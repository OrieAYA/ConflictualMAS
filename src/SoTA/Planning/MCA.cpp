#include "SoTA/Planning/MCA.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Policy/PolicyKit.hpp"   // kCostScale
#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

// Anytime LNS post-improvement [Chen et al. 2021, Alg. 3]. After cheapest
// insertion commits a feasible plan, destroy k random tasks and re-insert by
// RMCA cheapest order, keeping any improvement. No-op unless the strategy is
// enabled and the route has >= 2 stops. seg/max_carry/initial_load mirror the
// cheapest-insertion context in DeliveryAgent::receive_task.
void mca_lns_improve(DeliveryAgent& a, PDPTask& task, PDPGlobalMemory& memory) {
    if (!memory.planning_use_mca_lns || a.solution.sequence.size() < 2) return;

    const int tam_carry_cap = memory.task_agent.params.max_tasks_per_agent;
    const int max_carry = std::max(
        1, a.max_capacity > 0 ? a.max_capacity : tam_carry_cap);

    int initial_load = 0;
    for (const PDPTask* t : a.local_memory.tasks) {
        if (t == &task) continue;
        if (t->timeline.picked_step >= 0 && t->timeline.delivered_step < 0)
            ++initial_load;
    }

    auto seg = [&](osmium::object_id_type from,
                   osmium::object_id_type to) -> float {
        if (from == 0 || to == 0 || from == to) return 0.f;
        const auto* p = memory.get_or_compute_path(from, to, 1);
        return (p && p->valid()) ? p->cost : kCostScale;
    };

        constexpr int kLnsMaxIters  = 30;
        constexpr int kLnsGroupSize = 3;

        // ── Total cost helper (a.current_node → seq[0] → ... → seq[n-1]) ───────
        auto path_cost_of = [&](const std::vector<ObjectiveNode>& nodes) -> float {
            float c = 0.f;
            osmium::object_id_type prev = a.current_node;
            for (const auto& nd : nodes) {
                const float s = seg(prev, nd.id);
                if (s >= kCostScale) return std::numeric_limits<float>::infinity();
                c += s;
                prev = nd.id;
            }
            return c;
        };

        // Snapshot the current state — unwrap SolutionStep → ObjectiveNode for
        // direct mutation; commit_plan() (called by TAM after receive_task) will
        // rebuild the SolutionStep wrappers and estimated_arrival timestamps.
        std::vector<ObjectiveNode> best;
        best.reserve(a.solution.sequence.size());
        for (const auto& s : a.solution.sequence) best.push_back(s.node);
        float best_cost = path_cost_of(best);

        // Deterministic per-task seed → reproducibility across runs.
        std::mt19937 rng(0xC4EE2021u ^ static_cast<uint32_t>(task.task_id));

        for (int iter = 0; iter < kLnsMaxIters; ++iter) {
            std::vector<ObjectiveNode> trial = best;
            const int nt0 = static_cast<int>(trial.size());

            // ── destroyTasks(A, n) — identify swappable (pP, pD) pairs ───────
            struct SwapItem {
                int pP, pD;
                int task_id_within;
            };
            std::vector<SwapItem> swappable;
            std::unordered_map<osmium::object_id_type, int> node2pos;
            node2pos.reserve(static_cast<size_t>(nt0));
            for (int i = 0; i < nt0; ++i) node2pos[trial[i].id] = i;
            for (const PDPTask* t : a.local_memory.tasks) {
                if (t->timeline.picked_step >= 0) continue;     // already in transit
                auto itP = node2pos.find(t->pickup.id);
                auto itD = node2pos.find(t->delivery.id);
                if (itP == node2pos.end() || itD == node2pos.end()) continue;
                const int pp = itP->second, dp = itD->second;
                if (pp < 1 || dp < 1)  continue;                 // never touch head
                if (pp >= dp)          continue;                 // malformed
                swappable.push_back({pp, dp, t->task_id});
            }
            if (static_cast<int>(swappable.size()) < kLnsGroupSize) continue;

            // Sample kLnsGroupSize distinct indices without replacement.
            std::vector<int> pool(swappable.size());
            for (int i = 0; i < static_cast<int>(pool.size()); ++i) pool[i] = i;
            std::shuffle(pool.begin(), pool.end(), rng);
            pool.resize(static_cast<size_t>(kLnsGroupSize));

            // Build remove-positions list (sorted desc so erases keep earlier
            // indices valid) and the bank of removed tasks.
            std::vector<int> remove_positions;
            std::vector<int> bank_task_ids;
            remove_positions.reserve(static_cast<size_t>(kLnsGroupSize) * 2);
            bank_task_ids.reserve(static_cast<size_t>(kLnsGroupSize));
            for (int idx : pool) {
                const SwapItem& si = swappable[idx];
                remove_positions.push_back(si.pP);
                remove_positions.push_back(si.pD);
                bank_task_ids.push_back(si.task_id_within);
            }
            std::sort(remove_positions.begin(), remove_positions.end(), std::greater<int>());
            for (int rp : remove_positions) {
                if (rp >= 0 && rp < static_cast<int>(trial.size()))
                    trial.erase(trial.begin() + rp);
            }

            // ── RMCA(r) repair — greedy cheapest order ──────────────────────
            // Single-agent collapse of paper's regret rule (paper eq. 16): all
            // candidates target the same route, so positional regret reduces
            // to the cheapest-first insertion order.
            bool repair_ok = true;
            while (!bank_task_ids.empty()) {
                const int nt = static_cast<int>(trial.size());

                // Re-compute load profile after current trial mutations.
                std::vector<int> load_after_trial(nt, 0);
                {
                    int cur = initial_load;
                    for (int i = 0; i < nt; ++i) {
                        const PDPTask* tt = memory.get_task_for_node(trial[i].id);
                        if (tt) {
                            if      (tt->pickup.id   == trial[i].id) ++cur;
                            else if (tt->delivery.id == trial[i].id) --cur;
                        }
                        load_after_trial[i] = cur;
                    }
                }

                int   best_b = -1, b_pP = -1, b_pD = -1;
                float b_delta = std::numeric_limits<float>::max();

                for (int bi = 0; bi < static_cast<int>(bank_task_ids.size()); ++bi) {
                    const PDPTask* tk = memory.get_task(bank_task_ids[bi]);
                    if (!tk) continue;
                    const osmium::object_id_type pu_id = tk->pickup.id;
                    const osmium::object_id_type de_id = tk->delivery.id;

                    for (int pP = 1; pP <= nt; ++pP) {
                        for (int pD = pP; pD <= nt; ++pD) {
                            int peak = 0;
                            for (int k = pP - 1; k <= pD - 1; ++k)
                                if (load_after_trial[k] > peak) peak = load_after_trial[k];
                            if (peak + 1 > max_carry) continue;

                            float f_p, f_d;
                            if (pP == pD) {
                                const osmium::object_id_type before =
                                    (pP == 0) ? a.current_node : trial[pP - 1].id;
                                const osmium::object_id_type after =
                                    (pP == nt) ? 0 : trial[pP].id;
                                const float c_bp = seg(before, pu_id);
                                const float c_pd = seg(pu_id, de_id);
                                const float c_da = seg(de_id, after);
                                const float c_ba = seg(before, after);
                                f_p = c_bp + c_pd;
                                f_d = c_da - c_ba;
                                if (f_d < 0.f) f_d = 0.f;
                            } else {
                                const osmium::object_id_type before_P =
                                    (pP == 0) ? a.current_node : trial[pP - 1].id;
                                const osmium::object_id_type after_P  = trial[pP].id;
                                f_p = seg(before_P, pu_id)
                                    + seg(pu_id, after_P)
                                    - seg(before_P, after_P);
                                const osmium::object_id_type before_D = trial[pD - 1].id;
                                const osmium::object_id_type after_D  =
                                    (pD == nt) ? 0 : trial[pD].id;
                                f_d = seg(before_D, de_id)
                                    + seg(de_id, after_D)
                                    - seg(before_D, after_D);
                            }
                            const float delta = f_p + f_d;
                            if (delta < b_delta) {
                                b_delta = delta;
                                best_b  = bi;
                                b_pP    = pP;
                                b_pD    = pD;
                            }
                        }
                    }
                }
                if (best_b < 0) { repair_ok = false; break; }

                const PDPTask* tk_best = memory.get_task(bank_task_ids[best_b]);
                if (!tk_best)   { repair_ok = false; break; }

                // Apply: insert delivery first (higher index) so pickup index is preserved.
                if (b_pP == b_pD) {
                    trial.insert(trial.begin() + b_pP, tk_best->delivery);
                    trial.insert(trial.begin() + b_pP, tk_best->pickup);
                } else {
                    trial.insert(trial.begin() + b_pD, tk_best->delivery);
                    trial.insert(trial.begin() + b_pP, tk_best->pickup);
                }
                bank_task_ids.erase(bank_task_ids.begin() + best_b);
            }
            if (!repair_ok) continue;

            const float trial_cost = path_cost_of(trial);
            if (std::isfinite(trial_cost) && trial_cost <= best_cost) {
                best      = std::move(trial);
                best_cost = trial_cost;
            }
        }

        // Commit the LNS-improved sequence. By construction `best_cost` is the
        // running minimum across all accepted iterations starting from the
        // post-cheapest-insertion baseline — `best` is always at least as good
        // as the sequence that entered the LNS block. commit_plan() (called
        // by TAM after receive_task) rebuilds SolutionStep wrappers and
        // estimated_arrival timestamps from scratch, so only the node list
        // needs to be preserved.
        a.solution.sequence.clear();
        for (const auto& nd : best) a.solution.push_back(nd);
}
