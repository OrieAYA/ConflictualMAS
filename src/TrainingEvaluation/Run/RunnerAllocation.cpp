#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "DMASforPD/Agents/TaskAgent.hpp"
#include "SoTA/TaskAgentLike/TP.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

// ── Task offer ────────────────────────────────────────────────────────────────

EpisodeRunner::OfferResult EpisodeRunner::offer_task(
    int task_id, float reward, float importance,
    int n_active, const std::array<float, kGlobSz>& gs)
{
    // gs[0] = time_ratio, exposed to try_accept_task via GlobalMemory.
    memory_.cur_time_ratio = gs[0];

    // Default stats assume a full-scan baseline; the TAM branch overrides
    // them from the TAM getters before returning.
    last_offer_stats_ = LastOfferStats{};
    last_offer_stats_.agents_offered = n_active;
    last_offer_stats_.pre_marginal_costs.clear();

    // RAII timer: fills allocation/path/pure-alloc times on every return path.
    const auto offer_t0 = std::chrono::steady_clock::now();
    const long long path_us_before = PDPServerMemory::path_compute_time_us();
    struct TimerGuard {
        std::chrono::steady_clock::time_point t0;
        long long path_us_at_start;
        LastOfferStats& stats;
        ~TimerGuard() {
            const auto t1 = std::chrono::steady_clock::now();
            const long long total_us = std::chrono::duration_cast<
                std::chrono::microseconds>(t1 - t0).count();
            const long long path_us = PDPServerMemory::path_compute_time_us() - path_us_at_start;
            stats.allocation_time_us  = total_us;
            stats.path_time_us        = path_us;
            stats.pure_alloc_time_us  = (total_us > path_us) ? (total_us - path_us) : 0;
        }
    } _timer_guard{ offer_t0, path_us_before, last_offer_stats_ };

    // Full-scan oracle: marginal cost of every active agent for this task,
    // sampled BEFORE any allocation mutates state. Feeds the
    // marginal_cost_ratio_vs_oracle metric (chosen cost / min cost).
    PDPTask* task_for_oracle = memory_.get_task(task_id);
    if (task_for_oracle) {
        last_offer_stats_.pre_marginal_costs.reserve(n_active);
        for (int i = 0; i < n_active; ++i) {
            last_offer_stats_.pre_marginal_costs.push_back(
                compute_marginal_cost(*all_agents_[i], *task_for_oracle));
        }
    }

    // Physical carrying capacity is enforced at planning time inside
    // receive_task; every agent is eligible at offer time.
    auto has_cap = [&](const DeliveryAgent& /*a*/) { return true; };

    // ── Ablation baselines: sequential scan, no TAM, no buffer writes ─────────
    if (policy_mode == PolicyMode::Greedy) {
        for (int i = 0; i < n_active; ++i)
            if (has_cap(*all_agents_[i])) return { all_agents_[i]->agent_id, false };
        return { -1, false };
    }
    if (policy_mode == PolicyMode::Random) {
        std::bernoulli_distribution coin(0.5);
        for (int i = 0; i < n_active; ++i)
            if (has_cap(*all_agents_[i]) && coin(rng_))
                return { all_agents_[i]->agent_id, false };
        return { -1, false };
    }
    if (policy_mode == PolicyMode::InsertionGreedy) {
        TaskOffer offer{ task_id, reward, importance, {}, 0, 0 };
        for (int i = 0; i < n_active; ++i) {
            DeliveryAgent& a = *all_agents_[i];
            if (!has_cap(a)) continue;
            float bid = a.compute_bid(offer, memory_);
            if (bid > cfg_.insertion_greedy_threshold)
                return { a.agent_id, false };
        }
        return { -1, false };
    }

    // ── Token Passing [Ma et al. 2017, AAMAS] ─────────────────────────────────
    // A free agent picks the task minimising h(loc, pickup) on static cost.
    // Adaptation to the lifelong stream: when no agent is free, the same
    // argmin runs over all eligible agents (deferring would collapse
    // throughput); collision-free path commitments do not apply on a road
    // network where congestion is a soft cost, not a blocking constraint.
    if (policy_mode == PolicyMode::TokenPassing)
        return { tp_allocate(memory_, all_agents_, n_active, task_id), false };

    // ── Planning-level modes: allocation = cheapest marginal insertion ────────
    // DbVNS / ALNS: the planner itself is selected via the planning_use_*
    // flags in prepare_run; allocation is the same full-scan argmin.
    if (policy_mode == PolicyMode::DbVNS || policy_mode == PolicyMode::ALNS) {
        PDPTask* t = memory_.get_task(task_id);
        if (!t) return { -1, false };

        int   best_aid  = -1;
        float best_cost = std::numeric_limits<float>::max();
        for (int i = 0; i < n_active; ++i) {
            DeliveryAgent& a = *all_agents_[i];
            if (!has_cap(a)) continue;
            float mc = compute_marginal_cost(a, *t);
            if (mc < best_cost) { best_cost = mc; best_aid = a.agent_id; }
        }
        return { best_aid, false };
    }

    // Double-Horizon [Mitrovic-Minic et al. 2004]: same scan, but the cost
    // trades route length against slack preservation (alpha = 0.25).
    if (policy_mode == PolicyMode::DoubleHorizon) {
        PDPTask* t = memory_.get_task(task_id);
        if (!t) return { -1, false };
        constexpr float kAlpha = 0.25f;

        int   best_aid  = -1;
        float best_cost = std::numeric_limits<float>::max();
        for (int i = 0; i < n_active; ++i) {
            DeliveryAgent& a = *all_agents_[i];
            if (!has_cap(a)) continue;
            float c = compute_double_horizon_cost(a, *t, /*unused*/0, kAlpha);
            if (c < best_cost) { best_cost = c; best_aid = a.agent_id; }
        }
        return { best_aid, false };
    }

    // ── RL / RMCA / TamAlwaysAccept: drive the TAM ────────────────────────────
    // Each scored candidate records one experience; recent_records pairs the
    // i-th candidate with its (agent_id, buffer index).
    const int rec_before = active_policy_ ? active_policy_->n_recent_records() : 0;

    memory_.task_agent.on_new_task(task_id, memory_);
    auto* tam = memory_.task_agent.get_tam(task_id);
    if (!tam) return {-1, true};

    // The TAM terminates on its own conditions (allocated / exhausted /
    // deferred): adaptive budget x·ratio(x) and Dijkstra exhaustion.
    int tam_iter_count = 0;
    while (!tam->is_allocated() && !tam->is_exhausted()) {
        tam->step(memory_, cfg_.speed_mps);
        ++tam_iter_count;
    }
    last_offer_stats_.tam_dijkstra_steps = tam_iter_count;

    PDPTask* t = memory_.get_task(task_id);
    const bool allocated = tam->is_allocated() && t && t->agent_id >= 0;

    const bool shaping_active =
        (train_mode || (policy_mode == PolicyMode::Hybrid));

    auto resolve_buf_idx = [&](int i_scored) -> int {
        if (!active_policy_) return -1;
        const int k = rec_before + i_scored;
        if (k >= active_policy_->n_recent_records()) return -1;
        return active_policy_->recent_record(k).second;
    };

    auto add_reward_to = [&](int aid, int buf_idx, float delta) {
        if (active_policy_) active_policy_->add_to_reward(aid, buf_idx, delta);
    };

    {
        // TAM outcome → reward plumbing:
        //   - winner (only if it actually BID): bind its buffer entry to the
        //     task (pickup/delivery credit fires later) + congestion-creation
        //     penalty. A FORCED winner gets no binding — crediting a recorded
        //     "refuse" would teach the policy that refusing pays.
        //   - losing BIDDERS get the non-affected penalty; candidates that
        //     refused and lost made a consistent choice.
        const auto& cands  = tam->candidates_in_order();
        const auto& bids   = tam->bids_in_order();
        const bool  forced = tam->was_forced();
        if (!cands.empty() && t) {
            int win_pos = -1;
            if (allocated) {
                for (int i = 0; i < static_cast<int>(cands.size()); ++i) {
                    if (cands[i] == t->agent_id) { win_pos = i; break; }
                }
            }
            if (forced) win_pos = -1;

            if (win_pos >= 0) {
                const int win_buf_idx = resolve_buf_idx(win_pos);
                if (win_buf_idx >= 0) {
                    task_accept_buf_idx_[task_id] = win_buf_idx;

                    // Congestion-creation penalty (paper Table 2, winner row).
                    if (shaping_active && cfg_.congestion_creation_w > 0.f) {
                        DeliveryAgent* wa = memory_.get_delivery_agent(t->agent_id);
                        if (wa && cfg_.rs_congestion_route) {
                            // The winner pays the real BPR surcharge of its
                            // inserted route: congested replay minus free-flow
                            // cost of the same edges, over val_i. Own committed
                            // weight excluded (plan committed just before).
                            const int now    = memory_.congestion_map.current_step();
                            const int self_w = std::max(
                                1, memory_.congestion_map.params.load_per_agent);
                            auto leg = [&](osmium::object_id_type a,
                                           osmium::object_id_type b, int grp,
                                           float& cong, float& free) -> bool {
                                const auto* p = memory_.get_or_compute_path(a, b, grp);
                                if (!p || !p->valid()) return false;
                                const float c = memory_.bpr_path_cost(
                                    a, b, grp, now, cfg_.speed_mps, self_w);
                                if (!(c < std::numeric_limits<float>::max())) return false;
                                cong += c; free += p->cost;
                                return true;
                            };
                            float cong = 0.f, free = 0.f;
                            const bool ok =
                                leg(wa->current_node, t->pickup.id,
                                    t->pickup.group_id, cong, free)
                             && leg(t->pickup.id, t->delivery.id,
                                    t->delivery.group_id, cong, free);
                            if (ok) {
                                const float val_i = std::max(
                                    t->reward_original * t->importance_original, 1e-3f);
                                const float delta = std::clamp(
                                    (cong - free) / val_i, 0.f, 1.f);
                                // No ×importance: val_i already carries it.
                                add_reward_to(t->agent_id, win_buf_idx,
                                    -cfg_.congestion_creation_w * delta);
                            }
                        } else if (wa) {
                            // Legacy proxy: network mean BPR × normalised
                            // insertion distance.
                            const float to_pu_cost = [&]{
                                const auto* p = memory_.get_or_compute_path(
                                    wa->current_node, t->pickup.id, t->pickup.group_id);
                                return (p && p->valid()) ? p->cost : kCostScale;
                            }();
                            const float pu_del_cost = [&]{
                                const auto* p = memory_.get_or_compute_path(
                                    t->pickup.id, t->delivery.id, t->delivery.group_id);
                                return (p && p->valid()) ? p->cost : kCostScale;
                            }();
                            const float insertion_cost = to_pu_cost + pu_del_cost;
                            const float mean_load = memory_.congestion_map.mean_load_now();
                            const float ratio     = mean_load / 5.f;
                            const float r2        = ratio * ratio;
                            const float bpr_proxy = 1.f + 0.15f * r2 * r2;
                            const float norm_ins  = std::clamp(
                                insertion_cost / kCostScale, 0.f, 1.f);
                            const float delta     = std::clamp(
                                (bpr_proxy - 1.f) * norm_ins, 0.f, 1.f);
                            const float cong_pen  =
                                -cfg_.congestion_creation_w * delta
                                * t->importance_original;
                            add_reward_to(t->agent_id, win_buf_idx, cong_pen);
                        }
                    }
                }
            }

            // Non-affected penalty on losing bidders, scaled by their own bid
            // score mu — losing a confident bid hurts more.
            if (shaping_active && cfg_.non_affected_penalty_w > 0.f) {
                const auto& mus = tam->scores_in_order();
                const float w_naff = rs_w(cfg_.non_affected_penalty_w, 0.f);
                for (int i = 0; i < static_cast<int>(cands.size()); ++i) {
                    if (i == win_pos) continue;
                    if (i >= static_cast<int>(bids.size()) || !bids[i]) continue;
                    float pen = -w_naff * t->importance_original;
                    if (cfg_.rs_loser_mu && i < static_cast<int>(mus.size()))
                        pen *= mus[i];
                    const int aid = cands[i];
                    const int idx = resolve_buf_idx(i);
                    if (idx >= 0) add_reward_to(aid, idx, pen);
                }
            }
        }
    }

    const bool deferred = tam->is_deferred();

    // Capture TAM efficiency stats before erase_tam destroys them.
    last_offer_stats_.agents_offered    = tam->n_agents_offered();
    last_offer_stats_.recall_rounds     = tam->n_recall_rounds();
    last_offer_stats_.candidates_scored = tam->n_candidates_scored();

    memory_.task_agent.erase_tam(task_id);
    // Release this task's incremental-Dijkstra states (several MB per source
    // on large graphs) — they are reset per TAM anyway, keeping them all
    // episode accumulates GBs on task-heavy scenarios.
    if (t) {
        memory_.server_memory.reset_agent_search(t->pickup.id,   t->pickup.group_id);
        memory_.server_memory.reset_agent_search(t->delivery.id, t->delivery.group_id);
    }

    return { allocated ? t->agent_id : -1, true, deferred };
}

// ── Post-allocation bookkeeping (shared by arrival loop + retry loop) ─────────

void EpisodeRunner::commit_accepted_task(
    int task_id, int winner, bool tam_owned,
    osmium::object_id_type pickup_node, int step)
{
    DeliveryAgent* agent = memory_.get_delivery_agent(winner);
    if (!agent) return;
    PDPTask* task = memory_.get_task(task_id);

    bool was_idle;
    if (tam_owned) {
        // TAM already did assign + receive + commit; sole task ⇒ was Idle.
        was_idle = agent->local_memory.tasks.size() == 1;
    } else {
        was_idle = (agent->status == AgentStatus::Idle);
        if (task) {
            memory_.assign_task(task_id, winner);
            agent->receive_task(*task, memory_);
            memory_.commit_plan(winner, cfg_.speed_mps);
        }
    }

    // Start the leg only if the agent was Idle; an Active agent's cursor is
    // still valid and the new objectives follow the current delivery.
    if (was_idle)
        start_leg(winner, task_id, true, agent->current_node, pickup_node, step);
}

// ── Cheapest-insertion marginal cost ──────────────────────────────────────────
//
// Read-only mirror of DeliveryAgent::receive_task(): minimum route-cost delta
// over all admissible (pos_P, pos_D) with pos_P in [1, n] (preserves the
// in-flight head), pos_D in [pos_P, n], and peak carried load <= max_carry.
// Idle agent (n == 0): direct trip cost.

float EpisodeRunner::compute_marginal_cost(const DeliveryAgent& a,
                                           const PDPTask&       task) {
    const auto& seq = a.solution.sequence;
    const int   n   = static_cast<int>(seq.size());
    // Agent-effective capacity, not the TAM-global ceiling (heterogeneous
    // fleets would otherwise pick insertions the agent cannot honour).
    const int   tam_cap = memory_.task_agent.params.max_tasks_per_agent;
    const int   max_carry = std::max(
        1, a.max_capacity > 0 ? a.max_capacity : tam_cap);

    auto seg = [&](osmium::object_id_type from,
                   osmium::object_id_type to) -> float {
        if (from == 0 || to == 0 || from == to) return 0.f;
        const auto* p = memory_.get_or_compute_path(from, to, 1);
        return (p && p->valid()) ? p->cost : kCostScale;
    };

    if (n == 0)
        return seg(a.current_node, task.pickup.id)
             + seg(task.pickup.id, task.delivery.id);

    // Carry-load profile (task not yet in sequence).
    int initial_load = 0;
    for (const PDPTask* t : a.local_memory.tasks) {
        if (t->task_id == task.task_id) continue;
        if (t->timeline.picked_step >= 0 && t->timeline.delivered_step < 0)
            ++initial_load;
    }
    std::vector<int> load_after(n, 0);
    {
        int cur = initial_load;
        for (int i = 0; i < n; ++i) {
            const PDPTask* t = memory_.get_task_for_node(seq[i].node.id);
            if (t) {
                if      (t->pickup.id   == seq[i].node.id) ++cur;
                else if (t->delivery.id == seq[i].node.id) --cur;
            }
            load_after[i] = cur;
        }
    }

    float best_delta = std::numeric_limits<float>::max();
    for (int pP = 1; pP <= n; ++pP) {
        for (int pD = pP; pD <= n; ++pD) {
            int peak = 0;
            for (int k = pP - 1; k <= pD - 1; ++k)
                if (load_after[k] > peak) peak = load_after[k];
            if (peak + 1 > max_carry) continue;

            float delta;
            if (pP == pD) {
                osmium::object_id_type before = seq[pP - 1].node.id;
                osmium::object_id_type after  = (pP == n) ? 0 : seq[pP].node.id;
                delta = seg(before, task.pickup.id)
                      + seg(task.pickup.id, task.delivery.id)
                      + seg(task.delivery.id, after)
                      - seg(before, after);
            } else {
                osmium::object_id_type before_P = seq[pP - 1].node.id;
                osmium::object_id_type after_P  = seq[pP].node.id;
                float p_delta = seg(before_P, task.pickup.id)
                              + seg(task.pickup.id, after_P)
                              - seg(before_P, after_P);

                osmium::object_id_type before_D = seq[pD - 1].node.id;
                osmium::object_id_type after_D  = (pD == n) ? 0 : seq[pD].node.id;
                float d_delta = seg(before_D, task.delivery.id)
                              + seg(task.delivery.id, after_D)
                              - seg(before_D, after_D);

                delta = p_delta + d_delta;
            }
            if (delta < best_delta) best_delta = delta;
        }
    }
    return best_delta;
}

// ── Double-horizon insertion cost [Mitrovic-Minic et al. 2004] ────────────────
//
// Paper c3 cost adapted to the capacity-constrained L-GPDP:
//   c = [(1−α_p)·f_p + α_p·g_p] + [(1−α_d)·f_d + α_d·g_d]
// f = route-length increase at the insertion position; g = slack decrease at
// subsequent locations, expressed as (n − pos)·f; α = 0 when the estimated
// arrival at the inserted location falls in the short-term horizon, `alpha`
// otherwise. The horizon split is by estimated arrival TIME (paper-faithful).

float EpisodeRunner::compute_double_horizon_cost(const DeliveryAgent& a,
                                                  const PDPTask& task,
                                                  int /*unused_legacy*/,
                                                  float alpha) {
    const auto& seq = a.solution.sequence;
    const int   n   = static_cast<int>(seq.size());
    const int   tam_cap = memory_.task_agent.params.max_tasks_per_agent;
    const int   max_carry = std::max(
        1, a.max_capacity > 0 ? a.max_capacity : tam_cap);

    auto seg = [&](osmium::object_id_type from,
                   osmium::object_id_type to) -> float {
        if (from == 0 || to == 0 || from == to) return 0.f;
        const auto* p = memory_.get_or_compute_path(from, to, 1);
        return (p && p->valid()) ? p->cost : kCostScale;
    };

    // Current step recovered from the time ratio published by run().
    const int current_step = static_cast<int>(
        memory_.cur_time_ratio * static_cast<float>(memory_.total_steps));
    const float speed = std::max(0.1f, cfg_.speed_mps);

    // Paper uses a 1h short-term horizon on a 10h service period; mirror the
    // 10% ratio on the remaining episode, floored at 200 steps.
    const int remaining = std::max(0, memory_.total_steps - current_step);
    const int short_horizon_steps = std::max(200, remaining / 10);

    if (n == 0)
        return seg(a.current_node, task.pickup.id)
             + seg(task.pickup.id, task.delivery.id);

    // Load profile (same construction as compute_marginal_cost).
    int initial_load = 0;
    for (const PDPTask* t : a.local_memory.tasks) {
        if (t->task_id == task.task_id) continue;
        if (t->timeline.picked_step >= 0 && t->timeline.delivered_step < 0)
            ++initial_load;
    }
    std::vector<int> load_after(n, 0);
    {
        int cur = initial_load;
        for (int i = 0; i < n; ++i) {
            const PDPTask* t = memory_.get_task_for_node(seq[i].node.id);
            if (t) {
                if      (t->pickup.id   == seq[i].node.id) ++cur;
                else if (t->delivery.id == seq[i].node.id) --cur;
            }
            load_after[i] = cur;
        }
    }

    // Estimated arrival per existing position: prefer the commit_plan value,
    // else accumulate travel times from current_node.
    std::vector<int> arr(n, 0);
    {
        int t_acc = current_step;
        osmium::object_id_type prev = a.current_node;
        for (int i = 0; i < n; ++i) {
            const int committed = seq[i].estimated_arrival;
            if (committed >= current_step) {
                arr[i] = committed;
                t_acc = committed;
            } else {
                t_acc += static_cast<int>(std::ceil(seg(prev, seq[i].node.id) / speed));
                arr[i] = t_acc;
            }
            prev = seq[i].node.id;
        }
    }

    float best_cost = std::numeric_limits<float>::max();
    for (int pP = 1; pP <= n; ++pP) {
        for (int pD = pP; pD <= n; ++pD) {
            int peak = 0;
            for (int k = pP - 1; k <= pD - 1; ++k)
                if (load_after[k] > peak) peak = load_after[k];
            if (peak + 1 > max_carry) continue;

            float f_p, f_d;
            if (pP == pD) {
                osmium::object_id_type before = seq[pP - 1].node.id;
                osmium::object_id_type after  = (pP == n) ? 0 : seq[pP].node.id;
                const float c_bp = seg(before, task.pickup.id);
                const float c_pd = seg(task.pickup.id, task.delivery.id);
                const float c_da = seg(task.delivery.id, after);
                const float c_ba = seg(before, after);
                // Split the joint delta into p-side / d-side for the g weights.
                f_p = c_bp + c_pd;
                f_d = c_da - c_ba;
                if (f_d < 0.f) f_d = 0.f;
            } else {
                osmium::object_id_type before_P = seq[pP - 1].node.id;
                osmium::object_id_type after_P  = seq[pP].node.id;
                f_p = seg(before_P, task.pickup.id)
                    + seg(task.pickup.id, after_P)
                    - seg(before_P, after_P);
                osmium::object_id_type before_D = seq[pD - 1].node.id;
                osmium::object_id_type after_D  = (pD == n) ? 0 : seq[pD].node.id;
                f_d = seg(before_D, task.delivery.id)
                    + seg(task.delivery.id, after_D)
                    - seg(before_D, after_D);
            }

            // Estimated arrival of the new pickup / delivery.
            const int t_before_pP = (pP == 0) ? current_step : arr[pP - 1];
            const int t_before_pD = (pD == 0) ? current_step : arr[pD - 1];
            const int t_new_P = t_before_pP
                + static_cast<int>(std::ceil(seg(
                    (pP == 0 ? a.current_node : seq[pP - 1].node.id),
                    task.pickup.id) / speed));
            const int t_new_D = t_before_pD
                + static_cast<int>(std::ceil(seg(
                    (pD == pP ? task.pickup.id
                              : (pD == 0 ? a.current_node : seq[pD - 1].node.id)),
                    task.delivery.id) / speed));

            const bool p_short = (t_new_P - current_step) <= short_horizon_steps;
            const bool d_short = (t_new_D - current_step) <= short_horizon_steps;
            const float alpha_p = p_short ? 0.f : alpha;
            const float alpha_d = d_short ? 0.f : alpha;

            const float g_p = static_cast<float>(std::max(0, n - pP)) * f_p;
            const float g_d = static_cast<float>(std::max(0, n - pD)) * f_d;

            const float cost = (1.f - alpha_p) * f_p + alpha_p * g_p
                             + (1.f - alpha_d) * f_d + alpha_d * g_d;

            if (cost < best_cost) best_cost = cost;
        }
    }
    return best_cost;
}
