#include "DeliveryAgent.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include "DMASforPD/TaskAgent/TaskAllocationModule.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"
#include "DMASforPD/Policy/IPPOPolicy.hpp"
#include "DMASforPD/Policy/MapperPolicy.hpp"
#include "DMASforPD/Policy/FaithfulMapperPolicy.hpp"
#include "DMASforPD/Policy/HybridPolicy.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>

// ---- Construction -------------------------------------------------------

DeliveryAgent::DeliveryAgent(int id, osmium::object_id_type start_node)
    : agent_id(id), current_node(start_node) {
    solution = AgentSolution::make(&current_node);
}

// ---- Spatial state ------------------------------------------------------

void DeliveryAgent::start_edge(
    osmium::object_id_type edge_id,
    osmium::object_id_type to_node,
    int arrival_step
) {
    in_transit = EdgeTransit{ edge_id, current_node, to_node, arrival_step };
}

void DeliveryAgent::arrive_at_node() {
    if (!in_transit) return;
    current_node = in_transit->to_node;
    in_transit.reset();
    // current_node is pointed to by solution.current_position — automatically reflected.
}

// ---- GlobalMemory interface ---------------------------------------------

void DeliveryAgent::connect(PDPGlobalMemory& memory) {
    memory.register_delivery_agent(*this);
}

void DeliveryAgent::disconnect(PDPGlobalMemory& memory) {
    memory.unregister_delivery_agent(agent_id);
}

void DeliveryAgent::receive_task(PDPTask& task, PDPGlobalMemory& memory) {
    add_task_to_memory(task);

    // ── Capacity-aware insertion (DbVNS-PDP spirit, extended for LGPDP) ───────
    //
    // The agent's queue length is unbounded; what matters is the simultaneous
    // carrying load (picked-but-not-yet-delivered packages) at every step of
    // the planned sequence. This must never exceed the physical capacity
    // max_tasks_per_agent. We extend the existing DbVNS-PDP planning idea
    // (constraint-aware exploration of admissible positions) into an
    // online insertion routine that respects three constraints:
    //
    //   1. **In-flight invariant** — solution.sequence[0] is being traversed
    //      via edge_cursor; we never touch it (pos_P >= 1).
    //   2. **Sequence constraint** — for every task, pickup must precede
    //      delivery (enforced by construction: pos_D >= pos_P).
    //   3. **Capacity constraint** — at no point may the simultaneous carry
    //      exceed max_carry (verified against the load profile).
    //
    // Among admissible (pos_P, pos_D) pairs we pick the one minimising the
    // induced detour cost (cheapest-insertion in the Solomon / Savelsbergh
    // PDP literature, restricted to the constraint tree above). The (n, n)
    // pair — FIFO append at the end — is always admissible (carry = 1 around
    // adjacent P,D), so the search always terminates with a valid plan.
    //
    // Tasks already picked up but not yet delivered (i.e. whose delivery is
    // still in the sequence) contribute to the initial carrying load — without
    // this, an agent mid-task could be told to accept another package while
    // already at capacity.

    auto& seq = solution.sequence;
    const int n = static_cast<int>(seq.size());

    // ── ALNS global replanning branch (Ropke-Pisinger 2006 adapted) ─────────
    // Used ONLY by the planning-comparison test (PolicyMode::ALNS). Same
    // contract as DbVNS branch: preserve in-flight head, rebuild the remainder
    // via destroy-repair ALNS on the full set of remaining tasks.
    if (memory.planning_use_alns) {
        if (seq.empty()) {
            solution.push_back(task.pickup);
            solution.push_back(task.delivery);
            status = AgentStatus::Active;
            prefetch_next_path(memory);
            return;
        }
        const ObjectiveNode inflight = seq[0].node;
        const osmium::object_id_type plan_start = inflight.id;

        std::unordered_set<osmium::object_id_type> remaining_ids;
        for (int i = 1; i < n; ++i) remaining_ids.insert(seq[i].node.id);
        remaining_ids.insert(task.pickup.id);
        remaining_ids.insert(task.delivery.id);

        OperableEnvironment tmp_env;
        PairingMap           pickup_of_replan;
        for (int i = 1; i < n; ++i) tmp_env.add_single_node(seq[i].node);
        tmp_env.add_single_node(task.pickup);
        tmp_env.add_single_node(task.delivery);

        for (const PDPTask* t : local_memory.tasks) {
            if (t->timeline.picked_step >= 0) continue;
            const bool p_in = remaining_ids.count(t->pickup.id)   > 0;
            const bool d_in = remaining_ids.count(t->delivery.id) > 0;
            if (p_in && d_in && t->pickup.id != plan_start)
                pickup_of_replan[t->delivery.id] = t->pickup.id;
        }

        int tmp_n = tmp_env.size();
        std::vector<float> sc(tmp_n, kCostScale);
        for (int i = 0; i < tmp_n; ++i) {
            const auto* p = memory.get_or_compute_path(
                plan_start, tmp_env.nodes[i].id, 1);
            if (p && p->valid()) sc[i] = p->cost;
        }
        tmp_env.refresh_costs(memory);

        const int tam_cap = memory.task_agent.params.max_tasks_per_agent;
        const int max_cap = std::max(
            1, this->max_capacity > 0 ? this->max_capacity : tam_cap);
        int load_at_start = 0;
        for (const PDPTask* t : local_memory.tasks) {
            if (t == &task) continue;
            const bool picked    = t->timeline.picked_step    >= 0;
            const bool delivered = t->timeline.delivered_step >= 0;
            if (picked && !delivered) ++load_at_start;
        }
        if (const PDPTask* t0 = memory.get_task_for_node(plan_start)) {
            if (t0->pickup.id   == plan_start && t0->timeline.picked_step    < 0)
                ++load_at_start;
            else if (t0->delivery.id == plan_start && t0->timeline.delivered_step < 0)
                --load_at_start;
        }
        load_at_start = std::max(0, std::min(max_cap, load_at_start));

        auto planned = LocalSolutionAgent::plan_sequence_alns(
            tmp_env, pickup_of_replan, sc, max_cap, load_at_start);

        seq.clear();
        solution.push_back(inflight);
        for (const auto& node : planned) solution.push_back(node);

        status = AgentStatus::Active;
        prefetch_next_path(memory);
        return;
    }

    // ── DbVNS global replanning branch ──────────────────────────────────────
    // Discards the existing insertion order (except the in-flight head seq[0])
    // and reoptimises the full remaining sequence via forward DbVNS-PDP.
    if (memory.planning_use_dbvns) {
        if (seq.empty()) {
            // Idle agent: trivial first task.
            solution.push_back(task.pickup);
            solution.push_back(task.delivery);
            status = AgentStatus::Active;
            prefetch_next_path(memory);
            return;
        }

        // Save the in-flight destination (seq[0]) — it is fixed.
        const ObjectiveNode inflight = seq[0].node;
        const osmium::object_id_type plan_start = inflight.id;

        // Determine if inflight is a pickup (its delivery becomes immediately
        // available once the agent arrives at plan_start).
        osmium::object_id_type inflight_delivery_id = 0;
        {
            const PDPTask* t0 = memory.get_task_for_node(plan_start);
            if (t0 && t0->pickup.id == plan_start)
                inflight_delivery_id = t0->delivery.id;
        }

        // Build the set of remaining nodes (seq[1..n-1] + new pickup + new delivery).
        std::unordered_set<osmium::object_id_type> remaining_ids;
        for (int i = 1; i < n; ++i) remaining_ids.insert(seq[i].node.id);
        remaining_ids.insert(task.pickup.id);
        remaining_ids.insert(task.delivery.id);

        // Build temporary env and pickup_of for replanning.
        // pickup_of[delivery_id] = pickup_id  →  delivery is locked until pickup visited.
        OperableEnvironment tmp_env;
        PairingMap           pickup_of_replan;

        for (int i = 1; i < n; ++i) tmp_env.add_single_node(seq[i].node);
        tmp_env.add_single_node(task.pickup);
        tmp_env.add_single_node(task.delivery);

        // Lock deliveries of tasks not yet picked up, except when the pickup IS
        // the inflight head (pickup guaranteed on arrival → delivery is free).
        for (const PDPTask* t : local_memory.tasks) {
            const bool already_picked = (t->timeline.picked_step >= 0);
            if (already_picked) continue;  // carrying: delivery is free
            const bool p_in_remaining = remaining_ids.count(t->pickup.id) > 0;
            const bool d_in_remaining = remaining_ids.count(t->delivery.id) > 0;
            if (p_in_remaining && d_in_remaining
                && t->pickup.id != plan_start) {
                pickup_of_replan[t->delivery.id] = t->pickup.id;
            }
            // If pickup IS plan_start, delivery is immediately available.
        }

        // Compute costs from plan_start to every node in tmp_env.
        int tmp_n = tmp_env.size();
        std::vector<float> sc(tmp_n, kCostScale);
        for (int i = 0; i < tmp_n; ++i) {
            const auto* p = memory.get_or_compute_path(
                plan_start, tmp_env.nodes[i].id, 1);
            if (p && p->valid()) sc[i] = p->cost;
        }
        tmp_env.refresh_costs(memory);

        // Capacity & load at plan_start (i.e. AFTER arriving at the inflight head).
        const int tam_cap = memory.task_agent.params.max_tasks_per_agent;
        const int max_cap = std::max(
            1, this->max_capacity > 0 ? this->max_capacity : tam_cap);
        int load_at_start = 0;
        for (const PDPTask* t : local_memory.tasks) {
            if (t == &task) continue;                  // new task not picked yet
            const bool picked    = t->timeline.picked_step    >= 0;
            const bool delivered = t->timeline.delivered_step >= 0;
            if (picked && !delivered) ++load_at_start;
        }
        // Adjust for the inflight head once the agent reaches it.
        if (const PDPTask* t0 = memory.get_task_for_node(plan_start)) {
            if (t0->pickup.id   == plan_start && t0->timeline.picked_step    < 0)
                ++load_at_start;
            else if (t0->delivery.id == plan_start && t0->timeline.delivered_step < 0)
                --load_at_start;
        }
        load_at_start = std::max(0, std::min(max_cap, load_at_start));

        auto planned = LocalSolutionAgent::plan_sequence(
            tmp_env, pickup_of_replan, sc, max_cap, load_at_start);

        // Rebuild sequence: inflight head + DbVNS result.
        seq.clear();
        solution.push_back(inflight);
        for (const auto& node : planned)
            solution.push_back(node);

        status = AgentStatus::Active;
        prefetch_next_path(memory);
        return;
    }

    const int tam_carry_cap = memory.task_agent.params.max_tasks_per_agent;
    const int max_carry = std::max(
        1, this->max_capacity > 0 ? this->max_capacity : tam_carry_cap);

    // Idle agent ⇒ trivial: queue just becomes [P, D].
    if (n == 0) {
        solution.push_back(task.pickup);
        solution.push_back(task.delivery);
        status = AgentStatus::Active;
        prefetch_next_path(memory);
        return;
    }

    // ── Initial carrying load: count tasks already picked but not delivered.
    // Skip the just-added new task — it has not been picked yet.
    int initial_load = 0;
    for (const PDPTask* t : local_memory.tasks) {
        if (t == &task) continue;
        if (t->timeline.picked_step >= 0 && t->timeline.delivered_step < 0)
            ++initial_load;
    }

    // ── Build the carry-load profile of the current sequence.
    //   load_after[i] = simultaneous carry AFTER visiting seq[i]
    //   (load_before[0] = initial_load by definition)
    std::vector<int> load_after(n, 0);
    {
        int cur = initial_load;
        for (int i = 0; i < n; ++i) {
            const PDPTask* t = memory.get_task_for_node(seq[i].node.id);
            if (t) {
                if      (t->pickup.id   == seq[i].node.id) ++cur;
                else if (t->delivery.id == seq[i].node.id) --cur;
            }
            load_after[i] = cur;
        }
    }

    // ── Path-cost helper using the persistent A* cache (group 1).
    auto seg = [&](osmium::object_id_type from,
                   osmium::object_id_type to) -> float {
        if (from == 0 || to == 0 || from == to) return 0.f;
        const auto* p = memory.get_or_compute_path(from, to, 1);
        return (p && p->valid()) ? p->cost : kCostScale;
    };

    // ── Constraint-tree search over admissible (pos_P, pos_D).
    //   Range for pos_P: [1, n]  (preserve in-flight at index 0; n = append)
    //   Range for pos_D: [pos_P, n]  (D after P; pos_D == pos_P = D right after P)
    //
    //   Capacity test: max(load_after[pos_P-1 .. pos_D-1]) + 1 <= max_carry.
    //   If pos_P == pos_D, range is the single index pos_P-1.
    // ── Planning strategy setup (cheapest vs Double-Horizon, fixed) ─────────
    //
    // memory.planning_use_double_horizon is set by EpisodeRunner from cfg.
    // When false → classical Solomon cheapest insertion (score = f_p + f_d).
    // When true  → Mitrovic-Minic 2004 Double-Horizon, faithful adaptation:
    //
    //   cost = (1-α_p)·f_p + α_p·g_p + (1-α_d)·f_d + α_d·g_d
    //
    // where g_p (g_d) is the PROPER slack-time decrease over downstream nodes:
    //
    //   g_p = Σ_{k=pP..n-1} [ slack_before(k) − slack_after_pP(k) ]
    //
    // and slack(k) = max(0, deadline(task(k)) − estimated_arrival(k)).
    //
    // The deadline is synthetic — there are no real time windows in lifelong
    // GPDP — taken as `task.created_step + kDHTaskMaxAge`. This re-introduces
    // the slack semantics the paper relies on: distant insertions that push
    // downstream nodes near their SLA boundary become expensive in g_p, which
    // is exactly the long-term flexibility argument Mitrovic-Minic makes.
    //
    // α_p, α_d ∈ {0 short-term, 0.25 long-term}: short-term insertions ignore
    // slack and just minimise route extension (paper c3).
    const bool  use_dh        = memory.planning_use_double_horizon;
    const float speed         = std::max(memory.speed_mps, 0.1f);
    const int   current_step  = use_dh
        ? static_cast<int>(memory.cur_time_ratio * memory.total_steps) : 0;
    const int   short_horizon = use_dh
        ? std::max(60, memory.total_steps / 10) : 0;
    constexpr float kDHAlpha       = 0.25f;
    constexpr int   kDHTaskMaxAge  = 1500;  // synthetic SLA: steps after task arrival

    // Precompute estimated arrival time and slack-budget for each downstream
    // sequence position. Slack = max(0, deadline − arrival) when the node maps
    // to a known task; otherwise treated as infinite (no SLA pressure).
    std::vector<int> arr;
    std::vector<int> slack_budget;   // ≥ 0; 0 = no slack left, INT_MAX = no SLA
    if (use_dh) {
        arr.resize(n);
        slack_budget.resize(n);
        int t_acc = current_step;
        osmium::object_id_type prev = current_node;
        for (int i = 0; i < n; ++i) {
            const int committed = seq[i].estimated_arrival;
            if (committed >= current_step) {
                arr[i] = committed;
                t_acc  = committed;
            } else {
                t_acc += static_cast<int>(std::ceil(seg(prev, seq[i].node.id) / speed));
                arr[i] = t_acc;
            }
            const PDPTask* tk = memory.get_task_for_node(seq[i].node.id);
            if (tk) {
                const int deadline = tk->timeline.created_step + kDHTaskMaxAge;
                slack_budget[i] = std::max(0, deadline - arr[i]);
            } else {
                slack_budget[i] = std::numeric_limits<int>::max();
            }
            prev = seq[i].node.id;
        }
    }

    // Helper: total slack-decrease over downstream nodes when their arrival
    // is pushed by `extra_steps`. Capped per-node by their current slack
    // budget (you cannot lose more slack than you have).
    auto slack_loss = [&](int from_pos, int extra_steps) -> float {
        if (extra_steps <= 0) return 0.f;
        long long acc = 0;
        for (int k = from_pos; k < n; ++k) {
            const int sb = slack_budget[k];
            const int loss = (sb >= extra_steps) ? extra_steps : sb;
            acc += loss;
        }
        return static_cast<float>(acc) * speed;  // convert steps → metres (same unit as f_p)
    };

    float best_score = std::numeric_limits<float>::max();
    int   best_pP = -1, best_pD = -1;

    for (int pP = 1; pP <= n; ++pP) {
        for (int pD = pP; pD <= n; ++pD) {
            // Capacity branch pruning.
            int peak_load = 0;
            for (int k = pP - 1; k <= pD - 1; ++k)
                if (load_after[k] > peak_load) peak_load = load_after[k];
            if (peak_load + 1 > max_carry) continue;

            // ── Compute per-position cost components f_p, f_d ───────────────
            // (Same components used by both strategies; cheapest uses their
            //  sum, double-horizon uses the weighted formula.)
            float f_p, f_d;
            if (pP == pD) {
                osmium::object_id_type before =
                    (pP == 0) ? current_node : seq[pP - 1].node.id;
                osmium::object_id_type after =
                    (pP == n) ? 0 : seq[pP].node.id;
                const float c_bp = seg(before, task.pickup.id);
                const float c_pd = seg(task.pickup.id, task.delivery.id);
                const float c_da = seg(task.delivery.id, after);
                const float c_ba = seg(before, after);
                // For DH we need a clean split; for cheapest only the sum matters.
                f_p = c_bp + c_pd;
                f_d = c_da - c_ba;
                if (f_d < 0.f) f_d = 0.f;
            } else {
                osmium::object_id_type before_P =
                    (pP == 0) ? current_node : seq[pP - 1].node.id;
                osmium::object_id_type after_P = seq[pP].node.id;
                f_p = seg(before_P, task.pickup.id)
                    + seg(task.pickup.id, after_P)
                    - seg(before_P, after_P);

                osmium::object_id_type before_D = seq[pD - 1].node.id;
                osmium::object_id_type after_D =
                    (pD == n) ? 0 : seq[pD].node.id;
                f_d = seg(before_D, task.delivery.id)
                    + seg(task.delivery.id, after_D)
                    - seg(before_D, after_D);
            }

            float score;
            if (use_dh) {
                // Estimated arrival time of the freshly inserted P and D.
                osmium::object_id_type src_for_P =
                    (pP == 0) ? current_node : seq[pP - 1].node.id;
                const int t_before_pP = (pP == 0) ? current_step : arr[pP - 1];
                const int t_new_P = t_before_pP + static_cast<int>(
                    std::ceil(seg(src_for_P, task.pickup.id) / speed));

                // Source of the D-leg: the new P node if pD == pP, otherwise
                // the existing seq[pD-1] (because the new P is inserted before
                // pD only when pD > pP, in which case the route from new D
                // departs from seq[pD-1] in the new sequence).
                osmium::object_id_type src_for_D =
                    (pD == pP) ? task.pickup.id
                                : ((pD == 0) ? current_node : seq[pD - 1].node.id);
                const int t_before_pD =
                    (pD == pP) ? t_new_P
                                : ((pD == 0) ? current_step : arr[pD - 1]);
                const int t_new_D = t_before_pD + static_cast<int>(
                    std::ceil(seg(src_for_D, task.delivery.id) / speed));

                const bool p_short = (t_new_P - current_step) <= short_horizon;
                const bool d_short = (t_new_D - current_step) <= short_horizon;
                const float alpha_p = p_short ? 0.f : kDHAlpha;
                const float alpha_d = d_short ? 0.f : kDHAlpha;

                // Proper slack-decrease: downstream nodes lose extra_steps of
                // slack, capped per-node by their remaining SLA budget.
                const int extra_p = static_cast<int>(std::ceil(f_p / speed));
                const int extra_d = static_cast<int>(std::ceil(f_d / speed));
                const float g_p = slack_loss(pP, extra_p);
                const float g_d = slack_loss(pD, extra_d);

                score = (1.f - alpha_p) * f_p + alpha_p * g_p
                      + (1.f - alpha_d) * f_d + alpha_d * g_d;
            } else {
                // Classical cheapest insertion: minimise pure route-length delta.
                score = f_p + f_d;
            }

            if (score < best_score) {
                best_score = score;
                best_pP    = pP;
                best_pD    = pD;
            }
        }
    }
    // For legacy compatibility (logs / debug printing if any): keep best_delta name.
    const float best_delta = best_score;
    (void)best_delta;

    // ── Apply the chosen insertion. (n, n) is always admissible (FIFO append
    // at the tail with carry = 1), so best_pP >= 0 holds in practice; the
    // explicit fallback covers any pathological no-path case.
    if (best_pP < 0) {
        solution.push_back(task.pickup);
        solution.push_back(task.delivery);
    } else {
        // Snapshot the existing nodes (clearing the sequence loses the
        // SolutionStep wrappers, but estimated_arrival is recomputed by
        // commit_plan() called by the caller).
        std::vector<ObjectiveNode> snapshot;
        snapshot.reserve(static_cast<size_t>(n) + 2);
        for (int i = 0; i < best_pP; ++i) snapshot.push_back(seq[i].node);
        snapshot.push_back(task.pickup);
        for (int i = best_pP; i < best_pD; ++i) snapshot.push_back(seq[i].node);
        snapshot.push_back(task.delivery);
        for (int i = best_pD; i < n; ++i) snapshot.push_back(seq[i].node);

        seq.clear();
        for (const auto& node : snapshot) solution.push_back(node);
    }

    status = AgentStatus::Active;

    // Pre-fetched next_path may be stale when pos_P == 1 (the second
    // objective changed). Refresh unconditionally — the call is cheap and
    // safe regardless of whether the next objective actually moved.
    prefetch_next_path(memory);
}

void DeliveryAgent::remove_completed_task(int task_id) {
    auto it = std::find_if(local_memory.tasks.begin(), local_memory.tasks.end(),
        [task_id](const PDPTask* t){ return t->task_id == task_id; });
    if (it == local_memory.tasks.end()) return;

    PDPTask* task = *it;
    local_memory.operable_env.remove_task(task->pickup.id, task->delivery.id);

    auto la_it = std::find_if(local_memory.local_agents.begin(), local_memory.local_agents.end(),
        [&](const LocalSolutionAgent& la){ return la.starting_node.id == task->delivery.id; });
    if (la_it != local_memory.local_agents.end())
        local_memory.local_agents.erase(la_it);

    local_memory.tasks.erase(it);
}

float DeliveryAgent::try_accept_task(const TaskOffer& offer, PDPGlobalMemory& memory) {
    PDPTask* task = memory.get_task(offer.task_id);
    if (!task) return 0.f;

    // ── Estimate insertion cost: current_node → pickup → delivery (upper bound) ──
    // Uses the path cache; falls back to kCostScale per missing leg.
    float insertion_cost = kCostScale * 2.f;
    {
        const auto* to_pu = memory.get_or_compute_path(
            current_node, task->pickup.id, task->pickup.group_id);
        const auto* pu_del = memory.get_or_compute_path(
            task->pickup.id, task->delivery.id, task->delivery.group_id);

        float c_pu  = (to_pu  && to_pu->valid())  ? to_pu->cost  : kCostScale;
        float c_del = (pu_del && pu_del->valid())  ? pu_del->cost : kCostScale;
        insertion_cost = c_pu + c_del;
    }

    // ── Current planned route cost ─────────────────────────────────────────────
    float route_cost = solution.total_planned_cost(memory);
    if (route_cost >= 1e10f || route_cost < 0.f) route_cost = 0.f;

    const float eps = 1e-6f;

    // ── Build PolicyFeatures ───────────────────────────────────────────────────
    PolicyFeatures f;
    f.cost_diff      = std::clamp(insertion_cost / kCostScale, 0.f, 1.f);
    f.profit_rate    = std::clamp(offer.reward / (insertion_cost * 0.5f + eps), 0.f, 1.f);
    f.current_load   = std::clamp(
        static_cast<float>(local_memory.tasks.size()) / kMaxLoad, 0.f, 1.f);
    f.queue_duration = std::clamp(route_cost / kQueueScale, 0.f, 1.f);
    // Fractional route extension. Idle agents (route_cost ≈ 0) get 0 — they start
    // fresh, no efficiency loss. Busy agents get insertion/(route+insertion) ∈ [0,1]:
    // 0 = task is free relative to existing route, 1 = doubles the route.
    // Previous formula insertion/(route+ε) collapsed to 1.0 for idle agents,
    // signalling "worst case" exactly when the agent was most available.
    f.efficiency_loss = (route_cost < 1.f)
        ? 0.f
        : std::clamp(insertion_cost / (route_cost + insertion_cost + eps), 0.f, 1.f);
    f.rank_in_call   = 1.f - std::clamp(
        static_cast<float>(offer.prev_agents.size()) / kMaxAgents, 0.f, 1.f);
    f.task_importance= std::clamp(offer.importance / kImpMax, 0.f, 1.f);
    f.recall_round_norm = std::clamp(
        static_cast<float>(offer.recall_round)
        / static_cast<float>(std::max(offer.max_recalls, 1)),
        0.f, 1.f);

    int n_agents = static_cast<int>(memory.all_delivery_agents().size());
    int total    = memory.count_total();
    f.n_agents_ratio = std::clamp(static_cast<float>(n_agents) / kMaxAgents, 0.f, 1.f);
    f.n_alloc_ratio  = (total > 0)
        ? static_cast<float>(memory.count_allocated()) / total : 0.f;
    f.n_avail_ratio  = (total > 0)
        ? static_cast<float>(memory.count_available()) / total : 0.f;
    f.time_remaining = std::clamp(1.f - memory.cur_time_ratio, 0.f, 1.f);

    // (12-d feature vector — `deliverability` removed. Episode-end feasibility
    //  is implicit via `time_remaining` and `cost_diff`. This matches the
    //  May 17 baseline setup and keeps the local observation compact, in line
    //  with MAPPER's design philosophy where the per-agent critic sees only
    //  local features.)

    // ── Stochastic action sampling ─────────────────────────────────────────────
    // PPO requires the recorded action to be sampled from π(a|s). Bernoulli(μ)
    // naturally generates both action=1 and action=0 around any μ ∈ (0,1),
    // letting the gradient learn the accept/refuse trade-off.
    //
    // Dispatch to the active learning policy:
    //   - kMAPPO          → ObjectiveDMPolicy (shared actor, centralised critic)
    //   - kIPPO           → IPPOPolicy (shared actor, shared local critic — paper-faithful)
    //   - kMAPPER         → MapperPolicy (per-agent + enhanced Ev with mutation)
    //   - kFaithfulMAPPER → FaithfulMapperPolicy (per-agent + paper-faithful Ev: copy, no mutation)
    //   - kHybrid         → HybridPolicy (MAPPO base + per-agent residual)
    // The TAM is policy-agnostic; it just compares the returned score to 0.5.
    static thread_local std::mt19937 rng{std::random_device{}()};
    float mu;
    switch (memory.active_policy) {
        case PDPGlobalMemory::PolicyKind::kIPPO:
            mu = IPPOPolicy::shared().score(f);
            break;
        case PDPGlobalMemory::PolicyKind::kMAPPER:
            mu = MapperPolicy::shared().score(agent_id, f);
            break;
        case PDPGlobalMemory::PolicyKind::kFaithfulMAPPER:
            mu = FaithfulMapperPolicy::shared().score(agent_id, f);
            break;
        case PDPGlobalMemory::PolicyKind::kHybrid:
            mu = HybridPolicy::shared().score(agent_id, f);
            break;
        case PDPGlobalMemory::PolicyKind::kMAPPO:
        default:
            mu = ObjectiveDMPolicy::shared().score(f);
            break;
    }
    std::bernoulli_distribution dist(std::clamp(mu, 0.001f, 0.999f));
    float action = dist(rng) ? 1.f : 0.f;

    // Reward is 0 at decision time; EpisodeRunner::on_objective_reached calls
    // update_reward() with the actual task value when the task is delivered.
    switch (memory.active_policy) {
        case PDPGlobalMemory::PolicyKind::kIPPO:
            IPPOPolicy::shared().record(agent_id, f, action, 0.f);
            break;
        case PDPGlobalMemory::PolicyKind::kMAPPER:
            MapperPolicy::shared().record(agent_id, f, action, 0.f);
            break;
        case PDPGlobalMemory::PolicyKind::kFaithfulMAPPER:
            FaithfulMapperPolicy::shared().record(agent_id, f, action, 0.f);
            break;
        case PDPGlobalMemory::PolicyKind::kHybrid:
            HybridPolicy::shared().record(agent_id, f, action, 0.f);
            break;
        case PDPGlobalMemory::PolicyKind::kMAPPO:
        default:
            ObjectiveDMPolicy::shared().record(f, action, 0.f, agent_id);
            break;
    }
    return action;  // 0 or 1 — TAM still uses >= 0.5 threshold which works.
}

float DeliveryAgent::compute_bid(const TaskOffer& offer, PDPGlobalMemory& memory) {
    PDPTask* task = memory.get_task(offer.task_id);
    if (!task) return 0.f;

    const auto* to_pu = memory.get_or_compute_path(
        current_node, task->pickup.id, task->pickup.group_id);
    const auto* pu_del = memory.get_or_compute_path(
        task->pickup.id, task->delivery.id, task->delivery.group_id);

    float c_pu  = (to_pu  && to_pu->valid())  ? to_pu->cost  : kCostScale;
    float c_del = (pu_del && pu_del->valid()) ? pu_del->cost : kCostScale;
    float insertion_cost = c_pu + c_del;

    constexpr float eps = 1e-6f;
    return (offer.reward * offer.importance) / std::max(insertion_cost, eps);
}

// ---- Path management (two-path lookahead) --------------------------------

bool DeliveryAgent::begin_leg(const ObjectivePath* path, int task_id, bool is_pickup) {
    local_memory.current_path = path;

    if (!path || !path->valid() || path->nodes.size() < 2) {
        // Fallback: no usable cached path — build a synthetic direct-hop cursor.
        // The simulator will compute a haversine-based travel time for this hop.
        osmium::object_id_type dest = solution.empty() ? 0 : solution.next_objective().id;
        edge_cursor = EdgeCursor{ {current_node, dest}, {}, 0, task_id, is_pickup };
        return false;
    }

    edge_cursor = EdgeCursor{
        path->nodes,   // all nodes including start and objective
        path->edges,   // way IDs between consecutive nodes
        0,             // agent is at nodes[0] = current_node
        task_id,
        is_pickup
    };
    return true;
}

void DeliveryAgent::promote_next_path() {
    local_memory.current_path = local_memory.next_path;
    local_memory.next_path    = nullptr;
}

void DeliveryAgent::prefetch_next_path(PDPGlobalMemory& memory) {
    // Pre-fetch the path from sequence[0]→sequence[1] (the leg after the current one).
    // sequence[0] is the objective we are currently heading to.
    if (solution.num_remaining() >= 2)
        local_memory.next_path = solution.path_between(0, memory);
    else
        local_memory.next_path = nullptr;
}

void DeliveryAgent::push_updated_path(const ObjectivePath* new_path) {
    // GlobalMemory calls this when congestion reroutes the current leg.
    // Rebuild the cursor from the agent's current position within the new path.
    local_memory.current_path = new_path;
    if (!new_path || !new_path->valid() || new_path->nodes.size() < 2) return;

    // Find the agent's current node in the new path to resume from there.
    const auto& ns = new_path->nodes;
    int resume = 0;
    for (int i = 0; i < static_cast<int>(ns.size()); ++i) {
        if (ns[i] == current_node) { resume = i; break; }
    }

    if (edge_cursor) {
        edge_cursor->nodes    = new_path->nodes;
        edge_cursor->edges    = new_path->edges;
        edge_cursor->next_idx = resume;
    }
}

// ---- Legacy path helpers (kept for TAM / planning compatibility) ---------

void DeliveryAgent::fetch_current_path(PDPGlobalMemory& memory) {
    local_memory.current_path =
        solution.empty() ? nullptr : solution.path_to_next(memory);
}

void DeliveryAgent::fetch_next_path(PDPGlobalMemory& memory) {
    local_memory.next_path =
        (solution.num_remaining() < 2) ? nullptr : solution.path_between(0, memory);
}

// ---- Planning -----------------------------------------------------------

void DeliveryAgent::plan(PDPGlobalMemory& memory, float speed_mps) {
    // 1. Refresh the cost matrix from GlobalMemory path cache.
    local_memory.operable_env.refresh_costs(memory);

    if (local_memory.local_agents.empty()) return;

    // 2. Build pickup/delivery pairings from the assigned task list.
    PairingMap pickup_of, delivery_of;
    for (const PDPTask* t : local_memory.tasks) {
        pickup_of [t->delivery.id] = t->pickup.id;
        delivery_of[t->pickup.id]  = t->delivery.id;
    }

    // 3. Run all LocalSolutionAgents; keep the lowest-cost sequence.
    std::vector<ObjectiveNode> best_seq;
    float                      best_cost = std::numeric_limits<float>::max();

    for (const LocalSolutionAgent& la : local_memory.local_agents) {
        std::vector<ObjectiveNode> candidate =
            la.plan(local_memory.operable_env, pickup_of, delivery_of, current_node);
        if (candidate.empty()) continue;

        float cost = 0.0f;
        for (std::size_t i = 0; i + 1 < candidate.size(); ++i) {
            int ai = local_memory.operable_env.find_index(candidate[i].id);
            int bi = local_memory.operable_env.find_index(candidate[i + 1].id);
            if (ai < 0 || bi < 0) { cost = std::numeric_limits<float>::max(); break; }
            float c = local_memory.operable_env.get_cost(ai, bi);
            if (c < 0.0f)         { cost = std::numeric_limits<float>::max(); break; }
            cost += c;
        }

        if (cost < best_cost) {
            best_cost = cost;
            best_seq  = std::move(candidate);
        }
    }

    if (best_seq.empty()) return;

    // 4. Rebuild solution sequence with the best candidate.
    solution.sequence.clear();
    for (const auto& node : best_seq)
        solution.push_back(node);

    // 5. Commit the plan to GlobalMemory (updates congestion + estimated arrivals).
    memory.commit_plan(agent_id, speed_mps);
}

// ---- Private ------------------------------------------------------------

void DeliveryAgent::add_task_to_memory(PDPTask& task) {
    local_memory.tasks.push_back(&task);
}
