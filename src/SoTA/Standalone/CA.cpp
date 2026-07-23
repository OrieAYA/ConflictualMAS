#include "SoTA/Standalone/CA.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GraphSearch.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

void FaithfulCASolver::init(const SolverContext& ctx) {
    ctx_ = &ctx;

    agents_.clear();
    agents_.reserve(static_cast<size_t>(ctx.n_active_agents));
    for (int i = 0; i < ctx.n_active_agents; ++i) {
        AgentState a;
        a.current_node = (i < static_cast<int>(ctx.agent_start_nodes.size()))
                       ? ctx.agent_start_nodes[i]
                       : 0;
        a.capacity = (i < static_cast<int>(ctx.per_agent_capacity.size()) &&
                      !ctx.per_agent_capacity.empty())
                   ? ctx.per_agent_capacity[i]
                   : std::max(1, ctx.max_capacity_per_agent);
        a.plan_tail_node = a.current_node;
        a.plan_tail_step = 0;
        agents_.push_back(std::move(a));
    }

    pending_task_ids_.clear();
    tasks_.clear();
    appeared_ = completed_ = refused_ = 0;
    latency_sum_ = wait_sum_ = trip_sum_ = 0;
    road_pd_sum_ = 0.0;
    road_pd_count_ = 0;
    active_steps_sum_ = 0;
    wait_count_ = 0;
    capacity_violations_ = pairing_violations_ = 0;
    instr_.init(ctx.n_active_agents, ctx.speed_mps);
}

void FaithfulCASolver::inject_task(const ScheduledTask& task, int step) {
    TaskRecord r;
    r.task_id        = static_cast<int>(tasks_.size());
    r.pickup_node    = task.pickup_node_id;
    r.delivery_node  = task.delivery_node_id;
    r.arrival_step   = step;
    // STATIC shortest-path pickup→delivery road distance — the reference
    // "ideal" length behind mean_road_pd_m / delivery_route_efficiency. Must
    // be the free-flow shortest path, exactly like the RL pipeline
    // (get_or_compute_path → ObjectivePath::cost) and HAPC (seg_cost); using
    // the congestion-detoured BPR path here inflated the column for CA alone.
    {
        const auto& ways = ctx_->geo_box->data.ways;
        const auto edges = graph_search::shortest_path_edges(
            *ctx_->geo_box, r.pickup_node, r.delivery_node);
        for (auto eid : edges) {
            auto it = ways.find(eid);
            if (it != ways.end()) r.pd_road_dist += it->second.distance_meters;
        }
    }
    tasks_.push_back(r);
    pending_task_ids_.push_back(r.task_id);
    ++appeared_;
}

int FaithfulCASolver::edge_arrival_step(osmium::object_id_type edge_id,
                                         int t_enter) {
    if (!ctx_) return t_enter + 1;
    const auto& ways = ctx_->geo_box->data.ways;
    auto it = ways.find(edge_id);
    if (it == ways.end()) return t_enter + 1;
    const float length_m  = it->second.distance_meters;
    const float base_time = length_m / std::max(0.1f, ctx_->speed_mps);
    if (!ctx_->congestion_map)
        return t_enter + std::max(1, static_cast<int>(std::ceil(base_time)));

    // Traversal rule (parity with EpisodeRunner::schedule_next_edge): the agent
    // pays the BPR time of the n OTHERS on the edge, NOT n+1 — its own
    // committed weight is subtracted. recommit_route always runs before the
    // agent is scheduled onto an edge, so that weight is present in the map at
    // this point. Without this the solver paid for its own footprint while the
    // RL fleet did not — on a short edge (capacity = max(1, d·0.05) = 1 below
    // 20 m) that alone is a ×1.15 slowdown handed to the baseline.
    const int self_w = std::max(1, ctx_->congestion_map->params.load_per_agent);
    const float adj  = ctx_->congestion_map->adjusted_cost(
        edge_id, base_time, length_m, t_enter, self_w);
    // Exposure accumulators sampled at the edge's ENTRY step (t_enter = current
    // step at all call sites). Must be here, NOT at completion: CongestionMap::
    // advance() purges past steps, so a later query at the entry step reads
    // load 0 → BPR 1.0. `load_at_entry` is the FULL load (self included), same
    // as the RL pipeline's jam counter.
    instr_.record_edge_entry(base_time, adj,
                             ctx_->congestion_map->get_load(edge_id, t_enter));
    return t_enter + std::max(1, static_cast<int>(std::ceil(adj)));
}

// ── BPR-aware A* ─────────────────────────────────────────────────────────────
//
// Find the path from `from` to `to` that minimises BPR-adjusted travel
// time, given that traversal starts at `start_step` and edge load on each
// edge is read at the predicted ARRIVAL step at that edge's head.
//
// This is the FAITHFUL implementation of Asadi+2025 §3.2's "modified A*
// that incorporates estimated congestion". We use the SHARED CongestionMap
// as the estimator (substituting for the paper's CNN prediction).
//
// Admissible heuristic: euclidean distance / speed_mps — a strict lower
// bound on travel time under zero congestion (BPR factor ≥ 1).
FaithfulCASolver::BPRPath
FaithfulCASolver::bpr_a_star(osmium::object_id_type from,
                              osmium::object_id_type to,
                              int start_step) const {
    BPRPath result;
    if (!ctx_ || !ctx_->geo_box) return result;
    if (from == 0 || to == 0) return result;
    if (from == to) {
        result.valid = true;
        result.nodes = { from };
        result.trip_time = 0.f;
        return result;
    }

    const auto& nodes = ctx_->geo_box->data.nodes;
    const auto& ways  = ctx_->geo_box->data.ways;
    const float speed = std::max(0.1f, ctx_->speed_mps);

    auto end_it = nodes.find(to);
    if (end_it == nodes.end()) return result;

    auto h_to_goal = [&](osmium::object_id_type n) -> float {
        // Admissible time heuristic: great-circle distance / speed — a strict
        // lower bound on travel time under zero congestion (BPR factor >= 1).
        return graph_search::haversine_between(*ctx_->geo_box, n, to) / speed;
    };

    struct OpenEntry {
        float f;
        float g;
        int   t_arrive;        // predicted simulation step at this node
        osmium::object_id_type node;
        bool operator>(const OpenEntry& o) const { return f > o.f; }
    };

    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    std::unordered_map<osmium::object_id_type, float> g_score;
    // came_from[v] = (parent node, edge taken)
    std::unordered_map<osmium::object_id_type,
        std::pair<osmium::object_id_type, osmium::object_id_type>> came_from;
    std::unordered_set<osmium::object_id_type> closed;

    g_score[from] = 0.f;
    open.push({ h_to_goal(from), 0.f, start_step, from });

    while (!open.empty()) {
        OpenEntry cur = open.top();
        open.pop();
        if (closed.count(cur.node)) continue;
        closed.insert(cur.node);

        if (cur.node == to) {
            // Reconstruct.
            std::vector<osmium::object_id_type> rev_nodes;
            std::vector<osmium::object_id_type> rev_edges;
            osmium::object_id_type cn = to;
            rev_nodes.push_back(cn);
            while (cn != from) {
                auto it = came_from.find(cn);
                if (it == came_from.end()) break;
                rev_edges.push_back(it->second.second);
                cn = it->second.first;
                rev_nodes.push_back(cn);
            }
            std::reverse(rev_nodes.begin(), rev_nodes.end());
            std::reverse(rev_edges.begin(), rev_edges.end());
            result.nodes = std::move(rev_nodes);
            result.edges = std::move(rev_edges);
            result.trip_time = cur.g;
            result.valid = true;
            return result;
        }

        // Cap the search to avoid runaway expansion on disconnected graphs.
        // 1.5× upper bound on BPR-adjusted distance.
        const float ub = cur.g + h_to_goal(cur.node) * 3.f;
        (void)ub;  // currently unused; placeholder for future budget-based pruning.

        auto node_it = nodes.find(cur.node);
        if (node_it == nodes.end()) continue;
        const auto& pt = node_it->second;

        for (auto edge_id : pt.incident_ways) {
            auto wit = ways.find(edge_id);
            if (wit == ways.end()) continue;
            const auto& w = wit->second;
            const osmium::object_id_type nbr =
                (w.node1_id == cur.node) ? w.node2_id : w.node1_id;
            if (nbr == 0) continue;
            if (closed.count(nbr)) continue;

            const float length_m = w.distance_meters;
            if (length_m <= 0.f) continue;
            const float base_time = length_m / speed;
            const float adj = ctx_->congestion_map
                ? ctx_->congestion_map->adjusted_cost(edge_id, base_time,
                                                      length_m, cur.t_arrive)
                : base_time;
            const float tentative_g = cur.g + adj;

            auto gs = g_score.find(nbr);
            if (gs != g_score.end() && tentative_g >= gs->second) continue;

            g_score[nbr] = tentative_g;
            came_from[nbr] = { cur.node, edge_id };
            const int new_t = cur.t_arrive + std::max(1, static_cast<int>(std::ceil(adj)));
            const float f = tentative_g + h_to_goal(nbr);
            open.push({ f, tentative_g, new_t, nbr });
        }
    }

    return result;  // valid = false
}

float FaithfulCASolver::decision_cost(const AgentState& a,
                                       const TaskRecord& t,
                                       int step) const {
    // Marginal APPEND cost (paper §3.1: the order joins the tail of the
    // agent's sequence T). Priced from where and when the agent finishes what
    // it already holds; for an idle agent that is here and now.
    const bool is_busy = !a.task_queue.empty();
    const osmium::object_id_type from = is_busy ? a.plan_tail_node : a.current_node;
    const int from_step = is_busy ? std::max(step, a.plan_tail_step) : step;
    if (from == 0) return std::numeric_limits<float>::max();

    // Pickup leg.
    BPRPath leg1 = bpr_a_star(from, t.pickup_node, from_step);
    if (!leg1.valid) return std::numeric_limits<float>::max();

    // Delivery leg, starting at the predicted pickup arrival step.
    const int t_pickup = from_step + std::max(1, static_cast<int>(std::ceil(leg1.trip_time)));
    BPRPath leg2 = bpr_a_star(t.pickup_node, t.delivery_node, t_pickup);
    if (!leg2.valid) return std::numeric_limits<float>::max();

    // Completion time of the new order = wait for the sequence to drain + its
    // own two legs. Minimising it is the paper's M(Π) = max_a |π_a| objective.
    const float queue_delay = static_cast<float>(std::max(0, from_step - step));
    const float completion  = queue_delay + leg1.trip_time + leg2.trip_time;

    // γ-mode weighting (paper §3.4 γ_moving < γ_others, here γ_idle < γ_busy).
    return (is_busy ? hparams.gamma_busy : hparams.gamma_idle) * completion;
}

osmium::object_id_type
FaithfulCASolver::leg_target(const AgentState& a) const {
    if (a.task_queue.empty()) return 0;
    const int tid = a.task_queue.front();
    if (tid < 0 || tid >= static_cast<int>(tasks_.size())) return 0;
    const TaskRecord& t = tasks_[tid];
    return a.active_is_pickup_leg ? t.pickup_node : t.delivery_node;
}

void FaithfulCASolver::fire_stop(AgentState& a, int step) {
    if (a.task_queue.empty()) return;
    const int tid = a.task_queue.front();
    if (tid < 0 || tid >= static_cast<int>(tasks_.size())) {
        a.task_queue.erase(a.task_queue.begin());
        a.active_is_pickup_leg = true;
        return;
    }
    TaskRecord& t = tasks_[tid];

    if (a.active_is_pickup_leg) {
        t.picked_step = step;
        a.in_flight_task_ids.push_back(tid);
        if (static_cast<int>(a.in_flight_task_ids.size()) > a.capacity)
            ++capacity_violations_;
        wait_sum_ += (t.picked_step - t.arrival_step);
        ++wait_count_;
        // Paper §3.3: the pickup→delivery transition fires on fulfilling the
        // order; the delivery leg is then re-planned under current congestion.
        a.active_is_pickup_leg = false;
        return;
    }

    if (t.picked_step < 0) ++pairing_violations_;
    t.delivered_step = step;
    latency_sum_ += (t.delivered_step - t.arrival_step);
    trip_sum_    += (t.delivered_step - std::max(t.picked_step, t.arrival_step));
    if (t.pd_road_dist > 0.f) {
        road_pd_sum_ += t.pd_road_dist;
        ++road_pd_count_;
    }
    ++completed_;
    instr_.record_delivery(static_cast<int>(&a - agents_.data()));
    auto it = std::find(a.in_flight_task_ids.begin(),
                        a.in_flight_task_ids.end(), tid);
    if (it != a.in_flight_task_ids.end()) a.in_flight_task_ids.erase(it);
    // Order fulfilled → move on to the next one in the sequence.
    a.task_queue.erase(a.task_queue.begin());
    a.active_is_pickup_leg = true;
}

bool FaithfulCASolver::advance_plan(AgentState& a, int step) {
    // Bounded loop: at most 2 stops per queued order can be zero-length.
    for (int guard = 0; guard < 64; ++guard) {
        const osmium::object_id_type target = leg_target(a);
        if (target == 0) {                       // sequence drained
            a.current_path_nodes.clear();
            a.current_path_edges.clear();
            a.next_idx = 0;
            return true;
        }
        if (a.current_node != target) {
            BPRPath p = bpr_a_star(a.current_node, target, step);
            if (!p.valid || p.nodes.size() < 2) {
                // Unreachable target. Never freeze the agent (that silently
                // removes it from the fleet for the rest of the episode and
                // depresses throughput for a non-algorithmic reason):
                //  - pickup leg  → release the order back to the pending pool;
                //  - delivery leg→ the order is already on board and cannot be
                //    released, so back off and retry later.
                if (a.active_is_pickup_leg) {
                    const int tid = a.task_queue.front();
                    a.task_queue.erase(a.task_queue.begin());
                    if (tid >= 0 && tid < static_cast<int>(tasks_.size())) {
                        tasks_[tid].assigned_agent = -1;
                        pending_task_ids_.push_back(tid);
                    }
                    continue;
                }
                a.stalled_until = step + 32;
                return false;
            }
            a.current_path_nodes = std::move(p.nodes);
            a.current_path_edges = std::move(p.edges);
            a.next_idx = 0;
            a.current_edge_t_enter = step;
            // Footprint for the FULL remaining sequence is published by
            // recommit_route(); the caller runs it right after. Scheduling the
            // first edge therefore happens against a map that already holds
            // this agent's own weight — which edge_arrival_step subtracts.
            a.arrival_step_next_node = -1;
            return true;
        }
        fire_stop(a, step);                      // zero-length leg
    }
    return true;
}

void FaithfulCASolver::advance_agent(AgentState& a, int step) {
    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) return;

    // Deferred first-edge timing (advance_plan installs the path, the commit
    // runs before the schedule so self-exclusion is correct).
    if (a.arrival_step_next_node < 0)
        a.arrival_step_next_node =
            edge_arrival_step(a.current_path_edges[a.next_idx], step);

    // Route exposure: load on the edge this agent occupies right now — one
    // sample per in-transit agent per step (RL parity, Runner.cpp).
    if (ctx_ && ctx_->congestion_map)
        instr_.sample_route_exposure(ctx_->congestion_map->get_load(
            a.current_path_edges[a.next_idx], step));

    if (step < a.arrival_step_next_node) {
        ++active_steps_sum_;
        return;
    }

    // Track distance on the edge just completed (BPR is sampled at ENTRY inside
    // edge_arrival_step — past steps are purged from the CongestionMap).
    if (ctx_) {
        const auto& ways = ctx_->geo_box->data.ways;
        auto wit = ways.find(a.current_path_edges[a.next_idx]);
        if (wit != ways.end())
            instr_.record_edge_traversal(wit->second.distance_meters);
    }

    ++a.next_idx;
    a.current_node = a.current_path_nodes[a.next_idx];

    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) {
        // Leg target reached: advance_plan fires the pickup/delivery event and
        // re-plans the next leg under CURRENT congestion (paper §3.2 — each
        // agent continuously re-evaluates its A*).
        advance_plan(a, step);
        recommit_route(a, step);
        return;
    }

    a.current_edge_t_enter = step;
    const osmium::object_id_type next_edge = a.current_path_edges[a.next_idx];
    a.arrival_step_next_node = edge_arrival_step(next_edge, step);
    ++active_steps_sum_;
}

// Re-register the agent's full remaining route on the shared CongestionMap,
// exactly like Option O's commit_plan: unregister the previous footprint, then
// add every edge of every remaining leg with free-flow windows weighted by
// load_per_agent. Called on every plan change (allocation, stop reached).
void FaithfulCASolver::recommit_route(AgentState& a, int step) {
    if (!ctx_ || !ctx_->congestion_map || !ctx_->geo_box) return;

    // Stops = every remaining objective of the whole order sequence, in
    // execution order (the front order's pickup is skipped once it is done).
    std::vector<osmium::object_id_type> stops;
    stops.reserve(a.task_queue.size() * 2);
    for (std::size_t k = 0; k < a.task_queue.size(); ++k) {
        const int tid = a.task_queue[k];
        if (tid < 0 || tid >= static_cast<int>(tasks_.size())) continue;
        const TaskRecord& t = tasks_[tid];
        if (!(k == 0 && !a.active_is_pickup_leg)) stops.push_back(t.pickup_node);
        stops.push_back(t.delivery_node);
    }

    // The FIRST leg must be committed on the path the agent is actually
    // following, not on a fresh A*: commit_agent_route removes this agent's
    // own footprint first, so re-running the search here can return a
    // different route and we would publish load on edges nobody drives.
    bool first_leg = true;
    const int tail = commit_agent_route(
        *ctx_->congestion_map, *ctx_->geo_box, ctx_->speed_mps,
        a.current_node, stops, step, a.committed_occ,
        [&](osmium::object_id_type f, osmium::object_id_type to, int t) {
            if (first_leg) {
                first_leg = false;
                if (!a.current_path_edges.empty() &&
                    a.next_idx < static_cast<int>(a.current_path_edges.size()) &&
                    f == a.current_node && to == a.current_path_nodes.back()) {
                    return std::vector<osmium::object_id_type>(
                        a.current_path_edges.begin() + a.next_idx,
                        a.current_path_edges.end());
                }
            }
            return bpr_a_star(f, to, t).edges;
        });

    a.plan_tail_step = tail;
    a.plan_tail_node = stops.empty() ? a.current_node : stops.back();

    // Commit-before-schedule (§3.2 self-exclusion): the agent's own weight is
    // now on the map, so edge_arrival_step can subtract it.
    if (a.arrival_step_next_node < 0 &&
        a.next_idx < static_cast<int>(a.current_path_edges.size()))
        a.arrival_step_next_node =
            edge_arrival_step(a.current_path_edges[a.next_idx], step);
}

bool FaithfulCASolver::try_allocate_one(int step) {
    if (pending_task_ids_.empty()) return false;

    // Eligibility: the order sequence is not full (paper §3.1 — an agent holds
    // up to N_T orders). A busy agent is a legitimate candidate: the order is
    // appended to its sequence and its running leg is untouched.
    std::vector<int> eligible_idx;
    eligible_idx.reserve(agents_.size());
    for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
        const AgentState& a = agents_[i];
        if (static_cast<int>(a.task_queue.size()) >= a.capacity) continue;
        eligible_idx.push_back(i);
    }
    if (eligible_idx.empty()) return false;

    // ── DEFINING DECISION RULE (paper §3.4 + §3.3) ──────────────────────────
    // For each (eligible agent, pending task), compute γ-weighted full-trip
    // BPR-adjusted travel time. Then apply β_W tie-break.

    struct Cand { int agent_id; int task_idx; float cost; int load; };
    std::vector<Cand> cands;
    cands.reserve(eligible_idx.size() * pending_task_ids_.size());

    for (int ti = 0; ti < static_cast<int>(pending_task_ids_.size()); ++ti) {
        const int tid = pending_task_ids_[ti];
        const TaskRecord& t = tasks_[tid];
        for (int ai : eligible_idx) {
            const AgentState& a = agents_[ai];
            const float c = decision_cost(a, t, step);
            if (c >= std::numeric_limits<float>::max() * 0.5f) continue;
            cands.push_back({ ai, ti, c, static_cast<int>(a.task_queue.size()) });
        }
    }
    if (cands.empty()) return false;

    // Find global minimum cost.
    float best_cost = std::numeric_limits<float>::max();
    for (const auto& c : cands) if (c.cost < best_cost) best_cost = c.cost;
    const float band = best_cost * hparams.beta_tie_band;

    // β_π tie-break (paper §4.2-a — "the number of orders still to be
    // fulfilled"): within the cost band, prefer the agent with the SHORTEST
    // remaining order sequence. Ties broken by lowest cost, then by lowest
    // (agent, task) ids for determinism. This is what keeps M(Π) = max_a |π_a|
    // balanced across the fleet.
    //
    // NOTE: this is β_π, not β_W. Paper's β_W gives priority to
    // pickup/delivery (busy) agents over wandering/finished ones, which is the
    // opposite of what we want for LGPDP allocation (we want to balance load
    // by feeding less-loaded agents next). β_π is the correct paper analogue.
    int   best_aid       = -1;
    int   best_tidx      = -1;
    int   best_load      = std::numeric_limits<int>::max();
    float best_band_cost = std::numeric_limits<float>::max();
    for (const auto& c : cands) {
        if (c.cost > band) continue;
        bool better = false;
        if (c.load < best_load) better = true;
        else if (c.load == best_load && c.cost < best_band_cost) better = true;
        else if (c.load == best_load && c.cost == best_band_cost &&
                 (best_aid < 0 ||
                  std::make_pair(c.agent_id, c.task_idx) <
                  std::make_pair(best_aid, best_tidx))) better = true;
        if (better) {
            best_load      = c.load;
            best_band_cost = c.cost;
            best_aid       = c.agent_id;
            best_tidx      = c.task_idx;
        }
    }
    if (best_aid < 0 || best_tidx < 0) return false;

    // Commit allocation: the order joins the tail of the agent's sequence.
    const int tid = pending_task_ids_[best_tidx];
    pending_task_ids_.erase(pending_task_ids_.begin() + best_tidx);
    AgentState& a = agents_[best_aid];
    tasks_[tid].assigned_agent = best_aid;
    const bool was_idle = a.task_queue.empty();
    a.task_queue.push_back(tid);
    if (was_idle) {
        a.active_is_pickup_leg = true;
        advance_plan(a, step);
    }
    recommit_route(a, step);   // footprint of the FULL remaining sequence
    // advance_plan releases an order whose pickup turned out unreachable —
    // don't let the drain loop pick it again this step.
    return tasks_[tid].assigned_agent == best_aid;
}

void FaithfulCASolver::step(int timestep) {
    if (ctx_) {
        instr_.sample_congestion(ctx_->congestion_map);
        if (ctx_->ghost) instr_.sample_ghost(ctx_->ghost->n_active_now());
    }

    // Drain the pending pool. Timed per attempt so compute_time_per_decision_us
    // counts the same unit as HAPC (one allocation decision), and not timed at
    // all when nothing is pending.
    while (!pending_task_ids_.empty()) {
        bool progressed = false;
        instr_.time_allocation([&]{ progressed = try_allocate_one(timestep); });
        if (!progressed) break;
    }

    for (auto& a : agents_) {
        if (a.next_idx < static_cast<int>(a.current_path_edges.size())) {
            advance_agent(a, timestep);
            continue;
        }
        // Orders left but no running leg — an order appended while the agent
        // was mid-edge, or a target that was unreachable earlier. Re-plan
        // (backed off so a permanently unreachable target cannot trigger an
        // A* storm), never leave the agent frozen.
        if (a.task_queue.empty() || timestep < a.stalled_until) continue;
        a.stalled_until = -1;
        advance_plan(a, timestep);
        recommit_route(a, timestep);
    }
}

SolverMetrics FaithfulCASolver::finalize() {
    SolverMetrics m;
    m.tasks_appeared  = appeared_;
    m.tasks_completed = completed_;
    m.tasks_refused   = refused_;
    m.throughput_rate = (appeared_ > 0)
        ? std::min(1.f, static_cast<float>(completed_) / appeared_)
        : 0.f;
    m.accept_rate     = (appeared_ > 0)
        ? static_cast<float>(appeared_ - refused_) / appeared_
        : 0.f;
    if (completed_ > 0) {
        m.latency_mean    = static_cast<double>(latency_sum_) / completed_;
        m.mean_trip_steps = static_cast<double>(trip_sum_)    / completed_;
    }
    // Wait is accumulated at every PICKUP, so its denominator is the pickup
    // count — not the delivery count. Dividing by completed_ inflated the
    // column with the tasks picked up but still on board at episode end
    // (RL parity: Metrics.cpp wait_sum / wait_count).
    if (wait_count_ > 0)
        m.mean_wait_steps = static_cast<double>(wait_sum_) / wait_count_;
    if (road_pd_count_ > 0) {
        m.mean_road_pd_m = road_pd_sum_ / road_pd_count_;
    }
    if (ctx_ && ctx_->n_active_agents > 0 && ctx_->total_steps > 0) {
        m.agent_utilisation = static_cast<float>(active_steps_sum_) /
            (static_cast<float>(ctx_->n_active_agents) * ctx_->total_steps);
    }
    m.capacity_violations = capacity_violations_;
    m.pairing_violations  = pairing_violations_;
    // Per-agent latency is normalised by the MEAN NUMBER OF ACTIVE AGENTS, not
    // by the provisioned fleet (RL parity: Metrics.cpp active_sum/active_steps).
    if (ctx_ && ctx_->total_steps > 0) {
        const double mean_active =
            static_cast<double>(active_steps_sum_) / ctx_->total_steps;
        m.latency_per_agent = m.latency_mean / std::max(1.0, mean_active);
    }
    instr_.finalize_into(m);
    return m;
}