#include "TrafficFlowSolver.hpp"
#include "Environment/Congestion/CongestionMap.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

void TrafficFlowSolver::init(const SolverContext& ctx) {
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
        agents_.push_back(std::move(a));
    }

    pending_task_ids_.clear();
    tasks_.clear();
    flow_dir_.clear();
    vertex_inflow_.clear();
    appeared_ = completed_ = refused_ = 0;
    latency_sum_ = wait_sum_ = trip_sum_ = 0;
    road_pd_sum_ = 0.0;
    road_pd_count_ = 0;
    active_steps_sum_ = 0;
    capacity_violations_ = pairing_violations_ = 0;
    instr_.init(ctx.n_active_agents);
}

void TrafficFlowSolver::inject_task(const ScheduledTask& task, int step) {
    TaskRecord r;
    r.task_id        = static_cast<int>(tasks_.size());
    r.pickup_node    = task.pickup_node_id;
    r.delivery_node  = task.delivery_node_id;
    r.arrival_step   = step;
    // For metric purposes only: estimate p→d road distance via a quick
    // lex A* (no in-flight flow context — just the graph baseline).
    GuidePath warm = lex_a_star(r.pickup_node, r.delivery_node, step);
    if (warm.valid) {
        const auto& ways = ctx_->geo_box->data.ways;
        for (auto eid : warm.edges) {
            auto it = ways.find(eid);
            if (it != ways.end()) r.pd_road_dist += it->second.distance_meters;
        }
    }
    tasks_.push_back(r);
    pending_task_ids_.push_back(r.task_id);
    ++appeared_;
}

int TrafficFlowSolver::edge_arrival_step(osmium::object_id_type edge_id,
                                          int t_enter) const {
    if (!ctx_) return t_enter + 1;
    const auto& ways = ctx_->geo_box->data.ways;
    auto it = ways.find(edge_id);
    if (it == ways.end()) return t_enter + 1;
    const float length_m  = it->second.distance_meters;
    const float base_time = length_m / std::max(0.1f, ctx_->speed_mps);
    const float adj = ctx_->congestion_map
        ? ctx_->congestion_map->adjusted_cost(edge_id, base_time, length_m, t_enter)
        : base_time;
    return t_enter + std::max(1, static_cast<int>(std::ceil(adj)));
}

// ── Lex A* on the two-part edge weight (paper §4.2) ──────────────────────────
//
// State: (node, lex_cost_to_here, predicted_arrival_step). Frontier ordered
// by f = g + h where h is admissible: euclidean_distance / speed_mps. For
// the lex-cost we keep h's contraflow component at 0 (always admissible —
// minimum possible contraflow is 0) and only put the time component into h.
//
// Edge step cost when entering edge (u→v) at predicted step t:
//   - contraflow_step = flow_dir_[(v, u)] (marginal: candidate adds +1
//                       to f_{u,v}, so c_e_marginal = 1 × f_{v,u})
//   - bpr_time_step   = CongestionMap::adjusted_cost(e, length/speed,
//                       length, t)
//   - vertex pressure (optional, paper §4.1): += vertex_inflow_[v] to
//     the bpr_time as a secondary key softener; we fold it into bpr_time
//     so the lex order stays strict on contraflow first.
TrafficFlowSolver::GuidePath
TrafficFlowSolver::lex_a_star(osmium::object_id_type from,
                               osmium::object_id_type to,
                               int start_step) const {
    GuidePath result;
    if (!ctx_ || !ctx_->geo_box || !ctx_->pathfinder) return result;
    if (from == 0 || to == 0) return result;
    if (from == to) {
        result.valid = true;
        result.nodes = { from };
        return result;
    }

    const auto& nodes = ctx_->geo_box->data.nodes;
    const auto& ways  = ctx_->geo_box->data.ways;
    const float speed = std::max(0.1f, ctx_->speed_mps);

    auto end_it = nodes.find(to);
    if (end_it == nodes.end()) return result;

    auto h_time = [&](osmium::object_id_type n) -> float {
        const float h_m = const_cast<Pathfinder*>(ctx_->pathfinder)
                              ->heuristic(n, to);
        return h_m / speed;
    };

    struct OpenEntry {
        LexCost f;                  // g + (0, h_time)
        LexCost g;
        int     t_arrive;
        osmium::object_id_type node;
        bool operator>(const OpenEntry& o) const { return f > o.f; }
    };

    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    std::unordered_map<osmium::object_id_type, LexCost> g_score;
    std::unordered_map<osmium::object_id_type,
        std::pair<osmium::object_id_type, osmium::object_id_type>> came_from;
    std::unordered_set<osmium::object_id_type> closed;

    g_score[from] = LexCost{0, 0.f};
    open.push({ LexCost{0, h_time(from)}, LexCost{0, 0.f}, start_step, from });

    int expansions = 0;
    while (!open.empty()) {
        OpenEntry cur = open.top();
        open.pop();
        if (closed.count(cur.node)) continue;
        closed.insert(cur.node);

        if (cur.node == to) {
            // Reconstruct path.
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
            result.cost  = cur.g;
            result.valid = true;
            return result;
        }

        if (hparams.max_expansions > 0 && ++expansions > hparams.max_expansions) {
            break;
        }

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

            // ── Compute the TWO-PART edge cost (paper §4.1 / §4.2) ──────────
            // Primary: marginal contraflow on this candidate edge = flow in
            //          the OPPOSITE direction (the candidate adds 1 in
            //          (cur.node → nbr), so c_e_marginal = f_{nbr, cur.node}).
            long long contraflow_step = 0;
            {
                auto it_opp = flow_dir_.find(DirEdge{ nbr, cur.node });
                if (it_opp != flow_dir_.end()) contraflow_step = it_opp->second;
            }

            // Secondary: BPR-adjusted travel time, with optional vertex
            // pressure softener on the arrival vertex.
            float bpr_step = ctx_->congestion_map
                ? ctx_->congestion_map->adjusted_cost(edge_id, base_time,
                                                      length_m, cur.t_arrive)
                : base_time;
            if (hparams.enable_vertex_pressure) {
                auto it_vp = vertex_inflow_.find(nbr);
                if (it_vp != vertex_inflow_.end() && it_vp->second > 0) {
                    // Paper §4.1: per-agent vertex pressure p_v =
                    // ⌈(n_v - 1) / 2⌉ where n_v is the number of agents
                    // entering v. Exact formula, integer arithmetic.
                    const int n_v = it_vp->second;
                    const int p_v = (n_v - 1 + 1) / 2;  // ⌈(n_v - 1) / 2⌉
                    bpr_step += static_cast<float>(p_v);
                }
            }

            LexCost step_cost{ contraflow_step, bpr_step };
            LexCost tentative_g{ cur.g.contraflow + step_cost.contraflow,
                                  cur.g.bpr_time   + step_cost.bpr_time };

            auto gs = g_score.find(nbr);
            if (gs != g_score.end() && !(tentative_g < gs->second)) continue;

            g_score[nbr] = tentative_g;
            came_from[nbr] = { cur.node, edge_id };
            const int new_t = cur.t_arrive +
                std::max(1, static_cast<int>(std::ceil(bpr_step)));
            LexCost f{ tentative_g.contraflow,
                       tentative_g.bpr_time + h_time(nbr) };
            open.push({ f, tentative_g, new_t, nbr });
        }
    }

    return result;  // valid = false
}

void TrafficFlowSolver::register_path(
    const std::vector<osmium::object_id_type>& path_nodes, int from_idx)
{
    for (int k = std::max(0, from_idx);
         k + 1 < static_cast<int>(path_nodes.size()); ++k) {
        DirEdge e{ path_nodes[k], path_nodes[k + 1] };
        ++flow_dir_[e];
        ++vertex_inflow_[path_nodes[k + 1]];
    }
}

void TrafficFlowSolver::unregister_path(
    const std::vector<osmium::object_id_type>& path_nodes, int from_idx)
{
    for (int k = std::max(0, from_idx);
         k + 1 < static_cast<int>(path_nodes.size()); ++k) {
        DirEdge e{ path_nodes[k], path_nodes[k + 1] };
        auto it = flow_dir_.find(e);
        if (it != flow_dir_.end()) {
            if (--it->second <= 0) flow_dir_.erase(it);
        }
        auto vit = vertex_inflow_.find(path_nodes[k + 1]);
        if (vit != vertex_inflow_.end()) {
            if (--vit->second <= 0) vertex_inflow_.erase(vit);
        }
    }
}

TrafficFlowSolver::LexCost
TrafficFlowSolver::path_lex_cost(const GuidePath& gp) const {
    return gp.cost;
}

bool TrafficFlowSolver::begin_leg_guide(AgentState& a,
                                         osmium::object_id_type target_node,
                                         int step) {
    // Trivial path edge case (agent already at target): fire goal event
    // immediately and recurse for the follow-up leg if any.
    if (a.current_node == target_node) {
        if (a.active_task_id < 0 ||
            a.active_task_id >= static_cast<int>(tasks_.size())) {
            return true;
        }
        TaskRecord& t = tasks_[a.active_task_id];
        if (a.active_is_pickup_leg) {
            t.picked_step = step;
            a.in_flight_task_ids.push_back(a.active_task_id);
            if (static_cast<int>(a.in_flight_task_ids.size()) > a.capacity) {
                ++capacity_violations_;
            }
            wait_sum_ += (t.picked_step - t.arrival_step);
            a.active_is_pickup_leg = false;
            return begin_leg_guide(a, t.delivery_node, step);
        } else {
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
                                a.in_flight_task_ids.end(),
                                a.active_task_id);
            if (it != a.in_flight_task_ids.end()) a.in_flight_task_ids.erase(it);
            a.active_task_id = -1;
            a.current_path_nodes.clear();
            a.current_path_edges.clear();
            a.next_idx = 0;
            return true;
        }
    }

    GuidePath gp = lex_a_star(a.current_node, target_node, step);
    if (!gp.valid || gp.nodes.size() < 2) return false;

    // If the agent had a previous path, unregister its remaining edges.
    if (!a.current_path_nodes.empty()) {
        unregister_path(a.current_path_nodes, a.next_idx);
    }

    a.current_path_nodes = std::move(gp.nodes);
    a.current_path_edges = std::move(gp.edges);
    a.next_idx = 0;
    a.current_edge_t_enter = step;
    a.arrival_step_next_node = edge_arrival_step(a.current_path_edges.front(), step);

    // Register the FULL remaining path in the directional flow tracker so
    // subsequent agents see this guide path as committed traffic (paper §4.1
    // — committed paths contribute to contraflow seen by future planners).
    register_path(a.current_path_nodes, 0);

    if (ctx_->congestion_map) {
        ctx_->congestion_map->add_agent(a.current_path_edges.front(), step,
                                         a.arrival_step_next_node);
    }
    return true;
}

void TrafficFlowSolver::advance_agent(AgentState& a, int step) {
    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) return;
    if (step < a.arrival_step_next_node) {
        ++active_steps_sum_;
        return;
    }

    // Arrived at the next node — unregister the edge we just traversed
    // from the flow tracker (it's no longer "remaining").
    {
        DirEdge e{ a.current_path_nodes[a.next_idx],
                   a.current_path_nodes[a.next_idx + 1] };
        auto it = flow_dir_.find(e);
        if (it != flow_dir_.end()) {
            if (--it->second <= 0) flow_dir_.erase(it);
        }
        auto vit = vertex_inflow_.find(a.current_path_nodes[a.next_idx + 1]);
        if (vit != vertex_inflow_.end()) {
            if (--vit->second <= 0) vertex_inflow_.erase(vit);
        }
    }

    // Track distance traveled on the edge we just completed.
    if (ctx_ && a.next_idx < static_cast<int>(a.current_path_edges.size())) {
        const auto& ways = ctx_->geo_box->data.ways;
        auto wit = ways.find(a.current_path_edges[a.next_idx]);
        if (wit != ways.end())
            instr_.record_edge_traversal(wit->second.distance_meters);
    }

    ++a.next_idx;
    a.current_node = a.current_path_nodes[a.next_idx];

    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) {
        if (a.active_task_id >= 0 && a.active_task_id < static_cast<int>(tasks_.size())) {
            TaskRecord& t = tasks_[a.active_task_id];
            if (a.active_is_pickup_leg) {
                t.picked_step = step;
                a.in_flight_task_ids.push_back(a.active_task_id);
                if (static_cast<int>(a.in_flight_task_ids.size()) > a.capacity) {
                    ++capacity_violations_;
                }
                wait_sum_ += (t.picked_step - t.arrival_step);
                a.active_is_pickup_leg = false;
                // Re-plan delivery leg under current flow state. This is the
                // partial fidelity for paper §4.4 rolling re-planning — we
                // refresh the guide path on objective arrival.
                begin_leg_guide(a, t.delivery_node, step);
            } else {
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
                                    a.in_flight_task_ids.end(),
                                    a.active_task_id);
                if (it != a.in_flight_task_ids.end()) a.in_flight_task_ids.erase(it);
                a.active_task_id = -1;
                a.current_path_nodes.clear();
                a.current_path_edges.clear();
                a.next_idx = 0;
            }
        } else {
            a.current_path_nodes.clear();
            a.current_path_edges.clear();
            a.next_idx = 0;
        }
        return;
    }

    a.current_edge_t_enter = step;
    const osmium::object_id_type next_edge = a.current_path_edges[a.next_idx];
    a.arrival_step_next_node = edge_arrival_step(next_edge, step);
    if (ctx_->congestion_map) {
        ctx_->congestion_map->add_agent(next_edge, step, a.arrival_step_next_node);
    }
    ++active_steps_sum_;
}

bool TrafficFlowSolver::try_allocate_one(int step) {
    if (pending_task_ids_.empty()) return false;

    // Eligibility: idle journey, capacity not full.
    std::vector<int> eligible_idx;
    eligible_idx.reserve(agents_.size());
    for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
        const AgentState& a = agents_[i];
        if (a.active_task_id != -1) continue;
        if (!a.current_path_edges.empty() &&
            a.next_idx < static_cast<int>(a.current_path_edges.size())) continue;
        if (static_cast<int>(a.in_flight_task_ids.size()) >= a.capacity) continue;
        eligible_idx.push_back(i);
    }
    if (eligible_idx.empty()) return false;

    // ── DEFINING DECISION RULE (paper §4.2 lex two-part cost) ───────────────
    // For each (eligible agent, pending task), compute the lex A* cost of
    // a planned trip: agent.loc → pickup → delivery, under current flow.
    // argmin lex over (contraflow, BPR time).
    int     best_aid  = -1;
    int     best_tidx = -1;
    LexCost best_cost{ std::numeric_limits<long long>::max(),
                       std::numeric_limits<float>::max() };

    // We compute the pickup leg per (agent, task); the delivery leg per
    // task only (does not depend on agent). Cache delivery leg costs.
    std::unordered_map<int, LexCost> delivery_cost_cache;

    for (int ti = 0; ti < static_cast<int>(pending_task_ids_.size()); ++ti) {
        const int tid = pending_task_ids_[ti];
        const TaskRecord& t = tasks_[tid];

        for (int ai : eligible_idx) {
            const AgentState& a = agents_[ai];

            GuidePath leg1 = lex_a_star(a.current_node, t.pickup_node, step);
            if (!leg1.valid) continue;

            const int t_pickup = step +
                std::max(1, static_cast<int>(std::ceil(leg1.cost.bpr_time)));

            // Delivery leg cost: cache by task_id when start_step is the
            // same — different agents reach pickup at different times so
            // we can't cache by task alone. For simplicity, recompute.
            GuidePath leg2 = lex_a_star(t.pickup_node, t.delivery_node, t_pickup);
            if (!leg2.valid) continue;

            LexCost total{ leg1.cost.contraflow + leg2.cost.contraflow,
                            leg1.cost.bpr_time   + leg2.cost.bpr_time };

            if (total < best_cost) {
                best_cost = total;
                best_aid  = ai;
                best_tidx = ti;
            }
        }
    }
    if (best_aid < 0 || best_tidx < 0) return false;

    const int tid = pending_task_ids_[best_tidx];
    pending_task_ids_.erase(pending_task_ids_.begin() + best_tidx);
    AgentState& a = agents_[best_aid];
    TaskRecord& t = tasks_[tid];
    t.assigned_agent = best_aid;
    a.active_task_id = tid;
    a.active_is_pickup_leg = true;
    const bool ok = begin_leg_guide(a, t.pickup_node, step);
    if (!ok) {
        a.active_task_id = -1;
        pending_task_ids_.push_back(tid);
        return false;
    }
    return true;
}

void TrafficFlowSolver::step(int timestep) {
    if (ctx_) {
        instr_.sample_congestion(ctx_->congestion_map);
        if (ctx_->ghost) instr_.sample_ghost(ctx_->ghost->n_active_now());
    }

    while (try_allocate_one(timestep)) { /* drain pending */ }

    // ── Paper §4.4 online refinement (periodic replan) ─────────────────────
    // Every refinement_period steps, recompute each in-flight agent's
    // remaining path under the CURRENT flow state. This is the LGPDP
    // analogue of paper Algorithm 2's PathRefinement loop — single-agent
    // re-routing only, no cross-agent destruction (would create capacity
    // churn in online setting). Skips agents at the end of their current
    // path (no edges left to replan).
    if (hparams.refinement_period > 0 &&
        timestep > 0 &&
        (timestep % hparams.refinement_period) == 0) {
        for (auto& a : agents_) {
            if (a.active_task_id < 0) continue;
            // Determine the agent's CURRENT target (pickup or delivery).
            const TaskRecord& t = tasks_[a.active_task_id];
            const osmium::object_id_type tgt = a.active_is_pickup_leg
                ? t.pickup_node
                : t.delivery_node;
            if (a.current_node == tgt) continue;  // already arrived

            // Re-plan: unregister old path, compute fresh lex A*, register
            // new path. begin_leg_guide handles the bookkeeping.
            begin_leg_guide(a, tgt, timestep);
        }
    }

    for (auto& a : agents_) {
        if (a.current_path_edges.empty()) continue;
        if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) continue;
        advance_agent(a, timestep);
    }
}

SolverMetrics TrafficFlowSolver::finalize() {
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
        m.mean_wait_steps = static_cast<double>(wait_sum_)    / completed_;
        m.mean_trip_steps = static_cast<double>(trip_sum_)    / completed_;
    }
    if (road_pd_count_ > 0) {
        m.mean_road_pd_m = road_pd_sum_ / road_pd_count_;
    }
    if (ctx_ && ctx_->n_active_agents > 0 && ctx_->total_steps > 0) {
        m.agent_utilisation = static_cast<float>(active_steps_sum_) /
            (static_cast<float>(ctx_->n_active_agents) * ctx_->total_steps);
    }
    m.capacity_violations = capacity_violations_;
    m.pairing_violations  = pairing_violations_;
    if (ctx_ && ctx_->n_active_agents > 0)
        m.latency_per_agent = m.latency_mean / ctx_->n_active_agents;
    instr_.finalize_into(m);
    return m;
}