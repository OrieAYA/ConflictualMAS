#include "TokenPassingSolver.hpp"
#include "Environment/Congestion/CongestionMap.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

void TokenPassingSolver::init(const SolverContext& ctx) {
    ctx_ = &ctx;
    paths_ = std::make_unique<PathHelper>(*ctx.pathfinder);

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
    appeared_ = completed_ = refused_ = 0;
    latency_sum_ = wait_sum_ = trip_sum_ = 0;
    road_pd_sum_ = 0.0;
    road_pd_count_ = 0;
    active_steps_sum_ = 0;
    capacity_violations_ = pairing_violations_ = 0;
    instr_.init(ctx.n_active_agents);
}

void TokenPassingSolver::inject_task(const ScheduledTask& task, int step) {
    TaskRecord r;
    r.task_id        = static_cast<int>(tasks_.size());
    r.pickup_node    = task.pickup_node_id;
    r.delivery_node  = task.delivery_node_id;
    r.arrival_step   = step;
    // Pre-compute pickup→delivery road distance for the efficiency metric.
    if (paths_) {
        const auto& p = paths_->get(r.pickup_node, r.delivery_node);
        if (p.valid) r.pd_road_dist = p.cost;
    }
    tasks_.push_back(r);
    pending_task_ids_.push_back(r.task_id);
    ++appeared_;
}

int TokenPassingSolver::edge_arrival_step(osmium::object_id_type edge_id,
                                           int t_enter) const {
    if (!ctx_) return t_enter + 1;
    const auto& ways = ctx_->geo_box->data.ways;
    auto it = ways.find(edge_id);
    if (it == ways.end()) return t_enter + 1;
    const float length_m = it->second.distance_meters;
    const float base_time = length_m / std::max(0.1f, ctx_->speed_mps);
    // BPR-adjusted: read current edge load from the SHARED CongestionMap.
    const float adj = ctx_->congestion_map
        ? ctx_->congestion_map->adjusted_cost(edge_id, base_time, length_m, t_enter)
        : base_time;
    return t_enter + std::max(1, static_cast<int>(std::ceil(adj)));
}

bool TokenPassingSolver::begin_leg(AgentState& a,
                                    osmium::object_id_type target_node,
                                    int step) {
    if (!paths_) return false;

    // Trivial path: agent is already at the target node (e.g., new task's
    // pickup is exactly where the agent just delivered). Fire the goal
    // event immediately and, if there is a follow-up leg, recurse into it.
    if (a.current_node == target_node) {
        if (a.active_task_id < 0 ||
            a.active_task_id >= static_cast<int>(tasks_.size())) {
            // No active task — nothing to fire.
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
            return begin_leg(a, t.delivery_node, step);
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

    const SimplePath& sp = paths_->get(a.current_node, target_node);
    if (!sp.valid || sp.nodes.size() < 2) return false;

    a.current_path_nodes = sp.nodes;
    a.current_path_edges = sp.edges;
    a.next_idx = 0;
    a.current_edge_t_enter = step;
    a.arrival_step_next_node = edge_arrival_step(sp.edges.front(), step);

    // Register the first edge on the shared CongestionMap (load amplification
    // follows the SolverRunner's load_per_agent setting). Subsequent edges
    // are registered just-in-time when the agent enters them in
    // advance_agent().
    if (ctx_->congestion_map) {
        ctx_->congestion_map->add_agent(sp.edges.front(), step,
                                         a.arrival_step_next_node);
    }
    return true;
}

void TokenPassingSolver::advance_agent(AgentState& a, int step) {
    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) return;
    if (step < a.arrival_step_next_node) {
        // Still in transit on the current edge — count utilisation.
        ++active_steps_sum_;
        return;
    }

    // Arrived at the next node.
    // Track distance traveled on the edge we just completed.
    if (ctx_ && a.next_idx < static_cast<int>(a.current_path_edges.size())) {
        const auto& ways = ctx_->geo_box->data.ways;
        auto wit = ways.find(a.current_path_edges[a.next_idx]);
        if (wit != ways.end())
            instr_.record_edge_traversal(wit->second.distance_meters);
    }

    ++a.next_idx;
    a.current_node = a.current_path_nodes[a.next_idx];

    // Did we finish the path?
    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) {
        // Reached the objective.
        if (a.active_task_id >= 0 && a.active_task_id < static_cast<int>(tasks_.size())) {
            TaskRecord& t = tasks_[a.active_task_id];
            if (a.active_is_pickup_leg) {
                // Pickup completed. Now head to delivery (if not already
                // queued — we just unset active_task_id leg and start
                // delivery in the same step or the next).
                t.picked_step = step;
                a.in_flight_task_ids.push_back(a.active_task_id);
                if (static_cast<int>(a.in_flight_task_ids.size()) > a.capacity) {
                    ++capacity_violations_;
                }
                wait_sum_ += (t.picked_step - t.arrival_step);
                a.active_is_pickup_leg = false;
                // Start the delivery leg immediately.
                begin_leg(a, t.delivery_node, step);
            } else {
                // Delivery completed.
                if (t.picked_step < 0) {
                    ++pairing_violations_;
                }
                t.delivered_step = step;
                latency_sum_ += (t.delivered_step - t.arrival_step);
                trip_sum_    += (t.delivered_step - std::max(t.picked_step, t.arrival_step));
                if (t.pd_road_dist > 0.f) {
                    road_pd_sum_ += t.pd_road_dist;
                    ++road_pd_count_;
                }
                ++completed_;
                instr_.record_delivery(static_cast<int>(&a - agents_.data()));
                // Remove from in-flight list.
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
            // Path finished without active task — shouldn't happen in TP.
            a.current_path_nodes.clear();
            a.current_path_edges.clear();
            a.next_idx = 0;
        }
        return;
    }

    // More edges remain — schedule the next edge.
    a.current_edge_t_enter = step;
    const osmium::object_id_type next_edge = a.current_path_edges[a.next_idx];
    a.arrival_step_next_node = edge_arrival_step(next_edge, step);
    if (ctx_->congestion_map) {
        ctx_->congestion_map->add_agent(next_edge, step, a.arrival_step_next_node);
    }
    ++active_steps_sum_;
}

bool TokenPassingSolver::try_allocate_one(int step) {
    if (pending_task_ids_.empty()) return false;

    // Identify FREE agents (paper §3.1 — agent not currently executing a
    // task). With capacity = 1 paper-faithful: agents whose load (in-flight
    // tasks) is zero AND have no active in-progress journey.
    // For capacity > 1 (LGPDP extension): agents whose load < capacity.
    // We require the agent to be idle (no active path) too — TP's token
    // only goes to free agents.
    std::vector<int> free_idx;
    free_idx.reserve(agents_.size());
    for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
        const AgentState& a = agents_[i];
        if (a.active_task_id != -1) continue;          // currently moving
        if (!a.current_path_edges.empty() &&
            a.next_idx < static_cast<int>(a.current_path_edges.size())) continue;
        if (static_cast<int>(a.in_flight_task_ids.size()) >= a.capacity) continue;
        free_idx.push_back(i);
    }
    if (free_idx.empty()) return false;

    // For each pending task and each free agent, compute h(loc(a), s_τ)
    // and pick the (agent, task) pair with the globally minimum h.
    int   best_aid  = -1;
    int   best_tidx = -1;   // index into pending_task_ids_
    float best_h    = std::numeric_limits<float>::max();

    for (int ti = 0; ti < static_cast<int>(pending_task_ids_.size()); ++ti) {
        const int tid = pending_task_ids_[ti];
        const TaskRecord& t = tasks_[tid];
        for (int ai : free_idx) {
            const AgentState& a = agents_[ai];
            const SimplePath& sp = paths_->get(a.current_node, t.pickup_node);
            const float h = sp.valid ? sp.cost : paths_->heuristic(a.current_node, t.pickup_node);
            if (h < best_h) {
                best_h    = h;
                best_aid  = ai;
                best_tidx = ti;
            }
        }
    }

    if (best_aid < 0 || best_tidx < 0) return false;

    // Commit the allocation: remove task from pending, set up pickup leg.
    const int tid = pending_task_ids_[best_tidx];
    pending_task_ids_.erase(pending_task_ids_.begin() + best_tidx);
    AgentState& a = agents_[best_aid];
    TaskRecord& t = tasks_[tid];
    t.assigned_agent = best_aid;
    a.active_task_id = tid;
    a.active_is_pickup_leg = true;
    const bool ok = begin_leg(a, t.pickup_node, step);
    if (!ok) {
        // Path failed — fallback: put task back at the end of pending,
        // mark agent idle. This avoids an infinite-tight loop on
        // unreachable pickups.
        a.active_task_id = -1;
        pending_task_ids_.push_back(tid);
        return false;
    }
    return true;
}

void TokenPassingSolver::step(int timestep) {
    // Sample shared state (congestion, ghost traffic) for instrumentation.
    if (ctx_) {
        instr_.sample_congestion(ctx_->congestion_map);
        if (ctx_->ghost) instr_.sample_ghost(ctx_->ghost->n_active_now());
    }

    // Token-passing pass: keep trying to allocate while progress is made.
    // Multiple free agents may pick up multiple tasks at the same step.
    while (try_allocate_one(timestep)) { /* keep going */ }

    // Advance every agent along its current path one step.
    for (auto& a : agents_) {
        if (a.current_path_edges.empty()) continue;
        if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) continue;
        advance_agent(a, timestep);
    }
}

SolverMetrics TokenPassingSolver::finalize() {
    SolverMetrics m;
    m.tasks_appeared  = appeared_;
    m.tasks_completed = completed_;
    m.tasks_refused   = refused_;                      // TP never refuses; lifelong allocates eventually
    m.throughput_rate = (appeared_ > 0)
        ? std::min(1.f, static_cast<float>(completed_) / appeared_)
        : 0.f;
    m.accept_rate     = (appeared_ > 0)
        ? static_cast<float>(appeared_ - refused_) / appeared_
        : 0.f;
    if (completed_ > 0) {
        m.latency_mean   = static_cast<double>(latency_sum_) / completed_;
        m.mean_wait_steps = static_cast<double>(wait_sum_)   / completed_;
        m.mean_trip_steps = static_cast<double>(trip_sum_)   / completed_;
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
    // Latency per agent (paper §5 ratio metric).
    if (ctx_ && ctx_->n_active_agents > 0)
        m.latency_per_agent = m.latency_mean / ctx_->n_active_agents;
    instr_.finalize_into(m);
    return m;
}