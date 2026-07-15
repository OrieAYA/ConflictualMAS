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

void FaithfulCASolver::inject_task(const ScheduledTask& task, int step) {
    TaskRecord r;
    r.task_id        = static_cast<int>(tasks_.size());
    r.pickup_node    = task.pickup_node_id;
    r.delivery_node  = task.delivery_node_id;
    r.arrival_step   = step;
    // Pre-compute the FREE-FLOW pickup→delivery distance for the route
    // efficiency metric. We use a quick static BPR-A* at t=arrival_step
    // here; this is just for metrics, the real planning happens at
    // allocation time.
    BPRPath warm = bpr_a_star(r.pickup_node, r.delivery_node, step);
    if (warm.valid) {
        // Convert BPR time to a metric we can compare across solvers; we
        // store the road distance (sum of way lengths) instead so the
        // route_efficiency metric is comparable to TP/CA. Walk edges:
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

int FaithfulCASolver::edge_arrival_step(osmium::object_id_type edge_id,
                                         int t_enter) {
    if (!ctx_) return t_enter + 1;
    const auto& ways = ctx_->geo_box->data.ways;
    auto it = ways.find(edge_id);
    if (it == ways.end()) return t_enter + 1;
    const float length_m  = it->second.distance_meters;
    const float base_time = length_m / std::max(0.1f, ctx_->speed_mps);
    const float adj = ctx_->congestion_map
        ? ctx_->congestion_map->adjusted_cost(edge_id, base_time, length_m, t_enter)
        : base_time;
    // Record the BPR factor at the edge's ENTRY step (t_enter = current step at
    // all call sites). Must be sampled here, NOT at completion: CongestionMap::
    // advance() purges past steps, so a later query at the entry step reads
    // load 0 → BPR 1.0. Mirrors EpisodeRunner::schedule_next_edge.
    if (ctx_->congestion_map && base_time > 0.f)
        instr_.record_edge_bpr(adj / base_time);
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
    // Pickup leg.
    BPRPath leg1 = bpr_a_star(a.current_node, t.pickup_node, step);
    if (!leg1.valid) return std::numeric_limits<float>::max();

    // Delivery leg, starting at the predicted pickup arrival step.
    const int t_pickup = step + std::max(1, static_cast<int>(std::ceil(leg1.trip_time)));
    BPRPath leg2 = bpr_a_star(t.pickup_node, t.delivery_node, t_pickup);
    if (!leg2.valid) return std::numeric_limits<float>::max();

    const float trip = leg1.trip_time + leg2.trip_time;

    // γ-mode weighting (paper §3.4 γ_moving < γ_others, here γ_idle < γ_busy).
    const bool is_busy = !a.in_flight_task_ids.empty();
    const float gamma = is_busy ? hparams.gamma_busy : hparams.gamma_idle;
    return gamma * trip;
}

bool FaithfulCASolver::begin_leg_bpr(AgentState& a,
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
            return begin_leg_bpr(a, t.delivery_node, step);
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

    BPRPath p = bpr_a_star(a.current_node, target_node, step);
    if (!p.valid || p.nodes.size() < 2) return false;

    a.current_path_nodes = std::move(p.nodes);
    a.current_path_edges = std::move(p.edges);
    a.next_idx = 0;
    a.current_edge_t_enter = step;
    a.arrival_step_next_node = edge_arrival_step(a.current_path_edges.front(), step);
    // Congestion footprint is registered for the FULL remaining route by
    // recommit_route() (Option O commit_plan-style), not edge-by-edge here.
    return true;
}

void FaithfulCASolver::advance_agent(AgentState& a, int step) {
    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) return;
    if (step < a.arrival_step_next_node) {
        ++active_steps_sum_;
        return;
    }

    // Track distance on the edge just completed (BPR is sampled at ENTRY inside
    // edge_arrival_step — past steps are purged from the CongestionMap).
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
                // Re-plan delivery leg under current congestion (this is the
                // KEY behaviour of FaithfulCA — paper §3.2 has each agent
                // continuously re-evaluate its A* under current congestion).
                begin_leg_bpr(a, t.delivery_node, step);
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
        // Plan changed (stop reached) → refresh the full-route footprint.
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
    std::vector<osmium::object_id_type> stops;
    if (a.active_task_id >= 0 && a.active_task_id < static_cast<int>(tasks_.size())) {
        const TaskRecord& t = tasks_[a.active_task_id];
        if (a.active_is_pickup_leg) {
            stops.push_back(t.pickup_node);
            stops.push_back(t.delivery_node);
        } else {
            stops.push_back(t.delivery_node);
        }
    }
    commit_agent_route(
        *ctx_->congestion_map, *ctx_->geo_box, ctx_->speed_mps,
        a.current_node, stops, step, a.committed_occ,
        [this](osmium::object_id_type f, osmium::object_id_type to, int t) {
            return bpr_a_star(f, to, t).edges;
        });
}

bool FaithfulCASolver::try_allocate_one(int step) {
    if (pending_task_ids_.empty()) return false;

    // Eligibility: idle journey, capacity not full. Same as TP/CA.
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
            cands.push_back({ ai, ti, c, static_cast<int>(a.in_flight_task_ids.size()) });
        }
    }
    if (cands.empty()) return false;

    // Find global minimum cost.
    float best_cost = std::numeric_limits<float>::max();
    for (const auto& c : cands) if (c.cost < best_cost) best_cost = c.cost;
    const float band = best_cost * hparams.beta_tie_band;

    // β_π tie-break (paper §3.3 — "number of orders still to be fulfilled"):
    // within the cost band, prefer the agent with the LOWEST current load
    // (fewest in-flight tasks). Ties broken by lowest cost, then by lowest
    // (agent, task) ids for determinism.
    //
    // NOTE: this is β_π, not β_W. Earlier comments mistakenly said β_W —
    // paper's β_W gives priority to pickup/delivery (busy) agents over
    // wandering/finished ones, which is the opposite of what we want for
    // LGPDP allocation (we want to balance load by feeding less-loaded
    // agents next). β_π is the correct paper analogue for our use.
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

    // Commit allocation.
    const int tid = pending_task_ids_[best_tidx];
    pending_task_ids_.erase(pending_task_ids_.begin() + best_tidx);
    AgentState& a = agents_[best_aid];
    TaskRecord& t = tasks_[tid];
    t.assigned_agent = best_aid;
    a.active_task_id = tid;
    a.active_is_pickup_leg = true;
    const bool ok = begin_leg_bpr(a, t.pickup_node, step);
    if (!ok) {
        a.active_task_id = -1;
        pending_task_ids_.push_back(tid);
        return false;
    }
    recommit_route(a, step);   // register the full pickup→delivery route footprint
    return true;
}

void FaithfulCASolver::step(int timestep) {
    if (ctx_) {
        instr_.sample_congestion(ctx_->congestion_map);
        if (ctx_->ghost) instr_.sample_ghost(ctx_->ghost->n_active_now());
    }

    while (try_allocate_one(timestep)) { /* drain pending */ }

    for (auto& a : agents_) {
        if (a.current_path_edges.empty()) continue;
        if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) continue;
        advance_agent(a, timestep);
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