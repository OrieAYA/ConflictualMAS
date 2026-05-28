#include "RHCRSolver.hpp"
#include "Environment/Congestion/CongestionMap.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

void RHCRSolver::init(const SolverContext& ctx) {
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
    n_batches_ = n_replans_ = 0;
    instr_.init(ctx.n_active_agents);

    // Enforce h ≤ w invariant (paper §4).
    if (hparams.replan_period_h > hparams.time_horizon_w) {
        hparams.replan_period_h = hparams.time_horizon_w;
    }
    if (hparams.replan_period_h < 1) hparams.replan_period_h = 1;
}

void RHCRSolver::inject_task(const ScheduledTask& task, int step) {
    TaskRecord r;
    r.task_id        = static_cast<int>(tasks_.size());
    r.pickup_node    = task.pickup_node_id;
    r.delivery_node  = task.delivery_node_id;
    r.arrival_step   = step;
    // Metric-only pickup→delivery distance via a single-goal Multi-Label
    // A* (windowed BPR off because we're measuring a static distance).
    {
        std::vector<GoalEntry> single = { { r.task_id, r.delivery_node, false } };
        MultiLabelPath p = multi_label_a_star(r.pickup_node, single, step);
        if (p.valid) {
            const auto& ways = ctx_->geo_box->data.ways;
            for (auto eid : p.edges) {
                auto it = ways.find(eid);
                if (it != ways.end()) r.pd_road_dist += it->second.distance_meters;
            }
        }
    }
    tasks_.push_back(r);
    pending_task_ids_.push_back(r.task_id);
    ++appeared_;
}

int RHCRSolver::edge_arrival_step(osmium::object_id_type edge_id,
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

// ── Multi-Label A* (paper §4.1 Algorithm 1) ──────────────────────────────────
//
// Finds the shortest TIME path from `from` through `goal_seq` IN ORDER. State
// is (node, label) where label = number of goals visited so far. Edges use
// WINDOWED BPR cost: BPR-adjusted for edges entered in [start_step,
// start_step + w], free-flow time for edges entered after.
//
// Goal test: state with label == |goal_seq|.
// Admissible heuristic: euclidean remaining-travel-time through unvisited
// goals, divided by speed_mps (lower bound under zero congestion).
RHCRSolver::MultiLabelPath
RHCRSolver::multi_label_a_star(osmium::object_id_type from,
                                 const std::vector<GoalEntry>& goal_seq,
                                 int start_step) const {
    MultiLabelPath result;
    if (!ctx_ || !ctx_->geo_box || !ctx_->pathfinder) return result;
    if (from == 0) return result;
    if (goal_seq.empty()) {
        result.valid = true;
        result.nodes = { from };
        return result;
    }

    const auto& nodes = ctx_->geo_box->data.nodes;
    const auto& ways  = ctx_->geo_box->data.ways;
    const float speed = std::max(0.1f, ctx_->speed_mps);
    const int   w     = std::max(1, hparams.time_horizon_w);
    const int   t_window_end = start_step + w;

    // Cap the per-goal heuristic: total remaining travel through unvisited
    // goals. Pre-compute the suffix sum from label k to |goal_seq|.
    std::vector<float> h_suffix(goal_seq.size() + 1, 0.f);
    for (int k = static_cast<int>(goal_seq.size()) - 1; k > 0; --k) {
        const float dist_m = const_cast<Pathfinder*>(ctx_->pathfinder)
            ->heuristic(goal_seq[k - 1].node, goal_seq[k].node);
        h_suffix[k - 1] = h_suffix[k] + dist_m / speed;
    }

    auto h_value = [&](osmium::object_id_type n, int label) -> float {
        if (label >= static_cast<int>(goal_seq.size())) return 0.f;
        const float to_next_m = const_cast<Pathfinder*>(ctx_->pathfinder)
            ->heuristic(n, goal_seq[label].node);
        return to_next_m / speed + h_suffix[label];
    };

    struct State {
        osmium::object_id_type node;
        int label;
        bool operator==(const State& o) const {
            return node == o.node && label == o.label;
        }
    };
    struct StateHash {
        std::size_t operator()(const State& s) const noexcept {
            const auto a = std::hash<osmium::object_id_type>{}(s.node);
            const auto b = std::hash<int>{}(s.label);
            return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
        }
    };

    struct OpenEntry {
        float f;
        float g;
        int   t_arrive;
        State s;
        bool operator>(const OpenEntry& o) const { return f > o.f; }
    };

    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    std::unordered_map<State, float, StateHash> g_score;
    std::unordered_map<State,
        std::tuple<State, osmium::object_id_type>, StateHash> came_from;
    std::unordered_set<State, StateHash> closed;

    State start{ from, 0 };
    // If `from` happens to be the first goal, increment label immediately.
    if (!goal_seq.empty() && goal_seq[0].node == from) start.label = 1;

    g_score[start] = 0.f;
    open.push({ h_value(from, start.label), 0.f, start_step, start });

    int expansions = 0;
    while (!open.empty()) {
        OpenEntry cur = open.top();
        open.pop();
        if (closed.count(cur.s)) continue;
        closed.insert(cur.s);

        if (cur.s.label >= static_cast<int>(goal_seq.size())) {
            // Reconstruct path.
            std::vector<osmium::object_id_type> rev_nodes;
            std::vector<osmium::object_id_type> rev_edges;
            State cs = cur.s;
            rev_nodes.push_back(cs.node);
            while (!(cs == start)) {
                auto it = came_from.find(cs);
                if (it == came_from.end()) break;
                rev_edges.push_back(std::get<1>(it->second));
                cs = std::get<0>(it->second);
                rev_nodes.push_back(cs.node);
            }
            std::reverse(rev_nodes.begin(), rev_nodes.end());
            std::reverse(rev_edges.begin(), rev_edges.end());
            result.nodes = std::move(rev_nodes);
            result.edges = std::move(rev_edges);
            result.trip_time = cur.g;
            result.valid = true;
            return result;
        }

        if (hparams.max_expansions > 0 && ++expansions > hparams.max_expansions) {
            break;
        }

        auto node_it = nodes.find(cur.s.node);
        if (node_it == nodes.end()) continue;
        const auto& pt = node_it->second;

        for (auto edge_id : pt.incident_ways) {
            auto wit = ways.find(edge_id);
            if (wit == ways.end()) continue;
            const auto& w_obj = wit->second;
            const osmium::object_id_type nbr =
                (w_obj.node1_id == cur.s.node) ? w_obj.node2_id : w_obj.node1_id;
            if (nbr == 0) continue;

            const float length_m = w_obj.distance_meters;
            if (length_m <= 0.f) continue;
            const float base_time = length_m / speed;

            // ── WINDOWED COST (paper §4.2) ─────────────────────────────────
            // Edges entered within [start_step, start_step + w] pay BPR cost.
            // Edges entered after pay free-flow only.
            float step_cost;
            if (cur.t_arrive < t_window_end && ctx_->congestion_map) {
                step_cost = ctx_->congestion_map->adjusted_cost(
                    edge_id, base_time, length_m, cur.t_arrive);
            } else {
                step_cost = base_time;
            }

            // Label transition: incremented if nbr is the NEXT goal we're
            // heading to.
            int new_label = cur.s.label;
            if (new_label < static_cast<int>(goal_seq.size()) &&
                goal_seq[new_label].node == nbr) {
                ++new_label;
            }

            State new_s{ nbr, new_label };
            if (closed.count(new_s)) continue;

            const float tentative_g = cur.g + step_cost;
            auto gs = g_score.find(new_s);
            if (gs != g_score.end() && tentative_g >= gs->second) continue;

            g_score[new_s] = tentative_g;
            came_from[new_s] = std::make_tuple(cur.s, edge_id);
            const int new_t = cur.t_arrive +
                std::max(1, static_cast<int>(std::ceil(step_cost)));
            const float f = tentative_g + h_value(nbr, new_label);
            open.push({ f, tentative_g, new_t, new_s });
        }
    }

    return result;  // valid = false
}

float RHCRSolver::insertion_cost(const AgentState& a,
                                   osmium::object_id_type pickup_node,
                                   osmium::object_id_type delivery_node,
                                   int task_id,
                                   int step) const {
    // Trip cost WITHOUT the new task — Multi-Label A* through current
    // goal_queue.
    float base_cost = 0.f;
    if (!a.goal_queue.empty()) {
        MultiLabelPath base = multi_label_a_star(a.current_node, a.goal_queue, step);
        if (!base.valid) return std::numeric_limits<float>::max();
        base_cost = base.trip_time;
    }

    // Trip cost WITH the new task appended (pickup then delivery at the
    // end of the queue). Paper §4 — RHCR appends new goal locations.
    std::vector<GoalEntry> extended = a.goal_queue;
    extended.push_back({ task_id, pickup_node, true });
    extended.push_back({ task_id, delivery_node, false });

    MultiLabelPath ext = multi_label_a_star(a.current_node, extended, step);
    if (!ext.valid) return std::numeric_limits<float>::max();
    return ext.trip_time - base_cost;
}

void RHCRSolver::allocate_batch(int step) {
    if (pending_task_ids_.empty()) return;
    ++n_batches_;

    // Greedy insertion-cost assignment with PER-AGENT BASE CACHE (Optim A).
    //
    // For each pending task, find the agent with the lowest marginal cost
    // of appending (pickup, delivery). The base path of each agent (Multi-
    // Label A* through its current goal_queue) is INVARIANT across tasks
    // within this batch — compute it ONCE per agent, then for each task
    // compute only the suffix cost (base_end_node → P → D).
    //
    // Mathematically equivalent to the previous "full extended A*" approach
    // because Multi-Label A* visits goals strictly in order — the optimal
    // path through [g_1, ..., g_n, P, D] is the optimal path through
    // [g_1, ..., g_n] concatenated with the optimal path from g_n to P to D.
    // (The user's new task is appended at the END, never inserted earlier.)
    std::vector<int> remaining = pending_task_ids_;
    pending_task_ids_.clear();

    const int n_agents = static_cast<int>(agents_.size());

    // ── Per-agent base cache (Optim A) ──────────────────────────────────────
    // base_valid[ai]    : whether the base path was computed and is reachable
    // base_cost[ai]     : Multi-Label A* trip_time through current goal_queue
    // base_end_node[ai] : last goal node (or current_node if queue empty)
    // base_end_step[ai] : predicted simulation step at end of base
    std::vector<bool>   base_valid(n_agents, false);
    std::vector<float>  base_cost(n_agents, 0.f);
    std::vector<osmium::object_id_type> base_end_node(n_agents, 0);
    std::vector<int>    base_end_step(n_agents, step);

    for (int ai = 0; ai < n_agents; ++ai) {
        const AgentState& a = agents_[ai];
        if (a.goal_queue.empty()) {
            // Empty queue — base is trivially 0, end is current location.
            base_valid[ai]    = true;
            base_cost[ai]     = 0.f;
            base_end_node[ai] = a.current_node;
            base_end_step[ai] = step;
            continue;
        }
        MultiLabelPath base = multi_label_a_star(a.current_node, a.goal_queue, step);
        if (!base.valid) {
            base_valid[ai] = false;
            continue;
        }
        base_valid[ai]    = true;
        base_cost[ai]     = base.trip_time;
        base_end_node[ai] = a.goal_queue.back().node;
        base_end_step[ai] = step +
            std::max(1, static_cast<int>(std::ceil(base.trip_time)));
    }

    for (int tid : remaining) {
        TaskRecord& t = tasks_[tid];

        int   best_aid  = -1;
        float best_cost = std::numeric_limits<float>::max();
        for (int ai = 0; ai < n_agents; ++ai) {
            const AgentState& a = agents_[ai];
            if (static_cast<int>(a.in_flight_task_ids.size()) >= a.capacity) continue;
            if (!base_valid[ai]) continue;

            // Suffix cost: from end of base, head to P then D.
            // Use a TWO-GOAL Multi-Label A* (much cheaper than the full
            // extended path through the whole queue).
            std::vector<GoalEntry> suffix = {
                { tid, t.pickup_node,   true  },
                { tid, t.delivery_node, false }
            };
            MultiLabelPath suf = multi_label_a_star(
                base_end_node[ai], suffix, base_end_step[ai]);
            if (!suf.valid) continue;

            // Marginal cost is just the suffix trip_time (the base is
            // shared across all candidates and cancels out in the argmin).
            const float c = suf.trip_time;
            if (c < best_cost) {
                best_cost = c;
                best_aid  = ai;
            }
        }

        if (best_aid < 0) {
            pending_task_ids_.push_back(tid);
            continue;
        }

        AgentState& a = agents_[best_aid];
        a.goal_queue.push_back({ tid, t.pickup_node, true });
        a.goal_queue.push_back({ tid, t.delivery_node, false });
        a.needs_replan = true;
        t.assigned_agent = best_aid;

        // ── Optim A continued ───────────────────────────────────────────────
        // The chosen agent's base path changed (queue grew by 2 goals).
        // Update its cache so subsequent task allocations within THIS batch
        // use the new base — this preserves correctness for batches with
        // multiple tasks all going to the same agent.
        base_cost[best_aid]    += best_cost;
        base_end_node[best_aid] = t.delivery_node;
        base_end_step[best_aid] = base_end_step[best_aid] +
            std::max(1, static_cast<int>(std::ceil(best_cost)));
        // base_valid[best_aid] stays true.
    }
}

void RHCRSolver::replan_all_agents(int step) {
    // ── Optim B: skip agents whose goal_queue is unchanged. ─────────────────
    // needs_replan is set by allocate_batch() when a new task is appended
    // and by advance_agent() when a goal is popped. If neither happened
    // for an agent, its committed guide path is still valid (BPR cost is
    // applied at edge traversal time, so congestion is still "felt").
    //
    // The paper recommends replanning ALL agents every h steps, but on OSM
    // continuous routing this re-runs Multi-Label A* on the full goal
    // sequence even when nothing changed — pure waste. The fidelity-
    // preserving compromise: replan only agents that have queue changes.
    // Ghost traffic adaptation is preserved at the COST level (BPR per
    // edge) even without re-routing.
    bool any_replan_needed = false;
    for (const auto& a : agents_) {
        if (a.needs_replan && !a.goal_queue.empty()) {
            any_replan_needed = true;
            break;
        }
        if (a.current_path_edges.empty() && !a.goal_queue.empty()) {
            any_replan_needed = true;
            break;
        }
    }
    if (!any_replan_needed) return;

    for (auto& a : agents_) {
        // Skip agents without queue changes — they keep their existing path.
        if (!a.needs_replan &&
            !a.current_path_edges.empty() &&
            a.next_idx < static_cast<int>(a.current_path_edges.size())) {
            continue;
        }

        // ── FIX (OSM continuous routing): skip replan if the agent is
        // currently mid-edge. On OSM, a single edge takes many steps to
        // traverse (length / speed); replanning while in transit would
        // wipe the path and reset next_idx=0, destroying the edge progress
        // the agent has already made. Paper RHCR assumes grid MAPF where
        // each step = one cell move, so this issue doesn't arise.
        //
        // Defer the agent's replan to the next h-boundary AFTER it
        // arrives at the next node. The goal_queue update from
        // allocate_batch is still applied — just the path computation
        // waits until the agent is at a node and safe to re-route.
        const bool agent_in_transit =
            !a.current_path_edges.empty() &&
            a.next_idx < static_cast<int>(a.current_path_edges.size()) &&
            step < a.arrival_step_next_node;
        if (agent_in_transit) {
            // Mark for lazy replan next time the agent arrives at a node.
            a.needs_replan = true;
            continue;
        }

        // ── Fix: fire any pending goal events at the agent's current
        // location before planning. This handles the edge case where the
        // newly-allocated pickup is at the agent's current node (paper
        // §4.1 Multi-Label A* would bump the label without firing the
        // event otherwise).
        while (!a.goal_queue.empty() &&
               a.current_node == a.goal_queue.front().node) {
            const GoalEntry g = a.goal_queue.front();
            a.goal_queue.erase(a.goal_queue.begin());
            if (g.task_id < 0 || g.task_id >= static_cast<int>(tasks_.size()))
                continue;
            TaskRecord& t = tasks_[g.task_id];
            if (g.is_pickup) {
                t.picked_step = step;
                a.in_flight_task_ids.push_back(g.task_id);
                if (static_cast<int>(a.in_flight_task_ids.size()) > a.capacity) {
                    ++capacity_violations_;
                }
                wait_sum_ += (t.picked_step - t.arrival_step);
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
                                    g.task_id);
                if (it != a.in_flight_task_ids.end()) a.in_flight_task_ids.erase(it);
            }
        }

        if (a.goal_queue.empty()) {
            // No goals — clear any stale path.
            a.current_path_nodes.clear();
            a.current_path_edges.clear();
            a.next_idx = 0;
            a.needs_replan = false;
            continue;
        }
        // Replan from current_node through the full remaining goal queue.
        MultiLabelPath p = multi_label_a_star(a.current_node, a.goal_queue, step);
        if (!p.valid) {
            // Path failed — keep the existing one if any; mark for retry
            // next batch.
            continue;
        }
        a.current_path_nodes = std::move(p.nodes);
        a.current_path_edges = std::move(p.edges);
        a.next_idx = 0;
        a.needs_replan = false;
        if (!a.current_path_edges.empty()) {
            a.arrival_step_next_node = edge_arrival_step(a.current_path_edges.front(), step);
            if (ctx_->congestion_map) {
                ctx_->congestion_map->add_agent(a.current_path_edges.front(),
                                                 step, a.arrival_step_next_node);
            }
        }
        ++n_replans_;
    }
}

void RHCRSolver::advance_agent(AgentState& a, int step) {
    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) return;
    if (step < a.arrival_step_next_node) {
        ++active_steps_sum_;
        return;
    }

    // Track distance traveled on the edge just completed.
    if (ctx_ && a.next_idx < static_cast<int>(a.current_path_edges.size())) {
        const auto& ways = ctx_->geo_box->data.ways;
        auto wit = ways.find(a.current_path_edges[a.next_idx]);
        if (wit != ways.end())
            instr_.record_edge_traversal(wit->second.distance_meters);
    }

    ++a.next_idx;
    a.current_node = a.current_path_nodes[a.next_idx];

    // Check if we've arrived at the head of goal_queue. The path passes
    // through goals in order, so the FRONT of goal_queue is the next
    // objective; when current_node matches it, fire the event.
    while (!a.goal_queue.empty() && a.current_node == a.goal_queue.front().node) {
        const GoalEntry g = a.goal_queue.front();
        a.goal_queue.erase(a.goal_queue.begin());

        if (g.task_id < 0 || g.task_id >= static_cast<int>(tasks_.size())) continue;
        TaskRecord& t = tasks_[g.task_id];
        if (g.is_pickup) {
            t.picked_step = step;
            a.in_flight_task_ids.push_back(g.task_id);
            if (static_cast<int>(a.in_flight_task_ids.size()) > a.capacity) {
                ++capacity_violations_;
            }
            wait_sum_ += (t.picked_step - t.arrival_step);
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
                                g.task_id);
            if (it != a.in_flight_task_ids.end()) a.in_flight_task_ids.erase(it);
        }
    }

    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) {
        // Reached end of path. Goals all processed (or this was a transient
        // edge between agent's location and the first goal that just popped).
        a.current_path_nodes.clear();
        a.current_path_edges.clear();
        a.next_idx = 0;
        // If goals remain, mark for replan at next batch boundary.
        if (!a.goal_queue.empty()) a.needs_replan = true;
        return;
    }

    const osmium::object_id_type next_edge = a.current_path_edges[a.next_idx];
    a.arrival_step_next_node = edge_arrival_step(next_edge, step);
    if (ctx_->congestion_map) {
        ctx_->congestion_map->add_agent(next_edge, step, a.arrival_step_next_node);
    }
    ++active_steps_sum_;
}

void RHCRSolver::step(int timestep) {
    if (ctx_) {
        instr_.sample_congestion(ctx_->congestion_map);
        if (ctx_->ghost) instr_.sample_ghost(ctx_->ghost->n_active_now());
    }

    // ── FIX (OSM continuous routing): advance BEFORE replanning. ────────────
    // Paper RHCR assumes grid MAPF where step movement is atomic (one cell
    // per step), so the order doesn't matter. On OSM continuous routing
    // each edge spans many steps, and replanning resets next_idx=0 which
    // would wipe partial edge progress. By advancing first:
    //   1. agents arriving at the next node at exactly this step fire their
    //      goal events (via advance_agent → the while loop on goal_queue.front),
    //   2. then replan_all_agents sees their updated current_node and queue,
    //   3. the transit-protect guard in replan_all_agents catches agents
    //      still mid-edge and skips them, preserving their progress.
    for (auto& a : agents_) {
        if (a.current_path_edges.empty()) continue;
        if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) continue;
        advance_agent(a, timestep);
    }

    // ── Rolling-horizon trigger (paper §4) ──────────────────────────────────
    // Every h steps: batch-allocate pending tasks, then replan ALL agents
    // through their (possibly updated) goal queues.
    const bool is_h_boundary =
        (timestep % std::max(1, hparams.replan_period_h)) == 0;
    if (is_h_boundary) {
        allocate_batch(timestep);
        replan_all_agents(timestep);
    } else {
        // Mid-window: if an agent's queue changed (e.g., it just finished
        // a goal) and it has more work, lazily replan that one agent only.
        for (auto& a : agents_) {
            // Same transit-protect guard as in replan_all_agents.
            const bool agent_in_transit =
                !a.current_path_edges.empty() &&
                a.next_idx < static_cast<int>(a.current_path_edges.size()) &&
                timestep < a.arrival_step_next_node;
            if (agent_in_transit) continue;
            if (a.needs_replan && !a.goal_queue.empty()) {
                MultiLabelPath p = multi_label_a_star(a.current_node, a.goal_queue, timestep);
                if (p.valid) {
                    a.current_path_nodes = std::move(p.nodes);
                    a.current_path_edges = std::move(p.edges);
                    a.next_idx = 0;
                    a.needs_replan = false;
                    if (!a.current_path_edges.empty()) {
                        a.arrival_step_next_node =
                            edge_arrival_step(a.current_path_edges.front(), timestep);
                        if (ctx_->congestion_map) {
                            ctx_->congestion_map->add_agent(a.current_path_edges.front(),
                                                             timestep,
                                                             a.arrival_step_next_node);
                        }
                    }
                    ++n_replans_;
                }
            }
        }
    }
    // Advance loop moved to the TOP of step() to fix the OSM-continuous
    // edge-progress wipe bug. See the "FIX" comment at the start of step().
}

SolverMetrics RHCRSolver::finalize() {
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
    m.n_replans = n_replans_;
    if (ctx_ && ctx_->n_active_agents > 0)
        m.latency_per_agent = m.latency_mean / ctx_->n_active_agents;
    instr_.finalize_into(m);
    return m;
}