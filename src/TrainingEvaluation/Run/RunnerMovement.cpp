#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Agents/TaskAgent.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// ── Edge-by-edge movement ─────────────────────────────────────────────────────

int EpisodeRunner::start_leg(int agent_id, int task_id, bool is_pickup,
                             osmium::object_id_type from,
                             osmium::object_id_type to,
                             int current_step) {
    DeliveryAgent* agent = memory_.get_delivery_agent(agent_id);
    if (!agent) return current_step + 1;

    const ObjectivePath* path = memory_.get_or_compute_path(from, to, 1);

    agent->begin_leg(path, task_id, is_pickup);

    // Protocol: the agent knows the current leg + one ahead, no further.
    agent->prefetch_next_path(memory_);

    return schedule_next_edge(agent_id, current_step);
}

int EpisodeRunner::schedule_next_edge(int agent_id, int current_step) {
    DeliveryAgent* agent = memory_.get_delivery_agent(agent_id);
    if (!agent || !agent->edge_cursor) return current_step + 1;

    EdgeCursor& cur = *agent->edge_cursor;

    osmium::object_id_type dest_node = cur.next_node();
    osmium::object_id_type edge_id   = cur.current_edge_id();

    float dist = 0.f;
    if (edge_id != 0) {
        auto it = memory_.geo_box.data.ways.find(edge_id);
        if (it != memory_.geo_box.data.ways.end())
            dist = it->second.distance_meters;
    }
    if (dist <= 0.f) {
        dist = fallback_cost(agent->current_node, dest_node);
    }

    // Traversal rule: the agent pays the BPR time of the n OTHERS on the edge,
    // excluding its own committed weight (commit_plan ran before this call).
    const int self_w = std::max(1, memory_.congestion_map.params.load_per_agent);
    int steps;
    float eff_dist = dist;
    if (edge_id != 0 && dist > 0.f) {
        steps = memory_.congestion_map.traversal_steps(
            edge_id, dist, current_step, cfg_.speed_mps, self_w);
        eff_dist = memory_.congestion_map.adjusted_cost(
            edge_id, dist, dist, current_step, self_w);
    } else {
        steps = std::max(1, static_cast<int>(std::ceil(dist / cfg_.speed_mps)));
    }
    int arrival = current_step + steps;

    bool is_last = cur.is_last_edge();

    // Real-impact accumulators, sampled at edge entry: the slowdown here is
    // the delay the agent actually pays (eff_dist drives `steps`).
    if (edge_id != 0 && dist > 0.f) {
        const float bpr_factor = eff_dist / dist;   // 1.0 = no slowdown
        acc_.bpr_along_route_sum         += bpr_factor;
        acc_.bpr_along_route_count       += 1;
        const float extra_dist = std::max(0.f, eff_dist - dist);
        acc_.time_lost_to_congestion_sum += extra_dist / std::max(0.1f, cfg_.speed_mps);
        // Load >= 5 at entry = real chokepoint.
        const int load_now = memory_.congestion_map.get_load(edge_id, current_step);
        if (load_now >= 5) ++acc_.n_traversals_in_jam;
    }

    agent->start_edge(edge_id, dest_node, arrival);
    arrivals_.push_back({ agent_id, cur.task_id, cur.is_pickup, is_last, arrival });
    return arrival;
}

// First edge of the agent's current leg path, oriented from its position.
static osmium::object_id_type first_planned_edge(const DeliveryAgent& agent) {
    const ObjectivePath* p = agent.local_memory.current_path;
    if (!p || !p->valid() || p->edges.empty() || p->nodes.size() < 2) return 0;
    const bool forward = (p->nodes.front() == agent.current_node)
                      || (p->nodes.back()  != agent.current_node);
    return forward ? p->edges.front() : p->edges.back();
}

void EpisodeRunner::on_objective_reached(int agent_id, int task_id,
                                         bool is_pickup, int current_step,
                                         osmium::object_id_type came_from) {
    DeliveryAgent* agent = memory_.get_delivery_agent(agent_id);
    PDPTask*       task  = memory_.get_task(task_id);
    if (!agent || !task) return;

    if (cfg_.use_movement_policy && cfg_.movement_train)
        movement_policy().credit_leg_outcome(agent_id, current_step,
                                             cfg_.mp_outcome_w);

    agent->solution.advance();
    agent->edge_cursor.reset();

    if (is_pickup) {
        task->mark_picked(current_step);

        // Response delay: task appearance → pickup reached.
        if (task->timeline.created_step >= 0)
            acc_.wait_sum   += current_step - task->timeline.created_step,
            acc_.wait_count += 1;

        // Partial credit at pickup: cuts the credit-assignment delay (the
        // delivery may complete 500-1500 steps later).
        if (active_policy_) {
            auto it = task_accept_buf_idx_.find(task_id);
            if (it != task_accept_buf_idx_.end()) {
                float pickup_credit = cfg_.pickup_reward_frac
                    * task->reward_original * task->importance_original
                    / scales_.task_value;
                // Same latency factor as delivery, measured from acceptance.
                if (cfg_.rs_pickup_latency && task->timeline.allocated_step >= 0)
                    pickup_credit *= latency_phi(
                        current_step - task->timeline.allocated_step,
                        cfg_.latency_shaping_lambda,
                        cfg_.latency_shaping_max_steps);
                active_policy_->add_to_reward(task->agent_id, it->second,
                                              pickup_credit);
            }
        }

        agent->promote_next_path();

        // Congestion reroute at the node boundary: refresh the upcoming leg
        // if traffic makes another route >= 5% faster. With the movement
        // policy on, the learned gate replaces the unconditional check.
        if (!cfg_.use_movement_policy)
            memory_.push_rerouted_path(agent_id, cfg_.speed_mps);
        else if (!agent->solution.empty())
            movement_decision(agent_id, current_step,
                              first_planned_edge(*agent), came_from);

        if (!agent->solution.empty()) {
            // The next objective is not necessarily this task's delivery: the
            // planner may interleave another task's pickup between P and D.
            const ObjectiveNode& next_obj = agent->solution.next_objective();
            int  next_task_id = -1;
            bool next_pickup  = false;
            resolve_next_leg(*agent, next_obj, next_task_id, next_pickup);

            const ObjectivePath* next_path = agent->local_memory.current_path;
            if (!next_path)
                next_path = memory_.get_or_compute_path(
                    agent->current_node, next_obj.id, 1);
            agent->begin_leg(next_path, next_task_id, next_pickup);
            agent->prefetch_next_path(memory_);

            memory_.commit_plan(agent_id, cfg_.speed_mps);   // before schedule (self-exclusion)
            schedule_next_edge(agent_id, current_step);
        }

    } else {
        // A task is delivered exactly once; this guard protects the
        // delivery-side metrics (throughput <= 1) against any re-reach.
        const bool first_delivery = (task->timeline.delivered_step < 0);
        task->mark_delivered(current_step);

        int latency = current_step - task->timeline.created_step;
        if (first_delivery) {
        acc_.latency_sum   += latency;
        acc_.latency_count += 1;

        acc_.value_delivered_sum += static_cast<double>(task->reward_original)
                              * task->importance_original;

        // Pickup→delivery traversal time (excludes pre-allocation wait).
        if (task->timeline.picked_step >= 0) {
            acc_.trip_sum   += current_step - task->timeline.picked_step;
            acc_.trip_count += 1;
        }

        // Delivery efficiency = reward per km of the pickup→delivery route,
        // rescaled into [0,1] for GlobalState::avg_efficiency.
        const auto* pu_del = memory_.get_or_compute_path(
            task->pickup.id, task->delivery.id, task->delivery.group_id);
        const float dist_m = (pu_del && pu_del->valid()) ? pu_del->cost : 1.f;
        if (dist_m > 1.f) {
            const float eff = task->reward_original * 1000.f / dist_m;
            acc_.efficiency_sum   += static_cast<double>(eff);
            acc_.efficiency_count += 1;
            acc_.road_pd_sum      += static_cast<double>(dist_m);
            acc_.road_pd_count    += 1;
        }
        }

        // Remaining delivery credit (paper eq. 6 latency factor). Uses the
        // immutable *_original values so recall-time reward boosts cannot be
        // gamed by refuse-then-accept.
        if (active_policy_) {
            auto it = task_accept_buf_idx_.find(task_id);
            if (it != task_accept_buf_idx_.end()) {
                float shape_factor = 1.f;
                if (cfg_.enable_latency_shaping
                    && task->timeline.picked_step >= 0
                    && cfg_.latency_shaping_max_steps > 0)
                {
                    const float trip = static_cast<float>(
                        current_step - task->timeline.picked_step);
                    const float trip_norm = std::min(1.f,
                        trip / static_cast<float>(cfg_.latency_shaping_max_steps));
                    shape_factor = std::max(0.f,
                        1.f - cfg_.latency_shaping_lambda * trip_norm);
                }
                const float delivery_credit = (1.f - cfg_.pickup_reward_frac)
                    * task->reward_original * task->importance_original
                    * shape_factor / scales_.task_value;
                active_policy_->add_to_reward(task->agent_id, it->second,
                                              delivery_credit);
                task_accept_buf_idx_.erase(it);
            }
        }

        // Completes the task (agent goes Idle if no tasks remain).
        memory_.task_agent.on_task_event(task_id, TaskEvent::Finished, memory_);

        if (!agent->solution.empty()) {
            agent->promote_next_path();

            if (!cfg_.use_movement_policy)
                memory_.push_rerouted_path(agent_id, cfg_.speed_mps);
            else
                movement_decision(agent_id, current_step,
                                  first_planned_edge(*agent), came_from);

            const ObjectivePath* next_path = agent->local_memory.current_path;
            const ObjectiveNode& next_obj  = agent->solution.next_objective();

            if (!next_path)
                next_path = memory_.get_or_compute_path(
                    agent->current_node, next_obj.id, 1);

            int  next_task_id = -1;
            bool next_pickup  = false;
            resolve_next_leg(*agent, next_obj, next_task_id, next_pickup);
            agent->begin_leg(next_path, next_task_id, next_pickup);
            agent->prefetch_next_path(memory_);
            memory_.commit_plan(agent_id, cfg_.speed_mps);
            schedule_next_edge(agent_id, current_step);
        }
    }
}

// Disambiguated by task STATE: a not-yet-picked task at this node ⇒ pickup
// leg; a picked, not-yet-delivered task ⇒ delivery leg. The global node→task
// map is only a last-resort fallback (wrong when two tasks share a node).
void EpisodeRunner::resolve_next_leg(const DeliveryAgent& agent,
                                     const ObjectiveNode& next_obj,
                                     int& next_task_id, bool& next_pickup) {
    next_task_id = -1;
    next_pickup  = false;
    for (const PDPTask* t : agent.local_memory.tasks) {
        if (!t) continue;
        if (t->timeline.picked_step < 0 && t->pickup.id == next_obj.id) {
            next_task_id = t->task_id; next_pickup = true;  return;
        }
        if (t->timeline.picked_step >= 0 && t->timeline.delivered_step < 0
            && t->delivery.id == next_obj.id) {
            next_task_id = t->task_id; next_pickup = false; return;
        }
    }
    PDPTask* nt = memory_.get_task_for_node(next_obj.id);
    next_task_id = nt ? nt->task_id : -1;
    next_pickup  = nt && (nt->pickup.id == next_obj.id);
}

float EpisodeRunner::fallback_cost(osmium::object_id_type from,
                                   osmium::object_id_type to) const {
    const auto& nodes = memory_.geo_box.data.nodes;
    auto ia = nodes.find(from);
    auto ib = nodes.find(to);
    if (ia == nodes.end() || ib == nodes.end()) return kCostScale;
    double d = calculate_haversine_distance(ia->second.lat, ia->second.lon,
                                            ib->second.lat, ib->second.lon);
    return static_cast<float>(d) * 1.4f;
}

// ── Arrival processing ────────────────────────────────────────────────────────

void EpisodeRunner::process_arrivals(int current_step) {
    // Index-based loop: on_objective_reached() may push entries beyond n.
    const int n = static_cast<int>(arrivals_.size());
    for (int i = 0; i < n; ++i) {
        if (arrivals_[i].arrival_step > current_step) continue;

        // Copy before any potential reallocation.
        const int  agent_id     = arrivals_[i].agent_id;
        const int  task_id      = arrivals_[i].task_id;
        const bool is_pickup    = arrivals_[i].is_pickup;
        const bool is_objective = arrivals_[i].is_objective;

        DeliveryAgent* agent = memory_.get_delivery_agent(agent_id);
        if (!agent) { arrivals_[i].arrival_step = -1; continue; }

        const osmium::object_id_type came_from =
            agent->in_transit ? agent->in_transit->edge_id : 0;

        agent->arrive_at_node();

        if (!is_objective) {
            // Intermediate road node: next edge of the same leg.
            if (agent->edge_cursor)
                agent->edge_cursor->next_idx++;
            if (cfg_.use_movement_policy && agent->edge_cursor
                && agent->edge_cursor->has_more_edges())
                movement_decision(agent_id, current_step,
                                  agent->edge_cursor->current_edge_id(),
                                  came_from);
            schedule_next_edge(agent_id, current_step);
        } else {
            on_objective_reached(agent_id, task_id, is_pickup, current_step,
                                 came_from);
        }

        arrivals_[i].arrival_step = -1;
    }

    arrivals_.erase(
        std::remove_if(arrivals_.begin(), arrivals_.end(),
            [](const ScheduledArrival& a){ return a.arrival_step < 0; }),
        arrivals_.end());
}

// ── Movement decision policy ──────────────────────────────────────────────────

void EpisodeRunner::movement_decision(int agent_id, int current_step,
                                      osmium::object_id_type next_edge,
                                      osmium::object_id_type came_from) {
    DeliveryAgent* agent = memory_.get_delivery_agent(agent_id);
    if (!agent || next_edge == 0 || agent->solution.empty()) return;

    auto node_it = memory_.geo_box.data.nodes.find(agent->current_node);
    if (node_it == memory_.geo_box.data.nodes.end()) return;

    const auto& plan_cong = agent->local_memory.plan_cong;
    auto cong_now = [&](osmium::object_id_type way_id) -> float {
        auto wit = memory_.geo_box.data.ways.find(way_id);
        const float dist = (wit != memory_.geo_box.data.ways.end())
            ? wit->second.distance_meters : 0.f;
        const float cap = memory_.congestion_map.edge_capacity(dist);
        const float x = memory_.congestion_map.get_load(way_id, current_step)
                        / std::max(1.f, cap);
        return x / (1.f + x);
    };
    auto plan_of = [&](osmium::object_id_type way_id) -> float {
        auto it = plan_cong.find(way_id);
        return it != plan_cong.end() ? it->second : 0.f;
    };

    MoveObs o;
    float min_other = 1.f;   // 1 = no visible alternative
    int   slot      = 0;
    auto put = [&](osmium::object_id_type way_id) {
        if (slot >= kMoveEdgeSlots) return;
        float* e = o.x.data() + kMoveSelfFeat + slot * kMoveEdgeFeat;
        e[0] = cong_now(way_id);
        e[1] = plan_of(way_id);
        e[2] = (way_id == next_edge) ? 1.f : 0.f;
        e[3] = (way_id == came_from) ? 1.f : 0.f;
        if (way_id != next_edge && way_id != came_from)
            min_other = std::min(min_other, e[0]);
        ++slot;
    };
    put(next_edge);
    for (auto way_id : node_it->second.incident_ways)
        if (way_id != next_edge) put(way_id);
    o.n_edges = slot;
    o.x[0] = o.x[kMoveSelfFeat];       // congestion of the planned edge now
    o.x[1] = o.x[kMoveSelfFeat + 1];   // same edge at planning time
    o.x[2] = min_other;                // best visible alternative
    if (auto ait = lsm_alert_.find(agent_id); ait != lsm_alert_.end())
        o.x[3] = ait->second;          // LSM area alert (0 = none / LSM off)

    MovementPolicy& mp = movement_policy();
    const float mu     = mp.score(o);
    const bool  replan = mp.decide(mu, cfg_.movement_train);

    int idx = -1;
    if (cfg_.movement_train) {
        int pred = 1;
        if (!agent->solution.sequence.empty())
            pred = std::max(1, agent->solution.sequence[0].estimated_arrival
                               - current_step);
        idx = mp.record(agent_id, o, mu, replan ? 1.f : 0.f,
                        current_step, pred);
    }
    ++mv_stats_.n_decisions;
    if (!replan) return;

    ++mv_stats_.n_replans;
    PDPGlobalMemory::RerouteOutcome out;
    memory_.push_rerouted_path(agent_id, cfg_.speed_mps, &out);
    if (out.adopted) ++mv_stats_.n_adopted;

    float g = 0.f;
    if (out.attempted && out.cur_steps > 0)
        g = static_cast<float>(std::max(0, out.cur_steps - out.tda_steps))
            / static_cast<float>(out.cur_steps);
    mv_stats_.gain_sum += g;

    if (idx >= 0) {
        float r = g - cfg_.mp_replan_cost;
        if (g < cfg_.mp_gain_eps) r -= cfg_.mp_null_gain_penalty;
        mp.add_reward(agent_id, idx, r);
    }
}
