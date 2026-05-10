#include "EpisodeRunner.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"
#include "DMASforPD/TaskAgent/TaskAgent.hpp"
#include "DMASforPD/DeliveryAgent/OperableEnvironment.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// ── Construction / destruction ────────────────────────────────────────────────

EpisodeRunner::EpisodeRunner(const EpisodeConfig& cfg,
                             GeoBox&              geo_box,
                             Pathfinder&          pathfinder,
                             uint32_t             seed)
    : cfg_(cfg), memory_(geo_box, pathfinder), gen_(cfg, geo_box, seed), rng_(seed)
{
    // Synchronise TaskAgent speed with episode speed so plan() and commit_plan()
    // use the same value when computing estimated arrivals and congestion entries.
    memory_.task_agent.params.default_speed_mps = cfg_.speed_mps;

    // Limit to one task per agent. With edge-by-edge movement, assigning a second
    // task to an Active agent while it is traversing would cause plan() to reorder
    // solution.sequence while existing ScheduledArrivals still carry the old
    // task_id/is_pickup — leading to wrong mark_picked / mark_delivered calls.
    // Single-task mode is correct and sufficient for LGPDP training.
    memory_.task_agent.params.max_tasks_per_agent = 1;

    // Single shared group cache for all synthetic task paths.
    memory_.ensure_task_group(0);

    // Find a valid start node for agents (must be connected to at least one way).
    osmium::object_id_type start = 0;
    for (const auto& [id, pt] : geo_box.data.nodes) {
        if (!pt.incident_ways.empty()) { start = id; break; }
    }
    if (start == 0)
        throw std::runtime_error("EpisodeRunner: no valid road node for agent start");

    int n = cfg_.max_agents();
    all_agents_.reserve(n);
    for (int i = 0; i < n; ++i) {
        auto a = std::make_unique<DeliveryAgent>(i, start);
        a->connect(memory_);
        memory_.task_agent.add_delivery_agent(*a);
        all_agents_.push_back(std::move(a));
    }
}

EpisodeRunner::~EpisodeRunner() {
    for (auto& a : all_agents_)
        a->disconnect(memory_);
}

// ── Main episode loop ─────────────────────────────────────────────────────────

RunResult EpisodeRunner::run(int city_index, int num_cities) {
    auto t0 = std::chrono::steady_clock::now();
    arrivals_.clear();
    global_states_.clear();
    task_accept_buf_idx_.clear();
    // Discard any eval experiences that accumulated since the last train_epoch.
    // (Eval runs with train_mode=false skip train_epoch, so the shared buffer
    //  accumulates stale data that must not contaminate the next training update.)
    if (!train_mode)
        ObjectiveDMPolicy::shared().clear_buffer();
    n_accepted_    = 0;
    n_refused_     = 0;
    latency_sum_   = 0;
    latency_count_ = 0;
    active_sum_    = 0;
    active_steps_  = 0;

    // Reset per-episode GlobalMemory state (tasks, plans, congestion, clock).
    // Preserves the A* path cache so costs computed in prior episodes reuse.
    memory_.reset_episode();

    for (auto& a : all_agents_) {
        a->status = AgentStatus::Idle;
        a->in_transit.reset();
        a->edge_cursor.reset();
        a->local_memory.current_path = nullptr;
        a->local_memory.next_path    = nullptr;
        a->solution.sequence.clear();
        a->local_memory.tasks.clear();
        a->local_memory.local_agents.clear();
        a->local_memory.operable_env = OperableEnvironment{};
    }

    auto stream      = gen_.generate();
    auto phase_table = gen_.build_phase_table();
    int  total_steps = cfg_.total_steps();
    int  stream_idx  = 0;

    float city_norm = (num_cities > 1)
        ? static_cast<float>(city_index) / (num_cities - 1) : 0.f;

    ComparisonMetrics metrics;
    metrics.method = "DMAS-MAPPO";

    for (int step = 0; step < total_steps; ++step) {
        memory_.advance_time(step);

        // Determine current phase parameters.
        float phase_label = cfg_.phases.empty() ? 0.f : cfg_.phases.front().label;
        float lambda      = cfg_.phases.empty() ? 0.f : cfg_.phases.front().lambda;
        int   n_active    = cfg_.phases.empty() ? (int)all_agents_.size()
                                                : cfg_.phases.front().n_agents;
        for (const auto& ph : phase_table) {
            if (step >= ph.step_begin && step < ph.step_end) {
                phase_label = ph.label;
                lambda      = ph.lambda;
                n_active    = ph.n_agents;
                break;
            }
        }
        n_active = std::min(n_active, static_cast<int>(all_agents_.size()));

        // Process arrivals before injecting new tasks so agents that just
        // completed a delivery are Idle and can receive new work this step.
        process_arrivals(step);

        // Build the global state once per step; passed into offer_task so that
        // each MAPPO record() call gets the correctly aligned global state.
        // (global_states_ is populated per-offer, not per-step, so that
        //  global_states_.size() == buffer_.size() after the episode.)
        auto cur_gs = build_global_state(step, total_steps, phase_label, lambda,
                                         city_norm, n_active).to_array();

        // Inject tasks arriving at this step.
        while (stream_idx < (int)stream.size() &&
               stream[stream_idx].arrival_step == step) {
            const auto& st = stream[stream_idx++];

            // All synthetic tasks share group_id = 0 (path cache reuse).
            ObjectiveNode pickup  { st.pickup_node_id,   0 };
            ObjectiveNode delivery{ st.delivery_node_id, 0 };
            int task_id = memory_.add_task(pickup, delivery);

            PDPTask* task = memory_.get_task(task_id);
            task->reward     = st.reward;
            task->importance = st.importance;

            int winner = offer_task(task_id, st.reward, st.importance, n_active, cur_gs);
            if (winner >= 0) {
                ++n_accepted_;
                memory_.assign_task(task_id, winner);
                DeliveryAgent* agent = memory_.get_delivery_agent(winner);

                // Snapshot status before receive_task changes it to Active.
                const bool was_idle = (agent->status == AgentStatus::Idle);

                agent->receive_task(*task, memory_);
                memory_.commit_plan(winner, cfg_.speed_mps);

                // Record the buffer index of this accept decision so we can set
                // the real completion reward when the task is delivered.
                if (policy_mode == PolicyMode::MAPPO)
                    task_accept_buf_idx_[task_id] =
                        ObjectiveDMPolicy::shared().buffer_size() - 1;

                // Only start the edge-by-edge leg if the agent was Idle.
                // If the agent was already Active (multi-task), plan() reordered
                // the solution but the current cursor is still valid; the new
                // objectives will be picked up when the current delivery completes.
                if (was_idle)
                    start_leg(winner, task_id, true,
                              agent->current_node, st.pickup_node_id, step);
            } else {
                ++n_refused_;
                // Task stays in available_tasks but won't be re-offered.
            }
        }

        // Count active agents for utilisation metric.
        int n_now = 0;
        for (int i = 0; i < n_active; ++i)
            if (all_agents_[i]->status == AgentStatus::Active) ++n_now;
        active_sum_   += n_now;
        active_steps_ += 1;
    }

    // Populate metrics.
    int tasks_appeared    = n_accepted_ + n_refused_;
    metrics.tasks_appeared  = tasks_appeared;
    metrics.tasks_completed = latency_count_;
    metrics.accept_rate     = tasks_appeared > 0
        ? static_cast<float>(n_accepted_) / tasks_appeared : 0.f;
    metrics.refuse_rate     = 1.f - metrics.accept_rate;
    metrics.throughput_rate = tasks_appeared > 0
        ? static_cast<float>(latency_count_) / tasks_appeared : 0.f;
    if (latency_count_ > 0)
        metrics.latency_mean = static_cast<float>(latency_sum_) / latency_count_;
    int mean_active = (active_steps_ > 0) ? (active_sum_ / active_steps_) : 1;
    if (mean_active > 0)
        metrics.latency_per_agent = metrics.latency_mean / mean_active;
    metrics.agent_utilisation = (active_steps_ > 0 && n_accepted_ > 0)
        ? static_cast<float>(active_sum_) / active_steps_ / cfg_.max_agents() : 0.f;
    metrics.total_steps = total_steps;

    RunResult result;
    result.metrics = metrics;

    if (train_mode && policy_mode == PolicyMode::MAPPO && !global_states_.empty())
        result.train_stats = ObjectiveDMPolicy::shared().train_epoch(global_states_);

    auto t1 = std::chrono::steady_clock::now();
    result.wallclock_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return result;
}

// ── Task offer ────────────────────────────────────────────────────────────────

int EpisodeRunner::offer_task(int task_id, float reward, float importance,
                              int n_active,
                              const std::array<float, kGlobSz>& gs) {
    // Greedy and Random baselines bypass try_accept_task to avoid writing to
    // the MAPPO training buffer.
    const int task_cap = memory_.task_agent.params.max_tasks_per_agent;
    auto has_cap = [&](const DeliveryAgent& a) {
        return a.status == AgentStatus::Idle ||
               (a.status == AgentStatus::Active &&
                (int)a.local_memory.tasks.size() < task_cap);
    };

    if (policy_mode == PolicyMode::Greedy) {
        for (int i = 0; i < n_active; ++i)
            if (has_cap(*all_agents_[i])) return all_agents_[i]->agent_id;
        return -1;
    }
    if (policy_mode == PolicyMode::Random) {
        std::bernoulli_distribution coin(0.5);
        for (int i = 0; i < n_active; ++i)
            if (has_cap(*all_agents_[i]) && coin(rng_))
                return all_agents_[i]->agent_id;
        return -1;
    }

    // MAPPO: score each available agent (Idle or Active with capacity).
    // Push gs once per try_accept_task call (= once per record()) so that
    // global_states_[i] aligns exactly with buffer_[i] in train_epoch.
    TaskOffer offer{ task_id, reward, importance, {} };
    for (int i = 0; i < n_active; ++i) {
        DeliveryAgent& agent = *all_agents_[i];
        if (!has_cap(agent)) {
            offer.prev_agents.push_back(agent.agent_id);
            continue;
        }
        float score = agent.try_accept_task(offer, memory_);
        global_states_.push_back(gs);  // aligned with the record() call above
        if (score >= 0.5f) return agent.agent_id;
        offer.prev_agents.push_back(agent.agent_id);
    }
    return -1;
}

// ── Edge-by-edge movement ─────────────────────────────────────────────────────

int EpisodeRunner::start_leg(int agent_id, int task_id, bool is_pickup,
                             osmium::object_id_type from,
                             osmium::object_id_type to,
                             int current_step) {
    DeliveryAgent* agent = memory_.get_delivery_agent(agent_id);
    if (!agent) return current_step + 1;

    // Fetch (or compute) the path from the group-0 cache.
    const ObjectivePath* path = memory_.get_or_compute_path(from, to, 0);

    // Load path into agent cursor and set local_memory.current_path.
    agent->begin_leg(path, task_id, is_pickup);

    // Pre-fetch the NEXT leg's path (sequence[0]→sequence[1]) into next_path.
    // Protocol: agent knows current + one step ahead, no further.
    agent->prefetch_next_path(memory_);

    return schedule_next_edge(agent_id, current_step);
}

int EpisodeRunner::schedule_next_edge(int agent_id, int current_step) {
    DeliveryAgent* agent = memory_.get_delivery_agent(agent_id);
    if (!agent || !agent->edge_cursor) return current_step + 1;

    EdgeCursor& cur = *agent->edge_cursor;

    osmium::object_id_type dest_node = cur.next_node();
    osmium::object_id_type edge_id   = cur.current_edge_id();

    // Compute traversal time for this edge.
    float dist = 0.f;
    if (edge_id != 0) {
        auto it = memory_.geo_box.data.ways.find(edge_id);
        if (it != memory_.geo_box.data.ways.end())
            dist = it->second.distance_meters;
    }
    if (dist <= 0.f) {
        // No edge or fallback: haversine between current node and destination.
        dist = fallback_cost(agent->current_node, dest_node);
    }

    int steps   = std::max(1, static_cast<int>(std::ceil(dist / cfg_.speed_mps)));
    int arrival = current_step + steps;

    bool is_last = cur.is_last_edge();

    agent->start_edge(edge_id, dest_node, arrival);
    arrivals_.push_back({ agent_id, cur.task_id, cur.is_pickup, is_last, arrival });
    return arrival;
}

void EpisodeRunner::on_objective_reached(int agent_id, int task_id,
                                         bool is_pickup, int current_step) {
    DeliveryAgent* agent = memory_.get_delivery_agent(agent_id);
    PDPTask*       task  = memory_.get_task(task_id);
    if (!agent || !task) return;

    // Pop the just-reached objective from the solution sequence.
    agent->solution.advance();
    agent->edge_cursor.reset();

    if (is_pickup) {
        task->mark_picked(current_step);

        // Promote pre-fetched next_path (pickup→delivery) to current_path.
        agent->promote_next_path();

        // Congestion reroute: agent is at a node boundary — refresh the dynamic
        // cost of the upcoming leg (delivery) if traffic makes another route ≥5%
        // faster. push_rerouted_path updates current_path and commits the plan.
        memory_.push_rerouted_path(agent_id, cfg_.speed_mps);

        const ObjectivePath* delivery_path = agent->local_memory.current_path;
        agent->begin_leg(delivery_path, task_id, false);
        agent->prefetch_next_path(memory_);

        schedule_next_edge(agent_id, current_step);
        memory_.commit_plan(agent_id, cfg_.speed_mps);

    } else {
        task->mark_delivered(current_step);

        int latency = current_step - task->timeline.created_step;
        latency_sum_   += latency;
        latency_count_ += 1;

        // Update the MAPPO experience for this task with the actual completion
        // reward. This replaces the placeholder 0 recorded at decision time,
        // giving correct credit assignment to the accept decision.
        if (policy_mode == PolicyMode::MAPPO) {
            auto it = task_accept_buf_idx_.find(task_id);
            if (it != task_accept_buf_idx_.end()) {
                float completion_reward = task->reward * task->importance;
                ObjectiveDMPolicy::shared().update_reward(it->second, completion_reward);
                task_accept_buf_idx_.erase(it);
            }
        }

        // Complete the task (sets status→Idle if no more tasks remain).
        memory_.task_agent.on_task_event(task_id, TaskEvent::Finished, memory_);

        // If the agent still has tasks queued, start the next leg immediately.
        if (!agent->solution.empty()) {
            agent->promote_next_path();

            // Congestion reroute at leg boundary (agent is at a node).
            memory_.push_rerouted_path(agent_id, cfg_.speed_mps);

            const ObjectivePath* next_path = agent->local_memory.current_path;
            const ObjectiveNode& next_obj  = agent->solution.next_objective();

            if (!next_path)
                next_path = memory_.get_or_compute_path(
                    agent->current_node, next_obj.id, 0);

            PDPTask* next_task    = memory_.get_task_for_node(next_obj.id);
            int      next_task_id = next_task ? next_task->task_id : -1;
            bool     next_pickup  = next_task && (next_task->pickup.id == next_obj.id);
            agent->begin_leg(next_path, next_task_id, next_pickup);
            agent->prefetch_next_path(memory_);
            schedule_next_edge(agent_id, current_step);
            memory_.commit_plan(agent_id, cfg_.speed_mps);
        }
    }
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
    // Index-based loop: on_objective_reached() may push new entries beyond n.
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

        // Advance agent to the node at the end of the completed edge.
        agent->arrive_at_node();

        if (!is_objective) {
            // ── Intermediate road node ────────────────────────────────────
            // Advance cursor and schedule the next edge of the same leg.
            if (agent->edge_cursor)
                agent->edge_cursor->next_idx++;
            schedule_next_edge(agent_id, current_step);
        } else {
            // ── Objective node (pickup or delivery) ───────────────────────
            on_objective_reached(agent_id, task_id, is_pickup, current_step);
        }

        arrivals_[i].arrival_step = -1;
    }

    arrivals_.erase(
        std::remove_if(arrivals_.begin(), arrivals_.end(),
            [](const ScheduledArrival& a){ return a.arrival_step < 0; }),
        arrivals_.end());
}

// ── GlobalState assembly ──────────────────────────────────────────────────────

GlobalState EpisodeRunner::build_global_state(int step, int total_steps,
                                              float phase_label, float lambda,
                                              float city_norm, int n_active) const {
    GlobalState gs;
    int tasks_total = n_accepted_ + n_refused_;

    gs.time_ratio     = total_steps > 0
        ? static_cast<float>(step) / total_steps : 0.f;
    gs.n_agents_ratio = cfg_.max_agents() > 0
        ? static_cast<float>(n_active) / cfg_.max_agents() : 0.f;
    gs.avail_ratio    = tasks_total > 0
        ? static_cast<float>(memory_.count_available()) / tasks_total : 0.f;
    gs.alloc_ratio    = tasks_total > 0
        ? static_cast<float>(memory_.count_allocated()) / tasks_total : 0.f;
    gs.done_ratio     = tasks_total > 0
        ? static_cast<float>(latency_count_) / tasks_total : 0.f;

    float load_sum = 0.f, max_load = 0.f;
    for (const auto& a : all_agents_) {
        float l = static_cast<float>(a->local_memory.tasks.size());
        load_sum += l;
        max_load  = std::max(max_load, l);
    }
    gs.avg_load = all_agents_.empty() ? 0.f
        : (load_sum / all_agents_.size()) / kMaxLoad;
    gs.max_load = max_load / kMaxLoad;

    // Arrival rate normalised to [0,1] against the high-density phase ceiling (0.20).
    gs.arrival_rate = std::clamp(lambda / 0.20f, 0.f, 1.f);

    gs.throughput  = tasks_total > 0
        ? static_cast<float>(latency_count_) / tasks_total : 0.f;
    gs.avg_latency = (latency_count_ > 0 && cfg_.time_window_steps > 0)
        ? (static_cast<float>(latency_sum_) / latency_count_)
          / cfg_.time_window_steps : 0.f;
    gs.accept_rate = tasks_total > 0
        ? static_cast<float>(n_accepted_) / tasks_total : 0.f;
    gs.avg_efficiency = 0.f; // placeholder — no route-quality signal in simplified model

    gs.city_id_norm  = city_norm;
    gs.density_phase = phase_label;
    gs.cluster_ratio = 0.f; // could track hot-zone fraction if needed
    gs.congestion    = 0.f; // placeholder

    return gs;
}
