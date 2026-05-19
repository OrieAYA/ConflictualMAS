#include "EpisodeRunner.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"
#include "DMASforPD/Policy/IPPOPolicy.hpp"
#include "DMASforPD/Policy/MapperPolicy.hpp"
#include "DMASforPD/Policy/HybridPolicy.hpp"
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

    // Agent task capacity (configurable via EpisodeConfig).
    // Default = 1 (single-task mode, safe). For small-graph training, single-task
    // is sufficient; for large graphs >1 boosts experience volume but risks
    // ScheduledArrival/sequence-reorder mismatch — keep at 1 unless validated.
    memory_.task_agent.params.max_tasks_per_agent = cfg_.max_tasks_per_agent;

    // Single shared group cache (group 1) for all synthetic task paths.
    // Group 0 stays empty (reserved as the "no-group" sentinel).
    memory_.ensure_task_group(1);

    // Find a valid start node for agents (must be connected to at least one way).
    osmium::object_id_type start = 0;
    for (const auto& [id, pt] : geo_box.data.nodes) {
        if (!pt.incident_ways.empty()) { start = id; break; }
    }
    if (start == 0)
        throw std::runtime_error("EpisodeRunner: no valid road node for agent start");

    // Allocate the agent pool with a 1.5× overhead so the per-episode
    // EpisodeScenario.agents_mult can sample values above 1.0 (slack regime)
    // without being clamped down to the nominal max_agents() at run-time.
    // Idle surplus agents are inert (never offered tasks when n_active < pool)
    // and cost only a small per-instance memory footprint.
    int n = std::max(1, static_cast<int>(std::ceil(cfg_.max_agents() * 1.5)));
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

RunResult EpisodeRunner::run(int city_index, int num_cities,
                             EpisodeScenario scenario) {
    auto t0 = std::chrono::steady_clock::now();
    arrivals_.clear();
    global_states_.clear();
    task_accept_buf_idx_.clear();
    // Discard any eval experiences that accumulated since the last train_epoch.
    // (Eval runs with train_mode=false skip train_epoch, so the shared buffer
    //  accumulates stale data that must not contaminate the next training update.)
    if (!train_mode) {
        ObjectiveDMPolicy::shared().clear_buffer();
        IPPOPolicy::shared().clear_buffer_all();
        MapperPolicy::shared().clear_buffer_all();
    }
    // Hybrid is ALWAYS cleared at episode start (train or eval): its REINFORCE
    // update runs at the end of every episode regardless of train_mode — that's
    // the defining property of Hybrid (continual online adaptation), not the
    // gradient-based offline batch training of MAPPO/IPPO/MAPPER.
    if (policy_mode == PolicyMode::Hybrid)
        HybridPolicy::shared().clear_buffer_all();

    // Always start a fresh recent-records log for the per-agent baselines
    // (IPPO, MAPPER, Hybrid); stale offers from a previous episode must not be
    // picked up by the refusal-penalty bracketing during this one.
    IPPOPolicy::shared().clear_recent_records();
    MapperPolicy::shared().clear_recent_records();
    HybridPolicy::shared().clear_recent_records();

    // Publish the active learning policy to PDPGlobalMemory so that
    // DeliveryAgent::try_accept_task routes its calls to the right backend.
    switch (policy_mode) {
        case PolicyMode::IPPO:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kIPPO;
            break;
        case PolicyMode::MAPPER:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kMAPPER;
            break;
        case PolicyMode::Hybrid:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kHybrid;
            break;
        default:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kMAPPO;
            break;
    }
    n_accepted_    = 0;
    n_refused_     = 0;
    latency_sum_   = 0;
    latency_count_ = 0;
    active_sum_    = 0;
    active_steps_  = 0;
    efficiency_sum_   = 0.0;
    efficiency_count_ = 0;
    congestion_sum_   = 0.0;
    congestion_steps_ = 0;
    trip_sum_         = 0;
    trip_count_       = 0;

    // Propagate the always_accept flag to the TAM for TamAlwaysAccept ablation.
    // Must be set before the episode loop so all TAMs created during the episode
    // inherit it.
    memory_.task_agent.params.tam_params.always_accept =
        (policy_mode == PolicyMode::TamAlwaysAccept);

    // Reset per-episode GlobalMemory state (tasks, plans, congestion, clock).
    // Preserves the A* path cache so costs computed in prior episodes reuse.
    memory_.reset_episode();

    // Publish episode-level constants so the policy's per-decision features
    // can reference them (deliverability = steps_remaining / delivery_steps).
    memory_.speed_mps   = cfg_.speed_mps;
    memory_.total_steps = cfg_.total_steps();

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
        float lambda      = cfg_.phases.empty() ? 0.f : 0.f;
        int   n_active    = cfg_.phases.empty() ? (int)all_agents_.size()
                                                : cfg_.phases.front().n_agents_start;
        for (const auto& ph : phase_table) {
            if (step >= ph.step_begin && step < ph.step_end) {
                phase_label = ph.label;
                lambda      = ph.lambda;
                n_active    = ph.n_agents_at(step);
                break;
            }
        }
        // Apply scenario multipliers (varies difficulty per episode).
        lambda   *= scenario.density_mult;
        n_active  = std::max(1, static_cast<int>(std::round(n_active * scenario.agents_mult)));
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

            // All synthetic tasks share group_id = 1 (group 0 is reserved / empty).
            ObjectiveNode pickup  { st.pickup_node_id,   1 };
            ObjectiveNode delivery{ st.delivery_node_id, 1 };
            int task_id = memory_.add_task(pickup, delivery);

            PDPTask* task = memory_.get_task(task_id);
            task->reward     = st.reward;
            task->importance = st.importance;
            task->reward_original     = st.reward;
            task->importance_original = st.importance;

            OfferResult res = offer_task(task_id, st.reward, st.importance,
                                         n_active, cur_gs);
            const int winner = res.agent_id;
            if (winner >= 0) {
                ++n_accepted_;
                DeliveryAgent* agent = memory_.get_delivery_agent(winner);

                bool was_idle;
                if (res.tam_owned) {
                    // TAM already did assign + receive + commit. After receive_task
                    // the agent's task list contains this task plus whatever it had
                    // before. Sole task ⇒ was Idle.
                    was_idle = agent->local_memory.tasks.size() == 1;
                } else {
                    was_idle = (agent->status == AgentStatus::Idle);
                    memory_.assign_task(task_id, winner);
                    agent->receive_task(*task, memory_);
                    memory_.commit_plan(winner, cfg_.speed_mps);
                }

                // Record buffer index of the accept decision so the completion
                // reward can be set at delivery. In the TAM path the accept
                // entry is the LAST record produced by offer_to_agent — in the
                // shared buffer for MAPPO, or in the winning agent's own buffer
                // for IPPO/MAPPER.
                if (policy_mode == PolicyMode::MAPPO) {
                    task_accept_buf_idx_[task_id] =
                        ObjectiveDMPolicy::shared().buffer_size() - 1;
                } else if (policy_mode == PolicyMode::IPPO) {
                    task_accept_buf_idx_[task_id] =
                        IPPOPolicy::shared().buffer_size(winner) - 1;
                } else if (policy_mode == PolicyMode::MAPPER) {
                    task_accept_buf_idx_[task_id] =
                        MapperPolicy::shared().buffer_size(winner) - 1;
                } else if (policy_mode == PolicyMode::Hybrid) {
                    task_accept_buf_idx_[task_id] =
                        HybridPolicy::shared().buffer_size(winner) - 1;
                }

                // Only start the edge-by-edge leg if the agent was Idle. If
                // already Active (multi-task queue), the current cursor is
                // still valid; the new objectives are picked up after the
                // current delivery completes.
                if (was_idle)
                    start_leg(winner, task_id, true,
                              agent->current_node, st.pickup_node_id, step);
            } else {
                ++n_refused_;
                // Task stays in available_tasks but won't be re-offered.
            }
        }

        // Sample network congestion once per step (mean edge load over all edges
        // with at least one planned agent passage). Averaged at episode end to
        // produce ComparisonMetrics::mean_congestion.
        congestion_sum_   += memory_.congestion_map.mean_load_now();
        congestion_steps_ += 1;

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
    metrics.mean_congestion = (congestion_steps_ > 0)
        ? static_cast<float>(congestion_sum_ / congestion_steps_) : 0.f;
    metrics.mean_trip_steps = (trip_count_ > 0)
        ? static_cast<float>(trip_sum_) / trip_count_ : 0.f;

    // ── Spatial complexity over served tasks ──────────────────────────────
    // Walk all accepted tasks (allocated + finished) and compute spatial spread
    // metrics: bbox area, convex hull area, mean P→D distance, mean nearest
    // neighbour pickup distance. Cheap relative to the simulation cost.
    {
        struct PtLL { double lat, lon; };
        std::vector<PtLL> pus, dels;
        auto add_task_points = [&](const PDPTask* t) {
            if (!t) return;
            auto itp = memory_.geo_box.data.nodes.find(t->pickup.id);
            auto itd = memory_.geo_box.data.nodes.find(t->delivery.id);
            if (itp == memory_.geo_box.data.nodes.end() ||
                itd == memory_.geo_box.data.nodes.end()) return;
            pus .push_back({itp->second.lat, itp->second.lon});
            dels.push_back({itd->second.lat, itd->second.lon});
        };
        for (const PDPTask* t : memory_.finished_tasks)  add_task_points(t);
        for (const PDPTask* t : memory_.allocated_tasks) add_task_points(t);

        constexpr double kPi = 3.14159265358979323846;
        auto hav = [&](double la1, double lo1, double la2, double lo2) {
            constexpr double R = 6371000.0;
            const double dlat = (la2 - la1) * kPi / 180.0;
            const double dlon = (lo2 - lo1) * kPi / 180.0;
            const double a1 = la1 * kPi / 180.0;
            const double a2 = la2 * kPi / 180.0;
            const double a = std::sin(dlat/2)*std::sin(dlat/2)
                           + std::cos(a1)*std::cos(a2)*std::sin(dlon/2)*std::sin(dlon/2);
            return R * 2.0 * std::atan2(std::sqrt(a), std::sqrt(1-a));
        };

        if (!pus.empty()) {
            double mn_lat=+1e9, mx_lat=-1e9, mn_lon=+1e9, mx_lon=-1e9;
            for (size_t i = 0; i < pus.size(); ++i) {
                mn_lat = std::min({mn_lat, pus[i].lat, dels[i].lat});
                mx_lat = std::max({mx_lat, pus[i].lat, dels[i].lat});
                mn_lon = std::min({mn_lon, pus[i].lon, dels[i].lon});
                mx_lon = std::max({mx_lon, pus[i].lon, dels[i].lon});
            }
            const double w = hav(mn_lat, mn_lon, mn_lat, mx_lon);
            const double h = hav(mn_lat, mn_lon, mx_lat, mn_lon);
            metrics.bbox_area_km2 = static_cast<float>((w * h) / 1e6);

            // Convex hull (Andrew's monotone chain) on local-flat projection.
            const double cos_lat = std::cos((mn_lat + mx_lat) * 0.5 * kPi / 180.0);
            struct P2 { double x, y; };
            std::vector<P2> pts;
            pts.reserve(pus.size() * 2);
            for (size_t i = 0; i < pus.size(); ++i) {
                pts.push_back({pus [i].lon * 111320.0 * cos_lat, pus [i].lat * 111320.0});
                pts.push_back({dels[i].lon * 111320.0 * cos_lat, dels[i].lat * 111320.0});
            }
            std::sort(pts.begin(), pts.end(),
                      [](const P2& a, const P2& b){
                          return (a.x != b.x) ? a.x < b.x : a.y < b.y;
                      });
            auto crossp = [](const P2& O, const P2& A, const P2& B){
                return (A.x-O.x)*(B.y-O.y) - (A.y-O.y)*(B.x-O.x);
            };
            const int nP = static_cast<int>(pts.size());
            std::vector<P2> hull(2 * nP);
            int k = 0;
            for (int i = 0; i < nP; ++i) {
                while (k >= 2 && crossp(hull[k-2], hull[k-1], pts[i]) <= 0) --k;
                hull[k++] = pts[i];
            }
            for (int i = nP-2, tlim = k+1; i >= 0; --i) {
                while (k >= tlim && crossp(hull[k-2], hull[k-1], pts[i]) <= 0) --k;
                hull[k++] = pts[i];
            }
            hull.resize(k - 1);
            double sarea = 0;
            for (size_t i = 0; i < hull.size(); ++i) {
                const auto& A = hull[i];
                const auto& B = hull[(i+1) % hull.size()];
                sarea += A.x * B.y - B.x * A.y;
            }
            metrics.convex_hull_area_km2 =
                static_cast<float>(std::abs(sarea) * 0.5 / 1e6);

            // Mean P→D direct distance.
            double sum_pd = 0.0;
            for (size_t i = 0; i < pus.size(); ++i)
                sum_pd += hav(pus[i].lat, pus[i].lon, dels[i].lat, dels[i].lon);
            metrics.mean_pd_distance_m = static_cast<float>(sum_pd / pus.size());

            // Mean nearest-neighbour pickup distance (O(N²), N is small here).
            double sum_nn = 0.0; int n_nn = 0;
            for (size_t i = 0; i < pus.size(); ++i) {
                double best = 1e18;
                for (size_t j = 0; j < pus.size(); ++j) {
                    if (i == j) continue;
                    const double d = hav(pus[i].lat, pus[i].lon,
                                          pus[j].lat, pus[j].lon);
                    if (d < best) best = d;
                }
                if (best < 1e17) { sum_nn += best; ++n_nn; }
            }
            metrics.mean_nn_pickup_m = (n_nn > 0)
                ? static_cast<float>(sum_nn / n_nn) : 0.f;
        }
    }

    // ── Validity sanity check over the served tasks ────────────────────────
    // These should remain 0 — non-zero indicates a planner bug we must catch.
    {
        const int max_carry = std::max(
            1, memory_.task_agent.params.max_tasks_per_agent);
        for (const PDPTask* t : memory_.finished_tasks) {
            if (!t) continue;
            if (t->timeline.delivered_step >= 0 &&
                t->timeline.picked_step    >= 0 &&
                t->timeline.picked_step >= t->timeline.delivered_step)
                ++metrics.pairing_violations_runtime;
            if (t->timeline.picked_step    >= 0 &&
                t->timeline.created_step   >= 0 &&
                t->timeline.picked_step < t->timeline.created_step)
                ++metrics.pairing_violations_runtime;
        }
        // Capacity: replay per-agent timeline.
        struct Ev { int step; int delta; };
        std::unordered_map<int, std::vector<Ev>> by_agent;
        auto push_task = [&](const PDPTask* t){
            if (!t || t->agent_id < 0) return;
            if (t->timeline.picked_step    >= 0)
                by_agent[t->agent_id].push_back({t->timeline.picked_step,    +1});
            if (t->timeline.delivered_step >= 0)
                by_agent[t->agent_id].push_back({t->timeline.delivered_step, -1});
        };
        for (const PDPTask* t : memory_.finished_tasks)  push_task(t);
        for (const PDPTask* t : memory_.allocated_tasks) push_task(t);
        for (auto& [aid, evs] : by_agent) {
            std::sort(evs.begin(), evs.end(),
                      [](const Ev& a, const Ev& b){ return a.step < b.step; });
            int load = 0, peak = 0;
            for (const auto& e : evs) { load += e.delta; if (load > peak) peak = load; }
            if (peak > max_carry)
                metrics.capacity_violations_runtime += peak - max_carry;
        }
    }

    RunResult result;
    result.metrics = metrics;

    // Apply unfinished-acceptance penalty: any task still in task_accept_buf_idx_
    // was accepted but never delivered before episode end. The penalty scales
    // with the task value the agent failed to capture, so failing a high-value
    // task hurts more than failing a low-value one. This keeps the policy from
    // accepting tasks it cannot deliver, while a flat penalty would collapse
    // into "accept everything" once it falls below the refuse penalty.
    const bool is_learning_mode =
        (policy_mode == PolicyMode::MAPPO) ||
        (policy_mode == PolicyMode::IPPO)  ||
        (policy_mode == PolicyMode::MAPPER);
    // Hybrid receives the unfinished penalty regardless of train_mode because
    // its REINFORCE update runs at every episode end (online adaptation).
    const bool apply_unfinished_penalty =
        (train_mode && is_learning_mode) || (policy_mode == PolicyMode::Hybrid);
    if (apply_unfinished_penalty && cfg_.unfinished_factor > 0.f) {
        for (const auto& [tid, buf_idx] : task_accept_buf_idx_) {
            PDPTask* t = memory_.get_task(tid);
            if (!t) continue;
            const float val    = t->reward_original * t->importance_original;
            const bool  picked = (t->timeline.picked_step >= 0);
            const float lost_frac = picked ? (1.f - cfg_.pickup_reward_frac) : 1.f;
            const float penalty   = -cfg_.unfinished_factor * lost_frac * val;
            if (policy_mode == PolicyMode::MAPPO) {
                ObjectiveDMPolicy::shared().add_to_reward(buf_idx, penalty);
            } else if (policy_mode == PolicyMode::IPPO) {
                IPPOPolicy::shared().add_to_reward(t->agent_id, buf_idx, penalty);
            } else if (policy_mode == PolicyMode::MAPPER) {
                MapperPolicy::shared().add_to_reward(t->agent_id, buf_idx, penalty);
            } else if (policy_mode == PolicyMode::Hybrid) {
                HybridPolicy::shared().add_to_reward(t->agent_id, buf_idx, penalty);
            }
        }
    }

    // Per-episode learning update.
    //
    // MAPPO/IPPO/MAPPER: PPO updates only when train_mode=true.
    // Hybrid: REINFORCE update + rollback check runs at the end of EVERY
    // episode regardless of train_mode — this is the defining property of
    // Hybrid (continual online adaptation).
    if (train_mode) {
        if (policy_mode == PolicyMode::MAPPO && !global_states_.empty()) {
            result.train_stats =
                ObjectiveDMPolicy::shared().train_epoch(global_states_);
        } else if (policy_mode == PolicyMode::IPPO &&
                   IPPOPolicy::shared().total_buffer_size() > 0) {
            result.train_stats = IPPOPolicy::shared().train_epoch();
        } else if (policy_mode == PolicyMode::MAPPER &&
                   MapperPolicy::shared().total_buffer_size() > 0) {
            result.train_stats = MapperPolicy::shared().train_epoch();
        }
    }
    if (policy_mode == PolicyMode::Hybrid &&
        HybridPolicy::shared().total_buffer_size() > 0) {
        result.train_stats = HybridPolicy::shared().train_epoch();
    }

    auto t1 = std::chrono::steady_clock::now();
    result.wallclock_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Derived temporal complexity metrics (in result.metrics for CSV export).
    const int tasks_total = result.metrics.tasks_appeared;
    const int n_active_pool = std::max(1, cfg_.max_agents());
    result.metrics.compute_time_per_task_ms = (tasks_total > 0)
        ? static_cast<float>(result.wallclock_ms) / tasks_total : 0.f;
    result.metrics.compute_time_per_decision_us = (tasks_total > 0)
        ? static_cast<float>(result.wallclock_ms) * 1000.f
              / (tasks_total * n_active_pool)
        : 0.f;

    return result;
}

// ── Task offer ────────────────────────────────────────────────────────────────

EpisodeRunner::OfferResult EpisodeRunner::offer_task(
    int task_id, float reward, float importance,
    int n_active, const std::array<float, kGlobSz>& gs)
{
    // Make the current time ratio visible to try_accept_task via GlobalMemory.
    // gs[0] = GlobalState::time_ratio = step / total_steps.
    memory_.cur_time_ratio = gs[0];

    // Queue length is no longer a filter — physical carrying capacity is
    // enforced at planning time inside receive_task (capacity-aware insertion
    // respecting max_tasks_per_agent simultaneously-held packages). For
    // baselines, this means any agent can be considered eligible at offer
    // time; sequence-level capacity is settled by the insertion routine.
    auto has_cap = [&](const DeliveryAgent& /*a*/) { return true; };

    // ── Baselines: sequential scan, no TAM, no MAPPO buffer writes ────────────
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

    // ── LaCAM*-inspired baseline: optimal-myopic best-insertion ──────────────
    // Real LaCAM* is grid-MAPF specific; we adapt its "always pick the globally
    // best assignment" spirit by scanning ALL eligible agents and selecting the
    // one whose route_cost(current → pickup → delivery) is minimum. Unlike
    // InsertionGreedy this never short-circuits on the first acceptable bid.
    if (policy_mode == PolicyMode::LaCAM) {
        PDPTask* t = memory_.get_task(task_id);
        if (!t) return { -1, false };
        const auto* pu_del = memory_.get_or_compute_path(
            t->pickup.id, t->delivery.id, t->delivery.group_id);
        const float c_del = (pu_del && pu_del->valid()) ? pu_del->cost : kCostScale;

        int   best_aid  = -1;
        float best_cost = std::numeric_limits<float>::max();
        for (int i = 0; i < n_active; ++i) {
            DeliveryAgent& a = *all_agents_[i];
            if (!has_cap(a)) continue;
            const auto* to_pu = memory_.get_or_compute_path(
                a.current_node, t->pickup.id, t->pickup.group_id);
            const float c_pu = (to_pu && to_pu->valid()) ? to_pu->cost : kCostScale;
            const float cost = c_pu + c_del;
            if (cost < best_cost) { best_cost = cost; best_aid = a.agent_id; }
        }
        return { best_aid, false };
    }

    // ── PIBT-inspired baseline: load-balanced priority allocation ────────────
    // PIBT [Okumura et al.] assigns higher priority to agents that need to move
    // first to avoid deadlocks. In our PDP setting (no grid conflicts), we adapt
    // the priority concept to a load-balancing rule: the agent with the smallest
    // current task queue wins. This avoids the first-by-index pathology of
    // Greedy where the same agent monopolises all early tasks.
    if (policy_mode == PolicyMode::PIBT) {
        int best_aid  = -1;
        int min_load  = std::numeric_limits<int>::max();
        for (int i = 0; i < n_active; ++i) {
            DeliveryAgent& a = *all_agents_[i];
            if (!has_cap(a)) continue;
            const int load = static_cast<int>(a.local_memory.tasks.size());
            if (load < min_load) { min_load = load; best_aid = a.agent_id; }
        }
        return { best_aid, false };
    }

    // ── Congestion-Aware baseline (Liu, Saha et al.) ─────────────────────────
    // Picks the eligible agent whose pickup-side path has the lowest dynamic
    // (congestion-adjusted) cost. Falls back to static cost when the dynamic
    // cost has not yet been computed for that path. Reuses the existing
    // CongestionMap + TD-A* machinery in PDPServerMemory.
    if (policy_mode == PolicyMode::CongestionAware) {
        PDPTask* t = memory_.get_task(task_id);
        if (!t) return { -1, false };

        int   best_aid  = -1;
        float best_cong = std::numeric_limits<float>::max();
        for (int i = 0; i < n_active; ++i) {
            DeliveryAgent& a = *all_agents_[i];
            if (!has_cap(a)) continue;
            const auto* to_pu = memory_.get_or_compute_path(
                a.current_node, t->pickup.id, t->pickup.group_id);
            float cong = kCostScale;
            if (to_pu && to_pu->valid()) {
                cong = to_pu->has_dynamic_cost()
                    ? to_pu->dynamic_cost   // congestion-adjusted travel time
                    : to_pu->cost;          // static fallback (no congestion data)
            }
            if (cong < best_cong) { best_cong = cong; best_aid = a.agent_id; }
        }
        return { best_aid, false };
    }

    // ── Token Passing (TP) [Ma+2017, AAMAS] ───────────────────────────────────
    // Original TP: agent ai with the token picks the task τ from the available
    // set T' such that h(loc(ai), pickup(τ)) is minimal, where T' filters out
    // tasks whose pickup/delivery is occupied by another agent's path-end.
    //
    // Online L-GPDP adaptation: tasks arrive one at a time. For each arriving
    // task, we pick the agent with the minimum static A* distance from its
    // current_node to the pickup. The collision filter is trivial here (no
    // physical collisions). Capacity is enforced downstream by receive_task().
    //
    // Difference from LaCAM: TP uses only the pickup-leg cost (matching the
    // paper's h-value criterion), while LaCAM minimises c_pu + c_del. Both
    // scan all eligible agents, but TP captures the "closest agent picks
    // the task" decentralised-token spirit, not the global-best assignment.
    if (policy_mode == PolicyMode::TokenPassing) {
        PDPTask* t = memory_.get_task(task_id);
        if (!t) return { -1, false };

        int   best_aid = -1;
        float best_h   = std::numeric_limits<float>::max();
        for (int i = 0; i < n_active; ++i) {
            DeliveryAgent& a = *all_agents_[i];
            if (!has_cap(a)) continue;
            const auto* to_pu = memory_.get_or_compute_path(
                a.current_node, t->pickup.id, t->pickup.group_id);
            const float h = (to_pu && to_pu->valid()) ? to_pu->cost : kCostScale;
            if (h < best_h) { best_h = h; best_aid = a.agent_id; }
        }
        return { best_aid, false };
    }

    // ── Double-Horizon Insertion [Mitrovic-Minic+2004] ─────────────────────────
    // Same scan as MCA but uses compute_double_horizon_cost which trades route
    // length against slack-time preservation for long-term insertions.
    //
    // Hyperparameters (paper-recommended):
    //   - alpha = 0.25
    //   - short_horizon_pos = max(1, n_active / 2) → first half of agent route
    //
    // Adapted to L-GPDP: paper's PDPTW has time windows; we approximate with
    // sequence-position-based horizon (early positions = short-term).
    if (policy_mode == PolicyMode::DoubleHorizon) {
        PDPTask* t = memory_.get_task(task_id);
        if (!t) return { -1, false };
        constexpr float kAlpha = 0.25f;   // paper's recommended value

        int   best_aid  = -1;
        float best_cost = std::numeric_limits<float>::max();
        for (int i = 0; i < n_active; ++i) {
            DeliveryAgent& a = *all_agents_[i];
            if (!has_cap(a)) continue;
            // Horizon split is now by TIME inside compute_double_horizon_cost
            // (paper-faithful); the second arg is kept for ABI compatibility.
            float c = compute_double_horizon_cost(a, *t, /*unused*/0, kAlpha);
            if (c < best_cost) { best_cost = c; best_aid = a.agent_id; }
        }
        return { best_aid, false };
    }

    // ── MCA: true marginal-cost assignment [Chen+2021] ────────────────────────
    // Scans ALL eligible agents. For each, computes the true cheapest-insertion
    // delta over all admissible (pos_P, pos_D) positions in the agent's route
    // (capacity-constrained, preserving the in-flight head). Assigns to the
    // agent with the globally minimum marginal cost. Unlike LaCAM, which ignores
    // existing routes, this correctly accounts for route detour cost.
    if (policy_mode == PolicyMode::MCA) {
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

    // ── TrafficFlow: GP-PIBT spirit full-trip congestion-aware [Chen+2024] ───
    // Extends CongestionAware by using the congestion-adjusted (dynamic) cost
    // for BOTH the current→pickup leg AND the pickup→delivery leg, capturing
    // the full planned trip cost under current traffic conditions.
    // CongestionAware weights only the first leg; TrafficFlow weights both.
    if (policy_mode == PolicyMode::TrafficFlow) {
        PDPTask* t = memory_.get_task(task_id);
        if (!t) return { -1, false };

        int   best_aid  = -1;
        float best_cost = std::numeric_limits<float>::max();
        for (int i = 0; i < n_active; ++i) {
            DeliveryAgent& a = *all_agents_[i];
            if (!has_cap(a)) continue;
            const auto* to_pu = memory_.get_or_compute_path(
                a.current_node, t->pickup.id, t->pickup.group_id);
            const auto* pu_del = memory_.get_or_compute_path(
                t->pickup.id, t->delivery.id, t->delivery.group_id);
            float c_pu  = kCostScale;
            float c_del = kCostScale;
            if (to_pu && to_pu->valid())
                c_pu  = to_pu->has_dynamic_cost()  ? to_pu->dynamic_cost  : to_pu->cost;
            if (pu_del && pu_del->valid())
                c_del = pu_del->has_dynamic_cost() ? pu_del->dynamic_cost : pu_del->cost;
            const float total = c_pu + c_del;
            if (total < best_cost) { best_cost = total; best_aid = a.agent_id; }
        }
        return { best_aid, false };
    }

    // ── MAPPO / TamAlwaysAccept: drive the Task Allocation Module ──────────────
    //
    // TAM algorithm (matches the paper's design):
    //   Two incremental Dijkstra expansions run alternately from the pickup node
    //   and from the delivery node. Each step grows a path-distance cache.
    //   When the expansion reaches an objective node already owned by some agent
    //   (pickup/delivery of one of that agent's planned tasks), the agent is
    //   added to the TAM matrix with its side-cost. An agent that appears on
    //   BOTH sides becomes a candidate; offers are tried in ascending combined
    //   cost. If all expansions terminate without allocation, the task is recalled
    //   (importance boosted) up to max_recalls before being declared exhausted.
    //
    // MAPPO           – shared actor + centralised critic; records go to the
    //                   single ObjectiveDMPolicy buffer.
    // IPPO            – shared actor + per-agent local critics; records spread
    //                   across per-agent buffers, tracked via recent_records.
    // MAPPER          – per-agent actor + per-agent critic + Evolutionary RL;
    //                   records distributed across per-agent buffers.
    // TamAlwaysAccept – TAM always accepts (params.always_accept); no buffer
    //                   writes, no refusal penalties — pure routing ablation.
    const int  buf_before     = ObjectiveDMPolicy::shared().buffer_size();
    const int  ippo_before    = IPPOPolicy::shared().n_recent_records();
    const int  mapper_before  = MapperPolicy::shared().n_recent_records();
    const int  hybrid_before  = HybridPolicy::shared().n_recent_records();

    memory_.task_agent.on_new_task(task_id, memory_);
    auto* tam = memory_.task_agent.get_tam(task_id);
    if (!tam) return {-1, true};

    // Safety budget on Dijkstra expansion per task. Each TAM step expands one
    // node on the pickup side and one on the delivery side. With tasks_per_agent
    // tuned for ~150 tasks/episode this budget is rarely consumed in practice.
    constexpr int kMaxTamSteps = 300;
    int steps_used = 0;
    while (steps_used < kMaxTamSteps
           && !tam->is_allocated() && !tam->is_exhausted()) {
        tam->step(memory_, cfg_.speed_mps);
        ++steps_used;
    }

    const int buf_after    = ObjectiveDMPolicy::shared().buffer_size();
    const int ippo_after   = IPPOPolicy::shared().n_recent_records();
    const int mapper_after = MapperPolicy::shared().n_recent_records();
    const int hybrid_after = HybridPolicy::shared().n_recent_records();

    // Align global_states_ with every record() the TAM produced (MAPPO critic).
    if (policy_mode == PolicyMode::MAPPO) {
        for (int i = buf_before; i < buf_after; ++i)
            global_states_.push_back(gs);
    }

    PDPTask* t = memory_.get_task(task_id);
    const bool allocated = tam->is_allocated() && t && t->agent_id >= 0;

    // Refusal penalty.
    //   MAPPO  → walk the shared-buffer range [buf_before, last_excl).
    //   IPPO   → walk the recent-records (agent_id, buf_idx) range.
    //   MAPPER → same as IPPO but on its own recent-records log.
    //   Hybrid → same recent-records pattern; applied regardless of train_mode
    //            because online adaptation runs every episode.
    const bool apply_refuse_penalty =
        train_mode || (policy_mode == PolicyMode::Hybrid);
    if (apply_refuse_penalty && cfg_.refuse_penalty_w > 0.f && t) {
        const float pen = -cfg_.refuse_penalty_w * t->importance_original;
        if (policy_mode == PolicyMode::MAPPO) {
            const int last_excl = allocated ? buf_after - 1 : buf_after;
            for (int i = buf_before; i < last_excl; ++i)
                ObjectiveDMPolicy::shared().update_reward(i, pen);
        } else if (policy_mode == PolicyMode::IPPO) {
            const int last_excl = allocated ? ippo_after - 1 : ippo_after;
            for (int i = ippo_before; i < last_excl; ++i) {
                auto [aid, buf_idx] = IPPOPolicy::shared().recent_record(i);
                IPPOPolicy::shared().update_reward(aid, buf_idx, pen);
            }
        } else if (policy_mode == PolicyMode::MAPPER) {
            const int last_excl = allocated ? mapper_after - 1 : mapper_after;
            for (int i = mapper_before; i < last_excl; ++i) {
                auto [aid, buf_idx] = MapperPolicy::shared().recent_record(i);
                MapperPolicy::shared().update_reward(aid, buf_idx, pen);
            }
        } else if (policy_mode == PolicyMode::Hybrid) {
            const int last_excl = allocated ? hybrid_after - 1 : hybrid_after;
            for (int i = hybrid_before; i < last_excl; ++i) {
                auto [aid, buf_idx] = HybridPolicy::shared().recent_record(i);
                HybridPolicy::shared().update_reward(aid, buf_idx, pen);
            }
        }
    }

    memory_.task_agent.erase_tam(task_id);

    return {allocated ? t->agent_id : -1, true};
}

// ── Edge-by-edge movement ─────────────────────────────────────────────────────

int EpisodeRunner::start_leg(int agent_id, int task_id, bool is_pickup,
                             osmium::object_id_type from,
                             osmium::object_id_type to,
                             int current_step) {
    DeliveryAgent* agent = memory_.get_delivery_agent(agent_id);
    if (!agent) return current_step + 1;

    const ObjectivePath* path = memory_.get_or_compute_path(from, to, 1);

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

        // Reward shaping: deliver partial credit at pickup so the decision-time
        // experience gets a signal long before the actual delivery (which can be
        // 500-1500 steps later). Reduces credit-assignment delay drastically.
        if (policy_mode == PolicyMode::MAPPO ||
            policy_mode == PolicyMode::IPPO  ||
            policy_mode == PolicyMode::MAPPER ||
            policy_mode == PolicyMode::Hybrid) {
            auto it = task_accept_buf_idx_.find(task_id);
            if (it != task_accept_buf_idx_.end()) {
                const float pickup_credit = cfg_.pickup_reward_frac
                    * task->reward_original * task->importance_original;
                if (policy_mode == PolicyMode::MAPPO) {
                    ObjectiveDMPolicy::shared().add_to_reward(
                        it->second, pickup_credit);
                } else if (policy_mode == PolicyMode::IPPO) {
                    IPPOPolicy::shared().add_to_reward(
                        task->agent_id, it->second, pickup_credit);
                } else if (policy_mode == PolicyMode::MAPPER) {
                    MapperPolicy::shared().add_to_reward(
                        task->agent_id, it->second, pickup_credit);
                } else /* Hybrid */ {
                    HybridPolicy::shared().add_to_reward(
                        task->agent_id, it->second, pickup_credit);
                }
            }
        }

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

        // Pickup→delivery traversal time (excludes wait time before allocation).
        if (task->timeline.picked_step >= 0) {
            trip_sum_   += current_step - task->timeline.picked_step;
            trip_count_ += 1;
        }

        // Track per-task delivery efficiency = reward / pickup→delivery distance.
        // Reward is normalised in [0.1, 5.0] and distance in metres; dividing by
        // 1000 yields a [0, ~10] range, finally rescaled into [0, 1] for the
        // critic's GlobalState::avg_efficiency. Higher = the route delivered
        // more reward per meter (good geographic match between task and agent).
        const auto* pu_del = memory_.get_or_compute_path(
            task->pickup.id, task->delivery.id, task->delivery.group_id);
        const float dist_m = (pu_del && pu_del->valid()) ? pu_del->cost : 1.f;
        if (dist_m > 1.f) {
            const float eff = task->reward_original * 1000.f / dist_m;
            efficiency_sum_   += static_cast<double>(eff);
            efficiency_count_ += 1;
        }

        // Reward shaping: add the remaining delivery credit on top of the
        // partial pickup credit already given. Total accumulated reward equals
        // task->reward_original × importance_original (i.e., 1.0 × imp by default).
        //
        // Use *_original (immutable) so "refuse-first-accept-on-recall" cannot
        // game a larger reward than immediate acceptance — TAM may boost
        // reward/importance to attract bidders, but the policy must be trained
        // on the true task value.
        if (policy_mode == PolicyMode::MAPPO ||
            policy_mode == PolicyMode::IPPO  ||
            policy_mode == PolicyMode::MAPPER ||
            policy_mode == PolicyMode::Hybrid) {
            auto it = task_accept_buf_idx_.find(task_id);
            if (it != task_accept_buf_idx_.end()) {
                const float delivery_credit = (1.f - cfg_.pickup_reward_frac)
                    * task->reward_original * task->importance_original;
                if (policy_mode == PolicyMode::MAPPO) {
                    ObjectiveDMPolicy::shared().add_to_reward(
                        it->second, delivery_credit);
                } else if (policy_mode == PolicyMode::IPPO) {
                    IPPOPolicy::shared().add_to_reward(
                        task->agent_id, it->second, delivery_credit);
                } else if (policy_mode == PolicyMode::MAPPER) {
                    MapperPolicy::shared().add_to_reward(
                        task->agent_id, it->second, delivery_credit);
                } else /* Hybrid */ {
                    HybridPolicy::shared().add_to_reward(
                        task->agent_id, it->second, delivery_credit);
                }
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
                    agent->current_node, next_obj.id, 1);

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

// ── MCA marginal-cost dry-run ─────────────────────────────────────────────────
//
// Read-only mirror of DeliveryAgent::receive_task() cheapest-insertion search.
// Returns the minimum route-cost delta (detour) for inserting task into agent
// a's sequence over all admissible (pos_P, pos_D) positions:
//   - pos_P in [1, n]   — preserves in-flight head at index 0
//   - pos_D in [pos_P, n]
//   - capacity constraint: peak load in [pos_P-1 .. pos_D-1] + 1 <= max_carry
// For an idle agent (n==0) returns the point-to-point direct trip cost.

float EpisodeRunner::compute_marginal_cost(const DeliveryAgent& a,
                                           const PDPTask&       task) {
    const auto& seq = a.solution.sequence;
    const int   n   = static_cast<int>(seq.size());
    const int   max_carry = std::max(
        1, memory_.task_agent.params.max_tasks_per_agent);

    auto seg = [&](osmium::object_id_type from,
                   osmium::object_id_type to) -> float {
        if (from == 0 || to == 0 || from == to) return 0.f;
        const auto* p = memory_.get_or_compute_path(from, to, 1);
        return (p && p->valid()) ? p->cost : kCostScale;
    };

    if (n == 0)
        return seg(a.current_node, task.pickup.id)
             + seg(task.pickup.id, task.delivery.id);

    // Build carry-load profile (mirrors receive_task — task not yet in sequence).
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

// ── Double-horizon insertion cost [Mitrovic-Minic+2004] — faithful adaptation ─
//
// Translation of the paper's c3 cost into our capacity-constrained L-GPDP:
//   c = [(1−α_p)·f_p + α_p·g_p] + [(1−α_d)·f_d + α_d·g_d]
//
// where:
//   f_p, f_d : route-length increase (in metres) from inserting P at pos_P
//              and D at pos_D — same as MCA's per-position delta.
//
//   g_p, g_d : DECREASE in total slack at all subsequent locations due to the
//              insertion. Inserting at pos_P pushes every later location by
//              the time-equivalent of f_p (= f_p / speed). The slack at each
//              of the (n − pos_P) later locations decreases by that amount,
//              so total slack decrease (kept in metres for unit consistency):
//                  g_p = (n − pos_P) · f_p
//                  g_d = (n − pos_D) · f_d
//
//   α_p, α_d : 0 if estimated arrival time at the inserted location lies
//              within the SHORT-TERM horizon (≤ current_step + H_short),
//              0.25 otherwise. Matches paper's c3.
//
// Horizon partition is by ESTIMATED ARRIVAL TIME, not by sequence position,
// matching the paper. Estimated arrival uses agent.solution.sequence[k].
// estimated_arrival (filled by commit_plan); if -1 we accumulate travel times
// from current_node onward.

float EpisodeRunner::compute_double_horizon_cost(const DeliveryAgent& a,
                                                  const PDPTask& task,
                                                  int /*unused_legacy*/,
                                                  float alpha) {
    const auto& seq = a.solution.sequence;
    const int   n   = static_cast<int>(seq.size());
    const int   max_carry = std::max(
        1, memory_.task_agent.params.max_tasks_per_agent);

    auto seg = [&](osmium::object_id_type from,
                   osmium::object_id_type to) -> float {
        if (from == 0 || to == 0 || from == to) return 0.f;
        const auto* p = memory_.get_or_compute_path(from, to, 1);
        return (p && p->valid()) ? p->cost : kCostScale;
    };

    // Recover current simulation step from the time ratio published by run().
    const int current_step = static_cast<int>(
        memory_.cur_time_ratio * static_cast<float>(memory_.total_steps));
    const float speed = std::max(0.1f, cfg_.speed_mps);

    // Short-term horizon length (steps). Paper uses 1h on a 10h service period
    // = 10% of total. We mirror that ratio: 10% of remaining episode steps,
    // with a minimum of 200 steps to avoid degenerate splits late in episode.
    const int remaining = std::max(0, memory_.total_steps - current_step);
    const int short_horizon_steps = std::max(200, remaining / 10);

    // Idle agent (n=0): trivial direct trip, short-term by construction.
    if (n == 0)
        return seg(a.current_node, task.pickup.id)
             + seg(task.pickup.id, task.delivery.id);

    // Load profile (same as MCA).
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

    // Estimated arrival time per existing sequence position. Prefer the
    // commit_plan-published value; fall back to forward simulation from
    // current_node if not yet committed.
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

            // f_p, f_d — route-length increases (same as MCA).
            float f_p, f_d;
            if (pP == pD) {
                osmium::object_id_type before = seq[pP - 1].node.id;
                osmium::object_id_type after  = (pP == n) ? 0 : seq[pP].node.id;
                const float c_bp = seg(before, task.pickup.id);
                const float c_pd = seg(task.pickup.id, task.delivery.id);
                const float c_da = seg(task.delivery.id, after);
                const float c_ba = seg(before, after);
                // Split the joint insertion delta into a "p-side" and a "d-side"
                // for the g_p / g_d weighting to apply at the right pos.
                f_p = c_bp + c_pd;          // travelled before reaching D
                f_d = c_da - c_ba;          // residual delta after D
                if (f_d < 0.f) f_d = 0.f;   // numeric safety
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

            // Estimated arrival time of the NEW pickup / delivery.
            const int t_before_pP = (pP == 0) ? current_step : arr[pP - 1];
            const int t_before_pD = (pD == 0) ? current_step : arr[pD - 1];
            // Time at the freshly inserted P / D themselves:
            const int t_new_P = t_before_pP
                + static_cast<int>(std::ceil(seg(
                    (pP == 0 ? a.current_node : seq[pP - 1].node.id),
                    task.pickup.id) / speed));
            const int t_new_D = t_before_pD
                + static_cast<int>(std::ceil(seg(
                    (pD == pP ? task.pickup.id
                              : (pD == 0 ? a.current_node : seq[pD - 1].node.id)),
                    task.delivery.id) / speed));

            // Horizon membership per location (paper c3 rule).
            const bool p_short = (t_new_P - current_step) <= short_horizon_steps;
            const bool d_short = (t_new_D - current_step) <= short_horizon_steps;
            const float alpha_p = p_short ? 0.f : alpha;
            const float alpha_d = d_short ? 0.f : alpha;

            // g_p, g_d — slack decrease at subsequent locations (paper definition,
            // converted to metres via shared speed for unit consistency with f).
            // g = (n - pos) · f  expresses "the insertion pushes every later
            // location by f metres of travel distance" in route-length units.
            const float g_p = static_cast<float>(std::max(0, n - pP)) * f_p;
            const float g_d = static_cast<float>(std::max(0, n - pD)) * f_d;

            const float cost = (1.f - alpha_p) * f_p + alpha_p * g_p
                             + (1.f - alpha_d) * f_d + alpha_d * g_d;

            if (cost < best_cost) best_cost = cost;
        }
    }
    return best_cost;
}

// ── GlobalState assembly ──────────────────────────────────────────────────────

GlobalState EpisodeRunner::build_global_state(int step, int total_steps,
                                              float phase_label, float lambda,
                                              float city_norm, int n_active) const {
    GlobalState gs;
    int tasks_total = n_accepted_ + n_refused_;

    gs.time_ratio     = total_steps > 0
        ? static_cast<float>(step) / total_steps : 0.f;
    // Normalise against the actual agent pool (sized 1.5× the nominal
    // max_agents() in the constructor) so the ratio stays in [0,1] even
    // when EpisodeScenario.agents_mult exceeds 1.0.
    const int pool = static_cast<int>(all_agents_.size());
    gs.n_agents_ratio = pool > 0
        ? std::clamp(static_cast<float>(n_active) / pool, 0.f, 1.f) : 0.f;
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
    // Running mean delivery efficiency (reward per km traversed), squashed
    // into [0,1] for the critic. Provides a real route-quality signal that
    // shifts with the spatial mix of accepted tasks.
    if (efficiency_count_ > 0) {
        const float mean_eff = static_cast<float>(
            efficiency_sum_ / efficiency_count_);
        gs.avg_efficiency = std::clamp(mean_eff / 10.f, 0.f, 1.f);
    } else {
        gs.avg_efficiency = 0.f;
    }

    gs.city_id_norm  = city_norm;
    gs.density_phase = phase_label;
    gs.cluster_ratio = 0.f; // could track hot-zone fraction if needed
    // Mean per-edge agent load at the current step, normalised to [0,1] using
    // the BPR capacity scaling: a load equal to capacity_per_meter implies
    // strong congestion. With default 0.05 slots/m, an average load > 2 is
    // already heavy congestion in most edge sizes, so clamp at 5.
    gs.congestion = std::clamp(memory_.congestion_map.mean_load_now() / 5.f, 0.f, 1.f);

    return gs;
}
