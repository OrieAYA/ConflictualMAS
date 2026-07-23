#include "TrainingEvaluation/Run/Runner.hpp"
#include "Environment/Structure/EpisodeManager.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "DMASforPD/Agents/TaskAgent.hpp"
#include "DMASforPD/Structures/OperableEnvironment.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// ── Construction / destruction ────────────────────────────────────────────────

EpisodeRunner::EpisodeRunner(const EpisodeConfig& cfg,
                             GeoBox&              geo_box,
                             uint32_t             seed)
    : cfg_(cfg), memory_(geo_box), gen_(cfg, geo_box, seed), rng_(seed)
{
    // Fleet load amplifier: a smaller fleet keeps the same congestion
    // footprint (each commit registers load_per_agent units per edge).
    memory_.congestion_map.params.load_per_agent =
        std::max(1, cfg_.fleet_load_per_agent);

    // plan() and commit_plan() must share the same speed for arrivals/loads.
    memory_.task_agent.params.default_speed_mps = cfg_.speed_mps;
    memory_.task_agent.params.max_tasks_per_agent = cfg_.max_tasks_per_agent;

    // Single shared group cache for synthetic task paths (group 0 = sentinel).
    memory_.ensure_task_group(1);

    // Any road-connected node works as the default agent start.
    osmium::object_id_type start = 0;
    for (const auto& [id, pt] : geo_box.data.nodes) {
        if (!pt.incident_ways.empty()) { start = id; break; }
    }
    if (start == 0)
        throw std::runtime_error("EpisodeRunner: no valid road node for agent start");

    // Pool over-provisioned by agent_pool_multiplier so scenarios with
    // agents_mult > 1 are not clamped; idle surplus agents are inert.
    const float pool_mult = std::max(1.0f, cfg_.agent_pool_multiplier);
    int n = std::max(1, static_cast<int>(std::ceil(cfg_.max_agents() * pool_mult)));
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

void EpisodeRunner::release_episode_memory() {
    memory_.server_memory.clear_paths();
    last_episode_seed_ = 0;
}

// ── Episode preparation ───────────────────────────────────────────────────────

void EpisodeRunner::prepare_run(const EpisodeScenario& scenario,
                                uint32_t episode_seed,
                                const SharedEpisodeSetup* setup) {
    arrivals_.clear();
    task_accept_buf_idx_.clear();

    // Deterministic replay: re-anchor the RNGs so the same (city, scenario,
    // episode) slot faces identical task streams and capacity draws whatever
    // the policy. The ghost controller gets its own derived seed later.
    if (episode_seed != 0) {
        gen_.reset_seed(episode_seed);
        rng_.seed(episode_seed);
    }

    // Resolve the active policy and publish its kind to the Manager so
    // DeliveryAgent::bid_for_task routes to the right backend.
    active_policy_ = nullptr;
    switch (policy_mode) {
        case PolicyMode::IPPO:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kIPPO;
            active_policy_ = &bid_policy(BidPolicyKind::IPPO);
            break;
        case PolicyMode::MAPPER:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kMAPPER;
            active_policy_ = &bid_policy(BidPolicyKind::MAPPER);
            break;
        case PolicyMode::Hybrid:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kHybrid;
            active_policy_ = &bid_policy(BidPolicyKind::Hybrid);
            break;
        case PolicyMode::RMCA:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kRMCA;
            break;
        case PolicyMode::MAPPO:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kMAPPO;
            active_policy_ = &bid_policy(BidPolicyKind::MAPPO);
            break;
        default:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kMAPPO;
            break;
    }

    // Always start from clean buffers: eval episodes also record decisions,
    // and stale records must not leak into the next training update.
    // reset_episode() also zeroes Hybrid's online residuals so each episode
    // restarts from the frozen base (état 0), order-independent.
    if (active_policy_) {
        active_policy_->clear_buffers();
        active_policy_->clear_recent_records();
        active_policy_->reset_episode();
    }

    memory_.record_plan_congestion = cfg_.use_movement_policy;
    mv_stats_ = {};
    if (cfg_.use_movement_policy) movement_policy().clear_buffers();

    lsm_hist_.clear();
    lsm_alert_.clear();
    lsm_stats_ = {};
    if (cfg_.use_lsm) lsm_module().reset_state();

    acc_.reset();
    PDPServerMemory::reset_path_compute_time();

    scales_ = make_reward_scales(cfg_);
    cins_sum_ = 0.0; cins_n_ = 0;

    // TAM parameters must be in place before any TAM is created this episode.
    memory_.task_agent.params.tam_params.always_accept =
        (policy_mode == PolicyMode::TamAlwaysAccept);
    {
        auto& tp = memory_.task_agent.params.tam_params;
        tp.force_assign   = cfg_.tam_mc_force_assign;
        tp.max_candidates = cfg_.tam_mc_max_candidates;
        tp.ratio_min      = cfg_.tam_mc_ratio_min;
        tp.ratio_max      = cfg_.tam_mc_ratio_max;
        tp.ratio_scale    = cfg_.tam_mc_ratio_scale;
    }
    retry_queue_.clear();

    // Bids are sampled (exploration) only in training; eval uses mu >= 0.5.
    memory_.exploration_enabled = train_mode;

    // Reset per-episode state (tasks, plans, congestion, clock). The memoised
    // path cache is kept only across replays of the SAME episode (identical
    // task nodes); new episode content → purge, otherwise it grows without
    // bound over a training grid (~24 GB observed).
    memory_.reset_episode();
    if (episode_seed == 0 || episode_seed != last_episode_seed_) {
        memory_.server_memory.clear_paths();
        last_episode_seed_ = episode_seed;
    }

    memory_.speed_mps   = cfg_.speed_mps;
    memory_.total_steps = cfg_.total_steps();

    // Planning dispatcher — at most one flag active. DbVNS wins over DH when
    // both are set in cfg; PolicyMode entries force their respective flag.
    const bool cfg_dbvns = cfg_.use_dbvns_planning;
    const bool cfg_dh    = cfg_.use_double_horizon_planning && !cfg_dbvns;
    memory_.planning_use_dbvns =
        cfg_dbvns || (policy_mode == PolicyMode::DbVNS);
    memory_.planning_use_double_horizon =
        (cfg_dh || policy_mode == PolicyMode::DoubleHorizon)
        && !memory_.planning_use_dbvns;
    memory_.planning_use_alns  = (policy_mode == PolicyMode::ALNS);

    // Heterogeneous fleet: per-agent capacity draw in [min, max]; the TAM
    // global ceiling is bumped to the max so container sizing stays valid.
    // max_capacity == 0 => agent inherits the TAM global.
    if (cfg_.enable_heterogeneous_capacity) {
        const int cmin = std::max(1, cfg_.hetero_capacity_min);
        const int cmax = std::max(cmin, cfg_.hetero_capacity_max);
        memory_.task_agent.params.max_tasks_per_agent = cmax;
        if (setup && !setup->per_agent_capacity.empty()) {
            // Canonical draws shared with SolverRunner (byte-identical fleet).
            for (size_t i = 0; i < all_agents_.size(); ++i) {
                all_agents_[i]->max_capacity =
                    (i < setup->per_agent_capacity.size())
                        ? setup->per_agent_capacity[i]
                        : 0;
            }
        } else {
            std::uniform_int_distribution<int> cap_dist(cmin, cmax);
            for (auto& a : all_agents_)
                a->max_capacity = cap_dist(rng_);
        }
    } else {
        for (auto& a : all_agents_)
            a->max_capacity = 0;
    }

    for (auto& a : all_agents_) {
        a->status = AgentStatus::Idle;
        a->n_tasks_ever_assigned = 0;
        a->in_transit.reset();
        a->edge_cursor.reset();
        a->local_memory.current_path = nullptr;
        a->local_memory.next_path    = nullptr;
        a->solution.sequence.clear();
        a->local_memory.tasks.clear();
        a->local_memory.local_agents.clear();
        a->local_memory.operable_env = OperableEnvironment{};
        a->local_memory.plan_cong.clear();
    }

    // Canonical start positions when a SharedEpisodeSetup is provided (same
    // sampling as SolverRunner); surplus agents keep the default start.
    if (setup && !setup->agent_start_nodes.empty()) {
        const size_t n_canon = setup->agent_start_nodes.size();
        for (size_t i = 0; i < all_agents_.size() && i < n_canon; ++i)
            all_agents_[i]->current_node = setup->agent_start_nodes[i];
    }

    // Effective active fleet = provisioned pool × agents_mult (same value as
    // SolverRunner's n_active_agents).
    episode_fleet_size_ = (setup && setup->n_active_agents > 0)
        ? std::min(static_cast<int>(all_agents_.size()), setup->n_active_agents)
        : std::min(static_cast<int>(all_agents_.size()),
                   std::max(1, static_cast<int>(std::round(
                       cfg_.max_agents() * scenario.agents_mult))));
}

// ── Main episode loop ─────────────────────────────────────────────────────────

RunResult EpisodeRunner::run(int city_index, int num_cities,
                             EpisodeScenario scenario,
                             uint32_t episode_seed,
                             const SharedEpisodeSetup* setup) {
    auto t0 = std::chrono::steady_clock::now();

    prepare_run(scenario, episode_seed, setup);

    // Task stream: consume the shared setup verbatim (density already applied)
    // or generate the full episode stream now and apply the scenario density
    // with the same canonical sub/supersample the cross-method eval uses.
    auto stream = setup ? setup->task_stream
                        : gen_.generate(scenario.task_profile);
    if (!setup && scenario.density_mult != 1.0f && !stream.empty()) {
        const uint32_t dseed = (episode_seed != 0)
            ? (episode_seed ^ 0xA17EFEEDu)
            : static_cast<uint32_t>(rng_());
        apply_density_mult(stream, scenario.density_mult, dseed);
    }
    auto phase_table = gen_.build_phase_table();
    int  total_steps = cfg_.total_steps();
    int  stream_idx  = 0;

    // Expected arrivals per step (GlobalState arrival_rate feature): stream
    // size distributed along the scenario's task profile shape.
    std::vector<float> step_rate(static_cast<size_t>(std::max(0, total_steps)), 0.f);
    if (total_steps > 0 && !stream.empty()) {
        double norm = 0.0;
        for (int s = 0; s < total_steps; ++s)
            norm += std::max(0.f, temporal_profile_value(
                scenario.task_profile, (s + 0.5f) / total_steps));
        for (int s = 0; s < total_steps; ++s)
            step_rate[s] = (norm > 0.0)
                ? static_cast<float>(stream.size()
                    * std::max(0.f, temporal_profile_value(
                          scenario.task_profile, (s + 0.5f) / total_steps)) / norm)
                : static_cast<float>(stream.size()) / total_steps;
    }

    const bool ghost_on = cfg_.enable_ghost_traffic && scenario.density_mult >= 0.f;
    setup_ghost_traffic(scenario, setup, total_steps, city_index, ghost_on);

    float city_norm = (num_cities > 1)
        ? static_cast<float>(city_index) / (num_cities - 1) : 0.f;

    for (int step = 0; step < total_steps; ++step) {
        memory_.advance_time(step);

        // Background traffic before any agent decision, so policy features
        // and TAM cost queries see the up-to-date load.
        if (ghost_on) ghost_traffic_.step(step);

        // Current phase parameters (fleet ramp + label).
        float phase_label = cfg_.phases.empty() ? 0.f : cfg_.phases.front().label;
        int   n_active    = cfg_.phases.empty() ? (int)all_agents_.size()
                                                : cfg_.phases.front().n_agents_start;
        for (const auto& ph : phase_table) {
            if (step >= ph.step_begin && step < ph.step_end) {
                phase_label = ph.label;
                n_active    = ph.n_agents_at(step);
                break;
            }
        }
        const float lambda =
            (step < (int)step_rate.size()) ? step_rate[step] : 0.f;
        n_active  = std::max(1, static_cast<int>(std::round(n_active * scenario.agents_mult)));
        n_active = std::min(n_active, static_cast<int>(all_agents_.size()));

        if (cfg_.use_lsm && step % std::max(1, cfg_.lsm_every) == 0)
            lsm_tick(step, total_steps, lambda, n_active);

        // Arrivals first, so agents that just delivered are Idle for new work.
        process_arrivals(step);

        // Publish the global state once per step; bid_for_task snapshots it
        // into every experience (the centralised critic trains against it).
        auto cur_gs = build_global_state(step, total_steps, phase_label, lambda,
                                         city_norm, n_active).to_array();
        std::copy(cur_gs.begin(), cur_gs.end(), memory_.cur_global_state);

        retry_deferred_tasks(step, total_steps, n_active, cur_gs);

        // Inject tasks arriving at this step (stream is pre-generated; a task
        // materialises only once its arrival step is reached).
        while (stream_idx < (int)stream.size() &&
               stream[stream_idx].arrival_step == step) {
            const auto& st = stream[stream_idx++];

            ObjectiveNode pickup  { st.pickup_node_id,   1 };
            ObjectiveNode delivery{ st.delivery_node_id, 1 };
            int task_id = memory_.add_task(pickup, delivery);

            PDPTask* task = memory_.get_task(task_id);
            task->reward     = st.reward;
            task->importance = st.importance;
            task->reward_original     = st.reward;
            task->importance_original = st.importance;

            // Spatial heatmap feed (PolicyFeatures::area_heat_pickup).
            {
                auto it = memory_.geo_box.data.nodes.find(pickup.id);
                if (it != memory_.geo_box.data.nodes.end()) {
                    memory_.region_grid.register_task(
                        it->second.lat, it->second.lon, step);
                }
            }

            acc_.value_appeared_sum += static_cast<double>(st.reward) * st.importance;

            OfferResult res = offer_with_metrics(*task, step, n_active, cur_gs);
            if (res.deferred) {
                const int retry_interval = std::max(1, static_cast<int>(
                    cfg_.tam_mc_recall_time_frac * static_cast<float>(total_steps)));
                retry_queue_.push_back({ task_id, step + retry_interval });
            }
            // Refused tasks stay in available_tasks but are not re-offered.
        }

        sample_step_state(step, n_active);
    }

    // Tasks still deferred at episode end count as refusals so that
    // accepted + refused = appeared stays consistent.
    for (const auto& re : retry_queue_) {
        PDPTask* rt = memory_.get_task(re.task_id);
        acc_.n_refused += 1;
        if (rt) {
            acc_.imp_refused_sum += rt->importance_original;
            acc_.imp_refused_n   += 1;
            acc_.value_refused_sum += static_cast<double>(rt->reward_original)
                                    * rt->importance_original;
        }
    }
    retry_queue_.clear();

    // End-of-episode metric aggregation (Metrics.cpp — Part 2).
    acc_.path_compute_time_us_ep = PDPServerMemory::path_compute_time_us();
    ComparisonMetrics metrics = finalize_episode_metrics(
        acc_, memory_, all_agents_, cfg_, total_steps, episode_fleet_size_,
        ghost_on, ghost_traffic_.mean_active(),
        temporal_profile_label(scenario.congestion_profile));

    ghost_traffic_.purge();

    RunResult result;
    result.metrics = metrics;

    // Unfinished-acceptance penalty: accepted but never delivered. Scales with
    // the lost task value so the policy learns not to over-accept; Hybrid
    // receives it regardless of train_mode (online adaptation every episode).
    const bool apply_unfinished_penalty = active_policy_
        && (train_mode || active_policy_->trains_online());
    if (apply_unfinished_penalty && cfg_.unfinished_factor > 0.f) {
        const float w_unf = cfg_.unfinished_factor;   // constant (not annealed)
        const float cins_mean = (cins_n_ > 0)
            ? static_cast<float>(cins_sum_ / cins_n_) : 0.f;
        for (const auto& [tid, buf_idx] : task_accept_buf_idx_) {
            PDPTask* t = memory_.get_task(tid);
            if (!t) continue;
            const float val    = t->reward_original * t->importance_original;
            const bool  picked = (t->timeline.picked_step >= 0);
            const float lost_frac = picked ? (1.f - cfg_.pickup_reward_frac) : 1.f;
            float pen = -w_unf * lost_frac * val / scales_.task_value;
            // Failing an expensive-to-insert task hurts more (ratio to the
            // episode's mean accepted insertion cost).
            if (cfg_.rs_unfinished_cost_ratio
                && t->c_ins_at_accept >= 0.f && cins_mean > 0.f)
                pen *= std::clamp(t->c_ins_at_accept / cins_mean,
                                  cfg_.rs_unfinished_clip_lo,
                                  cfg_.rs_unfinished_clip_hi);
            active_policy_->add_to_reward(t->agent_id, buf_idx, pen);
        }
    }

    // Idle-agent penalty: offer entries of agents that stayed idle are debited.
    const bool apply_idle_penalty = active_policy_
        && (train_mode || active_policy_->trains_online())
        && cfg_.idle_penalty_w > 0.f;
    if (apply_idle_penalty) {
        const float w_idle = rs_w(cfg_.idle_penalty_w, 0.f);
        for (const auto& a : all_agents_) {
            if (!a) continue;
            // Refined condition: idle = never assigned any task over the
            // whole episode, not just the end-of-episode status.
            const bool idle_all_episode = cfg_.rs_idle_refine
                ? (a->n_tasks_ever_assigned == 0)
                : (a->status == AgentStatus::Idle && a->local_memory.tasks.empty());
            if (!idle_all_episode) continue;
            const int sz = active_policy_->buffer_size(a->agent_id);
            for (int i = 0; i < sz; ++i) {
                // Only debit hesitant bids (mu inside the config band);
                // confident refuses/bids made a consistent choice.
                if (cfg_.rs_idle_refine) {
                    const float mu = active_policy_->entry_mu(a->agent_id, i);
                    if (mu < cfg_.rs_mu_idle_lo || mu > cfg_.rs_mu_idle_hi)
                        continue;
                }
                active_policy_->add_to_reward(a->agent_id, i, -w_idle);
            }
        }
    }

    // Per-episode learning update: PPO policies only in train_mode, Hybrid at
    // the end of every episode (trains_online).
    if (active_policy_ && active_policy_->total_buffer_size() > 0
        && (train_mode || active_policy_->trains_online())) {
        result.train_stats = active_policy_->train_round();
    }

    if (cfg_.use_movement_policy && cfg_.movement_train
        && movement_policy().total_buffer_size() > 0)
        mv_stats_.train = movement_policy().train_round();

    auto t1 = std::chrono::steady_clock::now();
    result.wallclock_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

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

// ── Episode-loop helpers ──────────────────────────────────────────────────────

// Hot ways are resampled per episode so congested zones rotate. The seed mixes
// master seed + city + scenario label for deterministic replay; the shared
// setup's canonical ghost seed takes precedence (byte-identical traffic).
void EpisodeRunner::setup_ghost_traffic(const EpisodeScenario& scenario,
                                        const SharedEpisodeSetup* setup,
                                        int total_steps, int city_index,
                                        bool ghost_on) {
    if (!ghost_on) {
        ghost_traffic_.purge();
        return;
    }
    GhostTrafficController::Config gcfg;
    gcfg.total_steps      = total_steps;
    gcfg.window_steps     = cfg_.ghost_window_steps;
    gcfg.hot_way_fraction = cfg_.ghost_hot_way_frac;
    gcfg.hot_way_count    = cfg_.ghost_hot_way_count;
    gcfg.profile          = scenario.congestion_profile;
    gcfg.n_events = cfg_.ghost_n_events_user_set
        ? cfg_.ghost_n_events
        : event_tuning::derived_ghost_count(cfg_.env_scale, cfg_.ratio_mult);

    std::hash<std::string> shash;
    const uint32_t ghost_seed = setup
        ? setup->ghost_seed
        : static_cast<uint32_t>(rng_()
            ^ shash(scenario.label)
            ^ (static_cast<uint32_t>(city_index) << 16));
    ghost_traffic_.reset(memory_.geo_box, memory_.congestion_map, gcfg, ghost_seed);
}

EpisodeRunner::OfferResult EpisodeRunner::offer_with_metrics(
    PDPTask& task, int step, int n_active,
    const std::array<float, kGlobSz>& gs)
{
    // Congestion at the exact step the policy is consulted (distinct from the
    // per-step mean over the whole episode).
    const float cong_at_decision = memory_.congestion_map.mean_load_now();
    const bool  high_cong_now    = (cong_at_decision >= 1.0f);
    acc_.congestion_at_decision_sum   += cong_at_decision;
    acc_.congestion_at_decision_count += 1;

    OfferResult res = offer_task(task.task_id, task.reward_original,
                                 task.importance_original, n_active, gs);

    acc_.on_offer_sample(last_offer_stats_.agents_offered,
                         last_offer_stats_.recall_rounds,
                         last_offer_stats_.candidates_scored,
                         last_offer_stats_.allocation_time_us,
                         last_offer_stats_.pure_alloc_time_us,
                         last_offer_stats_.tam_dijkstra_steps);
    if (res.agent_id >= 0)
        acc_.on_allocation_choice(last_offer_stats_.pre_marginal_costs,
                                  res.agent_id,
                                  policy_mode == PolicyMode::RMCA);
    if (res.deferred) return res;

    if (res.agent_id >= 0) {
        acc_.on_accept(task.importance_original, high_cong_now);
        if (task.c_ins_at_accept >= 0.f) {
            cins_sum_ += task.c_ins_at_accept;
            cins_n_   += 1;
        }
        commit_accepted_task(task.task_id, res.agent_id, res.tam_owned,
                             task.pickup.id, step);
    } else {
        // No valid candidate anywhere (capacity/topology) — not a policy
        // refusal; Format A force-assigns whenever one candidate exists.
        acc_.n_no_candidate += 1;
    }
    return res;
}

// Re-offer deferred tasks whose retry step is due. Past the reject cutoff a
// task counts as a genuine refusal; still-deferred tasks are re-queued.
void EpisodeRunner::retry_deferred_tasks(int step, int total_steps, int n_active,
                                         const std::array<float, kGlobSz>& gs)
{
    const int retry_interval = std::max(1, static_cast<int>(
        cfg_.tam_mc_recall_time_frac * static_cast<float>(total_steps)));
    const int reject_cutoff  = static_cast<int>(
        cfg_.tam_mc_reject_time_frac * static_cast<float>(total_steps));
    for (size_t qi = 0; qi < retry_queue_.size(); ) {
        if (retry_queue_[qi].retry_step > step) { ++qi; continue; }
        const int rtask = retry_queue_[qi].task_id;
        PDPTask*  rt    = memory_.get_task(rtask);
        auto drop = [&]{
            retry_queue_[qi] = retry_queue_.back();
            retry_queue_.pop_back();
        };
        if (!rt) { drop(); continue; }

        if (step > reject_cutoff) {
            acc_.n_refused += 1;
            acc_.imp_refused_sum += rt->importance_original;
            acc_.imp_refused_n   += 1;
            drop();
            continue;
        }

        OfferResult res = offer_with_metrics(*rt, step, n_active, gs);
        if (res.deferred) {
            retry_queue_[qi].retry_step = step + retry_interval;
            ++qi;
        } else {
            drop();
        }
    }
}

void EpisodeRunner::sample_step_state(int step, int n_active) {
    const float load_now_step = memory_.congestion_map.mean_load_now();
    acc_.congestion_sum   += load_now_step;
    acc_.congestion_steps += 1;

    const int peak_step = memory_.congestion_map.peak_load_now();
    if (peak_step > acc_.peak_load_episode) acc_.peak_load_episode = peak_step;
    acc_.overlap_edges_sum += memory_.congestion_map.n_edges_load_ge(2);
    acc_.overlap_steps     += 1;
    acc_.load_now_sum_lin  += static_cast<double>(load_now_step);
    acc_.load_now_sum_sq   += static_cast<double>(load_now_step) * load_now_step;
    acc_.load_now_n        += 1;

    // Route exposure: load on edges currently traversed by real agents.
    for (const auto& a : all_agents_) {
        if (a->in_transit.has_value()) {
            const int load_here = memory_.congestion_map.get_load(
                a->in_transit->edge_id, step);
            acc_.route_exposure_sum += static_cast<double>(load_here);
            acc_.route_exposure_n   += 1;
        }
    }

    int n_now = 0;
    for (int i = 0; i < n_active; ++i)
        if (all_agents_[i]->status == AgentStatus::Active) ++n_now;
    acc_.active_sum   += n_now;
    acc_.active_steps += 1;
}

// ── GlobalState assembly ──────────────────────────────────────────────────────

GlobalState EpisodeRunner::build_global_state(int step, int total_steps,
                                              float phase_label, float lambda,
                                              float city_norm, int n_active) const {
    GlobalState gs;
    // n_no_candidate stays in the denominator so every gs feature keeps the
    // exact values the existing checkpoints were trained on.
    int tasks_total = acc_.n_accepted + acc_.n_refused + acc_.n_no_candidate;

    gs.time_ratio     = total_steps > 0
        ? static_cast<float>(step) / total_steps : 0.f;
    // Normalised against the actual (over-provisioned) pool so the ratio
    // stays in [0,1] even when agents_mult exceeds 1.0.
    const int pool = static_cast<int>(all_agents_.size());
    gs.n_agents_ratio = pool > 0
        ? std::clamp(static_cast<float>(n_active) / pool, 0.f, 1.f) : 0.f;
    gs.avail_ratio    = tasks_total > 0
        ? static_cast<float>(memory_.count_available()) / tasks_total : 0.f;
    gs.alloc_ratio    = tasks_total > 0
        ? static_cast<float>(memory_.count_allocated()) / tasks_total : 0.f;
    gs.done_ratio     = tasks_total > 0
        ? static_cast<float>(acc_.latency_count) / tasks_total : 0.f;

    float load_sum = 0.f, max_load = 0.f;
    for (const auto& a : all_agents_) {
        float l = static_cast<float>(a->local_memory.tasks.size());
        load_sum += l;
        max_load  = std::max(max_load, l);
    }
    gs.avg_load = all_agents_.empty() ? 0.f
        : (load_sum / all_agents_.size()) / kMaxLoad;
    gs.max_load = max_load / kMaxLoad;

    // Normalised against the high-density phase ceiling (0.20).
    gs.arrival_rate = std::clamp(lambda / 0.20f, 0.f, 1.f);

    gs.throughput  = tasks_total > 0
        ? static_cast<float>(acc_.latency_count) / tasks_total : 0.f;
    gs.avg_latency = (acc_.latency_count > 0 && cfg_.time_window_steps > 0)
        ? (static_cast<float>(acc_.latency_sum) / acc_.latency_count)
          / cfg_.time_window_steps : 0.f;
    gs.accept_rate = tasks_total > 0
        ? static_cast<float>(acc_.n_accepted) / tasks_total : 0.f;
    // Running mean delivery efficiency (reward per km), squashed into [0,1].
    if (acc_.efficiency_count > 0) {
        const float mean_eff = static_cast<float>(
            acc_.efficiency_sum / acc_.efficiency_count);
        gs.avg_efficiency = std::clamp(mean_eff / 10.f, 0.f, 1.f);
    } else {
        gs.avg_efficiency = 0.f;
    }

    gs.city_id_norm  = city_norm;
    gs.density_phase = phase_label;
    gs.cluster_ratio = 0.f;
    // Mean per-edge load now, clamped at 5 (heavy congestion under the
    // default 0.05 slots/m capacity scaling).
    gs.congestion = std::clamp(memory_.congestion_map.mean_load_now() / 5.f, 0.f, 1.f);

    return gs;
}
