#include "TrainingEvaluation/Run/Runner.hpp"
#include "Environment/Structure/EpisodeManager.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "DMASforPD/Agents/TaskAgent.hpp"
#include "SoTA/TaskAgentLike/TP.hpp"
#include "DMASforPD/Structures/OperableEnvironment.hpp"
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
    // Propagate the fleet load amplifier into the CongestionMap. Each real
    // agent's commit_plan registers load_per_agent units per edge, letting
    // the user shrink the fleet while keeping the same congestion footprint.
    memory_.congestion_map.params.load_per_agent =
        std::max(1, cfg_.fleet_load_per_agent);

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

    // Allocate the agent pool with cfg_.agent_pool_multiplier × overhead so
    // the per-episode EpisodeScenario.agents_mult can sample values above 1.0
    // (slack regime, or "over-provisioned fleet" stress scenarios at up to 10×)
    // without being clamped down to the nominal max_agents() at run-time.
    // Idle surplus agents are inert (never offered tasks when n_active < pool)
    // and cost only a small per-instance memory footprint. Default 1.5 (Option
    // T/M/X). Option Y bumps to 10 for the high-fleet sweep scenarios.
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

// ── Main episode loop ─────────────────────────────────────────────────────────

RunResult EpisodeRunner::run(int city_index, int num_cities,
                             EpisodeScenario scenario,
                             uint32_t episode_seed,
                             const SharedEpisodeSetup* setup) {
    auto t0 = std::chrono::steady_clock::now();
    arrivals_.clear();
    task_accept_buf_idx_.clear();

    // Deterministic episode replay: when the caller passes a non-zero seed,
    // re-anchor all RNGs so that this (city, scenario, episode) combination
    // produces the same task stream / ghost traffic / hetero capacity draw
    // regardless of which policy is being evaluated. The ghost controller is
    // reset later in the function with a derived seed that incorporates the
    // scenario label, so we only need to reset gen_ and rng_ here.
    if (episode_seed != 0) {
        gen_.reset_seed(episode_seed);
        rng_.seed(episode_seed);
    }
    // Resolve the active learning policy and publish its kind to the Manager
    // so DeliveryAgent::bid_for_task routes to the right backend. RMCA scores
    // deterministically inside the agent (no IBidPolicy instance).
    active_policy_ = nullptr;
    switch (policy_mode) {
        case PolicyMode::IPPO:
            memory_.active_policy = PDPGlobalMemory::PolicyKind::kIPPO;
            active_policy_ = &bid_policy(BidPolicyKind::IPPO);
            break;
        case PolicyMode::MAPPER:
        case PolicyMode::FaithfulMAPPER:   // legacy alias — same MAPPER policy
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

    // Start the episode with clean experience buffers — ALWAYS. An eval
    // episode also records decisions (Hybrid needs them; the others record
    // unconditionally); without this clear, those eval records would leak
    // into the next training update.
    if (active_policy_) {
        active_policy_->clear_buffers();
        active_policy_->clear_recent_records();
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
    wait_sum_         = 0;
    wait_count_       = 0;
    road_pd_sum_      = 0.0;
    road_pd_count_    = 0;
    congestion_at_decision_sum_   = 0.0;
    congestion_at_decision_count_ = 0;
    imp_accepted_sum_  = 0.0;
    imp_accepted_n_    = 0;
    imp_refused_sum_   = 0.0;
    imp_refused_n_     = 0;
    accepts_high_cong_ = 0;
    refuses_high_cong_ = 0;
    accepts_low_cong_  = 0;
    refuses_low_cong_  = 0;
    tam_agents_offered_sum_    = 0;
    tam_recall_rounds_sum_     = 0;
    tam_candidates_scored_sum_ = 0;
    tam_offer_samples_         = 0;
    value_appeared_sum_  = 0.0;
    value_delivered_sum_ = 0.0;
    value_refused_sum_   = 0.0;
    bpr_along_route_sum_         = 0.0;
    bpr_along_route_count_       = 0;
    time_lost_to_congestion_sum_ = 0.0;
    n_traversals_in_jam_         = 0;
    marginal_ratio_sum_     = 0.0;
    marginal_ratio_count_   = 0;
    rmca_regret_sum_        = 0.0;
    rmca_mc_k1_sum_         = 0.0;
    rmca_mc_k2_sum_         = 0.0;
    rmca_regret_count_      = 0;
    allocation_time_us_sum_ = 0;
    allocation_time_count_  = 0;
    tam_dijkstra_steps_sum_ = 0;
    tam_dijkstra_count_     = 0;
    pure_alloc_time_us_sum_ = 0;
    path_compute_time_us_ep_ = 0;
    // Reset the global path-compute timer so this episode's value starts at 0.
    PDPServerMemory::reset_path_compute_time();
    peak_load_episode_   = 0;
    overlap_edges_sum_   = 0;
    overlap_steps_       = 0;
    load_now_sum_sq_     = 0.0;
    load_now_sum_lin_    = 0.0;
    load_now_n_          = 0;
    route_exposure_sum_  = 0.0;
    route_exposure_n_    = 0;

    // Propagate the always_accept flag to the TAM for TamAlwaysAccept ablation.
    // Must be set before the episode loop so all TAMs created during the episode
    // inherit it.
    memory_.task_agent.params.tam_params.always_accept =
        (policy_mode == PolicyMode::TamAlwaysAccept);

    // Propagate the TAM parameters (the TAM is multi-candidate only).
    {
        auto& tp = memory_.task_agent.params.tam_params;
        tp.force_assign   = cfg_.tam_mc_force_assign;
        tp.max_candidates = cfg_.tam_mc_max_candidates;
        tp.ratio_min      = cfg_.tam_mc_ratio_min;
        tp.ratio_max      = cfg_.tam_mc_ratio_max;
        tp.ratio_scale    = cfg_.tam_mc_ratio_scale;
    }
    retry_queue_.clear();

    // Bids are sampled (exploration) only while training; evaluation uses the
    // deterministic μ >= 0.5 threshold so measured performance is the
    // policy's, not the sampler's.
    memory_.exploration_enabled = train_mode;

    // Reset per-episode GlobalMemory state (tasks, plans, congestion, clock).
    // Preserves the A* path cache so costs computed in prior episodes reuse.
    memory_.reset_episode();

    // Publish episode-level constants so the policy's per-decision features
    // can reference them (deliverability = steps_remaining / delivery_steps).
    memory_.speed_mps   = cfg_.speed_mps;
    memory_.total_steps = cfg_.total_steps();

    // Planning strategy flags.  At most one of the three modes is active.
    // DoubleHorizon and DbVNS override cfg_ so PlanningComparisonTest can
    // switch modes without touching EpisodeConfig.
    // Planning dispatcher — DbVNS takes precedence over DH if both are set
    // in cfg (defensive against accidental double-enable). The PlanningMode
    // entries (DoubleHorizon / DbVNS / ALNS) always force their respective
    // flag regardless of cfg.
    const bool cfg_dbvns = cfg_.use_dbvns_planning;
    const bool cfg_dh    = cfg_.use_double_horizon_planning && !cfg_dbvns;
    memory_.planning_use_dbvns =
        cfg_dbvns || (policy_mode == PolicyMode::DbVNS);
    memory_.planning_use_double_horizon =
        (cfg_dh || policy_mode == PolicyMode::DoubleHorizon)
        && !memory_.planning_use_dbvns;
    memory_.planning_use_alns  = (policy_mode == PolicyMode::ALNS);
    // MCA + anytime LNS improvement (Chen et al. 2021, ICRA — Algorithm 3).
    // Activated only on the MCA branch of the planning comparison; mutually
    // exclusive with DbVNS / DH / ALNS (those have their own replanner). The
    // LNS body lives in DeliveryAgent::receive_task right after the regular
    // cheapest-insertion commit — see the audit fix described there.
    memory_.planning_use_mca_lns =
        (policy_mode == PolicyMode::MCA)
        && !memory_.planning_use_dbvns
        && !memory_.planning_use_double_horizon
        && !memory_.planning_use_alns;

    // Per-agent capacity sampling (heterogeneous fleet, Option M).
    // When disabled, max_capacity stays 0 → DeliveryAgent falls back to
    // task_agent.params.max_tasks_per_agent (= cfg_.max_tasks_per_agent).
    // When enabled, each agent gets a uniform draw in [min, max]. TAM global
    // is bumped to hetero_capacity_max so container sizing stays valid.
    if (cfg_.enable_heterogeneous_capacity) {
        const int cmin = std::max(1, cfg_.hetero_capacity_min);
        const int cmax = std::max(cmin, cfg_.hetero_capacity_max);
        memory_.task_agent.params.max_tasks_per_agent = cmax;
        if (setup && !setup->per_agent_capacity.empty()) {
            // Canonical hetero capacities — same draws as SolverRunner sees,
            // so RL and SoTA face byte-identical fleet heterogeneity. Apply to
            // the FIRST n_active_agents (per index); idle surplus stays at 0.
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
            a->max_capacity = 0;  // inherit TAM global
    }

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

    // ── Canonical agent start positions (publication-grade eval only) ──────
    // When a SharedEpisodeSetup is provided, override the legacy
    // "all-agents-at-first-road-node" placement with the per-agent sampled
    // positions that SolverRunner also uses. Idle surplus agents (index >=
    // setup->n_active_agents) keep their legacy position — they are never
    // offered tasks so their start node is irrelevant.
    if (setup && !setup->agent_start_nodes.empty()) {
        const size_t n_canon = setup->agent_start_nodes.size();
        for (size_t i = 0; i < all_agents_.size() && i < n_canon; ++i)
            all_agents_[i]->current_node = setup->agent_start_nodes[i];
    }

    // ── Effective active fleet for this episode ───────────────────────────
    // The provisioned pool actually used, scaled by scenario.agents_mult — the
    // SAME value SolverRunner logs (setup->n_active_agents). Reported as
    // n_agents and used as the agent_utilisation denominator, replacing the
    // nominal cfg_.max_agents() which ignored agents_mult and under-reported
    // the fleet by the agents_mult factor (e.g. 6 vs 60 in over_fleet), even
    // though the candidate pool (n_active) was already scaled internally.
    episode_fleet_size_ = (setup && setup->n_active_agents > 0)
        ? std::min(static_cast<int>(all_agents_.size()), setup->n_active_agents)
        : std::min(static_cast<int>(all_agents_.size()),
                   std::max(1, static_cast<int>(std::round(
                       cfg_.max_agents() * scenario.agents_mult))));

    // Stream override: when a SharedEpisodeSetup is provided, consume its
    // pre-generated stream verbatim (already has density_mult applied via the
    // canonical subsample/supersample). Otherwise fall back to the legacy
    // gen_.generate() — used by training and Option Y.
    auto stream      = (setup ? setup->task_stream : gen_.generate());
    auto phase_table = gen_.build_phase_table();
    int  total_steps = cfg_.total_steps();
    int  stream_idx  = 0;

    // ── Optional background traffic controller (Option M diversification) ──
    // Built on top of CongestionMap::add_ghost_load; only spins up when the
    // EpisodeConfig flag is set. Hot ways are resampled per episode so the
    // congested zones rotate across the training distribution. RNG seed
    // composed from the runner's master seed + city + scenario label so
    // identical (seed, city, scenario) replay yields identical traffic.
    const bool ghost_on = cfg_.enable_ghost_traffic && scenario.density_mult >= 0.f;
    if (ghost_on) {
        GhostTrafficController::Config gcfg;
        gcfg.n_max              = cfg_.ghost_n_max;
        gcfg.total_steps        = total_steps;
        gcfg.window_steps       = cfg_.ghost_window_steps;
        gcfg.hot_way_fraction   = cfg_.ghost_hot_way_frac;
        gcfg.hot_way_count      = cfg_.ghost_hot_way_count;
        gcfg.density_per_hot_way = scenario.ghost_density_per_hot_way > 0.f
            ? scenario.ghost_density_per_hot_way : cfg_.ghost_density_per_hot_way;
        gcfg.load_per_ghost      = std::max(1, cfg_.ghost_load_per_unit);
        gcfg.profile            = scenario.congestion_profile;

        std::hash<std::string> shash;
        // Canonical ghost seed when a SharedEpisodeSetup is provided (Option O
        // eval) — same formula as SolverRunner so ghost traffic is byte-
        // identical across RL and SoTA. Otherwise use legacy formula
        // (training paths, Option Y unchanged).
        const uint32_t episode_seed = setup
            ? setup->ghost_seed
            : static_cast<uint32_t>(rng_()
                ^ shash(scenario.label ? scenario.label : "")
                ^ (static_cast<uint32_t>(city_index) << 16));
        ghost_traffic_.reset(memory_.geo_box,
                             memory_.congestion_map,
                             gcfg,
                             episode_seed);
    } else {
        ghost_traffic_.purge();
    }

    float city_norm = (num_cities > 1)
        ? static_cast<float>(city_index) / (num_cities - 1) : 0.f;

    ComparisonMetrics metrics;
    metrics.method = "DMAS-MAPPO";

    for (int step = 0; step < total_steps; ++step) {
        memory_.advance_time(step);

        // Inject background traffic for this step before any agent decision
        // so policy features and TAM cost queries see the up-to-date load.
        if (ghost_on) ghost_traffic_.step(step);

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

        // Build the global state once per step and publish it to the Manager:
        // bid_for_task snapshots it into every experience recorded this step,
        // which is what the MAPPO centralised critic trains against.
        auto cur_gs = build_global_state(step, total_steps, phase_label, lambda,
                                         city_norm, n_active).to_array();
        std::copy(cur_gs.begin(), cur_gs.end(), memory_.cur_global_state);

        // ── Retry deferred tasks (TAM multi-candidate Format B only) ─────────
        // retry_queue_ is empty in legacy mode and in Format A, so this loop
        // is a strict no-op there. A task is deferred when every candidate
        // scored < 0.5; it is re-offered after tam_mc_recall_time_frac of the
        // episode, or dropped as a genuine refusal past tam_mc_reject_time_frac.
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
                    // Past the cutoff — give up; counts as a genuine refusal.
                    ++n_refused_;
                    imp_refused_sum_ += rt->importance_original;
                    imp_refused_n_   += 1;
                    drop();
                    continue;
                }

                const float r_cong = memory_.congestion_map.mean_load_now();
                const bool  r_high = (r_cong >= 1.0f);
                congestion_at_decision_sum_   += r_cong;
                congestion_at_decision_count_ += 1;

                OfferResult rres = offer_task(rtask, rt->reward_original,
                                              rt->importance_original,
                                              n_active, cur_gs);
                tam_agents_offered_sum_    += last_offer_stats_.agents_offered;
                tam_recall_rounds_sum_     += last_offer_stats_.recall_rounds;
                tam_candidates_scored_sum_ += last_offer_stats_.candidates_scored;
                tam_offer_samples_         += 1;
                allocation_time_us_sum_    += last_offer_stats_.allocation_time_us;
                pure_alloc_time_us_sum_    += last_offer_stats_.pure_alloc_time_us;
                allocation_time_count_     += 1;
                if (last_offer_stats_.tam_dijkstra_steps > 0) {
                    tam_dijkstra_steps_sum_ += last_offer_stats_.tam_dijkstra_steps;
                    tam_dijkstra_count_     += 1;
                }
                // Oracle ratio: only meaningful if we actually allocated AND we
                // have pre-marginal costs for every agent at decision time.
                if (rres.agent_id >= 0
                    && static_cast<size_t>(rres.agent_id) < last_offer_stats_.pre_marginal_costs.size())
                {
                    const auto& pmc = last_offer_stats_.pre_marginal_costs;
                    const float chosen = pmc[rres.agent_id];
                    float oracle = std::numeric_limits<float>::max();
                    for (float c : pmc) if (c < oracle) oracle = c;
                    if (oracle > 0.f && std::isfinite(oracle) && std::isfinite(chosen)) {
                        marginal_ratio_sum_   += static_cast<double>(chosen) / oracle;
                        marginal_ratio_count_ += 1;
                    }
                    accumulate_rmca_regret(pmc);   // RMCA(r) eq.16 (no-op otherwise)
                }
                if (rres.deferred) {
                    retry_queue_[qi].retry_step = step + retry_interval;
                    ++qi;
                } else if (rres.agent_id >= 0) {
                    imp_accepted_sum_ += rt->importance_original;
                    imp_accepted_n_   += 1;
                    if (r_high) ++accepts_high_cong_; else ++accepts_low_cong_;
                    ++n_accepted_;
                    commit_accepted_task(rtask, rres.agent_id, rres.tam_owned,
                                         rt->pickup.id, step);
                    drop();
                } else {
                    ++n_refused_;
                    imp_refused_sum_ += rt->importance_original;
                    imp_refused_n_   += 1;
                    value_refused_sum_ += static_cast<double>(rt->reward_original)
                                        * rt->importance_original;
                    if (r_high) ++refuses_high_cong_; else ++refuses_low_cong_;
                    drop();
                }
            }
        }

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

            // Track pickup arrival in the spatial heatmap (RegionStatsGrid).
            // The grid powers PolicyFeatures::area_heat_pickup at scoring time.
            {
                auto it = memory_.geo_box.data.nodes.find(pickup.id);
                if (it != memory_.geo_box.data.nodes.end()) {
                    memory_.region_grid.register_task(
                        it->second.lat, it->second.lon, step);
                }
            }

            // Value tracking: this task contributes reward × importance to the
            // appeared-value pool. value_throughput_rate divides delivered by
            // appeared at episode end.
            value_appeared_sum_ += static_cast<double>(st.reward) * st.importance;

            // Snapshot congestion at the exact step the policy is consulted.
            // Distinct from per-step mean_congestion: this only counts the
            // network load when an accept/refuse call actually happens, which
            // is what the policy effectively conditioned on.
            const float cong_at_decision = memory_.congestion_map.mean_load_now();
            const bool  high_cong_now    = (cong_at_decision >= 1.0f);
            congestion_at_decision_sum_   += cong_at_decision;
            congestion_at_decision_count_ += 1;

            OfferResult res = offer_task(task_id, st.reward, st.importance,
                                         n_active, cur_gs);
            tam_agents_offered_sum_    += last_offer_stats_.agents_offered;
            tam_recall_rounds_sum_     += last_offer_stats_.recall_rounds;
            tam_candidates_scored_sum_ += last_offer_stats_.candidates_scored;
            tam_offer_samples_         += 1;
            allocation_time_us_sum_    += last_offer_stats_.allocation_time_us;
            allocation_time_count_     += 1;
            if (last_offer_stats_.tam_dijkstra_steps > 0) {
                tam_dijkstra_steps_sum_ += last_offer_stats_.tam_dijkstra_steps;
                tam_dijkstra_count_     += 1;
            }
            if (res.agent_id >= 0
                && static_cast<size_t>(res.agent_id) < last_offer_stats_.pre_marginal_costs.size())
            {
                const auto& pmc = last_offer_stats_.pre_marginal_costs;
                const float chosen = pmc[res.agent_id];
                float oracle = std::numeric_limits<float>::max();
                for (float c : pmc) if (c < oracle) oracle = c;
                if (oracle > 0.f && std::isfinite(oracle) && std::isfinite(chosen)) {
                    marginal_ratio_sum_   += static_cast<double>(chosen) / oracle;
                    marginal_ratio_count_ += 1;
                }
                accumulate_rmca_regret(pmc);   // RMCA(r) eq.16 (no-op otherwise)
            }
            if (res.deferred) {
                // TAM multi-candidate Format B — neither accepted nor refused
                // yet. Queue for a later retry; not counted in any metric now.
                const int retry_interval = std::max(1, static_cast<int>(
                    cfg_.tam_mc_recall_time_frac * static_cast<float>(total_steps)));
                retry_queue_.push_back({ task_id, step + retry_interval });
            } else if (res.agent_id >= 0) {
                // Selectivity + context diagnostics (Option X analysis).
                imp_accepted_sum_ += st.importance;
                imp_accepted_n_   += 1;
                if (high_cong_now) ++accepts_high_cong_;
                else               ++accepts_low_cong_;
                ++n_accepted_;
                commit_accepted_task(task_id, res.agent_id, res.tam_owned,
                                     st.pickup_node_id, step);
            } else {
                ++n_refused_;
                imp_refused_sum_ += st.importance;
                imp_refused_n_   += 1;
                value_refused_sum_ += static_cast<double>(st.reward) * st.importance;
                if (high_cong_now) ++refuses_high_cong_;
                else               ++refuses_low_cong_;
                // Task stays in available_tasks but won't be re-offered.
            }
        }

        // Sample network congestion once per step (mean edge load over all edges
        // with at least one planned agent passage). Averaged at episode end to
        // produce ComparisonMetrics::mean_congestion.
        const float load_now_step = memory_.congestion_map.mean_load_now();
        congestion_sum_   += load_now_step;
        congestion_steps_ += 1;

        // Network impact metrics (per-step samples).
        const int peak_step = memory_.congestion_map.peak_load_now();
        if (peak_step > peak_load_episode_) peak_load_episode_ = peak_step;
        overlap_edges_sum_ += memory_.congestion_map.n_edges_load_ge(2);
        overlap_steps_     += 1;
        load_now_sum_lin_  += static_cast<double>(load_now_step);
        load_now_sum_sq_   += static_cast<double>(load_now_step) * load_now_step;
        load_now_n_        += 1;

        // Route exposure: load_now on edges currently traversed by real agents.
        // Captures the congestion that the policy's chosen routes are actually
        // exposed to (distinct from network-wide average).
        for (const auto& a : all_agents_) {
            if (a->in_transit.has_value()) {
                const int load_here = memory_.congestion_map.get_load(
                    a->in_transit->edge_id, step);
                route_exposure_sum_ += static_cast<double>(load_here);
                route_exposure_n_   += 1;
            }
        }

        // Count active agents for utilisation metric.
        int n_now = 0;
        for (int i = 0; i < n_active; ++i)
            if (all_agents_[i]->status == AgentStatus::Active) ++n_now;
        active_sum_   += n_now;
        active_steps_ += 1;
    }

    // Flush any tasks still deferred at episode end (TAM multi-candidate
    // Format B). They were never resolved → count as refusals so that
    // tasks_appeared stays consistent (accepted + refused = appeared).
    for (const auto& re : retry_queue_) {
        PDPTask* rt = memory_.get_task(re.task_id);
        ++n_refused_;
        if (rt) {
            imp_refused_sum_ += rt->importance_original;
            imp_refused_n_   += 1;
            value_refused_sum_ += static_cast<double>(rt->reward_original)
                                * rt->importance_original;
        }
    }
    retry_queue_.clear();

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
    // Utilisation against the EFFECTIVE provisioned fleet (scenario-scaled),
    // matching Option OP's denominator. Was cfg_.max_agents() (nominal peak),
    // which ignored scenario.agents_mult → ~10× inflation in over_fleet.
    const int util_fleet = std::max(1, episode_fleet_size_);
    metrics.agent_utilisation = (active_steps_ > 0 && n_accepted_ > 0)
        ? static_cast<float>(active_sum_) / active_steps_ / util_fleet : 0.f;
    // Report the effective (scaled) fleet as n_agents — identical to the value
    // SolverRunner logs for the same (city, scenario, episode), restoring
    // cross-method fleet-size homogeneity in the CSV.
    metrics.n_agents_max = episode_fleet_size_;
    metrics.total_steps = total_steps;
    metrics.mean_congestion = (congestion_steps_ > 0)
        ? static_cast<float>(congestion_sum_ / congestion_steps_) : 0.f;
    metrics.mean_trip_steps = (trip_count_ > 0)
        ? static_cast<float>(trip_sum_) / trip_count_ : 0.f;
    metrics.mean_wait_steps = (wait_count_ > 0)
        ? static_cast<float>(wait_sum_) / wait_count_ : 0.f;
    metrics.mean_road_pd_m  = (road_pd_count_ > 0)
        ? static_cast<float>(road_pd_sum_ / road_pd_count_) : 0.f;

    // ── Selectivity diagnostics (new metrics for Option M analysis) ────────
    metrics.completion_per_accepted = (n_accepted_ > 0)
        ? static_cast<float>(latency_count_) / static_cast<float>(n_accepted_) : 0.f;
    metrics.unfinished_accept_rate  = (n_accepted_ > 0)
        ? 1.f - metrics.completion_per_accepted : 0.f;
    metrics.mean_congestion_at_decision = (congestion_at_decision_count_ > 0)
        ? static_cast<float>(congestion_at_decision_sum_ / congestion_at_decision_count_) : 0.f;
    metrics.n_ghost_active_mean         = ghost_on ? ghost_traffic_.mean_active() : 0.f;
    metrics.congestion_profile_label    = ghost_on
        ? std::string(congestion_profile_label(scenario.congestion_profile))
        : std::string();

    // TAM efficiency means (one sample per offer call). For TAM-driven modes
    // the agents_offered tells how many distinct deliveries the TAM contacted
    // — the smaller, the lower the communication overhead vs SoTA full-scan
    // baselines (which report n_active by construction).
    if (tam_offer_samples_ > 0) {
        metrics.mean_agents_offered_per_task = static_cast<float>(
            tam_agents_offered_sum_) / tam_offer_samples_;
        metrics.mean_recall_rounds_per_task = static_cast<float>(
            tam_recall_rounds_sum_)  / tam_offer_samples_;
        metrics.mean_candidates_scored_per_task = static_cast<float>(
            tam_candidates_scored_sum_) / tam_offer_samples_;
    }

    // ── Selection intelligence (delivery quality, not just count) ──────────
    metrics.value_throughput_rate = (value_appeared_sum_ > 0.0)
        ? static_cast<float>(value_delivered_sum_ / value_appeared_sum_) : 0.f;
    metrics.mean_completion_value = (latency_count_ > 0)
        ? static_cast<float>(value_delivered_sum_ / latency_count_) : 0.f;
    metrics.value_loss_to_refusal = static_cast<float>(value_refused_sum_);

    // ── Real impact on edge traversal (BPR factors actually paid) ─────────
    metrics.mean_bpr_along_route = (bpr_along_route_count_ > 0)
        ? static_cast<float>(bpr_along_route_sum_ / bpr_along_route_count_)
        : 1.f;
    metrics.time_lost_to_congestion_steps = static_cast<float>(
        time_lost_to_congestion_sum_);
    metrics.n_traversals_in_jam = n_traversals_in_jam_;

    // ── Allocation optimality vs MCA full-scan oracle ─────────────────────
    metrics.marginal_cost_ratio_vs_oracle = (marginal_ratio_count_ > 0)
        ? static_cast<float>(marginal_ratio_sum_ / marginal_ratio_count_)
        : 1.f;

    // ── RMCA(r) regret diagnostics (Chen+2021 eq.13/14/16; RMCA mode only) ─
    metrics.rmca_relative_regret  = (rmca_regret_count_ > 0)
        ? static_cast<float>(rmca_regret_sum_ / rmca_regret_count_) : 0.f;
    metrics.rmca_marginal_cost_k1 = (rmca_regret_count_ > 0)
        ? static_cast<float>(rmca_mc_k1_sum_ / rmca_regret_count_) : 0.f;
    metrics.rmca_marginal_cost_k2 = (rmca_regret_count_ > 0)
        ? static_cast<float>(rmca_mc_k2_sum_ / rmca_regret_count_) : 0.f;

    // ── Temporal complexity (allocation-only wallclock + TAM Dijkstra) ────
    metrics.mean_allocation_time_us = (allocation_time_count_ > 0)
        ? static_cast<float>(allocation_time_us_sum_) / allocation_time_count_
        : 0.f;
    metrics.mean_tam_dijkstra_steps = (tam_dijkstra_count_ > 0)
        ? static_cast<float>(tam_dijkstra_steps_sum_) / tam_dijkstra_count_
        : 0.f;

    // ── Path-compute breakdown (ms units) ──────────────────────────────────
    //   path_compute_time_ms    : total time spent in get_or_compute_path
    //                              across the whole episode
    //   mean_pure_alloc_time_ms : per-offer TAM+policy time (no path-cache)
    path_compute_time_us_ep_ = PDPServerMemory::path_compute_time_us();
    metrics.path_compute_time_ms = static_cast<float>(path_compute_time_us_ep_) / 1000.f;
    metrics.mean_pure_alloc_time_ms = (allocation_time_count_ > 0)
        ? (static_cast<float>(pure_alloc_time_us_sum_) / allocation_time_count_) / 1000.f
        : 0.f;

    // ── Multi-axis performance diagnostics (Option X) ──────────────────────
    metrics.mean_imp_accepted = (imp_accepted_n_ > 0)
        ? static_cast<float>(imp_accepted_sum_ / imp_accepted_n_) : 0.f;
    metrics.mean_imp_refused  = (imp_refused_n_ > 0)
        ? static_cast<float>(imp_refused_sum_  / imp_refused_n_)  : 0.f;
    {
        const int hi = accepts_high_cong_ + refuses_high_cong_;
        const int lo = accepts_low_cong_  + refuses_low_cong_;
        metrics.accept_rate_high_cong = (hi > 0)
            ? static_cast<float>(accepts_high_cong_) / hi : 0.f;
        metrics.accept_rate_low_cong  = (lo > 0)
            ? static_cast<float>(accepts_low_cong_)  / lo : 0.f;
    }

    // Per-agent load balance over delivered tasks. Counts include only
    // agents that received at least one allocation during the episode
    // (Gini / CV over the active fleet, not the full agent pool — the
    // pool oversizes by 1.5× for scenario.agents_mult > 1 head-room,
    // which would otherwise bias toward "very unequal").
    {
        std::vector<int> per_agent_completed(all_agents_.size(), 0);
        std::vector<int> per_agent_allocated(all_agents_.size(), 0);
        for (const PDPTask* t : memory_.finished_tasks)
            if (t && t->agent_id >= 0 && t->agent_id < (int)all_agents_.size())
                ++per_agent_completed[t->agent_id];
        for (const PDPTask* t : memory_.allocated_tasks)
            if (t && t->agent_id >= 0 && t->agent_id < (int)all_agents_.size())
                ++per_agent_allocated[t->agent_id];

        std::vector<int> active_completed;
        active_completed.reserve(per_agent_completed.size());
        for (size_t i = 0; i < per_agent_completed.size(); ++i)
            if (per_agent_completed[i] > 0 || per_agent_allocated[i] > 0)
                active_completed.push_back(per_agent_completed[i]);

        if (!active_completed.empty()) {
            std::sort(active_completed.begin(), active_completed.end());
            const size_t n = active_completed.size();
            double sum = 0.0;
            for (int v : active_completed) sum += v;
            const double mean = sum / static_cast<double>(n);

            // Gini (sorted ascending: sum_i (2*i - n + 1) * x_i) / (n * sum)
            double gnum = 0.0;
            for (size_t i = 0; i < n; ++i)
                gnum += (2.0 * static_cast<double>(i + 1) - static_cast<double>(n) - 1.0)
                        * static_cast<double>(active_completed[i]);
            metrics.agent_completed_gini = (sum > 0.0)
                ? static_cast<float>(gnum / (static_cast<double>(n) * sum))
                : 0.f;

            // Coefficient of variation (std / mean), normalised so cities
            // of different fleet sizes are comparable.
            double var = 0.0;
            for (int v : active_completed) {
                const double d = static_cast<double>(v) - mean;
                var += d * d;
            }
            var /= static_cast<double>(n);
            metrics.agent_completed_std = (mean > 0.0)
                ? static_cast<float>(std::sqrt(var) / mean) : 0.f;
        }
    }

    // Extra steps per delivered task = actual trip - ideal trip (in steps).
    // Magnitude version of delivery_route_efficiency (which is a ratio).
    {
        const float ideal_steps = (cfg_.speed_mps > 0.f && metrics.mean_road_pd_m > 0.f)
            ? metrics.mean_road_pd_m / cfg_.speed_mps : 0.f;
        metrics.mean_extra_steps_per_task = (metrics.mean_trip_steps > 0.f && ideal_steps > 0.f)
            ? std::max(0.f, metrics.mean_trip_steps - ideal_steps) : 0.f;
    }

    // ── Network-level congestion impact ─────────────────────────────────
    metrics.peak_congestion = peak_load_episode_;
    metrics.mean_overlap_edges = (overlap_steps_ > 0)
        ? static_cast<float>(overlap_edges_sum_) / overlap_steps_ : 0.f;
    if (load_now_n_ > 1) {
        const double m = load_now_sum_lin_ / load_now_n_;
        const double var = std::max(0.0, load_now_sum_sq_ / load_now_n_ - m * m);
        metrics.congestion_variance = static_cast<float>(std::sqrt(var));
    } else {
        metrics.congestion_variance = 0.f;
    }
    metrics.route_congestion_exposure = (route_exposure_n_ > 0)
        ? static_cast<float>(route_exposure_sum_ / route_exposure_n_) : 0.f;

    // ── Per-agent task distribution (raw min/max + total fleet distance) ──
    {
        int mx = 0, mn = std::numeric_limits<int>::max();
        bool any = false;
        std::vector<int> per_agent_completed_for_extrema(all_agents_.size(), 0);
        std::vector<int> per_agent_allocated_for_extrema(all_agents_.size(), 0);
        for (const PDPTask* t : memory_.finished_tasks)
            if (t && t->agent_id >= 0 && t->agent_id < (int)all_agents_.size())
                ++per_agent_completed_for_extrema[t->agent_id];
        for (const PDPTask* t : memory_.allocated_tasks)
            if (t && t->agent_id >= 0 && t->agent_id < (int)all_agents_.size())
                ++per_agent_allocated_for_extrema[t->agent_id];
        for (size_t i = 0; i < per_agent_completed_for_extrema.size(); ++i) {
            // Only count agents that participated (received at least one allocation).
            if (per_agent_completed_for_extrema[i] > 0 || per_agent_allocated_for_extrema[i] > 0) {
                if (per_agent_completed_for_extrema[i] > mx) mx = per_agent_completed_for_extrema[i];
                if (per_agent_completed_for_extrema[i] < mn) mn = per_agent_completed_for_extrema[i];
                any = true;
            }
        }
        metrics.max_agent_completed = any ? mx : 0;
        metrics.min_agent_completed = any ? mn : 0;
    }
    // Total fleet distance = sum of P→D road distances over completed tasks.
    // road_pd_sum_ is in metres (accumulated above for completed tasks).
    metrics.total_fleet_distance_m = static_cast<float>(road_pd_sum_);

    // Clean up ghost loads now that the episode is done so the next run()
    // starts from a clean CongestionMap (memory_.reset_episode() also clears
    // it, but purging here keeps ghost_traffic_ internal state consistent).
    ghost_traffic_.purge();
    // Delivery route efficiency: ideal travel distance / actual travel distance.
    // Ideal = A* shortest path at full speed (road_pd_m / speed_mps steps).
    // Actual = mean_trip_steps. Ratio ∈ (0,1]: 1=optimal, <1=detour overhead.
    {
        const float ideal_steps = (cfg_.speed_mps > 0.f && metrics.mean_road_pd_m > 0.f)
            ? metrics.mean_road_pd_m / cfg_.speed_mps : 0.f;
        metrics.delivery_route_efficiency = (metrics.mean_trip_steps > 0.f && ideal_steps > 0.f)
            ? std::min(1.f, ideal_steps / metrics.mean_trip_steps) : 0.f;
    }

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
            // Use the agent's OWN capacity, not the global TAM ceiling. With a
            // heterogeneous fleet (max_capacity ∈ [min,max]) the global ceiling
            // equals hetero_capacity_max, so a cap=3 agent carrying 5 would be a
            // real violation (5>3) yet slip past a 5>7 test. Fall back to the
            // global ceiling only for homogeneous fleets (max_capacity == 0).
            const DeliveryAgent* a = memory_.get_delivery_agent(aid);
            const int agent_cap = (a && a->max_capacity > 0) ? a->max_capacity
                                                             : max_carry;
            if (peak > agent_cap)
                metrics.capacity_violations_runtime += peak - agent_cap;
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
    // Hybrid receives the unfinished penalty regardless of train_mode because
    // its online update runs at every episode end (continual adaptation).
    const bool apply_unfinished_penalty = active_policy_
        && (train_mode || active_policy_->trains_online());
    if (apply_unfinished_penalty && cfg_.unfinished_factor > 0.f) {
        for (const auto& [tid, buf_idx] : task_accept_buf_idx_) {
            PDPTask* t = memory_.get_task(tid);
            if (!t) continue;
            const float val    = t->reward_original * t->importance_original;
            const bool  picked = (t->timeline.picked_step >= 0);
            const float lost_frac = picked ? (1.f - cfg_.pickup_reward_frac) : 1.f;
            active_policy_->add_to_reward(t->agent_id, buf_idx,
                                          -cfg_.unfinished_factor * lost_frac * val);
        }
    }

    // ── Idle-agent penalty (encourages active bidding) ────────────────────
    //
    // Sweep every delivery agent: if the agent finished the episode IDLE
    // (status==Idle AND no task ever assigned), every buffer entry they
    // produced this episode (offers they lost) is debited by idle_penalty_w
    // × importance. The signal trains the policy to bid HIGHER when its
    // alternative is sitting idle. Only fires for IPPO/MAPPER/FaithfulMAPPER
    // (per-agent buffers carry agent_id, so we know which entries belong to
    // whom). MAPPO's shared buffer is not partitioned per agent → would
    // require an extra tracker; skipped for now (the shared critic still
    // learns global idle-rate signals via the global state).
    // Idle-agent penalty: every offer entry produced by an agent that ended
    // the episode idle is debited — "sitting idle is wasted capacity". Now
    // uniform across all learning policies (per-agent buffers everywhere).
    const bool apply_idle_penalty = active_policy_
        && (train_mode || active_policy_->trains_online())
        && cfg_.idle_penalty_w > 0.f;
    if (apply_idle_penalty) {
        for (const auto& a : all_agents_) {
            if (!a) continue;
            if (a->status != AgentStatus::Idle || !a->local_memory.tasks.empty())
                continue;
            const int sz = active_policy_->buffer_size(a->agent_id);
            for (int i = 0; i < sz; ++i)
                active_policy_->add_to_reward(a->agent_id, i, -cfg_.idle_penalty_w);
        }
    }

    // Per-episode learning update. Offline policies (MAPPO/IPPO/MAPPER) run
    // their PPO update only when train_mode=true; Hybrid adapts its residual
    // at the end of EVERY episode (trains_online).
    if (active_policy_ && active_policy_->total_buffer_size() > 0
        && (train_mode || active_policy_->trains_online())) {
        result.train_stats = active_policy_->train_round();
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

// ── RMCA(r) relative regret [Chen et al. 2021, eq.13/14/16] ──────────────────
//
// From the pre-allocation marginal costs of every eligible agent (= R in the
// paper), read mc(k1) = cheapest and mc(k2) = 2nd-cheapest insertion cost, and
// accumulate the relative regret mc(k2)/mc(k1). Only active for RMCA. mc(k1) is
// floored at 1 m to avoid division blow-ups on near-coincident pickups, and the
// ratio is clamped to [1, 50] to keep the episode mean robust to such outliers.
// A single eligible candidate ⇒ no second-best ⇒ regret = 1 (no competition).
void EpisodeRunner::accumulate_rmca_regret(
    const std::vector<float>& pre_marginal_costs)
{
    if (policy_mode != PolicyMode::RMCA) return;

    // compute_marginal_cost returns FLT_MAX for capacity-infeasible agents
    // (eq.9 violated); treat anything above this threshold as "no insertion".
    constexpr float kInfeasible = 1e30f;
    float mc1 = std::numeric_limits<float>::max();   // cheapest        (k1)
    float mc2 = std::numeric_limits<float>::max();   // second cheapest (k2)
    for (float c : pre_marginal_costs) {
        if (!std::isfinite(c) || c < 0.f || c > kInfeasible) continue;
        if      (c < mc1) { mc2 = mc1; mc1 = c; }
        else if (c < mc2) { mc2 = c; }
    }
    if (mc1 > kInfeasible) return;                   // no feasible agent at all

    const bool  has_k2 = (mc2 <= kInfeasible);       // a genuine 2nd-best exists
    const float mc1f   = std::max(mc1, 1.0f);        // floor at 1 m (avoid /0)
    const float mc2f   = has_k2 ? mc2 : mc1;         // single candidate ⇒ k2 := k1
    const float regret = std::clamp(mc2f / mc1f, 1.0f, 50.0f);

    rmca_mc_k1_sum_    += static_cast<double>(mc1);
    rmca_mc_k2_sum_    += static_cast<double>(mc2f);
    rmca_regret_sum_   += static_cast<double>(regret);
    rmca_regret_count_ += 1;
}

// ── Task offer ────────────────────────────────────────────────────────────────

EpisodeRunner::OfferResult EpisodeRunner::offer_task(
    int task_id, float reward, float importance,
    int n_active, const std::array<float, kGlobSz>& gs)
{
    // Make the current time ratio visible to try_accept_task via GlobalMemory.
    // gs[0] = GlobalState::time_ratio = step / total_steps.
    memory_.cur_time_ratio = gs[0];

    // Default per-offer stats — assume a "full-scan" baseline (n_active agents
    // consulted, no recalls, no MC). The TAM branch overrides these from the
    // TAM getters before returning, so the values seen at the call site
    // correctly reflect either branch's actual behaviour.
    last_offer_stats_ = LastOfferStats{};
    last_offer_stats_.agents_offered = n_active;
    last_offer_stats_.pre_marginal_costs.clear();

    // Wallclock timing for the entire offer_task call (TAM Dijkstra + every
    // policy score). Read at function exit by recording the value via a small
    // RAII guard that updates last_offer_stats_.allocation_time_us regardless
    // of which return path is taken. The guard also snapshots the global
    // path-compute counter to compute, in addition:
    //   path_time_us       — time spent inside get_or_compute_path during this offer
    //   pure_alloc_time_us — TAM + policy time only (= allocation_time - path_time)
    const auto offer_t0 = std::chrono::steady_clock::now();
    const long long path_us_before = PDPServerMemory::path_compute_time_us();
    struct TimerGuard {
        std::chrono::steady_clock::time_point t0;
        long long path_us_at_start;
        LastOfferStats& stats;
        ~TimerGuard() {
            const auto t1 = std::chrono::steady_clock::now();
            const long long total_us = std::chrono::duration_cast<
                std::chrono::microseconds>(t1 - t0).count();
            const long long path_us = PDPServerMemory::path_compute_time_us() - path_us_at_start;
            stats.allocation_time_us  = total_us;
            stats.path_time_us        = path_us;
            stats.pure_alloc_time_us  = (total_us > path_us) ? (total_us - path_us) : 0;
        }
    } _timer_guard{ offer_t0, path_us_before, last_offer_stats_ };

    // Pre-compute marginal cost of inserting THIS task into every eligible
    // agent's current sequence — BEFORE any allocation mutates state. Used
    // at the end of the TAM branch to compute marginal_cost_ratio_vs_oracle:
    //   chosen_agent_cost / min_over_all_agents_cost.
    // This is the "oracle" comparison: would MCA full-scan have picked a
    // strictly better agent than the TAM's localised search? 1.0 = TAM
    // matched MCA's choice; > 1.0 = TAM is suboptimal by that ratio.
    // We do this for ALL modes (baselines also report it — they minimise
    // a different criterion, so their ratio vs route-length oracle is
    // typically > 1.0 too, which is itself publishable: it shows how each
    // baseline trades off route-length for its own objective).
    PDPTask* task_for_oracle = memory_.get_task(task_id);
    if (task_for_oracle) {
        last_offer_stats_.pre_marginal_costs.reserve(n_active);
        for (int i = 0; i < n_active; ++i) {
            last_offer_stats_.pre_marginal_costs.push_back(
                compute_marginal_cost(*all_agents_[i], *task_for_oracle));
        }
    }

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

    // ── PIBT-inspired baseline: load-balanced priority allocation ────────────
    //
    // References for the original algorithm:
    //   - Okumura, Machida, Defago, Tamura (2022), "Priority Inheritance with
    //     Backtracking for Iterative Multi-Agent Path Finding", Artificial
    //     Intelligence 310. Defines PIBT as a one-step path-finding rule on a
    //     grid: in each timestep each agent locally picks its next vertex by
    //     priority order with backtracking when a chain of pushes fails.
    //   - Matsui (2025), "Improvement of PIBT-based Solution Method for
    //     Lifelong MAPD Problems to Extend Applicable Graphs", ICAART. Extends
    //     PIBT to graphs with single dead-end aisles via cooperative swap
    //     tasks (Section 3.2).
    //
    // LGPDP adaptation (this branch):
    //   Neither the Okumura+2022 path-finding rule nor the Matsui+2025 swap
    //   extension translate directly to our setting — OSM road networks have
    //   no grid vertex conflicts, no biconnected-graph structure, no dead-end
    //   aisle semantics, and our agents perform continuous routing rather
    //   than one-step moves. What we keep is the *priority-coordination
    //   intent*: when multiple eligible agents could serve a task, prefer
    //   the one whose action would be most "PIBT-conformant" — i.e. the
    //   agent with the LIGHTEST current queue (= the one that would have
    //   the highest PIBT priority for movement in the original algorithm,
    //   since elapsed-time-since-goal is highest for the least-busy agent).
    //
    //   The resulting rule is: argmin over eligible agents of |queued tasks|.
    //   This is structurally a load-balancing tie-breaker on top of capacity
    //   feasibility, and matches the *priority-by-idleness* spirit of PIBT
    //   without trying to reimplement the grid-collision machinery.
    //
    // OWNERSHIP NOTE: this branch was authored by another project
    // contributor as part of the SOTA-baseline porting effort. Treat it as
    // read-only — coordinate with the original author before modifying the
    // selection rule, especially any change that would alter the priority
    // semantics (load → elapsed time, etc.).
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

    // ── Token Passing (TP) [Ma+2017, AAMAS — Paper 2] ─────────────────────────
    //
    // Paper TP (Algorithm 1, lines 7-12): a FREE agent (a_i with no current
    // task) holding the token picks τ* ∈ T' = { τ | no other path in token ends
    // in s_τ or g_τ } that minimises h(loc(a_i), s_τ). Token then passes to
    // next free agent.
    //
    // LGPDP adaptation (capacitated multi-task per agent, no depot, OSM road
    // network with online single-task arrivals):
    //
    //   ✓ Selection rule = argmin h(loc(a_i), s_τ) — paper Line 9, EXACT.
    //   ✓ Static h cost (NOT congestion-adjusted) — paper uses graph-distance
    //     heuristic; faithful to the deterministic-cost model.
    //   ✓ "Free agent first" preserved via two-stage preference:
    //       (a) any agent with load==0 → argmin restricted to free agents
    //           (matches paper TP exactly when free agents exist).
    //       (b) no free agent → fallback to argmin over ALL eligible agents.
    //           Paper TP would defer the task, but in our online task-stream
    //           setting deferring while the fleet is busy collapses throughput;
    //           the fallback preserves TP's "closest free agent" intuition while
    //           keeping the baseline comparable in saturated regimes.
    //
    //   ✗ Endpoint filter T' (paper Line 7): vacuous in our setting — each
    //     task has UNIQUE pickup/delivery nodes by EpisodeGenerator
    //     construction, so the filter never excludes anything.
    //   ✗ Path1/Path2 collision-free path planning (paper Lines 12,16):
    //     NOT APPLICABLE — OSM road network has no vertex-blocking semantics
    //     (multiple agents can occupy the same node/edge; congestion is
    //     metered via cost adjustment, not hard blocking).
    //   ✗ TPTS task-swap (paper Algorithm 2): DEGENERATE in our setting.
    //     TPTS lets an agent a_i with the token REASSIGN a task τ from another
    //     agent a_i' if a_i can reach s_τ faster than a_i'. The swap acts on
    //     the SET of tasks T held in the token. In our online lifelong setting:
    //         (i)  T contains at most ONE task at a time (each task is
    //              allocated synchronously on arrival),
    //         (ii) once allocated, a task moves into the agent's local queue
    //              and is not re-offered.
    //     With |T| = 1 at every token-passing moment, the TPTS swap collapses
    //     to "pick the best agent for the new task", which is exactly TP. A
    //     batch variant could re-examine in-flight (not-yet-picked-up) tasks
    //     and swap them; we leave this as documented future work to keep TP
    //     and TPTS comparable to the paper's batch semantics.
    //
    // Distinguishing TP from MCA / CongestionAware / TrafficFlow:
    //   TP              — argmin h(loc, pickup) on free agents (static cost).
    //   MCA             — argmin Δroute_length over admissible (q1,q2) insertions.
    //   CongestionAware — argmin dynamic c_pu (pickup leg only, time-dependent).
    //   TrafficFlow     — argmin dynamic (c_pu + c_del) (full trip, time-dep.).
    if (policy_mode == PolicyMode::TokenPassing)
        return { tp_allocate(memory_, all_agents_, n_active, task_id), false };

    // ── DbVNS-PDP Lifelong Replanning ─────────────────────────────────────────
    // Allocation: same as MCA (min marginal insertion cost across all agents).
    // Planning: handled by DeliveryAgent::receive_task() when planning_use_dbvns.
    if (policy_mode == PolicyMode::DbVNS) {
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

    // ── ALNS Lifelong Replanning ─────────────────────────────────────────────
    // Allocation: identical to MCA / DbVNS (min marginal insertion cost). The
    // planning side switches to ALNS via the planning_use_alns flag set in run().
    if (policy_mode == PolicyMode::ALNS) {
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

    // ── MCA: Marginal-Cost Assignment [Chen+2021, ICRA — Paper 1] ─────────────
    //
    // NOTE ON NAMING — MCA here stands for "Marginal-Cost Assignment", a
    // VRP/PDP-style INSERTION HEURISTIC. It has NO relation to congestion.
    // Do not confuse with any "Multi-Congestion-Aware" / "M.C.A." variant from
    // congestion-aware MAPP literature — this is the Chen+2021 ICRA paper that
    // proposes MCA + RMCA as marginal-cost vs regret-based assignment rules.
    //
    // In this project MCA is used as a METAHEURISTIC ALLOCATION BASELINE in the
    // VRP/PDP comparison cluster (alongside DbVNS / ALNS / Double-Horizon),
    // NOT in the LGPDP-SOTA cluster (which contains TokenPassing,
    // TrafficFlow, CongestionAware).
    //
    // Paper MCA: eq. (12) — for each unassigned task i and each robot k, find
    // (q1*, q2*) that minimise the route-cost delta of inserting (s_i at q1,
    // g_i at q2) into k's current sequence o_k while respecting capacity. Pick
    // the (k*, i*, q1*, q2*) that globally minimises this delta.
    //
    // Faithfulness in our lifelong GPDP context:
    //   ✓ Per-(agent, task) marginal cost = compute_marginal_cost() (see
    //     definition ~line 1572): exact (pP, pD) admissible search with
    //     capacity profile (peak ≤ max_carry), preserving the in-flight head
    //     seq[0]. Matches the paper's insert() routine (Section IV-B).
    //   ✓ Global argmin across agents: this branch's outer loop.
    //   ✓ Capacitated multi-task agents: handled by load_after[] profile.
    //   = MCA == RMCA(r) for the SELECTION rule in our setting: tasks arrive
    //     one-at-a-time (online lifelong), so there is no batch ordering for
    //     the regret-based variant (paper eq. 16) to exploit. Per-task winner
    //     is the same agent under MCA and RMCA(r) in the |P^u| = 1 regime.
    //   ✓ Anytime LNS improvement (Algorithm 3, paper Section IV-D): NOW
    //     implemented inside DeliveryAgent::receive_task, gated by
    //     memory_.planning_use_mca_lns (set above on the MCA branch). After
    //     the cheapest-insertion commits the new task, a short LNS loop
    //     (30 iters × destroy-3 random unpicked tasks + greedy cheapest
    //     re-insertion) refines the agent's local sequence. Already-picked
    //     tasks are never displaced (paper-faithful — lifelong constraint).
    //     Closes the audit's "MCA missing LNS" gap.
    //   ✗ Prioritised path planning (Section IV-A): we substitute DbVNS-PDP
    //     replan inside DeliveryAgent::receive_task. This means we measure
    //     route-distance delta, not collision-free TTD — but in OSM continuous
    //     routing there are no grid-MAPF style vertex collisions to resolve.
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

    // RL / RMCA / TamAlwaysAccept: drive the TAM (see TaskAllocationModule).
    // Each bid records an experience; the (agent_id, buf_idx) pairs between
    // rec_before and after pair candidates to their buffer entries.
    const int rec_before = active_policy_ ? active_policy_->n_recent_records() : 0;

    memory_.task_agent.on_new_task(task_id, memory_);
    auto* tam = memory_.task_agent.get_tam(task_id);
    if (!tam) return {-1, true};

    // The TAM terminates via its own conditions (allocated / exhausted / deferred).
    // No external step cap — the adaptive budget x*ratio(x) and the natural
    // Dijkstra exhaustion are the sole termination guarantees.
    int tam_iter_count = 0;
    while (!tam->is_allocated() && !tam->is_exhausted()) {
        tam->step(memory_, cfg_.speed_mps);
        ++tam_iter_count;
    }
    last_offer_stats_.tam_dijkstra_steps = tam_iter_count;

    PDPTask* t = memory_.get_task(task_id);
    const bool allocated = tam->is_allocated() && t && t->agent_id >= 0;

    const bool shaping_active =
        (train_mode || (policy_mode == PolicyMode::Hybrid));

    // ── Per-candidate buffer-index resolver ─────────────────────────────────
    // The i-th scored candidate produced the (rec_before + i)-th entry of the
    // active policy's recent-records log → (agent_id, buf_idx).
    auto resolve_buf_idx = [&](int i_scored) -> int {
        if (!active_policy_) return -1;
        const int k = rec_before + i_scored;
        if (k >= active_policy_->n_recent_records()) return -1;
        return active_policy_->recent_record(k).second;
    };

    auto add_reward_to = [&](int aid, int buf_idx, float delta) {
        if (active_policy_) active_policy_->add_to_reward(aid, buf_idx, delta);
    };

    {
        // ── Multi-candidate TAM outcome → reward plumbing ───────────────────
        // The TAM scored K candidates in the order of candidates_in_order().
        // Apply:
        //   - winner (only when it actually BID): bind its buffer entry to the
        //     task outcome (pickup/delivery credit fires later) + congestion-
        //     creation penalty. A FORCED winner (nobody bid, force_assign
        //     override) gets NO binding: crediting a delivery to a recorded
        //     "refuse" action would teach the policy that refusing pays.
        //   - non_affected_penalty to losing BIDDERS only ("your bid lost").
        //     Candidates that refused and lost made a consistent choice — no
        //     penalty on them.
        const auto& cands  = tam->candidates_in_order();
        const auto& bids   = tam->bids_in_order();
        const bool  forced = tam->was_forced();
        if (!cands.empty() && t) {
            // Locate winner position in scoring order.
            int win_pos = -1;
            if (allocated) {
                for (int i = 0; i < static_cast<int>(cands.size()); ++i) {
                    if (cands[i] == t->agent_id) { win_pos = i; break; }
                }
            }
            if (forced) win_pos = -1;   // no outcome binding for forced winners

            // Capture winner's buf_idx for downstream pickup/delivery credit.
            // Also apply the congestion-creation penalty here: the winner's
            // congestion_delta_contribution feature is already encoded in their
            // buffered observation, so this closes the loop between input
            // vector and reward (Axis B). Negative reward scales with how
            // much congestion the agent's plan will add to the network — uses
            // the same proxy the feature uses (network mean BPR × insertion
            // distance). Ghost-only jams contribute too, which biases the
            // agent to avoid jammed areas regardless of cause; this is the
            // simplest sane "system-aware" signal.
            if (win_pos >= 0) {
                const int win_buf_idx = resolve_buf_idx(win_pos);
                if (win_buf_idx >= 0) {
                    task_accept_buf_idx_[task_id] = win_buf_idx;

                    if (shaping_active && cfg_.congestion_creation_w > 0.f) {
                        // Recompute the proxy here so the reward matches what
                        // the policy saw in its feature vector.
                        DeliveryAgent* wa = memory_.get_delivery_agent(t->agent_id);
                        if (wa) {
                            const float to_pu_cost = [&]{
                                const auto* p = memory_.get_or_compute_path(
                                    wa->current_node, t->pickup.id, t->pickup.group_id);
                                return (p && p->valid()) ? p->cost : kCostScale;
                            }();
                            const float pu_del_cost = [&]{
                                const auto* p = memory_.get_or_compute_path(
                                    t->pickup.id, t->delivery.id, t->delivery.group_id);
                                return (p && p->valid()) ? p->cost : kCostScale;
                            }();
                            const float insertion_cost = to_pu_cost + pu_del_cost;
                            const float mean_load = memory_.congestion_map.mean_load_now();
                            const float ratio     = mean_load / 5.f;
                            const float r2        = ratio * ratio;
                            const float bpr_proxy = 1.f + 0.15f * r2 * r2;
                            const float norm_ins  = std::clamp(
                                insertion_cost / kCostScale, 0.f, 1.f);
                            const float delta     = std::clamp(
                                (bpr_proxy - 1.f) * norm_ins, 0.f, 1.f);
                            const float cong_pen  =
                                -cfg_.congestion_creation_w * delta
                                * t->importance_original;
                            add_reward_to(t->agent_id, win_buf_idx, cong_pen);
                        }
                    }
                }
            }

            // Non-affected penalty on every losing BIDDER.
            if (shaping_active && cfg_.non_affected_penalty_w > 0.f) {
                const float pen = -cfg_.non_affected_penalty_w * t->importance_original;
                for (int i = 0; i < static_cast<int>(cands.size()); ++i) {
                    if (i == win_pos) continue;
                    if (i >= static_cast<int>(bids.size()) || !bids[i]) continue;
                    const int aid = cands[i];
                    const int idx = resolve_buf_idx(i);
                    if (idx >= 0) add_reward_to(aid, idx, pen);
                }
            }
        }
    }

    const bool deferred = tam->is_deferred();

    // Capture TAM efficiency stats BEFORE erase_tam destroys the TAM state.
    // For the "paper minimise comm" claim: n_agents_offered is the count of
    // distinct agents the TAM actually contacted (offer_to_agent), which
    // is typically much smaller than n_active for the SoTA full-scan path.
    last_offer_stats_.agents_offered    = tam->n_agents_offered();
    last_offer_stats_.recall_rounds     = tam->n_recall_rounds();
    last_offer_stats_.candidates_scored = tam->n_candidates_scored();

    memory_.task_agent.erase_tam(task_id);

    return { allocated ? t->agent_id : -1, true, deferred };
}

// ── Post-allocation bookkeeping (shared by arrival loop + retry loop) ─────────
void EpisodeRunner::commit_accepted_task(
    int task_id, int winner, bool tam_owned,
    osmium::object_id_type pickup_node, int step)
{
    DeliveryAgent* agent = memory_.get_delivery_agent(winner);
    if (!agent) return;
    PDPTask* task = memory_.get_task(task_id);

    bool was_idle;
    if (tam_owned) {
        // TAM already did assign + receive + commit. After receive_task the
        // agent's task list contains this task plus whatever it had before.
        // Sole task ⇒ was Idle.
        was_idle = agent->local_memory.tasks.size() == 1;
    } else {
        was_idle = (agent->status == AgentStatus::Idle);
        if (task) {
            memory_.assign_task(task_id, winner);
            agent->receive_task(*task, memory_);
            memory_.commit_plan(winner, cfg_.speed_mps);
        }
    }

    // (The winner's buffer entry was already bound to the task in offer_task,
    // via the TAM's candidates/bids pairing — nothing to record here.)

    // Start the edge-by-edge leg only if the agent was Idle. If already Active
    // (multi-task queue) the current cursor is still valid; the new objectives
    // are picked up after the current delivery completes.
    if (was_idle)
        start_leg(winner, task_id, true, agent->current_node, pickup_node, step);
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

    // Exclude own committed weight: the agent sees n others, not n+1.
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

    // ── Real-impact accumulators: BPR along route + time lost + jam hits ──
    // Sampled at the moment this agent ENTERS the edge. Now that eff_dist
    // drives `steps`, this slowdown is the delay the agent ACTUALLY pays
    // (realized travel time), not just a measured externality.
    if (edge_id != 0 && dist > 0.f) {
        const float bpr_factor = eff_dist / dist;   // 1.0 = no slowdown
        bpr_along_route_sum_         += bpr_factor;
        bpr_along_route_count_       += 1;
        // Extra distance in metres -> divide by speed for extra steps.
        const float extra_dist = std::max(0.f, eff_dist - dist);
        time_lost_to_congestion_sum_ += extra_dist / std::max(0.1f, cfg_.speed_mps);
        // Edge with load >= 5 at the moment of entry = real chokepoint.
        const int load_now = memory_.congestion_map.get_load(edge_id, current_step);
        if (load_now >= 5) ++n_traversals_in_jam_;
    }

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

        // Response delay: steps from task appearance to agent arriving at pickup.
        if (task->timeline.created_step >= 0)
            wait_sum_   += current_step - task->timeline.created_step,
            wait_count_ += 1;

        // Reward shaping: deliver partial credit at pickup so the decision-time
        // experience gets a signal long before the actual delivery (which can be
        // 500-1500 steps later). Reduces credit-assignment delay drastically.
        if (active_policy_) {
            auto it = task_accept_buf_idx_.find(task_id);
            if (it != task_accept_buf_idx_.end()) {
                const float pickup_credit = cfg_.pickup_reward_frac
                    * task->reward_original * task->importance_original;
                active_policy_->add_to_reward(task->agent_id, it->second,
                                              pickup_credit);
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

        memory_.commit_plan(agent_id, cfg_.speed_mps);   // before schedule (self-exclusion)
        schedule_next_edge(agent_id, current_step);

    } else {
        // Invariant guard: a task is delivered exactly once, so throughput
        // (latency_count_ / appeared) can never exceed 1. The ROOT cause of the
        // rare double-count (next-leg identity mislabeled via the ambiguous
        // node→task map) is fixed above; this guard additionally protects every
        // delivery-side metric should any other path ever re-reach a delivered
        // task's node.
        const bool first_delivery = (task->timeline.delivered_step < 0);
        task->mark_delivered(current_step);

        int latency = current_step - task->timeline.created_step;
        if (first_delivery) {
        latency_sum_   += latency;
        latency_count_ += 1;

        // Value tracking: this delivered task contributes its value to the
        // delivered-value pool. value_throughput_rate and mean_completion_value
        // are computed from this sum at episode end.
        value_delivered_sum_ += static_cast<double>(task->reward_original)
                              * task->importance_original;

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
            road_pd_sum_      += static_cast<double>(dist_m);
            road_pd_count_    += 1;
        }
        }   // ── end "first delivery only" metric guard (throughput ≤ 1) ──

        // Reward shaping: add the remaining delivery credit on top of the
        // partial pickup credit already given. Total accumulated reward equals
        // task->reward_original × importance_original (i.e., 1.0 × imp by default).
        //
        // Use *_original (immutable) so "refuse-first-accept-on-recall" cannot
        // game a larger reward than immediate acceptance — TAM may boost
        // reward/importance to attract bidders, but the policy must be trained
        // on the true task value.
        if (active_policy_) {
            auto it = task_accept_buf_idx_.find(task_id);
            if (it != task_accept_buf_idx_.end()) {
                // Latency-aware delivery shaping (paper eq. 6):
                // factor = 1 − λ × min(1, trip_steps / max_steps).
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
                    * shape_factor;
                active_policy_->add_to_reward(task->agent_id, it->second,
                                              delivery_credit);
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

            // Resolve the next leg's task identity from the agent's OWN pending
            // tasks, disambiguated by pickup/delivery STATE — NOT from the global
            // node→task map (node_to_task_id_). That map keeps only ONE task per
            // node, so when two tasks share a node (e.g. one task's pickup node
            // coincides with another's delivery node) it returns the wrong task and
            // flips is_pickup, which makes a pickup leg get counted as a delivery
            // (and inflates throughput above 1). The agent's own task set is
            // unambiguous: a not-yet-picked task at this node ⇒ pickup leg; an
            // already-picked, not-yet-delivered task at this node ⇒ delivery leg.
            int  next_task_id = -1;
            bool next_pickup  = false;
            for (const PDPTask* t : agent->local_memory.tasks) {
                if (!t) continue;
                if (t->timeline.picked_step < 0 && t->pickup.id == next_obj.id) {
                    next_task_id = t->task_id; next_pickup = true;  break;
                }
                if (t->timeline.picked_step >= 0 && t->timeline.delivered_step < 0
                    && t->delivery.id == next_obj.id) {
                    next_task_id = t->task_id; next_pickup = false; break;
                }
            }
            if (next_task_id < 0) {
                // Defensive fallback (objective not found among the agent's tasks):
                // keep the legacy global lookup so the agent never stalls.
                PDPTask* nt = memory_.get_task_for_node(next_obj.id);
                next_task_id = nt ? nt->task_id : -1;
                next_pickup  = nt && (nt->pickup.id == next_obj.id);
            }
            agent->begin_leg(next_path, next_task_id, next_pickup);
            agent->prefetch_next_path(memory_);
            memory_.commit_plan(agent_id, cfg_.speed_mps);
            schedule_next_edge(agent_id, current_step);
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
    // Use the AGENT-effective carrying capacity, not the TAM-global ceiling.
    // With heterogeneous fleets (enable_heterogeneous_capacity) the TAM ceiling
    // is hetero_capacity_max — using it here would let MCA pick an insertion
    // the agent cannot honour at receive_task() time, causing capacity
    // overshoot via the fallback (append P, append D) branch.
    const int   tam_cap = memory_.task_agent.params.max_tasks_per_agent;
    const int   max_carry = std::max(
        1, a.max_capacity > 0 ? a.max_capacity : tam_cap);

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
    // Agent-effective capacity (see compute_marginal_cost for rationale).
    const int   tam_cap = memory_.task_agent.params.max_tasks_per_agent;
    const int   max_carry = std::max(
        1, a.max_capacity > 0 ? a.max_capacity : tam_cap);

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
