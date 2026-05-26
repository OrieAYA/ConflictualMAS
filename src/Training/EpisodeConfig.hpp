#ifndef TRAINING_EPISODE_CONFIG_HPP
#define TRAINING_EPISODE_CONFIG_HPP

#include "CityConfig.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"   // kGlobSz = 20
#include <osmium/osm/types.hpp>
#include <array>
#include <string>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// Density phase — one segment of the episode timeline
// ════════════════════════════════════════════════════════════════════════════
//
// Task arrival rate is expressed as tasks_per_agent (Amazon-scale reference:
// ~100–250 PDP tasks per driver per phase). Lambda is computed automatically:
//   lambda = tasks_per_agent * avg(n_agents_start, n_agents_end) / steps
// The generator uses a Poisson distribution so lambda > 1.0 is legal.
//
// n_agents is linearly interpolated from n_agents_start to n_agents_end
// within the phase, enabling smooth fleet-size ramps during the episode.
//
// n_hot_zones overrides the global EpisodeConfig.n_hot_zones for this phase.
// Set to -1 to inherit the global value.
struct DensityPhase {
    int   steps;              // duration in simulation steps
    float tasks_per_agent;    // expected tasks per agent in this phase
    int   n_agents_start;     // fleet size at phase start
    int   n_agents_end;       // fleet size at phase end (linear ramp)
    float label;              // [0,1] fed to GlobalState.density_phase
    int   n_hot_zones = -1;   // hot zones this phase; -1 = use global
};

// ════════════════════════════════════════════════════════════════════════════
// Episode configuration
// ════════════════════════════════════════════════════════════════════════════
struct EpisodeConfig {
    const CityConfig* city = nullptr;

    // Agent speed (m/step).  ~18 km/h = 5 m/s at 1 step/s.
    float speed_mps = 5.0f;

    // Phase schedule (executed in order).
    // Default: Amazon-scale three-phase curriculum over a pseudo-day (3600 steps).
    //   Low    (1000 steps):  8→10 agents, ~100 tasks/agent, 4 hot zones
    //   Medium (1500 steps): 10→13 agents, ~200 tasks/agent, 6 hot zones
    //   High   (1100 steps): 13→15 agents, ~250 tasks/agent, 8 hot zones
    // Expected lambda: low ≈ 0.90 | medium ≈ 1.53 | high ≈ 3.18 tasks/step.
    std::vector<DensityPhase> phases = {
        { 1000, 100.f,  8, 10, 0.0f,  4 },
        { 1500, 200.f, 10, 13, 0.5f,  6 },
        { 1100, 250.f, 13, 15, 1.0f,  8 },
    };

    // ── Spatial task pattern ───────────────────────────────────────────────
    float cluster_prob     = 0.35f;  // P(task in hot zone vs uniform)
    int   n_hot_zones      = 6;      // default hot-zone count (overridden per phase)
    float hot_zone_radius  = 600.f;  // hot-zone radius in metres
    float same_origin_prob = 0.0f;   // P(pickup == previous delivery) — lifelong reuse

    // ── Task distance constraints ──────────────────────────────────────────
    float min_task_dist_m  = 300.f;  // minimum haversine pickup→delivery distance
    float max_task_dist_m  = 8000.f; // maximum haversine pickup→delivery distance

    // ── Agent capacity ─────────────────────────────────────────────────────
    // Number of tasks an agent can hold simultaneously (1 = single-task mode,
    // safe; >1 enables queueing for higher throughput on long-distance graphs).
    int max_tasks_per_agent = 3;

    // ── Agent pool sizing ─────────────────────────────────────────────────
    // The EpisodeRunner pre-allocates a fleet of size
    //   pool = ceil(max_agents() × agent_pool_multiplier)
    // so that EpisodeScenario::agents_mult > 1 (over-provisioned fleet
    // scenarios like Y's "many idle agents, few tasks") can actually be
    // honoured without being clamped down to the nominal phase max. The
    // default 1.5 covers the slack regime (agents_mult ≤ ~1.4); set to 10
    // (or higher) for stress tests like "10× more agents than nominal" used
    // in the full SoTA comparison sweep. Idle surplus agents are inert
    // (never offered tasks when n_active < pool) and cost only a small
    // per-instance memory footprint.
    float agent_pool_multiplier = 1.5f;

    // ── Heterogeneous per-agent capacity (Option M diversification) ────────
    // When enable_heterogeneous_capacity is true, each agent receives a
    // capacity uniformly drawn in [hetero_capacity_min, hetero_capacity_max]
    // (inclusive) at the start of each episode. The per-agent value lives in
    // DeliveryAgent::max_capacity and overrides max_tasks_per_agent for that
    // agent only. The TAM-global max_tasks_per_agent is set to hetero_capacity_max
    // so dimensioning (path-cache, container sizes) stays valid for all draws.
    //
    // OFF by default → existing behaviour (uniform capacity = max_tasks_per_agent).
    bool enable_heterogeneous_capacity = false;
    int  hetero_capacity_min           = 3;
    int  hetero_capacity_max           = 7;

    // ── Reward shaping (MAPPO training only) ───────────────────────────────
    //
    // - delivered  : +task.reward_original * task.importance_original
    // - refused    : -refuse_penalty_w * task.importance_original
    // - unfinished : -unfinished_factor * task.reward_original * task.importance_original
    //                (proportional to the value the agent failed to capture)
    //
    // For default reward=1, importance=1:
    //   delivered  +1.00 | refused -0.15 | unfinished -0.50
    // Indifference threshold p* ≈ 0.23: the agent should refuse a task only if it
    // estimates < 23% probability of delivering it. A flat unfinished penalty
    // (e.g. -0.10) makes "accept-and-fail" dominant over "refuse" regardless of
    // p, so the policy collapses to acc ≈ 100%.
    float refuse_penalty_w   = 0.15f;  // legacy TAM only (Format B / first-fit)
    float unfinished_factor  = 0.7f;   // affected-but-not-delivered penalty (reduced
                                       //   from 1.0 — under strong congestion many
                                       //   undeliveries are structural, not policy fault)

    // ── MC TAM reward shaping (V2) ─────────────────────────────────────────
    //
    // With Format A force_assign, an agent does not "refuse" a task — the
    // argmax wins and the K-1 losers are simply NON-AFFECTED. This is a much
    // weaker signal than a real refusal, hence a lower default magnitude.
    //
    // non_affected_penalty_w : applied to each non-winning candidate at the
    //   moment the TAM finalises. Tells the policy "your score lost the
    //   argmax — slightly lower would have been more honest". Keep small to
    //   avoid pushing every loser toward 0 (would collapse the argmax range).
    float non_affected_penalty_w = 0.03f;

    // congestion_creation_w : negative reward applied to the WINNER at accept
    //   time, proportional to the congestion their plan will add to the
    //   network. Uses PolicyFeatures::congestion_delta_contribution as the
    //   signal — closes the loop between the input vector and the reward.
    //   0.0 = off. 0.1-0.3 = mild to firm "be system-aware".
    float congestion_creation_w  = 0.15f;

    // idle_penalty_w : per-offer penalty applied at episode end to every
    //   buffer entry of an agent that finished the episode IDLE (never
    //   received a task). Encodes "an idle agent is wasted capacity". Only
    //   takes effect when train_mode (or Hybrid online updates) is on.
    float idle_penalty_w         = 0.05f;

    // Reward shaping: split task reward between pickup and delivery events.
    // Shortens credit-assignment delay (decisions get partial signal already
    // when the agent reaches the pickup, not only at delivery 1000+ steps later).
    //
    //   buffer.reward at decision time     = 0
    //   ... agent reaches pickup           : +pickup_reward_frac × imp × reward
    //   ... agent reaches delivery         : +(1 − pickup_reward_frac) × imp × reward  (total = imp × reward)
    //   ... episode ends without delivery,
    //         pickup REACHED               : -unfinished_factor × (1 − pickup_reward_frac) × imp × reward
    //         pickup NOT reached           : -unfinished_factor × imp × reward
    //
    // Picking up therefore protects the agent from the worst penalty and the
    // refusal threshold becomes more informative for capacity-constrained cases.
    float pickup_reward_frac = 0.25f;  // fraction of total reward given at pickup
                                       //   (was 0.3 — slightly lowered so the
                                       //   policy stays motivated to follow
                                       //   through to delivery, not just grab pickup)

    // ── Latency-aware delivery shaping (Phase 2, opt-in) ───────────────────
    //
    // When enable_latency_shaping is true, the delivery credit (the (1−pickup_reward_frac)
    // share paid out at delivery) is scaled by a latency factor:
    //
    //   factor    = 1 − lambda × min(1, trip_steps / max_steps)
    //   delivery  = base_delivery × factor
    //
    // where trip_steps = current_step − task.timeline.picked_step (the time
    // the agent actually spent moving from pickup to delivery). Pickup credit
    // and refuse / unfinished penalties are NOT touched.
    //
    // Goal: make selectivity profitable. A task that ends up taking forever
    // pays less reward, so the policy is rewarded for choosing "fast" tasks
    // and refusing "slow" ones. lambda controls how aggressive the down-weight
    // is; max_steps anchors the normalisation.
    //
    // Defaults (lambda=0.4, max_steps=1500):
    //   trip_steps =    0 → factor = 1.00 (full credit)
    //   trip_steps =  750 → factor = 0.80
    //   trip_steps = 1500 → factor = 0.60
    //   trip_steps > 1500 → factor = 0.60 (clamped)
    //
    // Total accumulated reward on a completed task:
    //   pickup_credit + delivery_credit_scaled
    //   = (pickup_reward_frac + (1 − pickup_reward_frac) × factor) × imp × reward
    // Worst case (factor=0.6) = 0.72 × imp × reward — still positive and well
    // above the refuse penalty (-0.15 × imp), preserving the existing
    // indifference structure.
    //
    // OFF by default — existing options (T, N, X, Y, S, P) keep their reward
    // shape exactly. Option M opts in.
    bool  enable_latency_shaping  = false;
    float latency_shaping_lambda  = 0.4f;
    int   latency_shaping_max_steps = 1500;

    // ── Task-value heterogeneity (Phase 3, opt-in) ─────────────────────────
    //
    // Without this flag, reward is purely proportional to P→D distance and
    // importance is sampled in [0.5, 2.0] — so reward × importance lives in
    // a relatively narrow range, and the policy has no strong "task quality"
    // axis to discriminate on. Phase 2 reward shaping only penalises slow
    // tasks; it does NOT distinguish "valuable short task" from "cheap short
    // task".
    //
    // With the flag ON:
    //   - reward gets multiplied by a per-task random factor in
    //     [value_mul_min, value_mul_max] so two tasks with identical P→D
    //     can have very different rewards (decoupled from effort).
    //   - importance is sampled in [imp_min, imp_max] (wider than the
    //     default).
    //
    // Effective reward × imp range with defaults (mul ∈ [0.5, 2.0],
    // imp ∈ [0.3, 3.0]): a ratio of ~40× between the worst and best task.
    // The selectivity story becomes: refuse low-imp × low-mul (cheap and
    // unrewarding), accept high-imp × high-mul (valuable and unrewarding to
    // miss). Combined with Phase 2 latency shaping, the policy has TWO axes
    // of discrimination (value, effort) — what we want.
    //
    // OFF by default — existing options (T, N, X, Y, S, P) keep their reward
    // shape exactly. Option M opts in.
    bool  enable_task_value_heterogeneity = false;
    float task_value_mul_min = 0.5f;
    float task_value_mul_max = 2.0f;
    float task_imp_min       = 0.3f;
    float task_imp_max       = 3.0f;

    // ── Planning strategy (insertion algorithm) ─────────────────────────────
    // Mutually exclusive (EpisodeRunner enforces: dbvns wins if both set).
    //
    //   default       → Solomon cheapest insertion (pure route-length delta).
    //   use_dh        → Mitrovic-Minic 2004 Double-Horizon, slack-preserving.
    //   use_dbvns     → forward DbVNS-PDP: global replan of the full remaining
    //                    sequence on every task acceptance. Architecturally
    //                    clean: TAM → policy decides accept/refuse →
    //                    DbVNS does the routing.
    //
    // EpisodeRunner mirrors these to PDPGlobalMemory::planning_use_* flags at
    // the start of each run(). DeliveryAgent::receive_task() dispatches on
    // them (DbVNS branch first, then DH, otherwise cheapest insertion).
    bool use_double_horizon_planning = false;
    bool use_dbvns_planning          = false;

    // ── InsertionGreedy threshold ──────────────────────────────────────────
    // Accept task if (reward * importance) / max(insertion_cost_m, ε) > threshold.
    // Insertion cost is in meters, reward*importance ∈ [0.05, 10]. Typical bids
    // are 1e-5 to 1e-2, so threshold ≈ 1e-4 = "accept if at least 1 reward unit
    // per 10 km of routing cost". This filters genuinely unprofitable tasks.
    float insertion_greedy_threshold = 1e-4f;

    // ── TAM multi-candidate mode (opt-in, recoverable) ─────────────────────
    //
    // When tam_multi_candidate is false (default) the TAM behaves EXACTLY as
    // before: dual-Dijkstra, first agent scoring >= 0.5 wins (first-fit),
    // legacy recall on exhaustion. Setting it false restores the current
    // architecture byte-for-byte — nothing else is needed for recovery.
    //
    // When true, the TAM switches to a K-candidate auction:
    //   1. Dijkstra expands until the first valid candidate is found at
    //      x = max(pickup_cost, delivery_cost).
    //   2. Expansion continues until budget = x * ratio(x), or until
    //      tam_mc_max_candidates candidates are collected, or both searches
    //      exhaust. ratio(x) decreases from tam_mc_ratio_max (near x=0) to
    //      tam_mc_ratio_min (x→∞):  ratio(x) = min + (max-min)/(1 + x/scale).
    //   3. All candidates are scored via the policy (try_accept_task — same
    //      call path as legacy). The argmax wins.
    //   4. tam_mc_force_assign = true  → always allocate to the argmax.
    //      tam_mc_force_assign = false → if all scores < 0.5, the TAM marks
    //      the task DEFERRED. EpisodeRunner re-offers it after
    //      tam_mc_recall_time_frac × total_steps, unless the episode is
    //      already past tam_mc_reject_time_frac × total_steps → reject.
    //
    // Candidate = agent reachable from BOTH searches with a valid plan order
    // (pickup before delivery), OR an idle agent with no assigned task
    // (the "exception" — single-side discovery is enough).
    bool  tam_multi_candidate     = false;
    bool  tam_mc_force_assign     = true;    // Format A (true) vs Format B (false)
    int   tam_mc_max_candidates   = 5;
    float tam_mc_ratio_min        = 1.4f;
    float tam_mc_ratio_max        = 3.0f;
    float tam_mc_ratio_scale      = 2000.f;  // metres — ratio decay length
    float tam_mc_recall_time_frac = 0.20f;   // Format B: retry delay (× total_steps)
    float tam_mc_reject_time_frac = 0.70f;   // Format B: reject cutoff (× total_steps)

    // ── Optional depot (warehouse mode) ───────────────────────────────────
    bool                   use_depot   = false;
    osmium::object_id_type depot_node  = 0;

    // ── Dynamic background congestion (Option M training diversification) ──
    // When true, EpisodeRunner spins up a GhostTrafficController that injects
    // synthetic ghost loads into PDPGlobalMemory::congestion_map following the
    // scenario's congestion_profile. Hot ways are sampled at episode start so
    // congestion is spatially heterogeneous (low-cost routes always exist).
    //
    // OFF by default → all pre-existing options (T, N, X, Y) keep their
    // original behaviour unchanged. Only Option M (and any future explicit
    // opt-in) enables it.
    //
    // ghost_n_max_*: per-city-size peak ghost count (adaptive — see
    // MultiCityTrainer::customize_episode_for_city). Tokyo_Small≈20,
    // Medium≈40, Large≈80 by default.
    bool  enable_ghost_traffic = false;
    int   ghost_n_max          = 40;       // peak simultaneous ghost loads
    int   ghost_window_steps   = 5;        // duration each ghost occupies a way
    float ghost_hot_way_frac   = 0.30f;    // fraction of ways used as "hot" pool

    // ── Strong-congestion controls (Option Y high-impact eval) ─────────────
    //
    // When the diagnostic test (Option Z) reports BPR ×1.000-×1.010, the
    // ghost setup is too DIFFUSE to create meaningful edge cost adjustments:
    // 20-80 ghosts dispersed over 10000+ "hot ways" almost never collide,
    // so peak load stays at 1-2 and the BPR factor stays near 1.0. The two
    // knobs below close this gap.
    //
    // ghost_hot_way_count : when > 0, sets the ABSOLUTE number of hot ways
    //   sampled at episode start (overrides ghost_hot_way_frac). Use this to
    //   concentrate ghost traffic onto a small set of "arteries" so multiple
    //   ghosts pile up on the same edge and create real BPR cost increases.
    //   0 = use ghost_hot_way_frac (legacy behaviour).
    //
    // ghost_n_max_user_set : when true, MultiCityTrainer::customize_episode_for_city
    //   does NOT override ghost_n_max with the per-city tier (20/40/80). The
    //   user-supplied value is used as-is across every city. This is required
    //   when you want a SAME-density ghost regime across the eval cities.
    int  ghost_hot_way_count    = 0;
    bool ghost_n_max_user_set   = false;

    // When > 0, the GhostTrafficController derives ghost_n_max at episode
    // start as ceil(density × hot_way_count). This makes the per-edge BPR
    // PROFILE identical across all cities (same average ghost density on
    // each hot way) while letting the absolute ghost count scale with the
    // hot pool size automatically.
    //
    // The user-set values of ghost_n_max and ghost_n_max_user_set are then
    // ignored — they are superseded by the density-derived count. Set to 0
    // to keep the legacy "absolute n_max" behaviour.
    //
    // Default 0.0 preserves backwards-compat for every option except Y/Q
    // which now opt in to this scaling.
    float ghost_density_per_hot_way = 0.0f;

    // ── Compute-amortisation knobs ─────────────────────────────────────────
    //
    // Both knobs reduce simulation wallclock by amplifying the per-entry load
    // contribution, so the same congestion intensity is reached with fewer
    // CongestionMap hash operations (ghosts) and/or fewer real agents.
    //
    // ghost_load_per_unit : each ghost entry counts for K load units. Spawning
    //   target_load=N at K=4 means 250 hash ops per spawn cycle instead of
    //   1000. BPR is polynomial degree β so quantisation impact stays <1% for
    //   K ≤ 5 on the configured β=4 default. Default 4 = ×4 speedup on the
    //   dominant ghost spawn/expire path with negligible numerical change.
    //
    // fleet_load_per_agent : each real agent counts for K load units when its
    //   plan is registered in the CongestionMap. This lets the user shrink the
    //   actual fleet by K while preserving the congestion footprint of a K×
    //   larger fleet. Default 1 = no behavioural change; set to 2-3 in
    //   conjunction with reduced phase.n_agents_start/end for fast iteration.
    int  ghost_load_per_unit       = 4;
    int  fleet_load_per_agent      = 1;
    // When > 0, customize_episode_for_city divides the per-tier
    // n_agents_start / n_agents_end values by this factor at the start of an
    // episode. Combined with fleet_load_per_agent > 1 this reduces the active
    // fleet (less A*, less policy forward) while preserving the congestion
    // intensity of the larger fleet. Default 1 = no change.
    int  fleet_size_divisor        = 1;

    // ── Metrics ────────────────────────────────────────────────────────────
    bool collect_metrics    = true;
    int  time_window_steps  = 200;   // window used for throughput metric

    // ── Convenience ───────────────────────────────────────────────────────
    int  total_steps() const {
        int s = 0; for (const auto& p : phases) s += p.steps; return s;
    }
    int  max_agents()  const {
        int m = 0;
        for (const auto& p : phases)
            m = std::max(m, std::max(p.n_agents_start, p.n_agents_end));
        return m;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Global state vector — centralised critic input (kGlobSz = 20 floats)
// ════════════════════════════════════════════════════════════════════════════
//
// Assembled once per simulation step by the episode runner.
// One GlobalState per buffered Experience is passed to train_epoch().
struct GlobalState {
    // System (8)
    float time_ratio     = 0.f; // step / total_steps
    float n_agents_ratio = 0.f; // active_agents / max_agents
    float avail_ratio    = 0.f; // available_tasks / tasks_created
    float alloc_ratio    = 0.f; // allocated_tasks / tasks_created
    float done_ratio     = 0.f; // finished_tasks / tasks_created
    float avg_load       = 0.f; // mean(agent_tasks) / kMaxLoad
    float max_load       = 0.f; // max(agent_tasks)  / kMaxLoad
    float arrival_rate   = 0.f; // tasks_last_window / (lambda * window)

    // Performance (4)
    float throughput     = 0.f; // done / tasks_created
    float avg_latency    = 0.f; // mean_completion / time_window_steps
    float accept_rate    = 0.f; // accepted / (accepted + refused)
    float avg_efficiency = 0.f; // mean agent route quality ∈ [0,1]

    // Context (4)
    float city_id_norm   = 0.f; // city index / (num_cities-1)
    float density_phase  = 0.f; // current phase label (0/0.5/1)
    float cluster_ratio  = 0.f; // fraction of tasks in hot zones (running)
    float congestion     = 0.f; // placeholder: mean way congestion

    // Reserved (4) — available for future features
    float pad[4] = {};

    static_assert(kGlobSz == 20, "GlobalState size must match kGlobSz in ObjectiveDMPolicy");

    void to_array(float* dst) const {
        dst[ 0]=time_ratio;    dst[ 1]=n_agents_ratio;
        dst[ 2]=avail_ratio;   dst[ 3]=alloc_ratio;
        dst[ 4]=done_ratio;    dst[ 5]=avg_load;
        dst[ 6]=max_load;      dst[ 7]=arrival_rate;
        dst[ 8]=throughput;    dst[ 9]=avg_latency;
        dst[10]=accept_rate;   dst[11]=avg_efficiency;
        dst[12]=city_id_norm;  dst[13]=density_phase;
        dst[14]=cluster_ratio; dst[15]=congestion;
        dst[16]=pad[0];        dst[17]=pad[1];
        dst[18]=pad[2];        dst[19]=pad[3];
    }

    std::array<float, kGlobSz> to_array() const {
        std::array<float, kGlobSz> a{};
        to_array(a.data());
        return a;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Comparison metrics — one record per episode × method
// ════════════════════════════════════════════════════════════════════════════
//
// Covers all axes of the user's comparison specification:
//   Environment, VRP-related (DbVNS/IVNS/LKH planning),
//   MAPD-related (throughput & latency), RL policy.
struct ComparisonMetrics {
    // Identity
    std::string city;
    std::string method;      // "DMAS-DbVNS" | "IVNS" | "LKH" | "MAPPO" | …
    int         episode = 0;

    // ── Environment descriptors ────────────────────────────────────────────
    int   total_steps   = 0;
    int   n_agents_min  = 0;
    int   n_agents_max  = 0;
    bool  has_depot     = false;
    int   road_nodes    = 0;    // |V| of the road graph
    int   road_ways     = 0;    // |E|
    float area_km2      = 0.f;

    // ── VRP-related ────────────────────────────────────────────────────────
    // Objective: min distance/time  |  Constraint: P strictly before D
    float avg_t_ratio   = 0.f;  // mean(algo_cost / oracle_cost), lower = better
    float best_t_ratio  = 0.f;
    int   pdp_violations= 0;    // must remain 0

    // ── MAPD-related ───────────────────────────────────────────────────────
    // Metric 1: task throughput in time window
    int   tasks_appeared   = 0;
    int   tasks_completed  = 0;
    float throughput_rate  = 0.f;   // completed / appeared

    // Metric 2: completeness latency relative to available agents
    float latency_mean     = 0.f;   // mean steps appearance → completion
    float latency_per_agent= 0.f;   // latency_mean / mean_active_agents
    float agent_utilisation= 0.f;   // mean fraction of steps with active work
    float refuse_rate      = 0.f;   // tasks_refused / tasks_appeared

    // Metric 3: network congestion & travel efficiency
    float mean_congestion          = 0.f;  // mean edge load (mean_load_now) over all steps
    float mean_trip_steps          = 0.f;  // mean pickup→delivery travel time (steps)
    float mean_wait_steps          = 0.f;  // mean steps appearance → pickup arrival (response delay)
    float mean_road_pd_m           = 0.f;  // mean A* road distance pickup→delivery (metres)
    float delivery_route_efficiency= 0.f;  // mean_road_pd_m / (mean_trip_steps × speed_mps)
                                           // ∈ (0,1]: 1=shortest path at full speed, <1=detours

    // Metric 4: SPATIAL complexity over served tasks (pickup ∪ delivery points)
    //   bbox_area_km2        : bbox of all served task endpoints
    //   convex_hull_km2      : convex-hull area (finer dispersion measure)
    //   mean_pd_distance_m   : mean direct P→D distance per served task
    //   mean_nn_pickup_m     : mean nearest-neighbour pickup distance
    float bbox_area_km2          = 0.f;
    float convex_hull_area_km2   = 0.f;
    float mean_pd_distance_m     = 0.f;
    float mean_nn_pickup_m       = 0.f;

    // Metric 5: derived TEMPORAL complexity
    //   compute_time_per_task_ms      = wallclock_ms / tasks_appeared
    //   compute_time_per_decision_us  = wallclock_ms × 1000 / (tasks × n_agents)
    float compute_time_per_task_ms     = 0.f;
    float compute_time_per_decision_us = 0.f;

    // Metric 6: VALIDITY counters (sanity check — should stay at 0)
    //   pairing_violations  : tasks where picked_step ≥ delivered_step
    //                          OR delivered without pickup, OR before arrival
    //                          OR spatio-temporal teleportation suspect
    //   capacity_violations : agents with peak load > max_tasks_per_agent
    int   pairing_violations_runtime  = 0;
    int   capacity_violations_runtime = 0;

    // ── RL policy ──────────────────────────────────────────────────────────
    float accept_rate      = 0.f;
    float reward_mean      = 0.f;
    float actor_loss       = 0.f;
    float critic_loss      = 0.f;
    int   buffer_size      = 0;

    // ── Selectivity quality (Option M diagnostic columns) ──────────────────
    //
    // Designed to surface whether a policy's accept/refuse choices are
    // actually high-quality, independently of throughput. Two policies can
    // share a throughput but differ markedly here.
    //
    //   completion_per_accepted    = tasks_completed / tasks_accepted
    //                                 1.0 = every accepted task delivered;
    //                                 <1.0 = the policy committed to tasks it
    //                                 could not finish (poor selection).
    //   unfinished_accept_rate     = (accepted - completed) / accepted
    //                                 The complement above. Useful as a
    //                                 standalone "waste" indicator.
    //   mean_congestion_at_decision = mean of CongestionMap::mean_load_now()
    //                                 sampled only at accept/refuse decision
    //                                 steps (not at every step). Tells us how
    //                                 congested the network was when the
    //                                 policy made each call.
    //   n_ghost_active_mean        = mean active ghost loads across the
    //                                 episode (0 when ghost traffic disabled).
    //   congestion_profile_label   = profile name from the scenario sampler
    //                                 ("flat", "ramp_up_down", ...). Empty
    //                                 string when no ghost controller was
    //                                 active for this episode.
    float       completion_per_accepted     = 0.f;
    float       unfinished_accept_rate      = 0.f;
    float       mean_congestion_at_decision = 0.f;
    float       n_ghost_active_mean         = 0.f;
    std::string congestion_profile_label;

    // ── Performance diagnostics — multi-axis evaluation (Option X) ─────────
    //
    // Pure throughput hides most of what makes a policy good or bad. These
    // columns expose the dimensions that distinguish a policy that simply
    // accepts a lot from one that accepts selectively, balances load across
    // agents, and produces clean routes.
    //
    //   Load balance:
    //     agent_completed_gini   ∈ [0,1] Gini coefficient on per-agent
    //                            delivered-task counts. 0 = perfectly even,
    //                            1 = one agent does everything. Reported
    //                            over the active fleet (n_active inferred
    //                            from agents that received at least one
    //                            allocation).
    //     agent_completed_std    Standard deviation of per-agent deliveries,
    //                            divided by the mean (= coeff of variation,
    //                            unitless). Complements Gini for skewed
    //                            distributions.
    //
    //   Selectivity discrimination (does the policy refuse low-value tasks?):
    //     mean_imp_accepted      Mean task.importance_original on ACCEPTED
    //                            offers.
    //     mean_imp_refused       Mean task.importance_original on REFUSED
    //                            offers. A policy with real selectivity has
    //                            mean_imp_accepted > mean_imp_refused.
    //
    //   Context-aware acceptance (does the policy adapt to network state?):
    //     accept_rate_high_cong  Accept rate when decision-time congestion
    //                            (mean_load_now) ≥ 1.0 (≥ 1 unit of load
    //                            per loaded way on average). Should drop
    //                            under heavy traffic if the policy senses
    //                            saturation.
    //     accept_rate_low_cong   Same when < 1.0. The gap accept_low - accept_high
    //                            quantifies congestion sensitivity.
    //
    //   Route effort per delivery:
    //     mean_extra_steps_per_task  (mean_trip_steps − ideal_steps_per_task)
    //                            in steps. Quantifies detour overhead beyond
    //                            the straight A* path (delivery_route_efficiency
    //                            is its ratio counterpart, this is the magnitude).
    float agent_completed_gini    = 0.f;
    float agent_completed_std     = 0.f;
    float mean_imp_accepted       = 0.f;
    float mean_imp_refused        = 0.f;
    float accept_rate_high_cong   = 0.f;
    float accept_rate_low_cong    = 0.f;
    float mean_extra_steps_per_task = 0.f;

    // ── Network-level congestion impact (policy-attributable analysis) ─────
    //
    // These metrics measure HOW the policy's decisions affect the network
    // state, not just aggregate throughput. Designed for the comparison
    // "which policy generates the lowest congestion footprint at equal
    // throughput?" — i.e. for the Pareto-style analysis MAPPO vs MAPPER
    // vs Hybrid (without re-anchoring on TAA).
    //
    //   peak_congestion          Max load_now on any single edge observed
    //                            over the episode. A high peak indicates
    //                            policy decisions caused bursty pileups.
    //   mean_overlap_edges       Mean count of edges with load >= 2 per
    //                            step (= conflict density). Captures
    //                            "how often agents shared edges".
    //   congestion_variance      Std of mean_load_now() across episode
    //                            steps. High variance = volatile policy
    //                            choices, low = stable network usage.
    //   route_congestion_exposure  Mean load_now observed on edges that
    //                            real agents are actively traversing
    //                            (in_transit). Distinct from network-wide
    //                            mean: only counts what the agents
    //                            actually experience. The most directly
    //                            policy-attributable congestion metric.
    //   max_agent_completed      Max tasks delivered by any single agent
    //                            (raw, not normalised) — complements Gini.
    //   min_agent_completed      Min tasks delivered (over agents that
    //                            received at least one allocation).
    //   total_fleet_distance_m   Sum of pickup→delivery road distance over
    //                            all completed tasks (proxy for real
    //                            distance traveled by the fleet).
    int   peak_congestion         = 0;
    float mean_overlap_edges      = 0.f;
    float congestion_variance     = 0.f;
    float route_congestion_exposure = 0.f;
    int   max_agent_completed     = 0;
    int   min_agent_completed     = 0;
    float total_fleet_distance_m  = 0.f;

    // ── Selection intelligence (delivery quality, not just count) ──────────
    //
    // The throughput_rate column measures "how many tasks were delivered" —
    // it does NOT measure whether the delivered tasks were the VALUABLE ones.
    // A policy that refuses low-value tasks to capacity-protect high-value
    // ones can win on these columns at equal or lower throughput.
    //
    //   value_throughput_rate : Σ value(delivered) / Σ value(appeared)
    //                            where value = reward_original × importance_original.
    //                            1.0 = every value unit offered was eventually
    //                            captured; < 1.0 = value loss.
    //   mean_completion_value : Σ value(delivered) / N_delivered. Average
    //                            "size" of a delivered task. High = the
    //                            policy selects valuable tasks.
    //   value_loss_to_refusal : Σ value(refused). Absolute value the policy
    //                            DECIDED not to pursue. Low = no waste.
    float value_throughput_rate = 0.f;
    float mean_completion_value = 0.f;
    float value_loss_to_refusal = 0.f;

    // ── Real impact on edge traversal times ────────────────────────────────
    //
    // route_congestion_exposure (existing) reports load on edges traversed —
    // but load alone does not tell the agent how SLOW it actually was. The
    // BPR factor (dynamic_cost / static_cost) does. These columns close the
    // chain "ghost load → BPR factor → agent slowdown" so we can argue the
    // policy reduces real travel time, not just observed load.
    //
    //   mean_bpr_along_route          : mean (dynamic_cost / static_cost)
    //                                    over every edge any agent traversed.
    //                                    1.0 = no slowdown; > 1.0 = effective
    //                                    slowdown due to congestion encountered.
    //   time_lost_to_congestion_steps : Σ ((dynamic_cost - static_cost) / speed)
    //                                    over every traversal. The actual
    //                                    fleet-wide step budget lost to jams.
    //   n_traversals_in_jam           : count of (edge, traversal) where the
    //                                    edge had load ≥ 5 at the moment the
    //                                    agent entered it. Measures how often
    //                                    the policy's routes hit chokepoints.
    float mean_bpr_along_route          = 1.f;
    float time_lost_to_congestion_steps = 0.f;
    int   n_traversals_in_jam           = 0;

    // ── Allocation optimality vs oracle (MCA full-scan) ────────────────────
    //
    // The TAM contacts ~2-5 agents (mean_agents_offered_per_task). MCA scans
    // all n_active. Question: does the TAM converge on the SAME winner MCA
    // would have picked? marginal_cost_ratio_vs_oracle answers this directly.
    //
    //   marginal_cost_ratio_vs_oracle : mean over all allocations of
    //                                    (chosen_agent_insertion_cost /
    //                                     min_over_all_eligible_agents_insertion_cost).
    //                                    1.0 = MCA-optimal selection;
    //                                    > 1.0 = the TAM "missed" a better
    //                                    agent due to local search bound.
    //                                    Together with mean_agents_offered_per_task,
    //                                    this is THE Pareto plot for the
    //                                    paper's "minimize communication
    //                                    overhead WITHOUT losing allocation
    //                                    quality" claim.
    float marginal_cost_ratio_vs_oracle = 1.f;

    // ── Temporal complexity (per-allocation cost) ──────────────────────────
    //
    // compute_time_per_task_ms / compute_time_per_decision_us (existing) are
    // wallclock-divided-by-task averages over the WHOLE episode. They mix
    // movement, A*, plan() and offer cost. To isolate the ALLOCATION cost
    // specifically (the part the TAM + policy own), we add:
    //
    //   mean_allocation_time_us : wallclock per offer_task() call. Captures
    //                              the TAM Dijkstra + every policy score call
    //                              for that task, but not the subsequent
    //                              insertion / movement. Comparable across
    //                              modes because every mode goes through
    //                              offer_task on the same offers.
    //   mean_tam_dijkstra_steps : mean number of TAM step() iterations per
    //                              task. Proxy for incremental Dijkstra work
    //                              the TAM did. 0 for non-TAM baselines.
    float mean_allocation_time_us = 0.f;
    float mean_tam_dijkstra_steps = 0.f;

    // ── TAM efficiency metrics (paper's "minimize comm overhead" claim) ────
    //
    // Only meaningful for TAM-driven modes (MAPPO, FaithfulMAPPER, IPPO,
    // MAPPER, Hybrid, TamAlwaysAccept). For SoTA baselines that scan the
    // full active fleet (MCA, CongestionAware, TrafficFlow, TokenPassing,
    // PIBT, LaCAM, DoubleHorizon, MCA, InsertionGreedy, Greedy, Random),
    // EpisodeRunner reports n_active as a uniform sentinel so the column
    // stays comparable across modes.
    //
    //   mean_agents_offered_per_task    : distinct delivery agents the TAM
    //                                     actually asked (offer_to_agent called
    //                                     at least once). The CORE metric for
    //                                     the "minimize communication" claim.
    //   mean_recall_rounds_per_task     : recall rounds before allocation/
    //                                     exhaustion. 0 = first-pass success.
    //   mean_candidates_scored_per_task : multi-candidate mode only; size of
    //                                     the candidate set scored at finalise.
    //                                     0 in legacy mode.
    float mean_agents_offered_per_task    = 0.f;
    float mean_recall_rounds_per_task     = 0.f;
    float mean_candidates_scored_per_task = 0.f;

    // ── Accumulation helpers (called by episode runner) ────────────────────
    void on_task_completed(int latency_steps, float t_ratio);
    void on_task_refused();
    void on_offer(bool accepted, float reward);
    void finalise(int mean_active_agents, int steps_run);

private:
    int n_completed_acc_ = 0;  // running count for incremental mean
    int n_offered_acc_   = 0;
};

#endif // TRAINING_EPISODE_CONFIG_HPP
