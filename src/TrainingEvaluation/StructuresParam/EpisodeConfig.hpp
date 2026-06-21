#ifndef TRAINING_EPISODE_CONFIG_HPP
#define TRAINING_EPISODE_CONFIG_HPP

#include "TrainingEvaluation/StructuresParam/CityConfig.hpp"
#include "DMASforPD/Policy/PolicyKit.hpp"   // kGlobSz = 20
#include <osmium/osm/types.hpp>
#include <array>
#include <string>
#include <vector>

// One segment of the episode timeline. Task arrival is Poisson with
//   lambda = tasks_per_agent * avg(n_agents_start, n_agents_end) / steps
// (lambda > 1 is legal). Fleet size ramps linearly start→end across the phase.
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

    float speed_mps = 5.0f;          // agent speed (m/step); 5 m/s ≈ 18 km/h

    // Phase schedule (in order). Default: Amazon-scale 3-phase day (3600 steps),
    // lambda ≈ 0.90 / 1.53 / 3.18 tasks/step.
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
    float min_task_dist_m  = 300.f;  // min haversine pickup→delivery distance
    float max_task_dist_m  = 8000.f; // max haversine pickup→delivery distance

    // Tasks an agent can hold at once (1 = single-task; >1 = queueing).
    int max_tasks_per_agent = 3;

    // Fleet is pre-allocated as ceil(max_agents() × agent_pool_multiplier) so
    // over-provisioned scenarios (agents_mult > 1) are honoured (surplus agents
    // stay inert). 1.5 covers the slack regime; raise (e.g. 10) for stress sweeps.
    float agent_pool_multiplier = 1.5f;

    // Per-agent capacity drawn in [min,max] at episode start (→ DeliveryAgent::
    // max_capacity, overrides max_tasks_per_agent per agent). OFF = uniform.
    bool enable_heterogeneous_capacity = false;
    int  hetero_capacity_min           = 3;
    int  hetero_capacity_max           = 7;

    // ── Reward shaping (training only) ─────────────────────────────────────
    // delivered +reward·imp | refused -refuse_penalty_w·imp | unfinished
    // -unfinished_factor·reward·imp. Indifference threshold p* ≈ 0.23.
    float refuse_penalty_w   = 0.15f;  // legacy TAM only (Format B / first-fit)
    float unfinished_factor  = 0.7f;   // affected-but-not-delivered penalty

    // Format A: each non-winning candidate gets a small "you lost the argmax" penalty.
    float non_affected_penalty_w = 0.03f;
    // Winner penalty ∝ congestion their plan adds (congestion_delta_contribution).
    float congestion_creation_w  = 0.15f;
    // Per-offer penalty for agents that finished the episode idle (train only).
    float idle_penalty_w         = 0.05f;

    // Split task reward pickup/delivery to shorten credit-assignment delay.
    float pickup_reward_frac = 0.25f;  // fraction of total reward paid at pickup

    // Scale delivery credit by 1 − lambda·min(1, trip_steps/max_steps) so slow
    // trips pay less (selectivity becomes profitable). OFF = no latency shaping.
    bool  enable_latency_shaping  = false;
    float latency_shaping_lambda  = 0.4f;
    int   latency_shaping_max_steps = 1500;

    // Per-task reward ×U[mul_min,mul_max], importance ~U[imp_min,imp_max] —
    // decouples value from effort (a 2nd discrimination axis). OFF = narrow range.
    bool  enable_task_value_heterogeneity = false;
    float task_value_mul_min = 0.5f;
    float task_value_mul_max = 2.0f;
    float task_imp_min       = 0.3f;
    float task_imp_max       = 3.0f;

    // ── Planning strategy (mutually exclusive; dbvns wins if both set) ──────
    // default → cheapest insertion | use_dh → Double-Horizon (slack-preserving) |
    // use_dbvns → forward DbVNS-PDP global replan. Mirrored to
    // PDPGlobalMemory::planning_use_* and dispatched in receive_task().
    bool use_double_horizon_planning = false;
    bool use_dbvns_planning          = false;

    // InsertionGreedy: accept if reward·imp / max(insertion_cost_m, ε) > threshold.
    float insertion_greedy_threshold = 1e-4f;

    // ── TAM multi-candidate (paper Algorithm 1) ────────────────────────────
    // K-candidate auction: dual Dijkstra from pickup/delivery, budget x·ratio(x)
    // with ratio = min + (max-min)/(1 + x/scale); winner = argmax μ. force_assign:
    // Format A force-assigns the argmax if nobody bids; Format B defers (recall
    // after recall_frac, reject after reject_frac × total_steps).
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

    // ── Dynamic background (ghost) congestion ──────────────────────────────
    // GhostTrafficController injects synthetic loads into congestion_map per the
    // scenario profile; hot ways are sampled at episode start. OFF = no ghosts.
    // ghost_n_max is per-city-adaptive (see customize_episode_for_city).
    bool  enable_ghost_traffic = false;
    int   ghost_n_max          = 40;       // peak simultaneous ghost loads
    int   ghost_window_steps   = 5;        // duration each ghost occupies a way
    float ghost_hot_way_frac   = 0.30f;    // fraction of ways used as "hot" pool

    // hot_way_count > 0 sets the absolute hot-way pool (overrides frac) to make
    // ghosts pile up and create real BPR; n_max_user_set keeps a same-density
    // regime across cities (no per-tier override of ghost_n_max).
    int  ghost_hot_way_count    = 0;
    bool ghost_n_max_user_set   = false;

    // > 0: derive ghost_n_max = ceil(density × hot_way_count) at episode start so
    // the per-edge BPR profile is identical across cities (supersedes n_max). 0 = off.
    float ghost_density_per_hot_way = 0.0f;

    // Compute-amortisation: count each ghost / agent as K load units to reach the
    // same congestion with fewer hash ops / fewer real agents (BPR β=4 → <1% error).
    int  ghost_load_per_unit       = 4;
    int  fleet_load_per_agent      = 1;
    int  fleet_size_divisor        = 1;   // > 0: divide phase n_agents by this

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
// Global state vector — centralised critic input (kGlobSz = 20 floats).
// Assembled once per simulation step; one per buffered Experience → train_epoch.
// ════════════════════════════════════════════════════════════════════════════
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

    // Reserved (4) — future features
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
// Comparison metrics — one record per episode × method.
// Axes: environment, VRP (planning), MAPD (throughput/latency), RL policy,
// plus selectivity / congestion-impact / TAM-efficiency diagnostics.
// ════════════════════════════════════════════════════════════════════════════
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

    // ── VRP — min distance/time, P strictly before D ───────────────────────
    float avg_t_ratio   = 0.f;  // mean(algo_cost / oracle_cost), lower = better
    float best_t_ratio  = 0.f;
    int   pdp_violations= 0;    // must remain 0

    // ── MAPD — throughput ──────────────────────────────────────────────────
    int   tasks_appeared   = 0;
    int   tasks_completed  = 0;
    float throughput_rate  = 0.f;   // completed / appeared

    // ── MAPD — latency / utilisation ───────────────────────────────────────
    float latency_mean     = 0.f;   // mean steps appearance → completion
    float latency_per_agent= 0.f;   // latency_mean / mean_active_agents
    float agent_utilisation= 0.f;   // mean fraction of steps with active work
    float refuse_rate      = 0.f;   // tasks_refused / tasks_appeared

    // ── Network congestion & travel efficiency ─────────────────────────────
    float mean_congestion          = 0.f;  // mean edge load over all steps
    float mean_trip_steps          = 0.f;  // mean pickup→delivery travel time (steps)
    float mean_wait_steps          = 0.f;  // mean appearance → pickup arrival (response delay)
    float mean_road_pd_m           = 0.f;  // mean A* road distance pickup→delivery (m)
    float delivery_route_efficiency= 0.f;  // road_pd / (trip_steps × speed) ∈ (0,1]

    // ── Spatial complexity over served tasks ───────────────────────────────
    float bbox_area_km2          = 0.f;  // bbox of served endpoints
    float convex_hull_area_km2   = 0.f;  // convex-hull area
    float mean_pd_distance_m     = 0.f;  // mean direct P→D distance
    float mean_nn_pickup_m       = 0.f;  // mean nearest-neighbour pickup distance

    // ── Temporal complexity (derived) ──────────────────────────────────────
    float compute_time_per_task_ms     = 0.f;  // wallclock / tasks_appeared
    float compute_time_per_decision_us = 0.f;  // wallclock×1000 / (tasks × n_agents)

    // ── Validity counters (should stay 0) ──────────────────────────────────
    int   pairing_violations_runtime  = 0;   // picked ≥ delivered, etc.
    int   capacity_violations_runtime = 0;   // peak load > max_tasks_per_agent

    // ── RL policy ──────────────────────────────────────────────────────────
    float accept_rate      = 0.f;
    float reward_mean      = 0.f;
    float actor_loss       = 0.f;
    float critic_loss      = 0.f;
    int   buffer_size      = 0;

    // ── Selectivity quality (delivery quality, not just count) ─────────────
    float       completion_per_accepted     = 0.f;  // completed / accepted
    float       unfinished_accept_rate      = 0.f;  // (accepted-completed) / accepted
    float       mean_congestion_at_decision = 0.f;  // mean_load_now at decision steps
    float       n_ghost_active_mean         = 0.f;  // mean active ghost loads
    std::string congestion_profile_label;            // scenario profile name

    // ── Multi-axis diagnostics (load balance, selectivity, context) ────────
    float agent_completed_gini    = 0.f;  // Gini on per-agent deliveries
    float agent_completed_std     = 0.f;  // CoV of per-agent deliveries
    float mean_imp_accepted       = 0.f;  // mean importance on accepted offers
    float mean_imp_refused        = 0.f;  // mean importance on refused offers
    float accept_rate_high_cong   = 0.f;  // accept rate when load ≥ 1.0
    float accept_rate_low_cong    = 0.f;  // accept rate when load < 1.0
    float mean_extra_steps_per_task = 0.f; // trip_steps − ideal_steps (detour)

    // ── Network-level congestion impact (policy-attributable) ──────────────
    int   peak_congestion         = 0;    // max load_now on any edge
    float mean_overlap_edges      = 0.f;  // mean #edges with load ≥ 2 / step
    float congestion_variance     = 0.f;  // std of mean_load_now across steps
    float route_congestion_exposure = 0.f; // mean load on edges agents traverse
    int   max_agent_completed     = 0;
    int   min_agent_completed     = 0;
    float total_fleet_distance_m  = 0.f;  // Σ road P→D over completed tasks

    // ── Selection intelligence (value, not just count) ─────────────────────
    float value_throughput_rate = 0.f;  // Σvalue(delivered) / Σvalue(appeared)
    float mean_completion_value = 0.f;  // Σvalue(delivered) / N_delivered
    float value_loss_to_refusal = 0.f;  // Σvalue(refused)

    // ── Real impact on edge traversal times (ghost load → BPR → slowdown) ──
    float mean_bpr_along_route          = 1.f;  // mean dynamic/static over traversals
    float time_lost_to_congestion_steps = 0.f; // Σ (dynamic-static)/speed
    int   n_traversals_in_jam           = 0;    // traversals entered at load ≥ 5

    // ── Allocation optimality vs MCA oracle ────────────────────────────────
    float marginal_cost_ratio_vs_oracle = 1.f;  // chosen / min-over-eligible cost

    // ── RMCA(r) regret [Chen+2021]; populated only for PolicyMode::RMCA ─────
    float rmca_relative_regret  = 0.f;  // mc(k2)/mc(k1)
    float rmca_marginal_cost_k1 = 0.f;  // cheapest insertion cost (m)
    float rmca_marginal_cost_k2 = 0.f;  // 2nd-cheapest insertion cost (m)

    // ── Per-allocation cost breakdown ──────────────────────────────────────
    float mean_allocation_time_us = 0.f;  // wallclock per offer_task()
    float mean_tam_dijkstra_steps = 0.f;  // mean TAM step() iterations / task
    float path_compute_time_ms    = 0.f;  // total get_or_compute_path time
    float mean_pure_alloc_time_ms = 0.f;  // per-offer TAM+policy time (− path)

    // ── TAM efficiency ("minimize comm overhead"); TAM modes only ──────────
    float mean_agents_offered_per_task    = 0.f;  // distinct agents asked
    float mean_recall_rounds_per_task     = 0.f;  // recall rounds before allocation
    float mean_candidates_scored_per_task = 0.f;  // candidate-set size (MC mode)

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
