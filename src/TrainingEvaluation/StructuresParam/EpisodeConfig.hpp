#ifndef TRAINING_EPISODE_CONFIG_HPP
#define TRAINING_EPISODE_CONFIG_HPP

#include "TrainingEvaluation/StructuresParam/CityConfig.hpp"
#include "Environment/Structure/EventStream.hpp"   // event_tuning constants
#include "DMASforPD/Policy/PolicyKit.hpp"          // kGlobSz = 20
#include <osmium/osm/types.hpp>
#include <array>
#include <string>
#include <vector>

// One segment of the episode timeline: fleet size ramps linearly start→end
// across the phase. Task TIMING is no longer phase-driven — it follows the
// scenario's temporal profile (EpisodeGenerator::generate(profile)); phases
// keep the fleet ramp, the density label and the hot-zone count.
struct DensityPhase {
    int   steps;              // duration in simulation steps
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

    // SCE — environment size scalar (Small/Medium/Large = 1/2/5) and RM — ratio
    // multiplier (1 in training; incremented per seed in evaluation). Together
    // they fix the event counts: tasks = round(100·SCE·RM), ghosts =
    // round(10000·SCE·RM); fleet = round(10·SCE·AM) with AM the fleet regime.
    float env_scale  = 1.f;   // SCE
    float ratio_mult = 1.f;   // RM

    // Phase schedule (in order). Default: 3-phase day (3600 steps) with a
    // linearly growing fleet. Task timing follows the scenario profile.
    std::vector<DensityPhase> phases = {
        { 1000,  8, 10, 0.0f,  4 },
        { 1500, 10, 13, 0.5f,  6 },
        { 1100, 13, 15, 1.0f,  8 },
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

    // ── Reward shaping v2 (paper Table 2 revision) — per-term ablation flags ─
    // ALL default OFF: with a flag off, the legacy term above is the exact
    // fallback. enable_all_reward_shaping() (RewardShaping.hpp) turns the
    // whole revision on (used by apply_paper_environment for T/Y runs).
    bool rs_congestion_route      = false; // §1 winner pays real BPR surcharge of its route
    bool rs_loser_mu              = false; // §2 loser penalty × its own bid score μ
    bool rs_pickup_latency        = false; // §3 pickup credit × φ(t_pickup − t_accept)
    bool rs_unfinished_cost_ratio = false; // §4 unfinished × clip(c_ins/c̄_ins)
    bool rs_idle_refine           = false; // §5 μ-band entries + never-assigned agents only
    bool rs_normalization         = false; // §8 static λ_x scale division
    bool rs_weight_annealing      = false; // §9 linear anneal to 0 of w_naff + w_idle only

    int   rs_e_anneal = 450;   // §9 annealing horizon (episodes; = 50-round protocol)
    float rs_unfinished_clip_lo = 0.5f;  // §4 clip bounds on c_ins/c̄_ins
    float rs_unfinished_clip_hi = 2.0f;
    float rs_mu_idle_lo = 0.3f;          // §5 μ band actually debited
    float rs_mu_idle_hi = 0.7f;

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
    // The FULL ghost event stream (round(10000·SCE·RM) events) is generated and
    // injected on the congestion chains at episode start
    // (GhostTrafficController::reset). Each ghost occupies one hot edge over
    // [t, t+window] with a load drawn in [kGhostLoadMin, kGhostLoadMax], its
    // appearance step distributed per the congestion profile. OFF = no ghosts.
    bool  enable_ghost_traffic = false;
    int   ghost_window_steps   = 250;      // x: duration a ghost occupies a way
    // φh — hot-way pool fraction (default: event_tuning::kHotWayFraction).
    float ghost_hot_way_frac   = event_tuning::kHotWayFraction;

    // Test escape hatches: hot_way_count > 0 sets an absolute hot-way pool
    // (overrides φh); ghost_n_events_user_set uses ghost_n_events verbatim
    // instead of the round(10000·SCE·RM) derivation.
    int  ghost_hot_way_count     = 0;
    int  ghost_n_events          = 0;      // absolute ghost-event count (user-set only)
    bool ghost_n_events_user_set = false;

    // Congestion realism: count each real agent as K load units on its edges.
    int  fleet_load_per_agent    = 1;

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
    float accept_rate    = 0.f; // accepted / offered (incl. no-candidate)
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

#endif // TRAINING_EPISODE_CONFIG_HPP
