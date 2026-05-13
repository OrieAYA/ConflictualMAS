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
    int max_tasks_per_agent = 1;

    // ── Reward shaping (MAPPO training only) ───────────────────────────────
    // - delivered  : task->reward * task->importance  (positive, on completion)
    // - refused    : -refuse_penalty_w * task->importance  (slight bias toward acceptance)
    // - unfinished : unfinished_penalty (negative, applied at episode end for
    //                accepted tasks that never delivered — discourages saturating
    //                queues with tasks the agent can't complete in time)
    float refuse_penalty_w  = 0.05f;
    float unfinished_penalty = -1.0f;

    // ── InsertionGreedy threshold ──────────────────────────────────────────
    // Accept task if (reward * importance) / max(insertion_cost, ε) > threshold.
    // 1.0 means "expect at least 1 unit of reward per unit of insertion cost".
    float insertion_greedy_threshold = 0.5f;

    // ── Optional depot (warehouse mode) ───────────────────────────────────
    bool                   use_depot   = false;
    osmium::object_id_type depot_node  = 0;

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

    // ── RL policy ──────────────────────────────────────────────────────────
    float accept_rate      = 0.f;
    float reward_mean      = 0.f;
    float actor_loss       = 0.f;
    float critic_loss      = 0.f;
    int   buffer_size      = 0;

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
