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
// Matches the user spec: low → denser → very dense with limited agents.
// The three-phase curriculum makes the policy learn to balance workload
// under increasing pressure and shrinking fleet.
struct DensityPhase {
    int   steps;          // duration in simulation steps
    float lambda;         // mean tasks/step (Poisson rate)
    int   n_agents;       // fleet size for this phase
    float label;          // [0,1] fed to GlobalState.density_phase
};

// ════════════════════════════════════════════════════════════════════════════
// Episode configuration
// ════════════════════════════════════════════════════════════════════════════
struct EpisodeConfig {
    const CityConfig* city = nullptr;

    // Agent speed (m/step).  ~18 km/h = 5 m/s at 1 step/s.
    float speed_mps = 5.0f;

    // Phase schedule (executed in order).
    // Default: progressive stress — low → medium → high density, fleet shrinks.
    std::vector<DensityPhase> phases = {
        {  500, 0.02f,  8, 0.0f },   // low density,    8 agents
        { 1000, 0.08f,  6, 0.5f },   // medium density, 6 agents
        {  500, 0.20f,  4, 1.0f }    // high density,   4 agents  ← stress test
    };

    // ── Spatial task pattern ───────────────────────────────────────────────
    float cluster_prob     = 0.35f;  // P(task in hot zone vs uniform)
    int   n_hot_zones      = 4;      // hot-zone centers resampled each episode
    float hot_zone_radius  = 500.f;  // hot-zone radius in metres
    float same_origin_prob = 0.0f;   // P(pickup == previous delivery) — lifelong reuse

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
        int m = 0; for (const auto& p : phases) m = std::max(m, p.n_agents); return m;
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
