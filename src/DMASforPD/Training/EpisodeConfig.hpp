#ifndef EPISODE_CONFIG_HPP
#define EPISODE_CONFIG_HPP

#include "CityConfig.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"  // kGlobSz = 20
#include <array>
#include <string>

// ════════════════════════════════════════════════════════════════════════════
// Episode Configuration
// ════════════════════════════════════════════════════════════════════════════

// Arrival-rate phase (matches user spec: low → dense → very dense with limited agents)
struct DensityPhase {
    int   steps;         // duration of this phase in simulation steps
    float lambda;        // mean tasks/step (Poisson parameter)
    int   n_agents;      // fleet size during this phase
    float phase_norm;    // [0,1] label fed to GlobalState (0=low, 0.5=med, 1=high)
};

// Full episode definition
struct EpisodeConfig {
    const CityConfig* city = nullptr;

    // ── Fleet ──────────────────────────────────────────────────────────────
    float speed_mps = 5.0f;         // agent speed (m/step), ~18 km/h

    // ── Density phases (executed in order) ────────────────────────────────
    // Default: 3-phase schedule (low → medium → high density + reduced fleet)
    // matches the training curriculum in the user spec.
    std::vector<DensityPhase> phases = {
        { 500,  0.02f, 8,  0.0f },   // phase 1: low density,    8 agents
        { 1000, 0.08f, 6,  0.5f },   // phase 2: medium density, 6 agents
        { 500,  0.20f, 4,  1.0f }    // phase 3: high density,   4 agents (stress test)
    };

    // ── Task spatial pattern ───────────────────────────────────────────────
    float cluster_prob     = 0.35f;  // probability a task falls inside a hot zone
    int   n_hot_zones      = 4;      // number of hot zones sampled per episode
    float hot_zone_radius  = 500.f;  // hot zone radius in metres
    float large_task_prob  = 0.15f;  // probability of a large task (reserved, future)

    // ── Depot (optional warehouse) ─────────────────────────────────────────
    bool  use_depot        = false;  // if true, all pickups originate from depot_node
    osmium::object_id_type depot_node = 0;

    // ── Metrics collection ─────────────────────────────────────────────────
    bool  collect_metrics  = true;
    int   time_window_steps= 200;    // window for throughput metric

    // ── Convenience ───────────────────────────────────────────────────────
    int total_steps() const {
        int s = 0; for (const auto& p : phases) s += p.steps; return s;
    }
    int initial_agents() const {
        return phases.empty() ? 4 : phases.front().n_agents;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Global State Vector (critic input, kGlobSz = 20 floats)
// ════════════════════════════════════════════════════════════════════════════
//
// Built once per simulation step by the episode runner and stored alongside
// each buffered Experience so the centralised critic can estimate V(s).
//
// All fields are normalised to [0, 1] where possible.
struct GlobalState {
    // ── System state (8) ──────────────────────────────────────────────────
    float time_ratio        = 0.f; // current_step / total_steps
    float n_agents_ratio    = 0.f; // active_agents / max_agents_this_episode
    float avail_ratio       = 0.f; // available_tasks / total_tasks_created
    float alloc_ratio       = 0.f; // allocated_tasks / total_tasks_created
    float done_ratio        = 0.f; // finished_tasks / total_tasks_created
    float avg_load          = 0.f; // mean(tasks_per_agent) / kMaxLoad
    float max_load          = 0.f; // max(tasks_per_agent) / kMaxLoad
    float arrival_rate      = 0.f; // tasks_arrived_last_window / (lambda * window)

    // ── Performance (4) ───────────────────────────────────────────────────
    float throughput        = 0.f; // finished / total_tasks_created
    float avg_latency       = 0.f; // mean_completion_steps / time_window_steps
    float accept_rate       = 0.f; // accepted / (accepted + refused)
    float avg_efficiency    = 0.f; // mean agent route quality [0,1]

    // ── Episode context (4) ───────────────────────────────────────────────
    float city_id_norm      = 0.f; // city index / (num_cities - 1)
    float density_phase     = 0.f; // current phase label (0=low, 0.5=med, 1=high)
    float cluster_ratio     = 0.f; // fraction tasks in hot zones (running avg)
    float congestion_ratio  = 0.f; // placeholder: mean way congestion [0,1]

    // ── Reserved (4) ─────────────────────────────────────────────────────
    float pad0 = 0.f, pad1 = 0.f, pad2 = 0.f, pad3 = 0.f;

    static_assert(kGlobSz == 20, "GlobalState must match kGlobSz");

    void to_array(float* dst) const {
        dst[ 0] = time_ratio;     dst[ 1] = n_agents_ratio;
        dst[ 2] = avail_ratio;    dst[ 3] = alloc_ratio;
        dst[ 4] = done_ratio;     dst[ 5] = avg_load;
        dst[ 6] = max_load;       dst[ 7] = arrival_rate;
        dst[ 8] = throughput;     dst[ 9] = avg_latency;
        dst[10] = accept_rate;    dst[11] = avg_efficiency;
        dst[12] = city_id_norm;   dst[13] = density_phase;
        dst[14] = cluster_ratio;  dst[15] = congestion_ratio;
        dst[16] = pad0;           dst[17] = pad1;
        dst[18] = pad2;           dst[19] = pad3;
    }

    std::array<float, kGlobSz> to_array() const {
        std::array<float, kGlobSz> a{};
        to_array(a.data());
        return a;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Comparison Metrics  (collected per episode / per city)
// ════════════════════════════════════════════════════════════════════════════
//
// Covers all dimensions from the user's comparison spec:
//   Environment, VRP-related, MAPD-related, RL policy.
struct ComparisonMetrics {
    // ── Identification ─────────────────────────────────────────────────────
    std::string city_name;
    std::string method;          // "DMAS-DbVNS", "IVNS", "LKH", "MAPPO", …
    int         episode_id = 0;

    // ── Environment ────────────────────────────────────────────────────────
    int   total_steps          = 0;
    int   n_agents_min         = 0;
    int   n_agents_max         = 0;
    bool  has_depot            = false;
    // road network size (set from GeoBox at episode start)
    int   n_road_nodes         = 0;
    int   n_road_ways          = 0;
    float area_km2             = 0.f;

    // ── VRP-Related (Planning quality) ─────────────────────────────────────
    // Objective: minimize travel distance/time; constraint: P before D
    float avg_route_quality    = 0.f; // mean t_ratio (algo / oracle), lower = better
    float best_route_quality   = 0.f;
    int   pdp_violations       = 0;   // must be 0 for valid solutions

    // ── MAPD-Related (System efficiency) ──────────────────────────────────
    // Metric 1 — Task throughput within time window
    int   tasks_in_window      = 0;   // tasks completed inside time_window_steps
    int   tasks_total          = 0;   // tasks appeared during episode
    float throughput_rate      = 0.f; // tasks_in_window / tasks_total

    // Metric 2 — Completeness latency vs available agents
    float latency_mean_steps   = 0.f; // mean steps from task appearance to completion
    float latency_per_agent    = 0.f; // latency_mean / mean_active_agents
    float agent_utilization    = 0.f; // fraction of steps each agent is moving/working
    float tasks_refused        = 0.f; // fraction of tasks no agent accepted

    // ── RL Policy ──────────────────────────────────────────────────────────
    float policy_accept_rate   = 0.f; // fraction of offers accepted by policy
    float policy_reward_mean   = 0.f; // mean reward signal per completed task
    float policy_loss_actor    = 0.f; // last actor loss (training only)
    float policy_loss_critic   = 0.f; // last critic loss (training only)
    int   policy_buffer_size   = 0;   // experiences collected this episode

    // ── Accumulation helpers (called from episode runner) ──────────────────
    void accumulate_completion(int latency_steps, float quality);
    void accumulate_refusal();
    void accumulate_offer(bool accepted, float reward);
    void finalise(int n_agents_active, int total_steps_run);
};

#endif // EPISODE_CONFIG_HPP
