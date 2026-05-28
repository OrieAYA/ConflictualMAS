#ifndef SOTA_SOLVER_METRICS_HPP
#define SOTA_SOLVER_METRICS_HPP

#include <string>

// ════════════════════════════════════════════════════════════════════════════
// SolverMetrics
// ════════════════════════════════════════════════════════════════════════════
//
// Unified metric record produced by every standalone SOTA solver at the end
// of an episode. The fields mirror ComparisonMetrics (Training/EpisodeConfig
// .hpp:441) where it makes sense — so a SoTA standalone run and a MAPPO/
// Hybrid run can be compared apples-to-apples.
//
// Throughput is the LGPDP comparison axis (tasks_completed / tasks_appeared).
// Latency, utilisation, congestion-exposure, load balance, and compute cost
// are secondary metrics for the multi-axis comparison.
//
// All metrics that don't apply to a given solver should remain at their
// default zero values rather than be unset — the CSV writer accepts both.

struct SolverMetrics {
    // ── Identity / context (filled by SolverRunner) ─────────────────────────
    std::string solver_name;
    std::string city_label;
    std::string scenario_label;
    int         episode      = 0;
    int         total_steps  = 0;
    int         n_active_agents = 0;

    // ── Headline LGPDP metrics ──────────────────────────────────────────────
    int    tasks_appeared  = 0;
    int    tasks_completed = 0;
    int    tasks_refused   = 0;
    float  throughput_rate = 0.f;
    float  accept_rate     = 0.f;

    // ── Time metrics (mean over completed tasks) ────────────────────────────
    double latency_mean       = 0.;
    double latency_per_agent  = 0.;
    double mean_wait_steps    = 0.;
    double mean_trip_steps    = 0.;
    double mean_road_pd_m     = 0.;

    // ── Routing diagnostics ─────────────────────────────────────────────────
    double mean_extra_steps_per_task     = 0.;
    double delivery_route_efficiency     = 0.;
    double total_fleet_distance_m        = 0.;

    // ── Agent metrics ───────────────────────────────────────────────────────
    float  agent_utilisation             = 0.f;
    int    agent_completed_max           = 0;
    int    agent_completed_min           = 0;
    float  agent_completed_gini          = 0.f;
    float  agent_completed_std           = 0.f;   // coefficient of variation

    // ── Network-level congestion ───────────────────────────────────────────
    float  mean_congestion               = 0.f;   // mean(load_now) over all steps
    float  congestion_variance           = 0.f;
    int    peak_load                     = 0;     // max load_now across episode
    float  n_ghost_active_mean           = 0.f;   // SolverRunner fills this

    // ── Congestion exposure (along agent traversals) ────────────────────────
    float  mean_congestion_at_decision   = 0.f;
    double mean_bpr_along_route          = 1.;
    double time_lost_to_congestion       = 0.;
    int    n_traversals_in_jam           = 0;
    float  route_congestion_exposure     = 0.f;

    // ── Compute complexity ──────────────────────────────────────────────────
    long long wallclock_ms               = 0;
    long long n_allocation_calls         = 0;     // # try_allocate / batch calls
    double    compute_time_per_task_ms   = 0.;
    double    compute_time_per_decision_us = 0.;

    // ── Solver-specific bookkeeping ─────────────────────────────────────────
    int    n_replans                     = 0;
    int    n_collisions_avoided          = 0;

    // ── Validity audit — must remain zero ───────────────────────────────────
    int    capacity_violations           = 0;
    int    pairing_violations            = 0;
};

#endif // SOTA_SOLVER_METRICS_HPP