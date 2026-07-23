#ifndef TRAINING_EPISODE_METRICS_HPP
#define TRAINING_EPISODE_METRICS_HPP

#include <string>
#include <memory>
#include <vector>

class PDPGlobalMemory;
class DeliveryAgent;
struct EpisodeConfig;

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
    float accept_rate_high_cong   = 0.f;  // 1 if any accept at load ≥ 1.0
    float accept_rate_low_cong    = 0.f;  // 1 if any accept at load < 1.0
    float mean_extra_steps_per_task = 0.f; // trip_steps − ideal_steps (detour)

    // ── Network-level congestion impact (policy-attributable) ──────────────
    int   peak_congestion         = 0;    // max load_now on any edge
    float mean_overlap_edges      = 0.f;  // mean #edges with load ≥ 2 / step
    float congestion_variance     = 0.f;  // std of mean_load_now across steps
    float route_congestion_exposure = 0.f; // mean load on edges agents traverse
    int   max_agent_completed     = 0;
    int   min_agent_completed     = 0;
    float total_fleet_distance_m  = 0.f;  // Σ metres actually driven by the fleet

    // ── Selection intelligence (value, not just count) ─────────────────────
    float value_throughput_rate = 0.f;  // Σvalue(delivered) / Σvalue(appeared)
    float mean_completion_value = 0.f;  // Σvalue(delivered) / N_delivered
    float value_loss_to_refusal = 0.f;  // Σvalue(refused)

    // ── Real impact on edge traversal times (ghost load → BPR → slowdown) ──
    float mean_bpr_along_route          = 1.f;  // mean dynamic/static over traversals
    float time_lost_to_congestion_steps = 0.f; // Σ (dynamic-static)/speed
    int   n_traversals_in_jam           = 0;    // traversals entered at load ≥ 5

    // ── Allocation optimality vs full-scan cheapest-insertion oracle ───────
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
};

// ════════════════════════════════════════════════════════════════════════════
// Part 1 — COLLECTION. Per-episode accumulators, filled DURING the simulation.
// ════════════════════════════════════════════════════════════════════════════
//
// One instance lives in the EpisodeRunner; reset() at episode start, then the
// simulation loop and the movement/reward hooks write into it as events happen.
// Pure accumulation: no aggregation logic here — that is Part 2's job
// (finalize_episode_metrics below), keeping the two concerns separate.
struct EpisodeMetricsCollector {
    // ── Offers / decisions ─────────────────────────────────────────────────
    int    n_accepted = 0;
    int    n_refused  = 0;      // policy refusals only (Format B retry paths)
    int    n_no_candidate = 0;  // TAM/solver found no valid agent — not a refusal
    double imp_accepted_sum = 0.0;  int imp_accepted_n = 0;
    double imp_refused_sum  = 0.0;  int imp_refused_n  = 0;
    int    accepts_high_cong = 0;
    int    accepts_low_cong  = 0;
    double congestion_at_decision_sum = 0.0;  int congestion_at_decision_count = 0;
    double value_appeared_sum  = 0.0;
    double value_delivered_sum = 0.0;
    double value_refused_sum   = 0.0;

    // ── Completion / latency (filled at pickup / delivery events) ──────────
    long   latency_sum = 0;   int latency_count = 0;
    long   trip_sum    = 0;   int trip_count    = 0;
    long   wait_sum    = 0;   int wait_count    = 0;
    double road_pd_sum = 0.0; int road_pd_count = 0;
    // Metres actually driven by the fleet (Σ edge length over all traversals),
    // sampled at edge entry in schedule_next_edge. Distinct from road_pd_sum,
    // which is the Σ of ideal P→D distances over completed tasks.
    double fleet_distance_m = 0.0;
    double efficiency_sum = 0.0; int efficiency_count = 0;

    // ── Fleet activity (per-step sampling) ─────────────────────────────────
    int active_sum = 0;  int active_steps = 0;

    // ── Network congestion (per-step sampling) ─────────────────────────────
    double congestion_sum = 0.0;  int congestion_steps = 0;
    int    peak_load_episode = 0;
    long   overlap_edges_sum = 0;  int overlap_steps = 0;
    double load_now_sum_sq  = 0.0;   // for variance: Σ load²
    double load_now_sum_lin = 0.0;   //               Σ load
    int    load_now_n       = 0;     //               N
    double route_exposure_sum = 0.0;  int route_exposure_n = 0;

    // ── Edge-traversal impact (sampled at edge entry) ──────────────────────
    double bpr_along_route_sum = 0.0;  int bpr_along_route_count = 0;
    double time_lost_to_congestion_sum = 0.0;   // in steps
    int    n_traversals_in_jam = 0;

    // ── TAM / allocation cost (one sample per offer_task call) ─────────────
    long      tam_agents_offered_sum    = 0;
    long      tam_recall_rounds_sum     = 0;
    long      tam_candidates_scored_sum = 0;
    int       tam_offer_samples         = 0;
    long long allocation_time_us_sum = 0;  int allocation_time_count = 0;
    long long tam_dijkstra_steps_sum = 0;  int tam_dijkstra_count    = 0;
    long long pure_alloc_time_us_sum = 0;
    long long path_compute_time_us_ep = 0;

    // ── Allocation optimality / RMCA(r) regret ─────────────────────────────
    double marginal_ratio_sum = 0.0;  int marginal_ratio_count = 0;
    double rmca_regret_sum = 0.0, rmca_mc_k1_sum = 0.0, rmca_mc_k2_sum = 0.0;
    int    rmca_regret_count = 0;

    // Zero every accumulator (episode start).
    void reset() { *this = EpisodeMetricsCollector{}; }

    // One offer_task instrumentation sample (shared by arrival + retry loops).
    void on_offer_sample(int agents_offered, int recall_rounds,
                         int candidates_scored, long long alloc_us,
                         long long pure_alloc_us, int tam_dijkstra_steps);

    // One accept/refuse decision outcome (shared by arrival + retry loops).
    void on_accept(float importance, bool high_cong);

    // Oracle ratio (chosen / cheapest marginal cost) for an allocation, plus
    // the RMCA(r) relative-regret sample when rmca_mode is set
    // [Chen et al. 2021, eq.13/14/16 — mc(k2)/mc(k1), clamped to [1, 50]].
    void on_allocation_choice(const std::vector<float>& pre_marginal_costs,
                              int chosen_agent_index, bool rmca_mode);
};

// ════════════════════════════════════════════════════════════════════════════
// Part 2 — RETRIEVAL. End-of-episode aggregation: collector → ComparisonMetrics.
// ════════════════════════════════════════════════════════════════════════════
//
// Consumes the collector plus the final system state (task lists, agents) and
// computes every derived metric: rates, means, Gini/CoV load balance, spatial
// complexity of served tasks, and the pairing/capacity validity audit.
// compute_time_per_* need the wallclock and are filled by the caller after.
ComparisonMetrics finalize_episode_metrics(
    const EpisodeMetricsCollector&                     c,
    const PDPGlobalMemory&                             memory,
    const std::vector<std::unique_ptr<DeliveryAgent>>& agents,
    const EpisodeConfig&                               cfg,
    int                                                total_steps,
    int                                                episode_fleet_size,
    bool                                               ghost_on,
    float                                              ghost_mean_active,
    const char*                                        congestion_profile);

#endif // TRAINING_EPISODE_METRICS_HPP
