#ifndef MULTI_CITY_TRAINER_HPP
#define MULTI_CITY_TRAINER_HPP

#include "TrainingEvaluation/General/TrainingConfig.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/Run/Runner.hpp"
#include <string>
#include <fstream>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// EpisodeRecord — one CSV row, captures everything needed for analysis
// ════════════════════════════════════════════════════════════════════════════
struct EpisodeRecord {
    // Experimental identity
    int         seed           = 0;
    int         global_episode = 0;   // monotonically increasing across seeds
    std::string city;
    std::string phase           = "train"; // "train" | "eval"
    std::string policy_mode     = "MAPPO"; // "MAPPO" | "Greedy" | "Random"

    // Environment
    int   total_steps          = 0;
    int   n_agents_max         = 0;

    // MAPD performance (from ComparisonMetrics)
    int   tasks_appeared       = 0;
    int   tasks_completed      = 0;
    float throughput_rate      = 0.f;
    float accept_rate          = 0.f;
    float refuse_rate          = 0.f;
    float latency_mean         = 0.f;    // steps appearance → completion
    float latency_per_agent    = 0.f;
    float agent_utilisation    = 0.f;
    float mean_congestion           = 0.f;  // mean edge load over episode steps
    float mean_trip_steps           = 0.f;  // mean pickup→delivery steps
    float mean_wait_steps           = 0.f;  // mean appearance → pickup arrival (response delay)
    float mean_road_pd_m            = 0.f;  // mean A* road distance pickup→delivery (metres)
    float delivery_route_efficiency = 0.f;  // road_pd_m / (trip_steps × speed): ∈(0,1]

    // Spatial complexity (over served pickup+delivery points)
    float bbox_area_km2          = 0.f;
    float convex_hull_area_km2   = 0.f;
    float mean_pd_distance_m     = 0.f;
    float mean_nn_pickup_m       = 0.f;

    // Derived temporal complexity
    float compute_time_per_task_ms     = 0.f;
    float compute_time_per_decision_us = 0.f;

    // Validity counters (should stay 0)
    int   pairing_violations_runtime  = 0;
    int   capacity_violations_runtime = 0;

    // ── Selectivity diagnostics (Option M / RL utility analysis) ───────────
    // Surfaces the *quality* of accept/refuse choices instead of just volume.
    float       completion_per_accepted     = 0.f;
    float       unfinished_accept_rate      = 0.f;
    float       mean_congestion_at_decision = 0.f;
    float       n_ghost_active_mean         = 0.f;
    std::string congestion_profile_label;

    // Multi-axis performance diagnostics (Option X)
    float agent_completed_gini      = 0.f;
    float agent_completed_std       = 0.f;
    float mean_imp_accepted         = 0.f;
    float mean_imp_refused          = 0.f;
    float accept_rate_high_cong     = 0.f;
    float accept_rate_low_cong      = 0.f;
    float mean_extra_steps_per_task = 0.f;

    // Network-level congestion impact (policy-attributable analysis)
    int   peak_congestion          = 0;
    float mean_overlap_edges       = 0.f;
    float congestion_variance      = 0.f;
    float route_congestion_exposure = 0.f;
    int   max_agent_completed      = 0;
    int   min_agent_completed      = 0;
    float total_fleet_distance_m   = 0.f;

    // TAM efficiency (paper's "minimize communication overhead" claim)
    float mean_agents_offered_per_task    = 0.f;
    float mean_recall_rounds_per_task     = 0.f;
    float mean_candidates_scored_per_task = 0.f;

    // Selection intelligence (delivery quality, not just count)
    float value_throughput_rate = 0.f;
    float mean_completion_value = 0.f;
    float value_loss_to_refusal = 0.f;

    // Real impact on edge traversal (BPR factors paid by agents)
    float mean_bpr_along_route          = 1.f;
    float time_lost_to_congestion_steps = 0.f;
    int   n_traversals_in_jam           = 0;

    // Allocation optimality vs MCA full-scan oracle
    float marginal_cost_ratio_vs_oracle = 1.f;

    // RMCA(r) regret diagnostics [Chen et al. 2021] (RMCA mode only, else 0)
    float rmca_relative_regret  = 0.f;
    float rmca_marginal_cost_k1 = 0.f;
    float rmca_marginal_cost_k2 = 0.f;

    // Temporal complexity (allocation cost only)
    float mean_allocation_time_us = 0.f;
    float mean_tam_dijkstra_steps = 0.f;
    // Path-compute breakdown (ms units, see EpisodeConfig.hpp for semantics).
    float path_compute_time_ms    = 0.f;
    float mean_pure_alloc_time_ms = 0.f;

    // MAPPO training signal (zeroed for non-MAPPO or eval mode)
    float actor_loss           = 0.f;
    float critic_loss          = 0.f;
    float entropy              = 0.f;
    float adv_std              = 0.f;
    int   n_experiences        = 0;

    // Runtime
    long long wallclock_ms     = 0;
};

// ════════════════════════════════════════════════════════════════════════════
// TrainingLogger
// ════════════════════════════════════════════════════════════════════════════
//
// Writes one CSV per run (seed): a header then one EpisodeRecord per line.
// write_summary() appends a cross-seed mean ± std row.
class TrainingLogger {
public:
    TrainingLogger(const std::string& output_dir, int seed);
    ~TrainingLogger();

    void push(const EpisodeRecord& r);
    void flush();

    // Append a summary row (mean ± std) to a shared cross-seed CSV.
    // Call once per seed after all episodes are done.
    static void write_summary(const std::string& path,
                              const std::vector<EpisodeRecord>& records,
                              int seed);

    const std::vector<EpisodeRecord>& records() const { return records_; }

private:
    std::ofstream              file_;
    std::vector<EpisodeRecord> records_;
    bool                       header_written_ = false;

    void write_header();
    void write_row(const EpisodeRecord& r);
};

// Convenience: build an EpisodeRecord from RunResult + context.
EpisodeRecord make_record(const RunResult& result,
                          int seed, int global_episode,
                          const std::string& city,
                          const std::string& phase,
                          const std::string& policy_mode,
                          int n_agents_max);

#include "TrainingEvaluation/StructuresParam/CityConfig.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <memory>

// ════════════════════════════════════════════════════════════════════════════
// CityAssets — heap-stable resources for one city
// ════════════════════════════════════════════════════════════════════════════
//
// Pathfinder holds GeoBox by reference and EpisodeRunner holds const
// EpisodeConfig& → CityAssets must never move/copy after construction
// (allocate as unique_ptr). ep_cfg is the stable referent for the runner.
struct CityAssets {
    const CityConfig* config = nullptr;
    int               index  = 0;
    EpisodeConfig     ep_cfg;       // stable referent for EpisodeRunner
    GeoBox            geo_box;
    Pathfinder        pathfinder;   // holds GeoBox& geo_box above

    CityAssets(const CityConfig* cc, int idx, EpisodeConfig ep, GeoBox&& gb)
        : config(cc), index(idx), ep_cfg(std::move(ep)),
          geo_box(std::move(gb)), pathfinder(geo_box) {}

    CityAssets(const CityAssets&)            = delete;
    CityAssets& operator=(const CityAssets&) = delete;
    CityAssets(CityAssets&&)                 = delete;
    CityAssets& operator=(CityAssets&&)      = delete;
};

// ════════════════════════════════════════════════════════════════════════════
// MultiCityTrainer
// ════════════════════════════════════════════════════════════════════════════
//
// Full protocol: load all city graphs once; per seed → re-init MAPPO weights
// (optionally load a checkpoint), round-robin train over the train cities for
// n_rounds, periodic + final eval on all cities × policy modes, log each episode
// to per-seed CSV + cross-seed summary, save the checkpoint. One EpisodeRunner
// is kept per city across a seed so its A* path cache warms up; city graphs are
// shared across seeds, runners (and caches) recreated per seed.
class MultiCityTrainer {
public:
    void train(const TrainingConfig& cfg);

    // Public helper exposing the per-city EpisodeConfig customisation used
    // internally by train() (phases sized to area_km2, task distance range,
    // ghost_n_max tier, max_tasks_per_agent). Option O's Phase B calls this
    // so the SoTA standalone solvers face the EXACT same episode shape as
    // the RL policies in Phase A.
    static void customize_episode_for_city(EpisodeConfig& ep,
                                            const CityConfig& cc);

private:
    static std::unique_ptr<CityAssets> load_city(const CityConfig& cc, int idx,
                                                  EpisodeConfig ep,
                                                  const std::string& cache_root);

    static int run_eval(const TrainingConfig& cfg,
                        const std::vector<std::unique_ptr<CityAssets>>& assets,
                        std::vector<std::unique_ptr<EpisodeRunner>>& runners,
                        int global_ep, int seed,
                        TrainingLogger& logger);

    static int run_generalize_eval(const TrainingConfig& cfg,
                                   const std::vector<std::unique_ptr<CityAssets>>& gen_assets,
                                   std::vector<std::unique_ptr<EpisodeRunner>>& gen_runners,
                                   int global_ep, int seed,
                                   TrainingLogger& logger);

    // Stress evaluation: forced over-saturation scenario (density 2.0, agents 0.6×).
    // The system cannot deliver all tasks → tests whether the learned policy
    // refuses intelligently or collapses (always-accept agents waste capacity
    // on undeliverables, smart policies reserve capacity for feasible tasks).
    static int run_stress_eval(const TrainingConfig& cfg,
                               const std::vector<std::unique_ptr<CityAssets>>& assets,
                               std::vector<std::unique_ptr<EpisodeRunner>>& runners,
                               int global_ep, int seed,
                               TrainingLogger& logger);
};

#endif // MULTI_CITY_TRAINER_HPP
