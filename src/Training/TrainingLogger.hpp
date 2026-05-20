#ifndef TRAINING_LOGGER_HPP
#define TRAINING_LOGGER_HPP

#include "EpisodeConfig.hpp"
#include "EpisodeRunner.hpp"
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
// Writes one CSV per experimental run (seed).  The CSV has a header row
// followed by one EpisodeRecord per line.
//
// Usage:
//   TrainingLogger log("results/", seed);
//   log.push(record);          // call after each episode
//   log.flush();               // periodic or at the end
//   log.write_summary("results/summary.csv");  // aggregate across all seeds
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

#endif // TRAINING_LOGGER_HPP
