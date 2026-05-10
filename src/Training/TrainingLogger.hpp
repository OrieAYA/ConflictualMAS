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
