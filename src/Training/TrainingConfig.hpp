#ifndef TRAINING_CONFIG_HPP
#define TRAINING_CONFIG_HPP

#include "EpisodeConfig.hpp"
#include <string>

// ════════════════════════════════════════════════════════════════════════════
// TrainingConfig — full experimental protocol parameters
// ════════════════════════════════════════════════════════════════════════════
struct TrainingConfig {
    // ── Data paths ────────────────────────────────────────────────────────
    // GeoBox JSON caches are expected at: {cache_root}/{city_name}.json
    std::string cache_root   = "data/cache";
    // All output (CSVs, policy checkpoints, timing logs) goes here.
    std::string output_dir   = "results";

    // ── Episode parameters ────────────────────────────────────────────────
    EpisodeConfig episode_cfg;              // shared across all cities

    // ── Training schedule ─────────────────────────────────────────────────
    // Each "round" runs one episode per train city (round-robin, all 7).
    // Total training episodes = n_rounds × 7 cities × n_seeds.
    int n_rounds            = 200;         // rounds of multi-city training
    int n_eval_episodes     = 10;          // eval episodes per city × mode
    int n_seeds             = 3;           // independent seeds for significance

    // Evaluation is run after every eval_every rounds during training,
    // plus once at the very end.
    int eval_every          = 20;

    // ── Policy checkpoint ─────────────────────────────────────────────────
    bool        load_policy   = false;
    std::string policy_path   = "";        // path to .bin checkpoint to load

    // Save a checkpoint after each seed's training.
    bool        save_policy   = true;

    // ── Logging ───────────────────────────────────────────────────────────
    bool verbose              = true;
    int  log_every            = 1;         // log every N episodes

    // ── Derived helpers ───────────────────────────────────────────────────
    int total_train_episodes() const { return n_rounds * 7; } // 7 = all train cities
};

#endif // TRAINING_CONFIG_HPP
