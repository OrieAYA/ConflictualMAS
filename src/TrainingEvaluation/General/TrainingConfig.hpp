#ifndef TRAINING_CONFIG_HPP
#define TRAINING_CONFIG_HPP

#include "TrainingEvaluation/General/EvaluationConfig.hpp"     // base: eval-only knobs
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include <string>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// TrainingConfig — full experimental protocol. Inherits the evaluation knobs
// from EvaluationConfig (cfg.eval_modes etc. stay flat) and adds the training
// grid, checkpointing, city filters and scenario grid.
// ════════════════════════════════════════════════════════════════════════════
struct TrainingConfig : EvaluationConfig {
    // ── Data paths ────────────────────────────────────────────────────────
    std::string cache_root = "data/cache";   // GeoBox JSON at {cache_root}/{city}.json
    std::string output_dir = "results";      // CSVs, checkpoints, timing logs

    EpisodeConfig episode_cfg;               // episode parameters, shared across cities

    // ── Seeds ─────────────────────────────────────────────────────────────
    // train_grid: one independent policy per seed (seed s trains on seed
    // start_seed + s). evaluate: one eval pass per seed.
    int n_seeds    = 3;
    int start_seed = 42;

    // ── Policy checkpoints ────────────────────────────────────────────────
    // train_grid writes {output_dir}/{mappo,ippo,mapper}/*_seed{seed}.bin,
    // rewritten after every episode. evaluate() loads the paths below
    // (empty path = skip that policy; an RL mode in eval_modes with an empty
    // path aborts).
    bool        save_policy        = true;
    // train_grid: reprend automatiquement un run interrompu depuis
    // {output_dir}/episodes_train.csv + les checkpoints par seed.
    bool        resume             = true;
    std::string policy_path        = "";   // MAPPO checkpoint to evaluate
    std::string ippo_policy_path   = "";   // IPPO
    std::string mapper_policy_path = "";   // MAPPER
    std::string lsm_path           = "";   // frozen LSM checkpoint (train_movement)

    // ── Training mode subset ── empty = train MAPPO+IPPO+MAPPER.
    std::vector<PolicyMode> train_modes;

    // ── City subset ── empty = all TrainAndApply cities from CityRegistry.
    std::vector<std::string> train_city_filter;

    // ── Scenario grid ── empty = make_scenario_grid() (27 combos).
    std::vector<EpisodeScenario> train_scenarios;

    // ── Logging ───────────────────────────────────────────────────────────
    bool verbose   = true;
    int  log_every = 1;     // log every N episodes
};

#endif // TRAINING_CONFIG_HPP
