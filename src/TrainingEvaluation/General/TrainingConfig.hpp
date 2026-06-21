#ifndef TRAINING_CONFIG_HPP
#define TRAINING_CONFIG_HPP

#include "TrainingEvaluation/General/EvaluationConfig.hpp"     // base: eval-only knobs
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include <string>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// TrainingConfig — full experimental protocol. Inherits the evaluation knobs
// from EvaluationConfig (cfg.eval_only etc. stay flat) and adds the training
// schedule, checkpointing, city filters, and scenario sampler.
// ════════════════════════════════════════════════════════════════════════════
struct TrainingConfig : EvaluationConfig {
    // ── Data paths ────────────────────────────────────────────────────────
    std::string cache_root = "data/cache";   // GeoBox JSON at {cache_root}/{city}.json
    std::string output_dir = "results";      // CSVs, checkpoints, timing logs

    EpisodeConfig episode_cfg;               // episode parameters, shared across cities

    // ── Training schedule ─────────────────────────────────────────────────
    // Each round runs one episode per train city. Total = n_rounds × 7 × n_seeds.
    int n_rounds   = 200;   // rounds of multi-city training
    int n_seeds    = 3;     // independent seeds for significance
    int start_seed = 42;    // base seed; trainer uses s + start_seed for seed s
    int eval_every = 20;    // evaluate every N rounds during training (plus once at end)

    // ── Policy checkpoints ────────────────────────────────────────────────
    // load_policy loads checkpoints for MAPPO/IPPO/MAPPER at each seed start;
    // an empty path skips that policy (keeps Xavier init).
    bool        load_policy                 = false;
    std::string policy_path                 = "";  // MAPPO
    std::string ippo_policy_path            = "";  // IPPO
    std::string mapper_policy_path          = "";  // MAPPER (enhanced)
    std::string faithful_mapper_policy_path = "";  // FaithfulMAPPER (paper-faithful)
    bool        save_policy             = true;    // checkpoint after each seed
    int         checkpoint_every_rounds = 0;       // intermediate cadence; 0 = none

    // Pure training run: skip all eval phases (overrides eval_every + the skip_*
    // flags in EvaluationConfig). Checkpoints still honour save_policy + train_modes.
    bool train_only = false;

    // ── Training mode subset ── empty = train MAPPO+IPPO+MAPPER.
    std::vector<PolicyMode> train_modes;

    // ── City subsets ── empty = all cities from CityRegistry.
    std::vector<std::string> train_city_filter;  // training cities
    std::vector<std::string> gen_city_filter;    // held-out generalisation cities

    // ── Scenario sampler ──────────────────────────────────────────────────
    // false = 4-regime sampler (15% slack / 35% normal / 30/20 stress).
    // true  = drop the slack regime → 3-regime (replicates the May-17 baseline).
    bool disable_slack_regime = false;

    // Deterministic training scenario grid. Empty = make_scenario_grid() (9
    // task×congestion combos), walked across (round, city) slots.
    std::vector<EpisodeScenario> train_scenarios;

    // ── Logging ───────────────────────────────────────────────────────────
    bool verbose   = true;
    int  log_every = 1;     // log every N episodes

    int total_train_episodes() const { return n_rounds * 7; }  // 7 = all train cities
};

#endif // TRAINING_CONFIG_HPP
