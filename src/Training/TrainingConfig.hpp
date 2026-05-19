#ifndef TRAINING_CONFIG_HPP
#define TRAINING_CONFIG_HPP

#include "EpisodeConfig.hpp"
#include "EpisodeRunner.hpp"   // for PolicyMode
#include <string>
#include <vector>

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

    // ── Generalisation eval (unseen cities) ───────────────────────────────
    // When true, comparison cities (Kyoto, Fukuoka, LA, ...) are loaded with
    // graceful skip and evaluated at the end of each seed on MAPPO +
    // TamAlwaysAccept + InsertionGreedy. When false, only train cities are
    // loaded and evaluated (faster, simpler — useful for Tokyo-only runs).
    bool enable_generalization = true;

    // ── Policy checkpoints ────────────────────────────────────────────────
    // load_policy triggers checkpoint loading for ALL three RL policies
    // (MAPPO, IPPO, MAPPER) at the start of each seed. The respective paths
    // must point to checkpoints saved by the same seed's save() call.
    // Leave a path empty to skip loading that policy and keep Xavier init.
    bool        load_policy                  = false;
    std::string policy_path                  = "";   // MAPPO          checkpoint (.bin)
    std::string ippo_policy_path             = "";   // IPPO           checkpoint (.bin)
    std::string mapper_policy_path           = "";   // MAPPER         checkpoint (.bin) — enhanced variant
    std::string faithful_mapper_policy_path  = "";   // FaithfulMAPPER checkpoint (.bin) — paper-faithful variant

    // Save a checkpoint after each seed's training.
    bool        save_policy   = true;

    // Eval-only mode: if true, skip training rounds entirely (n_rounds is ignored)
    // and run run_eval + run_stress_eval + run_generalize_eval directly with the
    // loaded weights. Requires load_policy=true with valid paths for the three
    // policies you want to evaluate. The non-RL baselines run regardless.
    bool        eval_only          = false;

    // Custom eval mode subset. If non-empty, OVERRIDES the default 12-mode array
    // in run_eval / run_stress_eval and the 8-mode array in run_generalize_eval.
    // Use this to run a fast targeted eval, e.g. {MAPPER, MAPPO, TamAlwaysAccept,
    // Greedy} for a 4-mode lightweight comparison. Empty = run all default modes.
    std::vector<PolicyMode> eval_modes;

    // Skip the stress eval entirely (saves ~1.5h on a 7-city run).
    bool        skip_stress_eval   = false;

    // Skip the generalisation eval on held-out cities (saves ~45 min).
    bool        skip_generalize_eval = false;

    // Skip ALL evaluation phases (periodic, final, stress, generalize) for a
    // pure training run. Use when eval is done separately via Option X/Y on
    // the saved checkpoints. Overrides eval_every and the skip_* flags above.
    // Training still saves checkpoints honouring save_policy + train_modes.
    bool        train_only          = false;

    // ── Training mode subset ──────────────────────────────────────────────
    // Which RL policies to actually TRAIN per round. Default = all three.
    // For a MAPPER-only run, set to { PolicyMode::MAPPER } — IPPO and MAPPO
    // skip their PPO update phase, no buffer writes, ~3× faster per round.
    // Independent of eval_modes (which controls what's evaluated, not trained).
    std::vector<PolicyMode> train_modes;  // empty = train MAPPO+IPPO+MAPPER

    // ── Scenario sampler version ──────────────────────────────────────────
    // Default (false) uses the 4-regime sampler: 15% slack, 35% normal,
    // 30% stress_light, 20% stress_heavy.
    // Set to true to disable the "slack" regime → 3-regime sampler with
    // ~53% normal / 29% stress_light / 18% stress_heavy distribution.
    // This 3-regime version REPLICATES THE MAY 17 BASELINE TRAINING exactly
    // (verified from the 360-episode log: 0 slack samples observed).
    bool disable_slack_regime = false;

    // ── Eval scenario sweep ───────────────────────────────────────────────
    // List of EpisodeScenarios to evaluate under in run_eval / run_generalize_eval.
    // EMPTY (default) → single normal scenario (current T/M/X behaviour preserved).
    // NON-EMPTY → each scenario is run sequentially for every (mode, city, eps)
    //             tuple. The scenario.label becomes the suffix on EpisodeRecord.phase
    //             (e.g. "eval_slack", "generalize_stress_heavy"), letting pandas
    //             split results by scenario × city × mode for the final SoTA table.
    // This is what option Y uses to cover the full scenario × mode × map cross.
    std::vector<EpisodeScenario> eval_scenarios;

    // ── Logging ───────────────────────────────────────────────────────────
    bool verbose              = true;
    int  log_every            = 1;         // log every N episodes

    // ── Derived helpers ───────────────────────────────────────────────────
    int total_train_episodes() const { return n_rounds * 7; } // 7 = all train cities
};

#endif // TRAINING_CONFIG_HPP
