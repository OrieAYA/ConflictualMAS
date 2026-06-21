#ifndef EVALUATION_CONFIG_HPP
#define EVALUATION_CONFIG_HPP

#include "TrainingEvaluation/Run/Runner.hpp"                     // PolicyMode
#include "TrainingEvaluation/StructuresParam/ScenarioConfig.hpp" // EpisodeScenario
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// EvaluationConfig — evaluation-only knobs.
//
// Base of TrainingConfig: a training run reads these inherited fields flat
// (cfg.eval_only, cfg.eval_modes, ...), while a standalone evaluation can take
// an EvaluationConfig& without dragging in the training schedule.
// ════════════════════════════════════════════════════════════════════════════
struct EvaluationConfig {
    int  n_eval_episodes      = 10;     // eval episodes per (city × mode)
    bool enable_generalization = true;  // evaluate held-out cities at each seed end

    // Skip training rounds and evaluate loaded weights directly (needs
    // load_policy=true). Non-RL baselines run regardless.
    bool eval_only = false;

    // Mode subset for run_eval / run_stress_eval / run_generalize_eval.
    // Empty = all default modes.
    std::vector<PolicyMode> eval_modes;

    bool skip_stress_eval     = false;  // skip the stress eval
    bool skip_generalize_eval = false;  // skip the held-out generalisation eval

    // Build a canonical SharedEpisodeSetup per (city, scenario, episode) so the
    // RL-vs-SoTA comparison is byte-identical. Off = training byte-for-byte.
    bool use_shared_episode_setup = false;

    // Scenarios swept in run_eval / run_generalize_eval. Empty = single normal
    // scenario (legacy T/M/X behaviour).
    std::vector<EpisodeScenario> eval_scenarios;
};

#endif // EVALUATION_CONFIG_HPP
