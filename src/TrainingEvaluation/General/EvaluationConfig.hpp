#ifndef EVALUATION_CONFIG_HPP
#define EVALUATION_CONFIG_HPP

#include "TrainingEvaluation/Run/Runner.hpp"                     // PolicyMode
#include "TrainingEvaluation/StructuresParam/ScenarioConfig.hpp" // EpisodeScenario
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// EvaluationConfig — evaluation-only knobs.
//
// Base of TrainingConfig: a training run reads these inherited fields flat
// (cfg.eval_modes, cfg.eval_scenarios, ...), while a standalone evaluation can take
// an EvaluationConfig& without dragging in the training schedule.
// ════════════════════════════════════════════════════════════════════════════
struct EvaluationConfig {
    int n_eval_episodes = 10;           // eval episodes per (city × scenario)

    // Mode subset for run_eval. Empty = all default modes.
    std::vector<PolicyMode> eval_modes;

    // Scenarios swept in run_eval. Empty = single normal scenario.
    std::vector<EpisodeScenario> eval_scenarios;
};

#endif // EVALUATION_CONFIG_HPP
