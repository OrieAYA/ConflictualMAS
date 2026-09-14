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

    // Crash-resume: skip (city, scenario, episode, mode) rows whose
    // global_episode index (0-based, as logged in EpisodeRecord) is strictly
    // less than this value — they're assumed already logged by a prior run.
    // A scenario's standalone SoTA pass (CA/HAPC) is skipped only when the
    // ENTIRE scenario's mode block lies before this threshold; it reruns for
    // the scenario straddling the resume point (its modes weren't all done,
    // so its SoTA pass wasn't reached either) and every scenario after it.
    // 0 = no resume, run everything (default, fully backward compatible).
    int resume_from_episode = 0;
};

#endif // EVALUATION_CONFIG_HPP
