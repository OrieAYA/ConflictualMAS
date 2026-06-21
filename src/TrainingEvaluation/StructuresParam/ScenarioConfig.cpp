#include "TrainingEvaluation/StructuresParam/ScenarioConfig.hpp"

std::vector<EpisodeScenario> make_scenario_grid() {
    struct Task { float density, agents; const char* name; };
    struct Cong { CongestionProfile profile; float density; const char* name; };
    const Task tasks[3] = {
        { 0.7f, 1.2f, "task_easy"   },
        { 1.0f, 1.0f, "task_normal" },
        { 1.8f, 0.7f, "task_hard"   },
    };
    const Cong congs[3] = {
        { CongestionProfile::Flat,       1.0f, "cong_easy"   },
        { CongestionProfile::Wave,       2.0f, "cong_normal" },
        { CongestionProfile::ShockBurst, 4.0f, "cong_hard"   },
    };
    static const char* labels[9] = {
        "task_easy/cong_easy",   "task_easy/cong_normal",   "task_easy/cong_hard",
        "task_normal/cong_easy", "task_normal/cong_normal", "task_normal/cong_hard",
        "task_hard/cong_easy",   "task_hard/cong_normal",   "task_hard/cong_hard",
    };
    std::vector<EpisodeScenario> grid;
    grid.reserve(9);
    for (int t = 0; t < 3; ++t)
        for (int c = 0; c < 3; ++c)
            grid.push_back({ tasks[t].density, tasks[t].agents, labels[t*3+c],
                             congs[c].profile, congs[c].density });
    return grid;
}

