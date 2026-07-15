#include "TrainingEvaluation/StructuresParam/ScenarioConfig.hpp"

// Three temporal distribution functions fd(x), shared by task and congestion:
//   Normal   fd(x)=1        Uniform density
//   ShockPick fd(x)=x^1.5   accelerating late surge
//   Wave     fd(x)=sin(πx)  single bell
std::vector<TaskRegime> paper_task_regimes() {
    return {
        { TemporalProfile::Uniform,   "task_normal" },
        { TemporalProfile::ShockPick, "task_shock"  },
        { TemporalProfile::Wave,      "task_wave"   },
    };
}

std::vector<CongestionRegime> paper_congestion_regimes() {
    return {
        { TemporalProfile::Uniform,   "cong_normal" },
        { TemporalProfile::ShockPick, "cong_shock"  },
        { TemporalProfile::Wave,      "cong_wave"   },
    };
}

std::vector<FleetRegime> paper_fleet_regimes() {
    return {
        { 0.7f, "agents_low"  },
        { 1.0f, "agents_mid"  },
        { 2.5f, "agents_high" },
    };
}

std::vector<EpisodeScenario> build_scenarios(
    const std::vector<TaskRegime>&       tasks,
    const std::vector<CongestionRegime>& congestions,
    const std::vector<FleetRegime>&      fleets)
{
    std::vector<EpisodeScenario> grid;
    grid.reserve(tasks.size() * congestions.size() * fleets.size());
    for (const auto& t : tasks)
        for (const auto& c : congestions)
            for (const auto& a : fleets) {
                EpisodeScenario s;
                s.agents_mult        = a.agents_mult;
                s.task_profile       = t.profile;
                s.congestion_profile = c.profile;
                s.label              = t.name + "/" + c.name + "/" + a.name;
                grid.push_back(std::move(s));
            }
    return grid;
}

std::vector<EpisodeScenario> make_scenario_grid() {
    return build_scenarios(paper_task_regimes(),
                           paper_congestion_regimes(),
                           paper_fleet_regimes());
}
