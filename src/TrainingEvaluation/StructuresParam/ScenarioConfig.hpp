#ifndef SCENARIO_CONFIG_HPP
#define SCENARIO_CONFIG_HPP

#include "Environment/Structure/EventStream.hpp"
#include <string>
#include <vector>

// One episode combination. Task and congestion each pick a temporal profile
// fd(x); the fleet picks an agent multiplier AM (fleet = round(10·SCE·AM)).
// Event counts are fixed by SCE·RM (EpisodeConfig), NOT by the scenario.
// density_mult stays for the standalone planning comparison test only.
struct EpisodeScenario {
    float           agents_mult        = 1.0f;   // AM
    std::string     label              = "normal";
    TemporalProfile task_profile       = TemporalProfile::Uniform;
    TemporalProfile congestion_profile = TemporalProfile::Uniform;
    float           density_mult       = 1.0f;   // planning test only (no-op = 1)
};

struct TaskRegime {
    TemporalProfile profile;
    std::string     name;
};
struct CongestionRegime {
    TemporalProfile profile;
    std::string     name;
};
struct FleetRegime {
    float       agents_mult;   // AM
    std::string name;
};

std::vector<EpisodeScenario> build_scenarios(
    const std::vector<TaskRegime>&       tasks,
    const std::vector<CongestionRegime>& congestions,
    const std::vector<FleetRegime>&      fleets);

std::vector<TaskRegime>       paper_task_regimes();
std::vector<CongestionRegime> paper_congestion_regimes();
std::vector<FleetRegime>      paper_fleet_regimes();

std::vector<EpisodeScenario> make_scenario_grid();

#endif // SCENARIO_CONFIG_HPP
