#ifndef SCENARIO_CONFIG_HPP
#define SCENARIO_CONFIG_HPP

#include "Environment/Simulation/GhostTrafficController.hpp"   // CongestionProfile
#include <vector>

// One scenario = a task regime (density / fleet multipliers) combined with a
// congestion regime (ghost profile + intensity). ghost_density_per_hot_way > 0
// overrides EpisodeConfig's value so congestion is a per-scenario axis.
struct EpisodeScenario {
    float             density_mult              = 1.0f;
    float             agents_mult               = 1.0f;
    const char*       label                     = "normal";
    CongestionProfile congestion_profile        = CongestionProfile::Flat;
    float             ghost_density_per_hot_way = 0.f;
};

// 3 task levels × 3 congestion levels = 9 deterministic scenarios. Training and
// evaluation both walk this grid deterministically (no random sampling). Edit
// the two level tables in ScenarioConfig.cpp to retune difficulty.
std::vector<EpisodeScenario> make_scenario_grid();

#endif // SCENARIO_CONFIG_HPP
