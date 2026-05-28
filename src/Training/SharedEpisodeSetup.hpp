#ifndef TRAINING_SHARED_EPISODE_SETUP_HPP
#define TRAINING_SHARED_EPISODE_SETUP_HPP

#include "EpisodeConfig.hpp"
#include "EpisodeGenerator.hpp"
#include "EpisodeRunner.hpp"           // EpisodeScenario
#include "Environment/GeoBox/Box.hpp"
#include <cstdint>
#include <osmium/osm/types.hpp>
#include <vector>

struct CityConfig;

// ════════════════════════════════════════════════════════════════════════════
// SharedEpisodeSetup
// ════════════════════════════════════════════════════════════════════════════
//
// CANONICAL, deterministic snapshot of everything that must be byte-identical
// across methods in a cross-method publication-grade evaluation (Option O):
//
//   - task stream (pre-generated, density_mult already applied)
//   - agent start positions (per-agent sampling, seed-deterministic)
//   - per-agent carrying capacity (when heterogeneity enabled)
//   - fleet sizing (n_active_agents after agents_mult scaling)
//   - ghost-traffic seed (single canonical formula)
//   - simulation horizon and global capacity ceiling
//
// Built ONCE per (city, scenario, episode) by build_shared_episode_setup().
// Both EpisodeRunner (RL policies) and SolverRunner (SoTA standalone solvers)
// consume the same SharedEpisodeSetup and SKIP their internal RNG-driven
// re-derivations — guaranteeing the cross-method comparison is fair.
//
// Without this, the two runners use DIFFERENT formulas for ghost seed,
// hetero-capacity RNG, density_mult application, and agent positions, which
// makes any cross-method gap a confound of "method × environment" rather than
// a pure "method" effect.
struct SharedEpisodeSetup {
    uint32_t                            ep_seed = 0;
    const CityConfig*                   city    = nullptr;
    EpisodeScenario                     scenario;

    std::vector<ScheduledTask>          task_stream;
    std::vector<osmium::object_id_type> agent_start_nodes;
    std::vector<int>                    per_agent_capacity;  // empty = homogeneous
    int                                 n_active_agents        = 0;
    int                                 max_capacity_per_agent = 1;
    int                                 total_steps            = 0;
    uint32_t                            ghost_seed             = 0;
};

// Build a canonical SharedEpisodeSetup deterministically from (ep_seed, city,
// scenario, cfg, geo_box). The cfg should ALREADY have per-city customisation
// applied (e.g. via MultiCityTrainer::customize_episode_for_city) — this
// function does NOT re-run customisation.
SharedEpisodeSetup build_shared_episode_setup(
    uint32_t              ep_seed,
    const CityConfig&     city,
    const EpisodeScenario& scenario,
    const EpisodeConfig&  cfg,
    const GeoBox&         geo_box);

#endif // TRAINING_SHARED_EPISODE_SETUP_HPP