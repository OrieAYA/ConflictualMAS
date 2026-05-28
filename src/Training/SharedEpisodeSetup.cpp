#include "SharedEpisodeSetup.hpp"
#include "CityConfig.hpp"
#include "EpisodeGenerator.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace {

// Sample one valid (road-incident) node from the GeoBox. Mirrors the
// deterministic candidate collection used by SolverContext::sample_valid_node
// so SharedEpisodeSetup produces the same node IDs given the same rng state.
osmium::object_id_type sample_road_node(const GeoBox& geo_box,
                                         std::mt19937& rng) {
    if (geo_box.data.nodes.empty()) return 0;
    std::vector<osmium::object_id_type> candidates;
    candidates.reserve(geo_box.data.nodes.size() / 4);
    for (const auto& [id, pt] : geo_box.data.nodes)
        if (!pt.incident_ways.empty()) candidates.push_back(id);
    if (candidates.empty()) return 0;
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    return candidates[dist(rng)];
}

} // namespace

SharedEpisodeSetup build_shared_episode_setup(
    uint32_t              ep_seed,
    const CityConfig&     city,
    const EpisodeScenario& scenario,
    const EpisodeConfig&  cfg,
    const GeoBox&         geo_box)
{
    SharedEpisodeSetup setup;
    setup.ep_seed  = ep_seed;
    setup.city     = &city;
    setup.scenario = scenario;

    // ── 1. Task stream ─────────────────────────────────────────────────────
    // Generator is deterministic given ep_seed. We then apply density_mult by
    // subsample/supersample using a derived RNG (0xA17EFEEDu mask, same as the
    // legacy SolverRunner used) — that way the stream is the canonical one for
    // any solver/policy.
    {
        EpisodeGenerator gen(cfg, geo_box, ep_seed);
        setup.task_stream = gen.generate();

        if (scenario.density_mult != 1.0f && !setup.task_stream.empty()) {
            std::mt19937 rng(ep_seed ^ 0xA17EFEEDu);
            const int target = std::max<int>(
                1, static_cast<int>(std::round(setup.task_stream.size() *
                                                scenario.density_mult)));
            if (scenario.density_mult <= 1.0f) {
                std::shuffle(setup.task_stream.begin(), setup.task_stream.end(), rng);
                setup.task_stream.resize(static_cast<size_t>(target));
                std::sort(setup.task_stream.begin(), setup.task_stream.end(),
                          [](const ScheduledTask& a, const ScheduledTask& b) {
                              return a.arrival_step < b.arrival_step;
                          });
            } else {
                std::vector<ScheduledTask> scaled;
                scaled.reserve(static_cast<size_t>(target));
                const int max_step = setup.task_stream.back().arrival_step + 1;
                std::uniform_int_distribution<size_t> pick(
                    0, setup.task_stream.size() - 1);
                std::uniform_int_distribution<int> step_pick(
                    0, std::max(1, max_step - 1));
                for (int i = 0; i < target; ++i) {
                    ScheduledTask t = setup.task_stream[pick(rng)];
                    t.arrival_step = step_pick(rng);
                    scaled.push_back(t);
                }
                std::sort(scaled.begin(), scaled.end(),
                          [](const ScheduledTask& a, const ScheduledTask& b) {
                              return a.arrival_step < b.arrival_step;
                          });
                setup.task_stream = std::move(scaled);
            }
        }
    }

    // ── 2. Fleet sizing (peak agents × agents_mult) ────────────────────────
    {
        EpisodeGenerator gen(cfg, geo_box, ep_seed);
        const auto phases = gen.build_phase_table();
        int peak_agents = 1;
        int total_steps = 0;
        for (const auto& p : phases) {
            peak_agents = std::max(peak_agents,
                                    std::max(p.n_agents_start, p.n_agents_end));
            total_steps = std::max(total_steps, p.step_end);
        }
        setup.n_active_agents = std::max(
            1, static_cast<int>(std::round(peak_agents * scenario.agents_mult)));
        setup.total_steps = total_steps;
    }

    // Global capacity ceiling (homogeneous fallback).
    setup.max_capacity_per_agent = std::max(1, cfg.max_tasks_per_agent);

    // ── 3. Heterogeneous capacity (when enabled) ──────────────────────────
    // Canonical RNG = mt19937(ep_seed ^ 0xC0FFEE01u). The setup carries the
    // result vector so both runners use the SAME draws for the same agent
    // indices.
    if (cfg.enable_heterogeneous_capacity) {
        std::mt19937 rng(ep_seed ^ 0xC0FFEE01u);
        const int cmin = std::max(1, cfg.hetero_capacity_min);
        const int cmax = std::max(cmin, cfg.hetero_capacity_max);
        std::uniform_int_distribution<int> cap_dist(cmin, cmax);
        setup.per_agent_capacity.resize(static_cast<size_t>(setup.n_active_agents));
        for (int i = 0; i < setup.n_active_agents; ++i)
            setup.per_agent_capacity[i] = cap_dist(rng);
    }

    // ── 4. Agent start positions (canonical, per-agent sampling) ──────────
    // Canonical RNG = mt19937(ep_seed ^ 0xA9E47514u).
    {
        std::mt19937 rng(ep_seed ^ 0xA9E47514u);
        setup.agent_start_nodes.reserve(static_cast<size_t>(setup.n_active_agents));
        for (int i = 0; i < setup.n_active_agents; ++i)
            setup.agent_start_nodes.push_back(sample_road_node(geo_box, rng));
    }

    // ── 5. Ghost-traffic seed (single canonical formula) ──────────────────
    setup.ghost_seed = ep_seed ^ 0xBADCAFEEu;

    return setup;
}