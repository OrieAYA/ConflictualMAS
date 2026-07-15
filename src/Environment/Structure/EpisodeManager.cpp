#include "Environment/Structure/EpisodeManager.hpp"
#include "TrainingEvaluation/StructuresParam/CityConfig.hpp"
#include "Environment/Structure/Episode.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace {

// Process-scoped cache of "road-incident node" candidate vectors. Keyed by
// GeoBox pointer, with a content guard (node_count) to detect stack-frame
// reuse when main.cpp's city for-loop destroys/recreates GeoBox at the same
// address.
//
// CRITICAL HOMOGENEITY NOTE: we DO NOT sort the candidates vector. The
// previous (uncached) implementation iterated geo_box.data.nodes directly,
// producing candidates in unordered_map iteration order. That order is
// stable across calls for a given binary build → the agent_start_nodes
// produced by build_shared_episode_setup are byte-identical to those used
// by the prior TP/RL runs whose CSVs we want to join with HAPC's results.
// Sorting here would silently shuffle the sample index → different agent
// starts → broken homogeneity. Don't reintroduce the sort.
struct ValidNodesCacheEntry {
    size_t node_count = 0;
    std::vector<osmium::object_id_type> valid_nodes;
};
static std::unordered_map<const GeoBox*, ValidNodesCacheEntry>
    s_valid_nodes_cache;

const std::vector<osmium::object_id_type>&
get_valid_nodes(const GeoBox& geo_box) {
    auto& entry = s_valid_nodes_cache[&geo_box];
    if (entry.node_count != geo_box.data.nodes.size()) {
        // Either uncached or stale (same address, different city's data —
        // happens when main.cpp's stack-local geo_box gets reused).
        entry.node_count = geo_box.data.nodes.size();
        entry.valid_nodes.clear();
        entry.valid_nodes.reserve(geo_box.data.nodes.size() / 4);
        for (const auto& [id, pt] : geo_box.data.nodes)
            if (!pt.incident_ways.empty()) entry.valid_nodes.push_back(id);
        // ── NO SORT — preserves the exact order produced by the prior
        // uncached implementation so agent_start_nodes match byte-for-byte.
    }
    return entry.valid_nodes;
}

// Sample one valid (road-incident) node from the GeoBox. Uses the cached
// candidates vector — equivalent to the prior implementation but O(1) per
// call instead of O(V).
osmium::object_id_type sample_road_node(const GeoBox& geo_box,
                                         std::mt19937& rng) {
    const auto& valid = get_valid_nodes(geo_box);
    if (valid.empty()) return 0;
    std::uniform_int_distribution<size_t> dist(0, valid.size() - 1);
    return valid[dist(rng)];
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
    // Generator is deterministic given ep_seed; the scenario's task profile is
    // the single timing argument. density_mult is applied via the canonical
    // sub/supersample (0xA17EFEEDu mask) shared with the training path.
    {
        EpisodeGenerator gen(cfg, geo_box, ep_seed);
        setup.task_stream = gen.generate(scenario.task_profile);
        apply_density_mult(setup.task_stream, scenario.density_mult,
                           ep_seed ^ 0xA17EFEEDu);
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