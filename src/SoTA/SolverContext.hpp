#ifndef SOTA_SOLVER_CONTEXT_HPP
#define SOTA_SOLVER_CONTEXT_HPP

#include "Environment/GeoBox/Box.hpp"
#include "Environment/Congestion/CongestionMap.hpp"
#include "Environment/Congestion/GhostTrafficController.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include "Training/EpisodeConfig.hpp"
#include "Training/EpisodeGenerator.hpp"
#include <cstdint>
#include <osmium/osm/types.hpp>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// SolverContext
// ════════════════════════════════════════════════════════════════════════════
//
// EVERYTHING THAT MUST BE IDENTICAL ACROSS SOTA SOLVERS FOR A FAIR COMPARISON.
//
// The SolverRunner builds one SolverContext per episode and passes a const
// reference into every ISolver::init(). The context owns the shared simulation
// surface: road graph, congestion model, task stream, scenario parameters,
// fleet sizing. Two solvers given the same SolverContext face byte-identical
// task arrivals, byte-identical ghost-traffic injection, and byte-identical
// agent start positions.
//
// What each ISolver implementation owns ITSELF (NOT in the context):
//   - allocation algorithm
//   - path planning (faithful to the source paper)
//   - per-agent state (queues, current path, in_transit position)
//   - replanning / scheduling logic
//   - cost model used for its internal decisions
//
// Lifetime: the context is constructed at the start of each episode and lives
// for the entire run(). Solvers may keep a const SolverContext& after init().
// References inside the context are STABLE for the episode duration.

struct SolverContext {
    // ── World (read-only references) ────────────────────────────────────────
    const GeoBox*    geo_box    = nullptr;   // road graph (owned by caller)
    Pathfinder*      pathfinder = nullptr;   // shared A*/Dijkstra (caller-owned)

    // ── Mutable shared state ────────────────────────────────────────────────
    // Solvers WRITE to congestion_map (registering their agents' paths) so the
    // BPR feedback works across the system. Ghost traffic writes here too.
    CongestionMap*        congestion_map = nullptr;
    GhostTrafficController* ghost         = nullptr;   // may be nullptr

    // ── Episode definition ──────────────────────────────────────────────────
    const EpisodeConfig* episode_config = nullptr;   // user-configured params

    // The task stream. Pre-generated deterministically from the seed so every
    // solver replays the exact same arrivals. The solver does NOT mutate this.
    std::vector<ScheduledTask> task_stream;

    // Fleet sizing for this episode (post agents_mult scaling).
    int  n_active_agents       = 0;
    int  max_capacity_per_agent = 1;

    // Initial agent positions. solvers[i] should start agent i at
    // agent_start_nodes[i].
    std::vector<osmium::object_id_type> agent_start_nodes;

    // ── Per-agent capacity heterogeneity (paper Y, optional) ────────────────
    // When non-empty, capacities[i] is the carrying capacity of agent i. Empty
    // = use max_capacity_per_agent uniformly.
    std::vector<int> per_agent_capacity;

    // ── Sim parameters (already scaled to this episode's scenario) ──────────
    int   total_steps  = 0;
    float speed_mps    = 5.f;
    int   city_index   = 0;
    int   num_cities   = 1;

    // ── RNG seed (stable for this episode) ──────────────────────────────────
    // Solvers MUST NOT call rand() / std::random_device directly. Any internal
    // randomness should be seeded from this value for reproducibility.
    uint32_t episode_seed = 42;

    // ── Helpers ─────────────────────────────────────────────────────────────
    // Sample one valid road node (uniform over the road graph). Useful for
    // solvers that need to park idle agents at non-task endpoints.
    osmium::object_id_type sample_valid_node(std::mt19937& rng) const;
};

#endif // SOTA_SOLVER_CONTEXT_HPP
