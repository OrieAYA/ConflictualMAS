#ifndef SOTA_SOLVER_RUNNER_HPP
#define SOTA_SOLVER_RUNNER_HPP

#include "ISolver.hpp"
#include "SolverContext.hpp"
#include "SolverMetrics.hpp"
#include "Environment/Congestion/CongestionMap.hpp"
#include "Environment/Congestion/GhostTrafficController.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include "Training/EpisodeConfig.hpp"
#include "Training/EpisodeGenerator.hpp"
#include "Training/EpisodeRunner.hpp"   // EpisodeScenario
#include <cstdint>
#include <memory>

// ════════════════════════════════════════════════════════════════════════════
// SolverRunner
// ════════════════════════════════════════════════════════════════════════════
//
// Orchestrator that runs ONE EPISODE for ONE ISolver under a FIXED shared
// SolverContext. This is the entry point used by main-menu options and by
// the LGPDP throughput sweep.
//
// HOW IT GUARANTEES FAIR COMPARISON:
//
//   The SolverRunner pre-generates the entire task stream once via
//   EpisodeGenerator with the user-provided episode_seed. The same task
//   stream is then passed to every solver, in arrival_step order, via
//   ISolver::inject_task(). Likewise, the ghost-traffic controller is seeded
//   deterministically and ticks once per step, so every solver sees the
//   same background congestion.
//
//   The CongestionMap is RESET at the start of each run. Solvers register
//   their agents' edge loads on it during step(); ghost traffic also writes
//   to it. At the end of run() the map is reset again so the next solver
//   starts clean.
//
// USAGE EXAMPLE:
//   GeoBox  box = ...;
//   Pathfinder pf(box);
//   SolverRunner runner(cfg, box, pf, scenario, /*episode_seed*/ 42);
//   {
//       TokenPassingSolver tp;
//       SolverMetrics m = runner.run(tp);
//   }
//   {
//       RHCRSolver rhcr;
//       SolverMetrics m = runner.run(rhcr);   // SAME task stream as tp
//   }
//
// Note: this runner is INDEPENDENT from EpisodeRunner. They share no state.
// EpisodeRunner remains the entry point for MAPPO / IPPO / MAPPER / Hybrid
// training and the legacy allocation-only baseline comparison.

class SolverRunner {
public:
    SolverRunner(const EpisodeConfig& cfg,
                 GeoBox&              geo_box,
                 Pathfinder&          pathfinder,
                 EpisodeScenario      scenario      = {},
                 uint32_t             episode_seed  = 42);

    // Run ONE episode with the given solver.
    // The runner resets its shared congestion map before the run and again
    // after, so consecutive calls with different solvers are mutually
    // isolated. Returns the solver's metrics + wallclock + name.
    //
    // When `setup` is non-null, the runner consumes its canonical task stream,
    // agent_start_nodes, per_agent_capacity, n_active_agents, total_steps and
    // ghost_seed verbatim — bypassing internal RNG-driven re-derivation. This
    // is the publication-grade path used by Option O, ensuring SoTA solvers
    // see byte-identical environment to RL policies running on the SAME setup.
    SolverMetrics run(ISolver& solver,
                       const struct SharedEpisodeSetup* setup = nullptr);

    // Pre-built shared context. Exposed for tests that want to inspect the
    // pre-generated task stream or share the context across multiple runs.
    const SolverContext& context() const { return ctx_; }

private:
    const EpisodeConfig& cfg_;
    EpisodeScenario      scenario_;
    uint32_t             episode_seed_;

    // Owned shared state — built once at SolverRunner construction.
    CongestionMap                            congestion_map_;
    std::unique_ptr<GhostTrafficController>  ghost_;
    EpisodeGenerator                         gen_;
    SolverContext                            ctx_;

    // Build the SolverContext for this episode. Called from the constructor
    // and again at the start of each run() to reset task_stream + congestion
    // map. The task stream is regenerated with the SAME seed so identity is
    // preserved across run() calls.
    void prepare_episode();
};

#endif // SOTA_SOLVER_RUNNER_HPP