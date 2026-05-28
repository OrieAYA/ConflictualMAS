#ifndef SOTA_ISOLVER_HPP
#define SOTA_ISOLVER_HPP

#include "SolverContext.hpp"
#include "SolverMetrics.hpp"
#include "Training/EpisodeGenerator.hpp"   // ScheduledTask

// ════════════════════════════════════════════════════════════════════════════
// ISolver
// ════════════════════════════════════════════════════════════════════════════
//
// Abstract interface that every standalone SOTA solver must implement.
//
// CONTRACT:
//   - The solver owns and runs its OWN full LGPDP pipeline:
//       * allocation algorithm (which agent picks up which task),
//       * path planning (how each agent moves toward its objectives),
//       * any internal scheduling or replanning specific to the source paper.
//   - The solver READS the SolverContext for the simulation surface (graph,
//     congestion map, task stream, fleet sizing). It MUST NOT modify the task
//     stream or the GeoBox; it MAY register its agents' paths on the
//     CongestionMap (this is shared BPR feedback).
//   - The solver MUST advance simulation step-by-step via step(t). Calling
//     code never modifies the solver's internal agents directly.
//   - The solver MUST be deterministic given context.episode_seed — two runs
//     with the same context must produce identical metrics.
//
// LIFECYCLE called by SolverRunner:
//   init(ctx);
//   for (t = 0 ... ctx.total_steps - 1):
//       for (task in ctx.task_stream with arrival_step == t):
//           inject_task(task, t);
//       step(t);
//   metrics = finalize();
//
// Implementations live in src/SoTA/{MethodName}/. Each method gets its own
// folder so its allocation + planning code stays self-contained.

class ISolver {
public:
    virtual ~ISolver() = default;

    // ── One-time setup ──────────────────────────────────────────────────────
    // Called once at the start of an episode. Solver should allocate its
    // internal agent state, path caches, queues, etc. and store the context
    // reference if it needs it later. The context lives for the entire run.
    virtual void init(const SolverContext& ctx) = 0;

    // ── Task ingestion ──────────────────────────────────────────────────────
    // Called once for each task at its arrival_step, BEFORE step(t) runs for
    // that same step. Solver may allocate the task immediately or defer it
    // (e.g. RHCR's batch replanning).
    virtual void inject_task(const ScheduledTask& task, int step) = 0;

    // ── Simulation tick ─────────────────────────────────────────────────────
    // Called once per simulation step. Solver advances its agents along
    // their current paths, processes pickup/delivery events, may replan
    // and may register edge load on the congestion map.
    virtual void step(int timestep) = 0;

    // ── Wrap-up ─────────────────────────────────────────────────────────────
    // Called once at episode end. Solver returns its metrics; SolverRunner
    // augments them with wallclock and solver_name.
    virtual SolverMetrics finalize() = 0;

    // ── Identification ──────────────────────────────────────────────────────
    // Short solver name for logging and CSV columns. Matches the paper
    // identifier (e.g. "TokenPassing", "TrafficFlow", "RHCR").
    virtual const char* name() const = 0;
};

#endif // SOTA_ISOLVER_HPP
