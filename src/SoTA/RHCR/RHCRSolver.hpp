#ifndef SOTA_RHCR_SOLVER_HPP
#define SOTA_RHCR_SOLVER_HPP

#include "SoTA/ISolver.hpp"
#include "SoTA/SolverInstrumentation.hpp"
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// RHCRSolver  [Li, Tinka, Kiesel, Durham, Kumar, Koenig — 2021, AAAI]
// ════════════════════════════════════════════════════════════════════════════
//
// Paper "Lifelong Multi-Agent Path Finding in Large-Scale Warehouses".
// RHCR decomposes lifelong MAPF into a sequence of WINDOWED MAPF instances
// with user-specified time horizon w and replanning period h (h ≤ w). The
// paper's headline contribution is the framework + Multi-Label A* (Alg. 1)
// + bounded-horizon collision detection (§4.2).
//
// This standalone solver is the FULL-PIPELINE adaptation, distinct from
// the per-arrival allocation-only RHCR branch in EpisodeRunner (which
// did the cost-side fidelity but not the framework).
//
// ═══ ADAPTATION LOG ═══════════════════════════════════════════════════════
//
//   Section §4 — Rolling-Horizon Framework:
//   ┌ ✅ Time horizon `w` and replanning period `h` as user parameters.
//   ├ ✅ Per-h-step BATCH processing of pending tasks (lifelong allocation
//   │     buffered between h-boundaries).
//   ├ ✅ Per-h-step REPLANNING of every in-flight agent's path under the
//   │     current CongestionMap state.
//   ├ ⚠️  Paper's "ensure d ≥ h" goal-queue padding (paper §4): NOT
//   │     implemented as a forward-looking mechanism. Reason: in our LGPDP
//   │     setting tasks arrive online and we cannot fabricate goals to fill
//   │     queues. The natural analogue ("if agent is idle and has pending
//   │     tasks available, allocate") is what try_allocate_batch already
//   │     does.
//   └ ⚠️  Allocation rule (paper §4 outsources to an external task assigner):
//         we use GREEDY ARGMIN insertion cost (closest agent in Multi-Label
//         A* terms gets the task). Documented as LGPDP extension since the
//         paper does not specify.
//
//   Section §4.1 — Multi-Label A*:
//   ┌ ✅ State = (node, label) where label counts goals visited so far.
//   ├ ✅ Transition: increment label when reaching the next goal in the
//   │     agent's ordered sequence.
//   ├ ✅ Admissible heuristic = remaining-travel lower bound through the
//   │     unvisited goal sequence (paper's COMPUTEHVALUE on Lines 14-15).
//   ├ ✅ Goal test: label == |goal_sequence|.
//   └ ⚠️  Paper's COMPUTEHVALUE uses dist(); we use euclidean / speed_mps
//         to convert to time units, matching the cost domain of windowed
//         BPR. Adaptation neutral on admissibility.
//
//   Section §4.2 — Bounded-Horizon MAPF Solver:
//   ┌ ✅ Windowed cost: edges entered in [start_step, start_step + w] use
//   │     BPR-adjusted travel time; edges entered after start_step + w use
//   │     free-flow time. This is the LGPDP analogue of the paper's
//   │     "resolve collisions in first w timesteps; ignore beyond w" rule.
//   ├ ❌ Bounded-horizon (E)CBS / PBS variants (paper §4.2): NOT applicable
//   │     — OSM continuous routing has no vertex collisions to detect.
//   │     Windowed cost via BPR is the replacement.
//   └ ❌ Spatio-temporal constraints from higher-priority agents (CA*):
//         NOT applicable for the same reason.
//
//   Section §4.4 — Deadlock Avoidance (P(w) ≥ p):
//   ❌ Potential function P(w) and dynamic w growth: NOT implemented.
//      Reason: OSM continuous routing has no grid deadlocks (no two-corridor
//      head-on standoff; congestion only slows). The paper's deadlock
//      scenario (Example 2) cannot occur on our graph.
//
//   Beyond the paper (LGPDP extensions):
//   ┌ ⊕ Capacity > 1 support (paper assumes 1).
//   ├ ⊕ Greedy insertion-cost allocation (paper outsources allocation).
//   ├ ⊕ Lifelong online task arrival via inject_task + h-batch processing.
//   └ ⊕ BPR-driven cost as the windowed collision proxy (paper uses CBS
//       constraints; we use cost adjustment).
//
// ═══ COMPARISON AXIS ═════════════════════════════════════════════════════
//   TokenPassing    — static A*    + argmin distance       (single goal/path)
//   CongestionAware — static A*    + argmin BPR pickup     (single goal)
//   FaithfulCA      — BPR-A*       + γ-weighted full trip  (single goal)
//   TrafficFlow     — Lex-A*       + contraflow            (single goal)
//   RHCR            — Multi-Label  + batch + windowed       (GOAL SEQUENCE per agent)
//
//   Only RHCR plans through an ORDERED GOAL SEQUENCE per agent in a single
//   path — paper §4.1's Multi-Label A*. Other solvers re-plan at each goal.
//
// ═══ DEPENDENCIES ════════════════════════════════════════════════════════
//   - ISolver, SolverContext, SolverMetrics                 (Phase 0)
//   - GeoBox, CongestionMap                                 (Environment)
//   - Pathfinder::heuristic                                 (admissible h)
//   - NO dependency on PathHelper, other solvers, EpisodeRunner.

class RHCRSolver : public ISolver {
public:
    RHCRSolver() = default;
    ~RHCRSolver() override = default;

    // ── Tunable hyperparameters (paper §4) ──────────────────────────────────
    struct HParams {
        // Time horizon w (paper §4): edges entered in [start_step, start_step
        // + w] use BPR-adjusted cost; edges entered beyond use free-flow.
        // Paper's recommended w = 20 timesteps (§5).
        int   time_horizon_w  = 20;

        // Replanning period h (paper §4): batch task allocation and agent
        // replanning occur every h simulation steps. Paper recommends h = 5
        // (§5). MUST satisfy h ≤ w.
        int   replan_period_h = 5;

        // Multi-Label A* expansion cap. 0 = unlimited. Prevents runaway
        // expansion on pathological graphs. Reduced from 50000 → 30000 as
        // Optim C — on OSM Small/Medium graphs the optimal path is reached
        // well within 30k expansions; on Large graphs (Paris/NewYork) this
        // caps worst-case A* cost while keeping >99% of paths optimal.
        int   max_expansions  = 30000;
    };
    HParams hparams;

    void          init(const SolverContext& ctx) override;
    void          inject_task(const ScheduledTask& task, int step) override;
    void          step(int timestep) override;
    SolverMetrics finalize() override;
    const char*   name() const override { return "RHCR"; }

private:
    // ── Agent state — GOAL SEQUENCE model ───────────────────────────────────
    // Paper §4.1 — Multi-Label A* finds a single path through the agent's
    // ORDERED sequence of remaining goal locations. Each goal is either a
    // pickup or a delivery for a specific task.
    struct GoalEntry {
        int  task_id;
        osmium::object_id_type node;
        bool is_pickup;       // false = delivery
    };

    struct AgentState {
        osmium::object_id_type current_node = 0;

        // Ordered goal sequence (paper §4.1): the agent will visit these
        // nodes in the listed order. New tasks append (pickup, delivery)
        // pairs at the end. When a goal is reached, it pops from the front.
        std::vector<GoalEntry> goal_queue;

        // Path computed by the most recent Multi-Label A* call. nodes[0]
        // is current_node; subsequent nodes pass through goal_queue in
        // order until the last node = goal_queue.back().node.
        std::vector<osmium::object_id_type> current_path_nodes;
        std::vector<osmium::object_id_type> current_path_edges;
        int  next_idx                = 0;
        int  arrival_step_next_node  = -1;

        // Tasks currently in-flight (picked up, not delivered).
        std::vector<int> in_flight_task_ids;
        int  capacity = 1;

        // Set whenever the goal_queue changes so the next step() knows the
        // agent's path needs to be re-computed (defensive: we already
        // recompute at h-boundaries, this catches mid-window changes).
        bool needs_replan = false;
    };

    struct TaskRecord {
        int  task_id;
        osmium::object_id_type pickup_node;
        osmium::object_id_type delivery_node;
        int  arrival_step  = 0;
        int  picked_step   = -1;
        int  delivered_step = -1;
        int  assigned_agent = -1;
        float pd_road_dist = 0.f;
    };

    // ── Result of Multi-Label A* ────────────────────────────────────────────
    struct MultiLabelPath {
        std::vector<osmium::object_id_type> nodes;
        std::vector<osmium::object_id_type> edges;
        float trip_time = 0.f;       // windowed total travel time in steps
        bool  valid     = false;
    };

    const SolverContext* ctx_ = nullptr;

    std::vector<AgentState>    agents_;
    std::vector<int>           pending_task_ids_;
    std::vector<TaskRecord>    tasks_;

    int  appeared_         = 0;
    int  completed_        = 0;
    int  refused_          = 0;
    long latency_sum_      = 0;
    long wait_sum_         = 0;
    long trip_sum_         = 0;
    double road_pd_sum_    = 0.0;
    int  road_pd_count_    = 0;
    long active_steps_sum_ = 0;
    int  capacity_violations_ = 0;
    int  pairing_violations_  = 0;

    // RHCR-specific bookkeeping.
    int  n_batches_           = 0;
    int  n_replans_           = 0;

    SolverInstrumentation instr_;

    // ── Defining methods ────────────────────────────────────────────────────
    // Multi-Label A* (paper §4.1, Algorithm 1). Finds the shortest path from
    // `from` through `goal_seq` in order, using windowed BPR cost.
    MultiLabelPath multi_label_a_star(osmium::object_id_type from,
                                       const std::vector<GoalEntry>& goal_seq,
                                       int start_step) const;

    // Batch task allocation at the h-boundary. For each pending task,
    // greedily assigns it to the agent whose insertion cost (delta in
    // Multi-Label A* trip time) is minimum among eligible agents.
    void allocate_batch(int step);

    // Replan every in-flight agent's path from its current_node through
    // its remaining goal_queue under current congestion.
    void replan_all_agents(int step);

    void advance_agent(AgentState& a, int step);
    int  edge_arrival_step(osmium::object_id_type edge_id, int t_enter) const;

    // Helper used by allocate_batch: computes the insertion delta of
    // appending (pickup, delivery) to the agent's current goal sequence.
    // Returns +inf if unreachable. Uses Multi-Label A*.
    float insertion_cost(const AgentState& a,
                         osmium::object_id_type pickup_node,
                         osmium::object_id_type delivery_node,
                         int task_id,
                         int step) const;
};

#endif // SOTA_RHCR_SOLVER_HPP