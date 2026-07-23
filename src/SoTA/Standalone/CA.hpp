#ifndef SOTA_FAITHFUL_CA_SOLVER_HPP
#define SOTA_FAITHFUL_CA_SOLVER_HPP

#include "SoTA/SolverFramework.hpp"
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// FaithfulCASolver  [Asadi, Nowé, Ghofrani — 2025, GECCO]
// ════════════════════════════════════════════════════════════════════════════
//
// Paper "Congestion-Aware Multi-Agent Path Planning for Pick-up and Delivery
// Tasks". This is the FULL-PIPELINE paper-faithful adaptation, distinct from
// the in-house "CongestionAware" TD-Greedy baseline (Phase 2). The crucial
// algorithmic difference from CongestionAware is:
//
//                                    PATH PLANNING IS CONGESTION-AWARE.
//
// Agents do NOT follow the static shortest path; they route around hot edges
// via an A* whose edge cost is the BPR-adjusted travel time at the
// time-of-traversal estimate.
//
// FAITHFUL CORE — Asadi+2025 §3.2, §3.3, §3.4 adapted to OSM road network:
//
//   §3.2 Modified A* with congestion in the visit cost g(v):
//     ✓ Custom BPR-A* (defined privately in the .cpp). Edge cost =
//       CongestionMap::adjusted_cost(e, length/speed, length, t_enter)
//       where t_enter is the predicted arrival step at the edge head.
//       This makes the planned path actively avoid currently-loaded edges.
//     ✓ Admissible heuristic = euclidean distance / speed_mps (admissible
//       lower bound on travel time under zero congestion).
//
//   §3.1 Per-agent ORDER SEQUENCE T = (τ_a,1, …, τ_a,NT), executed in order:
//     ✓ Each agent owns a queue of assigned-but-undelivered orders, bounded
//       by its capacity (the LGPDP analogue of the paper's N_T, and the same
//       semantics as our DMAS pipeline where max_capacity bounds
//       DeliveryLocalMemory::tasks). Execution stays STRICTLY sequential per
//       order — pickup_i → delivery_i → pickup_{i+1} … — which is exactly
//       paper §3.3's mode transition ("the transition from pick up to
//       delivery occurs upon fulfilling an order"). Concurrent carrying is
//       therefore ≤ 1 by construction; the capacity is spent on the ORDER
//       QUEUE, not on the load, and no P/D interleaving heuristic (which
//       belongs to our DbVNS planner, not to this paper) is imported.
//     ✓ Allocation cost is therefore MARGINAL: an order is appended at the
//       tail of the sequence, so it is priced from where and WHEN the agent
//       finishes what it already holds (plan_tail_node / plan_tail_step,
//       both produced by the congestion-aware commit of the full route).
//
//   §3.3 Agent modes + priority resolution:
//     ✓ Two modes (LGPDP collapse from paper's three):
//         - IDLE   (paper Wandering / no assigned order)
//         - BUSY   (paper Pickup or Delivery — sequence non-empty)
//       Paper §3.4 weights γ_moving < γ_equals < γ_others. We expose this
//       as γ_idle and γ_busy with γ_idle < γ_busy so the BUSY agent's cost
//       gets INFLATED, biasing allocation toward IDLE agents — same effect
//       direction as paper's "goal-directed agents take precedence over
//       wandering agents" (paper §3.4) inverted because we choose ASSIGNEES
//       not OBSTACLES.
//     ✓ β_π tie-break: among agents whose γ-weighted cost is within
//       β_tie_band of the best candidate, prefer the agent with the FEWEST
//       ORDERS STILL TO BE FULFILLED (paper §4.2-a's β_π, verbatim — it is
//       the sequence length, not the carried load). Both this and γ above
//       are only meaningful because agents can now hold a queue: with the
//       previous one-order-at-a-time eligibility rule every candidate was
//       idle with an empty queue, so γ_busy and β_π were unreachable code.
//
//   §3.4 Global congestion message:
//     ✗ Paper's CNN-based prediction is replaced by the BPR-adjusted edge
//       cost computed from the SHARED CongestionMap. The CongestionMap's
//       time-dependent edge load over the configured horizon is the LGPDP
//       analogue of the paper's prediction window.
//     ✗ Multivariate-normal spatio-temporal smoothing — same reason.
//
// WHAT IS DELIBERATELY NOT IMPLEMENTED VS THE PAPER:
//   ✗ CNN training pipeline (§3.4): substituted by current CongestionMap.
//   ✗ Wandering mode as explicit deadlock-resolution state (§3.3): we
//     have IDLE agents that stay where they last delivered. OSM continuous
//     routing has no grid deadlocks to resolve.
//   ✗ Path1/Path2 collision-free planning of TP origin (paper §3.3): not
//     applicable — OSM has no vertex blocking.
//
// COST FUNCTION (decision time):
//   cost(agent, task) = γ_mode(agent)
//                     × ( queue_delay
//                       + BPR_cost(tail_node → task.pickup, tail_step)
//                       + BPR_cost(task.pickup → task.delivery, tail_step + Δp) )
//   where tail_node / tail_step are where and when the agent finishes the
//   orders it already holds (its own position / the current step when idle),
//   queue_delay = tail_step − step, and γ_mode = γ_idle when the agent's
//   sequence is empty, γ_busy otherwise. Summing the queue delay into the
//   cost makes the rule target the order's COMPLETION time, which is the
//   paper's objective M(Π) = max_a |π_a| (§3.1).
//   Δp is the BPR pickup-leg time used as the launch offset for the
//   delivery-leg cost — captures the "future state" the paper's CNN tries
//   to predict, via direct simulation of the agent's planned trajectory.
class FaithfulCASolver : public ISolver {
public:
    FaithfulCASolver() = default;
    ~FaithfulCASolver() override = default;

    // ── Tunable hyperparameters (paper §3.4 γ-mode + §3.3 β_W) ──────────────
    // Default values follow the paper's ordering γ_moving < γ_others without
    // claiming to reproduce the Bayesian-optimised values from the paper
    // (which were fit on a 15x15 grid scenario, not transferable to OSM).
    struct HParams {
        float gamma_idle    = 1.0f;   // weight for idle agents
        float gamma_busy    = 1.5f;   // weight for busy agents (γ_idle < γ_busy)
        float beta_tie_band = 1.05f;  // β_W band: within 5% of best cost is a tie
    };
    HParams hparams;

    void          init(const SolverContext& ctx) override;
    void          inject_task(const ScheduledTask& task, int step) override;
    void          step(int timestep) override;
    SolverMetrics finalize() override;
    const char*   name() const override { return "FaithfulCongestionAware"; }

private:
    struct AgentState {
        osmium::object_id_type current_node = 0;

        std::vector<osmium::object_id_type> current_path_nodes;
        std::vector<osmium::object_id_type> current_path_edges;
        int  next_idx                = 0;
        int  arrival_step_next_node  = -1;
        int  current_edge_t_enter    = 0;

        // Paper §3.1 order sequence T = (τ_a,1 … τ_a,NT): assigned orders not
        // yet delivered, served front-to-back. Bounded by `capacity`.
        std::vector<int> task_queue;
        // Which half of task_queue.front() the agent is heading for.
        bool active_is_pickup_leg = true;

        // Orders physically on board — ≤ 1 under the paper's strictly
        // sequential execution. Kept for the capacity-violation audit.
        std::vector<int> in_flight_task_ids;
        int  capacity = 1;

        // Where / when the agent finishes its whole current sequence, produced
        // by commit_agent_route (BPR-accumulated ETA). Used to price the
        // marginal append cost of a new order.
        osmium::object_id_type plan_tail_node = 0;
        int                    plan_tail_step = 0;

        // Set when planning failed; suppresses per-step A* retry storms.
        int  stalled_until = -1;

        // Full-route congestion footprint registered on the shared map
        // (replicates Option O's commit_plan); removed/re-added on every replan.
        CommittedOcc committed_occ;
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

    // ── Internal BPR-aware path representation ──────────────────────────────
    // Computed on demand by the private BPR-A*; not cached because the cost
    // depends on the START STEP (CongestionMap is time-dependent).
    struct BPRPath {
        std::vector<osmium::object_id_type> nodes;
        std::vector<osmium::object_id_type> edges;
        float trip_time = 0.f;   // total BPR-adjusted travel time in steps
        bool  valid     = false;
    };

    const SolverContext*       ctx_ = nullptr;

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
    int  wait_count_       = 0;
    int  capacity_violations_ = 0;
    int  pairing_violations_  = 0;

    SolverInstrumentation instr_;

    // ── Defining methods ────────────────────────────────────────────────────
    // BPR-aware A* from `from` to `to`, starting at `start_step`. Edge cost
    // is CongestionMap::adjusted_cost evaluated at the predicted arrival
    // step at each edge head. Admissible h = euclidean / speed.
    BPRPath bpr_a_star(osmium::object_id_type from,
                       osmium::object_id_type to,
                       int start_step) const;

    // γ-weighted marginal cost of APPENDING task t to agent a's sequence.
    // Returns +inf if any leg is unreachable.
    float decision_cost(const AgentState& a,
                        const TaskRecord& t,
                        int step) const;

    bool try_allocate_one(int step);

    // Node the agent's current leg is heading to (front order's pickup or
    // delivery), or 0 when the sequence is empty.
    osmium::object_id_type leg_target(const AgentState& a) const;

    // Fire the pickup / delivery event of the front order and move the
    // sequence cursor forward.
    void fire_stop(AgentState& a, int step);

    // Plan and install the leg toward leg_target(). Fires the stop events of
    // any target that already coincides with the agent's position, so callers
    // never recurse. Returns false when nothing could be planned.
    bool advance_plan(AgentState& a, int step);

    void advance_agent(AgentState& a, int step);
    int  edge_arrival_step(osmium::object_id_type edge_id, int t_enter);

    // Re-register the agent's full remaining route on the shared CongestionMap
    // (Option O commit_plan-style footprint) and refresh plan_tail_*. Called on
    // every plan change.
    void recommit_route(AgentState& a, int step);
};

#endif // SOTA_FAITHFUL_CA_SOLVER_HPP