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
//   §3.3 Agent modes + priority resolution (β_W):
//     ✓ Two modes (LGPDP collapse from paper's three):
//         - IDLE   (paper Wandering / no active task)
//         - BUSY   (paper Pickup or Delivery — actively executing a task)
//       Paper §3.4 weights γ_moving < γ_equals < γ_others. We expose this
//       as γ_idle and γ_busy with γ_idle < γ_busy so the BUSY agent's cost
//       gets INFLATED, biasing allocation toward IDLE agents — same effect
//       direction as paper's "goal-directed agents take precedence over
//       wandering agents" (paper §3.4) inverted because we choose ASSIGNEES
//       not OBSTACLES.
//     ✓ β_W-style tie-break: among agents whose γ-weighted cost is within
//       β_tie_band of the best candidate, prefer the agent with the
//       SMALLEST in-flight load (paper's β_π — fewer remaining orders —
//       generalised to lifelong setting).
//
//   §3.4 Global congestion message:
//     ✗ Paper's CNN-based prediction is replaced by the BPR-adjusted edge
//       cost computed from the SHARED CongestionMap. The CongestionMap's
//       sparse temporal load (load_(e,t)) over the configured horizon is
//       the LGPDP analogue of paper's prediction window.
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
//                     × ( BPR_cost(agent.loc → task.pickup, step)
//                       + BPR_cost(task.pickup → task.delivery, step + Δp) )
//   where γ_mode = γ_idle if agent has no in-flight tasks, γ_busy otherwise.
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
    // Same state model as TP/CA — see those for invariants. Each agent has
    // at most ONE in-flight objective at a time; capacity > 1 still allows
    // multiple tasks in the in_flight_task_ids queue once picked up.
    struct AgentState {
        osmium::object_id_type current_node = 0;

        std::vector<osmium::object_id_type> current_path_nodes;
        std::vector<osmium::object_id_type> current_path_edges;
        int  next_idx                = 0;
        int  arrival_step_next_node  = -1;
        int  current_edge_t_enter    = 0;

        int  active_task_id = -1;
        bool active_is_pickup_leg = true;

        std::vector<int> in_flight_task_ids;
        int  capacity = 1;

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

    // γ-weighted full-trip cost used at decision time. Returns +inf if any
    // leg is unreachable.
    float decision_cost(const AgentState& a,
                        const TaskRecord& t,
                        int step) const;

    bool try_allocate_one(int step);
    bool begin_leg_bpr(AgentState& a,
                       osmium::object_id_type target_node,
                       int step);
    void advance_agent(AgentState& a, int step);
    int  edge_arrival_step(osmium::object_id_type edge_id, int t_enter);

    // Re-register the agent's full remaining route on the shared CongestionMap
    // (Option O commit_plan-style footprint). Called on every plan change.
    void recommit_route(AgentState& a, int step);
};

#endif // SOTA_FAITHFUL_CA_SOLVER_HPP