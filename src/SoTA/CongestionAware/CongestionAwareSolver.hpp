#ifndef SOTA_CONGESTION_AWARE_SOLVER_HPP
#define SOTA_CONGESTION_AWARE_SOLVER_HPP

#include "SoTA/ISolver.hpp"
#include "SoTA/PathHelper.hpp"
#include "SoTA/SolverInstrumentation.hpp"
#include <memory>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// CongestionAwareSolver — IN-HOUSE TD-Greedy baseline (no source paper)
// ════════════════════════════════════════════════════════════════════════════
//
// This is the FULL-PIPELINE standalone version of the CongestionAware
// allocation branch documented at EpisodeRunner.cpp:1173. It does NOT
// implement any specific published method — it is a classical baseline
// formed by composing:
//
//   (a) "Argmin h(loc, pickup)" task allocation, structurally similar to
//       Token Passing [Ma+2017, AAMAS] but without the free-agent priority
//       and without the endpoint filter T'.
//   (b) BPR-adjusted travel time [Beckmann 1956 / LeBlanc 1975] as the cost
//       used at decision time. The path FOLLOWED is the static shortest
//       path; the difference vs TokenPassing is that the COST EVALUATED at
//       allocation reflects current edge load.
//
// This baseline appears in adjacent literatures under various names:
//   - "Nearest-idle-vehicle dispatch with time-dependent travel time"
//     (Maciejewski & Bischoff 2018, ride-hailing).
//   - "Closest-robot dispatch under congestion" (Pavone+2012, MoD).
//   - Decentralised greedy dispatch baseline in multi-robot TA.
// No single paper "owns" this method — it is presented as an in-house
// baseline in reports.
//
// FULL-PIPELINE BEHAVIOUR (this standalone solver):
//
//   Allocation rule:
//     For each pending task and each ELIGIBLE agent (idle journey, capacity
//     not full), compute the BPR-adjusted travel time from the agent's
//     current node to the pickup along the STATIC shortest path under the
//     CURRENT CongestionMap state. Pick the (agent, task) pair with the
//     globally minimum cost.
//
//   Path planning:
//     Static A* shortest path (via the shared PathHelper). Following the
//     static path under BPR-affected traversal time means the agent
//     EXPERIENCES congestion in execution (slower edges take more steps),
//     but its planned path does not actively avoid congestion. This is the
//     defining trait of a "TD-Greedy" baseline — congestion-aware at the
//     ALLOCATION level, congestion-naive at the ROUTING level.
//
//   Why it differs from a more sophisticated congestion-aware solver:
//     A truly "fully congestion-aware" solver would also REPLAN paths to
//     route around hot edges. That behaviour belongs to
//     FaithfulCongestionAwareSolver (Phase 3) which adapts Asadi+2025
//     GECCO with an A* whose g(v) includes a congestion penalty.
//
// SHARED BEHAVIOUR WITH ALL SOLVERS:
//   - same SolverContext (graph, congestion map, task stream, ghost traffic);
//   - registers traversal load on the SHARED CongestionMap so its traffic
//     footprint is visible to subsequent solvers in the comparison;
//   - capacity = 1 by default (paper-faithful for TP siblings); the
//     LGPDP extension to capacity > 1 is supported via the in_flight_task_ids
//     queue same as TokenPassingSolver.

class CongestionAwareSolver : public ISolver {
public:
    CongestionAwareSolver() = default;
    ~CongestionAwareSolver() override = default;

    void          init(const SolverContext& ctx) override;
    void          inject_task(const ScheduledTask& task, int step) override;
    void          step(int timestep) override;
    SolverMetrics finalize() override;
    const char*   name() const override { return "CongestionAware"; }

private:
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

    const SolverContext*       ctx_ = nullptr;
    std::unique_ptr<PathHelper> paths_;

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

    // ── Internal helpers ────────────────────────────────────────────────────
    // The KEY method that differentiates this solver from TokenPassing:
    // returns the BPR-adjusted travel time along the static shortest path,
    // given current edge loads on the shared CongestionMap. If the path
    // is invalid, returns +inf. If the CongestionMap is unavailable, falls
    // back to the static cost (= TokenPassing behaviour).
    float bpr_pickup_cost(const AgentState& a,
                          osmium::object_id_type pickup_node,
                          int step) const;

    bool try_allocate_one(int step);
    bool begin_leg(AgentState& a, osmium::object_id_type target_node, int step);
    void advance_agent(AgentState& a, int step);
    int  edge_arrival_step(osmium::object_id_type edge_id, int t_enter) const;
};

#endif // SOTA_CONGESTION_AWARE_SOLVER_HPP