#ifndef SOTA_TOKEN_PASSING_SOLVER_HPP
#define SOTA_TOKEN_PASSING_SOLVER_HPP

#include "SoTA/ISolver.hpp"
#include "SoTA/PathHelper.hpp"
#include "SoTA/SolverInstrumentation.hpp"
#include <memory>
#include <optional>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// TokenPassingSolver  [Ma, Li, Kumar, Koenig — 2017, AAMAS]
// ════════════════════════════════════════════════════════════════════════════
//
// Paper "Lifelong Multi-Agent Path Finding for Online Pickup and Delivery
// Tasks" §4.1, Algorithm 1 (TP).
//
// FAITHFUL CORE:
//   - At every step, FREE agents (those not currently executing a task)
//     request the token in turn.
//   - The agent holding the token picks τ* ∈ T' = {τ ∈ T | no other path
//     in the token ends in s_τ or g_τ} that minimises h(loc(a), s_τ).
//     In our LGPDP setting the endpoint filter T' is vacuous (unique
//     pickup/delivery nodes), so τ* = argmin_{τ ∈ T} h(loc(a), s_τ).
//   - h(loc, target) is the STATIC graph distance (paper §4.1 — deterministic
//     cost model). PathHelper::get(from, to) returns the A* shortest path.
//   - Once allocated, the agent traverses current_node → pickup → delivery
//     node-by-node, one path edge per simulation step (paper's discrete
//     time model translated to OSM continuous routing).
//   - Path1 (Algorithm 1 line 12) — paper plans a collision-free path
//     respecting other agents' paths in the token. In our LGPDP setting,
//     there are no vertex blocking semantics (multiple agents may share an
//     OSM node); the shortest A* path is the faithful analogue. We still
//     register the agent's path on the SHARED CongestionMap so the BPR cost
//     model sees the load (which is how downstream methods like the
//     congestion-aware solvers will see TP's traffic footprint).
//
// CAPACITY (LGPDP extension, paper has capacity = 1):
//   Paper §3.1 — an agent is "free" iff it is currently NOT executing any
//   task. This corresponds to our capacity = 1 case. For capacity > 1 the
//   paper does not specify; we extend by treating "free" as "queue has
//   space" (load < capacity). The lifetime of "load" is from the moment a
//   task is picked up to the moment it is delivered.
//
// WHAT IS NOT FAITHFUL TO THE PAPER:
//   ✗ TPTS (Algorithm 2 task swaps). With online single-task arrival the
//     swap collapses to the same argmin as TP — paper-faithful in our
//     regime; we document it at the call site.
//   ✗ Path2 (Algorithm 1 line 16, move to a non-task endpoint when
//     standing on a goal). OSM road networks have no "endpoint" topology.
//     Idle agents stay where they delivered, which is the lifelong analogue
//     of the paper's resting behaviour.

class TokenPassingSolver : public ISolver {
public:
    TokenPassingSolver() = default;
    ~TokenPassingSolver() override = default;

    void          init(const SolverContext& ctx) override;
    void          inject_task(const ScheduledTask& task, int step) override;
    void          step(int timestep) override;
    SolverMetrics finalize() override;
    const char*   name() const override { return "TokenPassing"; }

private:
    // ── Per-agent state ─────────────────────────────────────────────────────
    struct AgentState {
        osmium::object_id_type current_node = 0;

        // Active journey (if any): a current path nodes/edges + traversal idx.
        // While moving toward an objective, current_path is non-empty and
        // next_idx < current_path.edges.size(). Reaching the end of the path
        // means arrival at the current objective.
        std::vector<osmium::object_id_type> current_path_nodes;
        std::vector<osmium::object_id_type> current_path_edges;
        int  next_idx                = 0;     // index in current_path_edges
        int  arrival_step_next_node  = -1;    // step at which agent will reach next node
        int  current_edge_t_enter    = 0;     // step the agent entered current edge

        // What the agent is doing right now. Paper TP has one in-flight task
        // per agent (capacity = 1); we still track the task id to record
        // pickup/delivery events.
        int  active_task_id = -1;             // -1 = idle
        bool active_is_pickup_leg = true;     // true=moving to pickup; false=to delivery

        // Capacity tracking (LGPDP extension). Holds task IDs currently
        // picked up but not delivered. With paper TP (capacity=1) at most
        // one entry; with capacity>1, agent can still receive new tasks
        // while in transit.
        std::vector<int> in_flight_task_ids;
        int  capacity = 1;
    };

    // ── Task tracking ───────────────────────────────────────────────────────
    struct TaskRecord {
        int  task_id;
        osmium::object_id_type pickup_node;
        osmium::object_id_type delivery_node;
        int  arrival_step  = 0;
        int  picked_step   = -1;
        int  delivered_step = -1;
        int  assigned_agent = -1;
        float pd_road_dist = 0.f;   // shortest pickup→delivery road distance (m)
    };

    const SolverContext*       ctx_ = nullptr;
    std::unique_ptr<PathHelper> paths_;

    std::vector<AgentState>    agents_;
    std::vector<int>           pending_task_ids_;   // tasks injected, not yet assigned
    std::vector<TaskRecord>    tasks_;              // by task_id

    // ── Bookkeeping for metrics ─────────────────────────────────────────────
    int  appeared_         = 0;
    int  completed_        = 0;
    int  refused_          = 0;
    long latency_sum_      = 0;
    long wait_sum_         = 0;
    long trip_sum_         = 0;
    double road_pd_sum_    = 0.0;
    int  road_pd_count_    = 0;
    long active_steps_sum_ = 0;   // sum over (step, agent) where agent is in transit
    int  capacity_violations_ = 0;
    int  pairing_violations_  = 0;

    // Shared instrumentation (Gini, congestion sampling, fleet distance,
    // compute timing). Owned by the solver; never touches the solver's
    // task lifecycle directly.
    SolverInstrumentation instr_;

    // ── Internal helpers ────────────────────────────────────────────────────
    // Try to allocate a single pending task to a free agent using the TP
    // rule. Returns true if an allocation was made.
    bool try_allocate_one(int step);

    // Begin a new path leg for agent a from a.current_node to target_node.
    // Sets up current_path_nodes/edges/next_idx and schedules the first
    // edge arrival. Registers the path on the shared CongestionMap so other
    // solvers later see TP's traffic footprint as background flow.
    bool begin_leg(AgentState& a, osmium::object_id_type target_node, int step);

    // Advance agent one step along its current path. Handles edge entry
    // BPR cost (slowing on congested edges) by stretching arrival times.
    // Fires pickup / delivery events as the agent reaches objective nodes.
    void advance_agent(AgentState& a, int step);

    // Predict the integer step at which the agent will arrive at the END
    // of a single edge entered at t_enter. Uses BPR-adjusted travel time.
    int  edge_arrival_step(osmium::object_id_type edge_id, int t_enter) const;
};

#endif // SOTA_TOKEN_PASSING_SOLVER_HPP