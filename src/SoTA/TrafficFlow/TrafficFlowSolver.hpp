#ifndef SOTA_TRAFFIC_FLOW_SOLVER_HPP
#define SOTA_TRAFFIC_FLOW_SOLVER_HPP

#include "SoTA/ISolver.hpp"
#include "SoTA/SolverInstrumentation.hpp"
#include <vector>
#include <unordered_map>

// ════════════════════════════════════════════════════════════════════════════
// TrafficFlowSolver  [Chen, Harabor, Li, Stuckey — 2024, AAAI]
// ════════════════════════════════════════════════════════════════════════════
//
// Paper "Traffic Flow Optimisation for Lifelong Multi-Agent Path Finding".
// The paper's central contribution is a CONGESTION-AWARE PATH-PLANNING
// algorithm: each agent's path is a "guide path" computed via an A*-like
// search on a TWO-PART EDGE WEIGHT that captures both directional
// contraflow conflict and converging-vertex pressure. The guide paths
// then drive movement (the paper's setting uses PIBT; LGPDP continuous
// routing follows the guide path directly).
//
// This standalone solver is the FULL-PIPELINE paper-faithful adaptation.
// It is distinct from the allocation-only TrafficFlow branch in
// EpisodeRunner (which kept the spirit but skipped guide-path computation).
//
// ═══ ADAPTATION LOG ═══════════════════════════════════════════════════════
// Every adaptation we make to fit the LGPDP / OSM / continuous-routing
// context is listed here so a reviewer can audit fidelity:
//
//   Section §4.1 — Traffic costs:
//   ┌ ✅ Contraflow congestion c_e = f_{v1,v2} × f_{v2,v1} (paper Eq. §4.1).
//   │     Marginal contraflow on a candidate edge = 1 × f_{opposite} (the
//   │     candidate adds one unit to f_{u,v}, so marginal c_e = f_{v,u}).
//   ├ ✅ Directional flow f_{u,v} tracked from in-flight agents' remaining
//   │     edges in their committed guide paths.
//   ├ ⚠️  Vertex pressure p_v = ⌈(n-1)/2⌉ (paper Eq. §4.1) — simplified to
//   │     n_inflow on the candidate vertex. Reason: paper's formula assumes
//   │     synchronous grid arrivals, which doesn't hold on OSM continuous
//   │     routing. The simplified form preserves the "many agents converge
//   │     ⇒ pressure" intent.
//   └ ✅ BPR-adjusted travel time as secondary key (free-flow cost).
//
//   Section §4.2 — Path planning:
//   ┌ ✅ Two-part lexicographic edge weight (Σ contraflow, Σ 1+p_{v_2})
//   │     extended with BPR-adjusted time as tertiary tie-breaker for stable
//   │     ordering on graphs with many equal-contraflow paths.
//   ├ ⚠️  Frank-Wolfe / FOCAL bounded-suboptimal user-equilibrium guide-path
//   │     solver (paper Alg. 2 FindPaths). Replaced by a SINGLE-AGENT lex A*
//   │     evaluated against the CURRENT in-flight directional flow. Reason:
//   │     online lifelong arrival means we cannot do a batch UE iteration.
//   │     The single-agent A* captures the "select the path that minimises
//   │     conflict with currently-committed routes" intent.
//   └ ❌ LNS PathRefinement (paper Alg. 2 PathRefinement loop). Not
//         implemented because cross-agent re-routing would create capacity-
//         violation churn in online LGPDP. Future work.
//
//   Section §4.3 — Guide heuristic + PIBT:
//   ┌ ❌ Lazy backward BFS for guide heuristic h_i(v) = (d_p, d_g).
//   │     Not applicable: we do not use PIBT one-step movement on OSM.
//   ├ ❌ PIBT integration with guide heuristic as preference.
//   │     Replaced by: agents follow their pre-computed guide path
//   │     edge-by-edge (no per-step PIBT priority resolution).
//   └ Reason for both drops: PIBT requires grid vertex collision semantics
//     and a discretised time model that OSM continuous routing doesn't have.
//
//   Section §4.4 — Lifelong procedure:
//   ┌ ✅ Per-arrival allocation + guide path computation.
//   ├ ⚠️  Rolling re-planning ("relax" + refine, paper Alg. 3). We replan
//   │     the delivery leg at pickup arrival (re-evaluate under current
//   │     flow), but do NOT replan all in-flight agents periodically. The
//   │     pickup-replan is a partial fidelity gain at acceptable cost.
//   └ ❌ Online refinement (paper Re-i loop). Not implemented for the
//         same reason as PathRefinement above.
//
//   Beyond the paper (LGPDP extensions):
//   ┌ ⊕ Capacity > 1 support via in_flight_task_ids (paper assumes 1).
//   ├ ⊕ Capacity-aware eligibility filter at allocation.
//   └ ⊕ Allocation rule = argmin lex(contraflow, BPR cost) on (agent,
//       task) — the paper assumes tasks are already assigned; LGPDP needs
//       an allocation rule, so we elevate the path-cost evaluation to the
//       allocation level.
//
// ═══ COMPARISON AXIS ═════════════════════════════════════════════════════
//   TokenPassing    — static A* + argmin distance
//   CongestionAware — static A* + argmin BPR pickup
//   FaithfulCA      — BPR-A*    + γ-weighted full trip (decision + routing)
//   TrafficFlow     — Lex-A*    + contraflow + vertex pressure (this file)
//   RHCR            — Multi-Label batch + windowed (Phase 6)
//
// ═══ DEPENDENCIES ════════════════════════════════════════════════════════
//   - ISolver, SolverContext, SolverMetrics                 (Phase 0)
//   - GeoBox, CongestionMap                                 (Environment)
//   - Pathfinder::heuristic                                 (admissible h)
//   - NO dependency on PathHelper (lex A* is custom)
//   - NO dependency on other solvers
//   - NO dependency on EpisodeRunner / MAPPO

class TrafficFlowSolver : public ISolver {
public:
    TrafficFlowSolver() = default;
    ~TrafficFlowSolver() override = default;

    // ── Tunable hyperparameters ─────────────────────────────────────────────
    struct HParams {
        // Per-vertex pressure: 0 = ignore (only contraflow drives lex
        // ordering); 1 = enabled with exact paper §4.1 formula
        // p_v = ⌈(n_v − 1) / 2⌉.
        int   enable_vertex_pressure = 1;

        // Maximum number of A* expansions per planning call. Caps cost
        // for very long paths on dense maps. 0 = unlimited.
        int   max_expansions = 50000;

        // Paper §4.4 "online refinement" period: every `refinement_period`
        // simulation steps, re-plan the remaining leg of every in-flight
        // agent under the CURRENT directional flow state. 0 = disable.
        // This captures the LGPDP-amenable part of Algorithm 2's
        // PathRefinement (without the cross-agent destruction).
        int   refinement_period = 20;
    };
    HParams hparams;

    void          init(const SolverContext& ctx) override;
    void          inject_task(const ScheduledTask& task, int step) override;
    void          step(int timestep) override;
    SolverMetrics finalize() override;
    const char*   name() const override { return "TrafficFlow"; }

private:
    // ── Same agent state model as TP/CA/FaithfulCA ──────────────────────────
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

    // ── Lex cost type (paper §4.2 two-part edge weight) ─────────────────────
    // (contraflow first, then BPR-adjusted time). Used as the g_score in
    // the lex A*. Compare lexicographically.
    struct LexCost {
        long long contraflow = 0;
        float     bpr_time   = 0.f;

        bool operator<(const LexCost& o) const {
            if (contraflow != o.contraflow) return contraflow < o.contraflow;
            return bpr_time < o.bpr_time;
        }
        bool operator>(const LexCost& o) const { return o < *this; }
        bool operator==(const LexCost& o) const {
            return contraflow == o.contraflow && bpr_time == o.bpr_time;
        }
    };

    // ── Guide path = result of the lex A* ───────────────────────────────────
    struct GuidePath {
        std::vector<osmium::object_id_type> nodes;
        std::vector<osmium::object_id_type> edges;
        LexCost cost;
        bool    valid = false;
    };

    // ── Directional flow tracker (paper §4.1) ───────────────────────────────
    // Maps (from_node, to_node) → number of in-flight agents whose remaining
    // committed guide path traverses that directed edge. Updated when a
    // path is committed/released.
    using DirEdge = std::pair<osmium::object_id_type, osmium::object_id_type>;
    struct DirEdgeHash {
        std::size_t operator()(const DirEdge& p) const noexcept {
            const auto a = std::hash<osmium::object_id_type>{}(p.first);
            const auto b = std::hash<osmium::object_id_type>{}(p.second);
            return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
        }
    };

    const SolverContext* ctx_ = nullptr;

    std::vector<AgentState>    agents_;
    std::vector<int>           pending_task_ids_;
    std::vector<TaskRecord>    tasks_;

    // Live directional flow table.
    std::unordered_map<DirEdge, int, DirEdgeHash> flow_dir_;

    // Live per-vertex inflow counter (number of agents heading to this
    // vertex as part of their remaining path).
    std::unordered_map<osmium::object_id_type, int> vertex_inflow_;

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
    // Lex-A*: returns a guide path that lexicographically minimises
    // (Σ marginal contraflow, Σ BPR-adjusted travel time) given the
    // current in-flight directional flow state.
    GuidePath lex_a_star(osmium::object_id_type from,
                         osmium::object_id_type to,
                         int start_step) const;

    // Register / unregister a guide path's edges in the directional
    // flow tracker. Called when an agent commits a new path or completes
    // its journey.
    void register_path(const std::vector<osmium::object_id_type>& nodes,
                       int from_idx);
    void unregister_path(const std::vector<osmium::object_id_type>& nodes,
                         int from_idx);

    // Lex cost of an arbitrary path (used at allocation time to compare
    // candidates without committing the path).
    LexCost path_lex_cost(const GuidePath& gp) const;

    bool try_allocate_one(int step);
    bool begin_leg_guide(AgentState& a,
                         osmium::object_id_type target_node,
                         int step);
    void advance_agent(AgentState& a, int step);
    int  edge_arrival_step(osmium::object_id_type edge_id, int t_enter) const;
};

#endif // SOTA_TRAFFIC_FLOW_SOLVER_HPP