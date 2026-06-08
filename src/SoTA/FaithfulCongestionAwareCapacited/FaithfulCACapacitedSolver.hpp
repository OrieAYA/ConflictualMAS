#ifndef SOTA_FAITHFUL_CA_CAPACITED_SOLVER_HPP
#define SOTA_FAITHFUL_CA_CAPACITED_SOLVER_HPP

#include "SoTA/ISolver.hpp"
#include "SoTA/SolverInstrumentation.hpp"
#include "SoTA/SolverCongestion.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// FaithfulCACapacitedSolver  [Asadi+2025 GECCO — capacitated extension]
// ════════════════════════════════════════════════════════════════════════════
//
// Capacitated multi-task extension of FaithfulCASolver (Asadi, Nowé, Ghofrani
// 2025, "Congestion-Aware Multi-Agent Path Planning for Pick-up and Delivery
// Tasks", GECCO). The original paper is single-task MAPD — each agent carries
// one task at a time. This solver preserves the FAITHFUL Asadi mechanisms and
// adds ONLY what is structurally needed to exploit per-agent capacity > 1 in
// the LGPDP setting.
//
// MECHANISMS RETAINED VERBATIM FROM FaithfulCASolver (paper §3.2 + §3.3 + §3.4):
//   ✓ BPR-A* routing — every leg is planned by an A* whose edge cost is the
//     CongestionMap-adjusted travel time at the predicted arrival step at the
//     edge head. The agent ACTIVELY ROUTES AROUND HOT EDGES, not just static
//     shortest path. This is paper §3.2's defining trait and is preserved
//     unchanged.
//   ✓ γ-mode weighting (paper §3.4 γ_moving < γ_others, here γ_idle < γ_busy):
//     allocation cost is multiplied by γ depending on whether the candidate
//     agent currently has tasks in flight. Biases allocation toward IDLE
//     agents — same direction as the paper.
//   ✓ β_π tie-break (paper §3.3 "number of orders still to be fulfilled"):
//     within a cost band around the best candidate, prefer the agent with the
//     SMALLEST current load (fewest tasks already in flight).
//   ✓ Continuous re-planning under current congestion: each new leg toward
//     the next stop in the sequence is re-planned by BPR-A* at the moment
//     the agent departs (paper §3.2 — "modified A* incorporates ESTIMATED
//     congestion" evaluated at the time of planning).
//   ✓ CongestionMap traffic footprint: each traversed edge is registered on
//     the SHARED CongestionMap so downstream solvers and the global metric
//     correctly account for this solver's congestion contribution.
//
// WHAT IS ADDED FOR CAPACITY > 1 (the structural enabler — minimal):
//   • Per-agent SEQUENCE of stops, identical layout to HAPC's SequenceStop
//     {task_id, node, is_pickup}. The sequence is the ORDERED list of stops
//     the agent must visit (interleaved pickups and deliveries of tasks
//     currently assigned to it). A capacity-1 agent has at most 2 stops in
//     its sequence (one pickup, one delivery) — same behaviour as the
//     original FaithfulCA. A capacity-K agent can hold up to K simultaneous
//     pickups before each delivery.
//   • cheapest_insertion(agent, task): enumerate every (pos_p, pos_d) with
//     pos_p < pos_d ≤ |seq|+1, check capacity feasibility (load profile
//     across [pos_p..pos_d-1] ≤ capacity), and pick the insertion that
//     minimises the sequence's TOTAL BPR-time. This is the SAME enumeration
//     pattern as HAPC §3.3 BUT WITH A DIFFERENT COST: HAPC uses paper eq.11
//     (load-weighted travel + α·waiting); we use the pure BPR-A* travel time
//     sum to stay faithful to Asadi+2025's cost philosophy. HAPC contributes
//     ONLY the structural skeleton (sequence + insertion enumeration), NOT
//     its objective function.
//   • Eligibility extended: any agent with in_flight_task_ids.size() < cap
//     can receive a new task (the original FaithfulCA bans busy agents at
//     allocation; this is precisely the single-task restriction we lift).
//
// WHAT IS NOT BORROWED FROM HAPC:
//   ✗ Two-step lookahead and demand forecast (paper §3.3 / §4.4) — Asadi+2025
//     has no analogue, and including it would change which method the
//     comparison is testing.
//   ✗ Load-weighted travel + α·waiting (paper eq.11) — FaithfulCA's cost is
//     BPR travel time, not the dial-a-ride passenger-cost formula.
//   ✗ Particle Swarm Optimisation (paper §3.4) — we enumerate exhaustively,
//     same as our HAPC implementation does (yields the global optimum).
//
// COMPARISON AXIS (publication):
//   FaithfulCA           — congestion-aware routing, single-task per agent
//   FaithfulCACapacited  — congestion-aware routing + capacitated multi-task
//                          insertion (this solver)
// The clean delta between these two solvers isolates the value of multi-task
// capacity on top of Asadi+2025's congestion-aware mechanism — which is the
// methodological gap in the original paper for LGPDP applications.
//
// DEPENDENCIES (note for refactoring safety):
//   - ISolver, SolverContext, SolverMetrics                 (Phase 0)
//   - GeoBox, CongestionMap                                 (Environment)
//   - Pathfinder::heuristic                                 (for admissible h)
//   - NO dependency on PathHelper, FaithfulCASolver, or HybridAdaptive-
//     PredictiveSolver. The BPR-A* and the insertion enumeration are
//     re-implemented privately to keep this solver self-contained.

class FaithfulCACapacitedSolver : public ISolver {
public:
    FaithfulCACapacitedSolver() = default;
    ~FaithfulCACapacitedSolver() override = default;

    // ── Tunable hyperparameters ─────────────────────────────────────────────
    // γ-mode weighting INVERTED relative to FaithfulCASolver / Asadi+2025
    // §3.4. The paper's recommendation γ_moving < γ_others (idle preferred,
    // busy penalised) is appropriate for SINGLE-TASK MAPD: every assignment
    // commits an agent for the entire pickup→delivery cycle, so giving the
    // job to a fresh idle agent maximises parallel throughput.
    //
    // In the CAPACITATED setting it is the OPPOSITE — assigning a new task
    // to a busy agent is essentially FREE if the new pickup lies near the
    // agent's existing route (low insertion delta). Penalising busy agents
    // means the multi-task capacity is never exploited: each new task goes
    // to the closest idle agent, regenerating the single-task behaviour of
    // the parent FaithfulCA. To unlock the capacitated regime we PREFER
    // busy agents (γ_busy < γ_idle) when their cost delta is competitive.
    //
    // Documented departure from Asadi+2025 — necessary for fidelity to the
    // capacitated extension, not for fidelity to the original single-task
    // paper. The original FaithfulCASolver (single-task, untouched) keeps
    // the paper's γ_busy > γ_idle ordering.
    struct HParams {
        float gamma_idle    = 1.0f;   // weight for idle agents (no in-flight tasks)
        float gamma_busy    = 0.7f;   // weight for busy agents (γ_busy < γ_idle):
                                      // encourages piggyback on agents already
                                      // en route (their next leg cost is small).
        float beta_tie_band = 1.05f;  // β_π band: within 5% of best cost is a tie

        // ── Performance knobs (near-zero fidelity impact) ───────────────────
        // Per-allocation, consider only the K eligible agents closest to the
        // pickup by EUCLIDEAN distance (admissible lower bound on BPR-time).
        // The truly optimal agent under full BPR-A* is almost always among the
        // K closest by euclidean — Asadi+2025's allocation cost is a γ-weighted
        // sum of legs whose lower bound is dominated by the pickup leg, which
        // is bounded by euclidean / speed.
        // K=0 disables the filter (consider all eligible).
        int max_candidate_agents = 16;
    };
    HParams hparams;

    void          init(const SolverContext& ctx) override;
    void          inject_task(const ScheduledTask& task, int step) override;
    void          step(int timestep) override;
    SolverMetrics finalize() override;
    const char*   name() const override { return "FaithfulCongestionAwareCapacited"; }

private:
    // ── Per-stop element of an agent's planned route ────────────────────────
    // Same layout as HAPC's SequenceStop. is_pickup=true means visiting this
    // stop fires a pickup event for task_id; false means a delivery event.
    struct SequenceStop {
        int  task_id   = -1;
        osmium::object_id_type node = 0;
        bool is_pickup = true;
    };

    struct AgentState {
        osmium::object_id_type current_node = 0;

        // Planned route — interleaved pickup/delivery stops. Sequence is
        // mutated only at allocation time (insertion of a new task) and
        // popped from the front when the agent arrives at sequence[0].
        std::vector<SequenceStop> sequence;

        // Active leg currently being executed toward sequence[0]. Same fields
        // as FaithfulCASolver — replanned by BPR-A* at every new leg.
        std::vector<osmium::object_id_type> current_path_nodes;
        std::vector<osmium::object_id_type> current_path_edges;
        int  next_idx                = 0;
        int  arrival_step_next_node  = -1;
        int  current_edge_t_enter    = 0;

        std::vector<int> in_flight_task_ids;
        int  capacity = 1;

        // Full-route congestion footprint on the shared map (Option O
        // commit_plan-style); removed/re-added on every replan.
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
    // Identical layout to FaithfulCASolver::BPRPath — kept private to keep
    // this solver self-contained (no header coupling).
    struct BPRPath {
        std::vector<osmium::object_id_type> nodes;
        std::vector<osmium::object_id_type> edges;
        float trip_time = 0.f;
        bool  valid     = false;
    };

    // ── Capacitated insertion result ────────────────────────────────────────
    struct InsertionResult {
        float cost_delta = 0.f;    // ΔBPR-trip-time (new sequence − old)
        int   pos_p      = -1;
        int   pos_d      = -1;
        std::vector<SequenceStop> resulting_sequence;
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

    // ── BPR-A* memoization (per-step snapshot) ──────────────────────────────
    // cheapest_insertion calls bpr_a_star O(n²) times per (agent, task) with
    // overlapping (from, to) pairs. Without a cache, evaluating 60 agents on a
    // sequence-length 14 fleet blows up to ~80k A* calls per allocation round.
    // Strategy: snapshot the CongestionMap at the current allocation step and
    // cache all (from, to) results until the next step.
    //
    // Cache key MUST preserve full (from, to) identity: OSM node IDs in busy
    // areas (Tokyo, NYC) exceed 2^32, so naive XOR or 32-bit shifted hashing
    // produces silent collisions → wrong path returned → agent walks to
    // wrong destination → throughput collapses. Use std::pair with a 64-bit
    // mixing hash so unordered_map's bucket-chain equality check disambiguates
    // any hash collisions.
    struct NodePairKey {
        osmium::object_id_type from;
        osmium::object_id_type to;
        bool operator==(const NodePairKey& o) const {
            return from == o.from && to == o.to;
        }
    };
    struct NodePairHash {
        size_t operator()(const NodePairKey& k) const noexcept {
            // Splitmix-style 64-bit mixing combine.
            uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(k.from));
            uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(k.to));
            a ^= (a >> 33); a *= 0xff51afd7ed558ccdull; a ^= (a >> 33);
            b ^= (b >> 33); b *= 0xc4ceb9fe1a85ec53ull; b ^= (b >> 33);
            return static_cast<size_t>(a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2)));
        }
    };
    mutable std::unordered_map<NodePairKey, BPRPath, NodePairHash> bpr_cache_;
    mutable int cache_valid_at_step_ = -1;

    BPRPath bpr_lookup(osmium::object_id_type from,
                       osmium::object_id_type to,
                       int step) const;

    // ── Defining methods (mirror FaithfulCASolver where possible) ───────────
    // BPR-aware A* — same implementation philosophy as FaithfulCASolver.
    BPRPath bpr_a_star(osmium::object_id_type from,
                       osmium::object_id_type to,
                       int start_step) const;

    // Total BPR-time of an ordered sequence starting from `from_node` at
    // `start_step`. Returns +inf if any leg is unreachable.
    float sequence_bpr_time(osmium::object_id_type from_node,
                            const std::vector<SequenceStop>& seq,
                            int start_step) const;

    // Best (pos_p, pos_d) insertion of task t into agent a's sequence,
    // capacity-feasible. Cost = ΔBPR-time. γ-weighting is applied OUTSIDE
    // (in try_allocate_one) to keep insertion symmetric across agents.
    // `cutoff_raw_delta` is an UPPER BOUND on the raw (pre-γ) delta beyond
    // which the caller has no interest — used for early termination. Set to
    // +inf to disable pruning.
    InsertionResult cheapest_insertion(const AgentState& a,
                                       const TaskRecord& t,
                                       int step,
                                       float cutoff_raw_delta) const;

    bool try_allocate_one(int step);

    // Replan the current leg toward sequence[0] (called after sequence
    // mutation by insertion, or after firing an arrival event).
    bool begin_leg_bpr(AgentState& a, int step);

    // Fire pickup/delivery events on arrival at sequence[0] and pop it.
    // Recursively launches the next leg if the sequence still has stops.
    void on_arrival(AgentState& a, int step);

    void advance_agent(AgentState& a, int step);
    int  edge_arrival_step(osmium::object_id_type edge_id, int t_enter);

    // Re-register the agent's full remaining sequence on the shared
    // CongestionMap (Option O commit_plan-style). Called on every plan change.
    void recommit_route(AgentState& a, int step);
};

#endif // SOTA_FAITHFUL_CA_CAPACITED_SOLVER_HPP
