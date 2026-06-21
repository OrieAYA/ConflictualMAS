#ifndef SOTA_HYBRID_ADAPTIVE_PREDICTIVE_SOLVER_HPP
#define SOTA_HYBRID_ADAPTIVE_PREDICTIVE_SOLVER_HPP

#include "SoTA/SolverFramework.hpp"
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// HybridAdaptivePredictiveSolver  [Cortés, Sáez, Núñez, Muñoz-Carpintero —
//                                  2009, Transportation Science 43(1):27-42]
// ════════════════════════════════════════════════════════════════════════════
//
// Paper "Hybrid Adaptive Predictive Control for a Dynamic Pickup and Delivery
// Problem". The seminal capacitated-DARP baseline with PREDICTIVE rolling
// horizon: when a new request enters the system the dispatcher picks the
// insertion (vehicle, pickup_pos, delivery_pos) that minimises BOTH the
// immediate insertion cost C_j(k+1) AND the expected cost of inserting a
// PROBABLE future call C_j(k+2) drawn from per-OD-pair historical
// probabilities. This non-myopic two-steps-ahead lookahead is the paper's
// headline contribution (paper Tables 2-3: +13% waiting / +5% travel-time
// savings vs the one-step myopic baseline).
//
// ─────────────────────────────────────────────────────────────────────────
// FAITHFUL CORE — what we implement to match the paper:
// ─────────────────────────────────────────────────────────────────────────
//
//   §3.1 — Variable-step event-triggered dispatch:
//     ✓ Sequence S_j(k) for each vehicle, fixed between events.
//     ✓ Re-routing fires only on (a) new task arrival (insertion) or
//       (b) stop reached (sequence shrinks by one row). No periodic MPC —
//       the prior version added this and we removed it as a hors-papier
//       deviation.
//
//   §3.2 — State-space sequence representation (paper eq. 7):
//     ✓ Each stop carries the binary r_j^i ∈ {0,1} (pickup vs delivery)
//       and the task id ("label") for precedence checking.
//     ✓ Capacity feasibility enforced during insertion: load profile
//       across [pos_p .. pos_d-1] cannot exceed agent capacity.
//
//   §3.3 — Objective function (paper eq. 11):
//     ✓ C_j = Σ_i  ( L_j^(i-1) + 1 ) · ( T_j^i − T_j^(i-1) )         [travel]
//           + Σ_i  r_j^i · α · ( T_j^i − T_j^0 )                       [waiting]
//       where (load+1) inflates per-edge cost by the number of passengers
//       (+1 = driver = operational-cost proxy), and the waiting term fires
//       only at pickup stops (r_j^i = 1) with weight α.
//     ✓ α = 1 by default (paper §4.1 — travel time as important as waiting).
//
//   §3.3 — Two-steps-ahead lookahead (paper eq. 15-16):
//     ✓ For each candidate insertion (vehicle, pos_p, pos_d) of the
//       current call, compute the IMMEDIATE cost delta ΔC(k+1).
//     ✓ Then for each of H probable virtual future calls h with
//       probability p_h^ΔT (drawn from EMA-tracked OD-pair frequencies),
//       compute the BEST insertion cost ΔC(k+2)|h of that virtual call into
//       the post-insertion sequence.
//     ✓ Total cost = ΔC(k+1) + Σ_h p_h × ΔC(k+2)|h.
//     ✓ Pick argmin over (vehicle, pos_p, pos_d).
//
//   §3.4 — Solution method:
//     ✗ Paper uses a custom Particle Swarm Optimisation (PSO) with 10
//       particles × 10 iterations to scale up. We use exhaustive
//       enumeration (EE) over (pos_p, pos_d). For our typical sequence
//       lengths < 20, EE is tractable and yields the GLOBAL optimum that
//       paper's PSO approximates with a 2-3% gap (paper §4.2 note on EE
//       vs PSO). We disclose this as a strengthening of the paper.
//
//   §4.4 — Demand forecast (probabilities p_h^ΔT):
//     ✓ Paper computes probabilities offline from historical
//       origin-destination trip counts per zone pair per time interval.
//     ✓ Our solver substitutes ONLINE per-OD-cell-pair EMA over a coarse
//       spatial grid (8×8 default over the GeoBox bounding box). Each task
//       arrival bumps λ_{pickup_cell, delivery_cell}; all cells decay
//       slowly so the forecast tracks demand drift. The top-H OD-pairs
//       (by EMA mass) are read out as the virtual future calls; their
//       probabilities are normalised so Σ_h p_h = 1 inside top-H (matches
//       paper §4.1 where 90% of trips concentrate in H = 6 pairs).
//
// ─────────────────────────────────────────────────────────────────────────
// FIDELITY CAVEATS — what we DELIBERATELY adapt to our context:
// ─────────────────────────────────────────────────────────────────────────
//   ✗ Paper: Euclidean straight-line at constant 20 km/h.
//     Ours:  OSM road network with shortest-path A* (static distance,
//            i.e. free-flow travel time). We could BPR-adjust the segment
//            cost but the paper itself does not model congestion so we
//            stay faithful in keeping the routing cost time-independent.
//   ✗ Paper: PSO with rounding + repair.
//     Ours:  EE (yields global optimum, sub-2% closer than PSO).
//   ✗ Paper: τ tuned offline via sensitivity (paper Fig. 5 → 4.8 min for
//            9 zones, 2.4 min for 4 zones). τ is the assumed time between
//            calls and only affects the prediction of WHEN the virtual
//            future call lands.
//     Ours:  We do not need τ explicitly because we evaluate the future
//            insertion ON THE SAME SEQUENCE STATE (immediately after the
//            current insertion). This matches the paper's MIN argument —
//            see paper eq. 15: only the RELATIVE cost of competing
//            insertions matters, and τ is constant across them.
//
// CAPACITY (paper-native):
//   Agent capacity ≥ 1 (paper test: 4 passengers per vehicle). Heterogeneous
//   capacities (ctx.per_agent_capacity) supported — the paper holds capacity
//   constant but the algorithm generalises trivially.
//
// COMPARISON AXIS (publication):
//   Single-task SoTA (TP, FCA, TF) cannot exploit capacity > 1 and serve
//   tasks sequentially. HAPC is the capacitated-DARP baseline that
//   exploits both (a) multi-task carrying and (b) demand-prediction —
//   this is the methodological lever for the standalone evaluation.
//
// DEPENDENCIES:
//   - ISolver, SolverContext, SolverMetrics                 (Phase 0)
//   - GeoBox, CongestionMap                                 (Environment)
//   - PathHelper                                            (shared A*)
//   - NO dependency on other solvers.

class HybridAdaptivePredictiveSolver : public ISolver {
public:
    HybridAdaptivePredictiveSolver() = default;
    ~HybridAdaptivePredictiveSolver() override = default;

    // ── Tunable hyperparameters (paper §3.3 + §4.1) ─────────────────────────
    struct HParams {
        float alpha            = 1.0f;    // paper §3.3 waiting-cost weight (paper test α = 1)
        bool  enable_two_step  = true;    // §3.3 two-steps-ahead lookahead (false = myopic)
        int   forecast_top_h   = 6;       // §4.1 paper uses H = 6 for 9-zone disaggregation
        float forecast_min_p   = 0.02f;   // skip virtual calls with prob below this
        int   demand_grid_dim  = 8;       // coarse 8x8 spatial grid (paper uses 9 zones)
        float alpha_demand     = 0.02f;   // §4.4 EMA weight for online OD-pair learning
    };
    HParams hparams;

    void          init(const SolverContext& ctx) override;
    void          inject_task(const ScheduledTask& task, int step) override;
    void          step(int timestep) override;
    SolverMetrics finalize() override;
    const char*   name() const override { return "HybridAdaptivePredictive"; }

private:
    // ── Per-agent state ─────────────────────────────────────────────────────
    // Mirrors paper's S_j(k) (eq. 7) — a sequence of (r_j^i, label_j^i) pairs
    // for stop i, with r_j^i = is_pickup, label_j^i = task_id. The Γ_j^i
    // segment cost is recomputed on the fly from PathHelper.
    struct SequenceStop {
        int  task_id     = -1;
        osmium::object_id_type node = 0;
        bool is_pickup   = true;
    };

    struct AgentState {
        osmium::object_id_type current_node = 0;
        std::vector<SequenceStop> sequence;

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
        int  task_id        = -1;
        osmium::object_id_type pickup_node   = 0;
        osmium::object_id_type delivery_node = 0;
        int  arrival_step   = 0;
        int  picked_step    = -1;
        int  delivered_step = -1;
        int  assigned_agent = -1;
        float pd_road_dist  = 0.f;
        // Retry backoff (CRITICAL perf fix): per-step retry of an unassignable
        // task in HAPC's step() would invoke try_insert_task ~3600× per stuck
        // task per episode (~13 ms each = ~50 s wasted per stuck task). Under
        // saturation (over_fleet, low fleet count) many tasks remain stuck for
        // most of the episode → multiple-minutes wallclock overhead.
        //
        // With exponential backoff: a stuck task is retried at step+1, +2, +4,
        // +8, +16, +32, +64 (capped) — at most ~log2(3600) ≈ 12 retries per
        // episode instead of 3600. ZERO fidelity loss: paper §3.1 says re-
        // routing is event-triggered (call arrival / stop reached). Per-step
        // retry was our own non-paper addition; backoff is closer to paper.
        int next_retry_step = 0;
        int retry_count     = 0;
    };

    // ── Cost components (paper eq. 11) ──────────────────────────────────────
    // travel_cost is the load-weighted sum of segment travel times;
    // waiting_cost is the α-weighted sum of arrival times at pickup stops.
    struct PaperCost {
        float travel  = 0.f;
        float waiting = 0.f;
        float total() const { return travel + waiting; }
    };

    // ── Demand forecast (paper §4.4 — OD-pair × time-interval probabilities)
    // We track per-(pickup_cell, delivery_cell) EMA over a coarse grid built
    // at init() from the GeoBox bounding box. At decision time the top-H
    // cell-pairs by EMA mass are converted into VirtualCall (pickup_node,
    // delivery_node, normalised probability) tuples used by the two-step
    // lookahead. Cell representative nodes are picked at init() — one OSM
    // node per cell, found via single-pass scan.
    struct ZonePairForecast {
        int dim = 8;
        float lat_min = 0.f, lat_step = 0.f;
        float lon_min = 0.f, lon_step = 0.f;

        // λ_{i,j} for the OD-pair from cell i to cell j (encoded as i*n+j).
        std::unordered_map<int64_t, float> lambda_pair;

        // Representative OSM node per cell, used to instantiate virtual calls.
        // rep_node_of_cell[c] = some valid road node inside cell c (0 if cell
        // is empty / not populated by any node).
        std::vector<osmium::object_id_type> rep_node_of_cell;

        // Locate a node's cell. Returns -1 if out of bounds / box unset.
        int cell_of(osmium::object_id_type id, const GeoBox& g) const;

        // EMA update for an arrival event.
        void register_arrival(int p_cell, int d_cell, float alpha);

        // Slow uniform decay of all cells (paper §4.4 EMA implicitly decays
        // unobserved cells toward 0).
        void decay_all(float decay);

        // Top-H (pickup_node, delivery_node, prob) virtual calls. probs are
        // normalised so Σ = 1 within the returned subset. Filters out cells
        // with no rep_node (the OD pair is unrealised on the graph).
        struct VirtualCall {
            osmium::object_id_type pickup_node;
            osmium::object_id_type delivery_node;
            float probability;
        };
        std::vector<VirtualCall> top_h(int H, float min_p) const;
    };

    const SolverContext*       ctx_   = nullptr;

    // PathHelper is now process-scoped (one instance per city) via the static
    // shared_path_helper() singleton. paths_ is a non-owning view that the
    // current HAPC instance uses. Lifetime is tied to the static singleton,
    // which is reset only when the city (Pathfinder/GeoBox pair) changes.
    PathHelper*                paths_ = nullptr;
    // Static accessor — owns the per-city PathHelper. Returns a unique_ptr
    // reference so init() can replace it when the city changes.
    static std::unique_ptr<PathHelper>& shared_path_helper();

    // ── Persistent cross-episode distance cache (city-scoped) ───────────────
    // Mirrors the PDPGlobalMemory / ObjectiveCache pattern from the main DMAS
    // pipeline: shortest-path road distance is a property of the GRAPH alone,
    // not of any solver/episode. By keying a static cache on (Pathfinder*,
    // GeoBox*) we let every HAPC episode of the SAME city reuse all distances
    // computed by prior episodes — cold-cache cost is paid ONCE per city, not
    // once per (episode × city) combo. The cache is wiped on init() when a
    // different (Pathfinder*, GeoBox*) pair is detected (= city changed).
    //
    // Lookup priority in seg_cost:
    //   1. shared_cache_->data[(from, to)]                — O(1) hit
    //   2. PathHelper::get(...) → A* via Pathfinder       — cold path, then
    //      store into shared_cache_ so subsequent queries are O(1).
    struct SharedDistanceCache {
        const Pathfinder* pf = nullptr;
        const GeoBox*     gb = nullptr;
        // Content guard against stack-frame reuse: main.cpp's city for-loop
        // destroys/recreates GeoBox+Pathfinder as locals — addresses can be
        // reused. Comparing node count + first sentinel detects this case.
        size_t            gb_node_count = 0;
        // Per-source Dijkstra results (replaces the prior flat (from,to)→dist
        // map which bloated to ~4M entries on first city sweep, slowing every
        // lookup to ~500ns-1μs due to CPU cache misses on big buckets).
        //
        // dijkstra_from[src] = {target_node : shortest_path_distance}
        //
        // Lookup pattern in seg_cost (OSM road graph is undirected so
        // d(a,b) == d(b,a)):
        //   1. dijkstra_from[from][to]    if `from` was pre-warmed
        //   2. dijkstra_from[to][from]    if only `to` was pre-warmed
        //   3. fallback to PathHelper::get(from,to).cost
        //
        // Memory bound: O(#prewarmed × V) — ~32 MB on Tokyo_Small (200×10k),
        // ~640 MB on Paris if fully populated. Lookups: ~50ns outer hash +
        // ~100ns inner hash = ~150ns total vs ~500-1000ns on the prior flat
        // 4M-entry map.
        std::unordered_map<osmium::object_id_type,
                           std::unordered_map<osmium::object_id_type, float>>
            dijkstra_from;
    };
    static SharedDistanceCache& shared_cache();

    // Run a single-source Dijkstra from `src` and populate the persistent
    // cache with d(src, X) AND d(X, src) (OSM road graph is undirected, so
    // distances are symmetric). No-op if `src` has already been pre-warmed
    // within this city. Equivalent to N A* calls but typically 5-50× faster
    // when N is large (the pickup/delivery of a freshly-arrived task gets
    // queried ~60+N×n times during one allocation round).
    void prewarm_from_node(osmium::object_id_type src) const;

    // ── Precomputed sequence profile ─────────────────────────────────────
    // Built once per (agent, base or post-insertion sequence) and reused
    // across all H virtual-call inner cheapest_insertion calls. Mirrors the
    // PDPGlobalMemory pattern of building the cost matrix once and reusing
    // it across DbVNS local-search neighborhoods.
    struct SeqProfile {
        // t_seg[i]        = seg(pred(i), seq[i]) for i ∈ [0, n)
        // load_before[i]  = load BEFORE processing seq[i]
        // load_after[i]   = load AFTER  processing seq[i]
        // arrive_t[i]     = cumulative travel time at seq[i]
        // t_seg_prefix[i] = Σ t_seg[0..i-1]                (i ∈ [0, n+1])
        // pickup_suffix[i]= # of pickups in seq[i..n-1]    (i ∈ [0, n+1])
        std::vector<float> t_seg;
        std::vector<int>   load_before;
        std::vector<int>   load_after;
        std::vector<float> arrive_t;
        std::vector<float> t_seg_prefix;
        std::vector<int>   pickup_suffix;
        int                initial_load = 0;
        osmium::object_id_type from_node = 0;
        bool               valid = false;
    };
    bool build_profile(const std::vector<SequenceStop>& seq,
                       int initial_load,
                       osmium::object_id_type from_node,
                       SeqProfile& out) const;

    std::vector<AgentState>    agents_;
    std::vector<TaskRecord>    tasks_;
    ZonePairForecast           forecast_;

    // ── Scratch buffers (avoid 4 vec allocs per cheapest_insertion call) ───
    // cheapest_insertion_with_profile is called N_agents × (1+H) times per
    // task — ~420 calls per task on a 60-agent / H=6 config. Re-allocating 4
    // local vectors each call (heap malloc/free) burns ~1ms per task just on
    // allocator overhead. Hoisting them as mutable members reuses the same
    // capacity across the entire episode (assign() keeps capacity).
    mutable std::vector<float> scratch_seg_to_P_;
    mutable std::vector<float> scratch_seg_P_out_;
    mutable std::vector<float> scratch_seg_D_in_;
    mutable std::vector<float> scratch_seg_D_out_;

    // ── Bookkeeping for metrics ─────────────────────────────────────────────
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

    // Static A* shortest-path road distance from -> to. Returns +inf if no
    // path. PathHelper caches per (from, to).
    float seg_cost(osmium::object_id_type from,
                   osmium::object_id_type to) const;

    // Paper eq. 11 cost of an ordered sequence starting at `from` with
    // initial load = |in_flight|. Returns +inf if any leg unreachable.
    PaperCost paper_cost(osmium::object_id_type from,
                         const std::vector<SequenceStop>& seq,
                         int initial_load,
                         float alpha) const;

    // Best (pos_p, pos_d) insertion of task t into agent a (capacity-feasible
    // only). Returns the resulting paper-cost delta (new - old). If no
    // feasible insertion exists, delta = +inf and pos_* = -1.
    struct InsertionResult {
        float cost_delta = 0.f;
        int   pos_p      = -1;
        int   pos_d      = -1;
        std::vector<SequenceStop> resulting_sequence;
    };
    InsertionResult cheapest_insertion(const AgentState& a,
                                       const TaskRecord& t) const;

    // Same as cheapest_insertion but operates on a TEMPORARY sequence (the
    // post-insertion sequence of the outer iteration). Used by the two-step
    // lookahead to evaluate ΔC(k+2)|h for a virtual call h inserted into the
    // already-modified sequence. Does NOT mutate any state.
    InsertionResult cheapest_insertion_in_seq(
        const std::vector<SequenceStop>& sequence,
        int initial_load,
        int capacity,
        osmium::object_id_type from_node,
        osmium::object_id_type virtual_pickup,
        osmium::object_id_type virtual_delivery) const;

    // Variant that takes a pre-built SeqProfile. Skips the O(n) base-sequence
    // precompute (~5n seg_cost calls), reusing the profile across all H
    // virtual-call iterations of the 2-step lookahead. Same paper-faithful
    // delta formula as cheapest_insertion_in_seq.
    InsertionResult cheapest_insertion_with_profile(
        const std::vector<SequenceStop>& sequence,
        const SeqProfile& profile,
        int capacity,
        osmium::object_id_type virtual_pickup,
        osmium::object_id_type virtual_delivery) const;

    // Allocate task t to its (vehicle, pos_p, pos_d) under the 2-step
    // criterion (paper eq. 15). Returns true on commit.
    bool try_insert_task(int task_id, int step);

    // Replan the leg toward the new sequence[0].
    bool resync_leg(AgentState& a, int step);

    // Advance one step along the current path; fire pickup/delivery on arrival.
    void advance_agent(AgentState& a, int step);

    int edge_arrival_step(osmium::object_id_type edge_id, int t_enter);

    // Re-register the agent's full remaining sequence on the shared
    // CongestionMap (Option O commit_plan-style). Called on every plan change.
    void recommit_route(AgentState& a, int step);

    // Update forecast (called from inject_task).
    void register_arrival(osmium::object_id_type pickup_node,
                          osmium::object_id_type delivery_node);
};

#endif // SOTA_HYBRID_ADAPTIVE_PREDICTIVE_SOLVER_HPP
