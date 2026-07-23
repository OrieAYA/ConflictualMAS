#ifndef SOTA_HYBRID_ADAPTIVE_PREDICTIVE_SOLVER_HPP
#define SOTA_HYBRID_ADAPTIVE_PREDICTIVE_SOLVER_HPP

#include "SoTA/SolverFramework.hpp"
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

// Cortes+2009 (Transp. Sci.) — 2-step predictive insertion, eq.11/15. Details: ARCHITECTURE.md §6.
class HybridAdaptivePredictiveSolver : public ISolver {
public:
    HybridAdaptivePredictiveSolver() = default;
    ~HybridAdaptivePredictiveSolver() override = default;

    //  Tunable hyperparameters (paper §3.3 + §4.1)
    struct HParams {
        float alpha            = 1.0f;    // paper §3.3 waiting-cost weight (paper test α = 1)
        bool  enable_two_step  = true;    // §3.3 two-steps-ahead lookahead (false = myopic)
        int   forecast_top_h   = 4;       // §4.1 four-zone case uses H = 4
        float forecast_min_p   = 0.02f;   // skip virtual calls with prob below this
        int   demand_grid_dim  = 2;       // 2x2 = 4 zones, paper §4.2 preferred case
        int   forecast_n_intervals = 3;   // §4.1 three hourly ΔT over the episode
        int   forecast_min_samples = 12;  // below this, fall back to cumulative counts
    };
    HParams hparams;

    void          init(const SolverContext& ctx) override;
    void          inject_task(const ScheduledTask& task, int step) override;
    void          step(int timestep) override;
    SolverMetrics finalize() override;
    const char*   name() const override { return "HybridAdaptivePredictive"; }

private:
    // paper eq.7 sequence S_j(k)
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

        // congestion footprint on shared map
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
        // exponential retry backoff for stuck tasks
        int next_retry_step = 0;
        int retry_count     = 0;
    };

    // paper eq.11 cost split
    struct PaperCost {
        float travel  = 0.f;
        float waiting = 0.f;
        float total() const { return travel + waiting; }
    };

    // paper eq.12 demand forecast, OD-pair counts per interval
    struct ZonePairForecast {
        int dim = 3;
        float lat_min = 0.f, lat_step = 0.f;
        float lon_min = 0.f, lon_step = 0.f;

        // N_h counts keyed i*n+j; interval, then cumulative fallback
        std::unordered_map<int64_t, float> count_interval;
        std::unordered_map<int64_t, float> count_total;
        int   n_interval      = 0;   // samples in the current interval
        int   interval_index  = -1;  // which ΔT we are in
        int   interval_len    = 1;   // steps per ΔT

        // one road node per cell, 0 if empty
        std::vector<osmium::object_id_type> rep_node_of_cell;

        // node -> cell, -1 if out of bounds
        int cell_of(osmium::object_id_type id, const GeoBox& g) const;

        void register_arrival(int p_cell, int d_cell);

        // roll to the interval holding `step`
        void advance_interval(int step);

        // top-H virtual calls; prune before normalising (keeps sum p = 1)
        struct VirtualCall {
            osmium::object_id_type pickup_node;
            osmium::object_id_type delivery_node;
            float probability;
        };
        std::vector<VirtualCall> top_h(int H, float min_p, int min_samples) const;
    };

    const SolverContext*       ctx_   = nullptr;

    // non-owning view on the shared helper
    PathHelper*                paths_ = nullptr;
    static std::unique_ptr<PathHelper>& shared_path_helper();

    // per-episode distance cache, keyed on GeoBox
    struct SharedDistanceCache {
        const GeoBox*     gb = nullptr;
        // guards GeoBox address reuse across cities
        size_t            gb_node_count = 0;
        // dijkstra_from[src] = {node : distance}; symmetric lookup in seg_cost
        std::unordered_map<osmium::object_id_type,
                           std::unordered_map<osmium::object_id_type, float>>
            dijkstra_from;

        // entry count vs kPrewarmBudgetBytes
        size_t entries = 0;
        // sources refused: HAPC degraded to A*
        int    prewarm_refused = 0;
    };
    static SharedDistanceCache& shared_cache();

    // Dijkstra from src into the cache; skipped past the memory budget,
    // seg_cost then falls back to PathHelper A*
    void prewarm_from_node(osmium::object_id_type src) const;

    // per-episode ceiling
    static constexpr size_t kPrewarmBudgetBytes = 2ull * 1024 * 1024 * 1024;
    // conservative per-entry cost
    static constexpr size_t kPrewarmBytesPerEntry = 48;

    //  Precomputed sequence profile
    struct SeqProfile {
        // t_seg[i]        = seg(pred(i), seq[i]) for i ∈ [0, n)
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

    // reused across the ~420 insertion calls per task
    mutable std::vector<float> scratch_seg_to_P_;
    mutable std::vector<float> scratch_seg_P_out_;
    mutable std::vector<float> scratch_seg_D_in_;
    mutable std::vector<float> scratch_seg_D_out_;

    //  Bookkeeping for metrics
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

    //  Internal helpers

    // static shortest-path distance, +inf if none
    float seg_cost(osmium::object_id_type from,
                   osmium::object_id_type to) const;

    // paper eq.11 cost of a sequence
    PaperCost paper_cost(osmium::object_id_type from,
                         const std::vector<SequenceStop>& seq,
                         int initial_load,
                         float alpha) const;

    // cheapest capacity-feasible insertion; delta = +inf if none
    struct InsertionResult {
        float cost_delta = 0.f;
        int   pos_p      = -1;
        int   pos_d      = -1;
        std::vector<SequenceStop> resulting_sequence;
    };
    InsertionResult cheapest_insertion(const AgentState& a,
                                       const TaskRecord& t) const;

    // same, on a temporary sequence (2-step lookahead)
    InsertionResult cheapest_insertion_in_seq(
        const std::vector<SequenceStop>& sequence,
        int initial_load,
        int capacity,
        osmium::object_id_type from_node,
        osmium::object_id_type virtual_pickup,
        osmium::object_id_type virtual_delivery) const;

    // same, with a pre-built profile
    InsertionResult cheapest_insertion_with_profile(
        const std::vector<SequenceStop>& sequence,
        const SeqProfile& profile,
        int capacity,
        osmium::object_id_type virtual_pickup,
        osmium::object_id_type virtual_delivery) const;

    // allocate under paper eq.15
    bool try_insert_task(int task_id, int step);

    // replan toward sequence[0]
    bool resync_leg(AgentState& a, int step);

    // one step along the path; fires stops
    void advance_agent(AgentState& a, int step);

    int edge_arrival_step(osmium::object_id_type edge_id, int t_enter);

    // republish footprint on plan change
    void recommit_route(AgentState& a, int step);

    // forecast update
    void register_arrival(osmium::object_id_type pickup_node,
                          osmium::object_id_type delivery_node);
};

#endif // SOTA_HYBRID_ADAPTIVE_PREDICTIVE_SOLVER_HPP