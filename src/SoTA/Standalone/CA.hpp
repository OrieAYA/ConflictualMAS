#ifndef SOTA_FAITHFUL_CA_SOLVER_HPP
#define SOTA_FAITHFUL_CA_SOLVER_HPP

#include "SoTA/SolverFramework.hpp"
#include <vector>

// Asadi+2025 (GECCO) — congestion-aware A* + order queue. Details: ARCHITECTURE.md §6.
class FaithfulCASolver : public ISolver {
public:
    FaithfulCASolver() = default;
    ~FaithfulCASolver() override = default;

    // Hyperparams (untuned; paper Bayesian-optimises on a 15x15 grid)
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

        // assigned, undelivered orders; served front-to-back
        std::vector<int> task_queue;
        bool active_is_pickup_leg = true;

        // on board (<= 1 here); audit only
        std::vector<int> in_flight_task_ids;
        int  capacity = 1;

        // end of sequence, from commit_agent_route
        osmium::object_id_type plan_tail_node = 0;
        int                    plan_tail_step = 0;

        // planning failed: retry backoff
        int  stalled_until = -1;

        // congestion footprint on shared map
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

    // not cached: cost depends on start step
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

    // BPR-aware A*; h = euclidean / speed
    BPRPath bpr_a_star(osmium::object_id_type from,
                       osmium::object_id_type to,
                       int start_step) const;

    // gamma-weighted marginal append cost
    float decision_cost(const AgentState& a,
                        const TaskRecord& t,
                        int step) const;

    bool try_allocate_one(int step);

    // current leg target, 0 if idle
    osmium::object_id_type leg_target(const AgentState& a) const;

    // fire front stop, advance cursor
    void fire_stop(AgentState& a, int step);

    // plan next leg; handles zero-length legs
    bool advance_plan(AgentState& a, int step);

    void advance_agent(AgentState& a, int step);
    int  edge_arrival_step(osmium::object_id_type edge_id, int t_enter);

    // republish footprint, refresh plan_tail_*
    void recommit_route(AgentState& a, int step);
};

#endif // SOTA_FAITHFUL_CA_SOLVER_HPP