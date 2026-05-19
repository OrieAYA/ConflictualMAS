#ifndef GLOBAL_MEMORY_HPP
#define GLOBAL_MEMORY_HPP

#include "ObjectiveCache.hpp"
#include "Environment/Congestion/CongestionMap.hpp"
#include "DMASforPD/Utility/PDPTask.hpp"
#include "DMASforPD/Utility/TimedPath.hpp"
#include "DMASforPD/TaskAgent/TaskAgent.hpp"
#include <unordered_map>
#include <vector>

// Top-level server memory for the PDP multi-agent system.
//
// Central registry for:
//   - Environment (geo_box + congestion state)
//   - Tasks in three lifecycle categories (available / allocated / finished)
//   - Delivery agents + their committed plans
//   - ObjectiveNode → Task lookup (O(1))
//   - Task → DeliveryAgent lookup (via task.agent_id)
//
// Environment consistency rule:
//   Whenever an agent's solution changes, call commit_plan().
//   commit_plan() removes the old congestion contribution, rebuilds timed
//   paths from the new solution, and registers the new contribution.
class PDPGlobalMemory {
public:
    GeoBox&         geo_box;
    Pathfinder&     pathfinder;
    PDPServerMemory server_memory;
    CongestionMap   congestion_map;
    TaskAgent       task_agent;

    // ---- Task lists (non-owning pointers into tasks_) -------------------
    // Direct access: use .size(), iterate, dereference pointers.
    std::vector<PDPTask*> available_tasks;   // Idle
    std::vector<PDPTask*> allocated_tasks;   // Assigned to a delivery agent
    std::vector<PDPTask*> finished_tasks;    // Completed

    // Set by EpisodeRunner before each offer_task call so that try_accept_task
    // can use it as the time_remaining feature (1 − step/total_steps).
    float cur_time_ratio = 0.f;

    // Set by EpisodeRunner once per episode (in run()). Used by try_accept_task
    // to compute the `deliverability` feature: steps_remaining / delivery_steps.
    //   speed_mps   — agent speed in meters per step (typically 5.0 → ~18 km/h)
    //   total_steps — episode horizon in simulation steps (typically 3600)
    // Without these, the policy cannot judge whether an accepted task can
    // physically be completed before the episode ends.
    float speed_mps   = 5.0f;
    int   total_steps = 3600;

    // ── Planning strategy used by DeliveryAgent::receive_task() ─────────────
    // false (default) → Cheapest insertion: pick (pos_P, pos_D) minimising
    //                    route-length delta f_p + f_d. Classical Solomon-style.
    // true  → Double-Horizon insertion [Mitrovic-Minic et al. 2004]:
    //          cost = (1-α_p)·f_p + α_p·g_p + (1-α_d)·f_d + α_d·g_d
    //          where g_p = (n−pos_P)·f_p (slack-decrease proxy at subsequent
    //          locations) and α ∈ {0 short-term, 0.25 long-term} based on the
    //          estimated arrival time of the new pickup / delivery vs current.
    //
    // EpisodeRunner copies this from cfg.use_double_horizon_planning at the
    // start of each run(). Capacity + pickup-before-delivery invariants are
    // preserved by construction in both branches.
    bool  planning_use_double_horizon = false;

    // Active learning-policy dispatcher used by DeliveryAgent::try_accept_task.
    // Set by EpisodeRunner at the start of each run() based on policy_mode:
    //   - PolicyMode::MAPPO          → kMAPPO   (shared actor + centralised critic)
    //   - PolicyMode::IPPO           → kIPPO    (shared actor + shared local critic,
    //                                            de Witt+2020 paper-faithful)
    //   - PolicyMode::MAPPER         → kMAPPER  (per-agent actor + per-agent critic +
    //                                            enhanced Ev with Gaussian mutation)
    //   - PolicyMode::FaithfulMAPPER → kFaithfulMAPPER (same architecture as MAPPER
    //                                            but paper-faithful Ev: probabilistic
    //                                            replacement + exact copy of best,
    //                                            no mutation noise)
    //   - PolicyMode::Hybrid         → kHybrid  (MAPPO base + per-agent residual)
    // try_accept_task routes the feature vector to the corresponding policy
    // singleton (ObjectiveDMPolicy / IPPOPolicy / MapperPolicy / FaithfulMapperPolicy /
    // HybridPolicy) and records the experience into the right buffer. Other modes
    // (Greedy, Random, ...) bypass try_accept_task entirely so the kind is ignored.
    enum class PolicyKind { kMAPPO = 0, kIPPO = 1, kMAPPER = 2,
                            kHybrid = 3, kFaithfulMAPPER = 4 };
    PolicyKind active_policy = PolicyKind::kMAPPO;

    PDPGlobalMemory() = delete;
    PDPGlobalMemory(GeoBox& box, Pathfinder& pf,
                    const CongestionParams& cparams  = {},
                    const TaskAgentParams&  taparams = {});

    PDPGlobalMemory(const PDPGlobalMemory&)            = delete;
    PDPGlobalMemory& operator=(const PDPGlobalMemory&) = delete;

    // ---- Task management ------------------------------------------------

    // Create a task and place it in available_tasks. Returns task_id.
    int add_task(const ObjectiveNode& pickup, const ObjectiveNode& delivery);

    // Lifecycle transitions (also update the three category lists).
    void assign_task  (int task_id, int agent_id);  // available → allocated
    void unassign_task(int task_id);                 // allocated → available
    void complete_task(int task_id);                 // allocated → finished

    // ---- Task queries ---------------------------------------------------

    PDPTask*       get_task(int task_id);
    const PDPTask* get_task(int task_id) const;

    // Convenience counts (same as list.size() but named).
    int count_available() const { return static_cast<int>(available_tasks.size()); }
    int count_allocated() const { return static_cast<int>(allocated_tasks.size()); }
    int count_finished () const { return static_cast<int>(finished_tasks.size()); }
    int count_total    () const { return count_available() + count_allocated() + count_finished(); }

    // O(1) lookup: which task owns this objective node?
    PDPTask* get_task_for_node(osmium::object_id_type node_id);

    // Filter allocated_tasks by agent.
    std::vector<const PDPTask*> tasks_for_agent(int agent_id) const;

    // ---- Delivery agent registry ----------------------------------------

    void           register_delivery_agent  (DeliveryAgent& agent);
    void           unregister_delivery_agent(int agent_id);
    DeliveryAgent* get_delivery_agent       (int agent_id);
    DeliveryAgent* get_agent_for_task       (int task_id);

    const std::unordered_map<int, DeliveryAgent*>& all_delivery_agents() const;

    // Read-only solution access (delegates to delivery_agents_).
    const AgentSolution* get_solution(int agent_id) const;

    // ---- Plan commit (environment sync) ---------------------------------
    //
    // Must be called after any solution change.
    //
    // 1. Removes the agent's previously committed congestion contribution.
    // 2. Walks the current solution: current_node → seq[0] → seq[1] → …
    // 3. Builds a TimedPath per leg (steps = ceil(dist/speed_mps), min 1).
    // 4. Registers the new contribution into congestion_map.
    // 5. Fills estimated_arrival for each SolutionStep in the solution.
    //
    // speed_mps: agent speed in meters per step.
    void commit_plan(int agent_id, float speed_mps);

    // Remove an objective node from its group cache after pickup or delivery.
    // Also removes it from node_to_task_id_ so it's no longer findable.
    void clear_objective(osmium::object_id_type node_id, int group_id);

    // Push a congestion-rerouted path to a delivery agent mid-traversal.
    // Refreshes the dynamic cost of the cached path, then calls
    // agent->push_updated_path() so the agent's edge cursor resumes correctly.
    // No-op if the path has not changed or no improvement is found.
    void push_rerouted_path(int agent_id, float speed_mps);

    // ---- Path cache (delegates to server_memory) ------------------------

    // Create an empty group cache if none exists (for synthetic training tasks).
    void ensure_task_group(int group_id);

    const ObjectivePath* get_or_compute_path(
        osmium::object_id_type from, osmium::object_id_type to, int group_id);
    const ObjectivePath* discover_next_path(osmium::object_id_type from, int group_id);
    bool is_group_complete(int group_id) const;

    // One-shot TD-A*. start_time: integer step; agent_speed: meters per step.
    TDAStarResult time_dependent_astar(
        osmium::object_id_type from,
        osmium::object_id_type to,
        int   start_time,
        float agent_speed) const;

    // ---- Simulation clock -----------------------------------------------

    int  current_time() const;
    void advance_time(int t_now);

    // ---- Episode reset --------------------------------------------------
    //
    // Clears all per-episode state (tasks, plans, congestion, clock) while
    // preserving the A* path cache (server_memory / group_caches_) so that
    // path costs accumulated across episodes remain available.
    // Must be called at the start of each new episode when the same
    // PDPGlobalMemory instance is reused across episodes.
    void reset_episode();

    // ---- Congestion (delegates to congestion_map) -----------------------

    void  register_agent_path  (const ObjectivePath& path, int start_time, float speed_mps);
    void  unregister_agent_path(const ObjectivePath& path, int start_time, float speed_mps);
    float congestion_cost(osmium::object_id_type way_id,
                          float base_cost, float distance_meters, int t) const;

private:
    int current_time_ = 0;

    // Owned task storage (stable addresses — pointers in the category lists remain valid).
    std::unordered_map<int, PDPTask> tasks_;

    // ObjectiveNode id → task_id (covers both pickup and delivery nodes).
    std::unordered_map<osmium::object_id_type, int> node_to_task_id_;

    // Delivery agent registry.
    std::unordered_map<int, DeliveryAgent*> delivery_agents_;

    // Per-agent committed timed paths: list of (path, departure_step) pairs.
    // Used to undo congestion contributions when the plan changes.
    std::unordered_map<int, std::vector<std::pair<TimedPath, int>>> committed_plans_;

    // Remove a task pointer from whichever category list currently holds it.
    static void remove_from_lists(PDPTask* task,
                                  std::vector<PDPTask*>& available,
                                  std::vector<PDPTask*>& allocated,
                                  std::vector<PDPTask*>& finished);

    void unregister_committed_plan(int agent_id);
    void register_committed_plan  (int agent_id, float speed_mps);
};

#endif // GLOBAL_MEMORY_HPP
