#ifndef GLOBAL_MEMORY_HPP
#define GLOBAL_MEMORY_HPP

#include "DMASforPD/Structures/ObjectiveCache.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "Environment/Simulation/TemporalGraph.hpp"
#include "DMASforPD/Structures/PDPTask.hpp"
#include "DMASforPD/Structures/TimePaths.hpp"
#include "DMASforPD/Agents/TaskAgent.hpp"
#include <deque>
#include <unordered_map>
#include <vector>

// ── RegionStatsGrid — task density + congestion heatmap over the city bbo
struct RegionStatsGrid {
    static constexpr int kDim                = 32;
    static constexpr int kSize               = kDim * kDim;
    static constexpr int kMaxEdgesPerCell    = 16;     // sampled at init for BPR queries
    static constexpr int kCacheRefreshSteps  = 50;     // congestion cache cadence

    bool   inited      = false;
    double min_lat = 0, min_lon = 0;
    double cell_h_lat = 1, cell_w_lon = 1;

    // ── Density tracking (sliding window of task arrivals) ──────────────────
    int    task_counts[kSize]  = {};
    int    max_count           = 1;
    int    window_steps        = 600;
    struct Event { int step; int cell; };
    std::deque<Event> task_events;

    // ── Congestion cache (refreshed every kCacheRefreshSteps) ───────────────
    float  cell_cong_cache[kSize] = {};   // mean BPR multiplier (≥ 1.0)
    float  max_cell_cong          = 1.f;  // for normalisation
    int    last_cache_refresh     = -kCacheRefreshSteps;

    // Per-cell sampled edge IDs (precomputed at init, capped at kMaxEdgesPerCell).
    std::vector<osmium::object_id_type> cell_edges[kSize];

    // Initialise grid bounds + precompute per-cell edge samples from geo_box.
    void init(const GeoBox& gb);

    // Hard reset between episodes (clears counts, keeps the edge sampling).
    void reset_episode();

    int  cell_of(double lat, double lon) const;
    void register_task(double lat, double lon, int step);
    void purge(int now);

    void refresh_congestion_cache(const CongestionMap& cmap,
                                   const GeoBox& gb,
                                   int step);

    float density_norm(double lat, double lon) const;     // ∈ [0,1]
    float congestion_norm(double lat, double lon) const;  // ∈ [0,1]
    float area_heat(double lat, double lon) const;        // density × congestion
};

// Top-level server memory for the PDP multi-agent system.
class PDPGlobalMemory {
public:
    GeoBox&         geo_box;
    PDPServerMemory server_memory;
    CongestionMap   congestion_map;   // edge temporal layer (occupancy / BPR)
    TemporalGraph   node_events;      // node temporal layer (task/objective events)
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
    float speed_mps   = 5.0f;
    int   total_steps = 3600;

    // ── Planning strategy used by DeliveryAgent::receive_task() ─────

    // Double-Horizon insertion [Mitrovic-Minic et al. 2004]: insertion cost
    // weighted by slack preservation for long-horizon positions.
    bool  planning_use_double_horizon = false;

    // DbVNS replanning (the paper's planner): on every acceptance the agent
    // reoptimises its full remaining sequence (in-flight head preserved).
    bool  planning_use_dbvns = false;

    // ALNS-PDP lifelong replanning [Ropke & Pisinger 2006].
    bool  planning_use_alns  = false;

    // Active objective-policy dispatcher used by DeliveryAgent::bid_for_task:
    enum class PolicyKind { kMAPPO = 0, kIPPO = 1, kMAPPER = 2,
                            kHybrid = 3, kRMCA = 4 };
    PolicyKind active_policy = PolicyKind::kMAPPO;

    // System-state snapshot (kGlobSz = 20 floats) refreshed once per step by
    float cur_global_state[20] = {};

    // ── Spatial heatmap : task density + congestion per cell ──────
    RegionStatsGrid region_grid;

    PDPGlobalMemory() = delete;
    explicit PDPGlobalMemory(GeoBox& box,
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
    DeliveryAgent*       get_delivery_agent (int agent_id);
    const DeliveryAgent* get_delivery_agent (int agent_id) const;
    DeliveryAgent* get_agent_for_task       (int task_id);

    const std::unordered_map<int, DeliveryAgent*>& all_delivery_agents() const;

    // Read-only solution access (delegates to delivery_agents_).
    const AgentSolution* get_solution(int agent_id) const;

    // ---- Plan commit (environment sync) ---------------------------------
    void commit_plan(int agent_id, float speed_mps);

    // Remove an objective node from its group cache after pickup or delivery.
    // Also removes it from node_to_task_id_ so it's no longer findable.
    void clear_objective(osmium::object_id_type node_id, int group_id);

    // Push a congestion-rerouted path to a delivery agent mid-traversal.
    struct RerouteOutcome {
        bool attempted = false;   // both routes were costed
        bool adopted   = false;   // >5% improvement, path pushed
        int  cur_steps = 0;       // BPR-replayed remaining time, current route
        int  tda_steps = 0;       // TD-A* alternative time
    };
    void push_rerouted_path(int agent_id, float speed_mps,
                            RerouteOutcome* out = nullptr);

    // When true, register_committed_plan snapshots each committed edge's
    // congestion level into the agent's plan_cong (movement policy feature).
    bool record_plan_congestion = false;

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

    // BPR replay of a CACHED STATIC path: congestion-adjusted travel cost when
    float bpr_path_cost(osmium::object_id_type from, osmium::object_id_type to,
                        int group_id, int depart_step, float speed_mps,
                        int self_weight = 0);

    // Same BPR replay but in TIME units (steps), walking `path` oriented so
    int path_travel_steps(const ObjectivePath& path,
                          osmium::object_id_type from,
                          int depart_step, float speed_mps) const;

    // ---- Simulation clock -----------------------------------------------

    int  current_time() const;
    void advance_time(int t_now);

    // Sweep period (steps) for the retired-path purge; 0 disables it.
    static constexpr int kRetiredPurgeSteps = 100;

    // Drop cached geometry of objective nodes whose task is done, keeping the
    // paths an agent is currently walking. Called from advance_time.
    void purge_retired_paths();

    // ---- Episode reset --------------------------------------------------
    void reset_episode();

    // ---- Congestion (delegates to congestion_map) -----------------------

    float congestion_cost(osmium::object_id_type way_id,
                          float base_cost, float distance_meters, int t) const;

    // Exploration switch for the bid policies: when true (training), the
    bool exploration_enabled = false;

private:
    int current_time_ = 0;

    // Owned task storage (stable addresses — pointers in the category lists remain valid).
    std::unordered_map<int, PDPTask> tasks_;

    // ObjectiveNode id → task_id (covers both pickup and delivery nodes).
    std::unordered_map<osmium::object_id_type, int> node_to_task_id_;

    // Delivery agent registry.
    std::unordered_map<int, DeliveryAgent*> delivery_agents_;

    // Exact ledger of the load increments actually applied for each agent's
    struct LoadWindow {
        osmium::object_id_type way;
        int t_lo, t_hi;
        int weight;
    };
    std::unordered_map<int, std::vector<LoadWindow>> committed_loads_;

    // Remove a task pointer from whichever category list currently holds it.
    static void remove_from_lists(PDPTask* task,
                                  std::vector<PDPTask*>& available,
                                  std::vector<PDPTask*>& allocated,
                                  std::vector<PDPTask*>& finished);

    void unregister_committed_plan(int agent_id);
    void register_committed_plan  (int agent_id, float speed_mps);
};

#endif // GLOBAL_MEMORY_HPP