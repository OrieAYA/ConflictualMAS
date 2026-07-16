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

// ── RegionStatsGrid — task density + congestion heatmap over the city bbox ───
//
// Lightweight 32×32 grid over the GeoBox bbox. Tracks:
//   - Per-cell task arrival density (sliding window of `window_steps` steps)
//   - Per-cell mean BPR multiplier (refreshed every kCacheRefreshSteps)
//
// Used by PolicyFeatures::area_heat_pickup = density_norm × congestion_norm.
// Total memory: ~40 KB per episode (negligible). Total runtime: O(1) per task
// arrival + O(N_cells × N_sampled_edges) every kCacheRefreshSteps steps.
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
    // to compute the `deliverability` feature: steps_remaining / delivery_steps.
    //   speed_mps   — agent speed in meters per step (typically 5.0 → ~18 km/h)
    //   total_steps — episode horizon in simulation steps (typically 3600)
    // Without these, the policy cannot judge whether an accepted task can
    // physically be completed before the episode ends.
    float speed_mps   = 5.0f;
    int   total_steps = 3600;

    // ── Planning strategy used by DeliveryAgent::receive_task() ─────────────
    // At most one flag is set per run (EpisodeRunner::prepare_run dispatch);
    // all flags off = cheapest insertion (Solomon-style (pos_P, pos_D) argmin).
    // Capacity + pickup-before-delivery invariants hold in every branch.

    // Double-Horizon insertion [Mitrovic-Minic et al. 2004]: insertion cost
    // weighted by slack preservation for long-horizon positions.
    bool  planning_use_double_horizon = false;

    // DbVNS replanning (the paper's planner): on every acceptance the agent
    // reoptimises its full remaining sequence (in-flight head preserved).
    bool  planning_use_dbvns = false;

    // ALNS-PDP lifelong replanning [Ropke & Pisinger 2006].
    bool  planning_use_alns  = false;

    // Active objective-policy dispatcher used by DeliveryAgent::bid_for_task:
    //   kMAPPO / kIPPO / kMAPPER / kHybrid → the corresponding IBidPolicy
    //   kRMCA → non-learning marginal-cost scorer (Chen+2021), deterministic,
    //           no buffer write.
    // Modes that never consult a policy (Greedy, Random, ...) bypass
    // bid_for_task entirely, so the kind is ignored for them.
    enum class PolicyKind { kMAPPO = 0, kIPPO = 1, kMAPPER = 2,
                            kHybrid = 3, kRMCA = 4 };
    PolicyKind active_policy = PolicyKind::kMAPPO;

    // System-state snapshot (kGlobSz = 20 floats) refreshed once per step by
    // the episode runner. bid_for_task forwards it into each recorded
    // experience so the MAPPO critic sees the state the decision was made in.
    float cur_global_state[20] = {};

    // ── Spatial heatmap : task density + congestion per cell ───────────────
    // Initialised once at construction (via geo_box bbox), reset between
    // episodes. EpisodeRunner calls register_task() on arrival and refresh
    // happens lazily inside area_heat() via advance_time hooks.
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
    // `out` (optional) reports the TD-A* comparison for reward computation.
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
    // departing `depart_step`, with each edge sampled at its running arrival
    // time (deeper into the route = later t). Returns a distance-equivalent
    // cost (meters) in the SAME units as ObjectivePath::cost, so it is a
    // drop-in replacement for the static matrix entry. Single forward pass,
    // O(#edges), reads the load profile as-is (no fixed-point). Returns
    // FLT_MAX when the path is invalid. The walk is ORIENTED: edges are
    // replayed in the from→to direction regardless of the cached direction.
    // self_weight > 0 excludes the caller's own committed load from every
    // edge read (an agent evaluating its own route sees n others, not n+1).
    float bpr_path_cost(osmium::object_id_type from, osmium::object_id_type to,
                        int group_id, int depart_step, float speed_mps,
                        int self_weight = 0);

    // Same BPR replay but in TIME units (steps), walking `path` oriented so
    // the traversal starts at `from`. Used to compare the remaining travel
    // time of the current route against a TD-A* alternative in the SAME
    // units (push_rerouted_path). Returns INT_MAX if `from` is not on the path.
    int path_travel_steps(const ObjectivePath& path,
                          osmium::object_id_type from,
                          int depart_step, float speed_mps) const;

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

    float congestion_cost(osmium::object_id_type way_id,
                          float base_cost, float distance_meters, int t) const;

    // Exploration switch for the bid policies: when true (training), the
    // bid decision is SAMPLED from Bernoulli(μ); when false (evaluation),
    // it is the deterministic threshold μ >= 0.5. Set by the episode runner.
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
    // committed plan. Bounds are clamped to the CongestionMap window at ADD
    // time so removal is perfectly symmetric — removing un-clamped windows
    // after the clock advanced would subtract load that was never added
    // (corrupting other agents' / ghosts' contributions).
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
