#ifndef OBJECTIVE_CACHE_HPP
#define OBJECTIVE_CACHE_HPP

#include "DMASforPD/Structures/ObjectiveNode.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>

// ---- Data types --------------------------------------------------------

struct ObjectivePath {
    osmium::object_id_type              node_a = 0;  // canonical: smaller id
    osmium::object_id_type              node_b = 0;  // canonical: larger id
    std::vector<osmium::object_id_type> edges;
    std::vector<osmium::object_id_type> nodes;

    float cost         = std::numeric_limits<float>::max(); // static: Dijkstra distance (meters)
    float dynamic_cost = std::numeric_limits<float>::max(); // TD-A* travel time (steps, BPR-adjusted)
    int   dynamic_step = -1;   // simulation step when dynamic_cost was last computed (-1 = never)

    std::vector<osmium::object_id_type> intermediate_objectives;

    bool valid()            const { return cost         < std::numeric_limits<float>::max(); }
    bool has_dynamic_cost() const { return dynamic_step >= 0; }
};

// Result of a one-shot time-dependent A* query.
// total_time     : accumulated travel time in steps (BPR-adjusted, rounded)
// total_distance : total geometric distance in meters (unweighted)
struct TDAStarResult {
    std::vector<osmium::object_id_type> edges;
    std::vector<osmium::object_id_type> nodes;
    float total_distance = std::numeric_limits<float>::max();
    int   total_time     = std::numeric_limits<int>::max();

    bool valid() const { return total_time < std::numeric_limits<int>::max(); }
};

struct ObjPairHash {
    std::size_t operator()(
        const std::pair<osmium::object_id_type, osmium::object_id_type>& p
    ) const noexcept;
};

// ---- ObjectiveGroupCache -----------------------------------------------

class ObjectiveGroupCache {
public:
    using PathKey = std::pair<osmium::object_id_type, osmium::object_id_type>;

    int                        group_id = 0;
    std::vector<ObjectiveNode> objective_nodes;

    void build_objective_set();
    void register_in_set(osmium::object_id_type node_id);  // O(1) insert into objective_ids_
    void episode_reset();   // clear objectives + search states + pair count; keep paths_
    bool is_complete() const;

    bool                              has_path (osmium::object_id_type a, osmium::object_id_type b) const;
    const ObjectivePath*              get_path (osmium::object_id_type a, osmium::object_id_type b) const;
    ObjectivePath*                    get_path_mutable(osmium::object_id_type a, osmium::object_id_type b);
    void                              store_path(const ObjectivePath& path);
    std::vector<const ObjectivePath*> paths_from(osmium::object_id_type node_id) const;

    bool contains_objective(osmium::object_id_type node_id) const;

    // Remove an objective node and all cached paths involving it.
    // Called after a pickup or delivery is completed.
    void remove_node(osmium::object_id_type node_id);

    // Result of one TAM expansion step.
    struct DiscoveryStep {
        enum class Type { Path, AgentPosition, Exhausted } type = Type::Exhausted;
        const ObjectivePath*   path       = nullptr; // valid when type == Path
        osmium::object_id_type agent_node = 0;       // valid when type == AgentPosition
        float                  cost       = 0.0f;    // g_score at agent_node
    };

    // Incremental Dijkstra — resumes between calls.
    // Stops at the next uncached objective node OR at any node in agent_positions.
    // agent_positions: current nodes of idle delivery agents (checked from delivery side only).
    // max_cost: spatial pruning — stops expansion when the smallest open-set
    //           cost exceeds this bound. Does NOT permanently exhaust the search
    //           (the node is pushed back to the heap), so a subsequent call with
    //           a larger max_cost will resume past the previous frontier.
    DiscoveryStep discover_step(
        osmium::object_id_type from,
        const MyData&          data,
        const std::unordered_set<osmium::object_id_type>& agent_positions = {},
        float                  max_cost = std::numeric_limits<float>::max()
    );

    // Convenience wrapper (no agent detection, returns path or nullptr).
    const ObjectivePath* discover_next_path(osmium::object_id_type from, const MyData& data);

    // Reset the incremental Dijkstra state for a given starting node.
    // Called before each new TAM agent search so stale closed-sets and
    // exhausted flags from prior tasks don't block re-discovery of agents
    // that have since moved. Memoised paths_ are NOT affected.
    void reset_search_state(osmium::object_id_type from) {
        search_states_.erase(from);
    }

    // Drop every memoised path + search state (memory bound between episodes).
    void clear_paths() {
        paths_.clear();
        search_states_.clear();
        obj_pair_count_ = 0;
    }

private:
    std::unordered_map<PathKey, ObjectivePath, ObjPairHash> paths_;
    std::unordered_set<osmium::object_id_type>              objective_ids_;
    int                                                      obj_pair_count_ = 0; // paths where both endpoints are objectives

    struct SearchState {
        using OpenEntry = std::pair<float, osmium::object_id_type>;
        std::unordered_map<osmium::object_id_type, float>                          g_score;
        std::unordered_map<osmium::object_id_type,
            std::pair<osmium::object_id_type, osmium::object_id_type>>             came_from;
        std::unordered_set<osmium::object_id_type>                                 closed;
        std::vector<OpenEntry>                                                      open;
        bool exhausted = false;
    };

    std::unordered_map<osmium::object_id_type, SearchState> search_states_;

    static PathKey make_key(osmium::object_id_type a, osmium::object_id_type b);

    ObjectivePath reconstruct_path(
        const SearchState&     state,
        osmium::object_id_type from,
        osmium::object_id_type to,
        float                  cost
    ) const;
};

// ---- PDPServerMemory ---------------------------------------------------

class PDPServerMemory {
public:
    GeoBox& geo_box;
    float   length_constraint = 0.0f;

    PDPServerMemory() = delete;
    explicit PDPServerMemory(GeoBox& box);

    // ── Path-compute timing instrumentation ─────────────────────────────────
    // Accumulates the total wallclock time (in microseconds for precision)
    // spent inside get_or_compute_path. Reset at episode start by the
    // EpisodeRunner; read at finalize() to derive (a) per-episode path-compute
    // cost in ms, and (b) pure-allocation time by subtracting in-offer path
    // time from total offer_task time. Static because path cache is global
    // per GeoBox; the simulation is single-threaded so a plain
    // static counter is sufficient.
    static long long path_compute_time_us();
    static void      reset_path_compute_time();

    void initialize_from_geobox();

    // Create an empty group cache if it doesn't already exist.
    // Needed for synthetic training tasks that don't come from a GeoBox objective group.
    void ensure_group(int group_id);

    // Direct A*-based path lookup (lazy, stores on miss).
    const ObjectivePath* get_or_compute_path(
        osmium::object_id_type from,
        osmium::object_id_type to,
        int group_id
    );

    // Incremental Dijkstra — no agent detection (wraps discover_step).
    const ObjectivePath* discover_next_path(osmium::object_id_type from, int group_id);

    // TAM-aware step: expands one node, also checks agent_positions.
    // max_cost prunes the Dijkstra expansion radius (see ObjectiveGroupCache::discover_step).
    ObjectiveGroupCache::DiscoveryStep discover_step(
        osmium::object_id_type from,
        int group_id,
        const std::unordered_set<osmium::object_id_type>& agent_positions = {},
        float                  max_cost = std::numeric_limits<float>::max()
    );

    // Refresh the dynamic cost of a cached path using TD-A*.
    // No-op if the path is not cached or dynamic_step >= current_step.
    // Returns the updated path (or nullptr if not cached).
    ObjectivePath* refresh_dynamic_cost(
        osmium::object_id_type from,
        osmium::object_id_type to,
        int   group_id,
        float speed_mps,
        const CongestionMap& congestion,
        int   current_step
    );

    // Store an externally computed path into a group cache.
    void store_path_in_group(int group_id, const ObjectivePath& path);

    // Register a new objective node into its group cache (called when a task is created).
    void add_objective_node(osmium::object_id_type node_id, int group_id);

    // Clear all objective registrations for a group (between episodes).
    // Preserves paths_ so A* results computed in prior episodes are reused.
    void reset_objectives(int group_id);

    // Drop every memoised path in every group. Called when the episode
    // CONTENT changes (new ep_seed) — replays of the same episode keep the
    // cache; keeping it across different episodes grows without bound.
    void clear_paths();

    // Reset the Dijkstra search state for one starting node (TAM agent search).
    // Ensures each new task allocation starts from a clean state so agents that
    // moved since the previous search are not missed. Memoised paths_ are kept.
    void reset_agent_search(osmium::object_id_type from, int group_id);

    // Remove an objective node from its group cache (post pickup/delivery).
    void remove_objective_node(osmium::object_id_type node_id, int group_id);

    bool is_objective_node(osmium::object_id_type node_id, int group_id) const;
    bool is_group_complete(int group_id) const;

    std::vector<const ObjectivePath*> get_paths_from(osmium::object_id_type node_id, int group_id) const;

    // start_time: integer step; agent_speed: meters per step.
    TDAStarResult time_dependent_astar(
        osmium::object_id_type from,
        osmium::object_id_type to,
        int                    start_time,
        float                  agent_speed,
        const CongestionMap&   congestion
    ) const;

private:
    std::unordered_map<int, ObjectiveGroupCache> group_caches_;

    ObjectivePath compute_path(osmium::object_id_type from, osmium::object_id_type to, int group_id);
    void          tag_intermediate_objectives(ObjectivePath& path, int group_id);
};

#endif // OBJECTIVE_CACHE_HPP
