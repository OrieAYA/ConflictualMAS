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
    DiscoveryStep discover_step(
        osmium::object_id_type from,
        const MyData&          data,
        const std::unordered_set<osmium::object_id_type>& agent_positions = {},
        float                  max_cost = std::numeric_limits<float>::max()
    );

    // Convenience wrapper (no agent detection, returns path or nullptr).
    const ObjectivePath* discover_next_path(osmium::object_id_type from, const MyData& data);

    // Reset the incremental Dijkstra state for a given starting node.
    void reset_search_state(osmium::object_id_type from) {
        search_states_.erase(from);
    }

    // Drop every memoised path + search state (memory bound between episodes).
    void clear_paths() {
        paths_.clear();
        search_states_.clear();
        retired_.clear();
        obj_pair_count_ = 0;
    }

    // Erase the memoised paths of retired (delivered / picked-up) objective
    std::size_t purge_retired(
        const std::unordered_set<const ObjectivePath*>& live);

    std::size_t n_paths() const { return paths_.size(); }

private:
    std::unordered_map<PathKey, ObjectivePath, ObjPairHash> paths_;
    std::unordered_set<osmium::object_id_type>              objective_ids_;
    // Objective nodes whose task is done: their paths are dead weight, but can
    // only be freed once no agent points at them (see purge_retired).
    std::unordered_set<osmium::object_id_type>              retired_;
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

    //  Path-compute timing instrumentation
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
    void clear_paths();

    // Reset the Dijkstra search state for one starting node (TAM agent search).
    void reset_agent_search(osmium::object_id_type from, int group_id);

    // Remove an objective node from its group cache (post pickup/delivery).
    void remove_objective_node(osmium::object_id_type node_id, int group_id);

    // Free the memoised paths of retired objective nodes across every group,
    // skipping those still referenced by an agent. Returns the count dropped.
    std::size_t purge_retired_paths(
        const std::unordered_set<const ObjectivePath*>& live);

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