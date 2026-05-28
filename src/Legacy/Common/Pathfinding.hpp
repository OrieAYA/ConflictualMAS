#ifndef PATHFINDING_HPP
#define PATHFINDING_HPP

#include "Environment/GeoBox/Box.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <functional>

class Pathfinder;

struct Path {
    std::vector<osmium::object_id_type> path_edges;
    std::vector<osmium::object_id_type> path_nodes;

    osmium::object_id_type node_extremity_left;
    osmium::object_id_type node_extremity_right;

    float cost = 0.0;

    Path() = default;
    Path(std::vector<osmium::object_id_type> edges, 
         osmium::object_id_type left, 
         osmium::object_id_type right, 
         float c = 0.0)
        : path_edges(edges), 
          node_extremity_left(left), 
          node_extremity_right(right),
          cost(c) {}
    
    bool operator==(const Path& other) const {
        return node_extremity_left == other.node_extremity_left && 
               node_extremity_right == other.node_extremity_right && 
               cost == other.cost && 
               path_edges == other.path_edges &&
               path_nodes == other.path_nodes;
    }
};

struct PathHash {
    std::size_t operator()(const Path& p) const {
        std::size_t h = 0;

        auto hash_combine = [&h](auto const& v) {
            std::hash<std::decay_t<decltype(v)>> hasher;
            h ^= hasher(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };

        hash_combine(p.node_extremity_left);
        hash_combine(p.node_extremity_right);
        hash_combine(p.cost);

        for(auto e : p.path_edges) hash_combine(e);
        for(auto n : p.path_nodes) hash_combine(n);

        return h;
    }
};

class Pathfinder {
public:
    static std::mutex geobox_modification_mutex;
    GeoBox& geo_box;

    // NOUVEAU : Structure pour le cache de recherche par POI
    struct SearchCache {
        std::unordered_map<osmium::object_id_type, float> GScore;
        std::unordered_map<osmium::object_id_type, std::pair<osmium::object_id_type, osmium::object_id_type>> cameFrom;
        std::unordered_set<osmium::object_id_type> closed;
        std::vector<osmium::object_id_type> open;
        float max_explored_distance = 0.0f;
        std::vector<Path> found_pois;  // POIs déjà trouvés
    };
    
    // Cache par POI de départ
    std::unordered_map<osmium::object_id_type, SearchCache> search_caches;

    explicit Pathfinder(GeoBox& box);
    
    // Méthodes principales de pathfinding
    bool Subgraph_construction_threadsafe(
        Pathfinder& PfSystem,
        std::vector<osmium::object_id_type> objective_nodes,
        int path_group = 2
    );

    bool Complete_subgraph_construction(
        Pathfinder& PfSystem,
        std::vector<osmium::object_id_type> objective_nodes,
        int path_group = 2
    );

    bool Subgraph_construction(
        Pathfinder& PfSystem,
        std::vector<osmium::object_id_type> objective_nodes,
        int path_group = 2
    );

    bool Connected_subgraph_methode(
        Pathfinder& PfSystem,
        const std::vector<osmium::object_id_type>& objective_nodes,
        int path_group = 2
    );

    // Algorithmes de recherche de chemin
    std::vector<osmium::object_id_type> A_Star_Search(
        const osmium::object_id_type& start_point,
        const osmium::object_id_type& end_point
    );

    // Algorithmes de recherche de chemin
    std::vector<Path> Neighbor_Search(
        const osmium::object_id_type& start_point,
        const std::unordered_set<int>& available_groups,
        const std::unordered_set<osmium::object_id_type>& already_visited_pois = {}  // Optionnel
    );

    std::vector<osmium::object_id_type> reconstruct_path(
        const std::unordered_map<osmium::object_id_type, std::pair<osmium::object_id_type, osmium::object_id_type>>& cameFrom,
        osmium::object_id_type actual_node
    );

    float heuristic(osmium::object_id_type act_node, osmium::object_id_type end_point);

    // ── Single-source Dijkstra ─────────────────────────────────────────────
    // Compute shortest-path distances (in metres along OSM way edges) from
    // `source` to every reachable node in the GeoBox. Returns a map
    // node_id -> distance. Used by SoTA allocation rules (TokenPassing,
    // CongestionAware, FaithfulCongestionAware) to evaluate the
    // current-node→pickup leg cost for ALL candidate agents in O(V+E log V)
    // instead of O(N × A*) — i.e. one search per task arrival rather than
    // n_active independent A* calls. The result is a snapshot of free-flow
    // shortest-path distances (static cost); BPR-aware refinement is left
    // to the downstream DbVNS-PDP planner that handles route execution.
    std::unordered_map<osmium::object_id_type, float>
        dijkstra_distances_from(osmium::object_id_type source);
    
    // Méthodes utilitaires
    void update_way_group(osmium::object_id_type way_id, int new_group);
    void update_way_group_threadsafe(osmium::object_id_type way_id, int new_group);
    double calculate_distance(osmium::object_id_type node1, osmium::object_id_type node2);
    osmium::object_id_type find_nearest_node(double lat, double lon);
};

#endif // PATHFINDING_HPP