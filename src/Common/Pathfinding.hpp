#ifndef PATHFINDING_HPP
#define PATHFINDING_HPP

#include "../GeoBox/Box.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>

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
        float search_coefficient,
        const std::unordered_set<osmium::object_id_type>& already_visited_pois = {}  // Optionnel
    );

    std::vector<osmium::object_id_type> reconstruct_path(
        const std::unordered_map<osmium::object_id_type, std::pair<osmium::object_id_type, osmium::object_id_type>>& cameFrom,
        osmium::object_id_type actual_node
    );

    float heuristic(osmium::object_id_type act_node, osmium::object_id_type end_point);
    
    // Méthodes utilitaires
    void update_way_group(osmium::object_id_type way_id, int new_group);
    void update_way_group_threadsafe(osmium::object_id_type way_id, int new_group);
    double calculate_distance(osmium::object_id_type node1, osmium::object_id_type node2);
    osmium::object_id_type find_nearest_node(double lat, double lon);
};

#endif // PATHFINDING_HPP