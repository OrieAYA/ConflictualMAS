#ifndef PATHFINDING_HPP
#define PATHFINDING_HPP

#include "Box.hpp"
#include "Common/Hashes.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

// Classe principale pour le pathfinding
class Pathfinder {
private:
    static std::mutex geobox_modification_mutex;

public:
    GeoBox& geo_box;
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