#ifndef PATHFINDING_HPP
#define PATHFINDING_HPP

#include "Box.hpp"
#include <vector>
#include <unordered_map>
#include <memory>

struct PairHash {
    size_t operator()(const std::pair<osmium::object_id_type, osmium::object_id_type>& p) const {
        return std::hash<osmium::object_id_type>{}(p.first) ^ std::hash<osmium::object_id_type>{}(p.second);
    }
};

// Classe principale pour le pathfinding
class Pathfinder {
private:
    GeoBox& geo_box;
    
public:
    explicit Pathfinder(GeoBox& box);
    
    // Méthodes principales
    std::vector<osmium::object_id_type> A_Star_Search(
        const osmium::object_id_type& start_point,
        const osmium::object_id_type& end_point,
        int path_group = 2
    );

    std::vector<osmium::object_id_type> reconstruct_path(
        const std::unordered_map<osmium::object_id_type, std::pair<osmium::object_id_type, osmium::object_id_type>>& cameFrom,
        osmium::object_id_type actual_node,
        int path_group
    );

    float heuristic(osmium::object_id_type node);
    
    // Méthodes utilitaires
    void update_node_weight(osmium::object_id_type node_id, float new_weight);
    void update_way_group(osmium::object_id_type way_id, int new_group);
    double calculate_distance(osmium::object_id_type node1, osmium::object_id_type node2);
    osmium::object_id_type find_nearest_node(double lat, double lon);
};

#endif // PATHFINDING_HPP