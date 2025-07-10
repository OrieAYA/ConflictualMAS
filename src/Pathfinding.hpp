#ifndef PATHFINDING_HPP
#define PATHFINDING_HPP

#include "Box.hpp"
#include <vector>
#include <unordered_map>
#include <memory>

// Structure pour représenter un graphe avec cache des chemins
struct PathGraph {
    std::unordered_map<osmium::object_id_type, std::vector<osmium::object_id_type>> adjacency_list;
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, 
                      std::vector<osmium::object_id_type>, 
                      std::hash<std::string>> cached_paths;
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, 
                      double, 
                      std::hash<std::string>> path_distances;
    
    PathGraph() = default;
};

// Structure pour les phéromones (métaheuristique fourmis)
struct PheromoneMap {
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, 
                      double, 
                      std::hash<std::string>> pheromone_levels;
    double evaporation_rate = 0.1;
    double initial_pheromone = 1.0;
    
    PheromoneMap() = default;
};

// Classe principale pour le pathfinding
class Pathfinder {
private:
    const GeoBox& geo_box;
    PathGraph graph;
    PheromoneMap pheromones;
    
public:
    explicit Pathfinder(const GeoBox& box);
    
    // Méthodes principales
    void build_graph();
    std::vector<osmium::object_id_type> find_path_ant_colony(osmium::object_id_type start, 
                                                            osmium::object_id_type end);
    
    // Méthodes utilitaires
    double calculate_distance(osmium::object_id_type node1, osmium::object_id_type node2);
    osmium::object_id_type find_nearest_node(double lat, double lon);
    
    // Getters pour le graphe
    const PathGraph& get_graph() const { return graph; }
    const PheromoneMap& get_pheromones() const { return pheromones; }
};

#endif // PATHFINDING_HPP