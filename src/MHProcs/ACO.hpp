#ifndef ACO_HPP
#define ACO_HPP

#include "../Box.hpp"
#include "../Common/Hashes.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Structure représentant une fourmi
struct Ant {
    std::vector<osmium::object_id_type> tour;
    double tour_length;
    std::unordered_set<osmium::object_id_type> visited;
    
    Ant() : tour_length(0.0) {}
    
    void reset() {
        tour.clear();
        tour_length = 0.0;
        visited.clear();
    }
};

// Paramètres de configuration ACO
struct ACOParams {
    int num_ants = 25;
    int max_iterations = 50;
    double alpha = 1.0;      // Importance des phéromones
    double beta = 2.0;       // Importance de la distance (1/distance)
    double rho = 0.5;        // Taux d'évaporation
    double Q = 100.0;        // Constante de dépôt de phéromones
    double initial_pheromone = 1.0;
    
    ACOParams() = default;
};

// Forward declaration de Pathfinder pour éviter les dépendances circulaires
class Pathfinder;

// Classe dédiée aux algorithmes ACO
class ACOSolver {
private:
    GeoBox& geo_box;

public:
    explicit ACOSolver(GeoBox& box);
    
    // Méthode principale ACO pour un groupe
    bool solve_single_group(
        const std::vector<osmium::object_id_type>& objective_nodes,
        int group_id,
        const ACOParams& params = ACOParams{}
    );

private:
    // Méthodes internes ACO
    void construct_ant_tour(
        Ant& ant,
        const std::vector<osmium::object_id_type>& objective_nodes,
        const std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& distance_cache,
        const std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& pheromones,
        const ACOParams& params
    );

    osmium::object_id_type choose_next_node(
        osmium::object_id_type current_node,
        const std::vector<osmium::object_id_type>& objective_nodes,
        const std::unordered_set<osmium::object_id_type>& visited,
        const std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& distance_cache,
        const std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& pheromones,
        const ACOParams& params
    );

    void update_pheromones(
        std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& pheromones,
        const std::vector<Ant>& ants,
        const ACOParams& params
    );

    void apply_tour_to_ways(
        const std::vector<osmium::object_id_type>& tour,
        int group_id
    );

    // Méthodes utilitaires
    std::vector<osmium::object_id_type> find_shortest_path(
        osmium::object_id_type start, 
        osmium::object_id_type end
    );
    
    void update_way_group(osmium::object_id_type way_id, int new_group);
};

#endif // ACO_HPP