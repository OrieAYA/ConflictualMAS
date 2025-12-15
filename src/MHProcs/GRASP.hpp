#ifndef GRASP_HPP
#define GRASP_HPP

#include "../Box.hpp"
#include "../Common/Hashes.hpp"
#include "../Pathfinding.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>

// Structure pour stocker une solution GRASP
struct GRASPSolution {
    std::vector<osmium::object_id_type> tour;
    double total_distance;
    bool is_valid;
    
    GRASPSolution() : total_distance(0.0), is_valid(false) {}
    
    void reset() {
        tour.clear();
        total_distance = 0.0;
        is_valid = false;
    }
};

// Paramètres de configuration GRASP
struct GRASPParams {
    int max_iterations = 100;
    double alpha = 0.3;              // Paramètre de randomisation (0.0 = greedy pur, 1.0 = aléatoire pur)
    int local_search_iterations = 50; // Nombre d'itérations pour la recherche locale
    bool use_2opt = true;            // Utiliser l'amélioration 2-opt
    bool use_3opt = false;           // Utiliser l'amélioration 3-opt (plus coûteuse)
    
    GRASPParams() = default;
};

// Classe dédiée aux algorithmes GRASP
class GRASPSolver {
private:
    GeoBox& geo_box;
    std::mt19937 rng;
    
    // Cache des distances pour éviter les recalculs
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash> distance_cache;

public:
    explicit GRASPSolver(GeoBox& box);
    
    // Méthode principale GRASP pour un groupe
    bool solve_single_group(
        const std::vector<osmium::object_id_type>& objective_nodes,
        int group_id,
        const GRASPParams& params = GRASPParams{}
    );

private:
    // Phase de construction
    GRASPSolution greedy_randomized_construction(
        const std::vector<osmium::object_id_type>& objective_nodes,
        const GRASPParams& params
    );
    
    // Phase de recherche locale
    GRASPSolution local_search(
        const GRASPSolution& initial_solution,
        const GRASPParams& params
    );
    
    // Améliorations locales
    GRASPSolution two_opt_improvement(const GRASPSolution& solution);
    GRASPSolution three_opt_improvement(const GRASPSolution& solution);
    
    // Méthodes utilitaires
    void build_distance_cache(const std::vector<osmium::object_id_type>& nodes);
    double get_distance(osmium::object_id_type node1, osmium::object_id_type node2);
    double calculate_tour_distance(const std::vector<osmium::object_id_type>& tour);
    
    std::vector<osmium::object_id_type> find_shortest_path(
        osmium::object_id_type start, 
        osmium::object_id_type end
    );
    
    void apply_tour_to_ways(
        const std::vector<osmium::object_id_type>& tour,
        int group_id
    );
    
    void update_way_group(osmium::object_id_type way_id, int new_group);
    
    // Construction de la liste restreinte de candidats (RCL)
    std::vector<osmium::object_id_type> build_restricted_candidate_list(
        osmium::object_id_type current_node,
        const std::unordered_set<osmium::object_id_type>& unvisited,
        double alpha
    );
};

#endif // GRASP_HPP