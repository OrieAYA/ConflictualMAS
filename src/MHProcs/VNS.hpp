#ifndef VNS_HPP
#define VNS_HPP

#include "../Box.hpp"
#include "../Common/Hashes.hpp"
#include "../Pathfinding.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>

// Structure pour stocker une solution VNS
struct VNSSolution {
    std::vector<osmium::object_id_type> tour;
    double total_distance;
    bool is_valid;
    
    VNSSolution() : total_distance(0.0), is_valid(false) {}
    
    void reset() {
        tour.clear();
        total_distance = 0.0;
        is_valid = false;
    }
    
    VNSSolution(const VNSSolution& other) = default;
    VNSSolution& operator=(const VNSSolution& other) = default;
};

// Paramètres de configuration VNS
struct VNSParams {
    int max_iterations = 200;
    int max_neighborhoods = 4;          // Nombre de structures de voisinage différentes
    int local_search_iterations = 100;  // Itérations pour la recherche locale
    double shaking_intensity = 0.3;     // Intensité des perturbations (0.1-0.5)
    bool use_first_improvement = true;  // Arrêter dès la première amélioration
    bool diversification = true;        // Utiliser la diversification
    
    VNSParams() = default;
};

// Énumération des types de voisinage
enum class NeighborhoodType {
    TWO_OPT = 0,      // Échange de 2 arêtes
    THREE_OPT = 1,    // Échange de 3 arêtes  
    RELOCATE = 2,     // Déplacement d'un nœud
    SWAP = 3          // Échange de deux nœuds
};

// Classe dédiée aux algorithmes VNS
class VNSSolver {
private:
    GeoBox& geo_box;
    std::mt19937 rng;
    
    // Cache des distances pour éviter les recalculs
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash> distance_cache;

public:
    explicit VNSSolver(GeoBox& box);
    
    // Méthode principale VNS pour un groupe
    bool solve_single_group(
        const std::vector<osmium::object_id_type>& objective_nodes,
        int group_id,
        const VNSParams& params = VNSParams{}
    );

private:
    // Génération de solution initiale
    VNSSolution generate_initial_solution(const std::vector<osmium::object_id_type>& nodes);
    VNSSolution nearest_neighbor_heuristic(const std::vector<osmium::object_id_type>& nodes);
    
    // Phase de secousse (shaking)
    VNSSolution shaking(
        const VNSSolution& solution, 
        int neighborhood_index, 
        const VNSParams& params
    );
    
    // Recherche locale dans un voisinage donné
    VNSSolution local_search(
        const VNSSolution& solution, 
        NeighborhoodType neighborhood_type,
        const VNSParams& params
    );
    
    // Structures de voisinage spécifiques
    VNSSolution two_opt_neighborhood(const VNSSolution& solution, bool first_improvement = true);
    VNSSolution three_opt_neighborhood(const VNSSolution& solution, bool first_improvement = true);
    VNSSolution relocate_neighborhood(const VNSSolution& solution, bool first_improvement = true);
    VNSSolution swap_neighborhood(const VNSSolution& solution, bool first_improvement = true);
    
    // Perturbations pour la phase de secousse
    VNSSolution random_k_opt(const VNSSolution& solution, int k);
    VNSSolution random_relocations(const VNSSolution& solution, int num_relocations);
    VNSSolution random_swaps(const VNSSolution& solution, int num_swaps);
    
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
    
    // Vérification de validité
    bool is_valid_tour(const std::vector<osmium::object_id_type>& tour, 
                      const std::vector<osmium::object_id_type>& original_nodes);
};

#endif // VNS_HPP