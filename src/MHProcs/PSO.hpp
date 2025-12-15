#ifndef PSO_HPP
#define PSO_HPP

#include "../Box.hpp"
#include "../Common/Hashes.hpp"
#include "../Pathfinding.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>

// Structure pour représenter une particule (solution)
struct Particle {
    std::vector<osmium::object_id_type> position;    // Tour actuel
    std::vector<osmium::object_id_type> velocity;    // "Vitesse" (changements à appliquer)
    std::vector<osmium::object_id_type> best_position; // Meilleure position personnelle
    double fitness;                                   // Distance du tour actuel
    double best_fitness;                             // Meilleure distance personnelle
    
    Particle() : fitness(std::numeric_limits<double>::max()), 
                 best_fitness(std::numeric_limits<double>::max()) {}
    
    void reset() {
        position.clear();
        velocity.clear();
        best_position.clear();
        fitness = std::numeric_limits<double>::max();
        best_fitness = std::numeric_limits<double>::max();
    }
};

// Structure pour stocker une solution PSO
struct PSOSolution {
    std::vector<osmium::object_id_type> tour;
    double total_distance;
    bool is_valid;
    
    PSOSolution() : total_distance(0.0), is_valid(false) {}
    
    void reset() {
        tour.clear();
        total_distance = 0.0;
        is_valid = false;
    }
};

// Paramètres de configuration PSO
struct PSOParams {
    int num_particles = 30;        // Nombre de particules dans l'essaim
    int max_iterations = 150;      // Nombre maximum d'itérations
    double w = 0.7;               // Coefficient d'inertie
    double c1 = 1.5;              // Coefficient cognitif (attraction vers meilleur personnel)
    double c2 = 1.5;              // Coefficient social (attraction vers meilleur global)
    double mutation_rate = 0.1;   // Taux de mutation pour diversification
    bool use_local_search = true; // Utiliser une recherche locale
    
    PSOParams() = default;
};

// Classe dédiée aux algorithmes PSO
class PSOSolver {
private:
    GeoBox& geo_box;
    std::mt19937 rng;
    
    // Cache des distances
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash> distance_cache;
    
    // Essaim de particules
    std::vector<Particle> swarm;
    std::vector<osmium::object_id_type> global_best_position;
    double global_best_fitness;

public:
    explicit PSOSolver(GeoBox& box);
    
    // Méthode principale PSO pour un groupe
    bool solve_single_group(
        const std::vector<osmium::object_id_type>& objective_nodes,
        int group_id,
        const PSOParams& params = PSOParams{}
    );

private:
    // Initialisation de l'essaim
    void initialize_swarm(
        const std::vector<osmium::object_id_type>& nodes,
        const PSOParams& params
    );
    
    // Génération de positions initiales
    std::vector<osmium::object_id_type> generate_random_tour(
        const std::vector<osmium::object_id_type>& nodes
    );
    
    std::vector<osmium::object_id_type> nearest_neighbor_tour(
        const std::vector<osmium::object_id_type>& nodes,
        osmium::object_id_type start_node
    );
    
    // Mise à jour des particules
    void update_particle_velocity(
        Particle& particle,
        const PSOParams& params
    );
    
    void update_particle_position(
        Particle& particle,
        const std::vector<osmium::object_id_type>& nodes
    );
    
    // Opérateurs pour TSP adapté à PSO
    std::vector<osmium::object_id_type> apply_swap_sequence(
        const std::vector<osmium::object_id_type>& tour,
        const std::vector<osmium::object_id_type>& swaps
    );
    
    std::vector<osmium::object_id_type> compute_swap_sequence(
        const std::vector<osmium::object_id_type>& from_tour,
        const std::vector<osmium::object_id_type>& to_tour
    );
    
    std::vector<osmium::object_id_type> combine_swap_sequences(
        const std::vector<osmium::object_id_type>& seq1,
        const std::vector<osmium::object_id_type>& seq2,
        double weight1,
        double weight2
    );
    
    // Amélioration locale
    std::vector<osmium::object_id_type> local_search_2opt(
        const std::vector<osmium::object_id_type>& tour
    );
    
    // Mutation pour diversification
    std::vector<osmium::object_id_type> mutate_tour(
        const std::vector<osmium::object_id_type>& tour,
        double mutation_rate
    );
    
    // Évaluation des solutions
    double evaluate_fitness(const std::vector<osmium::object_id_type>& tour);
    
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
    
    // Validation
    bool is_valid_tour(
        const std::vector<osmium::object_id_type>& tour,
        const std::vector<osmium::object_id_type>& original_nodes
    );
};

#endif // PSO_HPP