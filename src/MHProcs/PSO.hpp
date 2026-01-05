#ifndef PSO_HPP
#define PSO_HPP

#include "../GeoBox/Box.hpp"
#include "../Common/Hashes.hpp"
#include "../Common/Pathfinding.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <chrono>

// Structure pour représenter une particule (solution MTTDS)
struct MTTDSParticle {
    std::vector<osmium::object_id_type> position;      // POIs visités
    std::vector<osmium::object_id_type> best_position; // Meilleure position personnelle
    double fitness;                                     // Récompense actuelle
    double best_fitness;                               // Meilleure récompense personnelle
    double distance;                                    // Distance actuelle
    
    MTTDSParticle() : fitness(0.0), best_fitness(0.0), distance(0.0) {}
};

// Paramètres de configuration PSO pour MTTDS
struct MTTDSPSOParams {
    int num_particles = 30;        // Nombre de particules dans l'essaim
    int max_iterations = 100;      // Nombre maximum d'itérations
    double w = 0.7;               // Coefficient d'inertie
    double c1 = 1.5;              // Coefficient cognitif
    double c2 = 1.5;              // Coefficient social
    double mutation_rate = 0.15;  // Taux de mutation
    bool use_local_search = true; // Utiliser recherche locale
    
    MTTDSPSOParams() = default;
};

// Résultat d'une résolution PSO MTTDS
struct MTTDSPSOResult {
    std::vector<osmium::object_id_type> solution;
    double fitness;
    double distance;
    int num_pois;
    long long execution_time_ms;
    bool is_valid;
    
    MTTDSPSOResult() : fitness(0.0), distance(0.0), num_pois(0), 
                       execution_time_ms(0), is_valid(false) {}
};

// Classe PSO adaptée pour MTTDS
class MTTDS_PSOSolver {
private:
    GeoBox& geo_box;
    Pathfinder& pathfinder;
    std::mt19937 rng;
    
    // Cache des distances et chemins
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash> distance_cache;
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, std::vector<osmium::object_id_type>, PairHash> path_cache;
    
    // Essaim de particules
    std::vector<MTTDSParticle> swarm;
    std::vector<osmium::object_id_type> global_best_position;
    double global_best_fitness;
    double global_best_distance;
    
    // Contraintes et paramètres du problème
    double max_distance_constraint;
    std::vector<int> characteristics;
    std::vector<osmium::object_id_type> all_pois;

public:
    MTTDS_PSOSolver(GeoBox& box, Pathfinder& pf);
    
    // Méthode principale de résolution
    MTTDSPSOResult solve(
        const std::vector<int>& agent_characteristics,
        double distance_constraint,
        const MTTDSPSOParams& params = MTTDSPSOParams{}
    );

private:
    // Initialisation
    void collect_rewardable_pois(const std::vector<int>& chars);
    void build_distance_cache();
    void initialize_swarm(const MTTDSPSOParams& params);
    
    // Génération de solutions initiales
    std::vector<osmium::object_id_type> generate_random_solution();
    std::vector<osmium::object_id_type> generate_greedy_solution();
    std::vector<osmium::object_id_type> generate_greedy_solution_from(osmium::object_id_type start);
    
    // Mise à jour des particules
    void update_particle(MTTDSParticle& particle, const MTTDSPSOParams& params);
    std::vector<osmium::object_id_type> apply_moves(
        const std::vector<osmium::object_id_type>& position,
        double cognitive_weight,
        double social_weight
    );
    
    // Opérateurs de voisinage
    std::vector<osmium::object_id_type> add_poi_operator(
        const std::vector<osmium::object_id_type>& solution
    );
    std::vector<osmium::object_id_type> remove_poi_operator(
        const std::vector<osmium::object_id_type>& solution
    );
    std::vector<osmium::object_id_type> swap_poi_operator(
        const std::vector<osmium::object_id_type>& solution
    );
    
    // Recherche locale
    std::vector<osmium::object_id_type> local_search(
        const std::vector<osmium::object_id_type>& solution
    );
    
    // Mutation
    std::vector<osmium::object_id_type> mutate_solution(
        const std::vector<osmium::object_id_type>& solution,
        double mutation_rate
    );
    
    // Évaluation
    double calculate_fitness(const std::vector<osmium::object_id_type>& solution);
    double calculate_distance(const std::vector<osmium::object_id_type>& solution);
    bool is_feasible(const std::vector<osmium::object_id_type>& solution);
    
    // Utilitaires
    double get_distance(osmium::object_id_type node1, osmium::object_id_type node2);
    std::vector<osmium::object_id_type> get_path(osmium::object_id_type node1, osmium::object_id_type node2);
    int get_poi_reward(osmium::object_id_type poi_id);
    
    // Réparation de solutions infaisables
    std::vector<osmium::object_id_type> repair_solution(
        const std::vector<osmium::object_id_type>& solution
    );
};

#endif // PSO_HPP