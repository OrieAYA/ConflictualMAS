#ifndef VNS_HPP
#define VNS_HPP

#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include "Legacy/Common/Memory.hpp"
#include <vector>
#include <random>
#include <unordered_set>

// ============================================================================
// STRUCTURES
// ============================================================================

struct VNSParams {
    int max_iterations;
    int k_max;                    // Nombre de voisinages
    int max_iterations_vnd;       // Itérations max pour VND
    bool use_two_opt;
    bool use_insertion;
    bool use_swap;
    bool use_add_remove;
    
    VNSParams() 
        : max_iterations(100), k_max(5), max_iterations_vnd(50),
          use_two_opt(true), use_insertion(true), use_swap(true),
          use_add_remove(true) {}
};

struct VNSResult {
    std::vector<osmium::object_id_type> solution;
    double fitness;
    double distance;
    int num_pois;
    long long execution_time_ms;
    bool is_valid;
    
    VNSResult() 
        : fitness(0.0), distance(0.0), num_pois(0), 
          execution_time_ms(0), is_valid(false) {}
};

// ============================================================================
// SOLVEUR VNS
// ============================================================================
class VNSSolver {
private:
    GeoBox& geo_box;
    Pathfinder& pathfinder;
    GlobalMemory& global_memory;

    bool verify_solution_sequence(const std::vector<osmium::object_id_type>& solution);
    void diagnose_poi_connectivity();
    
    std::vector<int> characteristics;
    double max_distance_constraint;
    
    std::vector<osmium::object_id_type> all_pois;
    std::mt19937 rng;
    
    // Solution courante et meilleure
    std::vector<osmium::object_id_type> current_solution;
    std::vector<osmium::object_id_type> best_solution;
    double best_fitness;
    
    // ========================================
    // INITIALISATION
    // ========================================
    void collect_rewardable_pois(const std::vector<int>& chars);
    std::vector<osmium::object_id_type> generate_initial_solution();
    std::vector<osmium::object_id_type> greedy_construction();
    std::vector<osmium::object_id_type> greedy_construction_from(osmium::object_id_type start);
    
    // ========================================
    // SHAKE (Perturbation)
    // ========================================
    std::vector<osmium::object_id_type> shake(
        const std::vector<osmium::object_id_type>& solution,
        int k
    );
    
    std::vector<osmium::object_id_type> random_remove(
        const std::vector<osmium::object_id_type>& solution,
        int num_to_remove
    );
    
    std::vector<osmium::object_id_type> random_swap(
        const std::vector<osmium::object_id_type>& solution,
        int num_swaps
    );
    
    std::vector<osmium::object_id_type> random_insert(
        const std::vector<osmium::object_id_type>& solution,
        int num_inserts
    );
    
    // ========================================
    // VND (Variable Neighborhood Descent)
    // ========================================
    std::vector<osmium::object_id_type> variable_neighborhood_descent(
        const std::vector<osmium::object_id_type>& solution,
        const VNSParams& params
    );
    
    // ========================================
    // VOISINAGES (Neighborhoods)
    // ========================================
    std::vector<osmium::object_id_type> two_opt(
        const std::vector<osmium::object_id_type>& solution
    );
    
    std::vector<osmium::object_id_type> best_insertion(
        const std::vector<osmium::object_id_type>& solution
    );
    
    std::vector<osmium::object_id_type> best_swap(
        const std::vector<osmium::object_id_type>& solution
    );
    
    std::vector<osmium::object_id_type> add_best_poi(
        const std::vector<osmium::object_id_type>& solution
    );
    
    std::vector<osmium::object_id_type> remove_worst_poi(
        const std::vector<osmium::object_id_type>& solution
    );
    
    std::vector<osmium::object_id_type> or_opt(
        const std::vector<osmium::object_id_type>& solution,
        int segment_size
    );
    
    // ========================================
    // RÉPARATION
    // ========================================
    std::vector<osmium::object_id_type> repair_solution(
        const std::vector<osmium::object_id_type>& solution
    );
    
    std::vector<osmium::object_id_type> greedy_repair(
        const std::vector<osmium::object_id_type>& solution
    );
    
    // ========================================
    // ÉVALUATION
    // ========================================
    double calculate_fitness(const std::vector<osmium::object_id_type>& solution);
    double calculate_distance(const std::vector<osmium::object_id_type>& solution);
    bool is_feasible(const std::vector<osmium::object_id_type>& solution);
    
    // ========================================
    // UTILITAIRES
    // ========================================
    double get_distance(osmium::object_id_type node1, osmium::object_id_type node2);
    int get_poi_reward(osmium::object_id_type poi_id);
    
    void print_iteration(int iteration, int k, double fitness, int num_pois, double distance);
    void print_improvement(const std::string& operator_name, double old_fitness, double new_fitness);

    int total_shake_calls;
    int total_vnd_calls;
    int total_improvements;
    std::vector<osmium::object_id_type> initial_solution_backup;

    void diagnose_vns_behavior(
        const std::vector<osmium::object_id_type>& initial_sol,
        const std::vector<osmium::object_id_type>& final_sol,
        int total_iterations,
        int improvements_count
    );

public:
    VNSSolver(GeoBox& box, Pathfinder& pf, GlobalMemory& mem);
    
    VNSResult solve(
        const std::vector<int>& agent_characteristics,
        double distance_constraint,
        const VNSParams& params
    );
};

#endif // VNS_HPP
