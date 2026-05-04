#ifndef PSO_HPP
#define PSO_HPP

#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Hashes.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include "Legacy/Common/Memory.hpp"
#include <vector>
#include <random>

struct MTTDSParticle {
    std::vector<osmium::object_id_type> position;
    std::vector<osmium::object_id_type> best_position;
    double fitness;
    double best_fitness;
    double distance;
    
    MTTDSParticle() : fitness(0.0), best_fitness(0.0), distance(0.0) {}
};

struct MTTDSPSOParams {
    int num_particles = 30;
    int max_iterations = 100;
    double w = 0.7;
    double c1 = 1.5;
    double c2 = 1.5;
    double mutation_rate = 0.15;
    bool use_local_search = true;
    
    MTTDSPSOParams() = default;
};

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

class MTTDS_PSOSolver {
private:
    GeoBox& geo_box;
    Pathfinder& pathfinder;
    GlobalMemory& global_memory;
    std::mt19937 rng;
    
    std::vector<MTTDSParticle> swarm;
    std::vector<osmium::object_id_type> global_best_position;
    double global_best_fitness;
    double global_best_distance;
    
    double max_distance_constraint;
    std::vector<int> characteristics;
    std::vector<osmium::object_id_type> all_pois;

    std::vector<osmium::object_id_type> crossover_with_gbest(
        const std::vector<osmium::object_id_type>& position,
        const std::vector<osmium::object_id_type>& gbest
    );

    std::vector<osmium::object_id_type> add_random_poi(
        const std::vector<osmium::object_id_type>& solution
    );

public:
    MTTDS_PSOSolver(GeoBox& box, Pathfinder& pf, GlobalMemory& mem);
    
    MTTDSPSOResult solve(
        const std::vector<int>& agent_characteristics,
        double distance_constraint,
        const MTTDSPSOParams& params = MTTDSPSOParams{}
    );

private:
    void collect_rewardable_pois(const std::vector<int>& chars);
    void initialize_swarm(const MTTDSPSOParams& params);
    
    std::vector<osmium::object_id_type> generate_random_solution();
    std::vector<osmium::object_id_type> generate_greedy_solution();
    std::vector<osmium::object_id_type> generate_greedy_solution_from(osmium::object_id_type start);
    
    void update_particle(MTTDSParticle& particle, const MTTDSPSOParams& params);
    std::vector<osmium::object_id_type> apply_moves(
        const std::vector<osmium::object_id_type>& position,
        double cognitive_weight,
        double social_weight
    );
    
    std::vector<osmium::object_id_type> add_poi_operator(const std::vector<osmium::object_id_type>& solution);
    std::vector<osmium::object_id_type> remove_poi_operator(const std::vector<osmium::object_id_type>& solution);
    std::vector<osmium::object_id_type> swap_poi_operator(const std::vector<osmium::object_id_type>& solution);
    std::vector<osmium::object_id_type> local_search(const std::vector<osmium::object_id_type>& solution);
    std::vector<osmium::object_id_type> mutate_solution(const std::vector<osmium::object_id_type>& solution, double mutation_rate);
    
    double calculate_fitness(const std::vector<osmium::object_id_type>& solution);
    double calculate_distance(const std::vector<osmium::object_id_type>& solution);
    bool is_feasible(const std::vector<osmium::object_id_type>& solution);
    
    double get_distance(osmium::object_id_type node1, osmium::object_id_type node2);
    std::vector<osmium::object_id_type> get_path(osmium::object_id_type node1, osmium::object_id_type node2);
    int get_poi_reward(osmium::object_id_type poi_id);
    
    std::vector<osmium::object_id_type> repair_solution(const std::vector<osmium::object_id_type>& solution);
};

#endif