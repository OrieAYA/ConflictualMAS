#ifndef AGENT_HPP
#define AGENT_HPP

#include "../Common/Pathfinding.hpp"
#include "../GeoBox/Box.hpp"
#include "../Common/Memory.hpp"
#include "MetaAgent.hpp"
#include <vector>
#include <deque>
#include <map>
#include <set>

class MetaAgent;

struct DecomposedSolution {
    Solution base;
    osmium::object_id_type forbidden_next;
};

class Agent {
private:
    GeoBox& geo_box;
    Pathfinder& PfSystem;
    GlobalMemory& memory;
    Solution initial_solution;
    MetaAgent* parent;
    std::unordered_set<int> characteristics;
    float max_reward_per_meter;
    int visited_POIs;
    std::unordered_map<Solution, std::unordered_set<osmium::object_id_type>> explored_solutions_cache;

    // Méthodes privées VNS Multi-Branch
    void calculate_max_reward_density();
    double estimate_upper_bound(const Solution& solution);
    int get_poi_reward(osmium::object_id_type poi);
    
    Solution greedy_construction(Solution& start_solution);
    Solution greedy_extend_from(const Solution& base, osmium::object_id_type forbidden_first);
    
    std::vector<DecomposedSolution> decompose_with_forbidden(const Solution& solution);
    
    std::vector<Solution> explore_multibranch(
        const Solution& base_solution,
        osmium::object_id_type forbidden_first,
        double threshold
    );
    
    Solution vnd_local_search(const Solution& solution);

public:
    Solution pbest;
    double pbest_fitness;

    // Constructeur - SIGNATURE IDENTIQUE
    Agent(
        GeoBox& box, 
        Pathfinder& pf,
        GlobalMemory& mem,
        std::vector<osmium::object_id_type> init_sol,
        MetaAgent* meta_agent
    );

    // Méthodes publiques - SIGNATURES IDENTIQUES
    bool evaluate_upper_bound(const Solution& solution);
    Solution agent_search(int max_iterations, int tabu_list_size);
    std::vector<Solution> explore_multibranch_adaptive(
        const Solution& base_solution,
        osmium::object_id_type forbidden_first,
        double threshold,
        int max_neighbors
    );
    
    const Solution& get_initial_solution() const { return initial_solution; }
    const Solution& get_pbest() const { return pbest; }
};

#endif // AGENT_HPP