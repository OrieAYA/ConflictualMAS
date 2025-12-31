#ifndef AGENT_HPP
#define AGENT_HPP

#include "../Common/Pathfinding.hpp"
#include "../GeoBox/Box.hpp"
#include "../Common/Memory.hpp"
#include <vector>
#include <deque>
#include <map>

class Agent {
private:
    GeoBox& geo_box;
    Pathfinder& PfSystem;
    GlobalMemory& memory;
    std::vector<int> agent_charact;
    Solution initial_solution;
    Solution actual_solution;
    double fitness;
    double best_fitness;

    std::map<Solution, float>* meta_local_memory;
    std::vector<Solution>* validated_pbest;
    double similarity_threshold;
    
    // NOUVEAU : Gestion exploration voisinage
    int max_neighbors_per_exploration;  // ← AJOUTER
    std::map<osmium::object_id_type, int> explored_neighbors_count;  // ← AJOUTER

    bool is_better_solution(const Solution& sol1, const Solution& sol2);
    std::vector<Solution> get_add_neighbors(const Solution& sol);
    std::vector<Solution> get_backtrack_neighbors(const Solution& sol, int backtrack_depth);
    std::vector<Solution> get_all_neighbors(const Solution& sol, int max_backtrack);
    bool is_tabu(const Solution& sol, const std::deque<std::vector<osmium::object_id_type>>& tabu_list);

public:
    Agent(
        GeoBox& box, 
        Pathfinder& pf,
        GlobalMemory& mem,
        std::vector<int> a_char, 
        std::vector<osmium::object_id_type> init_sol,
        std::map<Solution, float>* meta_memory = nullptr,
        std::vector<Solution>* pbest_list = nullptr,
        double sim_threshold = 1.0,
        int max_neighbors = 10  // ← AJOUTER
    );

    int objective_function(const Solution& sol);
    Solution tabu_search(int max_iterations, int tabu_list_size);
    
    const Solution& get_initial_solution() const { return initial_solution; }
    const Solution& get_actual_solution() const { return actual_solution; }
    double get_fitness() const { return fitness; }
    double get_best_fitness() const { return best_fitness; }
};

#endif // AGENT_HPP