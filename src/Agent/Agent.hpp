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
    DecomposedSolution* parent;
    std::unordered_map<osmium::object_id_type, DecomposedSolution*> childs;
    Solution base;
    std::unordered_set<osmium::object_id_type> forbidden;

    DecomposedSolution() : parent(nullptr) {}
    
    DecomposedSolution(const Solution& sol) 
        : parent(nullptr), base(sol) {
        std::unordered_set<osmium::object_id_type> visited(
            base.POIs.begin(), 
            base.POIs.end()
        );
        this->forbidden = visited;
    }
    
    DecomposedSolution(DecomposedSolution* p, const Solution& sol) 
        : parent(p), base(sol) {
        std::unordered_set<osmium::object_id_type> visited(
            base.POIs.begin(), 
            base.POIs.end()
        );
        this->forbidden = visited;
    }
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

    double estimate_upper_bound(DecomposedSolution* solution);
    int get_poi_reward(osmium::object_id_type poi);

    DecomposedSolution* greedy_construction(DecomposedSolution* current);
    
    std::vector<DecomposedSolution*> decompose_with_forbidden(DecomposedSolution* solution);
    
    void delete_decomposition_tree(DecomposedSolution* root);
    
    std::vector<Solution> explore_multibranch_adaptive(
        const Solution& base_solution,
        osmium::object_id_type forbidden_first,
        double threshold,
        int max_neighbors
    );

public:
    DecomposedSolution pbest;
    double pbest_fitness;

    Agent(
        GeoBox& box, 
        Pathfinder& pf,
        GlobalMemory& mem,
        std::vector<osmium::object_id_type> init_sol,
        MetaAgent* meta_agent
    );

    Solution agent_search(int max_iterations, int tabu_list_size);
    
    const Solution& get_initial_solution() const { return initial_solution; }
    const Solution& get_pbest() const { return pbest.base; }
};

#endif // AGENT_HPP