#ifndef METAAGENT_HPP
#define METAAGENT_HPP

#include "../GeoBox/Box.hpp"/
#include "../Common/Hashes.hpp"
#include "../Common/Pathfinding.hpp"
#include "../Common/Memory.hpp"
#include "Agent.hpp"
#include <vector>
#include <random>

struct MetaAgentParams {
    int max_iterations = 150;
    double w = 0.7;
    double c1 = 1.5;
    double c2 = 1.5;
    double admissible_similarity_degree = 0.3;
    double mutation_rate = 0.1;
    bool use_local_search = true;
    
    MetaAgentParams() = default;
};

class MetaAgent {
public:
    GeoBox& geo_box;
    GlobalMemory memory;
    std::vector<int> characteristics;
    std::map<Solution, float> local_visited_Solutions;
    std::unordered_map<std::vector<osmium::object_id_type>, Agent> population;

    explicit MetaAgent(
        GlobalMemory memory, 
        std::vector<int> characteristics, 
        std::map<Solution, float> local_visited_Solutions, 
        std::vector<osmium::object_id_type> POI_sorted_by_reward
    );

    void init_population();

    std::vector<std::vector<osmium::object_id_type>> get_neighbors(
        Agent& AgentSys,
        const std::vector<osmium::object_id_type>& objective_nodes
    );

    std::vector<osmium::object_id_type> tabu_search(
        Agent& AgentSys,
        int max_iterations, 
        int tabu_list_size
    );

};

#endif // METAAGENT_HPP