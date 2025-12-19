#ifndef AGENT_HPP
#define AGENT_HPP

#include "../Box.hpp"
#include "../Common/Hashes.hpp"
#include "../Pathfinding.hpp"
#include <vector>
#include <random>

class Agent {
public:
    GeoBox& geo_box;
    std::vector<int> agent_charact;
    std::vector<osmium::object_id_type> initial_solution;

    explicit Agent(GeoBox& box, std::vector<int> a_char, std::vector<osmium::object_id_type> init_sol);

    int objective_function(
        Agent& AgentSys,
        const std::vector<osmium::object_id_type>& objective_nodes
    );

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

#endif // AGENT_HPP