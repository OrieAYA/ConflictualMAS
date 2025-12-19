#include "Agent.hpp"
#include "Box.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>

Agent::Agent(GeoBox& box, std::vector<int> a_char, std::vector<osmium::object_id_type> init_sol) : geo_box(box), agent_charact(a_char), initial_solution(init_sol) {}

int Agent::objective_function(Agent& AgentSys, const std::vector<osmium::object_id_type>& objective_nodes) {
    int val = 0;

    for (const auto& node : objective_nodes){
        for (const int& vals : AgentSys.geo_box.data.nodes[node].groupes){
            val = val + AgentSys.agent_charact[vals];
        }
    }

    return val;
}

std::vector<std::vector<osmium::object_id_type>> Agent::get_neighbors(Agent& AgentSys, const std::vector<osmium::object_id_type>& objective_nodes) {
    std::vector<std::vector<osmium::object_id_type> > neighbors;
    for (size_t i = 0; i < objective_nodes.size(); i++) {
        for (size_t j = i + 1; j < objective_nodes.size(); j++) {
            std::vector<osmium::object_id_type> neighbor = objective_nodes;
            std::swap(neighbor[i], neighbor[j]);
            neighbors.push_back(neighbor);
        }
    }
    return neighbors;
}

std::vector<osmium::object_id_type> Agent::tabu_search(Agent& AgentSys, int max_iterations, int tabu_list_size) {
    std::vector<osmium::object_id_type> best_solution = AgentSys.initial_solution;
    std::vector<osmium::object_id_type> current_solution = AgentSys.initial_solution;
    std::vector<std::vector<osmium::object_id_type> > tabu_list;

    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<std::vector<osmium::object_id_type> > neighbors = AgentSys.get_neighbors(AgentSys, current_solution);
        std::vector<osmium::object_id_type> best_neighbor;
        int best_neighbor_fitness = std::numeric_limits<int>::max();

        for (const std::vector<osmium::object_id_type>& neighbor : neighbors) {
            if (std::find(tabu_list.begin(), tabu_list.end(), neighbor) == tabu_list.end()) {
                int neighbor_fitness = AgentSys.objective_function(AgentSys, neighbor);
                if (neighbor_fitness < best_neighbor_fitness) {
                    best_neighbor = neighbor;
                    best_neighbor_fitness = neighbor_fitness;
                }
            }
        }

        if (best_neighbor.empty()) {
            // No non-tabu neighbors found,
            // terminate the search
            break;
        }

        current_solution = best_neighbor;
        tabu_list.push_back(best_neighbor);
        if (tabu_list.size() > tabu_list_size) {
            // Remove the oldest entry from the
            // tabu list if it exceeds the size
            tabu_list.erase(tabu_list.begin());
        }

        if (AgentSys.objective_function(AgentSys, best_neighbor)
            < AgentSys.objective_function(AgentSys, best_solution)) {
            // Update the best solution if the
            // current neighbor is better
            best_solution = best_neighbor;
        }
    }

    return best_solution;
}