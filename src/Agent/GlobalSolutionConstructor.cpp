#include "GlobalSolutionConstructor.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>

GlobalSolutionConstructor::GlobalSolutionConstructor(
    GeoBox& box,
    Pathfinder& pf,
    GlobalMemory& mem
) : geo_box(box),
    pathfinder(pf),
    global_memory(mem) {}

float GlobalSolutionConstructor::objective_function(const GlobalSolution& solution) {
    float reward = 0.0f;
    
    for (const auto& [meta_agent_1, local_solution_1] : solution.solution_to_meta_agent) {
        for (const auto& [meta_agent_2, local_solution_2] : solution.solution_to_meta_agent) {
            if (meta_agent_1 != meta_agent_2) {
                float similarity = global_memory.calculate_similarity(local_solution_1, local_solution_2);
                reward = reward + similarity;
            }
        }
    }
    
    return reward;
}

std::vector<GlobalSolution> GlobalSolutionConstructor::get_neighbors(const GlobalSolution& solution) {
    std::vector<GlobalSolution> neighbors;
    
    for (const auto& [meta_agent, local_solution] : solution.solution_to_meta_agent) {
        for (const auto& best_solutions : meta_agent->validated_pbest) {
            if(best_solutions != solution.solution_to_meta_agent.at(meta_agent)){
                GlobalSolution neighbor = solution;
                neighbor.solution_to_meta_agent[meta_agent] = best_solutions;
                neighbors.push_back(neighbor);
            }
        }
    }
    return neighbors;
}

GlobalSolution GlobalSolutionConstructor::tabu_search(int max_iterations, int tabu_list_size) {

    GlobalSolution best_solution = initialize_global_solution();
    GlobalSolution current_solution = best_solution;
    std::vector<GlobalSolution> tabu_list;

    for (int iter = 0; iter < max_iterations; iter++) {
        std::cout << "On est ok ici 1." << iter << std::endl;
        std::vector<GlobalSolution> neighbors = get_neighbors(current_solution);
        std::cout << "On est ok ici 2." << iter << std::endl;
        GlobalSolution best_neighbor;
        float best_neighbor_fitness = std::numeric_limits<float>::max();

        for (const GlobalSolution& neighbor : neighbors) {
            if (std::find(tabu_list.begin(), tabu_list.end(), neighbor) == tabu_list.end()) {
                float neighbor_fitness = objective_function(neighbor);
                if (neighbor_fitness < best_neighbor_fitness) {
                    best_neighbor = neighbor;
                    best_neighbor_fitness = neighbor_fitness;
                }
            }
        }

        std::cout << "On est ok ici 3." << iter << std::endl;

        if (best_neighbor.empty()) {
            break;
        }

        current_solution = best_neighbor;
        tabu_list.push_back(best_neighbor);

        std::cout << "On est ok ici 4." << iter << std::endl;

        if (static_cast<int>(tabu_list.size()) > tabu_list_size) {
            tabu_list.erase(tabu_list.begin());
        }

        std::cout << "On est ok ici 5." << iter << std::endl;

        float current_reward = objective_function(current_solution);
        float best_reward = objective_function(best_solution);

        std::cout << "On est ok ici 6." << iter << std::endl;

        if (current_reward < best_reward) {
            best_solution = best_neighbor;
        }
    }

    std::cout << "On est ok ici final search" << std::endl;

    return best_solution;
}

void GlobalSolutionConstructor::add_meta_agent_config(const MetaAgentConfig& config) {
    meta_agent_configs.push_back(config);
}

void GlobalSolutionConstructor::run_all_meta_agents() {
    MetaAgents.clear();
    meta_agent_configurations.clear();
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "LANCEMENT DE " << meta_agent_configs.size() << " META-AGENT(S)\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    auto global_start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < meta_agent_configs.size(); i++) {
        const auto& config = meta_agent_configs[i];
        
        std::cout << "\n" << std::string(80, '-') << "\n";
        std::cout << "META-AGENT #" << (i + 1) << " : " << config.name << "\n";
        std::cout << std::string(80, '-') << "\n\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Créer et lancer le MetaAgent
        MetaAgent* meta_agent = new MetaAgent(
            geo_box,
            pathfinder,
            global_memory,
            config.characteristics,
            config.params
        );
        
        Solution gbest = meta_agent->run_meta_search();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        
        MetaAgents.push_back(meta_agent);
        meta_agent_configurations[meta_agent] = config;
        
        std::cout << "\n✓ META-AGENT #" << (i + 1) << " TERMINÉ\n";
        std::cout << "  Temps: " << duration.count() << " secondes\n";
        std::cout << "  GBest Fitness: " << meta_agent->local_visited_solutions[gbest] << "\n";
        std::cout << "  GBest POIs: " << gbest.POIs.size() << "\n";
        std::cout << "  GBest Distance: " << gbest.cost << " m\n";
        std::cout << "  Agents créés: " << static_cast<int>(meta_agent->get_agent_count()) << "\n";
        std::cout << "  Solutions validées: " << meta_agent->get_validated_pbest().size() << "\n";
    }
    
    auto global_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(global_end - global_start);
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TOUS LES META-AGENTS TERMINÉS\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Temps total: " << total_duration.count() << " secondes\n";
    std::cout << "Meta-Agents exécutés: " << MetaAgents.size() << "\n";
    
    int total_validated = 0;
    for (const auto& meta_agent : MetaAgents) {
        total_validated += static_cast<int>(meta_agent->validated_pbest.size());
    }
    std::cout << "Solutions validées totales: " << total_validated << "\n";
    
    std::cout << std::string(80, '=') << "\n\n";
}

GlobalSolution GlobalSolutionConstructor::initialize_global_solution() {
    if (MetaAgents.empty()) {
        std::cerr << "ERREUR: Aucun résultat de MetaAgent disponible!\n";
        std::cerr << "Appelez run_all_meta_agents() avant initialize_global_solution()\n";
        return GlobalSolution();
    }
    
    initial_solution.solution_to_meta_agent.clear();
    
    for (const auto& meta_agent : MetaAgents) {
        initial_solution.solution_to_meta_agent[meta_agent] = meta_agent->gbest_solution;
        std::cout << "  " << meta_agent_configurations[meta_agent].name << " → GBest (" 
                  << meta_agent->gbest_solution.POIs.size() << " POIs, fitness=" 
                  << meta_agent->local_visited_solutions[meta_agent->gbest_solution] << ")\n";
    }
    
    // Calculer reward et cost
    initial_solution.reward = objective_function(initial_solution);
    initial_solution.cost = 0.0f;
    for (const auto& [meta_result, solution] : initial_solution.solution_to_meta_agent) {
        initial_solution.cost += solution.cost;
    }

    return initial_solution;
}

void GlobalSolutionConstructor::print_summary() const {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RÉSUMÉ GLOBAL\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    if (MetaAgents.empty()) {
        std::cout << "Aucun résultat disponible.\n";
        return;
    }
    
    for (size_t i = 0; i < MetaAgents.size(); i++) {
        MetaAgent* meta_agent = MetaAgents[i];

        auto config_it = meta_agent_configurations.find(meta_agent);
        
        std::cout << "Meta-Agent #" << (i + 1) << ": " << config_it->second.name << "\n";
        std::cout << std::string(60, '-') << "\n";
        std::cout << "  GBest Fitness: " << meta_agent->local_visited_solutions[meta_agent->gbest_solution] << "\n";
        std::cout << "  GBest POIs: " << meta_agent->gbest_solution.POIs.size() << "\n";
        std::cout << "  GBest Distance: " << meta_agent->gbest_solution.cost << " m / " 
                  << global_memory.length_constraint << " m ";
        std::cout << "(" << std::fixed << std::setprecision(1)
                  << (meta_agent->gbest_solution.cost / global_memory.length_constraint * 100.0) << "%)\n";
        std::cout << "\n";
    }
    
    double best_fitness = 0;
    size_t best_index = 0;
    for (size_t i = 0; i < MetaAgents.size(); i++) {
        if (MetaAgents[i]->local_visited_solutions[MetaAgents[i]->gbest_solution] > best_fitness) {
            best_fitness = MetaAgents[i]->local_visited_solutions[MetaAgents[i]->gbest_solution];
            best_index = i;
        }
    }
    
    std::cout << "MEILLEUR OVERALL: " << meta_agent_configurations.find(MetaAgents[best_index])->second.name 
              << " (fitness=" << best_fitness << ")\n";
    std::cout << std::string(80, '=') << "\n\n";
}