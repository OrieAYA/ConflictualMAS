#include "GlobalSolutionConstructor.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

GlobalSolutionConstructor::GlobalSolutionConstructor(
    GeoBox& box,
    Pathfinder& pf,
    GlobalMemory& mem
) : geo_box(box),
    pathfinder(pf),
    global_memory(mem) {}

float GlobalSolutionConstructor::objective_function(const GlobalSolution& solution) {
    float reward = 0.0f;
    std::map<MetaAgentResult, Solution> actual_solution = solution.solution_to_meta_agent;
    
    for (const auto& [meta_agent_1, local_solution_1] : actual_solution) {
        for (const auto& [meta_agent_2, local_solution_2] : actual_solution) {
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
    std::map<MetaAgentResult, Solution> solution_map = solution.solution_to_meta_agent;
    
    for (const auto& [meta_agent, local_solution] : solution_map) {
        for (const auto& best_solutions : meta_agent.validated_pbest) {
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
        std::vector<GlobalSolution> neighbors
            = get_neighbors(current_solution);
        GlobalSolution best_neighbor;
        int best_neighbor_fitness
            = std::numeric_limits<int>::max();

        for (const GlobalSolution& neighbor : neighbors) {
            if (std::find(tabu_list.begin(),
                          tabu_list.end(), neighbor)
                == tabu_list.end()) {
                int neighbor_fitness
                    = objective_function(neighbor);
                if (neighbor_fitness
                    < best_neighbor_fitness) {
                    best_neighbor = neighbor;
                    best_neighbor_fitness
                        = neighbor_fitness;
                }
            }
        }

        if (best_neighbor.empty()) {
            break;
        }

        current_solution = best_neighbor;
        tabu_list.push_back(best_neighbor);
        if (tabu_list.size() > tabu_list_size) {
            tabu_list.erase(tabu_list.begin());
        }

        if (objective_function(best_neighbor)
            < objective_function(best_solution)) {
            best_solution = best_neighbor;
        }
    }

    return best_solution;
}

void GlobalSolutionConstructor::add_meta_agent_config(const MetaAgentConfig& config) {
    meta_agent_configs.push_back(config);
}

void GlobalSolutionConstructor::run_all_meta_agents() {
    results.clear();
    
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
        
        int gbest_fitness = 0;
        for (const auto& poi_id : gbest.POIs) {
            auto node_it = geo_box.data.nodes.find(poi_id);
            if (node_it != geo_box.data.nodes.end()) {
                for (const int group_id : node_it->second.groupes) {
                    if (group_id >= 0 && group_id < static_cast<int>(config.characteristics.size())) {
                        gbest_fitness += config.characteristics[group_id];
                    }
                }
            }
        }
        
        MetaAgentResult result;
        result.name = config.name;
        result.act_meta_agent = meta_agent;
        result.gbest = gbest;
        result.validated_pbest = meta_agent->get_validated_pbest();
        result.gbest_fitness = gbest_fitness;
        result.agent_count = static_cast<int>(meta_agent->get_agent_count());
        result.coverage_rate = meta_agent->get_coverage_rate();
        
        results.push_back(result);
        
        std::cout << "\n✓ META-AGENT #" << (i + 1) << " TERMINÉ\n";
        std::cout << "  Temps: " << duration.count() << " secondes\n";
        std::cout << "  GBest Fitness: " << gbest_fitness << "\n";
        std::cout << "  GBest POIs: " << gbest.POIs.size() << "\n";
        std::cout << "  GBest Distance: " << gbest.cost << " m\n";
        std::cout << "  Agents créés: " << result.agent_count << "\n";
        std::cout << "  Solutions validées: " << result.validated_pbest.size() << "\n";
        std::cout << "  Couverture: " << (result.coverage_rate * 100) << "%\n\n";
    }
    
    auto global_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(global_end - global_start);
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TOUS LES META-AGENTS TERMINÉS\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Temps total: " << total_duration.count() << " secondes\n";
    std::cout << "Meta-Agents exécutés: " << results.size() << "\n";
    
    int total_validated = 0;
    for (const auto& result : results) {
        total_validated += static_cast<int>(result.validated_pbest.size());
    }
    std::cout << "Solutions validées totales: " << total_validated << "\n";
    
    std::cout << std::string(80, '=') << "\n\n";
}

GlobalSolution GlobalSolutionConstructor::initialize_global_solution() {
    if (results.empty()) {
        std::cerr << "ERREUR: Aucun résultat de MetaAgent disponible!\n";
        std::cerr << "Appelez run_all_meta_agents() avant initialize_global_solution()\n";
        return GlobalSolution();
    }
    
    initial_solution.solution_to_meta_agent.clear();
    
    for (const auto& result : results) {
        initial_solution.solution_to_meta_agent[result] = result.gbest;
        std::cout << "  " << result.name << " → GBest (" 
                  << result.gbest.POIs.size() << " POIs, fitness=" 
                  << result.gbest_fitness << ")\n";
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
    
    if (results.empty()) {
        std::cout << "Aucun résultat disponible.\n";
        return;
    }
    
    for (size_t i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        
        std::cout << "Meta-Agent #" << (i + 1) << ": " << result.name << "\n";
        std::cout << std::string(60, '-') << "\n";
        std::cout << "  GBest Fitness: " << result.gbest_fitness << "\n";
        std::cout << "  GBest POIs: " << result.gbest.POIs.size() << "\n";
        std::cout << "  GBest Distance: " << result.gbest.cost << " m / " 
                  << global_memory.length_constraint << " m ";
        std::cout << "(" << std::fixed << std::setprecision(1)
                  << (result.gbest.cost / global_memory.length_constraint * 100.0) << "%)\n";
        std::cout << "\n";
    }
    
    int best_fitness = 0;
    size_t best_index = 0;
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].gbest_fitness > best_fitness) {
            best_fitness = results[i].gbest_fitness;
            best_index = i;
        }
    }
    
    std::cout << "MEILLEUR OVERALL: " << results[best_index].name 
              << " (fitness=" << best_fitness << ")\n";
    std::cout << std::string(80, '=') << "\n\n";
}

std::vector<Solution> GlobalSolutionConstructor::get_all_validated_pbest() const {
    std::vector<Solution> all_pbest;
    
    for (const auto& result : results) {
        all_pbest.insert(all_pbest.end(), 
                        result.validated_pbest.begin(), 
                        result.validated_pbest.end());
    }
    
    return all_pbest;
}

std::vector<Solution> GlobalSolutionConstructor::get_all_gbest() const {
    std::vector<Solution> all_gbest;
    
    for (const auto& result : results) {
        all_gbest.push_back(result.gbest);
    }
    
    return all_gbest;
}