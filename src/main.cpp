#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include "GeoBox/Box.hpp"
#include "Render/MapRenderer.hpp"
#include "GeoBox/GeoBoxManager.hpp"
#include "Common/Pathfinding.hpp"
#include "Common/Memory.hpp"
#include "Agent/Agent.hpp"
#include "Agent/MetaAgent.hpp"
#include "Agent/GlobalSolutionConstructor.hpp"
#include "utility.hpp"
#include <string>
#include "MHProcs/ACO.hpp"
#include "MHProcs/GRASP.hpp"
#include "MHProcs/VNS.hpp"
#include "MHProcs/PSO.hpp"
#include "OverpassAPI/OverpassAPI.hpp"
#include <random>

//====================================================================================================================================================
//=======================================================================GLOBAL=======================================================================
//====================================================================================================================================================

void test_global_solution_constructor(const std::string& cache_dir) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST GLOBAL SOLUTION CONSTRUCTOR\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // ========================================
    // CONFIGURATION 
    // ========================================
    const std::string input_cache_name = "asakusa_test_agent_raw";
    const std::string output_cache_name = "global_solution_result";
    const std::string output_map_name = "global_solution_map";
    
    const std::string input_cache_path = cache_dir + "\\" + input_cache_name + ".json";
    const std::string output_cache_path = cache_dir + "\\" + output_cache_name + ".json";
    
    // ========================================
    // 1. CHARGEMENT GEOBOX
    // ========================================
    GeoBox geo_box = GeoBoxManager::load_geobox(input_cache_path);
    
    if (!geo_box.is_valid) {
        std::cerr << "x ERREUR: Impossible de charger la GeoBox\n";
        std::cerr << "  Fichier: " << input_cache_path << "\n";
        return;
    }
    
    // ========================================
    // 2. INITIALISATION SYSTÈMES
    // ========================================
    Pathfinder pathfinder(geo_box);
    GlobalMemory global_memory(geo_box, pathfinder);
    global_memory.initialize_group_stats();
    
    float length_constraint = 1000.0f;
    float search_coefficient = 1.4f;
    
    global_memory.length_constraint = length_constraint;
    global_memory.search_coefficient = search_coefficient;
    
    // ========================================
    // 3. CRÉATION GLOBAL SOLUTION CONSTRUCTOR
    // ========================================
    GlobalSolutionConstructor constructor(geo_box, pathfinder, global_memory);
    
    // ========================================
    // 4. CONFIGURATION META-AGENTS
    // ========================================

    //Global Params
    MetaAgentConfig configBase("Agent_Config");
    configBase.params.max_divergence_from_gbest = 0.3;
    configBase.params.max_iterations_per_agent = 100;
    configBase.params.tabu_list_size = 20;
    
    MetaAgentConfig config1 = configBase;
    config1.name = "Agent_Temples";
    config1.characteristics[1] = 100.0;  // Temples
    config1.characteristics[2] = 0.0;   // Restaurants
    config1.characteristics[3] = 0.0;   // Shops
    
    constructor.add_meta_agent_config(config1);

    MetaAgentConfig config2 = configBase;
    config2.name = "Agent_Restaurants";
    config2.characteristics[1] = 0.0;  // Temples
    config2.characteristics[2] = 100.0;  // Restaurants
    config2.characteristics[3] = 0.0;   // Shops
    
    constructor.add_meta_agent_config(config2);
    
    MetaAgentConfig config3 = configBase;
    config3.name = "Agent_Shops";
    config3.characteristics[1] = 0.0;   // Temples
    config3.characteristics[2] = 0.0;  // Restaurants
    config3.characteristics[3] = 100.0;  // Shops
    
    constructor.add_meta_agent_config(config3);

    MetaAgentConfig config4 = configBase;
    config4.name = "Agent_All";
    config4.characteristics[1] = 100.0;   // Temples
    config4.characteristics[2] = 100.0;  // Restaurants
    config4.characteristics[3] = 100.0;  // Shops
    
    constructor.add_meta_agent_config(config4);
    
    // ========================================
    // 5. LANCEMENT DE TOUS LES META-AGENTS
    // ========================================
    auto start_time = std::chrono::high_resolution_clock::now();
    constructor.initialize_reward_densities();

    constructor.run_all_meta_agents();
    auto meta_agents_end_time = std::chrono::high_resolution_clock::now();
    auto meta_agents_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        start_time - meta_agents_end_time
    );

    // ========================================
    // 6. OPTIMISATION GLOBALE (TABU SEARCH)
    // ========================================
    GlobalSolution global_best = constructor.tabu_search(50, 10);
    std::cout << "On est ok ici final" << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );
    auto tabu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - meta_agents_end_time
    );
    
    // ========================================
    // 7. AFFICHAGE RÉSULTAT GLOBAL
    // ========================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RÉSULTAT FINAL GLOBAL\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Temps total: " << total_duration.count() << " ms\n";
    std::cout << "  - Meta-Agents: " << meta_agents_duration.count() << " ms\n";
    std::cout << "  - Tabu Search Global: " << tabu_duration.count() << " ms\n\n";
    
    // Afficher le résumé
    constructor.print_summary();
    
    // ========================================
    // 8. AFFICHAGE SOLUTION GLOBALE DÉTAILLÉE
    // ========================================
    std::cout << "\n--- Solution Globale Optimale ---\n";
    std::cout << "Reward global: " << global_best.reward << "\n";
    std::cout << "Cost total: " << global_best.cost << " m\n";
    std::cout << "Nombre de Meta-Agents: " << global_best.solution_to_meta_agent.size() << "\n\n";
    
    int solution_number = 1;
    for (const auto& [meta_result, solution] : global_best.solution_to_meta_agent) {
        std::cout << "Solution #" << solution_number << " (" << constructor.meta_agent_configurations[meta_result].name << "):\n";
        std::cout << "  POIs: " << solution.POIs.size() << "\n";
        std::cout << "  Distance: " << solution.cost << " m\n";
        std::cout << "  Fitness: " << meta_result->objective_function(solution) << "\n";
        
        // Calculer fitness de cette solution
        int solution_fitness = 0;
        for (const auto& poi_id : solution.POIs) {
            auto node_it = geo_box.data.nodes.find(poi_id);
            if (node_it != geo_box.data.nodes.end()) {
                for (const int group_id : node_it->second.groupes) {
                    if (group_id >= 0 && group_id < static_cast<int>(meta_result->gbest_solution.POIs.size())) {
                        // Utiliser les caractéristiques du config correspondant
                        const auto& results = constructor.MetaAgents;
                        for (const auto& result : results) {
                            if (constructor.meta_agent_configurations[result].name == constructor.meta_agent_configurations[meta_result].name) {
                                // Note: On ne peut pas accéder directement aux characteristics
                                // On affiche juste les POIs
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        std::cout << "  Parcours: ";
        for (size_t i = 0; i < std::min(size_t(5), solution.POIs.size()); i++) {
            std::cout << solution.POIs[i];
            if (i < std::min(size_t(5), solution.POIs.size()) - 1) std::cout << " → ";
        }
        if (solution.POIs.size() > 5) std::cout << " → ... (" << (solution.POIs.size() - 5) << " de plus)";
        std::cout << "\n\n";
        
        solution_number++;
    }
    
    // ========================================
    // 9. MARQUAGE DES CHEMINS DE TOUTES LES SOLUTIONS
    // ========================================

    //All_Meta_Agent_GBests
    GeoBox geo_box_All_meta_agent = GeoBoxManager::load_geobox(input_cache_path);
    if (!geo_box.is_valid) {
        std::cerr << "ERREUR: Impossible de charger la GeoBox\n";
        std::cerr << "  Fichier: " << input_cache_path << "\n";
        return;
    }
    Pathfinder pathfinder_All_meta_agent(geo_box_All_meta_agent);

    int path_group_meta_agent = 1;
    
    for (const auto& [meta_result, solution] : global_best.solution_to_meta_agent) {
        for (const auto& path : meta_result->gbest_solution.paths) {
            for (const auto& way_id : path.path_edges) {
                auto way_it = geo_box.data.ways.find(way_id);
                if (way_it != geo_box.data.ways.end()) {
                    pathfinder_All_meta_agent.update_way_group(way_id, path_group_meta_agent);
                }
            }
        }
        path_group_meta_agent++;
    }

    //Final_Solution
    int path_group = 1;
    
    for (const auto& [meta_result, solution] : global_best.solution_to_meta_agent) {
        for (const auto& path : solution.paths) {
            for (const auto& way_id : path.path_edges) {
                auto way_it = geo_box.data.ways.find(way_id);
                if (way_it != geo_box.data.ways.end()) {
                    pathfinder.update_way_group(way_id, path_group);
                }
            }
        }
        path_group++;
    }
    
    // ========================================
    // 10. SAUVEGARDE
    // ========================================
    std::cout << "--- Sauvegarde résultat ---\n";
    GeoBoxManager::save_geobox(geo_box, output_cache_path);
    std::cout << "GeoBox sauvegardée: " << output_cache_path << "\n\n";
    
    // ========================================
    // 11. RENDU CARTE
    // ========================================

    //All Meta Agent
    std::cout << "--- Génération carte All Meta Agent ---\n";
    bool render_success_meta_agent = GeoBoxManager::render_geobox(geo_box_All_meta_agent, "All_Meta_Agent_Solutions", 2000, 2000);
    
    if (render_success_meta_agent) {
        std::cout << "Carte générée: All_Meta_Agent_Solutions\n";
    } else {
        std::cout << "Erreur lors du rendu\n";
    }

    //Global Solution
    std::cout << "--- Génération carte Global Solution ---\n";
    bool render_success = GeoBoxManager::render_geobox(geo_box, output_map_name, 2000, 2000);
    
    if (render_success) {
        std::cout << "Carte générée: " << output_map_name << "\n";
    } else {
        std::cout << "Erreur lors du rendu\n";
    }
    
    std::cout << "\n--- Fichiers générés ---\n";
    std::cout << "  Cache: " << output_cache_name << ".json\n";
    std::cout << "  Carte: " << output_map_name << ".png\n";
    
    // ========================================
    // 12. STATISTIQUES FINALES
    // ========================================
    std::cout << "\n--- Statistiques finales ---\n";
    
    int total_pois = 0;
    float total_distance = 0.0f;
    for (const auto& [meta_result, solution] : global_best.solution_to_meta_agent) {
        total_pois += solution.POIs.size();
        total_distance += solution.cost;
    }
    
    std::cout << "POIs totaux visités: " << total_pois << "\n";
    std::cout << "Distance totale: " << total_distance << " m / " 
              << (length_constraint * global_best.solution_to_meta_agent.size()) << " m ("
              << (total_distance / (length_constraint * global_best.solution_to_meta_agent.size()) * 100.0) 
              << "%)\n";
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "FIN TEST GLOBAL SOLUTION CONSTRUCTOR\n";
    std::cout << std::string(80, '=') << "\n\n";
}

//====================================================================================================================================================
//========================================================================VNS=========================================================================
//====================================================================================================================================================

void test_vns_orienteering(const std::string& cache_dir) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST VNS - 4 AGENTS\n";
    std::cout << std::string(80, '=') << "\n\n";

    const std::string input_cache_name = "asakusa_test_agent_raw";
    const std::string input_cache_path = cache_dir + "\\" + input_cache_name + ".json";

    std::cout << "Chargement de la GeoBox..." << std::endl;
    GeoBox geo_box = GeoBoxManager::load_geobox(input_cache_path);

    if (!geo_box.is_valid) {
        std::cerr << "ERREUR: Impossible de charger la GeoBox\n";
        return;
    }

    std::cout << "GeoBox chargée avec succès\n";
    std::cout << "  Nodes: " << geo_box.data.nodes.size() << "\n";
    std::cout << "  Ways: " << geo_box.data.ways.size() << "\n\n";

    Pathfinder pathfinder(geo_box);
    GlobalMemory global_memory(geo_box, pathfinder);

    double distance_constraint = 1000.0;
    global_memory.length_constraint = distance_constraint;
    global_memory.search_coefficient = 1.5f;

    // ========================================
    // PARAMÈTRES VNS
    // ========================================
    VNSParams vns_params;
    vns_params.max_iterations = 100;
    vns_params.k_max = 5;
    vns_params.max_iterations_vnd = 50;
    vns_params.use_two_opt = true;
    vns_params.use_insertion = true;
    vns_params.use_swap = true;
    vns_params.use_add_remove = true;

    std::vector<int> chars_base(100, 0);

    struct AgentConfig {
        std::string name;
        std::vector<int> characteristics;
    };

    std::vector<AgentConfig> agents;

    AgentConfig agent1;
    agent1.name = "Agent_Temples";
    agent1.characteristics = chars_base;
    agent1.characteristics[1] = 100.0;
    agent1.characteristics[2] = 0.0;
    agent1.characteristics[3] = 0.0;
    agents.push_back(agent1);

    AgentConfig agent2;
    agent2.name = "Agent_Restaurants";
    agent2.characteristics = chars_base;
    agent2.characteristics[1] = 0.0;
    agent2.characteristics[2] = 100;
    agent2.characteristics[3] = 0.0;
    agents.push_back(agent2);

    AgentConfig agent3;
    agent3.name = "Agent_Shops";
    agent3.characteristics = chars_base;
    agent3.characteristics[1] = 0.0;
    agent3.characteristics[2] = 0.0;
    agent3.characteristics[3] = 100;
    agents.push_back(agent3);

    AgentConfig agent4;
    agent4.name = "Agent_All";
    agent4.characteristics = chars_base;
    agent4.characteristics[1] = 100;
    agent4.characteristics[2] = 100;
    agent4.characteristics[3] = 100;
    agents.push_back(agent4);

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RÉSOLUTION VNS - 4 AGENTS\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::vector<VNSResult> results;

    for (size_t i = 0; i < agents.size(); ++i) {
        const auto& agent = agents[i];

        std::cout << "\n" << std::string(80, '-') << "\n";
        std::cout << "AGENT #" << (i + 1) << " : " << agent.name << "\n";
        std::cout << std::string(80, '-') << "\n\n";

        VNSSolver vns_solver(geo_box, pathfinder, global_memory);

        VNSResult result = vns_solver.solve(
            agent.characteristics,
            distance_constraint,
            vns_params
        );

        results.push_back(result);

        std::cout << "\nv " << agent.name << " TERMINÉ\n";
        std::cout << "  Fitness: " << static_cast<int>(result.fitness) << "\n";
        std::cout << "  POIs: " << result.num_pois << "\n";
        std::cout << "  Distance: " << static_cast<int>(result.distance) << "m / " 
                  << static_cast<int>(distance_constraint) << "m ("
                  << std::fixed << std::setprecision(1)
                  << (result.distance / distance_constraint * 100.0) << "%)\n";
        std::cout << "  Temps: " << result.execution_time_ms << " ms\n";
        std::cout << "  Valide: " << (result.is_valid ? "v" : "x") << "\n\n";
    }

    // ========================================
    // RÉSUMÉ GLOBAL
    // ========================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RÉSUMÉ GLOBAL VNS\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << std::left << std::setw(20) << "Agent" 
              << std::right << std::setw(15) << "Fitness"
              << std::setw(10) << "POIs"
              << std::setw(15) << "Distance (m)"
              << std::setw(15) << "Temps (ms)"
              << std::setw(10) << "Valide"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    long long total_time = 0;
    int best_fitness = 0;
    std::string best_agent_name;

    for (size_t i = 0; i < agents.size(); ++i) {
        const auto& result = results[i];
        const auto& agent = agents[i];

        std::cout << std::left << std::setw(20) << agent.name
                  << std::right << std::setw(15) << static_cast<int>(result.fitness)
                  << std::setw(10) << result.num_pois
                  << std::setw(15) << static_cast<int>(result.distance)
                  << std::setw(15) << result.execution_time_ms
                  << std::setw(10) << (result.is_valid ? "v" : "x")
                  << "\n";

        total_time += result.execution_time_ms;

        if (static_cast<int>(result.fitness) > best_fitness) {
            best_fitness = static_cast<int>(result.fitness);
            best_agent_name = agent.name;
        }
    }

    std::cout << std::string(80, '-') << "\n";
    std::cout << "\nTemps total: " << total_time << " ms (" 
              << std::fixed << std::setprecision(2) 
              << (total_time / 1000.0) << " s)\n";
    std::cout << "Meilleur agent: " << best_agent_name 
              << " (fitness=" << best_fitness << ")\n";

    // ========================================
    // GÉNÉRATION DES CARTES (optionnel)
    // ========================================
    std::cout << "\n--- Génération des cartes ---\n";
    
    for (size_t i = 0; i < agents.size(); ++i) {
        const auto& agent = agents[i];
        const auto& result = results[i];

        if (!result.is_valid || result.solution.empty()) {
            continue;
        }

        GeoBox agent_geobox = GeoBoxManager::load_geobox(input_cache_path);
        Pathfinder agent_pathfinder(agent_geobox);

        int path_group = 1;
        for (size_t j = 0; j < result.solution.size() - 1; ++j) {
            osmium::object_id_type node1 = result.solution[j];
            osmium::object_id_type node2 = result.solution[j + 1];

            std::vector<Path> paths = global_memory.check_path(node1, node2);
            
            if (!paths.empty() && paths[0].cost != std::numeric_limits<float>::max()) {
                const Path& path = paths[0];
                
                for (const auto& way_id : path.path_edges) {
                    auto way_it = agent_geobox.data.ways.find(way_id);
                    if (way_it != agent_geobox.data.ways.end()) {
                        agent_pathfinder.update_way_group(way_id, path_group);
                    }
                }
            }
        }

        std::string output_map = "VNS_" + agent.name;
        bool render_success = GeoBoxManager::render_geobox(agent_geobox, output_map, 2000, 2000);

        if (render_success) {
            std::cout << "  v Carte générée: " << output_map << ".png\n";
        }
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST VNS TERMINÉ\n";
    std::cout << std::string(80, '=') << "\n\n";
}

//====================================================================================================================================================
//========================================================================PSO=========================================================================
//====================================================================================================================================================

void test_pso_mttds(const std::string& cache_dir) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST PSO MTTDS - 4 AGENTS\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    const std::string input_cache_name = "asakusa_test_agent_raw";
    const std::string input_cache_path = cache_dir + "/" + input_cache_name + ".json";
    
    std::cout << "Chargement de la GeoBox..." << std::endl;
    GeoBox geo_box = GeoBoxManager::load_geobox(input_cache_path);
    
    if (!geo_box.is_valid) {
        std::cerr << "ERREUR: Impossible de charger la GeoBox\n";
        std::cerr << "  Fichier: " << input_cache_path << "\n";
        return;
    }
    
    std::cout << "GeoBox chargée avec succès\n";
    std::cout << "  Nodes: " << geo_box.data.nodes.size() << "\n";
    std::cout << "  Ways: " << geo_box.data.ways.size() << "\n\n";
    
    Pathfinder pathfinder(geo_box);
    GlobalMemory global_memory(geo_box, pathfinder);
    
    double distance_constraint = 1000.0;
    global_memory.length_constraint = distance_constraint;
    global_memory.search_coefficient = 1.5f;
    
    MTTDSPSOParams pso_params;
    pso_params.num_particles = 30;
    pso_params.max_iterations = 100;
    pso_params.w = 0.9;
    pso_params.c1 = 2.0; 
    pso_params.c2 = 2.0; 
    pso_params.mutation_rate = 0.25;
    pso_params.use_local_search = true;
    
    std::vector<int> chars_base(100, 0);
    
    struct AgentConfig {
        std::string name;
        std::vector<int> characteristics;
    };
    
    std::vector<AgentConfig> agents;
    
    AgentConfig agent1;
    agent1.name = "Agent_Temples";
    agent1.characteristics = chars_base;
    agent1.characteristics[1] = 100;
    agents.push_back(agent1);
    
    AgentConfig agent2;
    agent2.name = "Agent_Restaurants";
    agent2.characteristics = chars_base;
    agent2.characteristics[2] = 100;
    agents.push_back(agent2);
    
    AgentConfig agent3;
    agent3.name = "Agent_Shops";
    agent3.characteristics = chars_base;
    agent3.characteristics[3] = 100;
    agents.push_back(agent3);
    
    AgentConfig agent4;
    agent4.name = "Agent_All";
    agent4.characteristics = chars_base;
    agent4.characteristics[1] = 100;
    agent4.characteristics[2] = 100;
    agent4.characteristics[3] = 100;
    agents.push_back(agent4);
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RÉSOLUTION PSO - 4 AGENTS\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::vector<MTTDSPSOResult> results;
    
    for (size_t i = 0; i < agents.size(); ++i) {
        const auto& agent = agents[i];
        
        std::cout << "\n" << std::string(80, '-') << "\n";
        std::cout << "AGENT #" << (i + 1) << " : " << agent.name << "\n";
        std::cout << std::string(80, '-') << "\n\n";
        
        MTTDS_PSOSolver pso_solver(geo_box, pathfinder, global_memory);
        
        MTTDSPSOResult result = pso_solver.solve(
            agent.characteristics,
            distance_constraint,
            pso_params
        );
        
        results.push_back(result);
        
        std::cout << "\n " << agent.name << " TERMINÉ\n";
        std::cout << "  Fitness GBest: " << static_cast<int>(result.fitness) << "\n";
        std::cout << "  POIs: " << result.num_pois << "\n";
        std::cout << "  Distance: " << static_cast<int>(result.distance) << "m / " 
                  << static_cast<int>(distance_constraint) << "m ("
                  << std::fixed << std::setprecision(1)
                  << (result.distance / distance_constraint * 100.0) << "%)\n";
        std::cout << "  Temps: " << result.execution_time_ms << " ms\n";
        std::cout << "  Valide: " << (result.is_valid ? "v" : "x") << "\n\n";
    }
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RÉSUMÉ GLOBAL PSO\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << std::left << std::setw(20) << "Agent" 
              << std::right << std::setw(15) << "Fitness GBest"
              << std::setw(10) << "POIs"
              << std::setw(15) << "Distance (m)"
              << std::setw(15) << "Temps (ms)"
              << std::setw(10) << "Valide"
              << "\n";
    std::cout << std::string(80, '-') << "\n";
    
    long long total_time = 0;
    int best_fitness = 0;
    std::string best_agent_name;
    
    for (size_t i = 0; i < agents.size(); ++i) {
        const auto& result = results[i];
        const auto& agent = agents[i];
        
        std::cout << std::left << std::setw(20) << agent.name
                  << std::right << std::setw(15) << static_cast<int>(result.fitness)
                  << std::setw(10) << result.num_pois
                  << std::setw(15) << static_cast<int>(result.distance)
                  << std::setw(15) << result.execution_time_ms
                  << std::setw(10) << (result.is_valid ? "v" : "x")
                  << "\n";
        
        total_time += result.execution_time_ms;
        
        if (static_cast<int>(result.fitness) > best_fitness) {
            best_fitness = static_cast<int>(result.fitness);
            best_agent_name = agent.name;
        }
    }
    
    std::cout << std::string(80, '-') << "\n";
    std::cout << "\nTemps total: " << total_time << " ms (" 
              << std::fixed << std::setprecision(2) 
              << (total_time / 1000.0) << " s)\n";
    std::cout << "Meilleur agent: " << best_agent_name 
              << " (fitness=" << best_fitness << ")\n";

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "GÉNÉRATION DES CARTES\n";
    std::cout << std::string(80, '=') << "\n\n";

    for (size_t i = 0; i < agents.size(); ++i) {
        const auto& agent = agents[i];
        const auto& result = results[i];

        if (!result.is_valid || result.solution.empty()) {
            std::cout << "⚠️  " << agent.name << ": Solution invalide, carte non générée\n";
            continue;
        }

        std::cout << "\n--- Génération carte pour " << agent.name << " ---\n";

        // Charger une GeoBox fraîche pour cet agent
        GeoBox agent_geobox = GeoBoxManager::load_geobox(input_cache_path);
        if (!agent_geobox.is_valid) {
            std::cerr << "x Erreur chargement GeoBox pour " << agent.name << "\n";
            continue;
        }

        Pathfinder agent_pathfinder(agent_geobox);

        // Marquer les chemins de la solution PSO
        int path_group = 1;
        for (size_t j = 0; j < result.solution.size() - 1; ++j) {
            osmium::object_id_type node1 = result.solution[j];
            osmium::object_id_type node2 = result.solution[j + 1];

            // Récupérer le chemin via check_path
            std::vector<Path> paths = global_memory.check_path(node1, node2);
            
            if (!paths.empty() && paths[0].cost != std::numeric_limits<float>::max()) {
                const Path& path = paths[0];
                
                // Marquer tous les ways du chemin avec le groupe de couleur
                for (const auto& way_id : path.path_edges) {
                    auto way_it = agent_geobox.data.ways.find(way_id);
                    if (way_it != agent_geobox.data.ways.end()) {
                        agent_pathfinder.update_way_group(way_id, path_group);
                    }
                }
            }
        }

        // Sauvegarder la GeoBox avec chemins marqués
        std::string output_cache = cache_dir + "\\pso_" + agent.name + "_result.json";
        GeoBoxManager::save_geobox(agent_geobox, output_cache);
        std::cout << "  Cache sauvegardé: " << output_cache << "\n";

        // Générer la carte
        std::string output_map = "PSO_" + agent.name;
        bool render_success = GeoBoxManager::render_geobox(agent_geobox, output_map, 2000, 2000);

        if (render_success) {
            std::cout << "  v Carte générée: " << output_map << ".png\n";
            std::cout << "    - POIs: " << result.num_pois << "\n";
            std::cout << "    - Fitness: " << static_cast<int>(result.fitness) << "\n";
            std::cout << "    - Distance: " << static_cast<int>(result.distance) << "m\n";
        } else {
            std::cout << "  x Erreur lors du rendu\n";
        }
    }
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST PSO TERMINÉ\n";
    std::cout << std::string(80, '=') << "\n\n";
}

//====================================================================================================================================================
//========================================================================MAIN========================================================================
//====================================================================================================================================================

int main() {
    
    const std::string osm_file = "C:\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    const std::string cache_dir = "C:\\ConflictualMAS\\src\\geobox_cache_folder";

    // ========== SELECTION DE LA LOCALISATION ==========
    std::cout << "\n=== Sélection de la localisation ===" << std::endl;
    std::cout << "1. Asakusa (Temples traditionnels)" << std::endl;
    std::cout << "2. Shibuya (Centre urbain)" << std::endl;
    std::cout << "3. Shinjuku (Quartier d'affaires)" << std::endl;
    std::cout << "Choisissez une localisation (1-3) : ";
    
    int location_choice;
    std::cin >> location_choice;
    
    // Variables de configuration selon la localisation
    std::string location_name;
    double min_lat, min_lon, max_lat, max_lon;
    std::string bbox;
    std::string cache_name;
    
    switch(location_choice) {
        case 1: // Asakusa
            location_name = "asakusa";
            min_lat = 35.705; min_lon = 139.785;
            max_lat = 35.718; max_lon = 139.800;
            bbox = "139.785,35.705,139.800,35.718";
            std::cout << "Localisation sélectionnée : Asakusa" << std::endl;
            break;
            
        case 2: // Shibuya
            location_name = "shibuya";
            min_lat = 35.658; min_lon = 139.699;
            max_lat = 35.661; max_lon = 139.704;
            bbox = "139.699,35.658,139.704,35.661";
            std::cout << "Localisation sélectionnée : Shibuya" << std::endl;
            break;
            
        case 3: // Shinjuku
            location_name = "shinjuku";
            min_lat = 35.689; min_lon = 139.698;
            max_lat = 35.702; max_lon = 139.710;
            bbox = "139.698,35.689,139.710,35.702";
            std::cout << "Localisation sélectionnée : Shinjuku" << std::endl;
            break;
            
        default:
            std::cout << "Choix invalide, utilisation d'Asakusa par défaut" << std::endl;
            location_name = "asakusa";
            min_lat = 35.705; min_lon = 139.785;
            max_lat = 35.718; max_lon = 139.800;
            bbox = "139.785,35.705,139.800,35.718";
            break;
    }
    
    const std::string cache_path = cache_dir + "\\" + location_name + ".json";
    std::cout << "=== Fin sélection localisation ===\n" << std::endl;

    // ========== MENU PRINCIPAL ==========
    std::string rep;
    std::cout << "\n=== MENU PRINCIPAL ===" << std::endl;
    std::cout << "G/g - Test Global" << std::endl;
    std::cout << "P/p - Test PSO" << std::endl;
    std::cout << "V/v - Test VNS" << std::endl;
    std::cout << "E/e - New Geobox and cache" << std::endl;
    std::cout << "I/i - Initialize POI" << std::endl;
    std::cout << "R/r - Render only" << std::endl;
    std::cout << "\nChoix: ";
    std::cin >> rep;

    FlickrConfig config;
    config.api_key = "9568c6342a890ef1ba342f54c4c1160f";
    config.bbox = bbox;  // Utilise la bbox de la localisation sélectionnée
    config.poi_assignment_radius = 15.0;
    config.min_date = "2020-01-01";
    config.max_date = "2024-12-31";
    
    if (rep == "G" || rep == "g") {
        test_global_solution_constructor(cache_dir);
        return 0;
    } else if (rep == "P" || rep == "p") {
        test_pso_mttds(cache_dir);
        return 0;
    } else if (rep == "V" || rep == "v") {
        test_vns_orienteering(cache_dir);
        return 0;
    } else if (rep == "E" || rep == "e"){

        complete_workflow(
            osm_file,
            min_lon, min_lat, max_lon, max_lat,  // Coordonnées de la localisation sélectionnée
            cache_dir,
            location_name,
            location_name + "_raw",
            2000, 2000,
            config,
            false  // Utiliser les objectifs Flickr
        );
        
    } else if (rep == "I" || rep == "i") {

        // ========== INITIALISATION DES POI FLICKR ==========
        std::cout << "\n=== Initialisation des Points d'Intérêt Flickr ===" << std::endl;
        
        std::string group_input;
        std::cout << "Numéro du groupe à créer : ";
        std::cin >> group_input;

        int group_nb = std::stoi(group_input);

        std::string objective_input;
        std::cout << "Objectifs Flickr (ex: temple, restaurant, shop, park) : ";
        std::cin >> objective_input;

        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur lors du rechargement du cache raw" << std::endl;
            return 0;
        }
        else {
            std::cout << "Succès du chargement du cache raw" << std::endl;
        }

        // Configuration Flickr avec les objectifs saisis
        config.search_word = objective_input;

        try {
            // Appliquer les objectifs Flickr à la GeoBox existante
            std::string flickr_cache = cache_dir + "\\flickr_" + location_name + "_" + objective_input + "_cache.json";
            geo_box = apply_objectives(geo_box, config, flickr_cache, true, group_nb);
            std::cout << "Objectifs Flickr appliqués avec succès pour: " << objective_input << std::endl;
            
            std::cout << "Cache Name to save : ";
            std::cin >> cache_name;
            cache_name = cache_dir + "//" + cache_name + ".json";
            GeoBoxManager::save_geobox(geo_box, cache_name);
            std::cout << "GeoBox avec POI sauvegardée: " << cache_name << std::endl;
            
            // Vérifier que le groupe a été créé
            auto group_it = geo_box.data.objective_groups.find(group_nb);
            if (group_it != geo_box.data.objective_groups.end()) {
                std::cout << "Groupe " << group_nb << " créé avec " << group_it->second.node_ids.size() << " POI" << std::endl;
            } else {
                std::cout << "Attention: Aucun POI trouvé pour les objectifs '" << objective_input << "'" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Erreur lors de l'application des objectifs Flickr: " << e.what() << std::endl;
        }
        
        std::cout << "=== Fin initialisation POI ===\n" << std::endl;
        
    } else if (rep == "R" || rep == "r") {

        // ========== RENDER ONLY ==========
        std::cout << "\n=== Rendu de carte uniquement ===" << std::endl;
        
        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur lors du chargement de la GeoBox" << std::endl;
            return 0;
        }
        else {
            std::cout << "Succès du chargement de la GeoBox" << std::endl;
        }
        
        std::string output_name;
        std::cout << "Nom de sortie pour la carte : ";
        std::cin >> output_name;
        
        // Rendu de la carte
        bool render_success = GeoBoxManager::render_geobox(geo_box, output_name, 2000, 2000);
        
        if (render_success) {
            std::cout << "Carte rendue avec succès: " << output_name << std::endl;
        } else {
            std::cout << "Erreur lors du rendu de la carte" << std::endl;
        }

    }

    std::cout << "Fin de l'application" << std::endl;

    return 0;
}