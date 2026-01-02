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
    
    const std::string input_cache_path = cache_dir + "//" + input_cache_name + ".json";
    const std::string output_cache_path = cache_dir + "//" + output_cache_name + ".json";
    
    // ========================================
    // 1. CHARGEMENT GEOBOX
    // ========================================
    GeoBox geo_box = GeoBoxManager::load_geobox(input_cache_path);
    
    if (!geo_box.is_valid) {
        std::cerr << "✗ ERREUR: Impossible de charger la GeoBox\n";
        std::cerr << "  Fichier: " << input_cache_path << "\n";
        return;
    }
    
    // ========================================
    // 2. INITIALISATION SYSTÈMES
    // ========================================
    Pathfinder pathfinder(geo_box);
    GlobalMemory global_memory(geo_box, pathfinder);
    
    float length_constraint = 1000.0f;
    float search_coefficient = 1.2f;
    
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
    configBase.params.admissible_similarity_degree = 0.4;
    configBase.params.coverage_rate = 0.5;
    configBase.params.max_divergence_from_gbest = 0.2;
    configBase.params.max_iterations_per_agent = 100;
    configBase.params.tabu_list_size = 10;
    configBase.params.max_agents = 25;
    
    MetaAgentConfig config1 = configBase;
    config1.name = "Agent_Temples";
    config1.characteristics.resize(100, 0);
    config1.characteristics[1] = 100;  // Temples
    config1.characteristics[2] = 0;   // Restaurants
    config1.characteristics[3] = 0;   // Shops
    
    constructor.add_meta_agent_config(config1);

    MetaAgentConfig config2 = configBase;
    config2.name = "Agent_Restaurants";
    config2.characteristics.resize(100, 0);
    config2.characteristics[1] = 0;  // Temples
    config2.characteristics[2] = 100;  // Restaurants
    config2.characteristics[3] = 0;   // Shops
    
    constructor.add_meta_agent_config(config2);
    
    MetaAgentConfig config3 = configBase;
    config3.name = "Agent_Shops";
    config3.characteristics.resize(100, 0);
    config3.characteristics[1] = 0;   // Temples
    config3.characteristics[2] = 0;  // Restaurants
    config3.characteristics[3] = 100;  // Shops
    
    constructor.add_meta_agent_config(config3);

    MetaAgentConfig config4 = configBase;
    config4.name = "Agent_All";
    config4.characteristics.resize(100, 0);
    config4.characteristics[1] = 100;   // Temples
    config4.characteristics[2] = 100;  // Restaurants
    config4.characteristics[3] = 100;  // Shops
    
    constructor.add_meta_agent_config(config4);
    
    // ========================================
    // 5. LANCEMENT DE TOUS LES META-AGENTS
    // ========================================
    auto start_time = std::chrono::high_resolution_clock::now();
    constructor.run_all_meta_agents();  // Initialise automatiquement la solution globale
    auto meta_agents_end_time = std::chrono::high_resolution_clock::now();
    auto meta_agents_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        meta_agents_end_time - start_time
    );
    
    // ========================================
    // 6. OPTIMISATION GLOBALE (TABU SEARCH)
    // ========================================
    GlobalSolution global_best = constructor.tabu_search(50, 10);
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
        std::cout << "Solution #" << solution_number << " (" << meta_result.name << "):\n";
        std::cout << "  POIs: " << solution.POIs.size() << "\n";
        std::cout << "  Distance: " << solution.cost << " m\n";
        std::cout << "  Fitness: " << meta_result.act_meta_agent->objective_function(solution) << " m\n";
        
        // Calculer fitness de cette solution
        int solution_fitness = 0;
        for (const auto& poi_id : solution.POIs) {
            auto node_it = geo_box.data.nodes.find(poi_id);
            if (node_it != geo_box.data.nodes.end()) {
                for (const int group_id : node_it->second.groupes) {
                    if (group_id >= 0 && group_id < static_cast<int>(meta_result.gbest.POIs.size())) {
                        // Utiliser les caractéristiques du config correspondant
                        const auto& results = constructor.get_results();
                        for (const auto& result : results) {
                            if (result.name == meta_result.name) {
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
        std::cerr << "✗ ERREUR: Impossible de charger la GeoBox\n";
        std::cerr << "  Fichier: " << input_cache_path << "\n";
        return;
    }
    Pathfinder pathfinder_All_meta_agent(geo_box_All_meta_agent);

    int path_group_meta_agent = 1;
    
    for (const auto& [meta_result, solution] : global_best.solution_to_meta_agent) {
        for (const auto& path : meta_result.gbest.paths) {
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
    std::cout << "✓ GeoBox sauvegardée: " << output_cache_path << "\n\n";
    
    // ========================================
    // 11. RENDU CARTE
    // ========================================

    //All Meta Agent
    std::cout << "--- Génération carte All Meta Agent ---\n";
    bool render_success_meta_agent = GeoBoxManager::render_geobox(geo_box_All_meta_agent, "All_Meta_Agent_Solutions", 2000, 2000);
    
    if (render_success_meta_agent) {
        std::cout << "✓ Carte générée: All_Meta_Agent_Solutions\n";
    } else {
        std::cout << "✗ Erreur lors du rendu\n";
    }

    //Global Solution
    std::cout << "--- Génération carte Global Solution ---\n";
    bool render_success = GeoBoxManager::render_geobox(geo_box, output_map_name, 2000, 2000);
    
    if (render_success) {
        std::cout << "✓ Carte générée: " << output_map_name << "\n";
    } else {
        std::cout << "✗ Erreur lors du rendu\n";
    }
    
    std::cout << "\n--- Fichiers générés ---\n";
    std::cout << "  Cache: " << output_cache_name << ".json\n";
    std::cout << "  Carte: " << output_map_name << ".png\n";
    
    // ========================================
    // 12. STATISTIQUES FINALES
    // ========================================
    std::cout << "\n--- Statistiques finales ---\n";
    std::cout << "Meta-Agents exécutés: " << constructor.get_result_count() << "\n";
    
    std::vector<Solution> all_gbest = constructor.get_all_gbest();
    std::cout << "GBest collectés: " << all_gbest.size() << "\n";
    
    std::vector<Solution> all_pbest = constructor.get_all_validated_pbest();
    std::cout << "PBest validés (total): " << all_pbest.size() << "\n";
    
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

void test_meta_agent(const std::string& cache_dir) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST META-AGENT\n";
    std::cout << std::string(80, '=') << "\n\n";

    // ========================================
    // CONFIGURATION
    // ========================================
    const std::string input_cache_name = "asakusa_test_agent_raw";
    const std::string output_cache_name = "meta_agent_result";
    const std::string output_map_name = "meta_agent_map";
    const int path_group_id = 99;
    
    const std::string input_cache_path = cache_dir + "//" + input_cache_name + ".json";
    const std::string output_cache_path = cache_dir + "//" + output_cache_name + ".json";
    
    // ========================================
    // 1. CHARGEMENT GEOBOX
    // ========================================
    std::cout << "--- Chargement GeoBox ---\n";
    GeoBox geo_box = GeoBoxManager::load_geobox(input_cache_path);
    
    if (!geo_box.is_valid) {
        std::cerr << "✗ ERREUR: Impossible de charger la GeoBox\n";
        std::cerr << "  Fichier: " << input_cache_path << "\n";
        return;
    }
    
    std::cout << "✓ GeoBox chargée\n";
    std::cout << "  Nœuds: " << geo_box.data.nodes.size() << "\n";
    std::cout << "  Ways: " << geo_box.data.ways.size() << "\n";
    std::cout << "  Groupes: " << geo_box.data.objective_groups.size() << "\n\n";
    
    // ========================================
    // 2. INITIALISATION SYSTÈMES
    // ========================================
    std::cout << "--- Initialisation systèmes ---\n";
    Pathfinder pathfinder(geo_box);
    GlobalMemory global_memory(geo_box, pathfinder);
    
    float length_constraint = 1000.0f;
    float search_coefficient = 1.4f;
    
    global_memory.length_constraint = length_constraint;
    global_memory.search_coefficient = search_coefficient;
    
    std::cout << "✓ Pathfinder initialisé\n";
    std::cout << "✓ Mémoire globale initialisée\n";
    std::cout << "  Contrainte distance: " << length_constraint << " m\n";
    std::cout << "  Coefficient recherche: " << search_coefficient << "\n\n";
    
    // ========================================
    // 3. CARACTÉRISTIQUES AGENT
    // ========================================
    std::vector<int> agent_characteristics(100, 0);
    agent_characteristics[0] = 100;  // Temples
    agent_characteristics[1] = 100;  // Restaurants  
    agent_characteristics[2] = 50;   // Shops
    
    std::cout << "--- Caractéristiques agent ---\n";
    std::cout << "  Groupe 0 (Temples): 100 points\n";
    std::cout << "  Groupe 1 (Restaurants): 100 points\n";
    std::cout << "  Groupe 2 (Shops): 50 points\n\n";
    
    // ========================================
    // 4. CONFIGURATION META-AGENT
    // ========================================
    MetaAgentParams meta_params;
    meta_params.admissible_similarity_degree = 0.3;
    meta_params.coverage_rate = 0.2;
    meta_params.max_divergence_from_gbest = 0.7;
    meta_params.max_iterations_per_agent = 100;
    meta_params.tabu_list_size = 15;
    meta_params.max_agents = 30;
    
    std::cout << "--- Configuration MetaAgent ---\n";
    std::cout << "Similarité max admissible: " << (meta_params.admissible_similarity_degree * 100) << "%\n";
    std::cout << "Taux couverture cible: " << (meta_params.coverage_rate * 100) << "%\n";
    std::cout << "Divergence max avec GBest: " << (meta_params.max_divergence_from_gbest * 100) << "%\n";
    std::cout << "Itérations par agent: " << meta_params.max_iterations_per_agent << "\n";
    std::cout << "Taille liste Tabu: " << meta_params.tabu_list_size << "\n";
    std::cout << "Agents max: " << meta_params.max_agents << "\n\n";
    
    // ========================================
    // 5. CRÉATION ET LANCEMENT META-AGENT
    // ========================================
    std::cout << "--- Création MetaAgent ---\n";
    MetaAgent meta_agent(
        geo_box,
        pathfinder,
        global_memory,
        agent_characteristics,
        meta_params
    );
    
    auto start_time = std::chrono::high_resolution_clock::now();
    Solution best_solution = meta_agent.run_meta_search();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    // ========================================
    // 6. AFFICHAGE RÉSULTAT DÉTAILLÉ
    // ========================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RÉSULTAT FINAL META-AGENT\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Temps total d'exécution: " << total_duration.count() << " secondes\n";
    
    if (best_solution.POIs.empty()) {
        std::cout << "\n⚠ Aucune solution trouvée\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << "FIN TEST META-AGENT\n";
        std::cout << std::string(80, '=') << "\n\n";
        return;
    }
    
    // Calculer fitness totale
    int total_fitness = 0;
    std::map<int, int> group_contributions;
    
    for (const auto& poi_id : best_solution.POIs) {
        auto node_it = geo_box.data.nodes.find(poi_id);
        if (node_it != geo_box.data.nodes.end()) {
            for (const int group_id : node_it->second.groupes) {
                if (group_id >= 0 && group_id < static_cast<int>(agent_characteristics.size())) {
                    int value = agent_characteristics[group_id];
                    total_fitness += value;
                    group_contributions[group_id] += value;
                }
            }
        }
    }
    
    std::cout << "\n--- Statistiques globales ---\n";
    std::cout << "POIs visités: " << best_solution.POIs.size() << "\n";
    std::cout << "Distance totale: " << best_solution.cost << " m / " 
              << length_constraint << " m ("
              << (best_solution.cost / length_constraint * 100.0) << "%)\n";
    std::cout << "Fitness totale: " << total_fitness << "\n";
    
    std::cout << "\n--- Contributions par groupe ---\n";
    for (const auto& [group_id, contribution] : group_contributions) {
        std::cout << "  Groupe " << group_id << ": " << contribution << " points\n";
    }
    
    // Parcours détaillé
    std::cout << "\n--- Parcours optimal (GBest) ---\n";
    float cumulative = 0.0f;
    for (size_t i = 0; i < best_solution.POIs.size(); i++) {
        osmium::object_id_type poi_id = best_solution.POIs[i];
        
        std::cout << "  " << (i+1) << ". POI " << poi_id;
        
        auto node_it = geo_box.data.nodes.find(poi_id);
        if (node_it != geo_box.data.nodes.end() && !node_it->second.groupes.empty()) {
            std::cout << " [";
            bool first = true;
            int node_value = 0;
            for (const int group_id : node_it->second.groupes) {
                if (!first) std::cout << ", ";
                std::cout << "G" << group_id;
                if (group_id >= 0 && group_id < static_cast<int>(agent_characteristics.size())) {
                    int val = agent_characteristics[group_id];
                    if (val > 0) {
                        std::cout << "(+" << val << ")";
                        node_value += val;
                    }
                }
                first = false;
            }
            std::cout << "] → " << node_value << " pts";
        }
        
        if (i < best_solution.paths.size()) {
            float dist = best_solution.paths[i].cost;
            cumulative += dist;
            std::cout << "\n     └→ " << dist << "m (cumulé: " << cumulative << "m)";
        }
        std::cout << "\n";
    }
    
    // ========================================
    // 7. MARQUAGE DES CHEMINS (comme test_single_agent)
    // ========================================
    std::cout << "\n--- Marquage des chemins ---\n";
    int marked_paths = 0;
    
    for (const auto& path : best_solution.paths) {
        for (const auto& way_id : path.path_edges) {
            auto way_it = geo_box.data.ways.find(way_id);
            if (way_it != geo_box.data.ways.end()) {
                way_it->second.groupes.insert(path_group_id);
                marked_paths++;
            }
        }
    }
    
    std::cout << "✓ Chemins marqués: " << marked_paths << " ways dans groupe " << path_group_id << "\n\n";
    
    // ========================================
    // 8. SAUVEGARDE (comme test_single_agent)
    // ========================================
    std::cout << "--- Sauvegarde résultat ---\n";
    GeoBoxManager::save_geobox(geo_box, output_cache_path);
    std::cout << "✓ GeoBox sauvegardée: " << output_cache_path << "\n\n";
    
    // ========================================
    // 9. RENDU CARTE (comme test_single_agent)
    // ========================================
    std::cout << "--- Génération carte ---\n";
    bool render_success = GeoBoxManager::render_geobox(geo_box, output_map_name, 2000, 2000);
    
    if (render_success) {
        std::cout << "✓ Carte générée: " << output_map_name << "\n";
    } else {
        std::cout << "✗ Erreur lors du rendu\n";
    }
    
    std::cout << "\n--- Fichiers générés ---\n";
    std::cout << "  Cache: " << output_cache_name << ".json\n";
    std::cout << "  Carte: " << output_map_name << ".png\n";
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "FIN TEST META-AGENT\n";
    std::cout << std::string(80, '=') << "\n\n";
}

// ====================================================================
// FONCTION DE TEST POUR UN SEUL AGENT - RECHERCHE TABU
// ====================================================================
void test_single_agent(const std::string& cache_dir) {
    
    // ========================================
    // CONFIGURATION MANUELLE - À MODIFIER ICI
    // ========================================
    
    // 1. Configuration du cache
    const std::string input_cache_name = "asakusa_test_agent_raw";
    const std::string output_cache_name = "asakusa_test_agent_result";
    const std::string output_map_name = "agent_tabu_map";
    
    // 2. Configuration de la mémoire
    const float length_constraint = 1000.0f;  // 1 km max
    const float search_coefficient = 1.4f;     // Pour le neighborhood search
    
    // 3. Configuration de l'agent - Caractéristiques/préférences par groupe
    std::vector<int> agent_characteristics = {
        0,    // Groupe 0
        100,  // Groupe 1 - Temples (priorité haute)
        50,   // Groupe 2 - Restaurants (priorité moyenne)
        30,   // Groupe 3 - Shops (priorité basse)
        0,    // Groupe 4
        0,    // Groupe 5
        0,    // Groupe 6
        0,    // Groupe 7
        0,    // Groupe 8
        0     // Groupe 9
    };
    
    // 4. Solution initiale - UN SEUL POI
    const bool random_start = true;  // true = POI aléatoire, false = premier POI
    
    // 5. Configuration Tabu Search
    const int max_iterations = 100;    // Nombre max d'itérations
    const int tabu_list_size = 15;     // Taille de la liste Tabu
    
    // 6. Options de sortie
    const bool save_result = true;
    const bool render_map = true;
    const int path_group_id = 99;  // ID du groupe pour marquer les chemins trouvés
    
    // ========================================
    // EXÉCUTION
    // ========================================
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST AGENT - RECHERCHE TABU MÉTAHEURISTIQUE\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    const std::string input_cache_path = cache_dir + "//" + input_cache_name + ".json";
    const std::string output_cache_path = cache_dir + "//" + output_cache_name + ".json";
    
    // Affichage configuration
    std::cout << "--- Configuration ---\n";
    std::cout << "Cache entrée: " << input_cache_name << "\n";
    std::cout << "Cache sortie: " << output_cache_name << "\n";
    std::cout << "Contrainte distance: " << length_constraint << " m\n";
    std::cout << "Search coefficient: " << search_coefficient << "\n";
    std::cout << "Max itérations: " << max_iterations << "\n";
    std::cout << "Taille liste Tabu: " << tabu_list_size << "\n";
    std::cout << "Départ: " << (random_start ? "POI aléatoire" : "Premier POI") << "\n\n";
    
    std::cout << "Préférences de l'agent:\n";
    for (size_t i = 0; i < agent_characteristics.size(); i++) {
        if (agent_characteristics[i] > 0) {
            std::cout << "  - Groupe " << i << ": valeur " << agent_characteristics[i] << "\n";
        }
    }
    std::cout << "\n";
    
    // 1. Chargement GeoBox
    std::cout << "--- Chargement GeoBox ---\n";
    GeoBox geo_box = GeoBoxManager::load_geobox(input_cache_path);
    
    if (!geo_box.is_valid) {
        std::cerr << "✗ ERREUR: Impossible de charger la GeoBox\n";
        std::cerr << "  Fichier: " << input_cache_path << "\n";
        return;
    }
    
    std::cout << "✓ GeoBox chargée\n";
    std::cout << "  Nœuds: " << geo_box.data.nodes.size() << "\n";
    std::cout << "  Ways: " << geo_box.data.ways.size() << "\n";
    std::cout << "  Groupes: " << geo_box.data.objective_groups.size() << "\n\n";
    
    // 2. Sélection POI de départ
    std::cout << "--- Sélection POI de départ ---\n";
    std::vector<osmium::object_id_type> candidate_pois;
    
    for (const auto& [node_id, node_data] : geo_box.data.nodes) {
        if (!node_data.groupes.empty()) {
            bool has_relevant_group = false;
            for (const auto& group_id : node_data.groupes) {
                if (group_id >= 0 && group_id < static_cast<int>(agent_characteristics.size())) {
                    if (agent_characteristics[group_id] > 0) {
                        has_relevant_group = true;
                        break;
                    }
                }
            }
            if (has_relevant_group) {
                candidate_pois.push_back(node_id);
            }
        }
    }
    
    if (candidate_pois.empty()) {
        std::cerr << "✗ ERREUR: Aucun POI trouvé\n";
        return;
    }
    
    std::cout << "POI candidats: " << candidate_pois.size() << "\n";
    
    // Sélection d'UN SEUL POI
    std::vector<osmium::object_id_type> initial_pois;
    if (random_start) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, candidate_pois.size() - 1);  // ← FIX: size_t au lieu de int
        initial_pois.push_back(candidate_pois[dis(gen)]);
        std::cout << "Sélection: POI aléatoire\n";
    } else {
        initial_pois.push_back(candidate_pois[0]);
        std::cout << "Sélection: Premier POI\n";
    }
    
    // Affichage POI de départ
    std::cout << "\nPOI de départ: " << initial_pois[0];
    auto node_it = geo_box.data.nodes.find(initial_pois[0]);
    if (node_it != geo_box.data.nodes.end() && !node_it->second.groupes.empty()) {
        std::cout << " [Groupes: ";
        int start_value = 0;
        size_t j = 0;
        // ← FIX: Utiliser itérateur au lieu d'index car groupes est un unordered_set
        for (const int& group_id : node_it->second.groupes) {
            std::cout << "G" << group_id;
            if (group_id >= 0 && group_id < static_cast<int>(agent_characteristics.size())) {
                int value = agent_characteristics[group_id];
                if (value > 0) {
                    std::cout << "(+" << value << ")";
                    start_value += value;
                }
            }
            if (j < node_it->second.groupes.size() - 1) std::cout << ", ";
            j++;
        }
        std::cout << "] → Valeur initiale: " << start_value;
    }
    std::cout << "\n\n";
    
    // 3. Initialisation systèmes
    std::cout << "--- Initialisation systèmes ---\n";
    Pathfinder pathfinder(geo_box);
    GlobalMemory global_memory(geo_box, pathfinder);
    global_memory.length_constraint = length_constraint;
    global_memory.search_coefficient = search_coefficient;
    std::cout << "✓ Pathfinder initialisé\n";
    std::cout << "✓ Mémoire globale initialisée\n\n";
    
    // 4. Création agent
    std::cout << "--- Création agent ---\n";
    Agent agent(
        geo_box, 
        pathfinder, 
        global_memory, 
        agent_characteristics, 
        initial_pois,
        nullptr,  // Pas de meta memory
        nullptr,  // Pas de pbest validées
        1.0,      // Pas de contrainte similarité
        10        // ← Max 10 voisins par exploration
    );
    std::cout << "✓ Agent créé\n\n";
    
    // 5. Exécution Tabu Search
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Solution best_solution = agent.tabu_search(max_iterations, tabu_list_size);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "RÉCAPITULATIF FINAL\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Temps total: " << duration.count() << " secondes\n";
    std::cout << "POIs visités: " << best_solution.POIs.size() << "\n";
    std::cout << "Distance totale: " << best_solution.cost << " m\n";
    std::cout << "Fitness: " << agent.objective_function(best_solution) << "\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // 6. Marquer les chemins dans la GeoBox
    if (save_result || render_map) {
        std::cout << "--- Marquage des chemins ---\n";
        int marked_paths = 0;
        
        for (const auto& path : best_solution.paths) {
            // ← FIX: Path utilise path_edges au lieu de edges
            for (const auto& way_id : path.path_edges) {
                auto way_it = geo_box.data.ways.find(way_id);
                if (way_it != geo_box.data.ways.end()) {
                    way_it->second.groupes.insert(path_group_id);
                    marked_paths++;
                }
            }
        }
        
        std::cout << "✓ Chemins marqués: " << marked_paths << " ways dans groupe " << path_group_id << "\n\n";
    }
    
    // 7. Sauvegarde
    if (save_result) {
        std::cout << "--- Sauvegarde résultat ---\n";
        GeoBoxManager::save_geobox(geo_box, output_cache_path);
        std::cout << "✓ GeoBox sauvegardée: " << output_cache_path << "\n\n";
    }
    
    // 8. Rendu carte
    if (render_map) {
        std::cout << "--- Génération carte ---\n";
        bool render_success = GeoBoxManager::render_geobox(geo_box, output_map_name, 2000, 2000);
        
        if (render_success) {
            std::cout << "✓ Carte générée: " << output_map_name << "\n";
        } else {
            std::cout << "✗ Erreur lors du rendu\n";
        }
    }
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST TERMINÉ\n";
    std::cout << std::string(80, '=') << "\n\n";
}

int main() {
    
    const std::string osm_file = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    const std::string cache_dir = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\geobox_cache_folder";

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
    std::cout << "T/t - Test Agent unique (Tabu Search)" << std::endl;
    std::cout << "M/m - Test Meta-Agent" << std::endl;
    std::cout << "G/g - Test Global" << std::endl;
    std::cout << "E/e - New Geobox and cache" << std::endl;
    std::cout << "I/i - Initialize POI" << std::endl;
    std::cout << "P/p - System Creation and Pathfinding" << std::endl;
    std::cout << "A/a - Metaheuristic procedures" << std::endl;
    std::cout << "V/v - Verify data" << std::endl;
    std::cout << "F/f - Verify Pathfinding" << std::endl;
    std::cout << "R/r - Render only" << std::endl;
    std::cout << "C/c - Complete Graph" << std::endl;
    std::cout << "\nChoix: ";
    std::cin >> rep;

    FlickrConfig config;
    config.api_key = "9568c6342a890ef1ba342f54c4c1160f";
    config.bbox = bbox;  // Utilise la bbox de la localisation sélectionnée
    config.poi_assignment_radius = 15.0;
    config.min_date = "2020-01-01";
    config.max_date = "2024-12-31";
    
    if (rep == "T" || rep == "t") {
        test_single_agent(cache_dir);
        return 0;
    } else if (rep == "M" || rep == "m") {
        test_meta_agent(cache_dir);
        return 0;
    } else if (rep == "G" || rep == "g") {
        test_global_solution_constructor(cache_dir);
        return 0;
    }else if (rep == "E" || rep == "e"){

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
        
    } else if (rep == "P" || rep == "p") {
        
        std::cout << "\n=== Pathfinding pour tous les groupes ===" << std::endl;
        
        // Charger la GeoBox avec les objectifs déjà initialisés
        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur: Cache d'objectifs introuvable. Utilisez d'abord l'option 'I' pour initialiser les POI." << std::endl;
            std::cout << "Fichier recherché: " << cache_name << std::endl;
            return 0;
        } else {
            std::cout << "Succès du chargement du cache d'objectifs" << std::endl;
            std::cout << "Nombre total de groupes: " << geo_box.data.objective_groups.size() << std::endl;
        }

        Pathfinder PfSystem(geo_box);

        auto debut = std::chrono::high_resolution_clock::now();
        
        bool global_success = false;
        int groups_processed = 0;
        int groups_successful = 0;
        
        // Parcourir tous les groupes existants
        for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
            
            std::cout << "\n--- TRAITEMENT GROUPE " << group_id << " ---" << std::endl;
            std::cout << "Nom: " << group_info.name << std::endl;
            std::cout << "POI disponibles: " << group_info.node_ids.size() << std::endl;
            
            groups_processed++;
            
            // Vérifier si le groupe a assez de POI
            if (group_info.node_ids.size() < 2) {
                std::cout << "GROUPE IGNORÉ: Minimum 2 POI requis pour le pathfinding" << std::endl;
                continue; // Passer au groupe suivant
            }
            
            std::cout << "Lancement du calcul des routes..." << std::endl;
            
            // Exécuter le pathfinding pour ce groupe
            bool group_success = PfSystem.Subgraph_construction(PfSystem, group_info.node_ids, group_id);
            
            if (group_success) {
                std::cout << "GROUPE " << group_id << ": SUCCÈS" << std::endl;
                groups_successful++;
                global_success = true; // Au moins un groupe a réussi
            } else {
                std::cout << "GROUPE " << group_id << ": ÉCHEC" << std::endl;
            }
        }

        auto fin = std::chrono::high_resolution_clock::now();
        
        // Résultats finaux
        std::cout << "\n=== RÉSULTATS FINAUX ===" << std::endl;
        std::cout << "Groupes traités: " << groups_processed << std::endl;
        std::cout << "Groupes réussis: " << groups_successful << std::endl;
        std::cout << "Taux de succès: " << (groups_processed > 0 ? (100.0 * groups_successful / groups_processed) : 0) << "%" << std::endl;
        
        if (global_success) {
            std::cout << "Construction du sous-graphe réussie globalement!" << std::endl;
            std::cout << "Cache Name to save : ";
            std::cin >> cache_name;
            cache_name = cache_dir + "//" + cache_name + ".json";
            GeoBoxManager::save_geobox(geo_box, cache_name);
            std::cout << "GeoBox avec pathfinding sauvegardée: " << cache_name << std::endl;
        } else {
            std::cout << "Aucun pathfinding réussi" << std::endl;
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

        auto duree = std::chrono::duration_cast<std::chrono::milliseconds>(fin - debut);
        
        std::cout << "Temps d'exécution: " << duree.count() << " ms" << std::endl;

    } else if (rep == "C" || rep == "c"){

        // Charger la GeoBox avec les objectifs déjà initialisés
        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur: Cache d'objectifs introuvable. Utilisez d'abord l'option 'I' pour initialiser les POI." << std::endl;
            std::cout << "Fichier recherché: " << cache_name << std::endl;
            return 0;
        }
        else {
            std::cout << "Succès du chargement du cache d'objectifs" << std::endl;
        }
        
        Pathfinder PfSystem(geo_box);

        int group_nb = 1;
        bool success = false;

        while(geo_box.data.objective_groups.find(group_nb) != geo_box.data.objective_groups.end()){

            auto& node_list = geo_box.data.objective_groups[group_nb].node_ids;

            if (node_list.size() < 2){
                std::cout << "Longueur de liste empêchant la création de routes (minimum 2 POI requis)" << std::endl;
                std::cout << "POI disponibles: " << node_list.size() << std::endl;
                group_nb++;
                continue;
            }
            else {
                std::cout << "Lancement du calcul des routes pour " << node_list.size() << " POI" << std::endl;
            }

            success = success | PfSystem.Subgraph_construction(PfSystem, node_list, group_nb);

            group_nb++;
        }

        if (success) {
            std::cout << "Construction du sous-graphe réussie!" << std::endl;
            std::cout << "Cache Name to save : ";
            std::cin >> cache_name;
            cache_name = cache_dir + "//" + cache_name + ".json";
            GeoBoxManager::save_geobox(geo_box, cache_name);
            std::cout << "GeoBox avec pathfinding sauvegardée: " << cache_name << std::endl;
        } else {
            std::cout << "Échec de la construction du sous-graphe" << std::endl;
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
    
    } else if (rep == "V" || rep == "v") {
        
        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur lors du rechargement de la GeoBox" << std::endl;
            return 0;
        }
        
        // Appel de la fonction de validation depuis utility
        validate_data_integrity(geo_box);
        
    } else if (rep == "O" || rep == "o") {
    
        std::cout << "\n=== Récupération de données Overpass API ===" << std::endl;
        
        try {
            OverpassAPI api;
            api.setTimeout(300); // 5 minutes
            
            std::cout << "\nLocalisation sélectionnée: " << location_name << std::endl;
            std::cout << "BBox: " << bbox << std::endl;
            
            // Construire la requête
            std::string query = api.getBoundingBoxQuery(min_lat, min_lon, max_lat, max_lon);
            
            // Nom du fichier de sortie
            std::string output_file = cache_dir + "\\overpass_" + location_name + "_raw.json";
            
            std::cout << "\nRécupération des données..." << std::endl;
            std::cout << "Cela peut prendre quelques minutes..." << std::endl;
            
            // Récupérer et sauvegarder
            bool success = api.queryToFile(query, output_file);
            
            if (success) {
                std::cout << "\n✓ Données récupérées avec succès!" << std::endl;
                std::cout << "Fichier: " << output_file << std::endl;
            } else {
                std::cout << "\n✗ Échec de la récupération des données" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Erreur: " << e.what() << std::endl;
        }
        
        std::cout << "=== Fin Overpass API ===\n" << std::endl;

    }else if (rep == "F" || rep == "f") {

        std::string group_input;
        std::cout << "Which group to check (nb) : ";
        std::cin >> group_input;

        int group_nb = std::stoi(group_input);

        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);

        if (!geo_box.is_valid) {
            std::cout << "Erreur lors du rechargement de la GeoBox" << std::endl;
            return 0;
        }

        Pathfinder PfSystem(geo_box);

        auto group_it = geo_box.data.objective_groups.find(group_nb);
        if (group_it == geo_box.data.objective_groups.end()) {
            std::cout << "Groupe " << group_nb << " n'existe pas!" << std::endl;
            return 0;
        }
        else {
            std::cout << "Groupe trouvé: " << group_it->second.name << std::endl;
        }

        auto& node_list = geo_box.data.objective_groups[group_nb].node_ids;

        if (node_list.size() < 2){
            std::cout << "Longeur de liste empechant la creation de routes" << std::endl;
            return 0;
        }
        else {
            std::cout << "Lancement du calcul des routes" << std::endl;
        }

        std::cout << "Nombre de points objectifs : " << node_list.size() << std::endl;

        bool success = verif_pathfinding(PfSystem, node_list, group_nb);

        if (success) {
            std::cout << "Le sous graph est complet" << std::endl;
        } else {
            std::cout << "Le sous graph a un probleme" << std::endl;
        }

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

    } else if (rep == "A" || rep == "a") {
        
        // ========== SELECTION DE LA METAHEURISTIQUE ==========
        std::cout << "\n=== Sélection de la méthode métaheuristique ===" << std::endl;
        std::cout << "1. ACO (Ant Colony Optimization)" << std::endl;
        std::cout << "2. GRASP (Greedy Randomized Adaptive Search)" << std::endl;
        std::cout << "3. VNS (Variable Neighborhood Search)" << std::endl;
        std::cout << "4. PSO (Particle Swarm Optimization)" << std::endl;
        std::cout << "Choisissez une méthode (1-4) : ";
        
        int metaheuristic_choice;
        std::cin >> metaheuristic_choice;

        // Charger la GeoBox avec les objectifs déjà initialisés
        std::cout << "\nCache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur: Cache d'objectifs introuvable. Utilisez d'abord l'option 'I' pour initialiser les POI." << std::endl;
            std::cout << "Fichier recherché: " << cache_name << std::endl;
            return 0;
        } else {
            std::cout << "Succès du chargement du cache d'objectifs" << std::endl;
        }

        auto debut = std::chrono::high_resolution_clock::now();
        bool overall_success = true;
        int processed_groups = 0;

        switch(metaheuristic_choice) {
            case 1: { // ACO
                std::cout << "\n=== PATHFINDING ACO POUR TOUS LES GROUPES ===" << std::endl;
                
                // Configuration des paramètres ACO
                ACOParams aco_params;
                
                std::cout << "\n=== Configuration ACO ===" << std::endl;
                std::cout << "Utiliser les paramètres par défaut? (y/n): ";
                char use_default;
                std::cin >> use_default;
                
                if (use_default != 'y' && use_default != 'Y') {
                    std::cout << "Nombre de fourmis [" << aco_params.num_ants << "]: ";
                    std::string input;
                    std::cin >> input;
                    if (!input.empty()) aco_params.num_ants = std::stoi(input);
                    
                    std::cout << "Nombre d'itérations [" << aco_params.max_iterations << "]: ";
                    std::cin >> input;
                    if (!input.empty()) aco_params.max_iterations = std::stoi(input);
                    
                    std::cout << "Alpha (importance phéromones) [" << aco_params.alpha << "]: ";
                    std::cin >> input;
                    if (!input.empty()) aco_params.alpha = std::stod(input);
                    
                    std::cout << "Beta (importance distance) [" << aco_params.beta << "]: ";
                    std::cin >> input;
                    if (!input.empty()) aco_params.beta = std::stod(input);
                    
                    std::cout << "Rho (évaporation) [" << aco_params.rho << "]: ";
                    std::cin >> input;
                    if (!input.empty()) aco_params.rho = std::stod(input);
                }

                std::cout << "\nParamètres ACO utilisés:" << std::endl;
                std::cout << "  Fourmis: " << aco_params.num_ants << std::endl;
                std::cout << "  Itérations: " << aco_params.max_iterations << std::endl;
                std::cout << "  Alpha: " << aco_params.alpha << std::endl;
                std::cout << "  Beta: " << aco_params.beta << std::endl;
                std::cout << "  Rho: " << aco_params.rho << std::endl;

                // Exécution ACO
                ACOSolver aco_solver(geo_box);
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() < 2) {
                        std::cout << "Groupe " << group_id << " ignoré (moins de 2 POI)" << std::endl;
                        continue;
                    }

                    std::cout << "\n--- TRAITEMENT ACO GROUPE " << group_id << " ---" << std::endl;
                    std::cout << "Nom: " << group_info.name << std::endl;
                    std::cout << "POI: " << group_info.node_ids.size() << std::endl;

                    bool group_success = aco_solver.solve_single_group(
                        group_info.node_ids, 
                        group_id, 
                        aco_params
                    );

                    if (group_success) {
                        std::cout << "Groupe " << group_id << ": SUCCÈS" << std::endl;
                        processed_groups++;
                    } else {
                        std::cout << "Groupe " << group_id << ": ÉCHEC" << std::endl;
                        overall_success = false;
                    }
                }
                break;
            }
            
            case 2: { // GRASP
                std::cout << "\n=== PATHFINDING GRASP POUR TOUS LES GROUPES ===" << std::endl;
                
                // Configuration des paramètres GRASP
                GRASPParams grasp_params;
                
                std::cout << "\n=== Configuration GRASP ===" << std::endl;
                std::cout << "Utiliser les paramètres par défaut? (y/n): ";
                char use_default;
                std::cin >> use_default;
                
                if (use_default != 'y' && use_default != 'Y') {
                    std::cout << "Nombre d'itérations [" << grasp_params.max_iterations << "]: ";
                    std::string input;
                    std::cin >> input;
                    if (!input.empty()) grasp_params.max_iterations = std::stoi(input);
                    
                    std::cout << "Alpha (randomisation 0.0-1.0) [" << grasp_params.alpha << "]: ";
                    std::cin >> input;
                    if (!input.empty()) grasp_params.alpha = std::stod(input);
                    
                    std::cout << "Itérations recherche locale [" << grasp_params.local_search_iterations << "]: ";
                    std::cin >> input;
                    if (!input.empty()) grasp_params.local_search_iterations = std::stoi(input);
                    
                    std::cout << "Utiliser 2-opt? (y/n) [" << (grasp_params.use_2opt ? "y" : "n") << "]: ";
                    std::cin >> input;
                    if (!input.empty()) grasp_params.use_2opt = (input[0] == 'y' || input[0] == 'Y');
                }

                std::cout << "\nParamètres GRASP utilisés:" << std::endl;
                std::cout << "  Itérations: " << grasp_params.max_iterations << std::endl;
                std::cout << "  Alpha: " << grasp_params.alpha << std::endl;
                std::cout << "  Recherche locale: " << grasp_params.local_search_iterations << std::endl;
                std::cout << "  2-opt: " << (grasp_params.use_2opt ? "Oui" : "Non") << std::endl;

                // Exécution GRASP
                GRASPSolver grasp_solver(geo_box);
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() < 2) {
                        std::cout << "Groupe " << group_id << " ignoré (moins de 2 POI)" << std::endl;
                        continue;
                    }

                    std::cout << "\n--- TRAITEMENT GRASP GROUPE " << group_id << " ---" << std::endl;
                    std::cout << "Nom: " << group_info.name << std::endl;
                    std::cout << "POI: " << group_info.node_ids.size() << std::endl;

                    bool group_success = grasp_solver.solve_single_group(
                        group_info.node_ids, 
                        group_id, 
                        grasp_params
                    );

                    if (group_success) {
                        std::cout << "Groupe " << group_id << ": SUCCÈS" << std::endl;
                        processed_groups++;
                    } else {
                        std::cout << "Groupe " << group_id << ": ÉCHEC" << std::endl;
                        overall_success = false;
                    }
                }
                break;
            }
            
            case 3: { // VNS
                std::cout << "\n=== PATHFINDING VNS POUR TOUS LES GROUPES ===" << std::endl;
                
                // Configuration des paramètres VNS
                VNSParams vns_params;
                
                std::cout << "\n=== Configuration VNS ===" << std::endl;
                std::cout << "Utiliser les paramètres par défaut? (y/n): ";
                char use_default;
                std::cin >> use_default;
                
                if (use_default != 'y' && use_default != 'Y') {
                    std::cout << "Nombre d'itérations [" << vns_params.max_iterations << "]: ";
                    std::string input;
                    std::cin >> input;
                    if (!input.empty()) vns_params.max_iterations = std::stoi(input);
                    
                    std::cout << "Nombre de voisinages [" << vns_params.max_neighborhoods << "]: ";
                    std::cin >> input;
                    if (!input.empty()) vns_params.max_neighborhoods = std::stoi(input);
                    
                    std::cout << "Intensité perturbation (0.1-0.5) [" << vns_params.shaking_intensity << "]: ";
                    std::cin >> input;
                    if (!input.empty()) vns_params.shaking_intensity = std::stod(input);
                    
                    std::cout << "First improvement? (y/n) [" << (vns_params.use_first_improvement ? "y" : "n") << "]: ";
                    std::cin >> input;
                    if (!input.empty()) vns_params.use_first_improvement = (input[0] == 'y' || input[0] == 'Y');
                }

                std::cout << "\nParamètres VNS utilisés:" << std::endl;
                std::cout << "  Itérations: " << vns_params.max_iterations << std::endl;
                std::cout << "  Voisinages: " << vns_params.max_neighborhoods << std::endl;
                std::cout << "  Intensité: " << vns_params.shaking_intensity << std::endl;
                std::cout << "  First improvement: " << (vns_params.use_first_improvement ? "Oui" : "Non") << std::endl;

                // Exécution VNS
                VNSSolver vns_solver(geo_box);
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() < 2) {
                        std::cout << "Groupe " << group_id << " ignoré (moins de 2 POI)" << std::endl;
                        continue;
                    }

                    std::cout << "\n--- TRAITEMENT VNS GROUPE " << group_id << " ---" << std::endl;
                    std::cout << "Nom: " << group_info.name << std::endl;
                    std::cout << "POI: " << group_info.node_ids.size() << std::endl;

                    bool group_success = vns_solver.solve_single_group(
                        group_info.node_ids, 
                        group_id, 
                        vns_params
                    );

                    if (group_success) {
                        std::cout << "Groupe " << group_id << ": SUCCÈS" << std::endl;
                        processed_groups++;
                    } else {
                        std::cout << "Groupe " << group_id << ": ÉCHEC" << std::endl;
                        overall_success = false;
                    }
                }
                break;
            }
            
            case 4: { // PSO
                std::cout << "\n=== PATHFINDING PSO POUR TOUS LES GROUPES ===" << std::endl;
                
                // Configuration des paramètres PSO
                PSOParams pso_params;
                
                std::cout << "\n=== Configuration PSO ===" << std::endl;
                std::cout << "Utiliser les paramètres par défaut? (y/n): ";
                char use_default;
                std::cin >> use_default;
                
                if (use_default != 'y' && use_default != 'Y') {
                    std::cout << "Nombre de particules [" << pso_params.num_particles << "]: ";
                    std::string input;
                    std::cin >> input;
                    if (!input.empty()) pso_params.num_particles = std::stoi(input);
                    
                    std::cout << "Nombre d'itérations [" << pso_params.max_iterations << "]: ";
                    std::cin >> input;
                    if (!input.empty()) pso_params.max_iterations = std::stoi(input);
                    
                    std::cout << "Inertie (w) [" << pso_params.w << "]: ";
                    std::cin >> input;
                    if (!input.empty()) pso_params.w = std::stod(input);
                    
                    std::cout << "Coefficient cognitif (c1) [" << pso_params.c1 << "]: ";
                    std::cin >> input;
                    if (!input.empty()) pso_params.c1 = std::stod(input);
                    
                    std::cout << "Coefficient social (c2) [" << pso_params.c2 << "]: ";
                    std::cin >> input;
                    if (!input.empty()) pso_params.c2 = std::stod(input);
                }

                std::cout << "\nParamètres PSO utilisés:" << std::endl;
                std::cout << "  Particules: " << pso_params.num_particles << std::endl;
                std::cout << "  Itérations: " << pso_params.max_iterations << std::endl;
                std::cout << "  Inertie (w): " << pso_params.w << std::endl;
                std::cout << "  Cognitif (c1): " << pso_params.c1 << std::endl;
                std::cout << "  Social (c2): " << pso_params.c2 << std::endl;

                // Exécution PSO
                PSOSolver pso_solver(geo_box);
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() < 2) {
                        std::cout << "Groupe " << group_id << " ignoré (moins de 2 POI)" << std::endl;
                        continue;
                    }

                    std::cout << "\n--- TRAITEMENT PSO GROUPE " << group_id << " ---" << std::endl;
                    std::cout << "Nom: " << group_info.name << std::endl;
                    std::cout << "POI: " << group_info.node_ids.size() << std::endl;

                    bool group_success = pso_solver.solve_single_group(
                        group_info.node_ids, 
                        group_id, 
                        pso_params
                    );

                    if (group_success) {
                        std::cout << "Groupe " << group_id << ": SUCCÈS" << std::endl;
                        processed_groups++;
                    } else {
                        std::cout << "Groupe " << group_id << ": ÉCHEC" << std::endl;
                        overall_success = false;
                    }
                }
                break;
            }
            
            default:
                std::cout << "Choix invalide, utilisation d'ACO par défaut" << std::endl;
                ACOSolver default_aco_solver(geo_box);
                ACOParams default_params;
                
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() >= 2) {
                        if (default_aco_solver.solve_single_group(group_info.node_ids, group_id, default_params)) {
                            processed_groups++;
                        } else {
                            overall_success = false;
                        }
                    }
                }
                break;
        }

        auto fin = std::chrono::high_resolution_clock::now();

        // Résultats finaux
        std::cout << "\n=== RÉSULTATS FINAUX ===" << std::endl;
        std::cout << "Groupes traités avec succès: " << processed_groups << std::endl;
        std::cout << "Statut global: " << (overall_success ? "SUCCÈS" : "PARTIEL") << std::endl;

        if (processed_groups > 0) {
            std::cout << "\nCache Name to save : ";
            std::cin >> cache_name;
            cache_name = cache_dir + "//" + cache_name + ".json";
            GeoBoxManager::save_geobox(geo_box, cache_name);
            std::cout << "GeoBox avec pathfinding métaheuristique sauvegardée: " << cache_name << std::endl;
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

        auto duree = std::chrono::duration_cast<std::chrono::milliseconds>(fin - debut);
        std::cout << "Temps d'exécution: " << duree.count() << " ms" << std::endl;
    } // <-- CORRECTION: Accolade manquante ajoutée ici

    std::cout << "Fin de l'application" << std::endl;

    return 0;
}

/* Paramètres Shibuya
create_save_render(
    osm_file,
    139.699, 35.658, 139.704, 35.661,  // Shibuya coordinates
    cache_dir + "\\shibuya_example.json",
    "shibuya_from_scratch",
    2000, 2000
);
*/

/* Paramètres Shibuya
GeoBox geo_box = GeoBoxManager::load_geobox(cache_dir + "\\shibuya_example.json");
*/

/*
with_flickr_objectives(
    osm_file,
    139.785, 35.705, 139.800, 35.718,  // Asakusa coordinates
    cache_dir + "\\asakusa_temples.json",
    config,
    "asakusa_temples_map",
    2000, 2000
);
*/

/*
complete_workflow(
    osm_file,
    139.785, 35.705, 139.800, 35.718,  // Asakusa coordinates
    cache_dir,
    "asakusa",
    "asakusa",
    2000, 2000,
    config,
    true  // Utiliser les objectifs Flickr
);
*/