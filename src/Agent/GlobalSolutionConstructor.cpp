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
    global_memory(mem)
{
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "GLOBAL SOLUTION CONSTRUCTOR INITIALISÉ\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "GeoBox: " << geo_box.data.nodes.size() << " nœuds\n";
    std::cout << "GlobalMemory: " << global_memory.length_constraint << "m max\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void GlobalSolutionConstructor::add_meta_agent_config(const MetaAgentConfig& config) {
    meta_agent_configs.push_back(config);
    
    std::cout << "✓ Configuration ajoutée: " << config.name << "\n";
    std::cout << "  Paramètres:\n";
    std::cout << "    - Similarité max: " << (config.params.admissible_similarity_degree * 100) << "%\n";
    std::cout << "    - Couverture cible: " << (config.params.coverage_rate * 100) << "%\n";
    std::cout << "    - Divergence: " << (config.params.max_divergence_from_gbest * 100) << "%\n";
    std::cout << "    - Itérations/agent: " << config.params.max_iterations_per_agent << "\n";
    std::cout << "    - Agents max: " << config.params.max_agents << "\n";
    
    // Afficher les caractéristiques non-nulles
    std::cout << "  Caractéristiques:\n";
    for (size_t i = 0; i < config.characteristics.size(); i++) {
        if (config.characteristics[i] > 0) {
            std::cout << "    - Groupe " << i << ": " << config.characteristics[i] << " points\n";
        }
    }
    std::cout << "\n";
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
        MetaAgent meta_agent(
            geo_box,
            pathfinder,
            global_memory,
            config.characteristics,
            config.params
        );
        
        Solution gbest = meta_agent.run_meta_search();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        
        // ✅ Calculer la fitness de GBest
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
        
        // ✅ Stocker les résultats avec validated_pbest
        MetaAgentResult result;
        result.name = config.name;
        result.gbest = gbest;
        result.validated_pbest = meta_agent.get_validated_pbest();  // ✅ Récupération !
        result.gbest_fitness = gbest_fitness;
        result.agent_count = static_cast<int>(meta_agent.get_agent_count());
        result.coverage_rate = meta_agent.get_coverage_rate();
        
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
    
    // ✅ Statistiques globales sur validated_pbest
    int total_validated = 0;
    for (const auto& result : results) {
        total_validated += static_cast<int>(result.validated_pbest.size());
    }
    std::cout << "Solutions validées totales: " << total_validated << "\n";
    
    std::cout << std::string(80, '=') << "\n\n";
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
    
    // Trouver le meilleur overall
    int best_fitness = 0;
    size_t best_index = 0;
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].gbest_fitness > best_fitness) {
            best_fitness = results[i].gbest_fitness;
            best_index = i;
        }
    }
    
    std::cout << "★ MEILLEUR OVERALL: " << results[best_index].name 
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