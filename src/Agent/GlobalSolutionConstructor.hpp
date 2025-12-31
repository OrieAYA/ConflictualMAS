#ifndef GLOBAL_SOLUTION_CONSTRUCTOR_HPP
#define GLOBAL_SOLUTION_CONSTRUCTOR_HPP

#include "../GeoBox/Box.hpp"
#include "../Common/Pathfinding.hpp"
#include "../Common/Memory.hpp"
#include "MetaAgent.hpp"
#include <vector>
#include <string>

// Structure pour configurer un MetaAgent
struct MetaAgentConfig {
    std::string name;                      // Nom descriptif (ex: "Tourist", "Foodie")
    std::vector<int> characteristics;      // Préférences par groupe
    MetaAgentParams params;                // Paramètres de recherche
    
    MetaAgentConfig(const std::string& agent_name = "Agent")
        : name(agent_name), characteristics(100, 0) {}
};

// Structure pour stocker les résultats d'un MetaAgent
struct MetaAgentResult {
    std::string name;                      // Nom du MetaAgent
    Solution gbest;                        // Meilleure solution globale
    std::vector<Solution> validated_pbest; // Toutes les solutions validées
    int gbest_fitness;                     // Fitness de la meilleure solution
    int agent_count;                       // Nombre d'agents dans la population
    float coverage_rate;                   // Taux de couverture atteint
};

class GlobalSolutionConstructor {
private:
    GeoBox& geo_box;
    Pathfinder& pathfinder;
    GlobalMemory& global_memory;
    
    // Configurations des MetaAgents à lancer
    std::vector<MetaAgentConfig> meta_agent_configs;
    
    // Résultats stockés après exécution
    std::vector<MetaAgentResult> results;

public:
    GlobalSolutionConstructor(
        GeoBox& box,
        Pathfinder& pf,
        GlobalMemory& mem
    );
    
    // Ajouter une configuration de MetaAgent
    void add_meta_agent_config(const MetaAgentConfig& config);
    
    // Lancer tous les MetaAgents configurés
    void run_all_meta_agents();
    
    // Accès aux résultats
    const std::vector<MetaAgentResult>& get_results() const { return results; }
    const MetaAgentResult& get_result(size_t index) const { return results[index]; }
    size_t get_result_count() const { return results.size(); }
    
    // Afficher un résumé de tous les résultats
    void print_summary() const;
    
    // Récupérer toutes les validated_pbest de tous les MetaAgents
    std::vector<Solution> get_all_validated_pbest() const;
    
    // Récupérer toutes les GBest de tous les MetaAgents
    std::vector<Solution> get_all_gbest() const;
};

#endif // GLOBAL_SOLUTION_CONSTRUCTOR_HPP