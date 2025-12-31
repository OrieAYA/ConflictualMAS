#ifndef METAAGENT_HPP
#define METAAGENT_HPP

#include "../GeoBox/Box.hpp"
#include "../Common/Pathfinding.hpp"
#include "../Common/Memory.hpp"
#include "Agent.hpp"
#include <vector>
#include <random>
#include <map>

struct MetaAgentParams {
    // Paramètres de convergence
    double admissible_similarity_degree = 0.3;  // Similarité max entre solutions (0-1)
    double coverage_rate = 0.3;                  // Taux de couverture des POIs (0-1)
    double max_divergence_from_gbest = 0.7;     // Accepter fitness >= GBest × ce paramètre
    
    // Paramètres de recherche
    int max_iterations_per_agent = 100;
    int tabu_list_size = 15;
    int max_agents = 50;
    
    MetaAgentParams() = default;
};

class MetaAgent {
private:
    GeoBox& geo_box;
    Pathfinder& pathfinder;
    GlobalMemory& global_memory;
    std::vector<int> characteristics;
    
    std::map<Solution, float> local_visited_solutions;
    std::vector<Agent*> agent_population;
    std::vector<Solution> validated_pbest;
    
    // Meilleure solution globale (copie)
    Solution gbest_solution;
    int gbest_fitness;
    
    // POIs utilisés comme départ
    std::unordered_set<osmium::object_id_type> used_starting_pois;
    std::vector<osmium::object_id_type> all_pois;
    
    MetaAgentParams params;
    std::mt19937 rng;

    // Méthodes privées
    void collect_all_pois();
    osmium::object_id_type select_random_unused_poi();
    float calculate_similarity(const Solution& sol1, const Solution& sol2);
    float calculate_coverage_rate();
    bool try_add_agent();
    void print_statistics();

public:
    // ✅ CORRECTION: Signature correcte du constructeur
    MetaAgent(
        GeoBox& box,
        Pathfinder& pf,
        GlobalMemory& mem,
        std::vector<int> agent_characteristics,
        MetaAgentParams parameters = MetaAgentParams()
    );
    
    ~MetaAgent();
    
    // Lancer la recherche et retourner la meilleure solution
    Solution run_meta_search();

    // Récupérer toutes les solutions validées
    const std::vector<Solution>& get_validated_pbest() const { 
        return validated_pbest; 
    }
    
    // Récupérer la population d'agents (pour stats)
    size_t get_agent_count() const { 
        return agent_population.size(); 
    }
    
    // Récupérer le taux de couverture atteint
    float get_coverage_rate() const {
        if (all_pois.empty()) return 0.0f;
        return static_cast<float>(used_starting_pois.size()) / static_cast<float>(all_pois.size());
    }
    
    // Récupérer la fitness de GBest
    int get_gbest_fitness() const {
        return gbest_fitness;
    }
    
    // Récupérer la solution GBest
    const Solution& get_gbest_solution() const {
        return gbest_solution;
    }
};

#endif // METAAGENT_HPP