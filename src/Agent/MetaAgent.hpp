#ifndef METAAGENT_HPP
#define METAAGENT_HPP

#include "../GeoBox/Box.hpp"
#include "../Common/Pathfinding.hpp"
#include "../Common/Memory.hpp"
#include "Agent.hpp"
#include <vector>
#include <random>
#include <map>

class Agent;

struct MetaAgentParams {
    double max_divergence_from_gbest = 0.7;
    
    int max_iterations_per_agent = 100;
    int tabu_list_size = 15;
    
    MetaAgentParams() = default;
    MetaAgentParams(
        double max_divergence,
        int max_iterations,
        int tabu_list_s
    ) : 
    max_divergence_from_gbest(max_divergence),
    max_iterations_per_agent(max_iterations),
    tabu_list_size(tabu_list_s)
    {}
};

class MetaAgent {
private:
    GeoBox& geo_box;
    Pathfinder& pathfinder;
    GlobalMemory& global_memory;
    
    std::vector<Agent*> agent_population;
    
    std::unordered_set<osmium::object_id_type> used_starting_pois;
    std::vector<osmium::object_id_type> all_pois;
    
    MetaAgentParams params;
    std::mt19937 rng;

    // Méthodes privées
    void collect_all_pois();
    osmium::object_id_type select_random_unused_poi();
    bool try_add_agent();
    void print_statistics();

public:

    double min_fitness = 0;
    Solution gbest_solution;
    std::map<int,double> characteristics;
    std::vector<Solution> validated_pbest;
    std::map<Solution, double> local_visited_solutions;
    
    MetaAgent(
        GeoBox& box,
        Pathfinder& pf,
        GlobalMemory& mem,
        std::map<int,double> agent_characteristics,
        MetaAgentParams parameters = MetaAgentParams()
    );
    
    ~MetaAgent();

    double objective_function(const Solution& sol) const;
    
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
    
    // Récupérer la solution GBest
    const Solution& get_gbest_solution() const {
        return gbest_solution;
    }
};

#endif // METAAGENT_HPP