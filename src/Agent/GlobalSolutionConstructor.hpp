#ifndef GLOBAL_SOLUTION_CONSTRUCTOR_HPP
#define GLOBAL_SOLUTION_CONSTRUCTOR_HPP

#include "../GeoBox/Box.hpp"
#include "../Common/Pathfinding.hpp"
#include "../Common/Memory.hpp"
#include "MetaAgent.hpp"
#include <vector>
#include <string>
#include <map>

// Structure pour configurer un MetaAgent
struct MetaAgentConfig {
    std::string name;
    std::map<int,double> characteristics;
    MetaAgentParams params;
    
    MetaAgentConfig(const std::string& agent_name = "Agent")
        : name(agent_name) {}
};

struct GlobalSolution {
    float reward;
    float cost;
    std::map<MetaAgent*, Solution> solution_to_meta_agent;
    
    bool empty() const {
        return solution_to_meta_agent.empty();
    }
    
    bool operator==(const GlobalSolution& other) const {
        return solution_to_meta_agent == other.solution_to_meta_agent;
    }
};

class GlobalSolutionConstructor {
private:
    GeoBox& geo_box;
    Pathfinder& pathfinder;
    GlobalMemory& global_memory;
    
    std::vector<MetaAgentConfig> meta_agent_configs;
    
    GlobalSolution initial_solution;

public:
    std::vector<MetaAgent*> MetaAgents;
    std::map<MetaAgent*, MetaAgentConfig> meta_agent_configurations;

    GlobalSolutionConstructor(
        GeoBox& box,
        Pathfinder& pf,
        GlobalMemory& mem
    );

    // Initialisation des densitÃ©s de reward (Ã  appeler AVANT run_all_meta_agents)
    void initialize_reward_densities();

    // Recherche TABU
    float objective_function(const GlobalSolution& solution);
    std::vector<GlobalSolution> get_neighbors(const GlobalSolution& solution);
    GlobalSolution tabu_search(int max_iterations, int tabu_list_size);
    
    void add_meta_agent_config(const MetaAgentConfig& config);
    void run_all_meta_agents();

    GlobalSolution initialize_global_solution();
    
    void print_summary() const;
    
};

#endif // GLOBAL_SOLUTION_CONSTRUCTOR_HPP