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
    std::vector<int> characteristics;
    MetaAgentParams params;
    
    MetaAgentConfig(const std::string& agent_name = "Agent")
        : name(agent_name), characteristics(100, 0) {}
};

// Structure pour stocker les résultats d'un MetaAgent
struct MetaAgentResult {
    std::string name;
    Solution gbest;
    std::vector<Solution> validated_pbest;
    int gbest_fitness;
    int agent_count;
    float coverage_rate;
    
    bool operator==(const MetaAgentResult& other) const {
        return name == other.name;
    }
    
    bool operator!=(const MetaAgentResult& other) const {
        return !(*this == other);
    }

    bool operator<(const MetaAgentResult& other) const {
        return name < other.name;
    }
};

struct GlobalSolution {
    float reward;
    float cost;
    std::map<MetaAgentResult, Solution> solution_to_meta_agent;
    
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
    std::vector<MetaAgentResult> results;
    
    GlobalSolution initial_solution;

public:
    GlobalSolutionConstructor(
        GeoBox& box,
        Pathfinder& pf,
        GlobalMemory& mem
    );

    // Recherche TABU
    float objective_function(const GlobalSolution& solution);
    std::vector<GlobalSolution> get_neighbors(const GlobalSolution& solution);
    GlobalSolution tabu_search(int max_iterations, int tabu_list_size);
    
    void add_meta_agent_config(const MetaAgentConfig& config);
    void run_all_meta_agents();

    GlobalSolution initialize_global_solution();
    
    const std::vector<MetaAgentResult>& get_results() const { return results; }
    const MetaAgentResult& get_result(size_t index) const { return results[index]; }
    size_t get_result_count() const { return results.size(); }
    
    void print_summary() const;
    
    std::vector<Solution> get_all_validated_pbest() const;
    std::vector<Solution> get_all_gbest() const;
};

#endif // GLOBAL_SOLUTION_CONSTRUCTOR_HPP