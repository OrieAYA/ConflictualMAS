#include "Agent.hpp"
#include "MetaAgent.hpp"
#include "../GeoBox/Box.hpp"
#include "../Common/Hashes.hpp"
#include "../Common/Memory.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>
#include <limits>
#include <random>
#include <tuple>

// ============================================================================
// CONSTRUCTEUR
// ============================================================================
Agent::Agent(
    GeoBox& box, 
    Pathfinder& pf,
    GlobalMemory& mem,
    std::vector<osmium::object_id_type> init_sol,
    MetaAgent* meta_agent
) : geo_box(box), 
    PfSystem(pf),
    memory(mem),
    parent(meta_agent)
{
    std::vector<Path> paths;
    initial_solution = Solution(init_sol, paths, 0.0f);
    pbest = initial_solution;
    pbest_fitness = 0.0;
    max_reward_per_meter = parent->max_reward_per_meter;  // Récupéré depuis MetaAgent
    
    for(auto it = parent->characteristics.begin(); it != parent->characteristics.end(); ++it){
        characteristics.insert(it->first);
    }
}

// ============================================================================
// ESTIMATE UPPER BOUND
// ============================================================================
double Agent::estimate_upper_bound(const Solution& solution) {
    double current_fitness = parent->objective_function(solution);
    float remaining_distance = memory.length_constraint - solution.cost;
    
    if(remaining_distance <= 0.0f){
        return current_fitness;
    }
    
    double max_additional_reward = remaining_distance * max_reward_per_meter;
    return current_fitness + max_additional_reward;
}

// ============================================================================
// GET POI REWARD
// ============================================================================
int Agent::get_poi_reward(osmium::object_id_type poi_id) {
    auto node_it = geo_box.data.nodes.find(poi_id);
    if(node_it == geo_box.data.nodes.end()) return 0;
    
    int reward = 0;
    for(const int group_id : node_it->second.groupes){
        auto char_it = parent->characteristics.find(group_id);
        if(char_it != parent->characteristics.end()){
            reward += static_cast<int>(char_it->second);
        }
    }
    
    return reward;
}

// ============================================================================
// GREEDY EXTEND FROM
// ============================================================================
Solution Agent::greedy_construction(
    const Solution& base, 
    osmium::object_id_type forbidden_first
) {
    Solution current = base;
    std::unordered_set<osmium::object_id_type> visited(
        base.POIs.begin(), 
        base.POIs.end()
    );
    
    bool first_add = true;
    
    while(current.cost < memory.length_constraint){
        
        osmium::object_id_type last_poi = current.POIs.back();
        
        osmium::object_id_type neighbor = memory.check_neighborhood(
            last_poi, 
            characteristics, 
            visited
        );
        
        if(neighbor == 0){
            break;  // Plus de voisins disponibles depuis ce POI
        }
        
        // Skip forbidden au premier ajout uniquement
        if(first_add && neighbor == forbidden_first){
            visited.insert(neighbor);
            first_add = false;
            continue;
        }
        
        std::vector<Path> paths = memory.check_path(last_poi, neighbor);
        
        if(paths.empty() || paths[0].cost == std::numeric_limits<float>::max()){
            visited.insert(neighbor);
            continue;
        }
        
        Path path = paths[0];
        
        if(current.cost + path.cost > memory.length_constraint){
            break;
        }
        
        current.add_node(neighbor, path);
        visited.insert(neighbor);
        first_add = false;
    }
    
    return current;
}

// ============================================================================
// DECOMPOSE WITH FORBIDDEN
// ============================================================================
std::vector<DecomposedSolution> Agent::decompose_with_forbidden(const Solution& solution) {
    std::vector<DecomposedSolution> decomposed;
    
    Solution current = solution;
    
    while(current.POIs.size() > 1){
        osmium::object_id_type forbidden = current.POIs.back();
        current.remove_node();
        
        if(!current.POIs.empty()){
            DecomposedSolution decomp;
            decomp.base = current;
            decomp.forbidden_next = forbidden;
            decomposed.push_back(decomp);
        }
    }
    
    return decomposed;
}

// ============================================================================
// AGENT SEARCH
// ============================================================================
Solution Agent::agent_search(int max_iterations, int tabu_list_size) {
    
    // Construction initiale
    Solution current = greedy_construction(initial_solution, 0);
    
    if(current.empty() || current.POIs.size() <= 1){
        return current;
    }
    
    pbest = current;
    pbest_fitness = parent->objective_function(pbest);
    
    double threshold = parent->min_fitness;

    if(pbest_fitness < threshold){
        return Solution();
    }
    
    // ========================================
    // PARAMETERS
    // ========================================
    int k_max = 4;
    double threshold_factor = parent->params.max_divergence_from_gbest;
    int max_decompositions = 5;
    
    int k = 1;
    int iter = 0;
    int consecutive_empty = 0;
    const int max_consecutive_empty = 3;
    
    while(iter < max_iterations && k <= k_max){
        
        // ========================================
        // A. DECOMPOSITION
        // ========================================
        Solution shaken = pbest;
        for(int i = 0; i < k && shaken.POIs.size() > 1; i++){
            shaken.remove_node();
        }

        std::vector<DecomposedSolution> decomposed = decompose_with_forbidden(shaken);
        
        if(decomposed.empty()){
            k++;
            iter++;
            continue;
        }
        
        // ========================================
        // B. FILTRAGE PROMISING ADAPTATIF
        // ========================================
        double local_threshold = pbest_fitness * threshold_factor;
        std::vector<DecomposedSolution> promising;
        int pruned = 0;
        
        for(const auto& decomp : decomposed){
            double upper = estimate_upper_bound(decomp.base);
            
            if(upper >= local_threshold){
                promising.push_back(decomp);
            } else {
                pruned++;
            }
        }
        
        // ========================================
        // LIMITE ADAPTATIVE
        // ========================================
        if(promising.empty()){
            consecutive_empty++;
            k++;
            iter++;
            
            if(consecutive_empty >= max_consecutive_empty){
                break;
            }
            
            continue;
        }

        if(promising.size() > max_decompositions){
            std::sort(promising.begin(), promising.end(),
                [this](const DecomposedSolution& a, const DecomposedSolution& b){
                    return estimate_upper_bound(a.base) > estimate_upper_bound(b.base);
                });
            promising.resize(max_decompositions);
        }
        
        consecutive_empty = 0;
        
        // ========================================
        // C. EXPLORATION - 1 GREEDY PAR DÉCOMPOSITION
        // ========================================
        Solution best_from_decomp;
        double best_decomp_fitness = -1.0;
        
        for(const auto& decomp : promising){
            Solution rebuilt = greedy_construction(decomp.base, decomp.forbidden_next);
            
            if(rebuilt.empty()) continue;
            
            double fitness = parent->objective_function(rebuilt);
            
            if(fitness > best_decomp_fitness){
                best_decomp_fitness = fitness;
                best_from_decomp = rebuilt;
            }
        }
        
        // ========================================
        // D. ACCEPTATION
        // ========================================
        if(best_decomp_fitness > pbest_fitness){
            pbest = best_from_decomp;
            pbest_fitness = best_decomp_fitness;
            current = pbest;
            k = 1;
        } else {
            k++;
        }
        
        iter++;
    }
    
    // ========================================
    // E. VÉRIFICATION FINALE
    // ========================================
    if(pbest_fitness < threshold){
        return Solution();
    }
    
    return pbest;
}