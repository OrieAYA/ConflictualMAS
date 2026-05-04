#include "Agent.hpp"
#include "MetaAgent.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Hashes.hpp"
#include "Legacy/Common/Memory.hpp"
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
    pbest_fitness = 0.0;
    max_reward_per_meter = parent->max_reward_per_meter;  // Récupéré depuis MetaAgent
    
    for(auto it = parent->characteristics.begin(); it != parent->characteristics.end(); ++it){
        characteristics.insert(it->first);
    }
}

// ============================================================================
// ESTIMATE UPPER BOUND
// ============================================================================
double Agent::estimate_upper_bound(DecomposedSolution* solution) {

    if(solution->base.POIs.empty()) return 0.0;

    double current_fitness = parent->objective_function(solution->base);
    float remaining_distance = memory.length_constraint - solution->base.cost;

    osmium::object_id_type next_neighbor = memory.check_neighborhood(
        solution->base.POIs.back(), 
        characteristics, 
        solution->forbidden
    );
    
    if(next_neighbor != 0) {
        std::vector<Path> paths = memory.check_path(solution->base.POIs.back(), next_neighbor);
        if(!paths.empty() && paths[0].cost != std::numeric_limits<float>::max()) {
            remaining_distance -= paths[0].cost;
        }
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
DecomposedSolution* Agent::greedy_construction(DecomposedSolution* current) {
    
    if(current->base.POIs.empty()) return current;
    
    osmium::object_id_type last_poi = current->base.POIs.back();
    
    osmium::object_id_type neighbor = memory.check_neighborhood(
        last_poi,
        characteristics, 
        current->forbidden
    );
    
    if(neighbor == 0) {
        return current;
    }

    if(current->childs.find(neighbor) != current->childs.end()) {
        return greedy_construction(current->childs[neighbor]);
    }
    
    std::vector<Path> paths = memory.check_path(last_poi, neighbor);
    
    if(paths.empty() || paths[0].cost == std::numeric_limits<float>::max()) {
        current->forbidden.insert(neighbor);
        return greedy_construction(current);
    }
    
    Path path = paths[0];
    
    if(current->base.cost + path.cost > memory.length_constraint) {
        return current;
    }

    Solution child_solution = current->base;
    child_solution.add_node(neighbor, path);

    DecomposedSolution* child = new DecomposedSolution(current, child_solution);
    current->childs[neighbor] = child;
    
    return greedy_construction(child);
}

// ============================================================================
// DECOMPOSE WITH FORBIDDEN
// ============================================================================
std::vector<DecomposedSolution*> Agent::decompose_with_forbidden(DecomposedSolution* solution) {
    std::vector<DecomposedSolution*> decomposed;
    
    DecomposedSolution* current = solution;
    
    while(current->parent != nullptr && current->base != initial_solution) {
        osmium::object_id_type forbidden = current->base.POIs.back();
        current = current->parent;
        
        current->forbidden.insert(forbidden);
        decomposed.push_back(current);
    }
    
    return decomposed;
}

// ============================================================================
// DELETE DECOMPOSITION TREE
// ============================================================================
void Agent::delete_decomposition_tree(DecomposedSolution* root) {
    if(root == nullptr) return;
    
    // Supprimer récursivement tous les enfants
    for(auto& [poi_id, child] : root->childs) {
        delete_decomposition_tree(child);
    }
    
    delete root;
}

// ============================================================================
// AGENT SEARCH
// ============================================================================
Solution Agent::agent_search(int max_iterations, int tabu_list_size) {
    
    DecomposedSolution* initial_decomposition = new DecomposedSolution(initial_solution);

    DecomposedSolution* current = greedy_construction(initial_decomposition);

    if(current->base.empty() || current->base.POIs.size() <= 1) {
        Solution result = current->base;
        delete_decomposition_tree(initial_decomposition);
        return result;
    }
    
    pbest = *current;
    pbest_fitness = parent->objective_function(pbest.base);
    
    double threshold = parent->min_fitness;
    
    // ========================================
    // PARAMETERS
    // ========================================
    int k_max = 4;
    double threshold_factor = 1 - parent->params.max_divergence_from_gbest;
    int max_decompositions = 5;
    
    int k = 1;
    int iter = 0;
    int consecutive_empty = 0;
    const int max_consecutive_empty = 3;
    
    while(iter < max_iterations) {
        
        // ========================================
        // A. SHAKE (Decomposition)
        // ========================================
        DecomposedSolution* shaken = &pbest;
        int until = std::min(
            int(shaken->base.POIs.size() / k_max) * (k - 1), 
            std::abs(2 - static_cast<int>(shaken->base.POIs.size()))
        );
        
        for(int i = 0; i < until; i++) {
            if(shaken->parent != nullptr) {
                shaken = shaken->parent;
            } else {
                break;
            }
        }

        std::vector<DecomposedSolution*> decomposed = decompose_with_forbidden(shaken);
        
        if(decomposed.empty()) {
            k++;
            iter++;
            continue;
        }
        
        // ========================================
        // B. FILTRAGE PROMISING ADAPTATIF
        // ========================================
        double local_threshold = pbest_fitness * threshold_factor;
        std::vector<DecomposedSolution*> promising;
        int pruned = 0;
        
        for(auto* decomp : decomposed) {
            double upper = estimate_upper_bound(decomp);
            
            if(upper >= local_threshold) {
                promising.push_back(decomp);
            } else {
                pruned++;
            }
        }
        
        // ========================================
        // LIMITE ADAPTATIVE
        // ========================================
        if(promising.empty()) {
            consecutive_empty++;
            k++;
            iter++;
            
            if(consecutive_empty >= max_consecutive_empty) {
                break;
            }
            
            continue;
        }

        if(promising.size() > max_decompositions) {
            std::sort(promising.begin(), promising.end(),
                [this](DecomposedSolution* a, DecomposedSolution* b) {
                    return estimate_upper_bound(a) > estimate_upper_bound(b);
                });
            promising.resize(max_decompositions);
        }
        
        consecutive_empty = 0;
        
        // ========================================
        // C. EXPLORATION - 1 GREEDY PAR DÉCOMPOSITION
        // ========================================
        DecomposedSolution* best_from_decomp = nullptr;
        double best_decomp_fitness = -1.0;
        
        for(auto* decomp : promising) {
            DecomposedSolution* rebuilt = greedy_construction(decomp);
            
            if(rebuilt->base.empty()) continue;
            
            double fitness = parent->objective_function(rebuilt->base);
            
            if(fitness > best_decomp_fitness) {
                best_decomp_fitness = fitness;
                best_from_decomp = rebuilt;
            }
        }

        if(best_from_decomp != nullptr) {
            current = best_from_decomp;
        }
        
        // ========================================
        // D. ACCEPTATION
        // ========================================
        if(best_decomp_fitness > pbest_fitness) {
            pbest = *current;
            pbest_fitness = best_decomp_fitness;
            k = 1;
        } else {
            if(k!=k_max)k++;
        }

        if(pbest_fitness < threshold) {
            Solution result = Solution();
            delete_decomposition_tree(initial_decomposition);
            return result;
        }
        
        iter++;
    }
    
    // ========================================
    // E. VÉRIFICATION FINALE
    // ========================================
    
    delete_decomposition_tree(initial_decomposition);
    
    return pbest.base;
}