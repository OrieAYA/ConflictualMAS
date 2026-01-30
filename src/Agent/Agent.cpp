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
    max_reward_per_meter = 0.0f;
    visited_POIs = 0;
    
    for(auto it = parent->characteristics.begin(); it != parent->characteristics.end(); ++it){
        characteristics.insert(it->first);
    }
}

// ============================================================================
// CALCULATE MAX REWARD DENSITY
// ============================================================================
void Agent::calculate_max_reward_density() {
    max_reward_per_meter = 0.0f;
    
    for(const auto& [poi_id, node_data] : geo_box.data.nodes){
        int reward = get_poi_reward(poi_id);
        if(reward <= 0) continue;
        
        float min_dist = 100.0f;  // Distance minimale estimée
        float density = static_cast<float>(reward) / min_dist;
        
        if(density > max_reward_per_meter){
            max_reward_per_meter = density;
        }
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
// EVALUATE UPPER BOUND (ancienne version pour compatibilité)
// ============================================================================
bool Agent::evaluate_upper_bound(const Solution& solution) {
    double upper = estimate_upper_bound(solution);
    return upper >= parent->min_fitness;
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
// GREEDY CONSTRUCTION
// ============================================================================
Solution Agent::greedy_construction(Solution& start_solution) {
    if(start_solution.POIs.empty()){
        return start_solution;
    }
    
    Solution current = start_solution;
    std::unordered_set<osmium::object_id_type> visited(
        current.POIs.begin(), 
        current.POIs.end()
    );
    
    int consecutive_failures = 0;
    const int MAX_FAILURES = 10;
    
    while(current.cost < memory.length_constraint){
        if(current.POIs.empty()) break;
        
        osmium::object_id_type last_poi = current.POIs.back();
        
        osmium::object_id_type neighbor = memory.check_neighborhood(
            last_poi, 
            characteristics, 
            visited
        );
        
        if(neighbor == 0){
            consecutive_failures++;
            if(consecutive_failures >= MAX_FAILURES){
                break;
            }
            continue;
        }
        
        consecutive_failures = 0;
        
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
    }
    
    return current;
}

// ============================================================================
// GREEDY EXTEND FROM
// ============================================================================
Solution Agent::greedy_extend_from(
    const Solution& base, 
    osmium::object_id_type forbidden_first
) {
    Solution current = base;
    std::unordered_set<osmium::object_id_type> visited(
        base.POIs.begin(), 
        base.POIs.end()
    );
    
    bool first_add = true;
    int consecutive_failures = 0;
    const int MAX_FAILURES = 5;
    
    while(current.cost < memory.length_constraint){
        if(current.POIs.empty()) break;
        
        osmium::object_id_type last_poi = current.POIs.back();
        
        osmium::object_id_type neighbor = memory.check_neighborhood(
            last_poi, 
            characteristics, 
            visited
        );
        
        if(neighbor == 0){
            consecutive_failures++;
            if(consecutive_failures >= MAX_FAILURES) break;
            continue;
        }
        
        // Skip forbidden au premier ajout
        if(first_add && neighbor == forbidden_first){
            visited.insert(neighbor);
            continue;
        }
        
        consecutive_failures = 0;
        
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
    
    if(solution.POIs.size() <= 1){
        return decomposed;
    }
    
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
// EXPLORE MULTIBRANCH - VERSION OPTIMISÉE
// ============================================================================
std::vector<Solution> Agent::explore_multibranch(
    const Solution& base_solution,
    osmium::object_id_type forbidden_first,
    double threshold
) {
    std::vector<Solution> promising_branches;
    
    if(base_solution.empty() || base_solution.POIs.empty()){
        return promising_branches;
    }
    
    osmium::object_id_type last_poi = base_solution.POIs.back();
    std::unordered_set<osmium::object_id_type> visited(
        base_solution.POIs.begin(), 
        base_solution.POIs.end()
    );
    
    // Récupérer premier voisin pour initialiser le cache
    osmium::object_id_type first_neighbor = memory.check_neighborhood(
        last_poi, 
        characteristics, 
        visited
    );
    
    if(first_neighbor == 0){
        return promising_branches;
    }
    
    // Récupérer cache de voisins
    auto cache_it = memory.visited_Neighborhoods.find(last_poi);
    if(cache_it == memory.visited_Neighborhoods.end()){
        return promising_branches;
    }
    
    const std::vector<osmium::object_id_type>& all_neighbors = cache_it->second;
    
    // ========================================
    // OPTIMISATION : LIMITER À 5 VOISINS MAX
    // ========================================
    int explored = 0;
    int pruned = 0;
    const int MAX_NEIGHBORS_TO_EXPLORE = 5;
    
    for(const auto& neighbor : all_neighbors){
        // EARLY STOP
        if(explored >= MAX_NEIGHBORS_TO_EXPLORE){
            break;
        }
        
        // Skip forbidden
        if(neighbor == forbidden_first) continue;
        
        // Skip déjà visités
        if(visited.count(neighbor) > 0) continue;
        
        // Vérifier distance
        std::vector<Path> paths = memory.check_path(last_poi, neighbor);
        if(paths.empty() || paths[0].cost == std::numeric_limits<float>::max()){
            continue;
        }
        
        Path to_neighbor = paths[0];
        
        if(base_solution.cost + to_neighbor.cost > memory.length_constraint){
            continue;
        }
        
        // Créer solution temporaire
        Solution temp = base_solution;
        temp.add_node(neighbor, to_neighbor);
        
        // Upper bound check
        double upper = estimate_upper_bound(temp);
        
        if(upper < threshold){
            pruned++;
            continue;
        }
        
        // Greedy extend
        Solution rebuilt = greedy_extend_from(temp, forbidden_first);
        
        if(!rebuilt.empty()){
            promising_branches.push_back(rebuilt);
            explored++;
        }
    }
    
    return promising_branches;
}

// ============================================================================
// VND LOCAL SEARCH
// ============================================================================
Solution Agent::vnd_local_search(const Solution& solution) {
    // Version simplifiée - juste retourner la solution
    // Tu peux implémenter swap, reverse, etc. plus tard
    return solution;
}

// ============================================================================
// AGENT SEARCH - MÉTHODE PRINCIPALE OPTIMISÉE
// ============================================================================
// ============================================================================
// AGENT SEARCH - VERSION ADAPTATIVE
// ============================================================================
Solution Agent::agent_search(int max_iterations, int tabu_list_size) {
    
    calculate_max_reward_density();
    
    // Construction initiale
    Solution current = greedy_construction(initial_solution);
    
    if(current.empty() || current.POIs.size() < 2){
        return current;
    }
    
    pbest = current;
    pbest_fitness = parent->objective_function(pbest);
    
    double meta_threshold = parent->min_fitness;
    
    // Early stop
    if(pbest_fitness < meta_threshold * 0.5){
        return Solution();
    }
    
    // ========================================
    // PARAMÈTRES ADAPTATIFS SELON TAILLE
    // ========================================
    int solution_size = static_cast<int>(pbest.POIs.size());
    
    int k_max;
    double threshold_factor;
    int max_decompositions;
    int max_neighbors;
    
    if(solution_size < 20){
        // PETIT SET : Recherche exhaustive, peu de pruning
        k_max = 3;
        threshold_factor = 0.4;  // Moins strict
        max_decompositions = 15;
        max_neighbors = 10;
    }
    else if(solution_size < 50){
        // MOYEN SET : Équilibré
        k_max = 4;
        threshold_factor = 0.6;
        max_decompositions = 10;
        max_neighbors = 7;
    }
    else{
        // GRAND SET : Pruning agressif
        k_max = 5;
        threshold_factor = 0.7;
        max_decompositions = 8;
        max_neighbors = 5;
    }
    
    int k = 1;
    int iter = 0;
    int consecutive_empty = 0;
    const int max_consecutive_empty = 3;
    
    while(iter < max_iterations && k <= k_max){
        
        // ========================================
        // A. SHAKE
        // ========================================
        Solution shaken = pbest;
        for(int i = 0; i < k && shaken.POIs.size() > 1; i++){
            shaken.remove_node();
        }
        
        if(shaken.POIs.empty()){
            k++;
            iter++;
            continue;
        }
        
        // ========================================
        // B. DÉCOMPOSITION
        // ========================================
        std::vector<DecomposedSolution> decomposed = decompose_with_forbidden(shaken);
        
        if(decomposed.empty()){
            k++;
            iter++;
            continue;
        }
        
        // ========================================
        // C. FILTRAGE PROMISING ADAPTATIF
        // ========================================
        double threshold = pbest_fitness * threshold_factor;
        std::vector<DecomposedSolution> promising;
        int pruned = 0;
        
        for(const auto& decomp : decomposed){
            double upper = estimate_upper_bound(decomp.base);
            
            if(upper >= threshold){
                promising.push_back(decomp);
            } else {
                pruned++;
            }
        }
        
        // ========================================
        // LIMITE ADAPTATIVE
        // ========================================
        if(promising.size() > max_decompositions){
            std::sort(promising.begin(), promising.end(),
                [this](const DecomposedSolution& a, const DecomposedSolution& b){
                    return estimate_upper_bound(a.base) > estimate_upper_bound(b.base);
                });
            promising.resize(max_decompositions);
        }
        
        if(promising.empty()){
            consecutive_empty++;
            k++;
            iter++;
            
            if(consecutive_empty >= max_consecutive_empty){
                break;
            }
            
            continue;
        }
        
        consecutive_empty = 0;
        
        // ========================================
        // D. EXPLORATION MULTI-BRANCH ADAPTATIVE
        // ========================================
        Solution best_from_decomp;
        double best_decomp_fitness = -1.0;
        
        for(const auto& decomp : promising){
            std::vector<Solution> branches = explore_multibranch_adaptive(
                decomp.base,
                decomp.forbidden_next,
                threshold,
                max_neighbors
            );
            
            if(branches.empty()) continue;
            
            // Meilleure branche
            Solution best_branch;
            double best_branch_fitness = -1.0;
            
            for(const auto& branch : branches){
                double fitness = parent->objective_function(branch);
                if(fitness > best_branch_fitness){
                    best_branch_fitness = fitness;
                    best_branch = branch;
                }
            }
            
            if(best_branch_fitness > best_decomp_fitness){
                best_decomp_fitness = best_branch_fitness;
                best_from_decomp = best_branch;
            }
        }
        
        // ========================================
        // E. MOVE OR NOT
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
    // F. VÉRIFICATION FINALE
    // ========================================
    if(pbest_fitness < meta_threshold){
        return Solution();
    }
    
    return pbest;
}

// ============================================================================
// EXPLORE MULTIBRANCH ADAPTATIF
// ============================================================================
std::vector<Solution> Agent::explore_multibranch_adaptive(
    const Solution& base_solution,
    osmium::object_id_type forbidden_first,
    double threshold,
    int max_neighbors  // ← Paramètre adaptatif
) {
    std::vector<Solution> promising_branches;
    
    if(base_solution.empty() || base_solution.POIs.empty()){
        return promising_branches;
    }
    
    osmium::object_id_type last_poi = base_solution.POIs.back();
    std::unordered_set<osmium::object_id_type> visited(
        base_solution.POIs.begin(), 
        base_solution.POIs.end()
    );
    
    osmium::object_id_type first_neighbor = memory.check_neighborhood(
        last_poi, 
        characteristics, 
        visited
    );
    
    if(first_neighbor == 0){
        return promising_branches;
    }
    
    auto cache_it = memory.visited_Neighborhoods.find(last_poi);
    if(cache_it == memory.visited_Neighborhoods.end()){
        return promising_branches;
    }
    
    const std::vector<osmium::object_id_type>& all_neighbors = cache_it->second;
    
    int explored = 0;
    int pruned = 0;
    
    for(const auto& neighbor : all_neighbors){
        if(explored >= max_neighbors) break;
        
        if(neighbor == forbidden_first) continue;
        if(visited.count(neighbor) > 0) continue;
        
        std::vector<Path> paths = memory.check_path(last_poi, neighbor);
        if(paths.empty() || paths[0].cost == std::numeric_limits<float>::max()){
            continue;
        }
        
        Path to_neighbor = paths[0];
        
        if(base_solution.cost + to_neighbor.cost > memory.length_constraint){
            continue;
        }
        
        Solution temp = base_solution;
        temp.add_node(neighbor, to_neighbor);
        
        double upper = estimate_upper_bound(temp);
        
        if(upper < threshold){
            pruned++;
            continue;
        }
        
        Solution rebuilt = greedy_extend_from(temp, forbidden_first);
        
        if(!rebuilt.empty()){
            promising_branches.push_back(rebuilt);
            explored++;
        }
    }
    
    return promising_branches;
}