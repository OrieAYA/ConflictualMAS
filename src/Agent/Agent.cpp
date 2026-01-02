#include "Agent.hpp"
#include "../GeoBox/Box.hpp"
#include "../Common/Hashes.hpp"
#include "../Common/Memory.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>
#include <limits>
#include <deque>
#include <random>
#include <tuple>

// Constructeur
Agent::Agent(
    GeoBox& box, 
    Pathfinder& pf,
    GlobalMemory& mem,
    std::vector<int> a_char, 
    std::vector<osmium::object_id_type> init_sol,
    std::map<Solution, float>* meta_memory,
    std::vector<Solution>* pbest_list,
    double sim_threshold,
    int max_neighbors
) : geo_box(box), 
    PfSystem(pf),
    memory(mem),
    agent_charact(a_char), 
    initial_solution(),
    actual_solution(),
    fitness(std::numeric_limits<double>::max()),
    best_fitness(std::numeric_limits<double>::max()),
    meta_local_memory(meta_memory),
    validated_pbest(pbest_list),
    similarity_threshold(sim_threshold),
    max_neighbors_per_exploration(max_neighbors)
{
    std::vector<Path> paths;
    float cost = 0.0f;
    
    initial_solution = Solution(init_sol, paths, cost);
    actual_solution = initial_solution;
}

int Agent::objective_function(const Solution& sol) {
    int val = 0;

    for (const auto& node : sol.POIs){
        auto node_it = geo_box.data.nodes.find(node);
        if(node_it != geo_box.data.nodes.end()){
            for (const int& group_id : node_it->second.groupes){
                val += agent_charact[group_id];
            }
        }
    }

    return val;
}

bool Agent::is_better_solution(const Solution& sol1, const Solution& sol2) {
    if(sol1.POIs.size() > sol2.POIs.size()) return true;
    if(sol1.POIs.size() < sol2.POIs.size()) return false;
    return objective_function(sol1) > objective_function(sol2);
}

std::vector<Solution> Agent::get_add_neighbors(const Solution& sol) {
    std::vector<Solution> neighbors;
    
    if(sol.POIs.empty()) return neighbors;
    
    osmium::object_id_type last_visited = sol.POIs.back();
    float remaining_distance = memory.length_constraint - sol.cost;
    
    std::unordered_set<osmium::object_id_type> visited_pois(sol.POIs.begin(), sol.POIs.end());
    
    if (remaining_distance < 10.0f) {
        return neighbors;
    }
    
    std::vector<osmium::object_id_type> nearby_pois = memory.check_neighborhood(last_visited, visited_pois);
    
    if (nearby_pois.empty()) {
        return neighbors;
    }
    
    int& explored_count = explored_neighbors_count[last_visited];
    
    //Param
    const int MAX_ATTEMPTS = 3;
    const int MAX_EXPANSION_ATTEMPTS = 1;
    int attempts = 0;
    int expansion_attempts = 0;
    
    while (neighbors.empty() && attempts < MAX_ATTEMPTS) {
        int start_index = explored_count;
        int end_index = std::min(start_index + max_neighbors_per_exploration, 
                                static_cast<int>(nearby_pois.size()));
        
        if (start_index >= static_cast<int>(nearby_pois.size())) {
            if (expansion_attempts >= MAX_EXPANSION_ATTEMPTS) {
                break;
            }
            
            if (remaining_distance > 100.0f) {
                
                size_t old_size = nearby_pois.size();
                nearby_pois = memory.continue_neighborhood_search(last_visited, visited_pois);
                
                expansion_attempts++;
                
                if (nearby_pois.size() <= old_size) {
                    break;
                }
                
                explored_count = static_cast<int>(old_size);
                start_index = explored_count;
                end_index = std::min(start_index + max_neighbors_per_exploration, 
                                    static_cast<int>(nearby_pois.size()));
            } else {
                break;
            }
        }
        
        if (start_index >= end_index) {
            std::cout << "    [ERREUR] Indices invalides (start=" << start_index 
                      << ", end=" << end_index << ")\n";
            break;
        }
        
        int valid_count = 0;
        int rejected_similarity = 0;
        int rejected_visited = 0;
        int rejected_distance = 0;
        
        // Explorer le batch actuel
        for (int i = start_index; i < end_index; i++) {
            osmium::object_id_type potential_neighbor = nearby_pois[i];
            
            if (visited_pois.count(potential_neighbor)) {
                rejected_visited++;
                continue;
            }
            
            Path to_neighbor = memory.check_path(last_visited, potential_neighbor);
            
            if (to_neighbor.cost <= 0.0f || to_neighbor.path_edges.empty()) {
                continue;
            }
            
            float new_total_cost = sol.cost + to_neighbor.cost;
            
            if (new_total_cost > memory.length_constraint) {
                rejected_distance++;
                continue;
            }
            
            Solution candidate = sol;
            candidate.add_node(potential_neighbor, to_neighbor);
            
            // Vérification similarité
            bool too_similar = false;
            if (validated_pbest != nullptr && !validated_pbest->empty()) {
                for (const auto& pbest : *validated_pbest) {
                    float similarity = memory.calculate_similarity(candidate, pbest);
                    
                    if (similarity > similarity_threshold) {
                        too_similar = true;
                        rejected_similarity++;
                        break;
                    }
                }
            }
            
            if (!too_similar) {
                neighbors.push_back(candidate);
                valid_count++;
                
                if (neighbors.size() >= static_cast<size_t>(max_neighbors_per_exploration)) {
                    break;
                }
            }
        }
        
        explored_count = end_index;
        
        if (!neighbors.empty()) {
            break;
        }
        
        attempts++;
        
        if (!(explored_count < static_cast<int>(nearby_pois.size()))) {
            break;
        }
    }
    
    return neighbors;
}

std::vector<Solution> Agent::get_backtrack_neighbors(const Solution& sol, int backtrack_depth) {
    std::vector<Solution> neighbors;
    
    if(sol.POIs.size() <= static_cast<size_t>(backtrack_depth)) return neighbors;
    
    // Créer une solution avec les N derniers POIs retirés
    Solution truncated = sol;
    for(int i = 0; i < backtrack_depth; i++){
        if(!truncated.POIs.empty()){
            truncated.POIs.pop_back();
            if(!truncated.paths.empty()){
                truncated.cost -= truncated.paths.back().cost;
                truncated.paths.pop_back();
            }
        }
    }
    
    // Depuis cette solution tronquée, ajouter de nouveaux POIs
    std::vector<Solution> add_neighbors = get_add_neighbors(truncated);
    
    // Ne garder que ceux qui sont différents de la solution originale
    for(const auto& neighbor : add_neighbors){
        if(neighbor.POIs != sol.POIs){
            neighbors.push_back(neighbor);
        }
    }
    
    return neighbors;
}

// Générer tous les voisins (ADD + BACKTRACK à différentes profondeurs)
std::vector<Solution> Agent::get_all_neighbors(const Solution& sol, int max_backtrack) {
    std::vector<Solution> all_neighbors;
    
    // Type 1: ADD (ajouter un POI)
    std::vector<Solution> add_neighbors = get_add_neighbors(sol);
    all_neighbors.insert(all_neighbors.end(), add_neighbors.begin(), add_neighbors.end());
    
    // Type 2: BACKTRACK (retirer 1, 2, ..., max_backtrack POIs et ajouter un nouveau)
    for(int depth = 1; depth <= max_backtrack && depth < static_cast<int>(sol.POIs.size()); depth++){
        std::vector<Solution> backtrack_neighbors = get_backtrack_neighbors(sol, depth);
        all_neighbors.insert(all_neighbors.end(), backtrack_neighbors.begin(), backtrack_neighbors.end());
    }
    
    return all_neighbors;
}

// Vérifier si une solution est dans la liste tabu
bool Agent::is_tabu(const Solution& sol, const std::deque<std::vector<osmium::object_id_type>>& tabu_list) {
    for(const auto& tabu_state : tabu_list){
        if(tabu_state == sol.POIs){
            return true;
        }
    }
    return false;
}

// Recherche Tabu métaheuristique
Solution Agent::tabu_search(int max_iterations, int tabu_list_size) {
    Solution best_solution = initial_solution;
    Solution current_solution = initial_solution;
    
    std::deque<std::vector<osmium::object_id_type>> tabu_list;
    
    int best_fitness = objective_function(best_solution);
    size_t best_poi_count = best_solution.POIs.size();
    
    int iterations_without_improvement = 0;
    int max_backtrack_depth = 2;  // Param : Profondeur de backtrack initiale
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(int iter = 0; iter < max_iterations; iter++){
        // Augmenter la profondeur de backtrack si pas d'amélioration
        if(iterations_without_improvement > 10){
            max_backtrack_depth = std::min(max_backtrack_depth + 1, 5);
            iterations_without_improvement = 0;
        }
        
        // Générer tous les voisins (ADD + BACKTRACK)
        std::vector<Solution> neighbors = get_all_neighbors(current_solution, max_backtrack_depth);
        
        if(neighbors.empty()){
            break;
        }
        
        // Chercher le meilleur voisin non-tabu ou satisfaisant le critère d'aspiration
        Solution best_neighbor;
        bool found_valid = false;
        bool used_aspiration = false;
        
        for(const Solution& neighbor : neighbors){
            bool is_tabu_solution = is_tabu(neighbor, tabu_list);
            
            // Critère d'aspiration : accepter une solution tabu si elle est meilleure que best_solution
            bool aspiration = is_tabu_solution && is_better_solution(neighbor, best_solution);
            
            if(!is_tabu_solution || aspiration){
                if(!found_valid || is_better_solution(neighbor, best_neighbor)){
                    best_neighbor = neighbor;
                    found_valid = true;
                    if(aspiration) used_aspiration = true;
                }
            }
        }
        
        if(!found_valid){
            break;
        }
        
        // Mise à jour
        current_solution = best_neighbor;
        tabu_list.push_back(best_neighbor.POIs);
        
        if(static_cast<int>(tabu_list.size()) > tabu_list_size){
            tabu_list.pop_front();
        }
        
        int current_fitness = objective_function(current_solution);
        size_t current_poi_count = current_solution.POIs.size();
        
        // Mettre à jour la meilleure solution
        if(is_better_solution(current_solution, best_solution)){
            best_solution = current_solution;
            best_fitness = current_fitness;
            best_poi_count = current_poi_count;
            iterations_without_improvement = 0;
            max_backtrack_depth = 2;  // Reset backtrack
        } else {
            iterations_without_improvement++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "FIN RECHERCHE TABU\n";
    std::cout << "Temps d'exécution: " << duration.count() << " ms\n";
    
    float cumulative_distance = 0.0f;
    for (size_t i = 0; i < best_solution.POIs.size(); i++) {
        osmium::object_id_type poi_id = best_solution.POIs[i];
        
        auto node_it = geo_box.data.nodes.find(poi_id);
        if (node_it != geo_box.data.nodes.end() && !node_it->second.groupes.empty()) {
            int node_value = 0;
            bool first = true;
            for (const auto& group_id : node_it->second.groupes) {
                int value = agent_charact[group_id];
                if (value > 0) {
                    node_value += value;
                }
                first = false;
            }
        }
        
        if (i < best_solution.paths.size()) {
            float segment_distance = best_solution.paths[i].cost;
            cumulative_distance += segment_distance;
        }
    }

    return best_solution;
}