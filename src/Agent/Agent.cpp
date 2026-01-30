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

// Constructeur - INCHANGÉ
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
    
    calculate_max_reward_density();
}

void Agent::calculate_max_reward_density() {
    max_reward_per_meter = 0.0f;
    
    for(const auto& group_id : characteristics) {
        auto char_it = parent->characteristics.find(group_id);
        if(char_it == parent->characteristics.end()) continue;
        
        double reward = char_it->second;
        float min_distance = 50.0f;
        
        float density = static_cast<float>(reward) / min_distance;
        max_reward_per_meter = std::max(max_reward_per_meter, density);
    }
}

bool Agent::evaluate_upper_bound(const Solution& solution){
    double current_fitness = parent->objective_function(solution);
    float remaining_distance = memory.length_constraint - solution.cost;

    double max_additional_reward = remaining_distance * max_reward_per_meter;

    if((current_fitness + max_additional_reward) < static_cast<float>(parent->min_fitness)) return false;

    return true;
}

double Agent::estimate_upper_bound(const Solution& solution){
    double current_fitness = parent->objective_function(solution);
    float remaining_distance = memory.length_constraint - solution.cost;
    double max_additional_reward = remaining_distance * max_reward_per_meter;
    return current_fitness + max_additional_reward;
}

int Agent::get_poi_reward(osmium::object_id_type poi){
    auto node_it = geo_box.data.nodes.find(poi);
    if(node_it == geo_box.data.nodes.end()) return 0;
    
    int total_reward = 0;
    for(const int& group_id : node_it->second.groupes){
        auto char_it = parent->characteristics.find(group_id);
        if(char_it != parent->characteristics.end()){
            total_reward += static_cast<int>(char_it->second);
        }
    }
    return total_reward;
}

// ============================================================================
// GREEDY CONSTRUCTION
// ============================================================================
Solution Agent::greedy_construction(Solution& start_solution){
    if(start_solution.POIs.empty()) return start_solution;
    
    Solution current = start_solution;
    std::unordered_set<osmium::object_id_type> visited_nodes;
    
    for(const auto& poi : current.POIs){
        visited_nodes.insert(poi);
    }

    double current_fitness = parent->objective_function(current);
    
    int max_pois = 50;
    int consecutive_failures = 0;
    
    std::cout << "  [Agent] Starting greedy from POI " << current.POIs[0] << std::endl;

    while(current.cost < memory.length_constraint && current.POIs.size() < max_pois){
        if(current.POIs.empty()) break;
        
        if(consecutive_failures >= 10){
            std::cout << "  [Agent] Greedy: 10 consecutive failures, stopping" << std::endl;
            break;
        }
        
        osmium::object_id_type last_poi = current.POIs.back();
        
        osmium::object_id_type nearest_neighbor = memory.check_neighborhood(
            last_poi, 
            this->characteristics, 
            visited_nodes
        );
        
        if(nearest_neighbor == 0){
            consecutive_failures++;
            break;
        }
        
        consecutive_failures = 0;
        
        std::vector<Path> paths = memory.check_path(last_poi, nearest_neighbor);
        if(paths.empty()){
            visited_nodes.insert(nearest_neighbor);
            consecutive_failures++;
            continue;
        }
        
        Path path_to_neighbor = paths[0];
        
        if(current.cost + path_to_neighbor.cost >= memory.length_constraint){
            visited_nodes.insert(nearest_neighbor);
            consecutive_failures++;
            continue;
        }
        
        Solution test_solution = current;
        test_solution.add_node(nearest_neighbor, path_to_neighbor);
        
        if(!evaluate_upper_bound(test_solution)){
            visited_nodes.insert(nearest_neighbor);
            consecutive_failures++;
            continue;
        }
        
        // Calcul incrémental de fitness
        auto node_it = geo_box.data.nodes.find(nearest_neighbor);
        double added_fitness = 0.0;
        if(node_it != geo_box.data.nodes.end()){
            for(const int& group_id : node_it->second.groupes){
                auto char_it = parent->characteristics.find(group_id);
                if(char_it != parent->characteristics.end()){
                    added_fitness += char_it->second;
                }
            }
        }
        
        current = test_solution;
        current_fitness += added_fitness;
        visited_nodes.insert(nearest_neighbor);
        
        if(current.POIs.size() % 5 == 0){
            std::cout << "  [Agent] Greedy: " << current.POIs.size() << " POIs, cost: " 
                      << current.cost << "/" << memory.length_constraint << std::endl;
        }
        
        visited_POIs++;
        if(path_to_neighbor.cost > 0){
            float reward_density = static_cast<float>(added_fitness) / path_to_neighbor.cost;
            max_reward_per_meter = (max_reward_per_meter * (visited_POIs - 1) + reward_density) / visited_POIs;
        }
    }

    return current;
}

// ============================================================================
// GREEDY EXTEND (avec POI interdit)
// ============================================================================
Solution Agent::greedy_extend_from(
    const Solution& base,
    osmium::object_id_type forbidden_first
){
    Solution current = base;
    std::unordered_set<osmium::object_id_type> visited(
        current.POIs.begin(), 
        current.POIs.end()
    );
    
    bool first_addition = true;
    int max_additions = 20;
    int additions = 0;
    
    while(current.cost < memory.length_constraint && additions < max_additions){
        if(current.POIs.empty()) break;
        
        osmium::object_id_type last = current.POIs.back();
        
        osmium::object_id_type best_neighbor = 0;
        double best_ratio = -1.0;
        Path best_path;
        
        // Chercher dans le voisinage
        osmium::object_id_type candidate = memory.check_neighborhood(last, characteristics, visited);
        
        if(candidate == 0) break;
        
        // Si premier ajout, interdire forbidden
        if(first_addition && candidate == forbidden_first){
            visited.insert(candidate);
            continue;
        }
        
        std::vector<Path> paths = memory.check_path(last, candidate);
        if(paths.empty()) break;
        
        Path to_candidate = paths[0];
        
        if(current.cost + to_candidate.cost >= memory.length_constraint){
            visited.insert(candidate);
            continue;
        }
        
        int reward = get_poi_reward(candidate);
        if(reward <= 0){
            visited.insert(candidate);
            continue;
        }
        
        current.add_node(candidate, to_candidate);
        visited.insert(candidate);
        first_addition = false;
        additions++;
    }
    
    return current;
}

// ============================================================================
// DÉCOMPOSITION avec POI interdit
// ============================================================================

std::vector<DecomposedSolution> Agent::decompose_with_forbidden(const Solution& solution){
    std::vector<DecomposedSolution> decompositions;
    
    if(solution.POIs.size() <= 1) return decompositions;
    
    Solution current = solution;
    
    while(current.POIs.size() > 1){
        osmium::object_id_type last = current.POIs.back();
        
        DecomposedSolution decomp;
        decomp.base = current;
        decomp.forbidden_next = last;
        
        decompositions.push_back(decomp);
        
        current.remove_node();
    }
    
    return decompositions;
}

// ============================================================================
// MULTI-BRANCH EXPLORATION
// ============================================================================
std::vector<Solution> Agent::explore_multibranch(
    const Solution& base_solution,
    osmium::object_id_type forbidden_first,
    double threshold
){
    std::vector<Solution> promising_branches;
    
    if(base_solution.POIs.empty()) return promising_branches;
    
    osmium::object_id_type last_poi = base_solution.POIs.back();
    std::unordered_set<osmium::object_id_type> visited(
        base_solution.POIs.begin(), 
        base_solution.POIs.end()
    );
    
    int max_branches = 5;
    int pruned_neighbors = 0;
    int explored = 0;
    
    // Explorer plusieurs voisins
    for(int branch = 0; branch < max_branches && explored < 3; branch++){
        
        osmium::object_id_type neighbor = memory.check_neighborhood(last_poi, characteristics, visited);
        
        if(neighbor == 0) break;
        
        // Skip le POI interdit
        if(neighbor == forbidden_first){
            visited.insert(neighbor);
            continue;
        }
        
        // PRUNING: Upper bound par voisin
        std::vector<Path> paths = memory.check_path(last_poi, neighbor);
        if(paths.empty()){
            visited.insert(neighbor);
            continue;
        }
        
        Path to_neighbor = paths[0];
        
        if(base_solution.cost + to_neighbor.cost >= memory.length_constraint){
            visited.insert(neighbor);
            continue;
        }
        
        Solution temp = base_solution;
        temp.add_node(neighbor, to_neighbor);
        
        double neighbor_upper_bound = estimate_upper_bound(temp);
        
        if(neighbor_upper_bound < threshold){
            pruned_neighbors++;
            visited.insert(neighbor);
            continue;
        }
        
        // Greedy rebuild depuis ce voisin
        Solution rebuilt = greedy_extend_from(temp, forbidden_first);
        promising_branches.push_back(rebuilt);
        explored++;
        
        visited.insert(neighbor);
    }
    
    std::cout << "    [MULTI-BRANCH] explored=" << explored
              << " pruned=" << pruned_neighbors << std::endl;
    
    return promising_branches;
}

// ============================================================================
// VND LOCAL SEARCH (simple swap)
// ============================================================================
Solution Agent::vnd_local_search(const Solution& solution){
    if(solution.POIs.size() < 3) return solution;
    
    Solution best = solution;
    double best_fitness = parent->objective_function(best);
    
    bool improved = true;
    int iterations = 0;
    
    while(improved && iterations < 5){
        improved = false;
        iterations++;
        
        // Essayer des swaps sur les 3 derniers POIs
        size_t start = std::max(1, static_cast<int>(best.POIs.size()) - 3);
        
        for(size_t i = start; i < best.POIs.size() - 1; i++){
            for(size_t j = i + 1; j < best.POIs.size(); j++){
                Solution neighbor = best;
                std::swap(neighbor.POIs[i], neighbor.POIs[j]);
                
                Solution rebuilt = memory.check_solution(neighbor.POIs);
                if(rebuilt.cost >= memory.length_constraint) continue;
                
                double fitness = parent->objective_function(rebuilt);
                
                if(fitness > best_fitness){
                    best = rebuilt;
                    best_fitness = fitness;
                    improved = true;
                }
            }
        }
    }
    
    return best;
}

// ============================================================================
// AGENT SEARCH - VNS avec Multi-Branch et Décomposition
// ============================================================================
Solution Agent::agent_search(int max_iterations, int tabu_list_size) {
    
    // 1. Construction initiale Greedy
    Solution current = greedy_construction(initial_solution);
    double initial_fitness = parent->objective_function(current);
    
    std::cout << "  [Agent] Greedy initial: " << current.POIs.size() 
              << " POIs, fitness: " << initial_fitness << std::endl;
    
    // Early stopping
    double meta_threshold = parent->min_fitness;
    if(initial_fitness < meta_threshold * 0.5){
        std::cout << "  [Agent] Initial solution too weak, DISCARD" << std::endl;
        this->pbest = current;
        this->pbest_fitness = initial_fitness;
        return current;
    }
    
    // 2. VNS avec Multi-Branch
    Solution best = current;
    double best_fitness = initial_fitness;
    
    int k = 1;
    const int k_max = std::min(5, static_cast<int>(current.POIs.size()) / 2);
    int iter = 0;
    
    this->pbest = best;
    this->pbest_fitness = best_fitness;
    
    while(iter < max_iterations && k <= k_max){
        
        // A. SHAKE: Retirer k POIs
        Solution shaken = best;
        for(int i = 0; i < k && shaken.POIs.size() > 1; i++){
            shaken.remove_node();
        }
        
        // B. DÉCOMPOSITION
        std::vector<DecomposedSolution> decomposed = decompose_with_forbidden(shaken);
        
        // C. POUR CHAQUE DÉCOMPOSITION
        Solution best_from_decompositions;
        double best_decomp_fitness = -1.0;
        int pruned_decompositions = 0;
        
        for(const auto& decomp : decomposed){
            
            // PRUNING NIVEAU 1: Upper bound décomposition
            double decomp_upper_bound = estimate_upper_bound(decomp.base);
            double local_threshold = best_fitness * 0.7;
            
            if(decomp_upper_bound < local_threshold){
                pruned_decompositions++;
                continue;
            }
            
            // D. MULTI-BRANCH depuis cette décomposition
            std::vector<Solution> branches = explore_multibranch(
                decomp.base, 
                decomp.forbidden_next,
                local_threshold
            );
            
            if(branches.empty()) continue;
            
            // E. LOCAL SEARCH sur meilleure branche
            Solution best_branch;
            double best_branch_fitness = -1.0;
            
            for(const auto& branch : branches){
                double fitness = parent->objective_function(branch);
                if(fitness > best_branch_fitness){
                    best_branch_fitness = fitness;
                    best_branch = branch;
                }
            }
            
            Solution improved = vnd_local_search(best_branch);
            double improved_fitness = parent->objective_function(improved);
            
            if(improved_fitness > best_decomp_fitness){
                best_decomp_fitness = improved_fitness;
                best_from_decompositions = improved;
            }
        }
        
        std::cout << "  [VNS] k=" << k 
                  << " decompositions=" << decomposed.size()
                  << " pruned=" << pruned_decompositions << std::endl;
        
        // F. MOVE OR NOT
        if(best_decomp_fitness > best_fitness){
            best = best_from_decompositions;
            best_fitness = best_decomp_fitness;
            current = best;
            k = 1;
            
            std::cout << "  [VNS] ✓ Amélioration fitness=" << static_cast<int>(best_fitness) << std::endl;
            
            this->pbest = best;
            this->pbest_fitness = best_fitness;
        } else {
            k++;
        }
        
        iter++;
    }
    
    // G. CHECK META-AGENT THRESHOLD
    if(best_fitness < meta_threshold){
        std::cout << "  [Agent] Final solution below threshold, DISCARD" << std::endl;
    }
    
    return this->pbest;
}