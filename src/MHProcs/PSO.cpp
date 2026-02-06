#include "PSO.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cmath>
#include <unordered_set>

MTTDS_PSOSolver::MTTDS_PSOSolver(GeoBox& box, Pathfinder& pf, GlobalMemory& mem) 
    : geo_box(box), pathfinder(pf), global_memory(mem), 
      global_best_fitness(0.0), global_best_distance(0.0) {
    rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
}

MTTDSPSOResult MTTDS_PSOSolver::solve(
    const std::vector<int>& agent_characteristics,
    double distance_constraint,
    const MTTDSPSOParams& params) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MTTDSPSOResult result;
    characteristics = agent_characteristics;
    max_distance_constraint = distance_constraint;
    
    std::cout << "\n=== PSO MTTDS ===" << std::endl;
    
    collect_rewardable_pois(agent_characteristics);
    
    if (all_pois.size() < 2) {
        std::cout << "Pas assez de POIs récompensables (minimum 2)" << std::endl;
        result.is_valid = false;
        return result;
    }
    
    std::cout << "POIs récompensables: " << all_pois.size() << std::endl;
    std::cout << "Contrainte de distance: " << distance_constraint << "m" << std::endl;
    
    size_t total_possible_paths = (all_pois.size() * (all_pois.size() - 1)) / 2;
    std::cout << "Chemins possibles: " << total_possible_paths << std::endl;
    std::cout << "Utilisation cache GlobalMemory (calcul à la demande)\n" << std::endl;
    
    std::cout << "Initialisation de l'essaim (" << params.num_particles << " particules)..." << std::endl;
    
    initialize_swarm(params);
    
    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        for (auto& particle : swarm) {
            update_particle(particle, params);
            
            if (particle.fitness > particle.best_fitness) {
                particle.best_fitness = particle.fitness;
                particle.best_position = particle.position;
            }
            
            if (particle.fitness > global_best_fitness) {
                global_best_fitness = particle.fitness;
                global_best_position = particle.position;
                global_best_distance = particle.distance;
            }
        }
        
        if (params.use_local_search && iteration % 5 == 0) {
            std::vector<osmium::object_id_type> improved = local_search(global_best_position);
            double improved_fitness = calculate_fitness(improved);
            double improved_distance = calculate_distance(improved);
            
            if (improved_fitness > global_best_fitness && improved_distance <= max_distance_constraint) {
                global_best_fitness = improved_fitness;
                global_best_position = improved;
                global_best_distance = improved_distance;
            }
        }
        
        if (iteration % 10 == 0) {
            for (auto& particle : swarm) {
                if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.4) {
                    particle.position = mutate_solution(particle.position, 0.3);
                    particle.fitness = calculate_fitness(particle.position);
                    particle.distance = calculate_distance(particle.position);
                }
            }
        }
        
        if (iteration % 20 == 0 || iteration == params.max_iterations - 1) {
            std::cout << "Itération " << iteration 
                      << " - Fitness: " << static_cast<int>(global_best_fitness)
                      << " | POIs: " << global_best_position.size()
                      << " | Distance: " << static_cast<int>(global_best_distance) << "m" 
                      << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    result.solution = global_best_position;
    result.fitness = global_best_fitness;
    result.distance = global_best_distance;
    result.num_pois = global_best_position.size();
    result.execution_time_ms = duration.count();
    result.is_valid = !global_best_position.empty() && global_best_distance <= max_distance_constraint;
    
    size_t paths_calculated = global_memory.visited_Paths.size();
    size_t total_possible = (all_pois.size() * (all_pois.size() - 1)) / 2;
    float cache_efficiency = total_possible > 0 ? (100.0f * paths_calculated / total_possible) : 0.0f;
    
    std::cout << "\n=== RÉSULTAT PSO ===" << std::endl;
    std::cout << "Fitness finale: " << static_cast<int>(result.fitness) << std::endl;
    std::cout << "POIs visités: " << result.num_pois << std::endl;
    std::cout << "Distance totale: " << static_cast<int>(result.distance) << "m" << std::endl;
    std::cout << "Temps d'exécution: " << result.execution_time_ms << " ms" << std::endl;
    std::cout << "\n=== STATISTIQUES CACHE ===" << std::endl;
    std::cout << "Chemins calculés: " << paths_calculated << " / " << total_possible 
              << " (" << std::fixed << std::setprecision(1) << cache_efficiency << "%)" << std::endl;
    
    return result;
    
    return result;
}

void MTTDS_PSOSolver::collect_rewardable_pois(const std::vector<int>& chars) {
    all_pois.clear();
    
    for (const auto& [node_id, node_data] : geo_box.data.nodes) {
        if (!node_data.groupes.empty()) {
            bool has_reward = false;
            for (const int group_id : node_data.groupes) {
                if (group_id >= 0 && group_id < static_cast<int>(chars.size())) {
                    if (chars[group_id] > 0) {
                        has_reward = true;
                        break;
                    }
                }
            }
            if (has_reward) {
                all_pois.push_back(node_id);
            }
        }
    }
}

void MTTDS_PSOSolver::initialize_swarm(const MTTDSPSOParams& params) {
    swarm.clear();
    swarm.resize(params.num_particles);
    global_best_fitness = 0.0;
    global_best_position.clear();
    
    for (int i = 0; i < params.num_particles; ++i) {
        MTTDSParticle& particle = swarm[i];
        
        if (i == 0) {
            particle.position = generate_greedy_solution();
        } else if (i < params.num_particles / 2) {
            std::uniform_int_distribution<size_t> dist(0, all_pois.size() - 1);
            particle.position = generate_greedy_solution_from(all_pois[dist(rng)]);
        } else {
            particle.position = generate_random_solution();
        }
        
        particle.fitness = calculate_fitness(particle.position);
        particle.distance = calculate_distance(particle.position);
        particle.best_position = particle.position;
        particle.best_fitness = particle.fitness;
        
        if (particle.fitness > global_best_fitness) {
            global_best_fitness = particle.fitness;
            global_best_position = particle.position;
            global_best_distance = particle.distance;
        }
    }
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::generate_random_solution() {
    std::vector<osmium::object_id_type> solution;
    std::vector<osmium::object_id_type> available = all_pois;
    std::shuffle(available.begin(), available.end(), rng);
    
    double current_distance = 0.0;
    
    for (const auto& poi : available) {
        std::vector<osmium::object_id_type> temp = solution;
        temp.push_back(poi);
        double new_distance = calculate_distance(temp);
        
        if (new_distance <= max_distance_constraint) {
            solution = temp;
            current_distance = new_distance;
        } else {
            break;
        }
    }
    
    return solution;
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::generate_greedy_solution() {
    if (all_pois.empty()) return {};
    
    std::uniform_int_distribution<size_t> dist(0, all_pois.size() - 1);
    return generate_greedy_solution_from(all_pois[dist(rng)]);
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::generate_greedy_solution_from(osmium::object_id_type start) {
    std::vector<osmium::object_id_type> solution;
    std::unordered_set<osmium::object_id_type> used;
    
    solution.push_back(start);
    used.insert(start);
    
    while (used.size() < all_pois.size()) {
        osmium::object_id_type best_next = 0;
        double best_ratio = -1.0;
        
        for (const auto& candidate : all_pois) {
            if (used.count(candidate)) continue;
            
            std::vector<osmium::object_id_type> temp = solution;
            temp.push_back(candidate);
            double new_distance = calculate_distance(temp);
            
            if (new_distance > max_distance_constraint) continue;
            
            double added_distance = new_distance - calculate_distance(solution);
            int added_reward = get_poi_reward(candidate);
            
            if (added_distance > 0) {
                double ratio = static_cast<double>(added_reward) / added_distance;
                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_next = candidate;
                }
            }
        }
        
        if (best_next == 0) break;
        
        solution.push_back(best_next);
        used.insert(best_next);
    }
    
    return solution;
}

void MTTDS_PSOSolver::update_particle(MTTDSParticle& particle, const MTTDSPSOParams& params) {
    double r1 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double r2 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

    std::vector<osmium::object_id_type> new_position = apply_moves(
        particle.position,
        params.c1 * r1,
        params.c2 * r2
    );

    // ========================================
    // ANCIEN CODE (BUGUÉ) :
    // if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) > params.w) {
    //     particle.position = new_position;
    // }
    // → 70% du temps, la particule IGNORE le mouvement !
    // ========================================
    
    // ========================================
    // NOUVEAU CODE (CORRECT) :
    // ========================================
    // w = inertie = probabilité de GARDER la vélocité (continuer dans la même direction)
    // Pour OP, on interprète w comme : probabilité d'ACCEPTER le nouveau mouvement
    
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < params.w) {
        // Accepter le mouvement avec probabilité w
        particle.position = new_position;
    } else {
        // Avec probabilité (1-w), faire un mouvement EXPLORATOIRE
        particle.position = mutate_solution(particle.position, 0.5);
    }

    // Réparer si non-faisable
    if (!is_feasible(particle.position)) {
        particle.position = repair_solution(particle.position);
    }

    // Mettre à jour fitness et distance
    particle.fitness = calculate_fitness(particle.position);
    particle.distance = calculate_distance(particle.position);
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::apply_moves(
    const std::vector<osmium::object_id_type>& position,
    double cognitive_weight,
    double social_weight) {

    std::vector<osmium::object_id_type> result = position;
    
    // ========================================
    // A. COGNITIVE COMPONENT (vers pbest)
    // ========================================
    // Cognitive = apprendre de SA PROPRE meilleure solution
    // Pour le OP, on ne peut pas vraiment avoir de pbest différent de position
    // → On utilise cognitive_weight comme probabilité de LOCAL SEARCH
    
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < cognitive_weight) {
        // Essayer d'AJOUTER un POI rentable
        result = add_poi_operator(result);
    }
    
    // ========================================
    // B. SOCIAL COMPONENT (vers gbest)
    // ========================================
    // Social = apprendre du MEILLEUR global
    
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < social_weight) {
        // NOUVEAU : Au lieu d'ajouter UN POI, on fait un CROSSOVER partiel
        result = crossover_with_gbest(result, global_best_position);
    }
    
    // ========================================
    // C. DIVERSIFICATION FORCÉE
    // ========================================
    // CRITIQUE : Forcer une modification même si cognitive + social n'ont rien fait
    
    bool modified = (result != position);
    
    if (!modified) {
        // Aucun mouvement effectué → FORCER un changement
        double rand = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        
        if (rand < 0.4) {
            // 40% : Ajouter un POI aléatoire
            result = add_random_poi(result);
        } else if (rand < 0.7) {
            // 30% : Retirer un POI
            result = remove_poi_operator(result);
        } else {
            // 30% : Swap deux POIs
            result = swap_poi_operator(result);
        }
    }

    return result;
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::crossover_with_gbest(
    const std::vector<osmium::object_id_type>& position,
    const std::vector<osmium::object_id_type>& gbest) {
    
    if (gbest.empty()) return position;
    
    std::vector<osmium::object_id_type> result = position;
    std::unordered_set<osmium::object_id_type> in_result(result.begin(), result.end());
    
    // Prendre 30-50% des POIs de gbest qui ne sont pas déjà dans position
    std::vector<osmium::object_id_type> candidates;
    for (const auto& poi : gbest) {
        if (in_result.find(poi) == in_result.end()) {
            candidates.push_back(poi);
        }
    }
    
    if (candidates.empty()) return result;
    
    // Shuffle et prendre les N premiers
    std::shuffle(candidates.begin(), candidates.end(), rng);
    int num_to_add = std::min(
        static_cast<int>(candidates.size()), 
        std::max(1, static_cast<int>(candidates.size() * 0.4))
    );
    
    for (int i = 0; i < num_to_add; i++) {
        std::vector<osmium::object_id_type> temp = result;
        temp.push_back(candidates[i]);
        
        if (calculate_distance(temp) <= max_distance_constraint) {
            result = temp;
        }
    }
    
    return result;
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::add_random_poi(
    const std::vector<osmium::object_id_type>& solution) {
    
    std::unordered_set<osmium::object_id_type> in_solution(solution.begin(), solution.end());
    std::vector<osmium::object_id_type> candidates;
    
    for (const auto& poi : all_pois) {
        if (in_solution.find(poi) == in_solution.end()) {
            candidates.push_back(poi);
        }
    }
    
    if (candidates.empty()) return solution;
    
    // Essayer d'ajouter un POI aléatoire
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    
    for (int attempts = 0; attempts < 5; attempts++) {
        osmium::object_id_type random_poi = candidates[dist(rng)];
        
        std::vector<osmium::object_id_type> temp = solution;
        temp.push_back(random_poi);
        
        if (calculate_distance(temp) <= max_distance_constraint) {
            return temp;
        }
    }
    
    return solution;
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::add_poi_operator(
    const std::vector<osmium::object_id_type>& solution) {
    
    std::vector<osmium::object_id_type> best_solution = solution;
    double best_improvement = 0.0;
    
    for (const auto& poi : all_pois) {
        if (std::find(solution.begin(), solution.end(), poi) != solution.end()) continue;
        
        std::vector<osmium::object_id_type> temp = solution;
        temp.push_back(poi);
        
        if (calculate_distance(temp) <= max_distance_constraint) {
            double improvement = calculate_fitness(temp) - calculate_fitness(solution);
            if (improvement > best_improvement) {
                best_improvement = improvement;
                best_solution = temp;
            }
        }
    }
    
    return best_solution;
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::remove_poi_operator(
    const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.size() <= 2) return solution;
    
    std::uniform_int_distribution<size_t> dist(0, solution.size() - 1);
    size_t pos = dist(rng);
    
    std::vector<osmium::object_id_type> result = solution;
    result.erase(result.begin() + pos);
    
    return result;
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::swap_poi_operator(
    const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.size() < 2) return solution;
    
    std::uniform_int_distribution<size_t> dist(0, solution.size() - 1);
    size_t pos1 = dist(rng);
    size_t pos2 = dist(rng);
    
    std::vector<osmium::object_id_type> result = solution;
    std::swap(result[pos1], result[pos2]);
    
    return result;
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::local_search(
    const std::vector<osmium::object_id_type>& solution) {
    
    std::vector<osmium::object_id_type> best = solution;
    double best_fitness = calculate_fitness(solution);
    bool improved = true;
    
    while (improved) {
        improved = false;
        
        for (const auto& poi : all_pois) {
            if (std::find(best.begin(), best.end(), poi) != best.end()) continue;
            
            std::vector<osmium::object_id_type> temp = best;
            temp.push_back(poi);
            
            if (calculate_distance(temp) <= max_distance_constraint) {
                double temp_fitness = calculate_fitness(temp);
                if (temp_fitness > best_fitness) {
                    best = temp;
                    best_fitness = temp_fitness;
                    improved = true;
                    break;
                }
            }
        }
        
        if (!improved && best.size() > 2) {
            for (size_t i = 0; i < best.size(); ++i) {
                std::vector<osmium::object_id_type> temp = best;
                temp.erase(temp.begin() + i);
                
                double temp_fitness = calculate_fitness(temp);
                if (temp_fitness > best_fitness) {
                    best = temp;
                    best_fitness = temp_fitness;
                    improved = true;
                    break;
                }
            }
        }
    }
    
    return best;
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::mutate_solution(
    const std::vector<osmium::object_id_type>& solution,
    double mutation_rate) {
    
    std::vector<osmium::object_id_type> mutated = solution;
    
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < mutation_rate) {
        mutated = add_poi_operator(mutated);
    }
    
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < mutation_rate) {
        mutated = remove_poi_operator(mutated);
    }
    
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < mutation_rate) {
        mutated = swap_poi_operator(mutated);
    }
    
    return mutated;
}

double MTTDS_PSOSolver::calculate_fitness(const std::vector<osmium::object_id_type>& solution) {
    double fitness = 0.0;
    
    for (const auto& poi : solution) {
        fitness += get_poi_reward(poi);
    }
    
    return fitness;
}

double MTTDS_PSOSolver::calculate_distance(const std::vector<osmium::object_id_type>& solution) {
    if (solution.size() < 2) return 0.0;
    
    double total_distance = 0.0;
    
    for (size_t i = 0; i < solution.size() - 1; ++i) {
        total_distance += get_distance(solution[i], solution[i + 1]);
    }
    
    return total_distance;
}

bool MTTDS_PSOSolver::is_feasible(const std::vector<osmium::object_id_type>& solution) {
    return calculate_distance(solution) <= max_distance_constraint;
}

double MTTDS_PSOSolver::get_distance(osmium::object_id_type node1, osmium::object_id_type node2) {
    Path path = global_memory.check_path(node1, node2)[0];
    return static_cast<double>(path.cost);
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::get_path(osmium::object_id_type node1, osmium::object_id_type node2) {
    Path path = global_memory.check_path(node1, node2)[0];
    return path.path_edges;
}

int MTTDS_PSOSolver::get_poi_reward(osmium::object_id_type poi_id) {
    auto node_it = geo_box.data.nodes.find(poi_id);
    if (node_it == geo_box.data.nodes.end()) return 0;
    
    int reward = 0;
    for (const int group_id : node_it->second.groupes) {
        if (group_id >= 0 && group_id < static_cast<int>(characteristics.size())) {
            reward += characteristics[group_id];
        }
    }
    
    return reward;
}

std::vector<osmium::object_id_type> MTTDS_PSOSolver::repair_solution(
    const std::vector<osmium::object_id_type>& solution) {
    
    std::vector<osmium::object_id_type> repaired = solution;
    
    while (!repaired.empty() && calculate_distance(repaired) > max_distance_constraint) {
        size_t worst_idx = 0;
        double worst_ratio = std::numeric_limits<double>::max();
        
        for (size_t i = 0; i < repaired.size(); ++i) {
            double reward = get_poi_reward(repaired[i]);
            double distance = 0.0;
            
            if (i > 0) distance += get_distance(repaired[i-1], repaired[i]);
            if (i < repaired.size() - 1) distance += get_distance(repaired[i], repaired[i+1]);
            
            if (distance > 0) {
                double ratio = reward / distance;
                if (ratio < worst_ratio) {
                    worst_ratio = ratio;
                    worst_idx = i;
                }
            }
        }
        
        repaired.erase(repaired.begin() + worst_idx);
    }
    
    return repaired;
}