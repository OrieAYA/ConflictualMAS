#include "PSO.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cmath>

// Constructeur
MTTDS_PSOSolver::MTTDS_PSOSolver(GeoBox& box, Pathfinder& pf) 
    : geo_box(box), pathfinder(pf), global_best_fitness(0.0), global_best_distance(0.0) {
    rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
}

// Méthode principale de résolution
MTTDSPSOResult MTTDS_PSOSolver::solve(
    const std::vector<int>& agent_characteristics,
    double distance_constraint,
    const MTTDSPSOParams& params) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MTTDSPSOResult result;
    characteristics = agent_characteristics;
    max_distance_constraint = distance_constraint;
    
    std::cout << "\n=== PSO MTTDS ===" << std::endl;
    
    // 1. Collecter les POIs récompensables
    collect_rewardable_pois(agent_characteristics);
    
    if (all_pois.size() < 2) {
        std::cout << "Pas assez de POIs récompensables (minimum 2)" << std::endl;
        result.is_valid = false;
        return result;
    }
    
    std::cout << "POIs récompensables: " << all_pois.size() << std::endl;
    std::cout << "Contrainte de distance: " << distance_constraint << "m" << std::endl;
    
    // 2. Construire le cache des distances
    std::cout << "Construction du cache des distances..." << std::endl;
    build_distance_cache();
    
    // 3. Initialiser l'essaim
    std::cout << "Initialisation de l'essaim (" << params.num_particles << " particules)..." << std::endl;
    initialize_swarm(params);
    
    // 4. Algorithme PSO principal
    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        
        // Mettre à jour chaque particule
        for (auto& particle : swarm) {
            update_particle(particle, params);
            
            // Mise à jour du meilleur personnel
            if (particle.fitness > particle.best_fitness) {
                particle.best_fitness = particle.fitness;
                particle.best_position = particle.position;
            }
            
            // Mise à jour du meilleur global
            if (particle.fitness > global_best_fitness) {
                global_best_fitness = particle.fitness;
                global_best_position = particle.position;
                global_best_distance = particle.distance;
            }
        }
        
        // Recherche locale sur la meilleure particule
        if (params.use_local_search && iteration % 10 == 0) {
            std::vector<osmium::object_id_type> improved = local_search(global_best_position);
            double improved_fitness = calculate_fitness(improved);
            double improved_distance = calculate_distance(improved);
            
            if (improved_fitness > global_best_fitness && improved_distance <= max_distance_constraint) {
                global_best_fitness = improved_fitness;
                global_best_position = improved;
                global_best_distance = improved_distance;
            }
        }
        
        // Mutation pour diversification
        if (iteration % 20 == 0) {
            for (auto& particle : swarm) {
                if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < params.mutation_rate) {
                    particle.position = mutate_solution(particle.position, 0.3);
                    particle.fitness = calculate_fitness(particle.position);
                    particle.distance = calculate_distance(particle.position);
                }
            }
        }
        
        // Affichage du progrès
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
    
    // Remplir le résultat
    result.solution = global_best_position;
    result.fitness = global_best_fitness;
    result.distance = global_best_distance;
    result.num_pois = global_best_position.size();
    result.execution_time_ms = duration.count();
    result.is_valid = !global_best_position.empty() && global_best_distance <= max_distance_constraint;
    
    std::cout << "\n=== RÉSULTAT PSO ===" << std::endl;
    std::cout << "Fitness finale: " << static_cast<int>(result.fitness) << std::endl;
    std::cout << "POIs visités: " << result.num_pois << std::endl;
    std::cout << "Distance totale: " << static_cast<int>(result.distance) << "m" << std::endl;
    std::cout << "Temps d'exécution: " << result.execution_time_ms << " ms" << std::endl;
    
    return result;
}

// Collecter les POIs récompensables
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

// Construire le cache des distances
void MTTDS_PSOSolver::build_distance_cache() {
    distance_cache.clear();
    path_cache.clear();
    
    for (size_t i = 0; i < all_pois.size(); ++i) {
        for (size_t j = i + 1; j < all_pois.size(); ++j) {
            osmium::object_id_type node1 = all_pois[i];
            osmium::object_id_type node2 = all_pois[j];
            
            std::vector<osmium::object_id_type> path = pathfinder.A_Star_Search(node1, node2);
            
            double distance = 0.0;
            if (!path.empty()) {
                for (const auto& way_id : path) {
                    auto way_it = geo_box.data.ways.find(way_id);
                    if (way_it != geo_box.data.ways.end()) {
                        distance += way_it->second.distance_meters;
                    }
                }
            } else {
                distance = std::numeric_limits<double>::max();
            }
            
            auto key = std::make_pair(std::min(node1, node2), std::max(node1, node2));
            distance_cache[key] = distance;
            path_cache[key] = path;
        }
    }
}

// Initialiser l'essaim
void MTTDS_PSOSolver::initialize_swarm(const MTTDSPSOParams& params) {
    swarm.clear();
    swarm.resize(params.num_particles);
    global_best_fitness = 0.0;
    global_best_position.clear();
    
    for (int i = 0; i < params.num_particles; ++i) {
        MTTDSParticle& particle = swarm[i];
        
        // Générer une solution initiale
        if (i == 0) {
            // Première particule: greedy
            particle.position = generate_greedy_solution();
        } else if (i < params.num_particles / 2) {
            // Moitié: greedy avec départ aléatoire
            std::uniform_int_distribution<size_t> dist(0, all_pois.size() - 1);
            particle.position = generate_greedy_solution_from(all_pois[dist(rng)]);
        } else {
            // Autre moitié: aléatoire
            particle.position = generate_random_solution();
        }
        
        // Évaluer la solution initiale
        particle.fitness = calculate_fitness(particle.position);
        particle.distance = calculate_distance(particle.position);
        particle.best_position = particle.position;
        particle.best_fitness = particle.fitness;
        
        // Mettre à jour le meilleur global
        if (particle.fitness > global_best_fitness) {
            global_best_fitness = particle.fitness;
            global_best_position = particle.position;
            global_best_distance = particle.distance;
        }
    }
}

// Générer une solution aléatoire faisable
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

// Générer une solution greedy
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

// Mettre à jour une particule
void MTTDS_PSOSolver::update_particle(MTTDSParticle& particle, const MTTDSPSOParams& params) {
    // Générer des coefficients aléatoires
    double r1 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double r2 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    
    // Appliquer les mouvements vers pbest et gbest
    std::vector<osmium::object_id_type> new_position = apply_moves(
        particle.position,
        params.c1 * r1,
        params.c2 * r2
    );
    
    // Appliquer l'inertie (garder une partie de la solution actuelle)
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) > params.w) {
        // Avec probabilité (1-w), appliquer les changements
        particle.position = new_position;
    }
    
    // Réparer si nécessaire
    if (!is_feasible(particle.position)) {
        particle.position = repair_solution(particle.position);
    }
    
    // Évaluer la nouvelle position
    particle.fitness = calculate_fitness(particle.position);
    particle.distance = calculate_distance(particle.position);
}

// Appliquer des mouvements vers pbest et gbest
std::vector<osmium::object_id_type> MTTDS_PSOSolver::apply_moves(
    const std::vector<osmium::object_id_type>& position,
    double cognitive_weight,
    double social_weight) {
    
    std::vector<osmium::object_id_type> result = position;
    
    // Mouvement vers pbest (cognitif)
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < cognitive_weight) {
        result = add_poi_operator(result);
    }
    
    // Mouvement vers gbest (social)
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < social_weight) {
        // Ajouter des POIs du gbest qui ne sont pas dans la solution
        for (const auto& poi : global_best_position) {
            if (std::find(result.begin(), result.end(), poi) == result.end()) {
                std::vector<osmium::object_id_type> temp = result;
                temp.push_back(poi);
                if (calculate_distance(temp) <= max_distance_constraint) {
                    result = temp;
                    break;
                }
            }
        }
    }
    
    return result;
}

// Opérateur: ajouter un POI
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

// Opérateur: retirer un POI
std::vector<osmium::object_id_type> MTTDS_PSOSolver::remove_poi_operator(
    const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.size() <= 2) return solution;
    
    std::uniform_int_distribution<size_t> dist(0, solution.size() - 1);
    size_t pos = dist(rng);
    
    std::vector<osmium::object_id_type> result = solution;
    result.erase(result.begin() + pos);
    
    return result;
}

// Opérateur: échanger deux POIs
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

// Recherche locale
std::vector<osmium::object_id_type> MTTDS_PSOSolver::local_search(
    const std::vector<osmium::object_id_type>& solution) {
    
    std::vector<osmium::object_id_type> best = solution;
    double best_fitness = calculate_fitness(solution);
    bool improved = true;
    
    while (improved) {
        improved = false;
        
        // Essayer d'ajouter des POIs
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
        
        // Essayer de retirer des POIs
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

// Mutation
std::vector<osmium::object_id_type> MTTDS_PSOSolver::mutate_solution(
    const std::vector<osmium::object_id_type>& solution,
    double mutation_rate) {
    
    std::vector<osmium::object_id_type> mutated = solution;
    
    // Avec une certaine probabilité, ajouter un POI
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < mutation_rate) {
        mutated = add_poi_operator(mutated);
    }
    
    // Avec une certaine probabilité, retirer un POI
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < mutation_rate) {
        mutated = remove_poi_operator(mutated);
    }
    
    // Avec une certaine probabilité, échanger deux POIs
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < mutation_rate) {
        mutated = swap_poi_operator(mutated);
    }
    
    return mutated;
}

// Calculer la fitness (récompense totale)
double MTTDS_PSOSolver::calculate_fitness(const std::vector<osmium::object_id_type>& solution) {
    double fitness = 0.0;
    
    for (const auto& poi : solution) {
        fitness += get_poi_reward(poi);
    }
    
    return fitness;
}

// Calculer la distance totale
double MTTDS_PSOSolver::calculate_distance(const std::vector<osmium::object_id_type>& solution) {
    if (solution.size() < 2) return 0.0;
    
    double total_distance = 0.0;
    
    for (size_t i = 0; i < solution.size() - 1; ++i) {
        total_distance += get_distance(solution[i], solution[i + 1]);
    }
    
    return total_distance;
}

// Vérifier la faisabilité
bool MTTDS_PSOSolver::is_feasible(const std::vector<osmium::object_id_type>& solution) {
    return calculate_distance(solution) <= max_distance_constraint;
}

// Obtenir la distance entre deux POIs
double MTTDS_PSOSolver::get_distance(osmium::object_id_type node1, osmium::object_id_type node2) {
    auto key = std::make_pair(std::min(node1, node2), std::max(node1, node2));
    auto it = distance_cache.find(key);
    return (it != distance_cache.end()) ? it->second : std::numeric_limits<double>::max();
}

// Obtenir le chemin entre deux POIs
std::vector<osmium::object_id_type> MTTDS_PSOSolver::get_path(osmium::object_id_type node1, osmium::object_id_type node2) {
    auto key = std::make_pair(std::min(node1, node2), std::max(node1, node2));
    auto it = path_cache.find(key);
    return (it != path_cache.end()) ? it->second : std::vector<osmium::object_id_type>{};
}

// Obtenir la récompense d'un POI
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

// Réparer une solution infaisable
std::vector<osmium::object_id_type> MTTDS_PSOSolver::repair_solution(
    const std::vector<osmium::object_id_type>& solution) {
    
    std::vector<osmium::object_id_type> repaired = solution;
    
    // Retirer des POIs jusqu'à ce que la solution soit faisable
    while (!repaired.empty() && calculate_distance(repaired) > max_distance_constraint) {
        // Retirer le POI avec le plus faible rapport reward/distance
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