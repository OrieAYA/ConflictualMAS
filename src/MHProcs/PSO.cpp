#include "PSO.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>

// Constructeur
PSOSolver::PSOSolver(GeoBox& box) : geo_box(box), global_best_fitness(std::numeric_limits<double>::max()) {
    // Initialiser le générateur de nombres aléatoires
    rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
}

// Méthode principale PSO pour un groupe
bool PSOSolver::solve_single_group(
    const std::vector<osmium::object_id_type>& objective_nodes,
    int group_id,
    const PSOParams& params) {

    if (objective_nodes.size() < 2) {
        std::cout << "PSO Groupe " << group_id << ": Pas assez de POI (minimum 2)" << std::endl;
        return false;
    }

    std::cout << "\n=== PSO GROUPE " << group_id << " ===" << std::endl;
    std::cout << "POI à optimiser: " << objective_nodes.size() << std::endl;

    // 1. Construire le cache des distances
    std::cout << "Construction du cache des distances..." << std::endl;
    build_distance_cache(objective_nodes);

    // 2. Initialiser l'essaim
    std::cout << "Initialisation de l'essaim (" << params.num_particles << " particules)..." << std::endl;
    initialize_swarm(objective_nodes, params);

    // 3. Algorithme PSO principal
    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        
        // Mettre à jour chaque particule
        for (auto& particle : swarm) {
            
            // Mettre à jour la vitesse
            update_particle_velocity(particle, params);
            
            // Mettre à jour la position
            update_particle_position(particle, objective_nodes);
            
            // Évaluer la nouvelle position
            particle.fitness = evaluate_fitness(particle.position);
            
            // Mise à jour du meilleur personnel
            if (particle.fitness < particle.best_fitness) {
                particle.best_fitness = particle.fitness;
                particle.best_position = particle.position;
            }
            
            // Mise à jour du meilleur global
            if (particle.fitness < global_best_fitness) {
                global_best_fitness = particle.fitness;
                global_best_position = particle.position;
            }
        }
        
        // Recherche locale sur la meilleure particule (optionnel)
        if (params.use_local_search && iteration % 10 == 0) {
            std::vector<osmium::object_id_type> improved = local_search_2opt(global_best_position);
            double improved_fitness = evaluate_fitness(improved);
            if (improved_fitness < global_best_fitness) {
                global_best_fitness = improved_fitness;
                global_best_position = improved;
            }
        }
        
        // Mutation pour diversification
        if (iteration % 20 == 0) {
            for (auto& particle : swarm) {
                if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < params.mutation_rate) {
                    particle.position = mutate_tour(particle.position, 0.1);
                    particle.fitness = evaluate_fitness(particle.position);
                }
            }
        }
        
        // Affichage du progrès
        if (iteration % 30 == 0 || iteration == params.max_iterations - 1) {
            std::cout << "Itération " << iteration << " - Meilleure distance: " 
                      << static_cast<int>(global_best_fitness) << "m" << std::endl;
        }
    }

    // 4. Vérifier et appliquer la solution
    if (global_best_position.empty() || !is_valid_tour(global_best_position, objective_nodes)) {
        std::cout << "PSO Groupe " << group_id << ": Aucune solution valide trouvée" << std::endl;
        return false;
    }

    std::cout << "Solution PSO trouvée - Distance totale: " << static_cast<int>(global_best_fitness) << "m" << std::endl;
    
    // Appliquer le tour optimal aux ways
    apply_tour_to_ways(global_best_position, group_id);

    return true;
}

// Initialisation de l'essaim
void PSOSolver::initialize_swarm(
    const std::vector<osmium::object_id_type>& nodes,
    const PSOParams& params) {

    swarm.clear();
    swarm.resize(params.num_particles);
    global_best_fitness = std::numeric_limits<double>::max();
    global_best_position.clear();

    for (int i = 0; i < params.num_particles; ++i) {
        Particle& particle = swarm[i];
        
        // Générer une position initiale
        if (i == 0) {
            // Première particule: heuristique du plus proche voisin
            particle.position = nearest_neighbor_tour(nodes, nodes[0]);
        } else if (i < params.num_particles / 2) {
            // Moitié des particules: plus proche voisin avec départ différent
            std::uniform_int_distribution<int> dist(0, nodes.size() - 1);
            particle.position = nearest_neighbor_tour(nodes, nodes[dist(rng)]);
        } else {
            // Autre moitié: complètement aléatoire
            particle.position = generate_random_tour(nodes);
        }
        
        // Évaluer la position initiale
        particle.fitness = evaluate_fitness(particle.position);
        particle.best_position = particle.position;
        particle.best_fitness = particle.fitness;
        
        // Mettre à jour le meilleur global
        if (particle.fitness < global_best_fitness) {
            global_best_fitness = particle.fitness;
            global_best_position = particle.position;
        }
        
        // Initialiser la vitesse (vide au début)
        particle.velocity.clear();
    }
}

// Génération de tours initiaux
std::vector<osmium::object_id_type> PSOSolver::generate_random_tour(
    const std::vector<osmium::object_id_type>& nodes) {
    
    std::vector<osmium::object_id_type> tour = nodes;
    std::shuffle(tour.begin(), tour.end(), rng);
    return tour;
}

std::vector<osmium::object_id_type> PSOSolver::nearest_neighbor_tour(
    const std::vector<osmium::object_id_type>& nodes,
    osmium::object_id_type start_node) {
    
    std::vector<osmium::object_id_type> tour;
    std::unordered_set<osmium::object_id_type> unvisited(nodes.begin(), nodes.end());
    
    osmium::object_id_type current = start_node;
    tour.push_back(current);
    unvisited.erase(current);
    
    while (!unvisited.empty()) {
        osmium::object_id_type nearest = 0;
        double min_distance = std::numeric_limits<double>::max();
        
        for (const auto& candidate : unvisited) {
            double distance = get_distance(current, candidate);
            if (distance < min_distance) {
                min_distance = distance;
                nearest = candidate;
            }
        }
        
        if (nearest != 0) {
            tour.push_back(nearest);
            unvisited.erase(nearest);
            current = nearest;
        } else {
            break;
        }
    }
    
    return tour;
}

// Mise à jour de la vitesse des particules
void PSOSolver::update_particle_velocity(
    Particle& particle,
    const PSOParams& params) {

    // Calculer les séquences de swaps vers le meilleur personnel et global
    std::vector<osmium::object_id_type> cognitive_component = 
        compute_swap_sequence(particle.position, particle.best_position);
    
    std::vector<osmium::object_id_type> social_component = 
        compute_swap_sequence(particle.position, global_best_position);
    
    // Générer des coefficients aléatoires
    double r1 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double r2 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    
    // Combiner les composantes pour former la nouvelle vitesse
    particle.velocity = combine_swap_sequences(
        cognitive_component, 
        social_component,
        params.c1 * r1,
        params.c2 * r2
    );
    
    // Appliquer l'inertie (réduire la taille de la vitesse)
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) > params.w) {
        if (!particle.velocity.empty()) {
            int reduction = particle.velocity.size() / 4;
            if (reduction > 0) {
                particle.velocity.erase(particle.velocity.end() - reduction, particle.velocity.end());
            }
        }
    }
}

// Mise à jour de la position des particules
void PSOSolver::update_particle_position(
    Particle& particle,
    const std::vector<osmium::object_id_type>& nodes) {
    
    // Appliquer la vitesse à la position
    particle.position = apply_swap_sequence(particle.position, particle.velocity);
    
    // Vérifier la validité du tour et corriger si nécessaire
    if (!is_valid_tour(particle.position, nodes)) {
        // Si le tour n'est pas valide, générer un nouveau tour aléatoire
        particle.position = generate_random_tour(nodes);
    }
}

// Opérateurs PSO adaptés au TSP
std::vector<osmium::object_id_type> PSOSolver::apply_swap_sequence(
    const std::vector<osmium::object_id_type>& tour,
    const std::vector<osmium::object_id_type>& swaps) {
    
    std::vector<osmium::object_id_type> result = tour;
    
    // Appliquer une partie aléatoire des swaps (pour l'aspect stochastique)
    int num_swaps = std::min(static_cast<int>(swaps.size()), static_cast<int>(tour.size() / 2));
    
    for (int i = 0; i < num_swaps; i += 2) {
        if (i + 1 < swaps.size()) {
            // Trouver les positions des éléments à échanger
            auto it1 = std::find(result.begin(), result.end(), swaps[i]);
            auto it2 = std::find(result.begin(), result.end(), swaps[i + 1]);
            
            if (it1 != result.end() && it2 != result.end()) {
                std::swap(*it1, *it2);
            }
        }
    }
    
    return result;
}

std::vector<osmium::object_id_type> PSOSolver::compute_swap_sequence(
    const std::vector<osmium::object_id_type>& from_tour,
    const std::vector<osmium::object_id_type>& to_tour) {
    
    std::vector<osmium::object_id_type> swaps;
    
    if (from_tour.size() != to_tour.size()) return swaps;
    
    std::vector<osmium::object_id_type> temp_tour = from_tour;
    
    for (size_t i = 0; i < to_tour.size(); ++i) {
        if (temp_tour[i] != to_tour[i]) {
            // Trouver où est l'élément voulu
            auto it = std::find(temp_tour.begin() + i + 1, temp_tour.end(), to_tour[i]);
            if (it != temp_tour.end()) {
                // Enregistrer le swap
                swaps.push_back(temp_tour[i]);
                swaps.push_back(to_tour[i]);
                
                // Effectuer le swap
                std::swap(temp_tour[i], *it);
            }
        }
    }
    
    return swaps;
}

std::vector<osmium::object_id_type> PSOSolver::combine_swap_sequences(
    const std::vector<osmium::object_id_type>& seq1,
    const std::vector<osmium::object_id_type>& seq2,
    double weight1,
    double weight2) {
    
    std::vector<osmium::object_id_type> combined;
    
    // Ajouter des éléments de seq1 selon weight1
    int num_from_seq1 = static_cast<int>(seq1.size() * std::min(weight1, 1.0));
    for (int i = 0; i < num_from_seq1; ++i) {
        combined.push_back(seq1[i]);
    }
    
    // Ajouter des éléments de seq2 selon weight2
    int num_from_seq2 = static_cast<int>(seq2.size() * std::min(weight2, 1.0));
    for (int i = 0; i < num_from_seq2; ++i) {
        combined.push_back(seq2[i]);
    }
    
    return combined;
}

// Amélioration locale 2-opt
std::vector<osmium::object_id_type> PSOSolver::local_search_2opt(
    const std::vector<osmium::object_id_type>& tour) {
    
    std::vector<osmium::object_id_type> best_tour = tour;
    double best_distance = calculate_tour_distance(tour);
    bool improved = true;
    
    while (improved) {
        improved = false;
        
        for (size_t i = 1; i < tour.size() - 1; ++i) {
            for (size_t j = i + 1; j < tour.size(); ++j) {
                // Créer nouveau tour en inversant le segment [i, j]
                std::vector<osmium::object_id_type> new_tour = best_tour;
                std::reverse(new_tour.begin() + i, new_tour.begin() + j + 1);
                
                double new_distance = calculate_tour_distance(new_tour);
                if (new_distance < best_distance) {
                    best_tour = new_tour;
                    best_distance = new_distance;
                    improved = true;
                }
            }
        }
    }
    
    return best_tour;
}

// Mutation pour diversification
std::vector<osmium::object_id_type> PSOSolver::mutate_tour(
    const std::vector<osmium::object_id_type>& tour,
    double mutation_rate) {
    
    std::vector<osmium::object_id_type> mutated = tour;
    
    int num_mutations = static_cast<int>(tour.size() * mutation_rate);
    
    for (int i = 0; i < num_mutations; ++i) {
        std::uniform_int_distribution<size_t> dist(0, tour.size() - 1);
        size_t pos1 = dist(rng);
        size_t pos2 = dist(rng);
        
        std::swap(mutated[pos1], mutated[pos2]);
    }
    
    return mutated;
}

// Évaluation des solutions
double PSOSolver::evaluate_fitness(const std::vector<osmium::object_id_type>& tour) {
    return calculate_tour_distance(tour);
}

// Méthodes utilitaires (similaires aux autres métaheuristiques)
void PSOSolver::build_distance_cache(const std::vector<osmium::object_id_type>& nodes) {
    distance_cache.clear();
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = i + 1; j < nodes.size(); ++j) {
            osmium::object_id_type node1 = nodes[i];
            osmium::object_id_type node2 = nodes[j];
            
            std::vector<osmium::object_id_type> path = find_shortest_path(node1, node2);
            
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
        }
    }
}

double PSOSolver::get_distance(osmium::object_id_type node1, osmium::object_id_type node2) {
    auto key = std::make_pair(std::min(node1, node2), std::max(node1, node2));
    auto it = distance_cache.find(key);
    return (it != distance_cache.end()) ? it->second : std::numeric_limits<double>::max();
}

double PSOSolver::calculate_tour_distance(const std::vector<osmium::object_id_type>& tour) {
    double total_distance = 0.0;
    
    for (size_t i = 0; i < tour.size(); ++i) {
        size_t next_i = (i + 1) % tour.size();
        total_distance += get_distance(tour[i], tour[next_i]);
    }
    
    return total_distance;
}

std::vector<osmium::object_id_type> PSOSolver::find_shortest_path(
    osmium::object_id_type start, 
    osmium::object_id_type end) {
    
    Pathfinder temp_pathfinder(geo_box);
    return temp_pathfinder.A_Star_Search(start, end);
}

void PSOSolver::apply_tour_to_ways(
    const std::vector<osmium::object_id_type>& tour,
    int group_id) {

    std::cout << "Application du tour PSO aux ways du groupe " << group_id << "..." << std::endl;
    int ways_marked = 0;

    for (size_t i = 0; i < tour.size(); ++i) {
        size_t next_i = (i + 1) % tour.size();
        osmium::object_id_type node1 = tour[i];
        osmium::object_id_type node2 = tour[next_i];

        std::vector<osmium::object_id_type> path = find_shortest_path(node1, node2);
        
        for (const auto& way_id : path) {
            update_way_group(way_id, group_id);
            ways_marked++;
        }
    }

    std::cout << "Ways marqués pour le groupe " << group_id << ": " << ways_marked << std::endl;
}

void PSOSolver::update_way_group(osmium::object_id_type way_id, int new_group) {
    auto it = geo_box.data.ways.find(way_id);
    if (it != geo_box.data.ways.end()) {
        it->second.add_group(new_group);
    } else {
        std::cerr << "Warning: Way " << way_id << " not found" << std::endl;
    }
}

bool PSOSolver::is_valid_tour(
    const std::vector<osmium::object_id_type>& tour,
    const std::vector<osmium::object_id_type>& original_nodes) {
    
    if (tour.size() != original_nodes.size()) return false;
    
    std::unordered_set<osmium::object_id_type> tour_set(tour.begin(), tour.end());
    std::unordered_set<osmium::object_id_type> original_set(original_nodes.begin(), original_nodes.end());
    
    return tour_set == original_set;
}