#include "ACO.hpp"
#include "../Pathfinding.hpp"  // Pour accéder à A_Star_Search
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdlib>

// Constructeur
ACOSolver::ACOSolver(GeoBox& box) : geo_box(box) {}

// Méthode principale ACO pour un groupe
bool ACOSolver::solve_single_group(
    const std::vector<osmium::object_id_type>& objective_nodes,
    int group_id,
    const ACOParams& params) {

    if (objective_nodes.size() < 2) {
        std::cout << "ACO Groupe " << group_id << ": Pas assez de POI (minimum 2)" << std::endl;
        return false;
    }

    std::cout << "\n=== ACO GROUPE " << group_id << " ===" << std::endl;
    std::cout << "POI à optimiser: " << objective_nodes.size() << std::endl;

    // 1. Cache des distances entre POI
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash> distance_cache;
    
    std::cout << "Calcul des distances entre POI..." << std::endl;
    for (size_t i = 0; i < objective_nodes.size(); ++i) {
        for (size_t j = i + 1; j < objective_nodes.size(); ++j) {
            osmium::object_id_type node1 = objective_nodes[i];
            osmium::object_id_type node2 = objective_nodes[j];
            
            // Utiliser le pathfinding pour calculer le chemin
            std::vector<osmium::object_id_type> path = find_shortest_path(node1, node2);
            
            double distance = 0.0;
            if (!path.empty()) {
                // Calculer la distance totale du chemin
                for (const auto& way_id : path) {
                    auto way_it = geo_box.data.ways.find(way_id);
                    if (way_it != geo_box.data.ways.end()) {
                        distance += way_it->second.distance_meters;
                    }
                }
            } else {
                // Si pas de chemin trouvé, distance infinie
                distance = std::numeric_limits<double>::max();
            }
            
            auto key = std::make_pair(std::min(node1, node2), std::max(node1, node2));
            distance_cache[key] = distance;
        }
    }

    // 2. Initialiser la matrice de phéromones
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash> pheromones;
    for (const auto& [key, distance] : distance_cache) {
        pheromones[key] = params.initial_pheromone;
    }

    // 3. Algorithme ACO principal
    std::vector<Ant> ants(params.num_ants);
    Ant best_ant;
    best_ant.tour_length = std::numeric_limits<double>::max();

    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        
        // Chaque fourmi construit une solution
        for (auto& ant : ants) {
            ant.reset();
            construct_ant_tour(ant, objective_nodes, distance_cache, pheromones, params);
        }

        // Trouver la meilleure fourmi de cette itération
        for (const auto& ant : ants) {
            if (ant.tour_length < best_ant.tour_length && ant.tour.size() == objective_nodes.size()) {
                best_ant = ant;
            }
        }

        // Mise à jour des phéromones
        update_pheromones(pheromones, ants, params);

        // Affichage du progrès
        if (iteration % 10 == 0 || iteration == params.max_iterations - 1) {
            std::cout << "Itération " << iteration << " - Meilleure distance: " 
                      << static_cast<int>(best_ant.tour_length) << "m" << std::endl;
        }
    }

    // 4. Appliquer la meilleure solution trouvée
    if (best_ant.tour.empty() || best_ant.tour.size() != objective_nodes.size()) {
        std::cout << "ACO Groupe " << group_id << ": Aucune solution valide trouvée" << std::endl;
        return false;
    }

    std::cout << "Solution trouvée - Distance totale: " << static_cast<int>(best_ant.tour_length) << "m" << std::endl;
    
    // Appliquer le tour optimal aux ways
    apply_tour_to_ways(best_ant.tour, group_id);

    return true;
}

// Construction d'un tour complet par une fourmi
void ACOSolver::construct_ant_tour(
    Ant& ant,
    const std::vector<osmium::object_id_type>& objective_nodes,
    const std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& distance_cache,
    const std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& pheromones,
    const ACOParams& params) {

    // Commencer par un POI aléatoire
    int start_idx = rand() % objective_nodes.size();
    osmium::object_id_type current_node = objective_nodes[start_idx];
    
    ant.tour.push_back(current_node);
    ant.visited.insert(current_node);

    // Visiter tous les autres POI
    while (ant.visited.size() < objective_nodes.size()) {
        osmium::object_id_type next_node = choose_next_node(
            current_node, objective_nodes, ant.visited, distance_cache, pheromones, params);
        
        if (next_node == 0) break; // Erreur - aucun nœud disponible
        
        ant.tour.push_back(next_node);
        ant.visited.insert(next_node);
        
        // Ajouter la distance au tour
        auto key = std::make_pair(std::min(current_node, next_node), std::max(current_node, next_node));
        auto dist_it = distance_cache.find(key);
        if (dist_it != distance_cache.end()) {
            ant.tour_length += dist_it->second;
        }
        
        current_node = next_node;
    }

    // Fermer le cycle (retour au point de départ)
    if (ant.tour.size() == objective_nodes.size()) {
        auto key = std::make_pair(std::min(current_node, ant.tour[0]), std::max(current_node, ant.tour[0]));
        auto dist_it = distance_cache.find(key);
        if (dist_it != distance_cache.end()) {
            ant.tour_length += dist_it->second;
        }
    }
}

// Choix du prochain POI selon les probabilités ACO
osmium::object_id_type ACOSolver::choose_next_node(
    osmium::object_id_type current_node,
    const std::vector<osmium::object_id_type>& objective_nodes,
    const std::unordered_set<osmium::object_id_type>& visited,
    const std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& distance_cache,
    const std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& pheromones,
    const ACOParams& params) {

    std::vector<std::pair<osmium::object_id_type, double>> probabilities;
    double total_probability = 0.0;

    // Calculer les probabilités pour chaque POI non visité
    for (const auto& candidate : objective_nodes) {
        if (visited.count(candidate) == 0 && candidate != current_node) {
            
            auto key = std::make_pair(std::min(current_node, candidate), std::max(current_node, candidate));
            
            auto dist_it = distance_cache.find(key);
            auto pher_it = pheromones.find(key);
            
            if (dist_it != distance_cache.end() && pher_it != pheromones.end() && 
                dist_it->second > 0 && dist_it->second < std::numeric_limits<double>::max()) {
                
                double tau = std::pow(pher_it->second, params.alpha);
                double eta = std::pow(1.0 / (dist_it->second + 1.0), params.beta);
                double probability = tau * eta;
                
                probabilities.push_back(std::make_pair(candidate, probability));
                total_probability += probability;
            }
        }
    }

    if (probabilities.empty() || total_probability <= 0.0) {
        return 0; // Erreur
    }

    // Sélection par roulette
    double random_value = ((double)rand() / RAND_MAX) * total_probability;
    double cumulative = 0.0;

    for (const auto& [node_id, probability] : probabilities) {
        cumulative += probability;
        if (cumulative >= random_value) {
            return node_id;
        }
    }

    // Fallback
    return probabilities[0].first;
}

// Mise à jour des phéromones
void ACOSolver::update_pheromones(
    std::unordered_map<std::pair<osmium::object_id_type, osmium::object_id_type>, double, PairHash>& pheromones,
    const std::vector<Ant>& ants,
    const ACOParams& params) {

    // Évaporation
    for (auto& [key, pheromone] : pheromones) {
        pheromone *= (1.0 - params.rho);
        pheromone = std::max(pheromone, 0.1);
    }

    // Dépôt par chaque fourmi
    for (const auto& ant : ants) {
        if (ant.tour.size() < 2 || ant.tour_length <= 0.0) continue;
        
        double delta = params.Q / ant.tour_length;
        
        // Pour chaque arête du tour
        for (size_t i = 0; i < ant.tour.size(); ++i) {
            size_t next_i = (i + 1) % ant.tour.size();
            auto key = std::make_pair(
                std::min(ant.tour[i], ant.tour[next_i]), 
                std::max(ant.tour[i], ant.tour[next_i])
            );
            
            auto it = pheromones.find(key);
            if (it != pheromones.end()) {
                it->second += delta;
            }
        }
    }
}

// Application du tour optimal aux ways
void ACOSolver::apply_tour_to_ways(
    const std::vector<osmium::object_id_type>& tour,
    int group_id) {

    std::cout << "Application du tour aux ways du groupe " << group_id << "..." << std::endl;
    int ways_marked = 0;

    // Pour chaque segment du tour optimal
    for (size_t i = 0; i < tour.size(); ++i) {
        size_t next_i = (i + 1) % tour.size();
        osmium::object_id_type node1 = tour[i];
        osmium::object_id_type node2 = tour[next_i];

        // Trouver le plus court chemin
        std::vector<osmium::object_id_type> path = find_shortest_path(node1, node2);
        
        // Marquer tous les ways de ce chemin avec le groupe
        for (const auto& way_id : path) {
            update_way_group(way_id, group_id);
            ways_marked++;
        }
    }

    std::cout << "Ways marqués pour le groupe " << group_id << ": " << ways_marked << std::endl;
}

// Méthodes utilitaires
std::vector<osmium::object_id_type> ACOSolver::find_shortest_path(
    osmium::object_id_type start, 
    osmium::object_id_type end) {
    
    // Créer un objet Pathfinder temporaire pour utiliser A_Star_Search
    // NOTE: Cette approche nécessite que Pathfinder soit accessible
    Pathfinder temp_pathfinder(geo_box);
    return temp_pathfinder.A_Star_Search(start, end);
}

void ACOSolver::update_way_group(osmium::object_id_type way_id, int new_group) {
    auto it = geo_box.data.ways.find(way_id);
    if (it != geo_box.data.ways.end()) {
        it->second.add_group(new_group);
    } else {
        std::cerr << "Warning: Way " << way_id << " not found" << std::endl;
    }
}