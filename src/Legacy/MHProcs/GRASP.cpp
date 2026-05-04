#include "GRASP.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>

// Constructeur
GRASPSolver::GRASPSolver(GeoBox& box) : geo_box(box) {
    // Initialiser le générateur de nombres aléatoires avec l'horloge système
    rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
}

// Méthode principale GRASP pour un groupe
bool GRASPSolver::solve_single_group(
    const std::vector<osmium::object_id_type>& objective_nodes,
    int group_id,
    const GRASPParams& params) {

    if (objective_nodes.size() < 2) {
        std::cout << "GRASP Groupe " << group_id << ": Pas assez de POI (minimum 2)" << std::endl;
        return false;
    }

    std::cout << "\n=== GRASP GROUPE " << group_id << " ===" << std::endl;
    std::cout << "POI à optimiser: " << objective_nodes.size() << std::endl;

    // 1. Construire le cache des distances
    std::cout << "Construction du cache des distances..." << std::endl;
    build_distance_cache(objective_nodes);

    // 2. GRASP principal
    GRASPSolution best_solution;
    best_solution.total_distance = std::numeric_limits<double>::max();

    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        
        // Phase de construction greedy randomisée
        GRASPSolution current_solution = greedy_randomized_construction(objective_nodes, params);
        
        // Phase de recherche locale
        if (current_solution.is_valid) {
            current_solution = local_search(current_solution, params);
        }
        
        // Mettre à jour la meilleure solution
        if (current_solution.is_valid && current_solution.total_distance < best_solution.total_distance) {
            best_solution = current_solution;
        }
        
        // Affichage du progrès
        if (iteration % 20 == 0 || iteration == params.max_iterations - 1) {
            std::cout << "Itération " << iteration << " - Meilleure distance: " 
                      << static_cast<int>(best_solution.total_distance) << "m" << std::endl;
        }
    }

    // 3. Appliquer la meilleure solution trouvée
    if (!best_solution.is_valid) {
        std::cout << "GRASP Groupe " << group_id << ": Aucune solution valide trouvée" << std::endl;
        return false;
    }

    std::cout << "Solution GRASP trouvée - Distance totale: " << static_cast<int>(best_solution.total_distance) << "m" << std::endl;
    
    // Appliquer le tour optimal aux ways
    apply_tour_to_ways(best_solution.tour, group_id);

    return true;
}

// Phase de construction greedy randomisée
GRASPSolution GRASPSolver::greedy_randomized_construction(
    const std::vector<osmium::object_id_type>& objective_nodes,
    const GRASPParams& params) {

    GRASPSolution solution;
    
    // Commencer par un POI aléatoire
    std::uniform_int_distribution<int> dist(0, objective_nodes.size() - 1);
    osmium::object_id_type current_node = objective_nodes[dist(rng)];
    
    solution.tour.push_back(current_node);
    std::unordered_set<osmium::object_id_type> unvisited(objective_nodes.begin(), objective_nodes.end());
    unvisited.erase(current_node);
    
    // Construire le tour en utilisant la Liste Restreinte de Candidats (RCL)
    while (!unvisited.empty()) {
        
        // Construire la RCL
        std::vector<osmium::object_id_type> rcl = build_restricted_candidate_list(
            current_node, unvisited, params.alpha);
        
        if (rcl.empty()) break;
        
        // Sélectionner aléatoirement dans la RCL
        std::uniform_int_distribution<int> rcl_dist(0, rcl.size() - 1);
        osmium::object_id_type next_node = rcl[rcl_dist(rng)];
        
        solution.tour.push_back(next_node);
        unvisited.erase(next_node);
        current_node = next_node;
    }
    
    // Calculer la distance totale du tour (incluant le retour)
    if (solution.tour.size() == objective_nodes.size()) {
        solution.total_distance = calculate_tour_distance(solution.tour);
        solution.is_valid = true;
    }
    
    return solution;
}

// Phase de recherche locale
GRASPSolution GRASPSolver::local_search(
    const GRASPSolution& initial_solution,
    const GRASPParams& params) {

    GRASPSolution current_solution = initial_solution;
    
    for (int iter = 0; iter < params.local_search_iterations; ++iter) {
        bool improved = false;
        
        // Amélioration 2-opt
        if (params.use_2opt) {
            GRASPSolution improved_solution = two_opt_improvement(current_solution);
            if (improved_solution.total_distance < current_solution.total_distance) {
                current_solution = improved_solution;
                improved = true;
            }
        }
        
        // Amélioration 3-opt (plus coûteuse)
        if (params.use_3opt && !improved) {
            GRASPSolution improved_solution = three_opt_improvement(current_solution);
            if (improved_solution.total_distance < current_solution.total_distance) {
                current_solution = improved_solution;
                improved = true;
            }
        }
        
        // Si aucune amélioration, arrêter la recherche locale
        if (!improved) {
            break;
        }
    }
    
    return current_solution;
}

// Amélioration 2-opt
GRASPSolution GRASPSolver::two_opt_improvement(const GRASPSolution& solution) {
    GRASPSolution best_solution = solution;
    
    for (size_t i = 1; i < solution.tour.size() - 2; ++i) {
        for (size_t j = i + 1; j < solution.tour.size(); ++j) {
            // Créer un nouveau tour en inversant le segment [i, j]
            std::vector<osmium::object_id_type> new_tour = solution.tour;
            std::reverse(new_tour.begin() + i, new_tour.begin() + j + 1);
            
            double new_distance = calculate_tour_distance(new_tour);
            if (new_distance < best_solution.total_distance) {
                best_solution.tour = new_tour;
                best_solution.total_distance = new_distance;
            }
        }
    }
    
    return best_solution;
}

// Amélioration 3-opt (version simplifiée)
GRASPSolution GRASPSolver::three_opt_improvement(const GRASPSolution& solution) {
    GRASPSolution best_solution = solution;
    
    for (size_t i = 0; i < solution.tour.size() - 3; ++i) {
        for (size_t j = i + 2; j < solution.tour.size() - 1; ++j) {
            for (size_t k = j + 2; k < solution.tour.size(); ++k) {
                // Essayer différentes reconnexions 3-opt
                std::vector<osmium::object_id_type> new_tour = solution.tour;
                
                // Version 1: inverser segment [i+1, j]
                std::reverse(new_tour.begin() + i + 1, new_tour.begin() + j + 1);
                
                double new_distance = calculate_tour_distance(new_tour);
                if (new_distance < best_solution.total_distance) {
                    best_solution.tour = new_tour;
                    best_solution.total_distance = new_distance;
                }
            }
        }
    }
    
    return best_solution;
}

// Construction de la liste restreinte de candidats (RCL)
std::vector<osmium::object_id_type> GRASPSolver::build_restricted_candidate_list(
    osmium::object_id_type current_node,
    const std::unordered_set<osmium::object_id_type>& unvisited,
    double alpha) {

    std::vector<std::pair<osmium::object_id_type, double>> candidates;
    
    // Calculer les distances vers tous les nœuds non visités
    for (const auto& candidate : unvisited) {
        double distance = get_distance(current_node, candidate);
        candidates.push_back({candidate, distance});
    }
    
    // Trier par distance croissante
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    if (candidates.empty()) return {};
    
    // Calculer les seuils pour la RCL
    double min_distance = candidates[0].second;
    double max_distance = candidates.back().second;
    double threshold = min_distance + alpha * (max_distance - min_distance);
    
    // Construire la RCL
    std::vector<osmium::object_id_type> rcl;
    for (const auto& candidate : candidates) {
        if (candidate.second <= threshold) {
            rcl.push_back(candidate.first);
        }
    }
    
    return rcl;
}

// Méthodes utilitaires
void GRASPSolver::build_distance_cache(const std::vector<osmium::object_id_type>& nodes) {
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

double GRASPSolver::get_distance(osmium::object_id_type node1, osmium::object_id_type node2) {
    auto key = std::make_pair(std::min(node1, node2), std::max(node1, node2));
    auto it = distance_cache.find(key);
    return (it != distance_cache.end()) ? it->second : std::numeric_limits<double>::max();
}

double GRASPSolver::calculate_tour_distance(const std::vector<osmium::object_id_type>& tour) {
    double total_distance = 0.0;
    
    for (size_t i = 0; i < tour.size(); ++i) {
        size_t next_i = (i + 1) % tour.size();
        total_distance += get_distance(tour[i], tour[next_i]);
    }
    
    return total_distance;
}

std::vector<osmium::object_id_type> GRASPSolver::find_shortest_path(
    osmium::object_id_type start, 
    osmium::object_id_type end) {
    
    Pathfinder temp_pathfinder(geo_box);
    return temp_pathfinder.A_Star_Search(start, end);
}

void GRASPSolver::apply_tour_to_ways(
    const std::vector<osmium::object_id_type>& tour,
    int group_id) {

    std::cout << "Application du tour GRASP aux ways du groupe " << group_id << "..." << std::endl;
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

void GRASPSolver::update_way_group(osmium::object_id_type way_id, int new_group) {
    auto it = geo_box.data.ways.find(way_id);
    if (it != geo_box.data.ways.end()) {
        it->second.add_group(new_group);
    } else {
        std::cerr << "Warning: Way " << way_id << " not found" << std::endl;
    }
}