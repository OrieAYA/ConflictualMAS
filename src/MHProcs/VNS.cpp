#include "VNS.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>

// Constructeur
VNSSolver::VNSSolver(GeoBox& box) : geo_box(box) {
    // Initialiser le générateur de nombres aléatoires
    rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
}

// Méthode principale VNS pour un groupe
bool VNSSolver::solve_single_group(
    const std::vector<osmium::object_id_type>& objective_nodes,
    int group_id,
    const VNSParams& params) {

    if (objective_nodes.size() < 2) {
        std::cout << "VNS Groupe " << group_id << ": Pas assez de POI (minimum 2)" << std::endl;
        return false;
    }

    std::cout << "\n=== VNS GROUPE " << group_id << " ===" << std::endl;
    std::cout << "POI à optimiser: " << objective_nodes.size() << std::endl;

    // 1. Construire le cache des distances
    std::cout << "Construction du cache des distances..." << std::endl;
    build_distance_cache(objective_nodes);

    // 2. Générer une solution initiale
    VNSSolution current_solution = generate_initial_solution(objective_nodes);
    if (!current_solution.is_valid) {
        std::cout << "VNS: Impossible de générer une solution initiale" << std::endl;
        return false;
    }

    VNSSolution best_solution = current_solution;
    std::cout << "Solution initiale: " << static_cast<int>(current_solution.total_distance) << "m" << std::endl;

    // 3. Algorithme VNS principal
    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        
        // Boucle sur les structures de voisinage
        for (int k = 0; k < params.max_neighborhoods; ++k) {
            
            // Phase de secousse (shaking)
            VNSSolution shaken_solution = shaking(current_solution, k, params);
            
            // Recherche locale
            NeighborhoodType neighborhood = static_cast<NeighborhoodType>(k % 4);
            VNSSolution local_optimum = local_search(shaken_solution, neighborhood, params);
            
            // Mouvement ou non
            if (local_optimum.is_valid && local_optimum.total_distance < current_solution.total_distance) {
                current_solution = local_optimum;
                
                // Mise à jour du meilleur
                if (current_solution.total_distance < best_solution.total_distance) {
                    best_solution = current_solution;
                }
                
                // Redémarrer depuis le premier voisinage
                k = -1; // Sera incrémenté à 0 par la boucle for
            }
        }
        
        // Diversification si pas d'amélioration
        if (params.diversification && iteration > 0 && iteration % 50 == 0) {
            current_solution = shaking(best_solution, params.max_neighborhoods / 2, params);
        }
        
        // Affichage du progrès
        if (iteration % 40 == 0 || iteration == params.max_iterations - 1) {
            std::cout << "Itération " << iteration << " - Distance actuelle: " 
                      << static_cast<int>(current_solution.total_distance) 
                      << "m, Meilleure: " << static_cast<int>(best_solution.total_distance) << "m" << std::endl;
        }
    }

    std::cout << "Solution VNS trouvée - Distance totale: " << static_cast<int>(best_solution.total_distance) << "m" << std::endl;
    
    // Appliquer le tour optimal aux ways
    apply_tour_to_ways(best_solution.tour, group_id);

    return true;
}

// Génération de solution initiale
VNSSolution VNSSolver::generate_initial_solution(const std::vector<osmium::object_id_type>& nodes) {
    // Utiliser l'heuristique du plus proche voisin
    return nearest_neighbor_heuristic(nodes);
}

VNSSolution VNSSolver::nearest_neighbor_heuristic(const std::vector<osmium::object_id_type>& nodes) {
    VNSSolution solution;
    
    if (nodes.empty()) return solution;
    
    // Commencer par le premier nœud
    osmium::object_id_type current = nodes[0];
    solution.tour.push_back(current);
    
    std::unordered_set<osmium::object_id_type> unvisited(nodes.begin() + 1, nodes.end());
    
    // Construire le tour avec l'heuristique du plus proche voisin
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
            solution.tour.push_back(nearest);
            unvisited.erase(nearest);
            current = nearest;
        } else {
            break;
        }
    }
    
    if (solution.tour.size() == nodes.size()) {
        solution.total_distance = calculate_tour_distance(solution.tour);
        solution.is_valid = true;
    }
    
    return solution;
}

// Phase de secousse (shaking)
VNSSolution VNSSolver::shaking(
    const VNSSolution& solution, 
    int neighborhood_index, 
    const VNSParams& params) {

    VNSSolution shaken = solution;
    
    // Intensité de la perturbation basée sur l'index du voisinage
    int perturbation_size = 1 + neighborhood_index;
    
    switch (neighborhood_index % 3) {
        case 0:
            // k-opt aléatoire
            shaken = random_k_opt(solution, perturbation_size);
            break;
            
        case 1:
            // Relocations aléatoires
            shaken = random_relocations(solution, perturbation_size);
            break;
            
        case 2:
            // Échanges aléatoires
            shaken = random_swaps(solution, perturbation_size);
            break;
    }
    
    return shaken;
}

// Recherche locale
VNSSolution VNSSolver::local_search(
    const VNSSolution& solution, 
    NeighborhoodType neighborhood_type,
    const VNSParams& params) {

    switch (neighborhood_type) {
        case NeighborhoodType::TWO_OPT:
            return two_opt_neighborhood(solution, params.use_first_improvement);
            
        case NeighborhoodType::THREE_OPT:
            return three_opt_neighborhood(solution, params.use_first_improvement);
            
        case NeighborhoodType::RELOCATE:
            return relocate_neighborhood(solution, params.use_first_improvement);
            
        case NeighborhoodType::SWAP:
            return swap_neighborhood(solution, params.use_first_improvement);
            
        default:
            return solution;
    }
}

// Structure de voisinage 2-opt
VNSSolution VNSSolver::two_opt_neighborhood(const VNSSolution& solution, bool first_improvement) {
    VNSSolution best = solution;
    
    for (size_t i = 1; i < solution.tour.size() - 1; ++i) {
        for (size_t j = i + 1; j < solution.tour.size(); ++j) {
            // Créer nouveau tour en inversant le segment [i, j]
            std::vector<osmium::object_id_type> new_tour = solution.tour;
            std::reverse(new_tour.begin() + i, new_tour.begin() + j + 1);
            
            double new_distance = calculate_tour_distance(new_tour);
            if (new_distance < best.total_distance) {
                best.tour = new_tour;
                best.total_distance = new_distance;
                
                if (first_improvement) {
                    return best;
                }
            }
        }
    }
    
    return best;
}

// Structure de voisinage 3-opt (version simplifiée)
VNSSolution VNSSolver::three_opt_neighborhood(const VNSSolution& solution, bool first_improvement) {
    VNSSolution best = solution;
    
    for (size_t i = 0; i < solution.tour.size() - 2; ++i) {
        for (size_t j = i + 2; j < solution.tour.size(); ++j) {
            // Version simplifiée: juste inverser un segment
            std::vector<osmium::object_id_type> new_tour = solution.tour;
            std::reverse(new_tour.begin() + i, new_tour.begin() + j + 1);
            
            double new_distance = calculate_tour_distance(new_tour);
            if (new_distance < best.total_distance) {
                best.tour = new_tour;
                best.total_distance = new_distance;
                
                if (first_improvement) {
                    return best;
                }
            }
        }
    }
    
    return best;
}

// Structure de voisinage relocate
VNSSolution VNSSolver::relocate_neighborhood(const VNSSolution& solution, bool first_improvement) {
    VNSSolution best = solution;
    
    for (size_t i = 0; i < solution.tour.size(); ++i) {
        for (size_t j = 0; j < solution.tour.size(); ++j) {
            if (i == j) continue;
            
            // Déplacer l'élément i vers la position j
            std::vector<osmium::object_id_type> new_tour = solution.tour;
            osmium::object_id_type element = new_tour[i];
            new_tour.erase(new_tour.begin() + i);
            new_tour.insert(new_tour.begin() + (j > i ? j - 1 : j), element);
            
            double new_distance = calculate_tour_distance(new_tour);
            if (new_distance < best.total_distance) {
                best.tour = new_tour;
                best.total_distance = new_distance;
                
                if (first_improvement) {
                    return best;
                }
            }
        }
    }
    
    return best;
}

// Structure de voisinage swap
VNSSolution VNSSolver::swap_neighborhood(const VNSSolution& solution, bool first_improvement) {
    VNSSolution best = solution;
    
    for (size_t i = 0; i < solution.tour.size(); ++i) {
        for (size_t j = i + 1; j < solution.tour.size(); ++j) {
            // Échanger les éléments i et j
            std::vector<osmium::object_id_type> new_tour = solution.tour;
            std::swap(new_tour[i], new_tour[j]);
            
            double new_distance = calculate_tour_distance(new_tour);
            if (new_distance < best.total_distance) {
                best.tour = new_tour;
                best.total_distance = new_distance;
                
                if (first_improvement) {
                    return best;
                }
            }
        }
    }
    
    return best;
}

// Perturbations pour la phase de secousse
VNSSolution VNSSolver::random_k_opt(const VNSSolution& solution, int k) {
    VNSSolution result = solution;
    
    for (int i = 0; i < k && result.tour.size() > 3; ++i) {
        std::uniform_int_distribution<size_t> dist(1, result.tour.size() - 2);
        size_t pos1 = dist(rng);
        size_t pos2 = dist(rng);
        
        if (pos1 > pos2) std::swap(pos1, pos2);
        if (pos2 - pos1 > 1) {
            std::reverse(result.tour.begin() + pos1, result.tour.begin() + pos2);
        }
    }
    
    result.total_distance = calculate_tour_distance(result.tour);
    return result;
}

VNSSolution VNSSolver::random_relocations(const VNSSolution& solution, int num_relocations) {
    VNSSolution result = solution;
    
    for (int i = 0; i < num_relocations && result.tour.size() > 2; ++i) {
        std::uniform_int_distribution<size_t> dist(0, result.tour.size() - 1);
        size_t from = dist(rng);
        size_t to = dist(rng);
        
        if (from != to) {
            osmium::object_id_type element = result.tour[from];
            result.tour.erase(result.tour.begin() + from);
            result.tour.insert(result.tour.begin() + (to > from ? to - 1 : to), element);
        }
    }
    
    result.total_distance = calculate_tour_distance(result.tour);
    return result;
}

VNSSolution VNSSolver::random_swaps(const VNSSolution& solution, int num_swaps) {
    VNSSolution result = solution;
    
    for (int i = 0; i < num_swaps && result.tour.size() > 1; ++i) {
        std::uniform_int_distribution<size_t> dist(0, result.tour.size() - 1);
        size_t pos1 = dist(rng);
        size_t pos2 = dist(rng);
        
        std::swap(result.tour[pos1], result.tour[pos2]);
    }
    
    result.total_distance = calculate_tour_distance(result.tour);
    return result;
}

// Méthodes utilitaires (similaires à GRASP et ACO)
void VNSSolver::build_distance_cache(const std::vector<osmium::object_id_type>& nodes) {
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

double VNSSolver::get_distance(osmium::object_id_type node1, osmium::object_id_type node2) {
    auto key = std::make_pair(std::min(node1, node2), std::max(node1, node2));
    auto it = distance_cache.find(key);
    return (it != distance_cache.end()) ? it->second : std::numeric_limits<double>::max();
}

double VNSSolver::calculate_tour_distance(const std::vector<osmium::object_id_type>& tour) {
    double total_distance = 0.0;
    
    for (size_t i = 0; i < tour.size(); ++i) {
        size_t next_i = (i + 1) % tour.size();
        total_distance += get_distance(tour[i], tour[next_i]);
    }
    
    return total_distance;
}

std::vector<osmium::object_id_type> VNSSolver::find_shortest_path(
    osmium::object_id_type start, 
    osmium::object_id_type end) {
    
    Pathfinder temp_pathfinder(geo_box);
    return temp_pathfinder.A_Star_Search(start, end);
}

void VNSSolver::apply_tour_to_ways(
    const std::vector<osmium::object_id_type>& tour,
    int group_id) {

    std::cout << "Application du tour VNS aux ways du groupe " << group_id << "..." << std::endl;
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

void VNSSolver::update_way_group(osmium::object_id_type way_id, int new_group) {
    auto it = geo_box.data.ways.find(way_id);
    if (it != geo_box.data.ways.end()) {
        it->second.add_group(new_group);
    } else {
        std::cerr << "Warning: Way " << way_id << " not found" << std::endl;
    }
}

bool VNSSolver::is_valid_tour(const std::vector<osmium::object_id_type>& tour, 
                             const std::vector<osmium::object_id_type>& original_nodes) {
    return tour.size() == original_nodes.size();
}