#include "VNS.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cmath>
#include <iomanip>

// ============================================================================
// CONSTRUCTEUR
// ============================================================================
VNSSolver::VNSSolver(GeoBox& box, Pathfinder& pf, GlobalMemory& mem) 
    : geo_box(box), pathfinder(pf), global_memory(mem), best_fitness(0.0) {
    rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
}

void VNSSolver::diagnose_vns_behavior(
    const std::vector<osmium::object_id_type>& initial_sol,
    const std::vector<osmium::object_id_type>& final_sol,
    int total_iterations,
    int improvements_count) {
    
    std::cout << "\n======================================================" << std::endl;
    std::cout << "|              DIAGNOSTIC VNS                        |" << std::endl;
    std::cout << "======================================================" << std::endl;
    
    // ========================================
    // TEST 1 : COMPARAISON INITIAL VS FINAL
    // ========================================
    std::cout << "\n[TEST 1] Evolution de la solution:" << std::endl;
    
    double initial_fitness = calculate_fitness(initial_sol);
    double final_fitness = calculate_fitness(final_sol);
    double improvement_pct = ((final_fitness - initial_fitness) / initial_fitness) * 100.0;
    
    std::cout << "  Solution initiale:" << std::endl;
    std::cout << "    - Fitness: " << static_cast<int>(initial_fitness) << std::endl;
    std::cout << "    - POIs: " << initial_sol.size() << std::endl;
    std::cout << "    - Distance: " << static_cast<int>(calculate_distance(initial_sol)) << "m" << std::endl;
    
    std::cout << "\n  Solution finale:" << std::endl;
    std::cout << "    - Fitness: " << static_cast<int>(final_fitness) << std::endl;
    std::cout << "    - POIs: " << final_sol.size() << std::endl;
    std::cout << "    - Distance: " << static_cast<int>(calculate_distance(final_sol)) << "m" << std::endl;
    
    std::cout << "\n  Amelioration: " << std::fixed << std::setprecision(2) 
              << improvement_pct << "%" << std::endl;
    
    if (improvement_pct < 1.0) {
        std::cout << "  [WARN]  PROBLEME: Amelioration < 1% - VNS n'a presque rien fait !" << std::endl;
    } else if (improvement_pct < 5.0) {
        std::cout << "  [WARN]  Amelioration faible (< 5%)" << std::endl;
    } else {
        std::cout << "  [OK] Amelioration significative" << std::endl;
    }
    
    std::cout << "\n  Iterations totales: " << total_iterations << std::endl;
    std::cout << "  Ameliorations acceptees: " << improvements_count << std::endl;
    
    if (improvements_count == 0) {
        std::cout << "  [WARN]  PROBLEME CRITIQUE: AUCUNE amelioration acceptee !" << std::endl;
        std::cout << "      -> La solution initiale est DEJA un optimum local" << std::endl;
        std::cout << "      -> OU les operateurs VND ne fonctionnent pas" << std::endl;
    } else if (improvements_count < 5) {
        std::cout << "  [WARN]  Tres peu d'ameliorations (< 5)" << std::endl;
    }
    
    // ========================================
    // TEST 2 : TESTER SHAKE MANUELLEMENT
    // ========================================
    std::cout << "\n[TEST 2] Test des operateurs SHAKE:" << std::endl;
    
    for (int k = 1; k <= 5; k++) {
        std::vector<osmium::object_id_type> shaken = shake(final_sol, k);
        double shaken_fitness = calculate_fitness(shaken);
        double diff = final_fitness - shaken_fitness;
        
        std::cout << "  k=" << k << " | Fitness apres shake: " 
                  << static_cast<int>(shaken_fitness) 
                  << " | Delta = " << static_cast<int>(diff)
                  << " | POIs: " << shaken.size() << std::endl;
        
        if (diff < 10) {
            std::cout << "      [WARN]  Shake trop faible (Delta < 10) - perturbation insuffisante !" << std::endl;
        }
    }
    
    // ========================================
    // TEST 3 : TESTER VND MANUELLEMENT
    // ========================================
    std::cout << "\n[TEST 3] Test des operateurs VND:" << std::endl;
    
    VNSParams test_params;
    test_params.max_iterations_vnd = 10;
    test_params.use_add_remove = true;
    test_params.use_swap = true;
    test_params.use_insertion = true;
    test_params.use_two_opt = true;
    
    // Test add_best_poi
    std::vector<osmium::object_id_type> with_add = add_best_poi(final_sol);
    double add_fitness = calculate_fitness(with_add);
    std::cout << "  add_best_poi: " << static_cast<int>(final_fitness) 
              << " -> " << static_cast<int>(add_fitness)
              << " (Delta=" << static_cast<int>(add_fitness - final_fitness) << ")" << std::endl;
    
    if (add_fitness <= final_fitness) {
        std::cout << "      [WARN]  add_best_poi n'ameliore PAS !" << std::endl;
        
        // Verifier pourquoi
        std::unordered_set<osmium::object_id_type> in_sol(final_sol.begin(), final_sol.end());
        int candidates = 0;
        for (const auto& poi : all_pois) {
            if (in_sol.count(poi) == 0) {
                candidates++;
            }
        }
        
        if (candidates == 0) {
            std::cout << "         Raison: TOUS les POIs sont deja dans la solution !" << std::endl;
        } else {
            std::cout << "         POIs disponibles: " << candidates << std::endl;
            std::cout << "         Raison: Contrainte distance trop stricte ?" << std::endl;
        }
    }
    
    // Test best_swap
    std::vector<osmium::object_id_type> with_swap = best_swap(final_sol);
    double swap_fitness = calculate_fitness(with_swap);
    std::cout << "  best_swap: " << static_cast<int>(final_fitness) 
              << " -> " << static_cast<int>(swap_fitness)
              << " (Delta=" << static_cast<int>(swap_fitness - final_fitness) << ")" << std::endl;
    
    if (swap_fitness <= final_fitness) {
        std::cout << "      [WARN]  best_swap n'ameliore PAS !" << std::endl;
    }
    
    // Test best_insertion
    std::vector<osmium::object_id_type> with_insert = best_insertion(final_sol);
    double insert_fitness = calculate_fitness(with_insert);
    std::cout << "  best_insertion: " << static_cast<int>(final_fitness) 
              << " -> " << static_cast<int>(insert_fitness)
              << " (Delta=" << static_cast<int>(insert_fitness - final_fitness) << ")" << std::endl;
    
    if (insert_fitness <= final_fitness) {
        std::cout << "      [WARN]  best_insertion n'ameliore PAS !" << std::endl;
    }
    
    // Test two_opt
    std::vector<osmium::object_id_type> with_2opt = two_opt(final_sol);
    double opt_fitness = calculate_fitness(with_2opt);
    double opt_dist = calculate_distance(with_2opt);
    double orig_dist = calculate_distance(final_sol);
    
    std::cout << "  two_opt: distance " << static_cast<int>(orig_dist) 
              << " -> " << static_cast<int>(opt_dist)
              << " (Delta=" << static_cast<int>(orig_dist - opt_dist) << "m)" << std::endl;
    
    if (opt_dist >= orig_dist) {
        std::cout << "      [WARN]  two_opt ne reduit PAS la distance !" << std::endl;
    }
    
    // ========================================
    // TEST 4 : SATURATION DE LA SOLUTION
    // ========================================
    std::cout << "\n[TEST 4] Saturation:" << std::endl;
    
    double current_dist = calculate_distance(final_sol);
    double remaining_dist = max_distance_constraint - current_dist;
    double usage_pct = (current_dist / max_distance_constraint) * 100.0;
    
    std::cout << "  Distance utilisee: " << static_cast<int>(current_dist) << "m / "
              << static_cast<int>(max_distance_constraint) << "m ("
              << std::fixed << std::setprecision(1) << usage_pct << "%)" << std::endl;
    std::cout << "  Distance restante: " << static_cast<int>(remaining_dist) << "m" << std::endl;
    
    std::unordered_set<osmium::object_id_type> in_sol(final_sol.begin(), final_sol.end());
    int pois_used = final_sol.size();
    int pois_available = all_pois.size() - pois_used;
    
    std::cout << "  POIs utilises: " << pois_used << " / " << all_pois.size() 
              << " (" << static_cast<int>((pois_used * 100.0) / all_pois.size()) << "%)" << std::endl;
    std::cout << "  POIs disponibles: " << pois_available << std::endl;
    
    if (pois_available == 0) {
        std::cout << "  [OK] Solution SATUREE - tous les POIs sont utilises" << std::endl;
        std::cout << "    -> Normal que VNS ne bouge plus" << std::endl;
    } else if (usage_pct > 98.0) {
        std::cout << "  [OK] Contrainte distance saturee (> 98%)" << std::endl;
        std::cout << "    -> Peu de marge pour ajouter des POIs" << std::endl;
    } else if (usage_pct < 70.0) {
        std::cout << "  [WARN]  Utilisation faible (< 70%)" << std::endl;
        std::cout << "      -> VNS devrait pouvoir ajouter plus de POIs !" << std::endl;
    }
    
    // ========================================
    // TEST 5 : VERIFIER GREEDY_REPAIR
    // ========================================
    std::cout << "\n[TEST 5] Test greedy_repair:" << std::endl;
    
    // Retirer 5 POIs et essayer de reparer
    std::vector<osmium::object_id_type> damaged = random_remove(final_sol, 5);
    double damaged_fitness = calculate_fitness(damaged);
    
    std::vector<osmium::object_id_type> repaired = greedy_repair(damaged);
    double repaired_fitness = calculate_fitness(repaired);
    
    std::cout << "  Solution degradee: " << static_cast<int>(damaged_fitness) 
              << " (" << damaged.size() << " POIs)" << std::endl;
    std::cout << "  Apres greedy_repair: " << static_cast<int>(repaired_fitness)
              << " (" << repaired.size() << " POIs)" << std::endl;
    std::cout << "  Recuperation: " << static_cast<int>(repaired_fitness - damaged_fitness) << std::endl;
    
    if (repaired_fitness < final_fitness * 0.9) {
        std::cout << "  [WARN]  greedy_repair ne reconstruit PAS bien !" << std::endl;
        std::cout << "      -> Probleme dans l'algorithme greedy" << std::endl;
    }
    
    // ========================================
    // CONCLUSION
    // ========================================
    std::cout << "\n[CONCLUSION]" << std::endl;
    
    if (pois_available == 0 || usage_pct > 98.0) {
        std::cout << "  [OK] NORMAL: Solution deja optimale/saturee des le depart" << std::endl;
        std::cout << "    -> Greedy construction est tres efficace" << std::endl;
        std::cout << "    -> VNS n'a rien a ameliorer" << std::endl;
    } else if (improvements_count == 0) {
        std::cout << "  [ERREUR] PROBLEME: VNS ne fonctionne PAS" << std::endl;
        std::cout << "    Causes possibles:" << std::endl;
        std::cout << "    1. Shake trop faible (ne perturbe pas assez)" << std::endl;
        std::cout << "    2. VND bloque (operateurs n'ameliorent pas)" << std::endl;
        std::cout << "    3. Condition d'acceptation trop stricte" << std::endl;
    } else if (improvement_pct < 5.0 && pois_available > 10) {
        std::cout << "  [WARN]  SOUS-OPTIMAL: VNS ameliore peu" << std::endl;
        std::cout << "    -> Augmenter k_max ou max_iterations" << std::endl;
        std::cout << "    -> Shake plus agressif (retirer plus de POIs)" << std::endl;
    } else {
        std::cout << "  [OK] VNS fonctionne correctement" << std::endl;
    }
    
    std::cout << "\n======================================================" << std::endl;
}

// ============================================================================
// COLLECTE DES POIs RECOMPENSABLES
// ============================================================================
void VNSSolver::collect_rewardable_pois(const std::vector<int>& chars) {
    all_pois.clear();
    std::unordered_set<osmium::object_id_type> seen;
    
    std::cout << "[VNS] Collecte POIs avec characteristics:" << std::endl;
    for(size_t i = 0; i < chars.size(); i++) {
        if(chars[i] > 0) {
            std::cout << "  Groupe " << i << " = " << chars[i] << std::endl;
        }
    }

    for (const auto& [node_id, node_data] : geo_box.data.nodes) {
        if (node_data.groupes.empty()) continue;
        
        // ✓ CORRECTION : Vérifier si ce POI appartient à UN groupe valorisé
        bool is_rewardable = false;
        for (const int group_id : node_data.groupes) {
            // ✓ Vérifier si characteristics[group_id] > 0
            if (group_id >= 0 && group_id < static_cast<int>(chars.size())) {
                if (chars[group_id] > 0) {  // ← BONNE VÉRIFICATION
                    is_rewardable = true;
                    break;
                }
            }
        }
        
        if (is_rewardable && seen.insert(node_id).second) {
            all_pois.push_back(node_id);
        }
    }
    
    std::cout << "[VNS] POIs récompensables trouvés: " << all_pois.size() << std::endl;
}

// ============================================================================
// GENERATION SOLUTION INITIALE
// ============================================================================
std::vector<osmium::object_id_type> VNSSolver::generate_initial_solution() {
    
    std::cout << "\n[DIAGNOSTIC] generate_initial_solution:" << std::endl;
    std::cout << "  all_pois.size() = " << all_pois.size() << std::endl;
    
    // ✓ NOUVEAU : Diagnostiquer la connectivité
    if (all_pois.size() >= 2 && all_pois.size() <= 20) {
        diagnose_poi_connectivity();
    }
    
    // Essayer plusieurs solutions greedy
    std::vector<osmium::object_id_type> best;
    double best_fit = 0.0;
    
    int num_tries = std::min(5, static_cast<int>(all_pois.size()));
    
    for (int i = 0; i < num_tries; i++) {
        std::vector<osmium::object_id_type> candidate = greedy_construction();
        
        double fit = calculate_fitness(candidate);
        if (fit > best_fit) {
            best_fit = fit;
            best = candidate;
        }
    }
    
    return best;
}

void VNSSolver::diagnose_poi_connectivity() {
    
    std::cout << "\n[DIAGNOSTIC] Connectivité des POIs:" << std::endl;
    std::cout << "Total POIs: " << all_pois.size() << std::endl;
    
    int total_pairs = 0;
    int connected_pairs = 0;
    int disconnected_pairs = 0;
    
    for (size_t i = 0; i < all_pois.size(); i++) {
        for (size_t j = i + 1; j < all_pois.size(); j++) {
            total_pairs++;
            
            std::vector<Path> paths = global_memory.check_path(all_pois[i], all_pois[j]);
            
            if (!paths.empty() && paths[0].cost != std::numeric_limits<float>::max()) {
                connected_pairs++;
            } else {
                disconnected_pairs++;
                
                // Afficher les 5 premières paires déconnectées
                if (disconnected_pairs <= 5) {
                    std::cout << "  ⚠️  Pas de chemin: POI " << all_pois[i] 
                              << " ↔ POI " << all_pois[j] << std::endl;
                }
            }
        }
    }
    
    double connectivity_pct = (connected_pairs * 100.0) / total_pairs;
    
    std::cout << "\nRÉSULTAT:" << std::endl;
    std::cout << "  Paires totales: " << total_pairs << std::endl;
    std::cout << "  Paires connectées: " << connected_pairs 
              << " (" << std::fixed << std::setprecision(1) << connectivity_pct << "%)" << std::endl;
    std::cout << "  Paires déconnectées: " << disconnected_pairs << std::endl;
    
    if (connectivity_pct < 50.0) {
        std::cout << "\n⚠️  PROBLÈME GRAVE: Graphe très peu connexe (<50%) !" << std::endl;
        std::cout << "   → Vérifier Neighbor_Search et check_path" << std::endl;
    } else if (connectivity_pct < 90.0) {
        std::cout << "\n⚠️  Graphe partiellement connexe" << std::endl;
    } else {
        std::cout << "\n✓ Graphe bien connexe" << std::endl;
    }
}

std::vector<osmium::object_id_type> VNSSolver::greedy_construction() {
    
    if (all_pois.empty()) {
        std::cout << "[VNS] all_pois est vide !" << std::endl;
        return {};
    }
    
    // Choisir un POI de départ aléatoire
    std::uniform_int_distribution<size_t> dist(0, all_pois.size() - 1);
    osmium::object_id_type start_poi = all_pois[dist(rng)];
    
    std::vector<osmium::object_id_type> solution;
    std::unordered_set<osmium::object_id_type> used;
    
    solution.push_back(start_poi);
    used.insert(start_poi);
    
    double current_distance = 0.0;
    
    std::cout << "[VNS] Greedy start depuis POI " << start_poi << std::endl;
    
    // ========================================
    // GREEDY LOOP AVEC VÉRIFICATION STRICTE
    // ========================================
    bool added = true;
    int iteration = 0;
    
    while (added && used.size() < all_pois.size() && iteration < 100) {
        added = false;
        iteration++;
        
        osmium::object_id_type best_next = 0;
        double best_ratio = -1.0;
        double best_edge_distance = 0.0;
        
        osmium::object_id_type last_poi = solution.back();
        
        int candidates_tested = 0;
        int candidates_no_path = 0;
        int candidates_too_far = 0;
        int candidates_feasible = 0;
        
        // ========================================
        // TESTER TOUS LES POIs NON-UTILISÉS
        // ========================================
        for (const auto& candidate : all_pois) {
            if (used.count(candidate)) continue;
            
            candidates_tested++;
            
            // ✓ VÉRIFICATION 1 : Existe-t-il un CHEMIN last_poi → candidate ?
            std::vector<Path> paths = global_memory.check_path(last_poi, candidate);
            
            if (paths.empty()) {
                candidates_no_path++;
                continue;  // Pas de chemin trouvé
            }
            
            // ✓ VÉRIFICATION 2 : Le chemin est-il valide (cost != infinity) ?
            if (paths[0].cost == std::numeric_limits<float>::max()) {
                candidates_no_path++;
                continue;  // Chemin invalide
            }
            
            double edge_distance = static_cast<double>(paths[0].cost);
            double new_total = current_distance + edge_distance;
            
            // ✓ VÉRIFICATION 3 : Contrainte distance
            if (new_total > max_distance_constraint) {
                candidates_too_far++;
                continue;
            }
            
            candidates_feasible++;
            
            // ✓ VÉRIFICATION 4 : Calculer ratio reward/distance
            int reward = get_poi_reward(candidate);
            
            if (edge_distance > 0 && reward > 0) {
                double ratio = static_cast<double>(reward) / edge_distance;
                
                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_next = candidate;
                    best_edge_distance = edge_distance;
                }
            }
        }
        
        // ========================================
        // DEBUG : Afficher statistiques
        // ========================================
        if (iteration % 5 == 0 || iteration == 1 || best_next == 0) {
            std::cout << "  Iter " << iteration 
                      << ": tested=" << candidates_tested 
                      << ", no_path=" << candidates_no_path
                      << ", too_far=" << candidates_too_far
                      << ", feasible=" << candidates_feasible;
        }
        
        // ========================================
        // AJOUTER LE MEILLEUR POI TROUVÉ
        // ========================================
        if (best_next != 0) {
            solution.push_back(best_next);
            used.insert(best_next);
            current_distance += best_edge_distance;
            added = true;
            
            if (iteration % 5 == 0 || iteration == 1) {
                std::cout << " -> Added POI " << best_next 
                          << " (ratio=" << std::fixed << std::setprecision(2) << best_ratio 
                          << ", total_dist=" << static_cast<int>(current_distance) << "m)" 
                          << std::endl;
            }
        } else {
            if (iteration % 5 == 0 || iteration == 1) {
                std::cout << " -> No feasible POI" << std::endl;
            }
            
            // ✓ DIAGNOSTIC : Pourquoi aucun POI n'est faisable ?
            if (candidates_no_path == candidates_tested) {
                std::cout << "  ⚠️  PROBLÈME: TOUS les POIs restants sont INACCESSIBLES !" << std::endl;
                std::cout << "      → Graphe non-connexe ou Neighbor_Search défaillant" << std::endl;
            } else if (candidates_too_far == candidates_tested) {
                std::cout << "  ℹ️  Contrainte distance saturée (tous POIs > " 
                          << static_cast<int>(max_distance_constraint) << "m)" << std::endl;
            }
            
            break;
        }
    }
    
    double final_fitness = calculate_fitness(solution);
    double final_distance = calculate_distance(solution);
    
    std::cout << "[VNS] Greedy final: " 
              << solution.size() << " POIs, "
              << "fitness=" << static_cast<int>(final_fitness) 
              << ", distance=" << static_cast<int>(final_distance) << "m"
              << std::endl;
    
    // ========================================
    // VÉRIFICATION FINALE DE LA SÉQUENCE
    // ========================================
    if (!verify_solution_sequence(solution)) {
        std::cout << "[ERREUR] Solution invalide : séquence non continue !" << std::endl;
    }
    
    return solution;
}

// ============================================================================
// SHAKE (Perturbation)
// ============================================================================
std::vector<osmium::object_id_type> VNSSolver::shake(
    const std::vector<osmium::object_id_type>& solution,
    int k) {
    
    if (solution.size() <= 2) return solution;
    
    std::vector<osmium::object_id_type> shaken = solution;
    
    // [OK] SHAKE PLUS AGRESSIF
    switch(k) {
        case 1:
            // k=1 : Retirer 20% (minimum 2)
            {
                int num = std::max(2, static_cast<int>(shaken.size() * 0.2));
                shaken = random_remove(shaken, num);
            }
            break;
            
        case 2:
            // k=2 : Retirer 30%
            {
                int num = std::max(3, static_cast<int>(shaken.size() * 0.3));
                shaken = random_remove(shaken, num);
            }
            break;
            
        case 3:
            // k=3 : Retirer 40%
            {
                int num = std::max(4, static_cast<int>(shaken.size() * 0.4));
                shaken = random_remove(shaken, num);
            }
            break;
            
        case 4:
            // k=4 : Retirer 50%
            {
                int num = std::max(5, static_cast<int>(shaken.size() * 0.5));
                shaken = random_remove(shaken, num);
            }
            break;
            
        case 5:
            // k=5 : Retirer 60% + swaps
            {
                int num = std::max(6, static_cast<int>(shaken.size() * 0.6));
                shaken = random_remove(shaken, num);
                shaken = random_swap(shaken, 3);
            }
            break;
            
        default:
            // k > 5 : Retirer 70%
            {
                int num = std::max(7, static_cast<int>(shaken.size() * 0.7));
                shaken = random_remove(shaken, num);
            }
            break;
    }
    
    // Reparer avec greedy
    shaken = greedy_repair(shaken);
    
    return shaken;
}

std::vector<osmium::object_id_type> VNSSolver::random_remove(
    const std::vector<osmium::object_id_type>& solution,
    int num_to_remove) {
    
    if (solution.size() <= 2) return solution;
    
    std::vector<osmium::object_id_type> result = solution;
    num_to_remove = std::min(num_to_remove, static_cast<int>(result.size()) - 1);
    
    for (int i = 0; i < num_to_remove; i++) {
        if (result.size() <= 1) break;
        
        std::uniform_int_distribution<size_t> dist(0, result.size() - 1);
        size_t pos = dist(rng);
        result.erase(result.begin() + pos);
    }
    
    return result;
}

std::vector<osmium::object_id_type> VNSSolver::random_swap(
    const std::vector<osmium::object_id_type>& solution,
    int num_swaps) {
    
    if (solution.size() < 2) return solution;
    
    std::vector<osmium::object_id_type> result = solution;
    
    for (int i = 0; i < num_swaps; i++) {
        std::uniform_int_distribution<size_t> dist(0, result.size() - 1);
        size_t pos1 = dist(rng);
        size_t pos2 = dist(rng);
        
        if (pos1 != pos2) {
            std::swap(result[pos1], result[pos2]);
        }
    }
    
    return result;
}

std::vector<osmium::object_id_type> VNSSolver::random_insert(
    const std::vector<osmium::object_id_type>& solution,
    int num_inserts) {
    
    if (solution.size() < 2) return solution;
    
    std::vector<osmium::object_id_type> result = solution;
    
    for (int i = 0; i < num_inserts; i++) {
        if (result.size() <= 1) break;
        
        std::uniform_int_distribution<size_t> dist(0, result.size() - 1);
        size_t from = dist(rng);
        size_t to = dist(rng);
        
        osmium::object_id_type poi = result[from];
        result.erase(result.begin() + from);
        
        if (to > from) to--;
        if (to < result.size()) {
            result.insert(result.begin() + to, poi);
        } else {
            result.push_back(poi);
        }
    }
    
    return result;
}

// ============================================================================
// VND (Variable Neighborhood Descent)
// ============================================================================
std::vector<osmium::object_id_type> VNSSolver::variable_neighborhood_descent(
    const std::vector<osmium::object_id_type>& solution,
    const VNSParams& params) {
    
    std::vector<osmium::object_id_type> current = solution;
    double current_fitness = calculate_fitness(current);
    
    int l = 1;
    const int l_max = 4;
    int iterations = 0;
    
    while (l <= l_max && iterations < params.max_iterations_vnd) {
        iterations++;
        
        std::vector<osmium::object_id_type> neighbor;
        
        switch(l) {
            case 1:
                // N1 : Ajouter meilleur POI
                if (params.use_add_remove) {
                    neighbor = add_best_poi(current);
                }
                break;
                
            case 2:
                // N2 : Best swap
                if (params.use_swap) {
                    neighbor = best_swap(current);
                }
                break;
                
            case 3:
                // N3 : Best insertion
                if (params.use_insertion) {
                    neighbor = best_insertion(current);
                }
                break;
                
            case 4:
                // N4 : 2-opt
                if (params.use_two_opt) {
                    neighbor = two_opt(current);
                }
                break;
        }
        
        if (neighbor.empty()) {
            l++;
            continue;
        }
        
        double neighbor_fitness = calculate_fitness(neighbor);
        
        if (neighbor_fitness > current_fitness && is_feasible(neighbor)) {
            current = neighbor;
            current_fitness = neighbor_fitness;
            l = 1;  // Recommencer avec le premier voisinage
        } else {
            l++;  // Passer au voisinage suivant
        }
    }
    
    return current;
}

// ============================================================================
// VOISINAGES
// ============================================================================
std::vector<osmium::object_id_type> VNSSolver::two_opt(
    const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.size() < 4) return solution;
    
    std::vector<osmium::object_id_type> best = solution;
    double best_distance = calculate_distance(solution);
    bool improved = false;
    
    for (size_t i = 0; i < solution.size() - 1; i++) {
        for (size_t j = i + 2; j < solution.size(); j++) {
            std::vector<osmium::object_id_type> candidate = solution;
            std::reverse(candidate.begin() + i + 1, candidate.begin() + j + 1);
            
            double dist = calculate_distance(candidate);
            if (dist < best_distance && dist <= max_distance_constraint) {
                best = candidate;
                best_distance = dist;
                improved = true;
            }
        }
    }
    
    return improved ? best : solution;
}

std::vector<osmium::object_id_type> VNSSolver::best_insertion(
    const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.size() < 2) return solution;
    
    std::vector<osmium::object_id_type> best = solution;
    double best_fitness = calculate_fitness(solution);
    
    for (size_t i = 0; i < solution.size(); i++) {
        for (size_t j = 0; j < solution.size(); j++) {
            if (i == j) continue;
            
            std::vector<osmium::object_id_type> candidate = solution;
            osmium::object_id_type poi = candidate[i];
            candidate.erase(candidate.begin() + i);
            
            size_t insert_pos = (j > i) ? j - 1 : j;
            candidate.insert(candidate.begin() + insert_pos, poi);
            
            if (is_feasible(candidate)) {
                double fitness = calculate_fitness(candidate);
                if (fitness > best_fitness) {
                    best = candidate;
                    best_fitness = fitness;
                }
            }
        }
    }
    
    return best;
}

std::vector<osmium::object_id_type> VNSSolver::best_swap(
    const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.size() < 2) return solution;
    
    std::vector<osmium::object_id_type> best = solution;
    double best_fitness = calculate_fitness(solution);
    
    for (size_t i = 0; i < solution.size(); i++) {
        for (size_t j = i + 1; j < solution.size(); j++) {
            std::vector<osmium::object_id_type> candidate = solution;
            std::swap(candidate[i], candidate[j]);
            
            if (is_feasible(candidate)) {
                double fitness = calculate_fitness(candidate);
                if (fitness > best_fitness) {
                    best = candidate;
                    best_fitness = fitness;
                }
            }
        }
    }
    
    return best;
}

std::vector<osmium::object_id_type> VNSSolver::add_best_poi(
    const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.empty()) return solution;
    
    std::unordered_set<osmium::object_id_type> in_solution(solution.begin(), solution.end());
    
    osmium::object_id_type best_poi = 0;
    double best_improvement = 0.0;
    
    osmium::object_id_type last_poi = solution.back();
    
    for (const auto& poi : all_pois) {
        if (in_solution.count(poi)) continue;
        
        // ✓ VÉRIFICATION : Existe-t-il un chemin last_poi → poi ?
        std::vector<Path> paths = global_memory.check_path(last_poi, poi);
        
        if (paths.empty() || paths[0].cost == std::numeric_limits<float>::max()) {
            continue;  // Pas de chemin valide
        }
        
        std::vector<osmium::object_id_type> temp = solution;
        temp.push_back(poi);
        
        double new_distance = calculate_distance(temp);
        
        // Vérifier distance ET validité
        if (new_distance > max_distance_constraint || 
            new_distance == std::numeric_limits<double>::max()) {
            continue;
        }
        
        double improvement = calculate_fitness(temp) - calculate_fitness(solution);
        
        if (improvement > best_improvement) {
            best_improvement = improvement;
            best_poi = poi;
        }
    }
    
    if (best_poi != 0) {
        std::vector<osmium::object_id_type> result = solution;
        result.push_back(best_poi);
        return result;
    }
    
    return solution;
}

std::vector<osmium::object_id_type> VNSSolver::remove_worst_poi(
    const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.size() <= 2) return solution;
    
    size_t worst_idx = 0;
    double worst_ratio = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < solution.size(); i++) {
        double reward = get_poi_reward(solution[i]);
        double distance = 0.0;
        
        if (i > 0) distance += get_distance(solution[i-1], solution[i]);
        if (i < solution.size() - 1) distance += get_distance(solution[i], solution[i+1]);
        
        if (distance > 0) {
            double ratio = reward / distance;
            if (ratio < worst_ratio) {
                worst_ratio = ratio;
                worst_idx = i;
            }
        }
    }
    
    std::vector<osmium::object_id_type> result = solution;
    result.erase(result.begin() + worst_idx);
    
    return result;
}

std::vector<osmium::object_id_type> VNSSolver::or_opt(
    const std::vector<osmium::object_id_type>& solution,
    int segment_size) {
    
    if (solution.size() < segment_size + 2) return solution;
    
    std::vector<osmium::object_id_type> best = solution;
    double best_distance = calculate_distance(solution);
    
    for (size_t i = 0; i + segment_size <= solution.size(); i++) {
        for (size_t j = 0; j <= solution.size(); j++) {
            if (j >= i && j <= i + segment_size) continue;
            
            std::vector<osmium::object_id_type> candidate = solution;
            
            // Extraire le segment
            std::vector<osmium::object_id_type> segment(
                candidate.begin() + i, 
                candidate.begin() + i + segment_size
            );
            
            // Supprimer le segment
            candidate.erase(
                candidate.begin() + i, 
                candidate.begin() + i + segment_size
            );
            
            // Inserer ailleurs
            size_t insert_pos = (j > i) ? j - segment_size : j;
            candidate.insert(
                candidate.begin() + insert_pos, 
                segment.begin(), 
                segment.end()
            );
            
            double dist = calculate_distance(candidate);
            if (dist < best_distance && dist <= max_distance_constraint) {
                best = candidate;
                best_distance = dist;
            }
        }
    }
    
    return best;
}

// ============================================================================
// REPARATION
// ============================================================================
std::vector<osmium::object_id_type> VNSSolver::repair_solution(
    const std::vector<osmium::object_id_type>& solution) {
    
    std::vector<osmium::object_id_type> repaired = solution;
    
    while (!repaired.empty() && calculate_distance(repaired) > max_distance_constraint) {
        repaired = remove_worst_poi(repaired);
    }
    
    return repaired;
}

std::vector<osmium::object_id_type> VNSSolver::greedy_repair(
    const std::vector<osmium::object_id_type>& solution) {
    
    std::vector<osmium::object_id_type> repaired = solution;
    
    // D'abord reparer si trop long
    while (!repaired.empty() && calculate_distance(repaired) > max_distance_constraint) {
        repaired = remove_worst_poi(repaired);
    }
    
    // Ensuite essayer d'ajouter greedy
    std::unordered_set<osmium::object_id_type> in_solution(repaired.begin(), repaired.end());
    
    bool added = true;
    while (added) {
        added = false;
        
        osmium::object_id_type best_poi = 0;
        double best_ratio = -1.0;
        
        for (const auto& poi : all_pois) {
            if (in_solution.count(poi)) continue;
            
            std::vector<osmium::object_id_type> temp = repaired;
            temp.push_back(poi);
            
            double new_distance = calculate_distance(temp);
            if (new_distance > max_distance_constraint) continue;
            
            double added_distance = new_distance - calculate_distance(repaired);
            int added_reward = get_poi_reward(poi);
            
            if (added_distance > 0) {
                double ratio = static_cast<double>(added_reward) / added_distance;
                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_poi = poi;
                }
            }
        }
        
        if (best_poi != 0) {
            repaired.push_back(best_poi);
            in_solution.insert(best_poi);
            added = true;
        }
    }
    
    return repaired;
}

// ============================================================================
// EVALUATION
// ============================================================================
double VNSSolver::calculate_fitness(const std::vector<osmium::object_id_type>& solution) {
    double fitness = 0.0;
    for (const auto& poi : solution) {
        fitness += get_poi_reward(poi);
    }
    return fitness;
}

double VNSSolver::calculate_distance(const std::vector<osmium::object_id_type>& solution) {
    if (solution.size() < 2) return 0.0;
    
    double total = 0.0;
    
    for (size_t i = 0; i < solution.size() - 1; ++i) {
        std::vector<Path> paths = global_memory.check_path(solution[i], solution[i + 1]);
        
        // ✓ VÉRIFICATION : Le chemin existe-t-il vraiment ?
        if (paths.empty() || paths[0].cost == std::numeric_limits<float>::max()) {
            // Chemin invalide → distance infinie
            return std::numeric_limits<double>::max();
        }
        
        total += static_cast<double>(paths[0].cost);
    }
    
    return total;
}

bool VNSSolver::verify_solution_sequence(const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.size() < 2) return true;
    
    std::cout << "\n[VERIFICATION] Vérification séquence solution:" << std::endl;
    
    bool all_valid = true;
    double total_distance = 0.0;
    
    for (size_t i = 0; i < solution.size() - 1; i++) {
        osmium::object_id_type from = solution[i];
        osmium::object_id_type to = solution[i + 1];
        
        // Vérifier qu'il existe un chemin
        std::vector<Path> paths = global_memory.check_path(from, to);
        
        if (paths.empty() || paths[0].cost == std::numeric_limits<float>::max()) {
            std::cout << "  ✗ Segment " << i << " (" << from << " → " << to << "): PAS DE CHEMIN" << std::endl;
            all_valid = false;
        } else {
            double dist = static_cast<double>(paths[0].cost);
            total_distance += dist;
            
            std::cout << "  ✓ Segment " << i << " (" << from << " → " << to 
                      << "): " << static_cast<int>(dist) << "m" << std::endl;
        }
    }
    
    std::cout << "  Distance totale: " << static_cast<int>(total_distance) << "m" << std::endl;
    std::cout << "  Contrainte: " << static_cast<int>(max_distance_constraint) << "m" << std::endl;
    
    if (total_distance > max_distance_constraint) {
        std::cout << "  ✗ CONTRAINTE VIOLÉE !" << std::endl;
        all_valid = false;
    } else {
        std::cout << "  ✓ Contrainte respectée" << std::endl;
    }
    
    if (all_valid) {
        std::cout << "[OK] Solution valide : séquence continue POI→POI" << std::endl;
    } else {
        std::cout << "[ERREUR] Solution INVALIDE !" << std::endl;
    }
    
    return all_valid;
}

bool VNSSolver::is_feasible(const std::vector<osmium::object_id_type>& solution) {
    
    if (solution.empty()) return true;
    if (solution.size() == 1) return true;
    
    // ✓ VÉRIFICATION 1 : Tous les segments POI→POI ont un chemin
    for (size_t i = 0; i < solution.size() - 1; i++) {
        std::vector<Path> paths = global_memory.check_path(solution[i], solution[i + 1]);
        
        if (paths.empty() || paths[0].cost == std::numeric_limits<float>::max()) {
            return false;  // Segment invalide
        }
    }
    
    // ✓ VÉRIFICATION 2 : Distance totale ≤ contrainte
    double dist = calculate_distance(solution);
    
    if (dist == std::numeric_limits<double>::max()) {
        return false;  // Chemin invalide détecté
    }
    
    return dist <= max_distance_constraint;
}

double VNSSolver::get_distance(osmium::object_id_type node1, osmium::object_id_type node2) {
    std::vector<Path> paths = global_memory.check_path(node1, node2);
    if (paths.empty() || paths[0].cost == std::numeric_limits<float>::max()) {
        return std::numeric_limits<double>::max();
    }
    return static_cast<double>(paths[0].cost);
}

int VNSSolver::get_poi_reward(osmium::object_id_type poi_id) {
    auto node_it = geo_box.data.nodes.find(poi_id);
    if (node_it == geo_box.data.nodes.end()) return 0;
    
    int reward = 0;
    for (const int group_id : node_it->second.groupes) {
        // ✓ CORRECTION : Accéder à characteristics[group_id]
        if (group_id >= 0 && group_id < static_cast<int>(characteristics.size())) {
            reward += characteristics[group_id];  // ← ACCÈS PAR INDEX, pas find()
        }
    }
    
    return reward;
}

// ============================================================================
// AFFICHAGE
// ============================================================================
void VNSSolver::print_iteration(int iteration, int k, double fitness, int num_pois, double distance) {
    std::cout << "Iter " << std::setw(3) << iteration 
              << " | k=" << k 
              << " | Fitness: " << std::setw(6) << static_cast<int>(fitness)
              << " | POIs: " << std::setw(3) << num_pois
              << " | Dist: " << std::setw(5) << static_cast<int>(distance) << "m"
              << std::endl;
}

void VNSSolver::print_improvement(const std::string& operator_name, double old_fitness, double new_fitness) {
    double improvement = ((new_fitness - old_fitness) / old_fitness) * 100.0;
    std::cout << "  [OK] " << operator_name << ": " 
              << static_cast<int>(old_fitness) << " -> " 
              << static_cast<int>(new_fitness) 
              << " (+" << std::fixed << std::setprecision(1) << improvement << "%)"
              << std::endl;
}

// ============================================================================
// SOLVE - ALGORITHME PRINCIPAL VNS
// ============================================================================
VNSResult VNSSolver::solve(
    const std::vector<int>& agent_characteristics,
    double distance_constraint,
    const VNSParams& params) {

    auto start_time = std::chrono::high_resolution_clock::now();

    VNSResult result;
    characteristics = agent_characteristics;
    max_distance_constraint = distance_constraint;

    std::cout << "\n======================================================" << std::endl;
    std::cout << "|         VNS AGGRESSIVE - DEBUG MODE                |" << std::endl;
    std::cout << "======================================================" << std::endl;

    collect_rewardable_pois(agent_characteristics);

    if (all_pois.size() < 2) {
        std::cout << "[WARN] Pas assez de POIs" << std::endl;
        result.is_valid = false;
        return result;
    }

    std::cout << "\nCONFIG: " << all_pois.size() << " POIs, " 
              << static_cast<int>(distance_constraint) << "m max" << std::endl;

    // ========================================
    // SOLUTION INITIALE
    // ========================================
    std::cout << "\n--- CONSTRUCTION INITIALE ---" << std::endl;
    current_solution = generate_initial_solution();
    best_solution = current_solution;
    best_fitness = calculate_fitness(best_solution);
    
    std::cout << "[OK] Solution initiale: Fitness=" << static_cast<int>(best_fitness) 
              << " | POIs=" << best_solution.size() 
              << " | Dist=" << static_cast<int>(calculate_distance(best_solution)) << "m"
              << std::endl;
    
    // [OK] BACKUP pour diagnostic
    std::vector<osmium::object_id_type> initial_backup = best_solution;

    std::cout << "\n--- DEBUT VNS ---" << std::endl;
    std::cout << "Format: Iter | k | Action -> Fitness (POIs) | Distance" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    int k = 1;
    int iteration = 0;
    int improvements = 0;
    int consecutive_no_improve = 0;
    
    while (iteration < params.max_iterations) {
        iteration++;
        
        double old_fitness = best_fitness;
        
        // ========================================
        // SHAKE
        // ========================================
        std::vector<osmium::object_id_type> shaken = shake(best_solution, k);
        double shaken_fitness = calculate_fitness(shaken);
        
        // [OK] LOG SHAKE
        std::cout << "Iter " << std::setw(3) << iteration 
                  << " | k=" << k 
                  << " | SHAKE -> " << static_cast<int>(shaken_fitness)
                  << " (" << shaken.size() << " POIs)"
                  << " | Delta=" << static_cast<int>(best_fitness - shaken_fitness)
                  << std::endl;
        
        // ========================================
        // VND
        // ========================================
        std::vector<osmium::object_id_type> improved = variable_neighborhood_descent(shaken, params);
        double improved_fitness = calculate_fitness(improved);
        
        // [OK] LOG VND
        std::cout << "        |    | VND   -> " << static_cast<int>(improved_fitness)
                  << " (" << improved.size() << " POIs)"
                  << " | " << static_cast<int>(calculate_distance(improved)) << "m"
                  << std::endl;
        
        // ========================================
        // ACCEPTATION
        // ========================================
        if (improved_fitness > best_fitness) {
            improvements++;
            consecutive_no_improve = 0;
            
            double gain = improved_fitness - best_fitness;
            double gain_pct = (gain / best_fitness) * 100.0;
            
            best_solution = improved;
            best_fitness = improved_fitness;
            k = 1;
            
            // [OK] LOG AMELIORATION
            std::cout << "        |    | [OK] ACCEPT +" << static_cast<int>(gain)
                      << " (+" << std::fixed << std::setprecision(2) << gain_pct << "%)"
                      << " | NEW BEST = " << static_cast<int>(best_fitness)
                      << std::endl;
            std::cout << std::string(80, '-') << std::endl;
            
        } else {
            consecutive_no_improve++;
            k++;
            
            if (k > params.k_max) {
                k = 1;
            }
            
            // [OK] LOG REJET (seulement si verbose)
            if (iteration % 20 == 0) {
                std::cout << "        |    | [X] Reject (no improve x " 
                          << consecutive_no_improve << ")" << std::endl;
            }
        }
        
        // ========================================
        // AFFICHAGE PERIODIQUE
        // ========================================
        if (iteration % 20 == 0) {
            std::cout << "\n[Iter " << iteration << "] Best=" 
                      << static_cast<int>(best_fitness)
                      << " | Improvements=" << improvements
                      << " | No-improve=" << consecutive_no_improve
                      << std::endl;
            std::cout << std::string(80, '-') << std::endl;
        }
    }

    std::cout << "\n--- FIN VNS ---" << std::endl;
    std::cout << "Raison arret: ";
    if (iteration >= params.max_iterations) {
        std::cout << "Max iterations atteintes" << std::endl;
    } else {
        std::cout << consecutive_no_improve << " iterations sans amelioration" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // ========================================
    // RESULTAT
    // ========================================
    result.solution = best_solution;
    result.fitness = best_fitness;
    result.distance = calculate_distance(best_solution);
    result.num_pois = best_solution.size();
    result.execution_time_ms = duration.count();
    result.is_valid = !best_solution.empty() && result.distance <= max_distance_constraint;

    // ========================================
    // STATISTIQUES FINALES
    // ========================================
    double initial_fitness = calculate_fitness(initial_backup);
    double improvement_pct = ((best_fitness - initial_fitness) / initial_fitness) * 100.0;

    std::cout << "\n======================================================" << std::endl;
    std::cout << "|               RESULTAT FINAL VNS                   |" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "|  Initiale: " << std::setw(37) << static_cast<int>(initial_fitness) << " |" << std::endl;
    std::cout << "|  Finale:   " << std::setw(37) << static_cast<int>(result.fitness) << " |" << std::endl;
    std::cout << "|  Amelioration: " << std::setw(27) << std::fixed << std::setprecision(2) 
              << improvement_pct << "% |" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "|  POIs: " << std::setw(41) << result.num_pois << " |" << std::endl;
    std::cout << "|  Distance: " << std::setw(31) << static_cast<int>(result.distance) << "m |" << std::endl;
    std::cout << "|  Temps: " << std::setw(36) << result.execution_time_ms << " ms |" << std::endl;
    std::cout << "|  Iterations: " << std::setw(33) << iteration << " |" << std::endl;
    std::cout << "|  Ameliorations: " << std::setw(30) << improvements << " |" << std::endl;
    std::cout << "======================================================" << std::endl;

    // ========================================
    // ANALYSE
    // ========================================
    std::cout << "\n--- ANALYSE ---" << std::endl;
    
    if (improvements == 0) {
        std::cout << "[ERREUR] PROBLEME: Aucune amelioration !" << std::endl;
        std::cout << "   Causes possibles:" << std::endl;
        std::cout << "   1. Solution initiale deja optimale" << std::endl;
        std::cout << "   2. Shake trop faible" << std::endl;
        std::cout << "   3. VND ne trouve pas de voisins meilleurs" << std::endl;
    } else if (improvement_pct < 1.0) {
        std::cout << "[WARN] Amelioration minime (< 1%)" << std::endl;
        std::cout << "   -> Solution initiale deja tres bonne" << std::endl;
    } else if (improvement_pct < 5.0) {
        std::cout << "[OK] Amelioration faible mais acceptable" << std::endl;
    } else {
        std::cout << "[OK][OK] Amelioration significative !" << std::endl;
    }
    
    // Usage distance
    double usage = (result.distance / max_distance_constraint) * 100.0;
    std::cout << "\nUtilisation distance: " << std::fixed << std::setprecision(1) 
              << usage << "%" << std::endl;
    
    if (usage > 98.0) {
        std::cout << "   [OK] Solution SATUREE (contrainte distance)" << std::endl;
    } else if (usage < 70.0) {
        std::cout << "   [WARN] Utilisation faible - peut ajouter plus de POIs ?" << std::endl;
    }
    
    // Usage POIs
    double poi_usage = (result.num_pois * 100.0) / all_pois.size();
    std::cout << "Utilisation POIs: " << std::fixed << std::setprecision(1) 
              << poi_usage << "% (" << result.num_pois << "/" << all_pois.size() << ")" << std::endl;
    
    if (poi_usage == 100.0) {
        std::cout << "   [OK] TOUS les POIs utilises !" << std::endl;
    }

    return result;
}