#include "MetaAgent.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <limits>

// ✅ CORRECTION: Constructeur avec signature correcte
MetaAgent::MetaAgent(
    GeoBox& box,
    Pathfinder& pf,
    GlobalMemory& mem,
    std::vector<int> agent_characteristics,
    MetaAgentParams parameters
) : geo_box(box),
    pathfinder(pf),
    global_memory(mem),
    characteristics(agent_characteristics),
    params(parameters),
    gbest_solution(),  // ✅ Initialisation de la solution
    gbest_fitness(std::numeric_limits<int>::min())
{
    std::random_device rd;
    rng.seed(rd());
    
    collect_all_pois();
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "META-AGENT INITIALISÉ\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "POIs disponibles: " << all_pois.size() << "\n";
    std::cout << "Couverture cible: " << (params.coverage_rate * 100) << "%\n";
    std::cout << "Similarité max: " << (params.admissible_similarity_degree * 100) << "%\n";
    std::cout << std::string(70, '=') << "\n\n";
}

MetaAgent::~MetaAgent() {
    for (auto* agent : agent_population) {
        delete agent;
    }
}

void MetaAgent::collect_all_pois() {
    for (const auto& [node_id, node_data] : geo_box.data.nodes) {
        if (!node_data.groupes.empty()) {
            bool has_relevant_group = false;
            for (const int group_id : node_data.groupes) {
                if (group_id >= 0 && group_id < static_cast<int>(characteristics.size())) {
                    if (characteristics[group_id] > 0) {
                        has_relevant_group = true;
                        break;
                    }
                }
            }
            if (has_relevant_group) {
                all_pois.push_back(node_id);
            }
        }
    }
}

osmium::object_id_type MetaAgent::select_random_unused_poi() {
    std::vector<osmium::object_id_type> available;
    for (const auto& poi : all_pois) {
        if (used_starting_pois.find(poi) == used_starting_pois.end()) {
            available.push_back(poi);
        }
    }
    
    if (available.empty()) return 0;
    
    std::uniform_int_distribution<size_t> dist(0, available.size() - 1);
    return available[dist(rng)];
}

float MetaAgent::calculate_similarity(const Solution& sol1, const Solution& sol2) {
    if (sol1.POIs.size() < 2 || sol2.POIs.size() < 2) {
        return 0.0f;
    }
    
    std::set<std::pair<osmium::object_id_type, osmium::object_id_type>> edges1, edges2;
    
    for (size_t i = 1; i < sol1.POIs.size(); i++) {
        auto edge = std::minmax(sol1.POIs[i-1], sol1.POIs[i]);
        edges1.insert(edge);
    }
    
    for (size_t i = 1; i < sol2.POIs.size(); i++) {
        auto edge = std::minmax(sol2.POIs[i-1], sol2.POIs[i]);
        edges2.insert(edge);
    }
    
    int common_edges = 0;
    for (const auto& edge : edges1) {
        if (edges2.count(edge)) {
            common_edges++;
        }
    }
    
    float total_edges = (edges1.size() + edges2.size()) / 2.0f;
    return static_cast<float>(common_edges) / total_edges;
}

float MetaAgent::calculate_coverage_rate() {
    if (all_pois.empty()) return 0.0f;
    return static_cast<float>(used_starting_pois.size()) / static_cast<float>(all_pois.size());
}

bool MetaAgent::try_add_agent() {
    std::cout << "\n--- AGENT #" << (agent_population.size() + 1) << " ---\n";
    
    osmium::object_id_type starting_poi = select_random_unused_poi();
    if (starting_poi == 0) {
        std::cout << "  Plus de POIs disponibles\n";
        return false;
    }
    
    std::cout << "  POI départ: " << starting_poi << "\n";
    
    // Créer agent
    std::vector<osmium::object_id_type> init_poi = {starting_poi};
    Agent* agent = new Agent(
        geo_box, 
        pathfinder, 
        global_memory,
        characteristics, 
        init_poi,
        &local_visited_solutions,
        &validated_pbest,
        params.admissible_similarity_degree,
        10
    );
    
    // Recherche Tabu
    Solution solution = agent->tabu_search(params.max_iterations_per_agent, params.tabu_list_size);
    int fitness = agent->objective_function(solution);
    
    std::cout << "  Solution trouvée: " << solution.POIs.size() 
              << " POIs, fitness=" << fitness 
              << ", distance=" << solution.cost << "m\n";
    
    // Validation 1 : Au moins 2 POIs
    if (solution.POIs.size() < 2) {
        std::cout << "  ✗ REJETÉ (solution triviale: " << solution.POIs.size() << " POI)\n";
        delete agent;
        return false;
    }
    
    // Validation 2 : Vérification de similarité
    if (!validated_pbest.empty()) {
        bool too_similar = false;
        float max_similarity = 0.0f;
        
        for (const auto& pbest : validated_pbest) {
            float similarity = calculate_similarity(solution, pbest);
            max_similarity = std::max(max_similarity, similarity);
            
            if (similarity > params.admissible_similarity_degree) {
                std::cout << "  ✗ REJETÉ (similarité: " << (similarity * 100) 
                          << "% > " << (params.admissible_similarity_degree * 100) 
                          << "% max)\n";
                too_similar = true;
                break;
            }
        }
        
        if (too_similar) {
            delete agent;
            return false;
        }
        
        std::cout << "  [Similarité max: " << (max_similarity * 100) << "%]\n";
    }
    
    // Validation 3 : Intervalle d'acceptation [max(min_pbest, GBest × divergence), +∞[
    if (!validated_pbest.empty()) {
        int gbest_threshold = static_cast<int>(gbest_fitness * params.max_divergence_from_gbest);
        
        int min_pbest_fitness = gbest_fitness;
        for (const auto& pbest : validated_pbest) {
            auto it = local_visited_solutions.find(pbest);
            if (it != local_visited_solutions.end()) {
                int pbest_fit = static_cast<int>(it->second);
                min_pbest_fitness = std::min(min_pbest_fitness, pbest_fit);
            }
        }
        
        int min_acceptable_fitness = std::max(min_pbest_fitness, gbest_threshold);
        
        if (fitness < min_acceptable_fitness) {
            std::cout << "  ✗ REJETÉ (fitness insuffisante: " << fitness 
                      << " < " << min_acceptable_fitness << ")\n";
            delete agent;
            return false;
        }
        
        std::cout << "  [Fitness OK: " << fitness << " >= " << min_acceptable_fitness << "]\n";
    }
    
    // ACCEPTÉ
    std::cout << "  ✓ ACCEPTÉ\n";
    
    agent_population.push_back(agent);
    used_starting_pois.insert(starting_poi);
    validated_pbest.push_back(solution);
    local_visited_solutions[validated_pbest.back()] = static_cast<float>(fitness);
    
    // Mettre à jour GBest
    if (fitness > gbest_fitness) {
        gbest_solution = solution;  // ✅ Copie de la solution
        gbest_fitness = fitness;
        std::cout << "  ★★★ NOUVEAU GBEST: " << gbest_fitness 
                  << " (" << gbest_solution.POIs.size() << " POIs)\n";
    }
    
    return true;
}

void MetaAgent::print_statistics() {
    std::cout << "  [Stats] Agents: " << agent_population.size() 
              << " | Couverture: " << (calculate_coverage_rate() * 100) << "% "
              << " | GBest: " << gbest_fitness << "\n";
}

Solution MetaAgent::run_meta_search() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "DÉBUT RECHERCHE META-AGENT\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Paramètres:\n";
    std::cout << "  - Similarité max: " << (params.admissible_similarity_degree * 100) << "%\n";
    std::cout << "  - Borne inférieure: max(min_pbest, GBest × " 
              << params.max_divergence_from_gbest << ")\n";
    std::cout << "  - Couverture cible: " << (params.coverage_rate * 100) << "%\n";
    std::cout << std::string(70, '=') << "\n";
    
    int consecutive_failures = 0;
    int max_failures = 10;
    
    // Boucle d'ajout d'agents
    while (calculate_coverage_rate() < params.coverage_rate &&
           agent_population.size() < static_cast<size_t>(params.max_agents) &&
           used_starting_pois.size() < all_pois.size()) {
        
        bool added = try_add_agent();
        
        if (added) {
            consecutive_failures = 0;
        } else {
            consecutive_failures++;
            if (consecutive_failures >= max_failures) {
                std::cout << "\n[STOP] " << max_failures << " échecs consécutifs\n";
                break;
            }
        }
        
        print_statistics();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "RÉSULTAT FINAL META-AGENT\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Temps total: " << duration.count() << " secondes\n";
    std::cout << "Agents dans la population: " << agent_population.size() << "\n";
    std::cout << "GBest Fitness: " << gbest_fitness << "\n";
    std::cout << "GBest POIs: " << gbest_solution.POIs.size() << "\n";
    std::cout << "GBest Distance: " << gbest_solution.cost << " m / " 
              << global_memory.length_constraint << " m\n";
    
    // Distribution des fitness
    if (!validated_pbest.empty()) {
        std::cout << "\nDistribution des fitness dans la population:\n";
        std::vector<std::pair<int, size_t>> fitness_with_index;
        
        for (size_t i = 0; i < validated_pbest.size(); i++) {
            const auto& pbest = validated_pbest[i];
            auto it = local_visited_solutions.find(pbest);
            if (it != local_visited_solutions.end()) {
                fitness_with_index.push_back({static_cast<int>(it->second), i});
            }
        }
        
        std::sort(fitness_with_index.begin(), fitness_with_index.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        int min_fitness = fitness_with_index.back().first;
        int max_fitness = fitness_with_index.front().first;
        
        for (size_t i = 0; i < fitness_with_index.size(); i++) {
            int fit = fitness_with_index[i].first;
            size_t agent_idx = fitness_with_index[i].second;
            float percent_of_gbest = (static_cast<float>(fit) / gbest_fitness) * 100.0f;
            
            std::cout << "  Agent " << (agent_idx + 1) << ": " << fit 
                      << " (" << std::fixed << std::setprecision(1) 
                      << percent_of_gbest << "% de GBest)";
            
            if (i == 0) std::cout << " ★ GBest";
            else if (i == fitness_with_index.size() - 1) std::cout << " ▼ Min";
            std::cout << "\n";
        }
        
        std::cout << "\nIntervalle de fitness: [" << min_fitness << ", " << max_fitness << "]\n";
        std::cout << "Ratio max/min: " << std::fixed << std::setprecision(2) 
                  << (static_cast<float>(max_fitness) / min_fitness) << "×\n";
    }
    
    std::cout << std::string(70, '=') << "\n\n";
    
    return gbest_solution;  // ✅ Retourner la copie de la solution
}