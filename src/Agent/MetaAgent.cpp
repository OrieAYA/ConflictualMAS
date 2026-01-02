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
    gbest_solution(),
    gbest_fitness(std::numeric_limits<int>::min())
{
    std::random_device rd;
    rng.seed(rd());
    
    collect_all_pois();
}

MetaAgent::~MetaAgent() {
    for (auto* agent : agent_population) {
        delete agent;
    }
}

int MetaAgent::objective_function(const Solution& sol) {
    int val = 0;

    for (const auto& node : sol.POIs){
        auto node_it = geo_box.data.nodes.find(node);
        if(node_it != geo_box.data.nodes.end()){
            for (const int& group_id : node_it->second.groupes){
                val += characteristics[group_id];
            }
        }
    }

    return val;
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

float MetaAgent::calculate_coverage_rate() {
    if (all_pois.empty()) return 0.0f;
    return static_cast<float>(used_starting_pois.size()) / static_cast<float>(all_pois.size());
}

bool MetaAgent::try_add_agent() {
    
    osmium::object_id_type starting_poi = select_random_unused_poi();
    if (starting_poi == 0) {
        return false;
    }
    
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
    
    // Validation 1 : Au moins 2 POIs
    if (solution.POIs.size() < 2) {
        delete agent;
        return false;
    }
    
    // Validation 2 : Vérification de similarité
    if (!validated_pbest.empty()) {
        bool too_similar = false;
        float max_similarity = 0.0f;
        
        for (const auto& pbest : validated_pbest) {
            float similarity = global_memory.calculate_similarity(solution, pbest);
            max_similarity = std::max(max_similarity, similarity);
            
            if (similarity > params.admissible_similarity_degree) {
                too_similar = true;
                break;
            }
        }
        
        if (too_similar) {
            delete agent;
            return false;
        }
        
    }

    if (!validated_pbest.empty()) {
        if (fitness < min_fitness) {
            delete agent;
            return false;
        }
    }
    
    agent_population.push_back(agent);
    used_starting_pois.insert(starting_poi);
    validated_pbest.push_back(solution);
    local_visited_solutions[validated_pbest.back()] = static_cast<float>(fitness);
    
    if (fitness > gbest_fitness) {
        gbest_solution = solution;
        gbest_fitness = fitness;
        int gbest_threshold = static_cast<int>(gbest_fitness * (1 - params.max_divergence_from_gbest));
        if (min_fitness == 0) {
            min_fitness = gbest_threshold;
        } else {
            min_fitness = std::min(min_fitness, gbest_threshold);
        }
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
                break;
            }
        }
        
        print_statistics();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "FIN RECHERCHE META-AGENT\n";
    std::cout << "Temps d'exécution: " << duration.count() << " ms\n";
    
    // Distribution des fitness
    if (!validated_pbest.empty()) {
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
        }
    }
    
    return gbest_solution;
}