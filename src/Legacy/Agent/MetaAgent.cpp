#include "MetaAgent.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <limits>

MetaAgent::MetaAgent(
    GeoBox& box,
    Pathfinder& pf,
    GlobalMemory& mem,
    std::map<int,double> agent_characteristics,
    MetaAgentParams parameters
) : geo_box(box),
    pathfinder(pf),
    global_memory(mem),
    characteristics(agent_characteristics),
    params(parameters),
    gbest_solution()
{
    std::random_device rd;
    rng.seed(rd());
    
    collect_all_pois();
    update_max_reward_per_meter();  // Initialisation
}

void MetaAgent::update_max_reward_per_meter() {
    max_reward_per_meter = 0.0f;
    
    // Calculer reward_per_meter adaptatif selon characteristics de CET agent
    for (const auto& [group_id, reward] : characteristics) {
        if (reward <= 0.0) continue;

        if (group_id < global_memory.avg_distance_by_group.size() &&
            global_memory.count_by_group[group_id] > 0)
        {
            float avg_distance =
                static_cast<float>(global_memory.avg_distance_by_group[group_id]);

            // Sécurité anti division par 0
            if (avg_distance > 0.0f) {
                float density = reward / avg_distance;
                max_reward_per_meter = std::max(max_reward_per_meter, density);
            }
        }
    }
    
    // Fallback
    if(max_reward_per_meter == 0.0f) {
        max_reward_per_meter = 2.0f;
    }
}

MetaAgent::~MetaAgent() {
    for (auto* agent : agent_population) {
        delete agent;
    }
}

double MetaAgent::objective_function(const Solution& sol) const {
    double val = 0;

    for (const auto& node : sol.POIs){
        auto node_it = geo_box.data.nodes.find(node);
        if(node_it != geo_box.data.nodes.end()){
            for (const int& group_id : node_it->second.groupes){
                val += characteristics.find(group_id)->second;
            }
        }
    }

    return val;
}

void MetaAgent::collect_all_pois() {
    std::unordered_set<osmium::object_id_type> seen;
    for (const auto& [group_id, reward] : characteristics) {
        if (reward <= 0.0) continue;  // Ignorer les groupes sans reward
        
        auto group_it = geo_box.data.objective_groups.find(group_id);
        if (group_it == geo_box.data.objective_groups.end()) continue;
        
        for (const auto& node_id : group_it->second.node_ids) {
            if (seen.insert(node_id).second) {  // Insérer seulement si pas déjà présent
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

bool MetaAgent::try_add_agent() {
    
    update_max_reward_per_meter();
    
    osmium::object_id_type starting_poi = select_random_unused_poi();
    used_starting_pois.insert(starting_poi);
    if (starting_poi == 0) {
        return false;
    }
    
    // Agent Creation
    std::vector<osmium::object_id_type> init_poi = {starting_poi};
    Agent* agent = new Agent(
        geo_box, 
        pathfinder, 
        global_memory,
        init_poi,
        this
    );
    
    // Agent Search
    Solution solution = agent->agent_search(params.max_iterations_per_agent, params.tabu_list_size);
    double fitness = objective_function(solution);

    std::cout << "Agent Fitness : " << fitness << std::endl;

    if (!validated_pbest.empty()) {
        if (fitness < min_fitness) {
            delete agent;
            return false;
        }
    }
    
    agent_population.push_back(agent);
    validated_pbest.push_back(solution);
    local_visited_solutions[validated_pbest.back()] = static_cast<double>(fitness);
    
    if (fitness > local_visited_solutions[gbest_solution]) {
        gbest_solution = solution;
        min_fitness = local_visited_solutions[gbest_solution] * (1 - params.max_divergence_from_gbest);
        std::cout << "Min Fitness : " << min_fitness << std::endl;
    }
    
    return true;
}

void MetaAgent::print_statistics() {
    std::cout << "  [Stats] Agents: " << agent_population.size() 
              << " | GBest: " << local_visited_solutions[gbest_solution] << "\n";
}

Solution MetaAgent::run_meta_search() {
    auto start = std::chrono::high_resolution_clock::now();
    
    int consecutive_failures = 0;
    int max_failures = 10;
    
    // Boucle d'ajout d'agents
    while (used_starting_pois.size() < all_pois.size()) {
        
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
    std::cout << "Agents créés: " << agent_population.size() << "\n";
    std::cout << "Solutions validées: " << validated_pbest.size() << "\n";
    
    return gbest_solution;
}