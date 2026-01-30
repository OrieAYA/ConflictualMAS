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
    for (const auto& [group_id, reward] : characteristics) {
        for (const auto& node_id : geo_box.data.objective_groups[group_id].node_ids) {
            all_pois.push_back(node_id);
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
            //delete agent;
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

    for(Agent* a : agent_population){
        std::cout << "Agent best fitness : " << objective_function(a->pbest) << std::endl;
        if (objective_function(a->pbest) < min_fitness) {
            agent_population.erase(find(agent_population.begin(), agent_population.end(), a));
            std::cout << "Agent discarded : " << a << std::endl;
            delete a;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "FIN RECHERCHE META-AGENT\n";
    std::cout << "Temps d'exécution: " << duration.count() << " ms\n";
    
    // Distribution des fitness
    if (!validated_pbest.empty()) {
        std::vector<std::pair<double, size_t>> fitness_with_index;
        
        for (size_t i = 0; i < validated_pbest.size(); i++) {
            const auto& pbest = validated_pbest[i];
            auto it = local_visited_solutions.find(pbest);
            if (it != local_visited_solutions.end()) {
                fitness_with_index.push_back({static_cast<double>(it->second), i});
            }
        }
        
        std::sort(fitness_with_index.begin(), fitness_with_index.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        double min_fitness = fitness_with_index.back().first;
        double max_fitness = fitness_with_index.front().first;
        
        for (size_t i = 0; i < fitness_with_index.size(); i++) {
            double fit = fitness_with_index[i].first;
            size_t agent_idx = fitness_with_index[i].second;
            double percent_of_gbest = (fit / local_visited_solutions[gbest_solution]) * 100.0f;
        }
    }
    
    return gbest_solution;
}