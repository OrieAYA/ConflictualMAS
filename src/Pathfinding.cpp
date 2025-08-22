#include "Pathfinding.hpp"
#include "Box.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <limits>

// ADDED: Static member definition
std::mutex Pathfinder::geobox_modification_mutex;

// Constructeur
Pathfinder::Pathfinder(GeoBox& box) : geo_box(box) {}

// FIXED: Proper method implementation with correct signature
void Pathfinder::update_way_group_threadsafe(osmium::object_id_type way_id, int new_group) {
    std::lock_guard<std::mutex> lock(geobox_modification_mutex);
    update_way_group(way_id, new_group);
}

// Version thread-safe de Subgraph_construction
bool Pathfinder::Subgraph_construction_threadsafe(
    Pathfinder& PfSystem,
    std::vector<osmium::object_id_type> objective_nodes,
    int path_group) {

        if(objective_nodes.size() < 2){
            return false;
        }

        osmium::object_id_type start_node = objective_nodes[0];
        osmium::object_id_type act_n = objective_nodes[0];

        std::unordered_map<osmium::object_id_type, bool> visited;

        for(const auto& init_node : objective_nodes){
            visited[init_node] = false;
        }

        while(objective_nodes.size() > 1){

            visited[act_n] = true;
            auto it = std::find(objective_nodes.begin(), objective_nodes.end(), act_n);
            if (it != objective_nodes.end()) {
                objective_nodes.erase(it);
            }

            float nearest_path_length = std::numeric_limits<float>::max();
            std::vector<osmium::object_id_type> nearest_path = {};
            osmium::object_id_type nearest_node = 0;

            for(const auto& tar_n : objective_nodes){
                if(!visited[tar_n]){

                    // A* Search est thread-safe car il ne modifie que des variables locales
                    std::vector<osmium::object_id_type> actual_path = PfSystem.A_Star_Search(act_n, tar_n);
                    float actual_path_length = 0.0f;

                    // Lecture thread-safe des distances (pas de modification)
                    for(const auto& act_path_way : actual_path){
                        auto way_it = PfSystem.geo_box.data.ways.find(act_path_way);
                        if (way_it != PfSystem.geo_box.data.ways.end()) {
                            actual_path_length += way_it->second.distance_meters;
                        }
                    }

                    if(actual_path_length < nearest_path_length){
                        nearest_path_length = actual_path_length;
                        nearest_path = actual_path;
                        nearest_node = tar_n;
                    }
                }
            }
            
            // Modification thread-safe des groupes de ways
            for(const auto& way_id : nearest_path){
                PfSystem.update_way_group_threadsafe(way_id, path_group);
            }

            act_n = nearest_node;
        }

        // Fermeture du cycle avec protection mutex
        std::vector<osmium::object_id_type> final_path = PfSystem.A_Star_Search(act_n, start_node);
        for(const auto& way_id : final_path){
            PfSystem.update_way_group_threadsafe(way_id, path_group);
        }
    
        return true;
}

bool Pathfinder::Subgraph_construction(
    Pathfinder& PfSystem,
    std::vector<osmium::object_id_type> objective_nodes,
    int path_group) {

        if(objective_nodes.size()<2){
            return false;
        }

        osmium::object_id_type start_node = objective_nodes[0];
        osmium::object_id_type act_n = objective_nodes[0];

        std::unordered_map<osmium::object_id_type, bool> visited;

        for(const auto& init_node : objective_nodes){
            visited[init_node] = false;
        }

        while(objective_nodes.size()>1){

            visited[act_n] = true;
            auto it = std::find(objective_nodes.begin(), objective_nodes.end(), act_n);
            if (it != objective_nodes.end()) {
                objective_nodes.erase(it);
            }

            float nearest_path_length = std::numeric_limits<float>::max();
            std::vector<osmium::object_id_type> nearest_path = {};
            osmium::object_id_type nearest_node = 0;

            for(const auto& tar_n : objective_nodes){
                if(!visited[tar_n]){

                    std::vector<osmium::object_id_type> actual_path = PfSystem.A_Star_Search(act_n, tar_n);
                    float actual_path_length = 0.0f;

                    for(const auto& act_path_way : actual_path){
                        actual_path_length = actual_path_length + PfSystem.geo_box.data.ways[act_path_way].distance_meters;
                    }

                    if(actual_path_length < nearest_path_length){
                        nearest_path_length = actual_path_length;
                        nearest_path = actual_path;
                        nearest_node = tar_n;
                    }
                }
            }
            
            for(const auto& way_id : nearest_path){
                PfSystem.update_way_group(way_id, path_group);
            }

            act_n = nearest_node;
        }

        std::vector<osmium::object_id_type> final_path = PfSystem.A_Star_Search(act_n, start_node);
        for(const auto& way_id : final_path){
            PfSystem.update_way_group(way_id, path_group);
        }
    
        return true;

    }

bool Pathfinder::Connected_subgraph_methode(Pathfinder& PfSystem,
    const std::vector<osmium::object_id_type>& objective_nodes,
    int path_group) {

        std::unordered_map<osmium::object_id_type, bool> visited;

        for(const auto& init_node : objective_nodes){
            visited[init_node] = false;
        }

        for(const auto& act_n : objective_nodes){

            float nearest_path_length = std::numeric_limits<float>::max();
            std::vector<osmium::object_id_type> nearest_path = {};
            osmium::object_id_type nearest_node = 0;

            for(const auto& tar_n : objective_nodes){
                if(act_n != tar_n && !visited[tar_n]){

                    std::vector<osmium::object_id_type> actual_path = PfSystem.A_Star_Search(act_n, tar_n);
                    float actual_path_length = 0.0f;

                    for(const auto& act_path_way : actual_path){
                        actual_path_length = actual_path_length + PfSystem.geo_box.data.ways[act_path_way].distance_meters;
                    }

                    if(actual_path_length < nearest_path_length){
                        nearest_path_length = actual_path_length;
                        nearest_path = actual_path;
                        nearest_node = tar_n;
                    }
                }
            }
            
            for(const auto& way_id : nearest_path){
                PfSystem.update_way_group(way_id, path_group);
            }

            visited[nearest_node] = true;
        }
    
        return true;
}

// Construction du graphe à partir de la GeoBox
std::vector<osmium::object_id_type> Pathfinder::A_Star_Search(
    const osmium::object_id_type& start_point,
    const osmium::object_id_type& end_point) {
        std::vector<osmium::object_id_type> open = {start_point};

        std::unordered_map<osmium::object_id_type, std::pair<osmium::object_id_type, osmium::object_id_type>> cameFrom;

        std::unordered_map<osmium::object_id_type, float> GScore;
        GScore[start_point] = 0.0f;

        std::unordered_map<osmium::object_id_type, float> FScore;
        FScore[start_point] = heuristic(start_point, end_point);

        osmium::object_id_type actual_node = start_point;
        osmium::object_id_type neighbor = start_point;

        float tentative_gScore = 0.0f;

        while (!open.empty()){

            osmium::object_id_type actual_node = open[0];
            size_t best_index = 0;
            
            for (size_t i = 1; i < open.size(); ++i) {
                if (FScore.count(open[i]) && FScore.count(actual_node)) {
                    if (FScore[open[i]] < FScore[actual_node]) {
                        actual_node = open[i];
                        best_index = i;
                    }
                }
            }

            if(actual_node == end_point){
                return reconstruct_path(cameFrom, actual_node);
            }

            open.erase(open.begin() + best_index);

            for(const auto& way_id : geo_box.data.nodes[actual_node].incident_ways){
                auto& way = geo_box.data.ways[way_id];
                if(way.node1_id == actual_node){
                    neighbor = way.node2_id;
                } else if(way.node2_id == actual_node) {
                    neighbor = way.node1_id;
                } else {
                    continue; // Way struct Error
                }

                tentative_gScore = GScore[actual_node] + way.distance_meters;
                if(!GScore.count(neighbor) || tentative_gScore < GScore[neighbor]){
                    cameFrom[neighbor] = std::make_pair(actual_node, way_id);
                    GScore[neighbor] = tentative_gScore;
                    FScore[neighbor] = tentative_gScore + heuristic(neighbor, end_point);
                    open.push_back(neighbor);
                }
            }
            
        }

        return {};
}

std::vector<osmium::object_id_type> Pathfinder::reconstruct_path(
    const std::unordered_map<osmium::object_id_type, std::pair<osmium::object_id_type, osmium::object_id_type>>& cameFrom,
    osmium::object_id_type actual_node) {
    
    std::vector<osmium::object_id_type> way_path;
    
    // Collecter les ways en ordre inverse
    while(cameFrom.count(actual_node)) {
        auto parent_node = cameFrom.at(actual_node).first;
        auto way_id = cameFrom.at(actual_node).second;
        
        way_path.push_back(way_id);
        actual_node = parent_node;
    }
    
    // Inverser pour avoir l'ordre correct (start → end)
    std::reverse(way_path.begin(), way_path.end());
    
    return way_path;
}

float Pathfinder::heuristic(osmium::object_id_type act_node, osmium::object_id_type end_point) {
    return 0.0f;

    /*
    auto& act_n = geo_box.data.nodes[act_node];
    auto& tar_n = geo_box.data.nodes[end_point];

    return calculate_haversine_distance(act_n.lat, act_n.lon, tar_n.lat, tar_n.lon);*/
}

// Méthode pour modifier le groupe d'un way
void Pathfinder::update_way_group(osmium::object_id_type way_id, int new_group) {
    auto it = geo_box.data.ways.find(way_id);
    if (it != geo_box.data.ways.end()) {
        it->second.groupe = new_group;
    } else {
        std::cerr << "Warning: Way " << way_id << " not found" << std::endl;
    }
}

// Calcul de distance entre deux nodes
double Pathfinder::calculate_distance(osmium::object_id_type node1, osmium::object_id_type node2) {
    
    return 0.0;
}

// Trouver le node le plus proche d'une coordonnée
osmium::object_id_type Pathfinder::find_nearest_node(double lat, double lon) {
    
    return 0;
}