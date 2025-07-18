#include "Pathfinding.hpp"
#include "Box.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>

// Constructeur
Pathfinder::Pathfinder(GeoBox& box) : geo_box(box) {}

// Construction du graphe à partir de la GeoBox
std::vector<osmium::object_id_type> Pathfinder::A_Star_Search(
    const osmium::object_id_type& start_point,
    const osmium::object_id_type& end_point,
    int path_group) {
        std::vector<osmium::object_id_type> open = {start_point};

        std::unordered_map<osmium::object_id_type, std::pair<osmium::object_id_type, osmium::object_id_type>> cameFrom;

        std::unordered_map<osmium::object_id_type, float> GScore;
        GScore[start_point] = 0.0f;

        std::unordered_map<osmium::object_id_type, float> FScore;
        FScore[start_point] = heuristic(start_point);

        osmium::object_id_type actual_node = start_point;
        osmium::object_id_type neighbor = start_point;

        float tentative_gScore = 0.0f;

        while (!open.empty()){
            for(const auto& [node_id, score] : FScore){
                if(score < FScore[actual_node]){
                    actual_node = node_id;
                }
            }

            if(actual_node == end_point){
                return reconstruct_path(cameFrom, actual_node, path_group);
            }

            auto it = std::find(open.begin(), open.end(), actual_node);
            if (it != open.end()) {
                open.erase(it);
            }

            for(const auto& way_id : geo_box.data.nodes[actual_node].incident_ways){
                auto& way = geo_box.data.ways[way_id];
                if(way.points[0].id == actual_node){
                    neighbor = way.points[1].id;
                }else{
                    neighbor = way.points[0].id;
                }

                tentative_gScore = GScore[actual_node] + way.distance_meters;
                if(tentative_gScore < GScore[neighbor]){
                    cameFrom[neighbor] = std::make_pair(actual_node, way_id);
                    GScore[neighbor] = tentative_gScore;
                    FScore[neighbor] = tentative_gScore + heuristic(neighbor);
                    if(std::find(open.begin(), open.end(), neighbor) == open.end()){
                        open.insert(open.begin(), neighbor);
                    }
                }
            }
            
        }

        return {};
}

std::vector<osmium::object_id_type> Pathfinder::reconstruct_path(
    const std::unordered_map<osmium::object_id_type, std::pair<osmium::object_id_type, osmium::object_id_type>>& cameFrom,
    osmium::object_id_type actual_node,
    int path_group) {

        std::vector<osmium::object_id_type> path = {actual_node};

        while(cameFrom.count(actual_node)){
            actual_node = cameFrom.at(actual_node).first;
            update_way_group(cameFrom.at(actual_node).second, path_group);
            path.insert(path.begin(), actual_node);
        }

        return path;

}

float Pathfinder::heuristic(osmium::object_id_type node) {
    const auto& point = geo_box.data.nodes[node];


    float lowest = geo_box.data.ways[point.incident_ways[0]].weight;

    for (const auto& way_id : point.incident_ways) {
        float way = geo_box.data.ways[way_id].weight;
        if (way < lowest){
            lowest = way;
        }
    }

    return lowest;
}

// Méthode pour modifier les weights des nodes
void Pathfinder::update_node_weight(osmium::object_id_type node_id, float new_weight) {
    auto it = geo_box.data.nodes.find(node_id);
    if (it != geo_box.data.nodes.end()) {
        it->second.weight = new_weight;
    } else {
        std::cerr << "Warning: node " << node_id << " not found" << std::endl;
    }
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