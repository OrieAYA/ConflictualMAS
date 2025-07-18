#include "Pathfinding.hpp"
#include "Box.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// Constructeur
Pathfinder::Pathfinder(const GeoBox& box) : geo_box(box) {
    build_graph();
}

// Construction du graphe à partir de la GeoBox
std::vector<osmium::object_id_type> Pathfinder::A_Star_Search(
    const MyData::Point& start_point,
    const MyData::Point& end_point,
    int path_group) {
        std::vector<osmium::object_id_type> path;

        

        return path;
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