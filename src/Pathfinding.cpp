#include "Pathfinding.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// Constructeur
Pathfinder::Pathfinder(const GeoBox& box) : geo_box(box) {
    build_graph();
}

// Construction du graphe à partir de la GeoBox
void Pathfinder::build_graph() {

}

// Algorithme de colonie de fourmis
std::vector<osmium::object_id_type> Pathfinder::find_path_ant_colony(osmium::object_id_type start, 
                                                                    osmium::object_id_type end) {
    
    std::vector<osmium::object_id_type> path;
    return path;
}

// Calcul de distance entre deux nodes
double Pathfinder::calculate_distance(osmium::object_id_type node1, osmium::object_id_type node2) {
    
    return 0.0;
}

// Trouver le node le plus proche d'une coordonnée
osmium::object_id_type Pathfinder::find_nearest_node(double lat, double lon) {
    
    return 0;
}