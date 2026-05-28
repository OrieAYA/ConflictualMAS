#include "Pathfinding.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <limits>
#include <queue>
#include <functional>

// Static member definition
std::mutex Pathfinder::geobox_modification_mutex;

// Constructeur
Pathfinder::Pathfinder(GeoBox& box) : geo_box(box) {}

// Thread-safe version of update_way_group
void Pathfinder::update_way_group_threadsafe(osmium::object_id_type way_id, int new_group) {
    std::lock_guard<std::mutex> lock(geobox_modification_mutex);
    update_way_group(way_id, new_group);
}

// ====================================================================
// MÉTHODES DE PATHFINDING PRINCIPALES
// ====================================================================

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

bool Pathfinder::Complete_subgraph_construction(
    Pathfinder& PfSystem,
    std::vector<osmium::object_id_type> objective_nodes,
    int path_group) {

    if(objective_nodes.size() < 2){
        return false;
    }

    osmium::object_id_type act_n;

    while(objective_nodes.size() > 1){

        act_n = objective_nodes[0];

        auto it = std::find(objective_nodes.begin(), objective_nodes.end(), act_n);
        if (it != objective_nodes.end()) {
            objective_nodes.erase(it);
        }

        for(const auto& tar_n : objective_nodes){
            std::vector<osmium::object_id_type> actual_path = PfSystem.A_Star_Search(act_n, tar_n);
            for(const auto& way_id : actual_path){
                PfSystem.update_way_group(way_id, path_group);
            }
        }
    }

    return true;
}

bool Pathfinder::Subgraph_construction(
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

bool Pathfinder::Connected_subgraph_methode(
    Pathfinder& PfSystem,
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

// ====================================================================
// ALGORITHMES DE RECHERCHE DE CHEMIN
// ====================================================================

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

std::vector<Path> Pathfinder::Neighbor_Search(
const osmium::object_id_type& start_point,
    const std::unordered_set<int>& available_groups,
    const std::unordered_set<osmium::object_id_type>& already_visited_pois) {

// Récupérer ou créer le cache pour ce POI
SearchCache& cache = search_caches[start_point];

    bool is_new_search = cache.found_pois.empty();
    bool found_new_poi = false;

    // Borne de distance pour empêcher l'exploration infinie du graphe.
    // Sans cette contrainte, si tous les POIs proches sont déjà visités,
    // le Dijkstra explore indéfiniment sans jamais poser found_new_poi = true.
    float length_constraint = std::numeric_limits<float>::max();

    if (is_new_search) {
        cache.GScore[start_point] = 0.0f;
        cache.open.push_back(start_point);
        cache.max_explored_distance = 0.0f;
    } else {
        // Continuation : borner à 3× le rayon déjà exploré (minimum 100m)
        length_constraint = cache.max_explored_distance * 3.0f;
        if (length_constraint < 100.0f) length_constraint = 100.0f;

        std::vector<osmium::object_id_type> nodes_to_reopen;
        for (const auto& [node_id, gscore] : cache.GScore) {
            if (gscore >= length_constraint) continue;  // Pas de rouvrir au-delà de la borne
            if (cache.closed.count(node_id)) {
                auto node_it = geo_box.data.nodes.find(node_id);
                if (node_it != geo_box.data.nodes.end()) {
                    for (const auto& way_id : node_it->second.incident_ways) {
                        auto way_it = geo_box.data.ways.find(way_id);
                        if (way_it == geo_box.data.ways.end()) continue;

                        auto& way = way_it->second;
                        osmium::object_id_type neighbor = (way.node1_id == node_id)
                            ? way.node2_id : way.node1_id;

                        float tentative_score = gscore + way.distance_meters;

                        if (tentative_score < length_constraint &&
                            (!cache.GScore.count(neighbor) || tentative_score < cache.GScore[neighbor])) {
                            nodes_to_reopen.push_back(node_id);
                            break;
                        }
                    }
                }
            }
        }

        for (const auto& node_id : nodes_to_reopen) {
            cache.closed.erase(node_id);
            if (std::find(cache.open.begin(), cache.open.end(), node_id) == cache.open.end()) {
                cache.open.push_back(node_id);
            }
        }
    }

int nodes_explored = 0;

    // DIJKSTRA - Continue jusqu'à trouver un nouveau POI
while (!cache.open.empty()) {
        // Si on a trouvé au moins un nouveau POI ET qu'on est en continuation, on peut arrêter
        if (found_new_poi) {
break;
}
        // Trouver le noeud avec le plus petit GScore
osmium::object_id_type actual_node = cache.open[0];
size_t best_index = 0;

for (size_t i = 1; i < cache.open.size(); ++i) {
if (cache.GScore.count(cache.open[i]) && cache.GScore.count(actual_node)) {
if (cache.GScore[cache.open[i]] < cache.GScore[actual_node]) {
actual_node = cache.open[i];
best_index = i;
}
}
}

cache.open.erase(cache.open.begin() + best_index);

if (cache.closed.count(actual_node)) {
continue;
}
cache.closed.insert(actual_node);

nodes_explored++;

cache.max_explored_distance = std::max(cache.max_explored_distance, cache.GScore[actual_node]);

auto node_it = geo_box.data.nodes.find(actual_node);
if (node_it == geo_box.data.nodes.end()) {
continue;
}

    // Vérifier si c'est un POI
    if (!node_it->second.groupes.empty() && actual_node != start_point) {
        std::vector<osmium::object_id_type> path_edges = reconstruct_path(cache.cameFrom, actual_node);
        float path_cost = cache.GScore[actual_node];

        Path new_path(path_edges, start_point, actual_node, path_cost);
        cache.found_pois.push_back(new_path);

        // Sur la première recherche, après le premier POI trouvé,
        // on borne à 5× cette distance pour collecter les POIs alentours
        if (cache.found_pois.size() == 1 && is_new_search) {
            length_constraint = path_cost * 5.0f;
        }

        bool is_new_for_solution = (already_visited_pois.find(actual_node) == already_visited_pois.end());

        if (is_new_for_solution) {
            for (const auto& i : available_groups) {
                if (node_it->second.groupes.find(i) != node_it->second.groupes.end()) {
                    found_new_poi = true;
                    break;
                }
            }
        }
    }

    // Ne pas explorer au-delà de la borne
    if (cache.GScore[actual_node] >= length_constraint) {
        continue;
    }

    // Explorer les voisins
for (const auto& way_id : node_it->second.incident_ways) {
auto way_it = geo_box.data.ways.find(way_id);
if (way_it == geo_box.data.ways.end()) {
continue;
}

auto& way = way_it->second;
osmium::object_id_type neighbor;

if (way.node1_id == actual_node) {
neighbor = way.node2_id;
} else if (way.node2_id == actual_node) {
neighbor = way.node1_id;
} else {
continue;
}

if (cache.closed.count(neighbor)) {
continue;
}

            float tentative_gScore = cache.GScore[actual_node] + way.distance_meters;

            if (tentative_gScore < length_constraint &&
                (!cache.GScore.count(neighbor) || tentative_gScore < cache.GScore[neighbor])) {
                cache.cameFrom[neighbor] = std::make_pair(actual_node, way_id);
                cache.GScore[neighbor] = tentative_gScore;
                
                if (std::find(cache.open.begin(), cache.open.end(), neighbor) == cache.open.end()) {
                    cache.open.push_back(neighbor);
                }
            }
        }
    }

    return cache.found_pois;
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
}

// ────────────────────────────────────────────────────────────────────────────
// Single-source Dijkstra — used by SoTA allocation rules
// ────────────────────────────────────────────────────────────────────────────
//
// Computes shortest-path distances from `source` to every reachable node.
// Returns an unordered_map node_id → distance (in metres along OSM way edges).
// O((V + E) log V) using binary heap; for OSM cities V ≈ 30k-800k.
//
// Why this exists: SoTA allocation rules (CongestionAware, FaithfulCA, etc.)
// need to evaluate cost(agent.current_node → task.pickup) for EVERY candidate
// agent at each task arrival. The naïve per-agent A* approach is O(N × E log V)
// per task, prohibitive when n_active is large (e.g. 60 agents in over_fleet).
// Replacing N A* calls with one reverse Dijkstra from the pickup yields the
// same information in one search.
std::unordered_map<osmium::object_id_type, float>
Pathfinder::dijkstra_distances_from(osmium::object_id_type source) {
    std::unordered_map<osmium::object_id_type, float> dist;

    auto src_it = geo_box.data.nodes.find(source);
    if (src_it == geo_box.data.nodes.end()) return dist;

    // Min-heap keyed by (distance, node). Use std::greater for min-heap.
    using QueueEntry = std::pair<float, osmium::object_id_type>;
    std::priority_queue<QueueEntry,
                        std::vector<QueueEntry>,
                        std::greater<QueueEntry>> pq;

    dist[source] = 0.0f;
    pq.push({0.0f, source});

    while (!pq.empty()) {
        const auto [d, u] = pq.top();
        pq.pop();

        // Stale heap entry: a shorter distance was already settled.
        auto du_it = dist.find(u);
        if (du_it == dist.end() || d > du_it->second) continue;

        auto node_it = geo_box.data.nodes.find(u);
        if (node_it == geo_box.data.nodes.end()) continue;

        for (const auto& way_id : node_it->second.incident_ways) {
            auto way_it = geo_box.data.ways.find(way_id);
            if (way_it == geo_box.data.ways.end()) continue;
            const auto& way = way_it->second;

            osmium::object_id_type v;
            if      (way.node1_id == u) v = way.node2_id;
            else if (way.node2_id == u) v = way.node1_id;
            else continue;  // malformed way

            const float nd = d + static_cast<float>(way.distance_meters);
            auto dv_it = dist.find(v);
            if (dv_it == dist.end() || nd < dv_it->second) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }

    return dist;
}

// ====================================================================
// MÉTHODES UTILITAIRES
// ====================================================================

void Pathfinder::update_way_group(osmium::object_id_type way_id, int new_group) {
    auto it = geo_box.data.ways.find(way_id);
    if (it != geo_box.data.ways.end()) {
        it->second.add_group(new_group);
    } else {
        std::cerr << "Warning: Way " << way_id << " not found" << std::endl;
    }
}

double Pathfinder::calculate_distance(osmium::object_id_type node1, osmium::object_id_type node2) {
    // À implémenter si nécessaire
    return 0.0;
}

osmium::object_id_type Pathfinder::find_nearest_node(double lat, double lon) {
    // À implémenter si nécessaire
    return 0;
}