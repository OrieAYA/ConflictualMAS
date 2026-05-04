#ifndef MEMORY_HPP
#define MEMORY_HPP

#include "Environment/GeoBox/Box.hpp"
#include "Pathfinding.hpp"
#include <iostream>
#include <map>
#include <vector>
#include <set>

// Forward declaration
class Pathfinder;

struct Solution {
    std::vector<osmium::object_id_type> POIs;
    std::vector<Path> paths;
    float cost = 0.0;

    Solution() = default;
    Solution(std::vector<osmium::object_id_type> pois, std::vector<Path> p, float c) 
        : POIs(pois), paths(p), cost(c) {}
    
    void add_node(osmium::object_id_type POI, const Path& path_to_add){
        this->POIs.push_back(POI);
        this->paths.push_back(path_to_add);
        this->cost += path_to_add.cost;
    }

    void remove_node(){
        if(POIs.empty()) return;
        
        this->POIs.pop_back();
        
        if(!this->paths.empty()){
            this->cost -= this->paths.back().cost;
            this->paths.pop_back();
        }
    }
    
    bool empty() const {
        return POIs.empty();
    }
    
    // Opérateur < pour utiliser Solution comme clé dans std::map
    bool operator<(const Solution& other) const {
        if (POIs.size() != other.POIs.size()) {
            return POIs.size() < other.POIs.size();
        }
        return POIs < other.POIs;
    }
    
    // Opérateur == pour comparaison
    bool operator==(const Solution& other) const {
        return POIs == other.POIs;
    }
};

namespace std {
    template<>
    struct hash<Solution> {
        size_t operator()(const Solution& sol) const {
            size_t h = 0;
            for(const auto& poi : sol.POIs){
                h ^= std::hash<osmium::object_id_type>{}(poi) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };
}

struct GlobalMemory {
    GeoBox& geo_box;
    Pathfinder& PfSystem;
    std::map<osmium::object_id_type, std::map<osmium::object_id_type, std::vector<Path>>> visited_Paths;
    std::unordered_map<Path, std::unordered_map<Path, float, PathHash>, PathHash> Path_similarities;
    std::map<osmium::object_id_type, std::vector<osmium::object_id_type>> visited_Neighborhoods;

    float length_constraint;
    float search_coefficient;
    
    std::vector<double> total_distance_by_group;
    std::vector<int> count_by_group;
    std::vector<double> avg_distance_by_group;

    GlobalMemory() = delete;
    GlobalMemory(GeoBox& box, Pathfinder& pf) 
        : geo_box(box), PfSystem(pf) {}
    
    void initialize_group_stats() {

        total_distance_by_group.clear();
        count_by_group.clear();
        avg_distance_by_group.clear();

        int max_group_id = 0;

        for (const auto& [gid, _] : geo_box.data.objective_groups) {
            if (gid > max_group_id)
                max_group_id = gid;
        }

        if (max_group_id == 0) {
            std::cout << "[Memory] Aucun groupe trouvé\n";
            return;
        }

        total_distance_by_group.resize(max_group_id + 1, 0.0);
        count_by_group.resize(max_group_id + 1, 0);
        avg_distance_by_group.resize(max_group_id + 1, 0.0);

        std::cout << "[Memory] Group stats initialised for "
                << max_group_id << " groups\n";
    }


    // OPTIMISÉ: utilise unordered_set au lieu de std::find
    float check_path_similarity(const Path& path1, const Path& path2){
        auto it_a = Path_similarities.find(path1);
        if(it_a != Path_similarities.end()){
            auto it_b = it_a->second.find(path2);
            if(it_b != it_a->second.end()){
                return it_b->second;
            }
        }

        if ((path1.path_edges.empty() && path1.path_nodes.empty()) || 
            (path2.path_edges.empty() && path2.path_nodes.empty())) {
            Path_similarities[path1][path2] = 0.0f;
            Path_similarities[path2][path1] = 0.0f;
            return 0.0f;
        }

        // OPTIMISATION: unordered_set pour recherche O(1)
        std::unordered_set<osmium::object_id_type> edges1(
            path1.path_edges.begin(), 
            path1.path_edges.end()
        );
        
        int common_ways = 0;
        for (const auto& way : path2.path_edges) {
            if (edges1.count(way) > 0) {
                common_ways++;
            }
        }

        std::unordered_set<osmium::object_id_type> nodes1(
            path1.path_nodes.begin(), 
            path1.path_nodes.end()
        );
        
        int common_nodes = 0;
        for (const auto& node : path2.path_nodes) {
            if (nodes1.count(node) > 0) {
                common_nodes++;
            }
        }

        float similarity_degree = (path2.path_nodes.size() + path2.path_edges.size()) > 0 
            ? static_cast<float>(common_ways + common_nodes) / static_cast<float>(path2.path_nodes.size() + path2.path_edges.size()) 
            : 0.0f;

        Path_similarities[path1][path2] = similarity_degree;
        Path_similarities[path2][path1] = similarity_degree;
        return similarity_degree;
    }

    float calculate_solution_similarity(const Solution& sol1, const Solution& sol2){
        float similarity_degree = 0.0f;
        int similarity_count = 0;

        // OPTIMISÉ: utilise min pour éviter out of bounds
        size_t min_size = std::min(sol1.POIs.size(), sol2.POIs.size());
        for(size_t i = 0; i < min_size; ++i){
            if(sol1.POIs[i] == sol2.POIs[i]) similarity_count++;
        }

        size_t max_size = std::max(sol1.POIs.size(), sol2.POIs.size());
        if(max_size != 0){
            similarity_degree = static_cast<float>(similarity_count) / static_cast<float>(max_size);
        }

        return similarity_degree;
    }

    float calculate_similarity(const Solution& sol1, const Solution& sol2){
        if (sol1.paths.empty() || sol2.paths.empty()) {
            return 0.0f;
        }

        std::unordered_set<osmium::object_id_type> ways_sol1;
        std::unordered_set<osmium::object_id_type> nodes_sol1;
        int total_ways_sol1 = 0;
        int total_nodes_sol1 = 0;
        for (const auto& path : sol1.paths) {
            for (const auto& way_id : path.path_edges) {
                ways_sol1.insert(way_id);
                total_ways_sol1++;
            }
            for (const auto& node_id : path.path_nodes) {
                nodes_sol1.insert(node_id);
                total_nodes_sol1++;
            }
        }


        int common_ways = 0;
        int total_ways_sol2 = 0;

        int common_nodes = 0;
        int total_nodes_sol2 = 0;
        
        for (const auto& path : sol2.paths) {
            for (const auto& way_id : path.path_edges) {
                total_ways_sol2++;
                if (ways_sol1.count(way_id)) {
                    common_ways++;
                }
            }
            for (const auto& node_id : path.path_nodes) {
                total_nodes_sol2++;
                if (nodes_sol1.count(node_id)) {
                    common_nodes++;
                }
            }
        }
        
        float similarity_degree = total_ways_sol2 + total_nodes_sol2 > 0 
            ? static_cast<float>(common_ways + common_nodes) / static_cast<float>(std::max(total_ways_sol2 + total_nodes_sol2,total_ways_sol1 + total_nodes_sol1))
            : 0.0f;

        return similarity_degree;
    }
    
    // OPTIMISÉ: comparaison limitée aux 3 derniers chemins
    Solution check_solution(std::vector<osmium::object_id_type> s){
        if(s.size() < 2){
            return Solution(s, {}, 0.0f);
        }
        
        std::vector<Path> paths;
        float cost = 0.0;
        
        for (size_t i = 1; i < s.size(); i++) {
            std::vector<Path> p = check_path(s[i-1], s[i]);
            
            if(p.empty() || p[0].cost == std::numeric_limits<float>::max()){
                return Solution({}, {}, std::numeric_limits<float>::max());
            }
            
            Path path_to_insert = p[0];
            
            /* OPTIMISATION: comparer seulement avec les 3 derniers chemins
            if(p.size() > 1 && !paths.empty()){
                float lowest_similarity = std::numeric_limits<float>::max();
                
                size_t start = paths.size() > 3 ? paths.size() - 3 : 0;
                
                for(const auto& insertable_path : p){
                    float total_similarity = 0.0f;
                    
                    for(size_t j = start; j < paths.size(); j++){
                        total_similarity += check_path_similarity(insertable_path, paths[j]);
                    }
                    
                    if(total_similarity < lowest_similarity){
                        path_to_insert = insertable_path;
                        lowest_similarity = total_similarity;
                    }
                }
            }
            */
           
            paths.push_back(path_to_insert);
            cost += path_to_insert.cost;
        }
        
        return Solution(s, paths, cost);
    }

    std::vector<Path> check_path(osmium::object_id_type A, osmium::object_id_type B){
        auto it_a = visited_Paths.find(A);
        if(it_a != visited_Paths.end()){
            auto it_b = it_a->second.find(B);
            if(it_b != it_a->second.end() && !it_b->second.empty()){
                return it_b->second;
            }
        }

        std::vector<osmium::object_id_type> edges = PfSystem.A_Star_Search(A, B);
        
        if(edges.empty()){
            Path empty_path;
            empty_path.cost = std::numeric_limits<float>::max();
            return {empty_path};
        }
        
        float cost = 0.0f;
        for(const auto& e : edges){
            auto way_it = geo_box.data.ways.find(e);
            if(way_it != geo_box.data.ways.end()){
                cost += way_it->second.distance_meters;
            }
        }
        
        Path A_to_B = Path(edges, A, B, cost);
        this->visited_Paths[A][B].push_back(A_to_B);
        this->visited_Paths[B][A].push_back(A_to_B);
        return this->visited_Paths[A][B];
    }

    // check_neighborhood : Retourne POI le plus proche du groupe demandé
    // Stocke TOUS les voisins, met à jour avg_distance pour le groupe demandé
    osmium::object_id_type check_neighborhood(
        osmium::object_id_type n,
        std::unordered_set<int>& available_groups,
        const std::unordered_set<osmium::object_id_type>& visited_pois = {}
    ){
        for(const auto& neighbor : visited_Neighborhoods[n]){
            if(visited_pois.find(neighbor) == visited_pois.end()){
                for(const auto& groupe : geo_box.data.nodes[neighbor].groupes){
                    if(available_groups.find(groupe) != available_groups.end()){
                        return neighbor;
                    }
                }
            }
        }

        std::vector<Path> sol = PfSystem.Neighbor_Search(n, available_groups, visited_pois);

        osmium::object_id_type nearest_available_neighbor = 0;
        float nearest_distance = std::numeric_limits<float>::max();
        
        for (const auto& p : sol) {
            osmium::object_id_type neighbor_id = (p.node_extremity_left != n) 
                ? p.node_extremity_left 
                : p.node_extremity_right;
            
            auto neighbor_node = geo_box.data.nodes.find(neighbor_id);
            if (neighbor_node == geo_box.data.nodes.end()) continue;
            
            // Stocker path (tous groupes)
            if (visited_Paths[n][neighbor_id].empty()) {
                visited_Paths[n][neighbor_id].push_back(p);
                visited_Paths[neighbor_id][n].push_back(p);
            }
            
            // Stocker dans visited_Neighborhoods (tous groupes)
            auto& neighborhood = visited_Neighborhoods[n];
            if (std::find(neighborhood.begin(), neighborhood.end(), neighbor_id) == neighborhood.end()) {
                neighborhood.push_back(neighbor_id);
            }

            // Vérifier si ce voisin appartient au groupe demandé
            bool is_in_requested_group = false;
            for (const int group_id : available_groups) {

                if (neighbor_node->second.groupes.count(group_id) > 0) {
                    is_in_requested_group = true;
                    
                    if (group_id >= 0 && group_id < total_distance_by_group.size()) {
                        total_distance_by_group[group_id] += p.cost;
                        count_by_group[group_id]++;
                        avg_distance_by_group[group_id] =
                            total_distance_by_group[group_id] / count_by_group[group_id];
                    }
                    
                    break;
                }
            }

            // Trouver le POI le plus proche du groupe demandé ET non-visité
            if (is_in_requested_group && visited_pois.count(neighbor_id) == 0) {
                if (p.cost < nearest_distance) {
                    nearest_distance = p.cost;
                    nearest_available_neighbor = neighbor_id;
                }
            }
        }

        return nearest_available_neighbor;
    }
    
};

class Memory {
public:
    GeoBox& geo_box;
    Pathfinder& PfSystem;
    GlobalMemory MemoryStructure;

    Memory(GeoBox& box, Pathfinder& pf);
};

#endif // MEMORY_HPP