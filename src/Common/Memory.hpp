#ifndef MEMORY_HPP
#define MEMORY_HPP

#include "../Geobox/Box.hpp"
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

struct GlobalMemory {
    GeoBox& geo_box;
    Pathfinder& PfSystem;
    std::map<std::vector<osmium::object_id_type>, Solution> visited_Solutions;
    std::map<osmium::object_id_type, std::map<osmium::object_id_type, Path>> visited_Paths;
    std::map<osmium::object_id_type, std::vector<osmium::object_id_type>> visited_Neighborhoods;

    float length_constraint = 5000.0f;
    float search_coefficient = 0.2f;

    GlobalMemory() = delete;
    GlobalMemory(GeoBox& box, Pathfinder& pf) 
        : geo_box(box), PfSystem(pf) {}

    Solution check_solution(std::vector<osmium::object_id_type> s){
        auto it = visited_Solutions.find(s);
        if(it != visited_Solutions.end()){
            return it->second;
        }
        std::vector<Path> path;
        float cost = 0.0;
        for (size_t i = 1; i < s.size(); i++) {
            Path p = check_path(s[i-1], s[i]);
            path.push_back(p);
            cost = cost + p.cost;
        }
        Solution sol = Solution(s, path, cost);
        visited_Solutions[s] = sol;
        return sol;
    }

    Path check_path(osmium::object_id_type A, osmium::object_id_type B){
        auto it_a = visited_Paths.find(A);
        if(it_a != visited_Paths.end()){
            auto it_b = it_a->second.find(B);
            if(it_b != it_a->second.end()){
                return it_b->second;
            }
        }
        
        std::vector<osmium::object_id_type> edges = PfSystem.A_Star_Search(A, B);
        float cost = 0.0f;
        for(const auto& e : edges){
            auto way_it = geo_box.data.ways.find(e);
            if(way_it != geo_box.data.ways.end()){
                cost += way_it->second.distance_meters;
            }
        }
        
        Path A_to_B = Path(edges, A, B, cost);
        this->visited_Paths[A][B] = A_to_B;
        this->visited_Paths[B][A] = A_to_B;
        return A_to_B;
    }

    std::vector<osmium::object_id_type> check_neighborhood(
        osmium::object_id_type n,
        const std::unordered_set<osmium::object_id_type>& visited_pois = {}
    ){
        auto it = visited_Neighborhoods.find(n);
        if(it != visited_Neighborhoods.end()){
            return it->second;
        }
        
        std::vector<Path> sol = PfSystem.Neighbor_Search(n, search_coefficient, visited_pois);
        std::vector<osmium::object_id_type> neighbors = {};
        
        for(const auto& p : sol){
            osmium::object_id_type neighbor_id = (p.node_extremity_left != n) 
                ? p.node_extremity_left 
                : p.node_extremity_right;
            
            neighbors.push_back(neighbor_id);
            this->visited_Paths[neighbor_id][n] = p;
            this->visited_Paths[n][neighbor_id] = p;
        }
        
        visited_Neighborhoods[n] = neighbors;
        return neighbors;
    }

    std::vector<osmium::object_id_type> continue_neighborhood_search(
        osmium::object_id_type n,
        const std::unordered_set<osmium::object_id_type>& visited_pois
    ){
        
        std::vector<Path> extended_paths = PfSystem.Neighbor_Search(n, search_coefficient, visited_pois);
        
        std::vector<osmium::object_id_type> neighbors = {};
        for(const auto& p : extended_paths){
            osmium::object_id_type neighbor_id = (p.node_extremity_left != n) 
                ? p.node_extremity_left 
                : p.node_extremity_right;
            
            neighbors.push_back(neighbor_id);
            this->visited_Paths[neighbor_id][n] = p;
            this->visited_Paths[n][neighbor_id] = p;
        }
        
        visited_Neighborhoods[n] = neighbors;
        
        return neighbors;
    }
    
    float calculate_similarity(const Solution& sol1, const Solution& sol2){
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
        return total_edges > 0 ? static_cast<float>(common_edges) / total_edges : 0.0f;
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