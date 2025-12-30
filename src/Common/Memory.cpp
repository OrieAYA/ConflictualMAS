#include "Memory.hpp"
#include "Pathfinding.hpp"
#include "../GeoBox/Box.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <limits>

// Constructeur - signature corrigée
Memory::Memory(GeoBox& box, Pathfinder& pf) 
    : geo_box(box), PfSystem(pf), MemoryStructure(box, pf) {}

float Memory::solutions_similarity(const Solution& a, const Solution& b) {
    if(a.paths.empty()) return 0.0f;
    
    float sim = 0.0f;
    float coef = 100.0f / static_cast<float>(a.paths.size());
    size_t min_size = std::min(a.paths.size(), b.paths.size());
    
    for (size_t i = 0; i < min_size; i++) {
        // Comparaison basique - vous pouvez améliorer cette logique
        if(a.paths[i].node_extremity_left == b.paths[i].node_extremity_left &&
           a.paths[i].node_extremity_right == b.paths[i].node_extremity_right){
            sim += coef;
        }
    }
    return sim;
}