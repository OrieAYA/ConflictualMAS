#include "Memory.hpp"
#include "Pathfinding.hpp"
#include "../GeoBox/Box.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <limits>

Memory::Memory(GeoBox& box, Pathfinder& pf) 
    : geo_box(box), PfSystem(pf), MemoryStructure(box, pf) {}