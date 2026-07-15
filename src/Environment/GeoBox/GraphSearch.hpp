#ifndef GRAPH_SEARCH_HPP
#define GRAPH_SEARCH_HPP

#include "Environment/GeoBox/Box.hpp"
#include <unordered_map>
#include <vector>

// Static (free-flow) searches over the GeoBox road graph. The time-dependent
// congestion-aware search lives in DMASforPD/Algorithms/TDAStar; the
// incremental memoised searches live in DMASforPD/Structures/ObjectiveCache.

namespace graph_search {

// Great-circle distance in metres between two graph nodes (0 when either
// node is unknown). Admissible lower bound on any road distance.
float haversine_between(const GeoBox& box,
                        osmium::object_id_type a,
                        osmium::object_id_type b);

// Static shortest path in metres: A* with haversine heuristic. Returns the
// ordered way-id sequence from -> to; empty when unreachable or from == to.
std::vector<osmium::object_id_type> shortest_path_edges(
    const GeoBox&          box,
    osmium::object_id_type from,
    osmium::object_id_type to);

// Single-source Dijkstra: static distance in metres from `source` to every
// reachable node. One call replaces N per-agent searches during allocation.
std::unordered_map<osmium::object_id_type, float> dijkstra_distances(
    const GeoBox&          box,
    osmium::object_id_type source);

// Tag a way with a path group (render colouring). No-op on unknown ways.
void set_way_group(GeoBox&                box,
                   osmium::object_id_type way_id,
                   int                    group);

} // namespace graph_search

#endif // GRAPH_SEARCH_HPP
