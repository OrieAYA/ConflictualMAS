#ifndef TDASTAR_HPP
#define TDASTAR_HPP

#include "DMASforPD/Structures/ObjectiveCache.hpp"   // TDAStarResult, GeoBox, CongestionMap

// Time-dependent A* on the road graph: shortest path from -> to departing at
// start_time, where each edge costs its BPR-adjusted travel time evaluated at
// the predicted arrival step on that edge (search time t* advances along the
// path). Heuristic = haversine / speed (admissible). self_weight is 0 here:
// this is a prediction, the caller is not yet on the edge.
TDAStarResult td_astar(const GeoBox&          geo_box,
                       osmium::object_id_type from,
                       osmium::object_id_type to,
                       int                    start_time,
                       float                  agent_speed,
                       const CongestionMap&   congestion);

#endif // TDASTAR_HPP
