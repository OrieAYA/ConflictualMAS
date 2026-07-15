#include "DMASforPD/Algorithms/TDAStar.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <unordered_set>

TDAStarResult td_astar(const GeoBox&          geo_box,
                       osmium::object_id_type from,
                       osmium::object_id_type to,
                       int                    start_time,
                       float                  agent_speed,
                       const CongestionMap&   congestion) {
    TDAStarResult result;
    if (from == to) {
        result.nodes = {from}; result.total_time = 0; result.total_distance = 0.0f;
        return result;
    }
    auto goal_it = geo_box.data.nodes.find(to);
    if (goal_it == geo_box.data.nodes.end()) return result;

    const double goal_lat  = goal_it->second.lat;
    const double goal_lon  = goal_it->second.lon;
    const float  inv_speed = (agent_speed > 0.0f) ? 1.0f / agent_speed : 0.0f;

    auto heuristic = [&](osmium::object_id_type n) -> float {
        auto it = geo_box.data.nodes.find(n);
        if (it == geo_box.data.nodes.end()) return 0.0f;
        return static_cast<float>(
            calculate_haversine_distance(it->second.lat, it->second.lon,
                                         goal_lat, goal_lon)) * inv_speed;
    };

    std::unordered_map<osmium::object_id_type, float> g;
    std::unordered_map<osmium::object_id_type,
        std::pair<osmium::object_id_type, osmium::object_id_type>> came_from;
    std::unordered_set<osmium::object_id_type> closed;
    using Entry = std::pair<float, osmium::object_id_type>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;

    g[from] = 0.0f;
    open.push({heuristic(from), from});

    while (!open.empty()) {
        auto [f, current] = open.top(); open.pop();
        if (closed.count(current)) continue;
        closed.insert(current);

        if (current == to) {
            std::vector<osmium::object_id_type> nodes_rev, edges;
            float dist = 0.0f;
            osmium::object_id_type cur = to;
            nodes_rev.push_back(cur);
            while (cur != from) {
                auto it = came_from.find(cur);
                if (it == came_from.end()) break;
                auto [prev, way_id] = it->second;
                edges.push_back(way_id);
                cur = prev;
                nodes_rev.push_back(cur);
                auto way_it = geo_box.data.ways.find(way_id);
                if (way_it != geo_box.data.ways.end()) dist += way_it->second.distance_meters;
            }
            std::reverse(nodes_rev.begin(), nodes_rev.end());
            std::reverse(edges.begin(), edges.end());
            result.nodes          = std::move(nodes_rev);
            result.edges          = std::move(edges);
            result.total_time     = static_cast<int>(std::round(g.at(to)));
            result.total_distance = dist;
            return result;
        }

        auto node_it = geo_box.data.nodes.find(current);
        if (node_it == geo_box.data.nodes.end()) continue;

        const int t_arrival = start_time + static_cast<int>(g.at(current));
        for (const auto& way_id : node_it->second.incident_ways) {
            auto way_it = geo_box.data.ways.find(way_id);
            if (way_it == geo_box.data.ways.end()) continue;
            const auto& way = way_it->second;
            const osmium::object_id_type nb =
                (way.node1_id == current) ? way.node2_id : way.node1_id;
            if (closed.count(nb)) continue;

            const float base_steps = way.distance_meters * inv_speed;
            const float edge_time  = congestion.adjusted_cost(
                way_id, base_steps, way.distance_meters, t_arrival);
            const float new_g = g.at(current) + edge_time;
            auto g_it = g.find(nb);
            if (g_it == g.end() || new_g < g_it->second) {
                g[nb]         = new_g;
                came_from[nb] = {current, way_id};
                open.push({new_g + heuristic(nb), nb});
            }
        }
    }
    return result;
}
