#include "Environment/GeoBox/GraphSearch.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <utility>

namespace graph_search {

float haversine_between(const GeoBox&          box,
                        osmium::object_id_type a,
                        osmium::object_id_type b) {
    const auto& nodes = box.data.nodes;
    auto ia = nodes.find(a);
    auto ib = nodes.find(b);
    if (ia == nodes.end() || ib == nodes.end()) return 0.f;
    return static_cast<float>(
        calculate_haversine_distance(ia->second.lat, ia->second.lon,
                                     ib->second.lat, ib->second.lon));
}

std::vector<osmium::object_id_type> shortest_path_edges(
    const GeoBox&          box,
    osmium::object_id_type from,
    osmium::object_id_type to) {

    const auto& nodes = box.data.nodes;
    const auto& ways  = box.data.ways;

    auto from_it = nodes.find(from);
    auto goal_it = nodes.find(to);
    if (from_it == nodes.end() || goal_it == nodes.end() || from == to)
        return {};

    const double goal_lat = goal_it->second.lat;
    const double goal_lon = goal_it->second.lon;
    auto h = [&](const MyData::Point& p) -> float {
        return static_cast<float>(
            calculate_haversine_distance(p.lat, p.lon, goal_lat, goal_lon));
    };

    std::unordered_map<osmium::object_id_type, float> g;
    // node -> (previous node, way taken to reach it)
    std::unordered_map<osmium::object_id_type,
        std::pair<osmium::object_id_type, osmium::object_id_type>> came_from;
    std::unordered_set<osmium::object_id_type> closed;
    using Entry = std::pair<float, osmium::object_id_type>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;

    g[from] = 0.f;
    open.push({h(from_it->second), from});

    while (!open.empty()) {
        const auto [f, cur] = open.top();
        open.pop();
        if (closed.count(cur)) continue;
        closed.insert(cur);

        if (cur == to) {
            std::vector<osmium::object_id_type> edges;
            osmium::object_id_type n = to;
            for (auto it = came_from.find(n); it != came_from.end();
                 it = came_from.find(n)) {
                edges.push_back(it->second.second);
                n = it->second.first;
            }
            std::reverse(edges.begin(), edges.end());
            return edges;
        }

        auto nit = nodes.find(cur);
        if (nit == nodes.end()) continue;

        for (const auto& way_id : nit->second.incident_ways) {
            auto wit = ways.find(way_id);
            if (wit == ways.end()) continue;
            const auto& w = wit->second;

            osmium::object_id_type nb;
            if      (w.node1_id == cur) nb = w.node2_id;
            else if (w.node2_id == cur) nb = w.node1_id;
            else continue; // malformed way

            const float ng = g[cur] + static_cast<float>(w.distance_meters);
            auto git = g.find(nb);
            if (git == g.end() || ng < git->second) {
                g[nb]         = ng;
                came_from[nb] = {cur, way_id};
                auto nbit = nodes.find(nb);
                const float hn = (nbit != nodes.end()) ? h(nbit->second) : 0.f;
                open.push({ng + hn, nb});
            }
        }
    }

    return {};
}

std::unordered_map<osmium::object_id_type, float> dijkstra_distances(
    const GeoBox&          box,
    osmium::object_id_type source) {

    std::unordered_map<osmium::object_id_type, float> dist;

    const auto& nodes = box.data.nodes;
    const auto& ways  = box.data.ways;
    if (nodes.find(source) == nodes.end()) return dist;

    using Entry = std::pair<float, osmium::object_id_type>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    dist[source] = 0.f;
    pq.push({0.f, source});

    while (!pq.empty()) {
        const auto [d, u] = pq.top();
        pq.pop();

        // Stale heap entry: a shorter distance was already settled.
        auto du_it = dist.find(u);
        if (du_it == dist.end() || d > du_it->second) continue;

        auto node_it = nodes.find(u);
        if (node_it == nodes.end()) continue;

        for (const auto& way_id : node_it->second.incident_ways) {
            auto way_it = ways.find(way_id);
            if (way_it == ways.end()) continue;
            const auto& way = way_it->second;

            osmium::object_id_type v;
            if      (way.node1_id == u) v = way.node2_id;
            else if (way.node2_id == u) v = way.node1_id;
            else continue; // malformed way

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

void set_way_group(GeoBox&                box,
                   osmium::object_id_type way_id,
                   int                    group) {
    auto it = box.data.ways.find(way_id);
    if (it != box.data.ways.end())
        it->second.add_group(group);
}

} // namespace graph_search
