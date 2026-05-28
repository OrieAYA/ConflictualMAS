#include "PathHelper.hpp"
#include <cmath>

PathHelper::PathHelper(Pathfinder& pf) : pf_(pf) {
    invalid_.valid = false;
}

float PathHelper::heuristic(osmium::object_id_type from,
                             osmium::object_id_type to) const {
    return const_cast<Pathfinder&>(pf_).heuristic(from, to);
}

const SimplePath& PathHelper::get(osmium::object_id_type from,
                                   osmium::object_id_type to) {
    if (from == to) {
        static SimplePath self;
        self.valid = true;
        self.cost  = 0.f;
        self.nodes = { from };
        self.edges.clear();
        return self;
    }

    Key key{ from, to };
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;

    // Call A* on the shared Pathfinder.
    std::vector<osmium::object_id_type> edge_seq = pf_.A_Star_Search(from, to);

    SimplePath p;
    if (edge_seq.empty()) {
        p.valid = false;
        cache_[key] = p;
        return cache_[key];
    }

    // Reconstruct node sequence from edges. Walk each edge in order; for
    // each edge, the "next" node is the endpoint that is NOT the current
    // node. Start at `from`.
    p.nodes.reserve(edge_seq.size() + 1);
    p.edges = edge_seq;
    p.nodes.push_back(from);

    osmium::object_id_type current = from;
    const auto& ways = pf_.geo_box.data.ways;
    float total = 0.f;

    for (auto eid : edge_seq) {
        auto wit = ways.find(eid);
        if (wit == ways.end()) {
            // Way not found — abort, treat as invalid.
            p.valid = false;
            p.nodes.clear();
            p.edges.clear();
            p.cost = 0.f;
            cache_[key] = p;
            return cache_[key];
        }
        const auto& w = wit->second;
        const osmium::object_id_type other =
            (w.node1_id == current) ? w.node2_id : w.node1_id;
        p.nodes.push_back(other);
        total += w.distance_meters;
        current = other;
    }

    // Final sanity check — last node should be `to`.
    if (!p.nodes.empty() && p.nodes.back() != to) {
        // A* and reconstruction disagree — flag invalid, keep partial path
        // off the cache.
        p.valid = false;
        cache_[key] = p;
        return cache_[key];
    }

    p.cost  = total;
    p.valid = true;
    cache_[key] = std::move(p);
    return cache_[key];
}