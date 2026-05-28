#ifndef SOTA_PATH_HELPER_HPP
#define SOTA_PATH_HELPER_HPP

#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <osmium/osm/types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// PathHelper
// ════════════════════════════════════════════════════════════════════════════
//
// SHARED utility for SOTA solvers. Wraps Pathfinder::A_Star_Search to return
// a structured (nodes, edges, cost) triple instead of the raw edge sequence,
// and caches results in a per-(from, to) map.
//
// Rationale: A* shortest paths are a function of the GeoBox alone — they do
// NOT depend on any solver's algorithm. Two different solvers asking for the
// same (from, to) shortest path MUST get the same answer (the graph itself
// is shared). So caching is safe and accelerates the comparison runs.
//
// What this DOES NOT do: cost adjustment, congestion-aware routing,
// guide-path computation, anything that would couple it to a specific
// solver's algorithm. Those live inside the solvers themselves.

struct SimplePath {
    std::vector<osmium::object_id_type> nodes;     // nodes[0] = from, nodes.back() = to
    std::vector<osmium::object_id_type> edges;     // edges[i] connects nodes[i] and nodes[i+1]
    float                                cost = 0.f; // total length in metres
    bool                                 valid = false;
};

class PathHelper {
public:
    explicit PathHelper(Pathfinder& pf);

    // Get (or compute and cache) the shortest path from `from` to `to`.
    // Returns a const reference into the cache; valid for the lifetime of
    // this PathHelper. If no path exists, the returned SimplePath has
    // valid=false.
    const SimplePath& get(osmium::object_id_type from,
                          osmium::object_id_type to);

    // Cache size diagnostic.
    int cache_size() const { return static_cast<int>(cache_.size()); }

    // Compute the EUCLIDEAN heuristic cost between two nodes (haversine in
    // metres). Used by solvers needing a quick h-value when no path exists
    // yet or when an upper-bound estimate is enough.
    float heuristic(osmium::object_id_type from,
                    osmium::object_id_type to) const;

private:
    Pathfinder& pf_;

    using Key = std::pair<osmium::object_id_type, osmium::object_id_type>;
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            const auto a = std::hash<osmium::object_id_type>{}(k.first);
            const auto b = std::hash<osmium::object_id_type>{}(k.second);
            return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
        }
    };

    std::unordered_map<Key, SimplePath, KeyHash> cache_;
    SimplePath                                    invalid_;  // returned when path fails
};

#endif // SOTA_PATH_HELPER_HPP