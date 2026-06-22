#ifndef ENVIRONMENT_TEMPORAL_GRAPH_HPP
#define ENVIRONMENT_TEMPORAL_GRAPH_HPP

#include "Environment/Simulation/TemporalChainList.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <osmium/osm/types.hpp>
#include <unordered_map>

// ════════════════════════════════════════════════════════════════════════════
// Temporal graph layer — one TemporalChainList per node and per edge.
// ════════════════════════════════════════════════════════════════════════════
//
// The runtime temporal state that sits ON TOP of the static (serialized) GeoBox
// without polluting it. Chains are created on demand and keyed by OSM id, so
// "every node/edge has its temporal chained list" holds logically while the
// GeoBox stays const-shareable and copy/serialize-safe.
//
// On first access a chain is initialised with start(0)/end(episode_end)
// sentinels and its static relations stored on the start sentinel's
// incident_elements: a node chain lists its incident EDGES, an edge chain lists
// its two endpoint NODES.
class TemporalGraph {
public:
    TemporalGraph(const GeoBox& geo_box, float episode_end);

    // Create-on-demand chains for a node / an edge (by OSM id).
    TemporalChainList& node_chain(osmium::object_id_type node_id);
    TemporalChainList& edge_chain(osmium::object_id_type edge_id);

    bool has_node_chain(osmium::object_id_type id) const { return node_chains_.count(id) != 0; }
    bool has_edge_chain(osmium::object_id_type id) const { return edge_chains_.count(id) != 0; }

    std::size_t n_node_chains() const { return node_chains_.size(); }
    std::size_t n_edge_chains() const { return edge_chains_.size(); }

    float episode_end() const { return episode_end_; }

    // Drop all chains (e.g. between episodes). The GeoBox is untouched.
    void reset();

private:
    const GeoBox& geo_box_;
    float         episode_end_;
    std::unordered_map<osmium::object_id_type, TemporalChainList> node_chains_;
    std::unordered_map<osmium::object_id_type, TemporalChainList> edge_chains_;
};

#endif // ENVIRONMENT_TEMPORAL_GRAPH_HPP
