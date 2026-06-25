#ifndef ENVIRONMENT_TEMPORAL_GRAPH_HPP
#define ENVIRONMENT_TEMPORAL_GRAPH_HPP

#include "Environment/Simulation/TemporalChainList.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <osmium/osm/types.hpp>
#include <unordered_map>

// Node-event temporal layer: one TemporalChainList per node (task/objective
// events over time). Edge occupancy/congestion lives in CongestionMap, which is
// the edge temporal layer — so there is a single store per concern.
//
// Chains are created on demand, keyed by OSM id, with the node's incident edges
// stored on the start sentinel's incident_elements. The GeoBox stays untouched.
class TemporalGraph {
public:
    TemporalGraph(const GeoBox& geo_box, float episode_end);

    TemporalChainList& node_chain(osmium::object_id_type node_id);

    bool        has_node_chain(osmium::object_id_type id) const { return node_chains_.count(id) != 0; }
    std::size_t n_node_chains() const { return node_chains_.size(); }
    float       episode_end()   const { return episode_end_; }

    void reset() { node_chains_.clear(); }

private:
    const GeoBox& geo_box_;
    float         episode_end_;
    std::unordered_map<osmium::object_id_type, TemporalChainList> node_chains_;
};

#endif // ENVIRONMENT_TEMPORAL_GRAPH_HPP
