#include "Environment/Simulation/TemporalGraph.hpp"

TemporalGraph::TemporalGraph(const GeoBox& geo_box, float episode_end)
    : geo_box_(geo_box), episode_end_(episode_end) {}

TemporalChainList& TemporalGraph::node_chain(osmium::object_id_type node_id) {
    auto it = node_chains_.find(node_id);
    if (it != node_chains_.end()) return it->second;

    auto& chain = node_chains_.try_emplace(node_id, episode_end_).first->second;
    auto nit = geo_box_.data.nodes.find(node_id);
    if (nit != geo_box_.data.nodes.end())
        chain.start()->incident_elements = nit->second.incident_ways;  // a node's edges
    return chain;
}
