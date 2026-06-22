#include "Environment/Simulation/TemporalGraph.hpp"

TemporalGraph::TemporalGraph(const GeoBox& geo_box, float episode_end)
    : geo_box_(geo_box), episode_end_(episode_end) {}

TemporalChainList& TemporalGraph::node_chain(osmium::object_id_type node_id) {
    auto it = node_chains_.find(node_id);
    if (it != node_chains_.end()) return it->second;

    auto& chain = node_chains_.try_emplace(node_id, episode_end_).first->second;
    // A node's static relations are its incident edges.
    auto nit = geo_box_.data.nodes.find(node_id);
    if (nit != geo_box_.data.nodes.end())
        chain.start()->incident_elements = nit->second.incident_ways;
    return chain;
}

TemporalChainList& TemporalGraph::edge_chain(osmium::object_id_type edge_id) {
    auto it = edge_chains_.find(edge_id);
    if (it != edge_chains_.end()) return it->second;

    auto& chain = edge_chains_.try_emplace(edge_id, episode_end_).first->second;
    // An edge's static relations are its two endpoint nodes.
    auto wit = geo_box_.data.ways.find(edge_id);
    if (wit != geo_box_.data.ways.end())
        chain.start()->incident_elements = { wit->second.node1_id, wit->second.node2_id };
    return chain;
}

void TemporalGraph::reset() {
    node_chains_.clear();
    edge_chains_.clear();
}
