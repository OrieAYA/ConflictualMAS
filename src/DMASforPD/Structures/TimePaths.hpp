#ifndef TIMED_PATH_HPP
#define TIMED_PATH_HPP

#include <osmium/osm/types.hpp>
#include <vector>

// A path between two nodes with per-edge traversal times, expressed as
// integer steps RELATIVE to an unspecified departure step.
//
// Design rationale (DTA / link-transmission model):
//   - The ObjectivePath cache is time-agnostic (reusable across agents/times).
//   - TimedPath instantiates a path at a given speed: edge_steps[i] is the
//     number of simulation steps needed to traverse edge i.
//   - Given a concrete departure step d, abs_entry(i,d) and abs_exit(i,d)
//     give the absolute simulation steps for congestion registration.
//
// total_steps = sum(edge_steps) = arrival offset from departure.
//
// Call seal() once after all edge_steps are pushed to enable O(1) rel_entry/rel_exit.
struct TimedPath {
    osmium::object_id_type from_node     = 0;
    osmium::object_id_type to_node       = 0;
    float                  total_distance = 0.0f;  // meters
    int                    total_steps    = 0;      // steps from departure to arrival

    std::vector<osmium::object_id_type> nodes;
    std::vector<osmium::object_id_type> edge_ids;
    std::vector<int>                    edge_steps;   // edge_steps[i] >= 1
    std::vector<int>                    prefix_steps; // prefix_steps[i] = sum(edge_steps[0..i-1])

    bool valid() const { return from_node != 0 && to_node != 0; }
    bool empty() const { return edge_ids.empty(); }

    // Build prefix_steps from edge_steps. Must be called after all edges are pushed.
    void seal();

    // Steps elapsed from departure before entering / upon exiting edge i.
    // O(1) after seal(), O(i) before (fallback).
    int rel_entry(std::size_t i) const;
    int rel_exit (std::size_t i) const;

    int abs_entry(std::size_t i, int departure) const { return departure + rel_entry(i); }
    int abs_exit (std::size_t i, int departure) const { return departure + rel_exit(i);  }
};

#endif // TIMED_PATH_HPP
