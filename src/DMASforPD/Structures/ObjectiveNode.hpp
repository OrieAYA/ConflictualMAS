#ifndef OBJECTIVE_NODE_HPP
#define OBJECTIVE_NODE_HPP

#include "Environment/GeoBox/Box.hpp"

// A node in the road graph that acts as a pickup or delivery location.
// The pickup/delivery distinction belongs to PDPTask, not to the node itself.
// group_id links this node to its ObjectiveGroupCache in PDPServerMemory.
struct ObjectiveNode {
    osmium::object_id_type id       = 0;
    int group_id = 0;

    bool valid() const { return id != 0; }

    bool operator==(const ObjectiveNode& o) const {
        return id == o.id && group_id == o.group_id;
    }
    bool operator!=(const ObjectiveNode& o) const { return !(*this == o); }
};

#endif // OBJECTIVE_NODE_HPP
