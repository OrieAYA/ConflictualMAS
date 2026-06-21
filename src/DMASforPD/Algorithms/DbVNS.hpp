#ifndef LOCAL_SOLUTION_AGENT_HPP
#define LOCAL_SOLUTION_AGENT_HPP

#include "DMASforPD/Structures/ObjectiveNode.hpp"
#include <unordered_map>
#include <vector>

struct OperableEnvironment;

// Pickup/delivery pairing lookup for DbVNS constraint management.
// pickup_of [delivery_id] = pickup_id
// delivery_of[pickup_id]  = delivery_id
using PairingMap = std::unordered_map<osmium::object_id_type, osmium::object_id_type>;

// Parameters for the DbVNS-PDP search.
struct DbVNSParams {
    int    max_iterations     = 100;
    double max_divergence     = 0.7;  // allow branches up to max_divergence worse than pbest
    int    max_decompositions = 5;    // max promising branches explored per shake
    int    k_max              = 4;    // shake depth factor
};


// Plans from a single delivery node using the operable environment cost matrix.
// One LocalSolutionAgent exists per delivery node in the DeliveryAgent's task list.
//
// Implements DbVNS-PDP (Decomposition-based VNS for Pickup and Delivery):
//   - Sequence is built in reverse, anchored at the delivery node (= last visited).
//   - Adding delivery D_i to the stack queues its pickup P_i in pending,
//     so P_i is always placed before D_i in the final (reversed) sequence.
//   - VNS shake: backtrack partial solutions, forbid tail choices, explore alternatives.
//   - Objective: minimize total travel distance.
struct LocalSolutionAgent {
    ObjectiveNode starting_node;
    DbVNSParams   params;

    explicit LocalSolutionAgent(const ObjectiveNode& node,
                                const DbVNSParams&   p = {});

    // Run DbVNS-PDP and return the visit sequence (pickup-before-delivery guaranteed).
    // pickup_of  : delivery_id -> pickup_id
    // delivery_of: pickup_id  -> delivery_id (reserved for future use)
    // agent_current: agent's current node id (reserved for first-leg cost)
    std::vector<ObjectiveNode> plan(
        const OperableEnvironment& env,
        const PairingMap&          pickup_of,
        const PairingMap&          delivery_of,
        osmium::object_id_type     agent_current
    ) const;

    // Forward DbVNS-PDP for lifelong replanning (mono-agent GPDP).
    //
    // Plans a visit order for all nodes in `env`, starting from an external
    // position (NOT itself a node to visit). Constraints enforced:
    //   - Pairing : pickup_of[delivery_id] = pickup_id  →  pickup before delivery
    //   - Capacity: simultaneous carry ≤ max_capacity at every step
    //
    // initial_load = number of packages already in the agent's hands when it
    //   reaches the plan-start position (e.g. the inflight head). The forward
    //   sequence builds up on top of this baseline.
    //
    // start_costs[i] = travel cost from plan-start to env.nodes[i].
    // Use kCostScale (a large sentinel) for unreachable nodes.
    //
    // Returns the forward visit sequence with pickup-before-delivery + capacity
    // guaranteed.
    static std::vector<ObjectiveNode> plan_sequence(
        const OperableEnvironment& env,
        const PairingMap&          pickup_of,
        const std::vector<float>&  start_costs,
        int                        max_capacity = 3,
        int                        initial_load = 0,
        const DbVNSParams&         params = {}
    );

};

#endif // LOCAL_SOLUTION_AGENT_HPP
