#ifndef GREEDY_PLANNING_HPP
#define GREEDY_PLANNING_HPP

#include "DMASforPD/Algorithms/DbVNS.hpp"   // PairingMap, ObjectiveNode, OperableEnvironment fwd
#include <vector>

// Forward greedy nearest-next-objective planner for lifelong GPDP, same I/O
// contract as LocalSolutionAgent::plan_sequence (DbVNS.hpp):
//   - Pairing : pickup_of[delivery_id] = pickup_id -> pickup before delivery
//   - Capacity: simultaneous carry <= max_capacity at every step
// Strategy: repeatedly select, among currently available objective nodes
// (unvisited pickups not blocking capacity + unlocked deliveries), the one
// with minimal cost from the current position, until none remain.
std::vector<ObjectiveNode> plan_sequence_greedy(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const std::vector<float>&  start_costs,
    int                        max_capacity = 3,
    int                        initial_load = 0);

#endif // GREEDY_PLANNING_HPP