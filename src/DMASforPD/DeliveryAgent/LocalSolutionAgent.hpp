#ifndef LOCAL_SOLUTION_AGENT_HPP
#define LOCAL_SOLUTION_AGENT_HPP

#include "DMASforPD/Utility/ObjectiveNode.hpp"
#include <vector>

struct OperableEnvironment;

// Plans from a single delivery node using the operable environment cost matrix.
// One LocalSolutionAgent exists per delivery node in the DeliveryAgent's task list.
//
// Each instance will run the DbVNS-PDP metaheuristic (structure only — not yet implemented).
// Equivalent to the local Agent in the legacy MetaAgent architecture, adapted for PDP.
struct LocalSolutionAgent {
    ObjectiveNode starting_node;   // delivery node this agent plans from

    explicit LocalSolutionAgent(const ObjectiveNode& node);

    // Returns a candidate visit sequence covering all objective nodes in env,
    // with pickup-before-delivery respected for each task.
    // Stub: returns nodes in insertion order until DbVNS-PDP is implemented.
    std::vector<ObjectiveNode> plan(const OperableEnvironment& env) const;
};

#endif // LOCAL_SOLUTION_AGENT_HPP
