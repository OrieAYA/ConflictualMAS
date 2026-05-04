#include "LocalSolutionAgent.hpp"
#include "OperableEnvironment.hpp"

LocalSolutionAgent::LocalSolutionAgent(const ObjectiveNode& node)
    : starting_node(node) {}

std::vector<ObjectiveNode> LocalSolutionAgent::plan(const OperableEnvironment& env) const {
    // Stub: DbVNS-PDP not yet implemented.
    // Returns nodes in insertion order; the planning algorithm will replace this.
    return env.nodes;
}
