#ifndef AGENT_SOLUTION_HPP
#define AGENT_SOLUTION_HPP

#include "DMASforPD/Structures/ObjectiveNode.hpp"
#include "DMASforPD/Structures/ObjectiveCache.hpp"
#include <vector>

class PDPGlobalMemory;

// One planned step in an agent's sequence.
// estimated_arrival is set by GlobalMemory::commit_plan(); -1 if not yet committed.
struct SolutionStep {
    ObjectiveNode node;
    int           estimated_arrival = -1;
};

// Plan of a single delivery agent.
//
// current_position: live pointer to the agent's real graph node — reflects
//                   the last node the agent reached (or its start).
//
// sequence: remaining objectives in order. Front = next to visit.
//           estimated_arrival is filled when the plan is committed to GlobalMemory.
//
// Sequence layout (link-transmission convention):
//   sequence[0] = next objective to reach
//   sequence[k] = k-th future objective
//   sequence.back() = final objective
struct AgentSolution {
    const osmium::object_id_type* current_position = nullptr;
    std::vector<SolutionStep>     sequence;

    // ---- Construction --------------------------------------------------

    static AgentSolution make(const osmium::object_id_type* position);

    // ---- State queries -------------------------------------------------

    bool        empty()         const { return sequence.empty(); }
    bool        valid()         const { return current_position != nullptr; }
    std::size_t num_remaining() const { return sequence.size(); }

    // Next objective node and its estimated arrival step.
    const SolutionStep&  next_step()                 const;
    ObjectiveNode        next_objective()             const;
    int                  next_estimated_arrival()     const;

    ObjectiveNode        objective_at(std::size_t i)      const;
    int                  estimated_arrival_at(std::size_t i) const;

    // ---- Plan mutations ------------------------------------------------

    // Append an objective (estimated_arrival will be set by commit_plan).
    void push_back(const ObjectiveNode& node);

    // Remove sequence.front(). Call after current_position has been updated
    // to reflect the reached node.
    void advance();

    // ---- Path retrieval ------------------------------------------------

    const ObjectivePath* path_to_next  (PDPGlobalMemory& memory) const;
    const ObjectivePath* path_between  (std::size_t i, PDPGlobalMemory& memory) const;
    float                total_planned_cost(PDPGlobalMemory& memory) const;
};

#endif // AGENT_SOLUTION_HPP
