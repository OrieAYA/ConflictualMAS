#include "AgentSolution.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include <stdexcept>

AgentSolution AgentSolution::make(const osmium::object_id_type* position) {
    AgentSolution sol;
    sol.current_position = position;
    return sol;
}

const SolutionStep& AgentSolution::next_step() const {
    if (sequence.empty())
        throw std::runtime_error("AgentSolution::next_step: empty sequence");
    return sequence.front();
}

ObjectiveNode AgentSolution::next_objective() const {
    return next_step().node;
}

int AgentSolution::next_estimated_arrival() const {
    return next_step().estimated_arrival;
}

ObjectiveNode AgentSolution::objective_at(std::size_t i) const {
    return sequence.at(i).node;
}

int AgentSolution::estimated_arrival_at(std::size_t i) const {
    return sequence.at(i).estimated_arrival;
}

void AgentSolution::push_back(const ObjectiveNode& node) {
    sequence.push_back({node, -1});
}

void AgentSolution::advance() {
    if (!sequence.empty())
        sequence.erase(sequence.begin());
}

const ObjectivePath* AgentSolution::path_to_next(PDPGlobalMemory& memory) const {
    if (!current_position || sequence.empty()) return nullptr;
    const ObjectiveNode& next = sequence.front().node;
    return memory.get_or_compute_path(*current_position, next.id, next.group_id);
}

const ObjectivePath* AgentSolution::path_between(std::size_t i, PDPGlobalMemory& memory) const {
    if (i + 1 >= sequence.size()) return nullptr;
    const ObjectiveNode& a = sequence[i].node;
    const ObjectiveNode& b = sequence[i + 1].node;
    return memory.get_or_compute_path(a.id, b.id, a.group_id);
}

float AgentSolution::total_planned_cost(PDPGlobalMemory& memory) const {
    if (!current_position || sequence.empty()) return 0.0f;
    float total = 0.0f;
    const ObjectivePath* first = path_to_next(memory);
    if (first && first->valid()) total += first->cost;
    for (std::size_t i = 0; i + 1 < sequence.size(); ++i) {
        const ObjectivePath* seg = path_between(i, memory);
        if (seg && seg->valid()) total += seg->cost;
    }
    return total;
}
