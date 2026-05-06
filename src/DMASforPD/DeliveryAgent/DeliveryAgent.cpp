#include "DeliveryAgent.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include "DMASforPD/TaskAgent/TaskAllocationModule.hpp"
#include <algorithm>
#include <limits>

// ---- Construction -------------------------------------------------------

DeliveryAgent::DeliveryAgent(int id, osmium::object_id_type start_node)
    : agent_id(id), current_node(start_node) {
    solution = AgentSolution::make(&current_node);
}

// ---- Spatial state ------------------------------------------------------

void DeliveryAgent::start_edge(
    osmium::object_id_type edge_id,
    osmium::object_id_type to_node,
    int arrival_step
) {
    in_transit = EdgeTransit{ edge_id, current_node, to_node, arrival_step };
}

void DeliveryAgent::arrive_at_node() {
    if (!in_transit) return;
    current_node = in_transit->to_node;
    in_transit.reset();
    // current_node is pointed to by solution.current_position — automatically reflected.
}

// ---- GlobalMemory interface ---------------------------------------------

void DeliveryAgent::connect(PDPGlobalMemory& memory) {
    memory.register_delivery_agent(*this);
}

void DeliveryAgent::disconnect(PDPGlobalMemory& memory) {
    memory.unregister_delivery_agent(agent_id);
}

void DeliveryAgent::receive_task(PDPTask& task, PDPGlobalMemory& /*memory*/) {
    add_task_to_memory(task);
    solution.push_back(task.pickup);
    solution.push_back(task.delivery);
    status = AgentStatus::Active;
    // assign_task and commit_plan are called by the TAM immediately after.
}

void DeliveryAgent::remove_completed_task(int task_id) {
    auto it = std::find_if(local_memory.tasks.begin(), local_memory.tasks.end(),
        [task_id](const PDPTask* t){ return t->task_id == task_id; });
    if (it == local_memory.tasks.end()) return;

    PDPTask* task = *it;
    local_memory.operable_env.remove_task(task->pickup.id, task->delivery.id);

    auto la_it = std::find_if(local_memory.local_agents.begin(), local_memory.local_agents.end(),
        [&](const LocalSolutionAgent& la){ return la.starting_node.id == task->delivery.id; });
    if (la_it != local_memory.local_agents.end())
        local_memory.local_agents.erase(la_it);

    local_memory.tasks.erase(it);
}

float DeliveryAgent::try_accept_task(const TaskOffer& /*offer*/, PDPGlobalMemory& /*memory*/) {
    // Default: always accept. Future versions will evaluate via reward/importance
    // and current workload reflected in local_memory.
    return 1.0f;
}

// ---- Path management ----------------------------------------------------

void DeliveryAgent::fetch_current_path(PDPGlobalMemory& memory) {
    local_memory.current_path =
        solution.empty() ? nullptr : solution.path_to_next(memory);
}

void DeliveryAgent::fetch_next_path(PDPGlobalMemory& memory) {
    local_memory.next_path =
        (solution.num_remaining() < 2) ? nullptr : solution.path_between(0, memory);
}

// ---- Planning -----------------------------------------------------------

void DeliveryAgent::plan(PDPGlobalMemory& memory, float speed_mps) {
    // 1. Refresh the cost matrix from GlobalMemory path cache.
    local_memory.operable_env.refresh_costs(memory);

    if (local_memory.local_agents.empty()) return;

    // 2. Build pickup/delivery pairings from the assigned task list.
    PairingMap pickup_of, delivery_of;
    for (const PDPTask* t : local_memory.tasks) {
        pickup_of [t->delivery.id] = t->pickup.id;
        delivery_of[t->pickup.id]  = t->delivery.id;
    }

    // 3. Run all LocalSolutionAgents; keep the lowest-cost sequence.
    std::vector<ObjectiveNode> best_seq;
    float                      best_cost = std::numeric_limits<float>::max();

    for (const LocalSolutionAgent& la : local_memory.local_agents) {
        std::vector<ObjectiveNode> candidate =
            la.plan(local_memory.operable_env, pickup_of, delivery_of, current_node);
        if (candidate.empty()) continue;

        float cost = 0.0f;
        for (std::size_t i = 0; i + 1 < candidate.size(); ++i) {
            int ai = local_memory.operable_env.find_index(candidate[i].id);
            int bi = local_memory.operable_env.find_index(candidate[i + 1].id);
            if (ai < 0 || bi < 0) { cost = std::numeric_limits<float>::max(); break; }
            float c = local_memory.operable_env.get_cost(ai, bi);
            if (c < 0.0f)         { cost = std::numeric_limits<float>::max(); break; }
            cost += c;
        }

        if (cost < best_cost) {
            best_cost = cost;
            best_seq  = std::move(candidate);
        }
    }

    if (best_seq.empty()) return;

    // 4. Rebuild solution sequence with the best candidate.
    solution.sequence.clear();
    for (const auto& node : best_seq)
        solution.push_back(node);

    // 5. Commit the plan to GlobalMemory (updates congestion + estimated arrivals).
    memory.commit_plan(agent_id, speed_mps);
}

// ---- Private ------------------------------------------------------------

void DeliveryAgent::add_task_to_memory(PDPTask& task) {
    local_memory.tasks.push_back(&task);
    local_memory.operable_env.add_task(task);
    local_memory.local_agents.emplace_back(task.delivery);
}
