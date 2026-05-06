#include "TaskAgent.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include <algorithm>
#include <memory>

TaskAgent::TaskAgent(int id, const TaskAgentParams& p)
    : agent_id(id), params(p) {}

// ---- Fleet management --------------------------------------------------

void TaskAgent::add_delivery_agent(DeliveryAgent& agent) {
    agents_.push_back(&agent);
}

void TaskAgent::remove_delivery_agent(int delivery_agent_id) {
    agents_.erase(
        std::remove_if(agents_.begin(), agents_.end(),
            [delivery_agent_id](const DeliveryAgent* a){
                return a->agent_id == delivery_agent_id;
            }),
        agents_.end()
    );
}

int TaskAgent::num_agents() const { return static_cast<int>(agents_.size()); }

const DeliveryAgent* TaskAgent::get_agent(int id) const {
    for (const auto* a : agents_) if (a->agent_id == id) return a;
    return nullptr;
}
DeliveryAgent* TaskAgent::get_agent(int id) {
    for (auto* a : agents_) if (a->agent_id == id) return a;
    return nullptr;
}

// ---- Task lifecycle ----------------------------------------------------

void TaskAgent::on_new_task(int task_id, PDPGlobalMemory& memory) {
    PDPTask* task = memory.get_task(task_id);
    if (!task) return;
    task->start_allocation();
    active_tams_.emplace(task_id,
        std::make_unique<TaskAllocationModule>(*task, params.tam_params));
}

void TaskAgent::on_task_event(int task_id, TaskEvent event, PDPGlobalMemory& memory) {
    PDPTask* task = memory.get_task(task_id);
    if (!task) return;

    const int t = memory.current_time();

    switch (event) {
    case TaskEvent::TowardsPickup:
        task->mark_towards_pickup(t);
        break;

    case TaskEvent::Picked:
        task->mark_picked(t);
        // Clear the pickup node from its group cache — avoids stale path computations.
        memory.clear_objective(task->pickup.id, task->pickup.group_id);
        break;

    case TaskEvent::TowardsDelivery:
        task->mark_towards_delivery(t);
        break;

    case TaskEvent::Delivered:
        task->mark_delivered(t);
        // Clear the delivery node from its group cache.
        memory.clear_objective(task->delivery.id, task->delivery.group_id);
        break;

    case TaskEvent::Finished:
        memory.complete_task(task_id);
        if (DeliveryAgent* agent = memory.get_agent_for_task(task_id)) {
            agent->remove_completed_task(task_id);
            agent->status = AgentStatus::Idle;
        }
        break;
    }
}

// ---- Simulation hook ---------------------------------------------------

void TaskAgent::step(PDPGlobalMemory& memory) {
    for (auto it = active_tams_.begin(); it != active_tams_.end(); ) {
        auto& [task_id, tam] = *it;

        if (tam->is_allocated() || tam->is_exhausted()) {
            it = active_tams_.erase(it);
            continue;
        }

        tam->step(memory, params.default_speed_mps);
        ++it;
    }
}
