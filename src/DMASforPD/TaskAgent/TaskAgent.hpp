#ifndef TASK_AGENT_HPP
#define TASK_AGENT_HPP

#include "DMASforPD/DeliveryAgent/DeliveryAgent.hpp"
#include "DMASforPD/TaskAgent/TaskAllocationModule.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

class PDPGlobalMemory;

struct TaskAgentParams {
    int   max_tasks_per_agent = 5;
    float default_speed_mps   = 10.0f;  // meters per step (used for commit_plan)
    TaskAllocationModule::Params tam_params;
};

// Events sent by delivery agents to notify task progress.
enum class TaskEvent {
    TowardsPickup,
    Picked,
    TowardsDelivery,
    Delivered,
    Finished
};

// Coordinator agent for the PDP multi-agent system.
//
// Responsibilities:
//   - Maintains the fleet of delivery agents.
//   - Launches a TaskAllocationModule (TAM) for each new task.
//   - Drives TAM steps via step() until allocation.
//   - Receives task events from delivery agents and updates task status.
//   - Clears completed objective nodes from the path cache.
class TaskAgent {
public:
    int             agent_id;
    TaskAgentParams params;

    TaskAgent() = delete;
    explicit TaskAgent(int id, const TaskAgentParams& p = {});

    TaskAgent(const TaskAgent&)            = delete;
    TaskAgent& operator=(const TaskAgent&) = delete;

    // ---- Fleet management -----------------------------------------------

    void add_delivery_agent   (DeliveryAgent& agent);
    void remove_delivery_agent(int delivery_agent_id);

    int                  num_agents()      const;
    const DeliveryAgent* get_agent(int id) const;
    DeliveryAgent*       get_agent(int id);

    // ---- Task lifecycle -------------------------------------------------

    // Register a new task and start the TAM for it.
    void on_new_task(int task_id, PDPGlobalMemory& memory);

    // Called by the simulation / delivery agent when a milestone is reached.
    // Updates task status and clears the objective cache on Picked / Delivered.
    void on_task_event(int task_id, TaskEvent event, PDPGlobalMemory& memory);

    // ---- Simulation hook ------------------------------------------------

    // Called once per simulation step.
    // Advances all running TAMs; allocates tasks as they are accepted.
    void step(PDPGlobalMemory& memory);

    // Access a running TAM for synchronous driving (training pipeline).
    // Returns nullptr if no TAM is active for this task.
    TaskAllocationModule* get_tam(int task_id) {
        auto it = active_tams_.find(task_id);
        return (it == active_tams_.end()) ? nullptr : it->second.get();
    }
    void erase_tam(int task_id) { active_tams_.erase(task_id); }

private:
    std::vector<DeliveryAgent*> agents_;

    // Active TAM instances indexed by task_id.
    // unique_ptr avoids issues with TAM's non-movable reference member (PDPTask&).
    std::unordered_map<int, std::unique_ptr<TaskAllocationModule>> active_tams_;
};

#endif // TASK_AGENT_HPP
