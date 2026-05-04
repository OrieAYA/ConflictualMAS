#ifndef DELIVERY_AGENT_HPP
#define DELIVERY_AGENT_HPP

#include "DMASforPD/Utility/AgentSolution.hpp"
#include "DMASforPD/Utility/PDPTask.hpp"
#include "DMASforPD/DeliveryAgent/OperableEnvironment.hpp"
#include "DMASforPD/DeliveryAgent/LocalSolutionAgent.hpp"
#include <optional>
#include <vector>

struct TaskOffer;    // defined in TaskAllocationModule.hpp
class PDPGlobalMemory;

// Status of a delivery agent at the task level.
enum class AgentStatus { Idle, Active, Done };

// State describing an agent currently traversing an edge (link-transmission model).
// While in transit, the agent is "on the edge" — not yet at to_node.
struct EdgeTransit {
    osmium::object_id_type edge_id;
    osmium::object_id_type from_node;
    osmium::object_id_type to_node;
    int                    arrival_step = -1;
};

// Local memory of a delivery agent — independent of GlobalMemory.
//
// Holds the agent's task list, operable environment (cost matrix), planning
// agents (one per delivery node), and the two pre-fetched paths.
//
// Update policy:
//   - tasks / operable_env / local_agents: updated on task assignment and completion.
//   - current_path / next_path: fetched once from GlobalMemory per leg.
//   - current_time: updated by the simulation each step.
//   - Cost matrix: refreshed from GlobalMemory only during planning.
struct DeliveryLocalMemory {
    std::vector<PDPTask*>          tasks;          // assigned tasks (non-owning pointers)
    const ObjectivePath*           current_path = nullptr; // path currently being traversed
    const ObjectivePath*           next_path    = nullptr; // queued for after current_path
    int                            current_time = 0;
    OperableEnvironment            operable_env;
    std::vector<LocalSolutionAgent> local_agents;  // one per delivery node
};

// Delivery agent — executes tasks assigned by the TaskAgent coordinator.
//
// Spatial state (link-transmission model):
//   - AtNode  : agent is at current_node, ready to move or make decisions.
//   - OnEdge  : agent is traversing in_transit.edge_id toward in_transit.to_node,
//               arrives at in_transit.arrival_step.
//
// Planning model (legacy MetaAgent pattern adapted for PDP):
//   - local_memory.local_agents run DbVNS-PDP from their delivery node.
//   - plan() collects candidates, selects best, rebuilds solution, commits to GlobalMemory.
class DeliveryAgent {
public:
    int                        agent_id     = 0;
    AgentStatus                status       = AgentStatus::Idle;
    osmium::object_id_type     current_node = 0;
    std::optional<EdgeTransit> in_transit;
    AgentSolution              solution;
    DeliveryLocalMemory        local_memory;

    DeliveryAgent() = delete;
    DeliveryAgent(int id, osmium::object_id_type start_node);

    DeliveryAgent(const DeliveryAgent&)            = delete;
    DeliveryAgent& operator=(const DeliveryAgent&) = delete;

    // ---- Spatial state --------------------------------------------------

    bool is_at_node() const { return !in_transit.has_value(); }
    bool is_on_edge() const { return  in_transit.has_value(); }

    // Start traversing an edge. Sets in_transit; current_node stays at from_node.
    void start_edge(osmium::object_id_type edge_id,
                    osmium::object_id_type to_node,
                    int arrival_step);

    // Call when the simulation step reaches in_transit.arrival_step.
    // Clears in_transit and advances current_node to the destination.
    void arrive_at_node();

    // ---- GlobalMemory interface -----------------------------------------

    void connect   (PDPGlobalMemory& memory);
    void disconnect(PDPGlobalMemory& memory);

    // Accept a task: register in local memory, push pickup/delivery into solution.
    // Does NOT call commit_plan — TAM does so immediately after.
    void receive_task(PDPTask& task, PDPGlobalMemory& memory);

    // Remove a completed task from local memory and the operable environment.
    // Called after a task's Finished event has been processed.
    void remove_completed_task(int task_id);

    // Called by TAM with a task offer. Returns score in [0,1]; >= 0.5 = accept.
    float try_accept_task(const TaskOffer& offer, PDPGlobalMemory& memory);

    // ---- Path management ------------------------------------------------

    // Fetch path to the next objective; store in local_memory.current_path.
    // Call once per leg (not on every step).
    void fetch_current_path(PDPGlobalMemory& memory);

    // Fetch path from next objective to the one after; store in local_memory.next_path.
    // Call once per leg (not on every step).
    void fetch_next_path(PDPGlobalMemory& memory);

    // ---- Planning -------------------------------------------------------

    // Refresh operable environment costs, run all LocalSolutionAgents, select
    // the best candidate sequence, rebuild solution, and commit_plan to GlobalMemory.
    void plan(PDPGlobalMemory& memory, float speed_mps);

private:
    // Register task in local_memory: add to task list, expand operable environment,
    // create a LocalSolutionAgent anchored at the delivery node.
    void add_task_to_memory(PDPTask& task);
};

#endif // DELIVERY_AGENT_HPP
