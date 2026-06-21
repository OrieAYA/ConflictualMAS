#ifndef DELIVERY_AGENT_HPP
#define DELIVERY_AGENT_HPP

#include "DMASforPD/Structures/AgentSolution.hpp"
#include "DMASforPD/Structures/PDPTask.hpp"
#include "DMASforPD/Structures/OperableEnvironment.hpp"
#include "DMASforPD/Algorithms/DbVNS.hpp"
#include "DMASforPD/Structures/ObjectiveCache.hpp"
#include <optional>
#include <vector>

struct TaskOffer;
class PDPGlobalMemory;

enum class AgentStatus { Idle, Active, Done };

// State of a single edge traversal (link-transmission model).
struct EdgeTransit {
    osmium::object_id_type edge_id;
    osmium::object_id_type from_node;
    osmium::object_id_type to_node;
    int                    arrival_step = -1;
};

// ════════════════════════════════════════════════════════════════════════════
// EdgeCursor — tracks position within a multi-edge path traversal
// ════════════════════════════════════════════════════════════════════════════
//
// When an agent starts a leg (current_node → objective), it loads the
// ObjectivePath's node and edge sequences into an EdgeCursor.
//
// Invariant: agent is currently at nodes[next_idx].
//   - edges[next_idx] is the next edge to traverse.
//   - When next_idx == edges.size(), the agent is at the objective (last node).
//
// Fallback (no cached path): edges is empty, nodes = {from, to}.  The
// simulator schedules one synthetic hop to cover the full distance.
struct EdgeCursor {
    std::vector<osmium::object_id_type> nodes;     // path nodes (incl. start and end)
    std::vector<osmium::object_id_type> edges;     // way IDs between consecutive nodes
    int  next_idx  = 0;    // index of next edge to traverse; agent is at nodes[next_idx]
    int  task_id   = -1;
    bool is_pickup = true;

    // True while there are still edges to traverse before reaching the objective.
    bool has_more_edges() const { return next_idx < static_cast<int>(edges.size()); }

    // True when the agent is one edge away from the objective.
    bool is_last_edge()   const { return next_idx + 1 == static_cast<int>(edges.size())
                                      || edges.empty(); }

    // The node the agent will arrive at after completing the current edge.
    osmium::object_id_type next_node() const {
        int dest_idx = next_idx + 1;
        return (dest_idx < static_cast<int>(nodes.size())) ? nodes[dest_idx] : 0;
    }

    // The way ID of the current edge (0 if fallback/no real edge).
    osmium::object_id_type current_edge_id() const {
        return (next_idx < static_cast<int>(edges.size())) ? edges[next_idx] : 0;
    }
};

// Local agent memory — independent of GlobalMemory.
//
// current_path : path being traversed right now (set at leg start, updated on
//                congestion-push from GlobalMemory).
// next_path    : pre-fetched path for the leg after the current objective
//                (promoted to current_path when the current objective is reached).
//
// The two-path lookahead means GlobalMemory can push an updated current_path
// mid-traversal without the agent losing its upcoming route.
struct DeliveryLocalMemory {
    std::vector<PDPTask*>           tasks;
    const ObjectivePath*            current_path = nullptr;
    const ObjectivePath*            next_path    = nullptr;
    int                             current_time = 0;
    OperableEnvironment             operable_env;
    std::vector<LocalSolutionAgent> local_agents;

    // Agent-owned storage for a congestion reroute. current_path points here
    // when the Manager pushes a TD-A* alternative (the path cache only stores
    // static shortest paths; rerouted geometry needs a stable owner).
    ObjectivePath                   reroute_path;
};

// ════════════════════════════════════════════════════════════════════════════
// DeliveryAgent
// ════════════════════════════════════════════════════════════════════════════
class DeliveryAgent {
public:
    int                        agent_id     = 0;
    AgentStatus                status       = AgentStatus::Idle;
    osmium::object_id_type     current_node = 0;
    std::optional<EdgeTransit> in_transit;
    std::optional<EdgeCursor>  edge_cursor;    // active path traversal state
    AgentSolution              solution;
    DeliveryLocalMemory        local_memory;

    // Per-agent carrying capacity. 0 = inherit TAM-global value
    // (task_agent.params.max_tasks_per_agent). >0 = override, used by Option M
    // to inject fleet heterogeneity (each agent gets a capacity drawn in
    // [min, max] at episode start). receive_task() respects this when
    // computing capacity-aware insertion positions.
    int                        max_capacity = 0;

    DeliveryAgent() = delete;
    DeliveryAgent(int id, osmium::object_id_type start_node);

    DeliveryAgent(const DeliveryAgent&)            = delete;
    DeliveryAgent& operator=(const DeliveryAgent&) = delete;

    // ---- Spatial state --------------------------------------------------

    bool is_at_node()    const { return !in_transit.has_value(); }
    bool is_on_edge()    const { return  in_transit.has_value(); }
    bool is_traversing() const { return  edge_cursor.has_value(); }

    void start_edge(osmium::object_id_type edge_id,
                    osmium::object_id_type to_node,
                    int arrival_step);

    // Advance current_node to the destination of the completed edge; clear in_transit.
    void arrive_at_node();

    // ---- GlobalMemory interface -----------------------------------------

    void connect   (PDPGlobalMemory& memory);
    void disconnect(PDPGlobalMemory& memory);

    void  receive_task        (PDPTask& task, PDPGlobalMemory& memory);
    void  remove_completed_task(int task_id);

    // Result of asking this agent to evaluate a task offer.
    //   mu  — the policy's action score ∈ [0,1] (paper: "promising-ness").
    //   bid — the agent's sampled/deterministic accept decision. The TAM
    //         allocates to the argmax-μ agent AMONG BIDDERS; if nobody bids
    //         it force-assigns the argmax-μ agent and flags the allocation
    //         as forced (the runner then does NOT credit task outcomes to
    //         that buffer entry — the assignment was the system's override,
    //         not the policy's action).
    struct BidResult { float mu = 0.f; bool bid = false; };

    // Build the 12-d feature vector, query the active policy, sample the bid
    // (Bernoulli(μ) when memory.exploration_enabled, μ>=0.5 otherwise) and
    // record the experience into the active policy's buffer. RMCA mode
    // returns a deterministic score with bid=true and records nothing.
    BidResult bid_for_task    (const TaskOffer& offer, PDPGlobalMemory& memory);

    // Compute a bid score for InsertionGreedy baseline (no buffer recording).
    // Returns (reward * importance) / max(insertion_cost, ε). Higher = better task.
    float compute_bid         (const TaskOffer& offer, PDPGlobalMemory& memory);

    // ---- Path management (two-path lookahead) ---------------------------

    // Load the path for the current leg into local_memory.current_path and
    // initialise edge_cursor.  Call at the start of each new leg.
    // Returns false if no valid path is available.
    bool begin_leg(const ObjectivePath* path, int task_id, bool is_pickup);

    // Promote next_path → current_path; reset next_path.
    // Called automatically when arriving at an objective.
    void promote_next_path();

    // Fetch the path from sequence[0] → sequence[1] into local_memory.next_path.
    // No-op if fewer than 2 objectives remain.
    void prefetch_next_path(PDPGlobalMemory& memory);

    // Push an updated path from GlobalMemory (congestion reroute).
    // Replaces current_path and rebuilds the edge cursor from the agent's
    // current position within the new path.
    void push_updated_path(const ObjectivePath* new_path);

    // Legacy helpers (kept for TAM / planning compatibility).
    void fetch_current_path(PDPGlobalMemory& memory);
    void fetch_next_path   (PDPGlobalMemory& memory);

    // ---- Planning -------------------------------------------------------

    void plan(PDPGlobalMemory& memory, float speed_mps);

private:
    void add_task_to_memory(PDPTask& task);
};

#endif // DELIVERY_AGENT_HPP
