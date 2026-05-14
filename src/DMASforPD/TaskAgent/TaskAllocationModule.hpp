#ifndef TASK_ALLOCATION_MODULE_HPP
#define TASK_ALLOCATION_MODULE_HPP

#include "DMASforPD/Utility/PDPTask.hpp"
#include "DMASforPD/GlobalMemory/ObjectiveCache.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>

class PDPGlobalMemory;

// ---- Task offer (task → agent) -----------------------------------------

// Information sent to a delivery agent when offering it a task.
struct TaskOffer {
    int              task_id;
    float            reward;        // current task reward (may be boosted on recall)
    float            importance;    // urgency / priority
    std::vector<int> prev_agents;   // agents called before (context for the recipient)
    int              recall_round = 0;  // 0 = first round, 1..N = recall index
    int              max_recalls  = 3;  // for normalisation in PolicyFeatures
};

// ---- Task Allocation Module (TAM) --------------------------------------
//
// Algorithm:
//   Two incremental Dijkstra searches run in parallel from the pickup node
//   (pickup_search_) and from the delivery node (delivery_search_).
//
//   Each expansion step:
//     1. Pickup search: finds the next objective node reachable from pickup.
//        If that node belongs to a task already assigned to an agent → record
//        the agent in the matrix with its pickup-side cost.
//     2. Delivery search: same, plus detect idle agents whose current_node
//        is expanded (direct allocation opportunity).
//     3. Any agent present in BOTH searches → plan-order check → offer task.
//        Score >= 0.5 → allocated.
//
//   Recall: if both searches are exhausted without allocation, re-offer the
//   task to the highest-scoring agents with an updated (boosted) reward.
//
// Call step() in a loop until is_allocated() or is_exhausted().
class TaskAllocationModule {
public:
    struct Params {
        int   max_recalls          = 3;
        float recall_reward_mult   = 1.5f;   // reward multiplier on each recall round
        float recall_import_mult   = 1.2f;   // importance multiplier on each recall round
        int   max_tasks_per_agent  = 1;       // capacity guard — TAM won't offer to full agents
        bool  always_accept        = false;   // bypass policy: accept unconditionally (TamAlwaysAccept ablation)
    };

    explicit TaskAllocationModule(PDPTask& task, Params p = {});

    // Expand one node from each side; try to allocate. Returns true if allocated.
    bool step(PDPGlobalMemory& memory, float speed_mps);

    bool is_allocated()  const { return allocated_; }
    bool is_exhausted()  const { return exhausted_;  }

private:
    // ---- Per-agent entry in the TAM matrix --------------------------------
    static constexpr float kNoContact = -1.0f;

    struct AgentEntry {
        float pickup_cost   = kNoContact;   // static cost from pickup-side Dijkstra
        float delivery_cost = kNoContact;   // static cost from delivery-side Dijkstra
        float score         = 0.0f;         // last score returned by the delivery agent
        int   call_count    = 0;
    };

    // ---- State ------------------------------------------------------------
    PDPTask& task_;
    Params   params_;

    // The TAM reuses ObjectiveGroupCache's incremental Dijkstra via discover_step().
    // pickup_group_id_ / delivery_group_id_ identify the caches to expand.
    int pickup_group_id_;
    int delivery_group_id_;

    // Matrix: agent_id → costs from each side.
    std::unordered_map<int, AgentEntry> matrix_;

    // Agents to recall, sorted by descending score.
    std::vector<int> recall_queue_;
    int recall_count_ = 0;

    bool allocated_ = false;
    bool exhausted_ = false;

    // ---- Helpers ----------------------------------------------------------

    // Build a set of current_node values for all agents with remaining capacity.
    std::unordered_set<osmium::object_id_type> idle_agent_positions(
        PDPGlobalMemory& memory) const;

    // Process a node found by the pickup-side search.
    void handle_pickup_node(osmium::object_id_type node, float cost, PDPGlobalMemory& memory);

    // Process a node found by the delivery-side search.
    // Returns true if a direct allocation was made (agent at that position).
    bool handle_delivery_node(osmium::object_id_type node, float cost,
                               PDPGlobalMemory& memory, float speed_mps);

    // Try all agents present on both sides; return true if one accepted.
    bool try_common_agents(PDPGlobalMemory& memory, float speed_mps);

    // Verify that the agent's plan has pickup before delivery (or neither).
    bool plan_order_valid(int agent_id, PDPGlobalMemory& memory) const;

    // Send offer to agent; update matrix score. Returns true if accepted (score >= 0.5).
    bool offer_to_agent(int agent_id, PDPGlobalMemory& memory, float speed_mps);

    // Execute one recall round (re-offer to best-scoring agents with boosted reward).
    bool do_recall(PDPGlobalMemory& memory, float speed_mps);

    // Rebuild recall_queue_ sorted by descending score.
    void build_recall_queue();
};

#endif // TASK_ALLOCATION_MODULE_HPP
