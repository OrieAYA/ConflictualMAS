#ifndef TASK_ALLOCATION_MODULE_HPP
#define TASK_ALLOCATION_MODULE_HPP

#include "DMASforPD/Structures/PDPTask.hpp"
#include "DMASforPD/Structures/ObjectiveCache.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>

class PDPGlobalMemory;

// ---- Task offer (task → agent) -----------------------------------------

// Information sent to a delivery agent when it is asked to evaluate a task.
struct TaskOffer {
    int              task_id;
    float            reward;
    float            importance;
    // Competition context, populated before each bid_for_task call so the
    // policy can build marginal_cost_relative and fleet_pressure without
    // re-scanning the TAM internals. Costs are static road distances (m).
    int   n_candidates_total  = 0;   // size of the candidate set this round
    float my_insertion_cost   = 0.f; // pickup_cost + delivery_cost for this agent
    float cheapest_other_cost = 0.f; // min insertion cost across the OTHER candidates
                                     // (0 if this agent is the only candidate)
    int   mc_max_candidates   = 5;   // K (for fleet_pressure normalisation)
};

// ---- Task Allocation Module (TAM) ---------------------------------------
//
// Multi-candidate allocation (paper Algorithm 1):
//
//   1. Two incremental Dijkstra searches expand from the pickup node and the
//      delivery node (static road distance, meters). An agent becomes a
//      VALID CANDIDATE when it is reachable from at least one side with a
//      plan-order-compatible sequence (pickup before delivery), or when it
//      is idle and discovered at its standing position.
//   2. When the first candidate is found at combined cost x, the expansion
//      budget is fixed at x * ratio(x), ratio decreasing from ratio_max to
//      ratio_min with decay length ratio_scale (meters).
//   3. Expansion stops at K candidates, budget exhaustion, or graph
//      exhaustion. Candidates are kept sorted by ascending combined cost.
//   4. Every candidate is asked to bid (DeliveryAgent::bid_for_task → policy
//      score μ + accept/refuse bid). The winner is the argmax-μ agent among
//      BIDDERS. If nobody bids:
//        - force_assign=true  → argmax-μ overall wins, allocation flagged
//          FORCED (the policy did not choose it; the runner must not credit
//          the task outcome to that buffer entry).
//        - force_assign=false → the task is DEFERRED (runner re-offers later).
//
// Call step() in a loop until is_allocated() or is_exhausted().
class TaskAllocationModule {
public:
    struct Params {
        int   max_tasks_per_agent = 5;     // capacity ceiling for idle discovery
        bool  always_accept       = false; // ablation: every candidate bids with μ=1

        bool  force_assign    = true;      // Format A (true) vs Format B (false)
        int   max_candidates  = 5;         // K
        float ratio_min       = 1.4f;
        float ratio_max       = 3.0f;
        float ratio_scale     = 2000.0f;   // meters — ratio(x) decay length
    };

    explicit TaskAllocationModule(PDPTask& task, Params p = {});

    // Expand one node from each side; finalise when the collection criterion
    // is met. Returns true once the session is finished.
    bool step(PDPGlobalMemory& memory, float speed_mps);

    bool is_allocated() const { return allocated_; }
    bool is_exhausted() const { return exhausted_; }

    // Format B only: every candidate refused (no bid) and force_assign is
    // off. The task is neither allocated nor unreachable — re-offer it later.
    bool is_deferred()  const { return deferred_; }

    // True when the winner did NOT bid (all candidates refused, Format A).
    // The allocation is the system's override, not the policy's action.
    bool was_forced()   const { return forced_; }

    int  winner_agent() const { return winner_; }

    // ── Reward-pairing accessors ──────────────────────────────────────────
    // candidates_in_order()[i] produced the i-th buffer record of this
    // session; bids_in_order()[i] tells whether that candidate bid. The
    // runner uses these to bind the winner's buffer entry to the task
    // outcome and to apply the non-affected penalty to losing bidders.
    const std::vector<int>&  candidates_in_order() const { return candidates_; }
    const std::vector<bool>& bids_in_order()       const { return bids_; }
    const std::vector<float>& scores_in_order()    const { return scores_; }

    // ── Observability ─────────────────────────────────────────────────────
    int n_agents_offered()    const { return static_cast<int>(candidates_.size()); }
    int n_candidates_scored() const { return static_cast<int>(candidates_.size()); }
    int n_recall_rounds()     const { return 0; }   // legacy metric — MC has no recall

private:
    static constexpr float kNoContact = -1.0f;

    struct AgentEntry {
        float pickup_cost   = kNoContact;
        float delivery_cost = kNoContact;
        osmium::object_id_type pickup_node   = 0;   // agent node reached per side
        osmium::object_id_type delivery_node = 0;
        bool idle_from_pickup = false;
    };

    PDPTask& task_;
    Params   params_;

    int pickup_group_id_;
    int delivery_group_id_;

    std::unordered_map<int, AgentEntry> matrix_;

    bool allocated_ = false;
    bool exhausted_ = false;
    bool deferred_  = false;
    bool forced_    = false;
    int  winner_    = -1;

    bool first_found_       = false;
    bool search_reset_done_ = false;
    std::vector<int>   candidates_;  // candidate agent ids, ascending cost order
    std::vector<bool>  bids_;        // filled by finalise(), aligned to candidates_
    std::vector<float> scores_;      // μ per candidate, aligned to candidates_

    // Expansion budget for both Dijkstra searches (meters). +inf until the
    // first candidate fixes it at x * ratio(x).
    float max_search_cost_ = -1.0f;

    // Snapshot of standing positions of agents with remaining capacity,
    // computed once per session.
    std::unordered_set<osmium::object_id_type> idle_positions_cache_;
    bool                                       idle_positions_dirty_ = true;

    std::unordered_set<osmium::object_id_type> idle_agent_positions(
        PDPGlobalMemory& memory) const;

    // ratio(x) = ratio_min + (ratio_max - ratio_min) / (1 + x/scale).
    float ratio(float x) const;

    // Rebuild candidates_ from matrix_ (valid = idle-from-pickup, or busy with
    // pickup node before delivery node in plan), sorted by ascending cost,
    // capped at max_candidates.
    void collect_candidates(PDPGlobalMemory& memory);

    // Ask every candidate to bid, pick the winner, allocate / defer.
    bool finalise(PDPGlobalMemory& memory, float speed_mps);

    // assign + receive_task + commit_plan.
    void allocate_to(int agent_id, PDPGlobalMemory& memory, float speed_mps);
};

#endif // TASK_ALLOCATION_MODULE_HPP
