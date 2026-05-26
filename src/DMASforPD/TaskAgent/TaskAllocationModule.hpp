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
    // ── MC TAM context (V2 input vector) ─────────────────────────────────────
    // Populated by mc_score_agent before calling try_accept_task so the policy
    // can build marginal_cost_relative and fleet_pressure without re-scanning
    // the TAM internals. Zero-defaults are safe for non-MC code paths.
    int   n_candidates_total = 0;   // size of mc_candidates_ on this round
    float my_insertion_cost   = 0.f;// pickup_cost + delivery_cost for this agent
    float cheapest_other_cost = 0.f;// min insertion cost across the OTHER candidates
                                    // (0 if this agent is the only candidate)
    int   mc_max_candidates   = 5;  // matches params_.mc_max_candidates (for norm)
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

        // Spatial pruning: bound the Dijkstra expansion (per side) to
        //   haversine(pickup, delivery) * search_radius_mult.
        // Each recall multiplies the budget by recall_radius_grow so the search
        // resumes past the previous frontier when more agents are needed.
        float search_radius_mult   = 3.0f;
        float recall_radius_grow   = 2.0f;

        // ── Multi-candidate mode (opt-in, see EpisodeConfig::tam_multi_candidate)
        // When multi_candidate=false (default) step() runs the legacy first-fit
        // path unchanged. When true, step() collects up to mc_max_candidates
        // agents via adaptive Dijkstra expansion, scores them all, and assigns
        // the argmax (force_assign) or defers if every score < 0.5.
        bool  multi_candidate  = false;
        bool  mc_force_assign  = true;     // Format A (true) vs Format B (false)
        int   mc_max_candidates= 5;
        float mc_ratio_min     = 1.4f;
        float mc_ratio_max     = 3.0f;
        float mc_ratio_scale   = 2000.0f;  // metres — ratio(x) decay length
    };

    explicit TaskAllocationModule(PDPTask& task, Params p = {});

    // Expand one node from each side; try to allocate. Returns true if allocated.
    bool step(PDPGlobalMemory& memory, float speed_mps);

    bool is_allocated()  const { return allocated_; }
    bool is_exhausted()  const { return exhausted_;  }

    // Multi-candidate Format B: true when the TAM finished a session with
    // every candidate scoring below 0.5 and force_assign disabled. The task
    // was neither allocated nor genuinely unreachable — the caller (EpisodeRunner)
    // should re-offer it later. Always false in legacy mode and in Format A.
    bool is_deferred()   const { return mc_deferred_; }

    // ── Observability (TAM-efficiency metrics) ─────────────────────────────
    // These getters let EpisodeRunner aggregate per-task TAM behaviour into
    // episode-level ComparisonMetrics columns. They are read-only and zero-
    // cost (just expose state already maintained).
    //
    // n_agents_offered  : how many distinct agents the TAM actually asked
    //                     (offer_to_agent called at least once). For the
    //                     paper's "minimise communication overhead" claim,
    //                     this is THE number to track — lower than the
    //                     fleet size means the responsibility-based
    //                     decomposition is paying off.
    // n_recall_rounds   : how many recall rounds the TAM ran before
    //                     allocating / exhausting. 0 = first-pass success.
    // n_candidates_scored : multi-candidate mode only; size of the
    //                     candidate set scored at finalise. 0 in legacy
    //                     mode (use n_agents_offered instead).
    int n_agents_offered()    const;
    int n_recall_rounds()     const { return recall_count_; }
    int n_candidates_scored() const { return static_cast<int>(mc_candidates_.size()); }

    // ── MC TAM reward-shaping accessor ───────────────────────────────────────
    // After mc_finalise() runs, EpisodeRunner reads mc_candidates_in_order()
    // to pair each scored agent with the buffer entry it produced (1 entry
    // per candidate, in the order they were scored). The winner's buf_idx
    // becomes task_accept_buf_idx_[task_id]; the others get the non-affected
    // penalty.
    const std::vector<int>& mc_candidates_in_order() const { return mc_candidates_; }

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

    // ── Multi-candidate state (only used when params_.multi_candidate) ──────
    bool  mc_deferred_          = false;  // Format B: all candidates scored < 0.5
    bool  mc_first_found_       = false;  // first valid candidate seen → budget set
    bool  mc_search_reset_done_ = false;  // search states reset once per allocation
    std::vector<int> mc_candidates_;      // collected candidate agent ids (scoring order)

    // Spatial pruning budget for both Dijkstra searches (meters).
    // Legacy mode : initialised on first step() as haversine(pickup,delivery)
    //   * search_radius_mult; each recall multiplies by recall_radius_grow.
    // MC mode     : starts at +inf (no budget) until the first candidate is
    //   found, then set to x*ratio(x) where x=max(pickup_cost,delivery_cost).
    // Negative sentinel means "not yet initialised".
    float max_search_cost_ = -1.0f;

    // Cached snapshot of agents with capacity for this TAM session. Computed
    // once per session (or per recall) instead of once per step() — the fleet
    // state does not change inside one TAM allocation, so re-scanning all
    // agents on every Dijkstra expansion is pure waste (up to 300×/task).
    std::unordered_set<osmium::object_id_type> idle_positions_cache_;
    bool                                       idle_positions_dirty_ = true;

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

    // ── Multi-candidate mode helpers ───────────────────────────────────────

    // Entry point for params_.multi_candidate. Called by step() instead of the
    // legacy path. Expands the dual Dijkstra, collects candidates, and once the
    // collection criterion is met scores them and allocates the argmax (or
    // defers). Returns true once the session is finished (allocated/deferred/
    // exhausted), matching step()'s contract.
    bool step_multi_candidate(PDPGlobalMemory& memory, float speed_mps);

    // ratio(x) for the adaptive expansion budget, decreasing in x:
    //   ratio(x) = mc_ratio_min + (mc_ratio_max - mc_ratio_min) / (1 + x/scale)
    // x=0 → mc_ratio_max ; x→∞ → mc_ratio_min.
    float mc_ratio(float x) const;

    // Rebuild mc_candidates_ from the current matrix_: any agent reachable
    // from at least one Dijkstra side with a valid plan order.
    // Sorted ascending by combined cost, capped at mc_max_candidates.
    void mc_collect_candidates(PDPGlobalMemory& memory);

    // Score every collected candidate via try_accept_task, pick the argmax,
    // and either allocate (force_assign or best >= 0.5) or set mc_deferred_.
    // Returns true (the session always ends here).
    bool mc_finalise(PDPGlobalMemory& memory, float speed_mps);

    // Score one agent without allocating (used by mc_finalise). Mirrors the
    // policy call in offer_to_agent but never assigns the task.
    float mc_score_agent(int agent_id, PDPGlobalMemory& memory);

    // Allocate the task to agent_id (assign + receive_task + commit). Shared
    // by the legacy offer_to_agent and mc_finalise.
    void mc_allocate(int agent_id, PDPGlobalMemory& memory, float speed_mps);
};

#endif // TASK_ALLOCATION_MODULE_HPP
