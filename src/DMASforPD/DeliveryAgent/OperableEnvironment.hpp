#ifndef OPERABLE_ENVIRONMENT_HPP
#define OPERABLE_ENVIRONMENT_HPP

#include "DMASforPD/Utility/PDPTask.hpp"
#include <unordered_map>
#include <vector>

class PDPGlobalMemory;

// N×N minimum-cost matrix over all objective nodes assigned to a delivery agent.
// N = 2 * number_of_tasks (one pickup + one delivery per task).
//
// Costs (meters) are populated from GlobalMemory's path cache during the planning phase.
// Entries not yet computed are stored as -1.
//
// Usage:
//   add_task()      → on task assignment (expands the matrix)
//   remove_task()   → on task completion (compacts the matrix)
//   refresh_costs() → during planning only
struct OperableEnvironment {
    std::vector<ObjectiveNode> nodes;  // nodes[i] — all objective nodes in insertion order

    // ---- Task management ------------------------------------------------

    void add_task       (const PDPTask& task);
    void add_single_node(const ObjectiveNode& node);  // add one node (no pairing)
    void remove_task(osmium::object_id_type pickup_id,
                     osmium::object_id_type delivery_id);

    // ---- Lookup ---------------------------------------------------------

    int  find_index(osmium::object_id_type node_id) const;  // -1 if absent
    int  size      ()                               const;

    // ---- Cost access ----------------------------------------------------

    void  set_cost(int i, int j, float cost);
    float get_cost(int i, int j) const;   // -1.0f if not yet computed
    bool  has_cost(int i, int j) const;

    // Query GlobalMemory path cache for all missing (i,j) pairs.
    // Must be called only during the planning phase.
    void refresh_costs(PDPGlobalMemory& memory);

    // ---- Time-aware (congestion-BPR) cost -------------------------------
    //
    // When a planning time context is set, get_cost_at(i,j,depart) returns the
    // BPR congestion-adjusted cost of the static path i→j departing at step
    // `depart` (memoised per (i,j,time-bucket)); without a context it falls
    // back to the static matrix entry. This is what turns the DbVNS
    // decomposition tree into the operable environment: each branch is costed
    // at the time it is actually traversed (deeper = later t). The static
    // matrix is retained as geometry source + admissible lower bound.
    void  set_time_context(PDPGlobalMemory& memory, float speed_mps, int base_step);
    bool  time_aware() const { return mem_ctx_ != nullptr; }
    float plan_speed() const { return speed_ctx_; }
    int   plan_base () const { return base_ctx_;  }
    float get_cost_at(int i, int j, int depart_step) const;

private:
    std::vector<float> costs_;   // flat N×N row-major; -1 = not computed
    // O(1) lookup maintained in sync with nodes[].
    std::unordered_map<osmium::object_id_type, int> index_map_;

    // Planning time context (set per replanning; null = static fallback).
    PDPGlobalMemory* mem_ctx_   = nullptr;
    float            speed_ctx_ = 1.f;
    int              base_ctx_  = 0;
    mutable std::unordered_map<long long, float> tcost_memo_;  // (i,j,bucket) → BPR cost
};

#endif // OPERABLE_ENVIRONMENT_HPP
