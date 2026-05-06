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

    void add_task   (const PDPTask& task);
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

private:
    std::vector<float> costs_;   // flat N×N row-major; -1 = not computed
    // O(1) lookup maintained in sync with nodes[].
    std::unordered_map<osmium::object_id_type, int> index_map_;
};

#endif // OPERABLE_ENVIRONMENT_HPP
