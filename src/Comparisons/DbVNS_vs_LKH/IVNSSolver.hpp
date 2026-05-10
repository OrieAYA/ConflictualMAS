#ifndef IVNS_SOLVER_HPP
#define IVNS_SOLVER_HPP

#include "DMASforPD/DeliveryAgent/OperableEnvironment.hpp"
#include "DMASforPD/DeliveryAgent/LocalSolutionAgent.hpp"
#include <vector>
#include <limits>

// Insertion-based VNS with free-zone 2-opt (IVNS).
//
// Key ideas vs DbVNS:
//   - No anchor: seed = most-spread pair, all others inserted at cheapest position.
//   - Free zones: maximal route subsequences with no intra-zone P-D pair.
//     These are pure TSP segments — unrestricted 2-opt applied there directly.
//   - VNS shake: remove k highest-cost pairs, reinsert at cheapest positions.
//   - After each shake: free-zone 2-opt + PDP-constrained 2-opt + or-opt.
struct IVNSSolver {
    struct Result {
        std::vector<ObjectiveNode> sequence;
        float     cost         = std::numeric_limits<float>::max();
        long long exec_ms      = 0;
        int       iters_done   = 0;
    };

    static Result solve(
        const OperableEnvironment& env,
        const PairingMap&          pickup_of,
        int max_iterations = 50,
        int max_k          = 4);

    // Insert a new (pickup, delivery) pair into an existing IVNS result in-place.
    //
    // Preconditions:
    //   - env.add_task() and env.refresh_costs() have already been called for the new pair.
    //   - pickup_of includes the new pair (delivery_id → pickup_id).
    //   - p_idx / d_idx are the env indices of the new pickup and delivery nodes.
    //
    // Finds the cheapest valid insertion position via pair_delta, applies it,
    // then runs one full refinement pass (free-zone 2-opt + PDP 2-opt + or-opt).
    // Handles empty existing.sequence gracefully.
    static void insert_pair(
        Result&                    existing,
        const OperableEnvironment& env,
        const PairingMap&          pickup_of,
        int                        p_idx,
        int                        d_idx);
};

#endif // IVNS_SOLVER_HPP
