#ifndef LKH_SOLVER_HPP
#define LKH_SOLVER_HPP

#include "DMASforPD/DeliveryAgent/OperableEnvironment.hpp"
#include "DMASforPD/DeliveryAgent/LocalSolutionAgent.hpp"
#include <vector>

// Lin-Kernighan-Helsgott inspired TSP-PC solver (LKH core).
//
// For N~130-190 stops with PDP precedence constraints:
//   1. Nearest Neighbor construction respecting P_i before D_i.
//   2. 2-opt: O(N^2) segment reversals that reduce cost and preserve PDP.
//      PDP check is O(segment_len/2) per candidate.
//   3. Or-opt (single-node relocation): O(1) PDP check per insertion position.
//   4. Multiple restarts from different pickup nodes; keep best solution.
//
// Expected: near-optimal tours for N<200 in under 200 ms.
struct LKHSolver {
    struct Result {
        std::vector<ObjectiveNode> sequence;
        float     cost;           // sum of env edge costs (depot leg excluded)
        long long exec_ms;
        int       restarts_done;
    };

    // Solve TSP-PC on env.nodes.
    // depot is NOT in env — caller adds depot leg via sequence_full_cost.
    static Result solve(const OperableEnvironment& env,
                        const PairingMap&          pickup_of,
                        int                        max_restarts = 10);
};

#endif // LKH_SOLVER_HPP
