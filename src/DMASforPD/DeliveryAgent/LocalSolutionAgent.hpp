#ifndef LOCAL_SOLUTION_AGENT_HPP
#define LOCAL_SOLUTION_AGENT_HPP

#include "DMASforPD/Utility/ObjectiveNode.hpp"
#include <unordered_map>
#include <vector>

struct OperableEnvironment;

// Pickup/delivery pairing lookup for DbVNS constraint management.
// pickup_of [delivery_id] = pickup_id
// delivery_of[pickup_id]  = delivery_id
using PairingMap = std::unordered_map<osmium::object_id_type, osmium::object_id_type>;

// Parameters for the DbVNS-PDP search.
struct DbVNSParams {
    int    max_iterations     = 100;
    double max_divergence     = 0.7;  // allow branches up to max_divergence worse than pbest
    int    max_decompositions = 5;    // max promising branches explored per shake
    int    k_max              = 4;    // shake depth factor
};

// Parameters for the ALNS-PDP search (Ropke & Pisinger 2006, mono-agent variant).
//
//   - Destroy operators : Shaw (similarity), random, worst removal
//   - Repair  operators : cheapest insertion, regret-2 insertion (positional)
//   - Acceptance        : simulated annealing
//   - Adaptive weights  : roulette wheel updated per segment of `segment_size`
struct ALNSParams {
    int    max_iterations = 200;     // total destroy-repair iterations
    float  removal_min    = 0.1f;    // [removal_min, removal_max] × n_pairs to remove
    float  removal_max    = 0.4f;
    float  p_shaw         = 6.f;     // Shaw determinism (higher = more deterministic)
    float  p_worst        = 3.f;     // Worst-removal determinism
    float  reaction_factor = 0.1f;   // adaptive-weight reaction speed (paper r)
    int    segment_size   = 25;      // iterations between weight updates
    float  w_dist         = 1.f;     // Shaw relatedness: distance weight
    float  sa_w_start     = 0.05f;   // start temperature s.t. solution w% worse accepted w.p. 0.5
    float  sa_cooling     = 0.9985f; // SA cooling rate per iteration
    int    sigma1         = 33;      // score: new global best (paper σ1)
    int    sigma2         = 9;       // score: accepted improving (paper σ2)
    int    sigma3         = 13;      // score: accepted worse (paper σ3)
    float  noise_factor   = 0.025f;  // insertion-cost noise η × max_edge
};

// Plans from a single delivery node using the operable environment cost matrix.
// One LocalSolutionAgent exists per delivery node in the DeliveryAgent's task list.
//
// Implements DbVNS-PDP (Decomposition-based VNS for Pickup and Delivery):
//   - Sequence is built in reverse, anchored at the delivery node (= last visited).
//   - Adding delivery D_i to the stack queues its pickup P_i in pending,
//     so P_i is always placed before D_i in the final (reversed) sequence.
//   - VNS shake: backtrack partial solutions, forbid tail choices, explore alternatives.
//   - Objective: minimize total travel distance.
struct LocalSolutionAgent {
    ObjectiveNode starting_node;
    DbVNSParams   params;

    explicit LocalSolutionAgent(const ObjectiveNode& node,
                                const DbVNSParams&   p = {});

    // Run DbVNS-PDP and return the visit sequence (pickup-before-delivery guaranteed).
    // pickup_of  : delivery_id -> pickup_id
    // delivery_of: pickup_id  -> delivery_id (reserved for future use)
    // agent_current: agent's current node id (reserved for first-leg cost)
    std::vector<ObjectiveNode> plan(
        const OperableEnvironment& env,
        const PairingMap&          pickup_of,
        const PairingMap&          delivery_of,
        osmium::object_id_type     agent_current
    ) const;

    // Forward DbVNS-PDP for lifelong replanning (mono-agent GPDP).
    //
    // Plans a visit order for all nodes in `env`, starting from an external
    // position (NOT itself a node to visit). Constraints enforced:
    //   - Pairing : pickup_of[delivery_id] = pickup_id  →  pickup before delivery
    //   - Capacity: simultaneous carry ≤ max_capacity at every step
    //
    // initial_load = number of packages already in the agent's hands when it
    //   reaches the plan-start position (e.g. the inflight head). The forward
    //   sequence builds up on top of this baseline.
    //
    // start_costs[i] = travel cost from plan-start to env.nodes[i].
    // Use kCostScale (a large sentinel) for unreachable nodes.
    //
    // Returns the forward visit sequence with pickup-before-delivery + capacity
    // guaranteed.
    static std::vector<ObjectiveNode> plan_sequence(
        const OperableEnvironment& env,
        const PairingMap&          pickup_of,
        const std::vector<float>&  start_costs,
        int                        max_capacity = 3,
        int                        initial_load = 0,
        const DbVNSParams&         params = {}
    );

    // ALNS-PDP variant of plan_sequence (Ropke & Pisinger 2006 adapted for
    // mono-agent lifelong GPDP). Same I/O contract as plan_sequence above —
    // pairing & capacity enforced, sequence forward from an external start.
    static std::vector<ObjectiveNode> plan_sequence_alns(
        const OperableEnvironment& env,
        const PairingMap&          pickup_of,
        const std::vector<float>&  start_costs,
        int                        max_capacity = 3,
        int                        initial_load = 0,
        const ALNSParams&          params = {}
    );
};

#endif // LOCAL_SOLUTION_AGENT_HPP
