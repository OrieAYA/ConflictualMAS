#ifndef SOTA_ALNS_HPP
#define SOTA_ALNS_HPP

#include "DMASforPD/Algorithms/DbVNS.hpp"   // PairingMap, OperableEnvironment fwd, ObjectiveNode
#include <vector>

// ALNS-PDP [Ropke & Pisinger 2006], mono-agent lifelong GPDP variant.
//   destroy: Shaw / random / worst   repair: cheapest / regret-2
//   acceptance: simulated annealing   adaptive operator weights (roulette).
struct ALNSParams {
    int    max_iterations = 200;
    float  removal_min    = 0.1f;
    float  removal_max    = 0.4f;
    float  p_shaw         = 6.f;
    float  p_worst        = 3.f;
    float  reaction_factor = 0.1f;
    int    segment_size   = 25;
    float  w_dist         = 1.f;
    float  sa_w_start     = 0.05f;
    float  sa_cooling     = 0.9985f;
    int    sigma1         = 33;
    int    sigma2         = 9;
    int    sigma3         = 13;
    float  noise_factor   = 0.025f;
};

// Forward visit sequence from an external start, pairing + capacity enforced.
// Same I/O contract as DbVNS plan_sequence.
std::vector<ObjectiveNode> plan_sequence_alns(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const std::vector<float>&  start_costs,
    int                        max_capacity = 3,
    int                        initial_load = 0,
    const ALNSParams&          params = {});

#endif // SOTA_ALNS_HPP
