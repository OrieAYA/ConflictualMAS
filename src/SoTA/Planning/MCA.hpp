#ifndef SOTA_MCA_HPP
#define SOTA_MCA_HPP

class DeliveryAgent;
class PDPGlobalMemory;
struct PDPTask;

// Multi-Agent Cheapest-insertion + anytime LNS [Chen et al. 2021] planning
// baseline: improves the route built by cheapest insertion. In-place on
// a.solution; no-op unless memory.planning_use_mca_lns and route has >= 2 stops.
void mca_lns_improve(DeliveryAgent& a, PDPTask& task, PDPGlobalMemory& memory);

#endif // SOTA_MCA_HPP
