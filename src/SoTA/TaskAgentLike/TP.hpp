#ifndef SOTA_TP_HPP
#define SOTA_TP_HPP

#include <memory>
#include <vector>

class DeliveryAgent;
class PDPGlobalMemory;

// Token Passing [Ma+2017] allocation baseline: replaces the TAM + policy. The
// task picks the closest FREE agent by h(loc, pickup) (static graph distance,
// one reverse Dijkstra from the pickup); falls back to all eligible agents when
// none is free. Returns the winner agent_id, or -1 if none.
int tp_allocate(PDPGlobalMemory& memory,
                const std::vector<std::unique_ptr<DeliveryAgent>>& agents,
                int n_active, int task_id);

#endif // SOTA_TP_HPP
