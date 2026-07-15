#include "SoTA/TaskAgentLike/TP.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Policy/PolicyKit.hpp"   // kCostScale
#include "Environment/GeoBox/GraphSearch.hpp"
#include <limits>

int tp_allocate(PDPGlobalMemory& memory,
                const std::vector<std::unique_ptr<DeliveryAgent>>& agents,
                int n_active, int task_id) {
    PDPTask* t = memory.get_task(task_id);
    if (!t) return -1;

    bool any_free = false;
    for (int i = 0; i < n_active; ++i)
        if (agents[i]->local_memory.tasks.empty()) { any_free = true; break; }

    // One reverse Dijkstra from pickup gives h(loc, pickup) for every agent.
    auto dist_from_pickup =
        graph_search::dijkstra_distances(memory.geo_box, t->pickup.id);

    int   best_aid = -1;
    float best_h   = std::numeric_limits<float>::max();
    for (int i = 0; i < n_active; ++i) {
        DeliveryAgent& a = *agents[i];
        if (any_free && !a.local_memory.tasks.empty()) continue;   // free agents first
        float h = kCostScale;
        auto it = dist_from_pickup.find(a.current_node);
        if (it != dist_from_pickup.end()) h = it->second;
        if (h < best_h) { best_h = h; best_aid = a.agent_id; }
    }
    return best_aid;
}
