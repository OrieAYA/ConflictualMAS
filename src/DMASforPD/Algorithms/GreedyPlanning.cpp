#include "DMASforPD/Algorithms/GreedyPlanning.hpp"
#include "DMASforPD/Structures/OperableEnvironment.hpp"
#include "DMASforPD/Policy/PolicyKit.hpp"
#include <limits>
#include <unordered_map>

std::vector<ObjectiveNode> plan_sequence_greedy(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const std::vector<float>&  start_costs,
    int                        max_capacity,
    int                        initial_load)
{
    const int n = env.size();
    std::vector<ObjectiveNode> result;
    if (n <= 0) return result;

    std::unordered_map<osmium::object_id_type, osmium::object_id_type> unlock_on_visit;
    for (const auto& [delivery_id, pickup_id] : pickup_of)
        unlock_on_visit[pickup_id] = delivery_id;

    std::vector<osmium::object_id_type> node_id(n);
    std::vector<bool> is_pickup(n, false);
    std::vector<bool> locked(n, false);
    std::vector<bool> visited(n, false);

    for (int i = 0; i < n; ++i) {
        node_id[i]   = env.nodes[i].id;
        is_pickup[i] = unlock_on_visit.count(node_id[i]) > 0;
        if (!is_pickup[i] && pickup_of.count(node_id[i]) > 0)
            locked[i] = true;
    }

    int remaining = n;
    int load      = initial_load;
    int last_idx  = -1;

    while (remaining > 0) {
        int   best_idx  = -1;
        float best_cost = std::numeric_limits<float>::max();

        for (int i = 0; i < n; ++i) {
            if (visited[i] || locked[i]) continue;
            if (is_pickup[i] && load >= max_capacity) continue;

            float c = (last_idx < 0) ? start_costs[i] : env.ensure_cost(last_idx, i);
            if (c < 0.f) c = kCostScale;

            if (c < best_cost) { best_cost = c; best_idx = i; }
        }

        if (best_idx < 0) break;

        visited[best_idx] = true;
        result.push_back(env.nodes[best_idx]);
        --remaining;
        last_idx = best_idx;

        if (is_pickup[best_idx]) {
            ++load;
            const int di = env.find_index(unlock_on_visit[node_id[best_idx]]);
            if (di >= 0) locked[di] = false;
        } else {
            --load;
        }
    }

    return result;
}