#include "SoTA/ScoringLike/RMCA.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Policy/PolicyKit.hpp"   // kCostScale
#include <algorithm>
#include <limits>
#include <vector>

// Marginal insertion cost (eq.13) over admissible (pickup_pos, delivery_pos)
// pairs of the agent's current sequence, capacity-checked (eq.9). Same cost
// model as the cheapest-insertion planner, so the chosen agent matches the
// regret's k1.
static float rmca_marginal_cost(const DeliveryAgent& a, const PDPTask& task,
                                PDPGlobalMemory& memory) {
    const auto& seq = a.solution.sequence;
    const int   n   = static_cast<int>(seq.size());
    const int   tam_cap   = memory.task_agent.params.max_tasks_per_agent;
    const int   max_carry = std::max(1, a.max_capacity > 0 ? a.max_capacity : tam_cap);

    auto seg = [&](osmium::object_id_type from, osmium::object_id_type to) -> float {
        if (from == 0 || to == 0 || from == to) return 0.f;
        const auto* p = memory.get_or_compute_path(from, to, 1);
        return (p && p->valid()) ? p->cost : kCostScale;
    };

    if (n == 0)
        return seg(a.current_node, task.pickup.id)
             + seg(task.pickup.id, task.delivery.id);

    int initial_load = 0;
    for (const PDPTask* t : a.local_memory.tasks) {
        if (t->task_id == task.task_id) continue;
        if (t->timeline.picked_step >= 0 && t->timeline.delivered_step < 0)
            ++initial_load;
    }
    std::vector<int> load_after(n, 0);
    {
        int cur = initial_load;
        for (int i = 0; i < n; ++i) {
            const PDPTask* t = memory.get_task_for_node(seq[i].node.id);
            if (t) {
                if      (t->pickup.id   == seq[i].node.id) ++cur;
                else if (t->delivery.id == seq[i].node.id) --cur;
            }
            load_after[i] = cur;
        }
    }

    float best = std::numeric_limits<float>::max();
    for (int pP = 1; pP <= n; ++pP) {
        for (int pD = pP; pD <= n; ++pD) {
            int peak = 0;
            for (int k = pP - 1; k <= pD - 1; ++k)
                if (load_after[k] > peak) peak = load_after[k];
            if (peak + 1 > max_carry) continue;

            float delta;
            if (pP == pD) {
                osmium::object_id_type before = seq[pP - 1].node.id;
                osmium::object_id_type after  = (pP == n) ? 0 : seq[pP].node.id;
                delta = seg(before, task.pickup.id)
                      + seg(task.pickup.id, task.delivery.id)
                      + seg(task.delivery.id, after)
                      - seg(before, after);
            } else {
                osmium::object_id_type before_P = seq[pP - 1].node.id;
                osmium::object_id_type after_P  = seq[pP].node.id;
                float p_delta = seg(before_P, task.pickup.id)
                              + seg(task.pickup.id, after_P) - seg(before_P, after_P);
                osmium::object_id_type before_D = seq[pD - 1].node.id;
                osmium::object_id_type after_D  = (pD == n) ? 0 : seq[pD].node.id;
                float d_delta = seg(before_D, task.delivery.id)
                              + seg(task.delivery.id, after_D) - seg(before_D, after_D);
                delta = p_delta + d_delta;
            }
            if (delta < best) best = delta;
        }
    }
    return best;
}

float rmca_score(const DeliveryAgent& a, const PDPTask& task,
                 PDPGlobalMemory& memory) {
    const float mc = rmca_marginal_cost(a, task, memory);
    return 1.0f / (1.0f + std::max(0.f, mc) / kCostScale);
}
