#include "TaskAllocationModule.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include "DMASforPD/DeliveryAgent/DeliveryAgent.hpp"
#include <algorithm>
#include <limits>

using DiscoveryStep = ObjectiveGroupCache::DiscoveryStep;

TaskAllocationModule::TaskAllocationModule(PDPTask& task, Params p)
    : task_(task), params_(p),
      pickup_group_id_(task.pickup.group_id),
      delivery_group_id_(task.delivery.group_id) {}

float TaskAllocationModule::ratio(float x) const {
    const float span = params_.ratio_max - params_.ratio_min;
    return params_.ratio_min + span / (1.f + x / std::max(1.f, params_.ratio_scale));
}

std::unordered_set<osmium::object_id_type>
TaskAllocationModule::idle_agent_positions(PDPGlobalMemory& memory) const {
    std::unordered_set<osmium::object_id_type> positions;
    for (const auto& [aid, agent] : memory.all_delivery_agents()) {
        if (!agent || !agent->is_at_node()) continue;
        const int cap = (agent->max_capacity > 0)
                        ? agent->max_capacity : params_.max_tasks_per_agent;
        if (static_cast<int>(agent->local_memory.tasks.size()) < cap)
            positions.insert(agent->current_node);
    }
    return positions;
}

bool TaskAllocationModule::plan_order_valid(int agent_id,
                                            PDPGlobalMemory& memory) const {
    const AgentSolution* sol = memory.get_solution(agent_id);
    if (!sol) return false;
    int pickup_pos = -1, delivery_pos = -1;
    for (int i = 0; i < static_cast<int>(sol->sequence.size()); ++i) {
        const auto nid = sol->sequence[i].node.id;
        if (nid == task_.pickup.id   && pickup_pos   < 0) pickup_pos   = i;
        if (nid == task_.delivery.id && delivery_pos < 0) delivery_pos = i;
    }
    if (pickup_pos < 0 && delivery_pos < 0) return true;      // not in plan yet
    if (pickup_pos >= 0 && delivery_pos >= 0) return pickup_pos < delivery_pos;
    return true;
}

void TaskAllocationModule::collect_candidates(PDPGlobalMemory& memory) {
    candidates_.clear();

    std::vector<std::pair<float, int>> scored;
    scored.reserve(matrix_.size());
    for (const auto& [aid, entry] : matrix_) {
        const bool has_p = entry.pickup_cost   >= 0.0f;
        const bool has_d = entry.delivery_cost >= 0.0f;
        if (!has_p && !has_d) continue;
        if (!plan_order_valid(aid, memory)) continue;
        const float pc = has_p ? entry.pickup_cost   : 0.0f;
        const float dc = has_d ? entry.delivery_cost : 0.0f;
        scored.push_back({ pc + dc, aid });
    }
    std::sort(scored.begin(), scored.end());
    for (const auto& [cost, aid] : scored) {
        (void)cost;
        if (static_cast<int>(candidates_.size()) >= params_.max_candidates) break;
        candidates_.push_back(aid);
    }
}

bool TaskAllocationModule::finalise(PDPGlobalMemory& memory, float speed_mps) {
    if (candidates_.empty()) {
        // No agent reachable at all — a topology/capacity failure, distinct
        // from a policy refusal.
        exhausted_ = true;
        return true;
    }

    bids_.assign(candidates_.size(), false);
    scores_.assign(candidates_.size(), 0.f);

    int   best_bidder = -1, best_any = -1;
    float best_bidder_mu = -1.f, best_any_mu = -1.f;

    for (std::size_t i = 0; i < candidates_.size(); ++i) {
        const int aid = candidates_[i];
        DeliveryAgent* agent = memory.get_delivery_agent(aid);
        if (!agent) continue;

        float mu  = 1.f;
        bool  bid = true;
        if (!params_.always_accept) {
            auto& entry = matrix_[aid];
            const float my_cost =
                std::max(0.f, entry.pickup_cost) + std::max(0.f, entry.delivery_cost);
            float cheapest_other = 0.f;
            bool  any_other      = false;
            for (int other : candidates_) {
                if (other == aid) continue;
                auto it = matrix_.find(other);
                if (it == matrix_.end()) continue;
                const float oc = std::max(0.f, it->second.pickup_cost)
                               + std::max(0.f, it->second.delivery_cost);
                if (!any_other || oc < cheapest_other) {
                    cheapest_other = oc;
                    any_other      = true;
                }
            }
            TaskOffer offer{ task_.task_id, task_.reward, task_.importance,
                             static_cast<int>(candidates_.size()),
                             my_cost, cheapest_other,
                             params_.max_candidates };
            const auto res = agent->bid_for_task(offer, memory);
            mu  = res.mu;
            bid = res.bid;
        }

        bids_[i]   = bid;
        scores_[i] = mu;
        if (mu > best_any_mu) { best_any_mu = mu; best_any = aid; }
        if (bid && mu > best_bidder_mu) { best_bidder_mu = mu; best_bidder = aid; }
    }

    if (best_bidder >= 0) {
        allocate_to(best_bidder, memory, speed_mps);
        return true;
    }
    if (params_.force_assign && best_any >= 0) {
        forced_ = true;
        allocate_to(best_any, memory, speed_mps);
        return true;
    }
    // Format B: everybody refused — defer.
    deferred_  = true;
    exhausted_ = true;
    return true;
}

void TaskAllocationModule::allocate_to(int agent_id, PDPGlobalMemory& memory,
                                       float speed_mps) {
    memory.assign_task(task_.task_id, agent_id);
    if (DeliveryAgent* agent = memory.get_delivery_agent(agent_id))
        agent->receive_task(task_, memory);
    memory.commit_plan(agent_id, speed_mps);
    winner_    = agent_id;
    allocated_ = true;
}

bool TaskAllocationModule::step(PDPGlobalMemory& memory, float speed_mps) {
    if (allocated_ || exhausted_) return true;

    if (max_search_cost_ < 0.0f)
        max_search_cost_ = std::numeric_limits<float>::max();

    if (idle_positions_dirty_) {
        idle_positions_cache_ = idle_agent_positions(memory);
        idle_positions_dirty_ = false;
    }

    // Reset the incremental Dijkstra frontiers once per allocation so stale
    // closed-sets from prior tasks don't hide agents that have moved since.
    if (!search_reset_done_) {
        memory.server_memory.reset_agent_search(task_.pickup.id,   pickup_group_id_);
        memory.server_memory.reset_agent_search(task_.delivery.id, delivery_group_id_);
        search_reset_done_ = true;
    }

    // Record-only side handler: writes side costs into matrix_.
    auto record_side = [&](const DiscoveryStep& d, bool is_pickup) {
        auto set_cost = [&](int aid, float cost) {
            if (is_pickup) matrix_[aid].pickup_cost   = cost;
            else           matrix_[aid].delivery_cost = cost;
        };
        if (d.type == DiscoveryStep::Type::Path && d.path) {
            const osmium::object_id_type tgt =
                is_pickup ? task_.pickup.id : task_.delivery.id;
            const osmium::object_id_type node =
                (d.path->node_b == tgt) ? d.path->node_a : d.path->node_b;
            // Objective node owned by an already-assigned task.
            PDPTask* related = memory.get_task_for_node(node);
            if (related && related->agent_id >= 0)
                set_cost(related->agent_id, d.path->cost);
            // Idle agents physically standing at this node.
            for (auto& [aid, agent] : memory.all_delivery_agents())
                if (agent->current_node == node && agent->is_at_node())
                    set_cost(aid, d.path->cost);
        } else if (d.type == DiscoveryStep::Type::AgentPosition) {
            for (auto& [aid, agent] : memory.all_delivery_agents())
                if (agent->current_node == d.agent_node && agent->is_at_node())
                    set_cost(aid, d.cost);
        }
    };

    // Expand both sides (static road distance — same units as the budget).
    DiscoveryStep ps = memory.server_memory.discover_step(
        task_.pickup.id, pickup_group_id_,
        idle_positions_cache_, max_search_cost_);
    record_side(ps, /*is_pickup=*/true);

    DiscoveryStep ds = memory.server_memory.discover_step(
        task_.delivery.id, delivery_group_id_,
        idle_positions_cache_, max_search_cost_);
    record_side(ds, /*is_pickup=*/false);

    collect_candidates(memory);

    // First candidate found → fix the adaptive expansion budget at x*ratio(x)
    // where x = max(pickup_cost, delivery_cost) of that candidate.
    if (!first_found_ && !candidates_.empty()) {
        first_found_ = true;
        const AgentEntry& e = matrix_[candidates_.front()];
        const float pc = (e.pickup_cost   >= 0.0f) ? e.pickup_cost   : 0.0f;
        const float dc = (e.delivery_cost >= 0.0f) ? e.delivery_cost : 0.0f;
        const float x  = std::max(pc, dc);
        max_search_cost_ = x * ratio(x);
    }

    const bool enough =
        static_cast<int>(candidates_.size()) >= params_.max_candidates;
    const bool both_done =
        (ps.type == DiscoveryStep::Type::Exhausted) &&
        (ds.type == DiscoveryStep::Type::Exhausted);

    if (enough || both_done)
        return finalise(memory, speed_mps);

    return false;
}
