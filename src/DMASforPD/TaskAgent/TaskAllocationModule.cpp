#include "TaskAllocationModule.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include "DMASforPD/DeliveryAgent/DeliveryAgent.hpp"
#include <algorithm>

using DiscoveryStep = ObjectiveGroupCache::DiscoveryStep;

TaskAllocationModule::TaskAllocationModule(PDPTask& task, Params p)
    : task_(task), params_(p),
      pickup_group_id_  (task.pickup.group_id),
      delivery_group_id_(task.delivery.group_id) {}

// ---- Public interface --------------------------------------------------

bool TaskAllocationModule::step(PDPGlobalMemory& memory, float speed_mps) {
    if (allocated_ || exhausted_) return allocated_;

    const auto idle_positions = idle_agent_positions(memory);

    // 1. Expand pickup side (no agent-position check — per TAM spec).
    DiscoveryStep ps = memory.server_memory.discover_step(
        task_.pickup.id, pickup_group_id_);
    if (ps.type == DiscoveryStep::Type::Path)
        handle_pickup_node(ps.path->node_b == task_.pickup.id
                           ? ps.path->node_a : ps.path->node_b,
                           ps.path->cost, memory);

    // 2. Expand delivery side (with idle-agent detection).
    DiscoveryStep ds = memory.server_memory.discover_step(
        task_.delivery.id, delivery_group_id_, idle_positions);

    if (ds.type == DiscoveryStep::Type::AgentPosition) {
        if (handle_delivery_node(ds.agent_node, ds.cost, memory, speed_mps))
            return true;
    } else if (ds.type == DiscoveryStep::Type::Path) {
        osmium::object_id_type found =
            (ds.path->node_b == task_.delivery.id) ? ds.path->node_a : ds.path->node_b;
        handle_delivery_node(found, ds.path->cost, memory, speed_mps);
    }

    // 3. Try agents reachable from both sides.
    if (try_common_agents(memory, speed_mps)) return true;

    // 4. If both searches exhausted → recall or declare failure.
    bool p_ex = (ps.type == DiscoveryStep::Type::Exhausted);
    bool d_ex = (ds.type == DiscoveryStep::Type::Exhausted);
    if (p_ex && d_ex) {
        if (recall_count_ < params_.max_recalls) {
            if (do_recall(memory, speed_mps)) return true;
        } else {
            exhausted_ = true;
        }
    }

    return false;
}

// ---- Helpers -----------------------------------------------------------

std::unordered_set<osmium::object_id_type>
TaskAllocationModule::idle_agent_positions(PDPGlobalMemory& memory) const {
    std::unordered_set<osmium::object_id_type> positions;
    for (const auto& [aid, agent] : memory.all_delivery_agents())
        if (agent->status == AgentStatus::Idle && agent->is_at_node())
            positions.insert(agent->current_node);
    return positions;
}

void TaskAllocationModule::handle_pickup_node(
    osmium::object_id_type node, float cost, PDPGlobalMemory& memory
) {
    // If this objective node belongs to an already-assigned task → record that agent.
    PDPTask* related = memory.get_task_for_node(node);
    if (related && related->agent_id >= 0)
        matrix_[related->agent_id].pickup_cost = cost;
}

bool TaskAllocationModule::handle_delivery_node(
    osmium::object_id_type node, float cost,
    PDPGlobalMemory& memory, float speed_mps
) {
    // Direct allocation: idle agent physically at this node (delivery-side discovery).
    for (auto& [aid, agent] : memory.all_delivery_agents()) {
        if (agent->current_node == node
            && agent->status == AgentStatus::Idle
            && agent->is_at_node())
        {
            matrix_[aid].delivery_cost = cost;
            if (offer_to_agent(aid, memory, speed_mps)) return true;
        }
    }

    // Objective node belonging to an assigned agent.
    PDPTask* related = memory.get_task_for_node(node);
    if (related && related->agent_id >= 0)
        matrix_[related->agent_id].delivery_cost = cost;

    return false;
}

bool TaskAllocationModule::try_common_agents(PDPGlobalMemory& memory, float speed_mps) {
    // Collect agents reachable from both sides, sorted by combined cost (ascending).
    std::vector<std::pair<float, int>> candidates;
    for (const auto& [aid, entry] : matrix_) {
        if (entry.pickup_cost  < 0.0f) continue;
        if (entry.delivery_cost < 0.0f) continue;
        candidates.push_back({entry.pickup_cost + entry.delivery_cost, aid});
    }
    std::sort(candidates.begin(), candidates.end());

    for (const auto& [combined_cost, aid] : candidates) {
        (void)combined_cost;
        if (!plan_order_valid(aid, memory)) continue;
        if (offer_to_agent(aid, memory, speed_mps)) return true;
    }
    return false;
}

bool TaskAllocationModule::plan_order_valid(int agent_id, PDPGlobalMemory& memory) const {
    const AgentSolution* sol = memory.get_solution(agent_id);
    if (!sol || sol->empty()) return true;  // no conflict

    int pickup_idx   = -1;
    int delivery_idx = -1;
    for (int i = 0; i < static_cast<int>(sol->sequence.size()); ++i) {
        osmium::object_id_type nid = sol->sequence[i].node.id;
        if (nid == task_.pickup.id)   pickup_idx   = i;
        if (nid == task_.delivery.id) delivery_idx = i;
    }
    // Neither in plan → OK to add. Both in plan → pickup must come first.
    if (pickup_idx < 0 && delivery_idx < 0) return true;
    if (pickup_idx >= 0 && delivery_idx >= 0) return pickup_idx < delivery_idx;
    return false;  // only one is already planned — inconsistent
}

bool TaskAllocationModule::offer_to_agent(
    int agent_id, PDPGlobalMemory& memory, float speed_mps
) {
    DeliveryAgent* agent = memory.get_delivery_agent(agent_id);
    if (!agent) return false;

    auto& entry = matrix_[agent_id];
    entry.call_count++;

    std::vector<int> prev;
    prev.reserve(matrix_.size());
    for (const auto& [aid, _] : matrix_) if (aid != agent_id) prev.push_back(aid);

    TaskOffer offer{task_.task_id, task_.reward, task_.importance, std::move(prev)};
    float score = agent->try_accept_task(offer, memory);
    entry.score = score;

    if (score >= 0.5f) {
        // Accepted — finalize allocation.
        memory.assign_task(task_.task_id, agent_id);
        agent->receive_task(task_, memory);
        memory.commit_plan(agent_id, speed_mps);
        allocated_ = true;
        return true;
    }
    return false;
}

void TaskAllocationModule::build_recall_queue() {
    recall_queue_.clear();
    for (const auto& [aid, entry] : matrix_)
        if (entry.score > 0.0f || entry.call_count > 0)
            recall_queue_.push_back(aid);
    std::sort(recall_queue_.begin(), recall_queue_.end(),
        [&](int a, int b){ return matrix_.at(a).score > matrix_.at(b).score; });
}

bool TaskAllocationModule::do_recall(PDPGlobalMemory& memory, float speed_mps) {
    build_recall_queue();
    task_.reward     *= params_.recall_reward_mult;
    task_.importance *= params_.recall_import_mult;
    ++recall_count_;

    for (int aid : recall_queue_) {
        if (offer_to_agent(aid, memory, speed_mps)) return true;
    }
    return false;
}
