#include "TaskAllocationModule.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include "DMASforPD/DeliveryAgent/DeliveryAgent.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <algorithm>
#include <cmath>

using DiscoveryStep = ObjectiveGroupCache::DiscoveryStep;

namespace {
// Great-circle distance in meters — local copy so TAM stays independent of Legacy.
float haversine_m(double lat1, double lon1, double lat2, double lon2) {
    constexpr double R    = 6371000.0;
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    const double phi1  = lat1 * kDeg, phi2 = lat2 * kDeg;
    const double dphi  = (lat2 - lat1) * kDeg, dlam = (lon2 - lon1) * kDeg;
    const double a = std::sin(dphi/2)*std::sin(dphi/2)
                   + std::cos(phi1)*std::cos(phi2)*std::sin(dlam/2)*std::sin(dlam/2);
    return static_cast<float>(R * 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a)));
}
}  // namespace

TaskAllocationModule::TaskAllocationModule(PDPTask& task, Params p)
    : task_(task), params_(p),
      pickup_group_id_  (task.pickup.group_id),
      delivery_group_id_(task.delivery.group_id) {}

// ---- Public interface --------------------------------------------------

bool TaskAllocationModule::step(PDPGlobalMemory& memory, float speed_mps) {
    if (allocated_ || exhausted_) return allocated_;

    // Lazy init of the spatial search budget on the first step.
    // task_dist = haversine(pickup, delivery); budget = task_dist * mult.
    if (max_search_cost_ < 0.0f) {
        const auto& nodes = memory.geo_box.data.nodes;
        auto pit = nodes.find(task_.pickup.id);
        auto dit = nodes.find(task_.delivery.id);
        if (pit != nodes.end() && dit != nodes.end()) {
            const float task_dist = haversine_m(
                pit->second.lat, pit->second.lon,
                dit->second.lat, dit->second.lon);
            max_search_cost_ = task_dist * params_.search_radius_mult;
        } else {
            max_search_cost_ = std::numeric_limits<float>::max();
        }
    }

    // Refresh idle-positions cache only when invalidated (initial step + after
    // each recall). Fleet state does not change between TAM step()s inside one
    // allocation, so the previous code repeated this O(n_agents) scan up to 300×
    // per task offer.
    if (idle_positions_dirty_) {
        idle_positions_cache_ = idle_agent_positions(memory);
        idle_positions_dirty_ = false;
    }

    // 1. Expand pickup side (no agent-position check — per TAM spec).
    DiscoveryStep ps = memory.server_memory.discover_step(
        task_.pickup.id, pickup_group_id_, {}, max_search_cost_);
    if (ps.type == DiscoveryStep::Type::Path)
        handle_pickup_node(ps.path->node_b == task_.pickup.id
                           ? ps.path->node_a : ps.path->node_b,
                           ps.path->cost, memory);

    // 2. Expand delivery side (with idle-agent detection).
    DiscoveryStep ds = memory.server_memory.discover_step(
        task_.delivery.id, delivery_group_id_, idle_positions_cache_, max_search_cost_);

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
    // Returns the positions of all agents currently at a node, regardless of
    // their queue length. The TAM's job is to call agents by planning-route
    // proximity; whether an agent can physically host one more package is a
    // capacity constraint enforced by DeliveryAgent::receive_task during
    // insertion planning, not a filter at offer time.
    std::unordered_set<osmium::object_id_type> positions;
    for (const auto& [aid, agent] : memory.all_delivery_agents())
        if (agent->is_at_node())
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
    // Direct allocation: any agent physically at this node is a candidate.
    // Queue length is no longer a filter — physical carrying capacity is
    // checked at planning time inside receive_task (capacity-aware insertion).
    for (auto& [aid, agent] : memory.all_delivery_agents()) {
        if (agent->current_node == node && agent->is_at_node()) {
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
    // No queue-length cap: an agent can always be offered a task. The
    // physical carrying capacity (max_tasks_per_agent = max simultaneous
    // packages in transit) is enforced inside DeliveryAgent::receive_task
    // via capacity-aware insertion. If no valid insertion exists, the
    // method falls back to a FIFO append (carry = 1, always valid).

    auto& entry = matrix_[agent_id];
    entry.call_count++;

    float score;
    if (params_.always_accept) {
        // Ablation mode: bypass MAPPO policy, accept unconditionally.
        // No buffer write — this is a non-learning baseline.
        score = 1.0f;
    } else {
        std::vector<int> prev;
        prev.reserve(matrix_.size());
        for (const auto& [aid, _] : matrix_) if (aid != agent_id) prev.push_back(aid);

        TaskOffer offer{task_.task_id, task_.reward, task_.importance,
                        std::move(prev), recall_count_, params_.max_recalls};
        score = agent->try_accept_task(offer, memory);
    }
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

    // Grow the search radius so the next step()s can discover agents that were
    // outside the previous budget. discover_step pushes back the frontier node
    // when it hits the budget, so the search resumes seamlessly with the
    // larger limit.
    if (max_search_cost_ < std::numeric_limits<float>::max())
        max_search_cost_ *= params_.recall_radius_grow;

    // Force a refresh: agents that finished a delivery during the previous
    // expansions might now be eligible.
    idle_positions_dirty_ = true;

    for (int aid : recall_queue_) {
        if (offer_to_agent(aid, memory, speed_mps)) return true;
    }
    return false;
}
