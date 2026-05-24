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

    // Multi-candidate mode branches off entirely — the legacy first-fit path
    // below is left byte-for-byte unchanged so that multi_candidate=false
    // restores the exact previous behaviour.
    if (params_.multi_candidate)
        return step_multi_candidate(memory, speed_mps);

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

// ════════════════════════════════════════════════════════════════════════════
// Multi-candidate mode (params_.multi_candidate == true)
// ════════════════════════════════════════════════════════════════════════════

float TaskAllocationModule::mc_ratio(float x) const {
    const float scale = std::max(1.0f, params_.mc_ratio_scale);
    const float span  = params_.mc_ratio_max - params_.mc_ratio_min;
    // x = 0 → mc_ratio_max ; x → ∞ → mc_ratio_min (monotone decreasing).
    return params_.mc_ratio_min + span / (1.0f + x / scale);
}

void TaskAllocationModule::mc_collect_candidates(PDPGlobalMemory& memory) {
    mc_candidates_.clear();

    // (combined_cost, agent_id) pairs, sorted ascending so the closest
    // candidates are kept when capping at mc_max_candidates.
    std::vector<std::pair<float, int>> scored;
    scored.reserve(matrix_.size());

    for (const auto& [aid, entry] : matrix_) {
        const bool has_p = entry.pickup_cost   >= 0.0f;
        const bool has_d = entry.delivery_cost >= 0.0f;

        // Include any agent reachable from at least one Dijkstra side.
        // Agents in transit after pickup (delivery node still in objectives,
        // pickup node removed) only have delivery_cost set — excluding them
        // caused ~24 % allocation failures even on a fully connected graph.
        if (!has_p && !has_d) continue;

        // Direction constraint: pickup before delivery in the agent's plan
        // (plan_order_valid already returns true for an empty plan).
        if (!plan_order_valid(aid, memory)) continue;

        const float pc = has_p ? entry.pickup_cost   : 0.0f;
        const float dc = has_d ? entry.delivery_cost : 0.0f;
        scored.push_back({ pc + dc, aid });
    }

    std::sort(scored.begin(), scored.end());
    for (const auto& [cost, aid] : scored) {
        (void)cost;
        if (static_cast<int>(mc_candidates_.size()) >= params_.mc_max_candidates)
            break;
        mc_candidates_.push_back(aid);
    }
}

float TaskAllocationModule::mc_score_agent(int agent_id, PDPGlobalMemory& memory) {
    DeliveryAgent* agent = memory.get_delivery_agent(agent_id);
    if (!agent) return 0.0f;

    auto& entry = matrix_[agent_id];
    entry.call_count++;

    float score;
    if (params_.always_accept) {
        // TamAlwaysAccept under multi-candidate: every candidate scores 1.0,
        // so the argmax degenerates to "nearest candidate" — a proper
        // always-accept + nearest-assignment baseline.
        score = 1.0f;
    } else {
        std::vector<int> prev;
        prev.reserve(matrix_.size());
        for (const auto& [aid, _] : matrix_)
            if (aid != agent_id) prev.push_back(aid);

        TaskOffer offer{task_.task_id, task_.reward, task_.importance,
                        std::move(prev), recall_count_, params_.max_recalls};
        // Same policy call path as the legacy offer_to_agent — no change to
        // how policies are invoked.
        score = agent->try_accept_task(offer, memory);
    }
    entry.score = score;
    return score;
}

void TaskAllocationModule::mc_allocate(
    int agent_id, PDPGlobalMemory& memory, float speed_mps
) {
    memory.assign_task(task_.task_id, agent_id);
    if (DeliveryAgent* agent = memory.get_delivery_agent(agent_id))
        agent->receive_task(task_, memory);
    memory.commit_plan(agent_id, speed_mps);
    allocated_ = true;
}

bool TaskAllocationModule::mc_finalise(PDPGlobalMemory& memory, float speed_mps) {
    if (mc_candidates_.empty()) {
        // No agent reachable at all — a genuine (capacity / topology) failure,
        // distinct from a policy refusal. Not deferred.
        exhausted_ = true;
        return true;
    }

    int   best_aid   = -1;
    float best_score = -1.0f;
    for (int aid : mc_candidates_) {
        const float s = mc_score_agent(aid, memory);
        if (s > best_score) { best_score = s; best_aid = aid; }
    }

    if (best_aid < 0) {           // defensive — should not happen
        exhausted_ = true;
        return true;
    }

    if (params_.mc_force_assign || best_score >= 0.5f) {
        // Format A (always) or Format B with an acceptable best score.
        mc_allocate(best_aid, memory, speed_mps);
        return true;
    }

    // Format B: every candidate scored below 0.5 → defer for a later retry.
    mc_deferred_ = true;
    exhausted_   = true;          // ends this TAM session; runner re-offers later
    return true;
}

bool TaskAllocationModule::step_multi_candidate(
    PDPGlobalMemory& memory, float speed_mps
) {
    // Phase-1: no budget — search expands freely until the first candidate is
    // found. The adaptive budget x*ratio(x) is applied below via mc_first_found_.
    // This guarantees at least one agent is always found (on a connected graph).
    // params_.search_radius_mult is only used by the legacy step() path.
    if (max_search_cost_ < 0.0f)
        max_search_cost_ = std::numeric_limits<float>::max();

    if (idle_positions_dirty_) {
        idle_positions_cache_ = idle_agent_positions(memory);
        idle_positions_dirty_ = false;
    }

    // Reset incremental Dijkstra states once per allocation so stale closed-sets
    // from prior tasks don't hide agents that have since moved to those nodes.
    // Memoised paths_ (A* results) are not affected — only the expansion frontier.
    if (!mc_search_reset_done_) {
        memory.server_memory.reset_agent_search(task_.pickup.id,   pickup_group_id_);
        memory.server_memory.reset_agent_search(task_.delivery.id, delivery_group_id_);
        mc_search_reset_done_ = true;
    }

    // Record-only side handler: writes pickup/delivery costs into matrix_
    // WITHOUT offering or allocating (offering happens later in mc_finalise).
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

    // Expand both sides. Idle positions are passed to BOTH searches so idle
    // agents are discovered on either side (legacy only checked delivery).
    DiscoveryStep ps = memory.server_memory.discover_step(
        task_.pickup.id, pickup_group_id_, idle_positions_cache_, max_search_cost_);
    record_side(ps, /*is_pickup=*/true);

    DiscoveryStep ds = memory.server_memory.discover_step(
        task_.delivery.id, delivery_group_id_, idle_positions_cache_, max_search_cost_);
    record_side(ds, /*is_pickup=*/false);

    // Rebuild the candidate set from the current matrix_.
    mc_collect_candidates(memory);

    // First valid candidate → fix the adaptive expansion budget.
    //   x = max(pickup_cost, delivery_cost) of that candidate.
    if (!mc_first_found_ && !mc_candidates_.empty()) {
        mc_first_found_ = true;
        const AgentEntry& e = matrix_[mc_candidates_.front()];
        const float pc = (e.pickup_cost   >= 0.0f) ? e.pickup_cost   : 0.0f;
        const float dc = (e.delivery_cost >= 0.0f) ? e.delivery_cost : 0.0f;
        const float x  = std::max(pc, dc);
        // budget = x * ratio(x) — guaranteed > x since ratio >= mc_ratio_min
        // (>1), so both searches still have room to find more candidates.
        max_search_cost_ = x * mc_ratio(x);
    }

    // Termination: enough candidates collected, or both searches exhausted.
    const bool enough =
        static_cast<int>(mc_candidates_.size()) >= params_.mc_max_candidates;
    const bool both_exhausted =
        (ps.type == DiscoveryStep::Type::Exhausted) &&
        (ds.type == DiscoveryStep::Type::Exhausted);

    if (enough || both_exhausted)
        return mc_finalise(memory, speed_mps);

    return false;   // keep expanding on the next step()
}
