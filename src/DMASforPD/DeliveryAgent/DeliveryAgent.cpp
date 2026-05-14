#include "DeliveryAgent.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include "DMASforPD/TaskAgent/TaskAllocationModule.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

// ---- Construction -------------------------------------------------------

DeliveryAgent::DeliveryAgent(int id, osmium::object_id_type start_node)
    : agent_id(id), current_node(start_node) {
    solution = AgentSolution::make(&current_node);
}

// ---- Spatial state ------------------------------------------------------

void DeliveryAgent::start_edge(
    osmium::object_id_type edge_id,
    osmium::object_id_type to_node,
    int arrival_step
) {
    in_transit = EdgeTransit{ edge_id, current_node, to_node, arrival_step };
}

void DeliveryAgent::arrive_at_node() {
    if (!in_transit) return;
    current_node = in_transit->to_node;
    in_transit.reset();
    // current_node is pointed to by solution.current_position — automatically reflected.
}

// ---- GlobalMemory interface ---------------------------------------------

void DeliveryAgent::connect(PDPGlobalMemory& memory) {
    memory.register_delivery_agent(*this);
}

void DeliveryAgent::disconnect(PDPGlobalMemory& memory) {
    memory.unregister_delivery_agent(agent_id);
}

void DeliveryAgent::receive_task(PDPTask& task, PDPGlobalMemory& memory) {
    add_task_to_memory(task);

    // FIFO append for both Idle and Active.
    // plan() reorder is intentionally NOT called for active agents: it would
    // rebuild solution.sequence (clear + reinsert) while the in-flight EdgeCursor
    // and ScheduledArrival still target the original front. solution.advance()
    // at arrival would then pop the new (reordered) front, breaking the
    // invariant front().id == edge_cursor.target. FIFO append preserves it.
    solution.push_back(task.pickup);
    solution.push_back(task.delivery);

    status = AgentStatus::Active;
}

void DeliveryAgent::remove_completed_task(int task_id) {
    auto it = std::find_if(local_memory.tasks.begin(), local_memory.tasks.end(),
        [task_id](const PDPTask* t){ return t->task_id == task_id; });
    if (it == local_memory.tasks.end()) return;

    PDPTask* task = *it;
    local_memory.operable_env.remove_task(task->pickup.id, task->delivery.id);

    auto la_it = std::find_if(local_memory.local_agents.begin(), local_memory.local_agents.end(),
        [&](const LocalSolutionAgent& la){ return la.starting_node.id == task->delivery.id; });
    if (la_it != local_memory.local_agents.end())
        local_memory.local_agents.erase(la_it);

    local_memory.tasks.erase(it);
}

float DeliveryAgent::try_accept_task(const TaskOffer& offer, PDPGlobalMemory& memory) {
    PDPTask* task = memory.get_task(offer.task_id);
    if (!task) return 0.f;

    // ── Estimate insertion cost: current_node → pickup → delivery (upper bound) ──
    // Uses the path cache; falls back to kCostScale per missing leg.
    float insertion_cost = kCostScale * 2.f;
    {
        const auto* to_pu = memory.get_or_compute_path(
            current_node, task->pickup.id, task->pickup.group_id);
        const auto* pu_del = memory.get_or_compute_path(
            task->pickup.id, task->delivery.id, task->delivery.group_id);

        float c_pu  = (to_pu  && to_pu->valid())  ? to_pu->cost  : kCostScale;
        float c_del = (pu_del && pu_del->valid())  ? pu_del->cost : kCostScale;
        insertion_cost = c_pu + c_del;
    }

    // ── Current planned route cost ─────────────────────────────────────────────
    float route_cost = solution.total_planned_cost(memory);
    if (route_cost >= 1e10f || route_cost < 0.f) route_cost = 0.f;

    const float eps = 1e-6f;

    // ── Build PolicyFeatures ───────────────────────────────────────────────────
    PolicyFeatures f;
    f.cost_diff      = std::clamp(insertion_cost / kCostScale, 0.f, 1.f);
    f.profit_rate    = std::clamp(offer.reward / (insertion_cost * 0.5f + eps), 0.f, 1.f);
    f.current_load   = std::clamp(
        static_cast<float>(local_memory.tasks.size()) / kMaxLoad, 0.f, 1.f);
    f.queue_duration = std::clamp(route_cost / kQueueScale, 0.f, 1.f);
    f.efficiency_loss= std::clamp(insertion_cost / (route_cost + eps), 0.f, 1.f);
    f.rank_in_call   = 1.f - std::clamp(
        static_cast<float>(offer.prev_agents.size()) / kMaxAgents, 0.f, 1.f);
    f.task_importance= std::clamp(offer.importance / kImpMax, 0.f, 1.f);
    f.recall_round_norm = std::clamp(
        static_cast<float>(offer.recall_round)
        / static_cast<float>(std::max(offer.max_recalls, 1)),
        0.f, 1.f);

    int n_agents = static_cast<int>(memory.all_delivery_agents().size());
    int total    = memory.count_total();
    f.n_agents_ratio = std::clamp(static_cast<float>(n_agents) / kMaxAgents, 0.f, 1.f);
    f.n_alloc_ratio  = (total > 0)
        ? static_cast<float>(memory.count_allocated()) / total : 0.f;
    f.n_avail_ratio  = (total > 0)
        ? static_cast<float>(memory.count_available()) / total : 0.f;
    f.time_remaining = std::clamp(1.f - memory.cur_time_ratio, 0.f, 1.f);

    // ── Stochastic action sampling ─────────────────────────────────────────────
    // PPO requires the recorded action to be sampled from π(a|s). A hard
    // threshold (action = μ ≥ 0.5) was used previously, which produced
    // mathematically inconsistent log_probs and locked the policy at acc ≈ 1.0
    // because μ never had to cross 0.5 to generate action=0 experiences.
    // Sampling Bernoulli(μ) naturally generates both action=1 and action=0
    // records around any μ ∈ (0,1), allowing the gradient to learn the true
    // accept/refuse trade-off.
    static thread_local std::mt19937 rng{std::random_device{}()};
    float mu = ObjectiveDMPolicy::shared().score(f);
    std::bernoulli_distribution dist(std::clamp(mu, 0.001f, 0.999f));
    float action = dist(rng) ? 1.f : 0.f;

    // Reward is 0 at decision time; EpisodeRunner::on_objective_reached calls
    // update_reward() with the actual task value when the task is delivered.
    ObjectiveDMPolicy::shared().record(f, action, 0.f, agent_id);
    return action;  // 0 or 1 — TAM still uses >= 0.5 threshold which works.
}

float DeliveryAgent::compute_bid(const TaskOffer& offer, PDPGlobalMemory& memory) {
    PDPTask* task = memory.get_task(offer.task_id);
    if (!task) return 0.f;

    const auto* to_pu = memory.get_or_compute_path(
        current_node, task->pickup.id, task->pickup.group_id);
    const auto* pu_del = memory.get_or_compute_path(
        task->pickup.id, task->delivery.id, task->delivery.group_id);

    float c_pu  = (to_pu  && to_pu->valid())  ? to_pu->cost  : kCostScale;
    float c_del = (pu_del && pu_del->valid()) ? pu_del->cost : kCostScale;
    float insertion_cost = c_pu + c_del;

    constexpr float eps = 1e-6f;
    return (offer.reward * offer.importance) / std::max(insertion_cost, eps);
}

// ---- Path management (two-path lookahead) --------------------------------

bool DeliveryAgent::begin_leg(const ObjectivePath* path, int task_id, bool is_pickup) {
    local_memory.current_path = path;

    if (!path || !path->valid() || path->nodes.size() < 2) {
        // Fallback: no usable cached path — build a synthetic direct-hop cursor.
        // The simulator will compute a haversine-based travel time for this hop.
        osmium::object_id_type dest = solution.empty() ? 0 : solution.next_objective().id;
        edge_cursor = EdgeCursor{ {current_node, dest}, {}, 0, task_id, is_pickup };
        return false;
    }

    edge_cursor = EdgeCursor{
        path->nodes,   // all nodes including start and objective
        path->edges,   // way IDs between consecutive nodes
        0,             // agent is at nodes[0] = current_node
        task_id,
        is_pickup
    };
    return true;
}

void DeliveryAgent::promote_next_path() {
    local_memory.current_path = local_memory.next_path;
    local_memory.next_path    = nullptr;
}

void DeliveryAgent::prefetch_next_path(PDPGlobalMemory& memory) {
    // Pre-fetch the path from sequence[0]→sequence[1] (the leg after the current one).
    // sequence[0] is the objective we are currently heading to.
    if (solution.num_remaining() >= 2)
        local_memory.next_path = solution.path_between(0, memory);
    else
        local_memory.next_path = nullptr;
}

void DeliveryAgent::push_updated_path(const ObjectivePath* new_path) {
    // GlobalMemory calls this when congestion reroutes the current leg.
    // Rebuild the cursor from the agent's current position within the new path.
    local_memory.current_path = new_path;
    if (!new_path || !new_path->valid() || new_path->nodes.size() < 2) return;

    // Find the agent's current node in the new path to resume from there.
    const auto& ns = new_path->nodes;
    int resume = 0;
    for (int i = 0; i < static_cast<int>(ns.size()); ++i) {
        if (ns[i] == current_node) { resume = i; break; }
    }

    if (edge_cursor) {
        edge_cursor->nodes    = new_path->nodes;
        edge_cursor->edges    = new_path->edges;
        edge_cursor->next_idx = resume;
    }
}

// ---- Legacy path helpers (kept for TAM / planning compatibility) ---------

void DeliveryAgent::fetch_current_path(PDPGlobalMemory& memory) {
    local_memory.current_path =
        solution.empty() ? nullptr : solution.path_to_next(memory);
}

void DeliveryAgent::fetch_next_path(PDPGlobalMemory& memory) {
    local_memory.next_path =
        (solution.num_remaining() < 2) ? nullptr : solution.path_between(0, memory);
}

// ---- Planning -----------------------------------------------------------

void DeliveryAgent::plan(PDPGlobalMemory& memory, float speed_mps) {
    // 1. Refresh the cost matrix from GlobalMemory path cache.
    local_memory.operable_env.refresh_costs(memory);

    if (local_memory.local_agents.empty()) return;

    // 2. Build pickup/delivery pairings from the assigned task list.
    PairingMap pickup_of, delivery_of;
    for (const PDPTask* t : local_memory.tasks) {
        pickup_of [t->delivery.id] = t->pickup.id;
        delivery_of[t->pickup.id]  = t->delivery.id;
    }

    // 3. Run all LocalSolutionAgents; keep the lowest-cost sequence.
    std::vector<ObjectiveNode> best_seq;
    float                      best_cost = std::numeric_limits<float>::max();

    for (const LocalSolutionAgent& la : local_memory.local_agents) {
        std::vector<ObjectiveNode> candidate =
            la.plan(local_memory.operable_env, pickup_of, delivery_of, current_node);
        if (candidate.empty()) continue;

        float cost = 0.0f;
        for (std::size_t i = 0; i + 1 < candidate.size(); ++i) {
            int ai = local_memory.operable_env.find_index(candidate[i].id);
            int bi = local_memory.operable_env.find_index(candidate[i + 1].id);
            if (ai < 0 || bi < 0) { cost = std::numeric_limits<float>::max(); break; }
            float c = local_memory.operable_env.get_cost(ai, bi);
            if (c < 0.0f)         { cost = std::numeric_limits<float>::max(); break; }
            cost += c;
        }

        if (cost < best_cost) {
            best_cost = cost;
            best_seq  = std::move(candidate);
        }
    }

    if (best_seq.empty()) return;

    // 4. Rebuild solution sequence with the best candidate.
    solution.sequence.clear();
    for (const auto& node : best_seq)
        solution.push_back(node);

    // 5. Commit the plan to GlobalMemory (updates congestion + estimated arrivals).
    memory.commit_plan(agent_id, speed_mps);
}

// ---- Private ------------------------------------------------------------

void DeliveryAgent::add_task_to_memory(PDPTask& task) {
    local_memory.tasks.push_back(&task);
    local_memory.operable_env.add_task(task);
    local_memory.local_agents.emplace_back(task.delivery);
}
