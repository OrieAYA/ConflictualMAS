#include "GlobalMemory.hpp"
#include "DMASforPD/Utility/AgentSolution.hpp"
#include "DMASforPD/DeliveryAgent/DeliveryAgent.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>

// ---- Construction -------------------------------------------------------

PDPGlobalMemory::PDPGlobalMemory(GeoBox& box, Pathfinder& pf,
                                 const CongestionParams& cparams,
                                 const TaskAgentParams& taparams)
    : geo_box(box), pathfinder(pf), server_memory(box, pf),
      congestion_map(cparams), task_agent(0, taparams) {
    server_memory.initialize_from_geobox();
}

// ---- Task management ---------------------------------------------------

int PDPGlobalMemory::add_task(const ObjectiveNode& pickup, const ObjectiveNode& delivery) {
    server_memory.add_objective_node(pickup.id,   pickup.group_id);
    server_memory.add_objective_node(delivery.id, delivery.group_id);

    int id = static_cast<int>(tasks_.size());
    PDPTask task;
    task.task_id  = id;
    task.pickup   = pickup;
    task.delivery = delivery;
    tasks_.emplace(id, task);
    PDPTask* ptr = &tasks_.at(id);
    ptr->timeline.created_step = current_time_;
    available_tasks.push_back(ptr);
    node_to_task_id_[pickup.id]   = id;
    node_to_task_id_[delivery.id] = id;
    return id;
}

void PDPGlobalMemory::remove_from_lists(PDPTask* task,
                                         std::vector<PDPTask*>& available,
                                         std::vector<PDPTask*>& allocated,
                                         std::vector<PDPTask*>& finished) {
    auto erase = [task](std::vector<PDPTask*>& v) {
        v.erase(std::remove(v.begin(), v.end(), task), v.end());
    };
    erase(available);
    erase(allocated);
    erase(finished);
}

void PDPGlobalMemory::assign_task(int task_id, int agent_id) {
    PDPTask* task = get_task(task_id);
    if (!task) throw std::out_of_range("assign_task: invalid task_id");
    remove_from_lists(task, available_tasks, allocated_tasks, finished_tasks);
    task->assign(agent_id, current_time_);
    allocated_tasks.push_back(task);
}

void PDPGlobalMemory::unassign_task(int task_id) {
    PDPTask* task = get_task(task_id);
    if (!task) throw std::out_of_range("unassign_task: invalid task_id");
    remove_from_lists(task, available_tasks, allocated_tasks, finished_tasks);
    task->release();
    available_tasks.push_back(task);
}

void PDPGlobalMemory::complete_task(int task_id) {
    PDPTask* task = get_task(task_id);
    if (!task) throw std::out_of_range("complete_task: invalid task_id");
    remove_from_lists(task, available_tasks, allocated_tasks, finished_tasks);
    task->complete(current_time_);
    finished_tasks.push_back(task);
}

// ---- Task queries -------------------------------------------------------

PDPTask* PDPGlobalMemory::get_task(int task_id) {
    auto it = tasks_.find(task_id);
    return it != tasks_.end() ? &it->second : nullptr;
}

const PDPTask* PDPGlobalMemory::get_task(int task_id) const {
    auto it = tasks_.find(task_id);
    return it != tasks_.end() ? &it->second : nullptr;
}

PDPTask* PDPGlobalMemory::get_task_for_node(osmium::object_id_type node_id) {
    auto it = node_to_task_id_.find(node_id);
    if (it == node_to_task_id_.end()) return nullptr;
    return get_task(it->second);
}

std::vector<const PDPTask*> PDPGlobalMemory::tasks_for_agent(int agent_id) const {
    std::vector<const PDPTask*> result;
    for (const auto* t : allocated_tasks)
        if (t->agent_id == agent_id) result.push_back(t);
    return result;
}

// ---- Delivery agent registry -------------------------------------------

void PDPGlobalMemory::register_delivery_agent(DeliveryAgent& agent) {
    delivery_agents_[agent.agent_id] = &agent;
}

void PDPGlobalMemory::unregister_delivery_agent(int agent_id) {
    unregister_committed_plan(agent_id);
    delivery_agents_.erase(agent_id);
}

DeliveryAgent* PDPGlobalMemory::get_delivery_agent(int agent_id) {
    auto it = delivery_agents_.find(agent_id);
    return it != delivery_agents_.end() ? it->second : nullptr;
}

DeliveryAgent* PDPGlobalMemory::get_agent_for_task(int task_id) {
    const PDPTask* task = get_task(task_id);
    if (!task || task->agent_id < 0) return nullptr;
    return get_delivery_agent(task->agent_id);
}

const std::unordered_map<int, DeliveryAgent*>& PDPGlobalMemory::all_delivery_agents() const {
    return delivery_agents_;
}

const AgentSolution* PDPGlobalMemory::get_solution(int agent_id) const {
    auto it = delivery_agents_.find(agent_id);
    return it != delivery_agents_.end() ? &it->second->solution : nullptr;
}

// ---- Plan commit (environment sync) ------------------------------------

namespace {

// Build a TimedPath from a cached ObjectivePath at a given speed.
// edge_steps[i] = ceil(dist_i / speed_mps), minimum 1.
TimedPath make_timed_path(const ObjectivePath& path, float speed_mps, const MyData& data) {
    TimedPath tp;
    tp.from_node      = path.nodes.empty() ? path.node_a : path.nodes.front();
    tp.to_node        = path.nodes.empty() ? path.node_b : path.nodes.back();
    tp.nodes          = path.nodes;
    tp.total_distance = path.cost;

    for (const auto& way_id : path.edges) {
        auto it = data.ways.find(way_id);
        float dist  = (it != data.ways.end()) ? it->second.distance_meters : 0.0f;
        int   steps = (speed_mps > 0.0f)
                      ? std::max(1, static_cast<int>(std::ceil(dist / speed_mps)))
                      : 1;
        tp.edge_ids.push_back(way_id);
        tp.edge_steps.push_back(steps);
        tp.total_steps += steps;
    }
    tp.seal();
    return tp;
}

}  // namespace

void PDPGlobalMemory::unregister_committed_plan(int agent_id) {
    auto it = committed_plans_.find(agent_id);
    if (it == committed_plans_.end()) return;
    for (auto& [tp, departure] : it->second) {
        for (std::size_t i = 0; i < tp.edge_ids.size(); ++i)
            congestion_map.remove_agent(tp.edge_ids[i],
                                        tp.abs_entry(i, departure),
                                        tp.abs_exit (i, departure));
    }
    committed_plans_.erase(it);
}

void PDPGlobalMemory::register_committed_plan(int agent_id, float speed_mps) {
    DeliveryAgent* agent = get_delivery_agent(agent_id);
    if (!agent) return;

    AgentSolution& sol = agent->solution;
    if (!sol.valid() || sol.empty()) return;

    std::vector<std::pair<TimedPath, int>>& plan = committed_plans_[agent_id];
    plan.clear();

    int t = current_time_;

    // First leg: current node → sequence[0]
    {
        const ObjectiveNode& next = sol.sequence[0].node;
        const ObjectivePath* path = get_or_compute_path(*sol.current_position, next.id, next.group_id);
        if (path && path->valid()) {
            TimedPath tp = make_timed_path(*path, speed_mps, geo_box.data);
            sol.sequence[0].estimated_arrival = t + tp.total_steps;
            for (std::size_t i = 0; i < tp.edge_ids.size(); ++i)
                congestion_map.add_agent(tp.edge_ids[i], tp.abs_entry(i, t), tp.abs_exit(i, t));
            plan.push_back({std::move(tp), t});
            t = sol.sequence[0].estimated_arrival;
        }
    }

    // Remaining legs: sequence[k] → sequence[k+1]
    for (std::size_t k = 0; k + 1 < sol.sequence.size(); ++k) {
        const ObjectiveNode& a = sol.sequence[k].node;
        const ObjectiveNode& b = sol.sequence[k + 1].node;
        const ObjectivePath* path = get_or_compute_path(a.id, b.id, a.group_id);
        if (path && path->valid()) {
            TimedPath tp = make_timed_path(*path, speed_mps, geo_box.data);
            sol.sequence[k + 1].estimated_arrival = t + tp.total_steps;
            for (std::size_t i = 0; i < tp.edge_ids.size(); ++i)
                congestion_map.add_agent(tp.edge_ids[i], tp.abs_entry(i, t), tp.abs_exit(i, t));
            plan.push_back({std::move(tp), t});
            t = sol.sequence[k + 1].estimated_arrival;
        }
    }
}

void PDPGlobalMemory::commit_plan(int agent_id, float speed_mps) {
    unregister_committed_plan(agent_id);
    register_committed_plan  (agent_id, speed_mps);
}

// ---- Congestion-push reroute -------------------------------------------

void PDPGlobalMemory::push_rerouted_path(int agent_id, float speed_mps) {
    DeliveryAgent* agent = get_delivery_agent(agent_id);
    if (!agent) return;

    const AgentSolution& sol = agent->solution;
    if (sol.empty()) return;
    // Works both mid-traversal (has edge_cursor) and at leg boundaries
    // (edge_cursor reset). At boundaries, push_updated_path updates current_path
    // only; begin_leg() then uses that refreshed path to create the cursor.

    // The agent is heading from current_node to sequence[0].
    osmium::object_id_type from = agent->current_node;
    const ObjectiveNode&   to   = sol.sequence[0].node;

    // Refresh the dynamic cost via TD-A*.
    ObjectivePath* path = server_memory.refresh_dynamic_cost(
        from, to.id, to.group_id, speed_mps, congestion_map, current_time_);
    if (!path) return;

    // Only push if the dynamic cost is meaningfully better than the static path.
    const float improvement_threshold = 1.05f; // >5% better
    if (!path->has_dynamic_cost()) return;
    if (agent->local_memory.current_path &&
        agent->local_memory.current_path->cost > 0.f &&
        path->dynamic_cost >= agent->local_memory.current_path->cost / improvement_threshold)
        return;

    agent->push_updated_path(path);
    commit_plan(agent_id, speed_mps);
}

// ---- Objective cleanup -------------------------------------------------

void PDPGlobalMemory::clear_objective(osmium::object_id_type node_id, int group_id) {
    server_memory.remove_objective_node(node_id, group_id);
    node_to_task_id_.erase(node_id);
}

// ---- Path cache --------------------------------------------------------

void PDPGlobalMemory::ensure_task_group(int group_id) {
    server_memory.ensure_group(group_id);
}

const ObjectivePath* PDPGlobalMemory::get_or_compute_path(
    osmium::object_id_type from, osmium::object_id_type to, int group_id
) {
    return server_memory.get_or_compute_path(from, to, group_id);
}

const ObjectivePath* PDPGlobalMemory::discover_next_path(
    osmium::object_id_type from, int group_id
) {
    return server_memory.discover_next_path(from, group_id);
}

bool PDPGlobalMemory::is_group_complete(int group_id) const {
    return server_memory.is_group_complete(group_id);
}

TDAStarResult PDPGlobalMemory::time_dependent_astar(
    osmium::object_id_type from, osmium::object_id_type to,
    int start_time, float agent_speed
) const {
    return server_memory.time_dependent_astar(from, to, start_time, agent_speed, congestion_map);
}

// ---- Simulation clock --------------------------------------------------

int PDPGlobalMemory::current_time() const { return current_time_; }

void PDPGlobalMemory::advance_time(int t_now) {
    if (t_now <= current_time_) return;
    current_time_ = t_now;
    congestion_map.advance(t_now);
}

// ---- Episode reset -----------------------------------------------------

void PDPGlobalMemory::reset_episode() {
    // Clear all committed congestion contributions before wiping plans.
    for (auto& [aid, plan] : committed_plans_) {
        for (auto& [tp, departure] : plan) {
            for (std::size_t i = 0; i < tp.edge_ids.size(); ++i)
                congestion_map.remove_agent(tp.edge_ids[i],
                                            tp.abs_entry(i, departure),
                                            tp.abs_exit (i, departure));
        }
    }
    committed_plans_.clear();

    tasks_.clear();
    available_tasks.clear();
    allocated_tasks.clear();
    finished_tasks.clear();
    node_to_task_id_.clear();

    // Reset the congestion map: clears load_ and resets t_now_ to 0.
    // Without this, CongestionMap::update_load filters all new-episode steps
    // (t < t_now_) so no congestion is ever tracked after the first episode.
    congestion_map.reset();

    // Reset clock to 0 so advance_time() accepts the new episode's steps.
    current_time_ = 0;

    // Clear dynamic objective registrations (task nodes from the just-finished episode).
    // Preserves paths_ so A* results are reused across episodes.
    server_memory.reset_objectives(1);
}

// ---- Congestion --------------------------------------------------------

static void apply_route(CongestionMap& cmap, const ObjectivePath& path,
                         int start_time, float speed_mps,
                         const GeoBox& geo_box, int delta) {
    int t = start_time;
    for (const auto& way_id : path.edges) {
        auto it = geo_box.data.ways.find(way_id);
        if (it == geo_box.data.ways.end()) continue;
        const float dist   = it->second.distance_meters;
        const int transit  = (speed_mps > 0.0f)
                             ? std::max(1, static_cast<int>(std::ceil(dist / speed_mps)))
                             : 1;
        if (delta > 0) cmap.add_agent   (way_id, t, t + transit);
        else           cmap.remove_agent(way_id, t, t + transit);
        t += transit;
    }
}

void PDPGlobalMemory::register_agent_path(
    const ObjectivePath& path, int start_time, float speed_mps
) {
    apply_route(congestion_map, path, start_time, speed_mps, geo_box, +1);
}

void PDPGlobalMemory::unregister_agent_path(
    const ObjectivePath& path, int start_time, float speed_mps
) {
    apply_route(congestion_map, path, start_time, speed_mps, geo_box, -1);
}

float PDPGlobalMemory::congestion_cost(
    osmium::object_id_type way_id, float base_cost, float distance_meters, int t
) const {
    return congestion_map.adjusted_cost(way_id, base_cost, distance_meters, t);
}
