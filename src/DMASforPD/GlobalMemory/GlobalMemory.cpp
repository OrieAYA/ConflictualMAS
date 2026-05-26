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
    region_grid.init(box);
}

// ── RegionStatsGrid implementation ─────────────────────────────────────────

void RegionStatsGrid::init(const GeoBox& gb) {
    // Bounding box from the GeoBox. Fallback to scanning nodes if osmium::Box
    // is invalid for this dataset.
    double mn_lat =  90.0, mx_lat = -90.0;
    double mn_lon = 180.0, mx_lon = -180.0;
    if (gb.bbox.valid()) {
        mn_lat = gb.bbox.bottom_left().lat();
        mn_lon = gb.bbox.bottom_left().lon();
        mx_lat = gb.bbox.top_right().lat();
        mx_lon = gb.bbox.top_right().lon();
    } else {
        for (const auto& [id, pt] : gb.data.nodes) {
            if (pt.lat < mn_lat) mn_lat = pt.lat;
            if (pt.lat > mx_lat) mx_lat = pt.lat;
            if (pt.lon < mn_lon) mn_lon = pt.lon;
            if (pt.lon > mx_lon) mx_lon = pt.lon;
        }
    }
    if (mx_lat <= mn_lat || mx_lon <= mn_lon) {
        // Degenerate bbox — disable grid silently.
        inited = false;
        return;
    }
    min_lat = mn_lat;
    min_lon = mn_lon;
    cell_h_lat = (mx_lat - mn_lat) / static_cast<double>(kDim);
    cell_w_lon = (mx_lon - mn_lon) / static_cast<double>(kDim);

    // Precompute up to kMaxEdgesPerCell edge IDs per cell, sampled from way
    // midpoints. Used by refresh_congestion_cache to estimate per-cell BPR
    // without scanning the whole graph every refresh.
    for (int i = 0; i < kSize; ++i) cell_edges[i].clear();
    for (const auto& [wid, way] : gb.data.ways) {
        // Use one of the way's endpoints (node1 if known, else first point) as
        // a cheap stand-in for its centroid.
        double slat = 0, slon = 0; bool ok = false;
        if (way.node1_id != 0) {
            auto it = gb.data.nodes.find(way.node1_id);
            if (it != gb.data.nodes.end()) {
                slat = it->second.lat; slon = it->second.lon; ok = true;
            }
        }
        if (!ok && !way.points.empty()) {
            slat = way.points.front().lat;
            slon = way.points.front().lon;
            ok = true;
        }
        if (!ok) continue;
        const int c = cell_of(slat, slon);
        if (c < 0) continue;
        if (static_cast<int>(cell_edges[c].size()) < kMaxEdgesPerCell)
            cell_edges[c].push_back(wid);
    }

    reset_episode();
    inited = true;
}

void RegionStatsGrid::reset_episode() {
    for (int i = 0; i < kSize; ++i) {
        task_counts[i]      = 0;
        cell_cong_cache[i]  = 1.f;
    }
    task_events.clear();
    max_count          = 1;
    max_cell_cong      = 1.f;
    last_cache_refresh = -kCacheRefreshSteps;
}

int RegionStatsGrid::cell_of(double lat, double lon) const {
    if (!inited && cell_h_lat <= 0) return -1;
    int row = static_cast<int>((lat - min_lat) / cell_h_lat);
    int col = static_cast<int>((lon - min_lon) / cell_w_lon);
    if (row < 0) row = 0; if (row >= kDim) row = kDim - 1;
    if (col < 0) col = 0; if (col >= kDim) col = kDim - 1;
    return row * kDim + col;
}

void RegionStatsGrid::register_task(double lat, double lon, int step) {
    if (!inited) return;
    const int c = cell_of(lat, lon);
    if (c < 0) return;
    task_counts[c]++;
    task_events.push_back({step, c});
    if (task_counts[c] > max_count) max_count = task_counts[c];
    purge(step);
}

void RegionStatsGrid::purge(int now) {
    const int cutoff = now - window_steps;
    while (!task_events.empty() && task_events.front().step < cutoff) {
        const int c = task_events.front().cell;
        if (task_counts[c] > 0) --task_counts[c];
        task_events.pop_front();
    }
}

void RegionStatsGrid::refresh_congestion_cache(const CongestionMap& cmap,
                                                const GeoBox& gb,
                                                int step) {
    if (!inited) return;
    if (step - last_cache_refresh < kCacheRefreshSteps) return;
    last_cache_refresh = step;

    float mx = 1.f;
    for (int c = 0; c < kSize; ++c) {
        if (cell_edges[c].empty()) { cell_cong_cache[c] = 1.f; continue; }
        float sum_bpr = 0.f;
        int   n_sampled = 0;
        for (osmium::object_id_type wid : cell_edges[c]) {
            auto it = gb.data.ways.find(wid);
            if (it == gb.data.ways.end()) continue;
            const float dist = it->second.distance_meters;
            // base_cost=1 → adjusted_cost returns the multiplier directly.
            const float mul  = cmap.adjusted_cost(wid, 1.f, dist, step);
            sum_bpr += mul;
            ++n_sampled;
        }
        const float mean = (n_sampled > 0) ? sum_bpr / n_sampled : 1.f;
        cell_cong_cache[c] = mean;
        if (mean > mx) mx = mean;
    }
    max_cell_cong = mx;
}

float RegionStatsGrid::density_norm(double lat, double lon) const {
    if (!inited || max_count <= 0) return 0.f;
    const int c = cell_of(lat, lon);
    if (c < 0) return 0.f;
    return std::clamp(static_cast<float>(task_counts[c])
                      / static_cast<float>(max_count), 0.f, 1.f);
}

float RegionStatsGrid::congestion_norm(double lat, double lon) const {
    if (!inited) return 0.f;
    const int c = cell_of(lat, lon);
    if (c < 0) return 0.f;
    // Normalise (mul - 1) by (max_mul - 1) so a "free flow" cell scores 0 and
    // the worst cell scores 1. Guards against the degenerate case where the
    // whole grid is at BPR 1.0 (no congestion anywhere).
    const float denom = std::max(1e-3f, max_cell_cong - 1.f);
    return std::clamp((cell_cong_cache[c] - 1.f) / denom, 0.f, 1.f);
}

float RegionStatsGrid::area_heat(double lat, double lon) const {
    return density_norm(lat, lon) * congestion_norm(lat, lon);
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
    const int w = std::max(1, congestion_map.params.load_per_agent);
    for (auto& [tp, departure] : it->second) {
        for (std::size_t i = 0; i < tp.edge_ids.size(); ++i)
            congestion_map.remove_agent(tp.edge_ids[i],
                                        tp.abs_entry(i, departure),
                                        tp.abs_exit (i, departure), w);
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
    const int w = std::max(1, congestion_map.params.load_per_agent);

    // First leg: current node → sequence[0]
    {
        const ObjectiveNode& next = sol.sequence[0].node;
        const ObjectivePath* path = get_or_compute_path(*sol.current_position, next.id, next.group_id);
        if (path && path->valid()) {
            TimedPath tp = make_timed_path(*path, speed_mps, geo_box.data);
            sol.sequence[0].estimated_arrival = t + tp.total_steps;
            for (std::size_t i = 0; i < tp.edge_ids.size(); ++i)
                congestion_map.add_agent(tp.edge_ids[i], tp.abs_entry(i, t), tp.abs_exit(i, t), w);
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
                congestion_map.add_agent(tp.edge_ids[i], tp.abs_entry(i, t), tp.abs_exit(i, t), w);
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
    // Periodic refresh of the per-cell congestion cache (every
    // kCacheRefreshSteps steps). Cheap: O(N_cells × N_sampled_edges_per_cell).
    region_grid.refresh_congestion_cache(congestion_map, geo_box, t_now);
}

// ---- Episode reset -----------------------------------------------------

void PDPGlobalMemory::reset_episode() {
    // Clear all committed congestion contributions before wiping plans.
    const int w = std::max(1, congestion_map.params.load_per_agent);
    for (auto& [aid, plan] : committed_plans_) {
        for (auto& [tp, departure] : plan) {
            for (std::size_t i = 0; i < tp.edge_ids.size(); ++i)
                congestion_map.remove_agent(tp.edge_ids[i],
                                            tp.abs_entry(i, departure),
                                            tp.abs_exit (i, departure), w);
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

    // Wipe per-episode heatmap stats (keeps precomputed cell→edge sampling).
    region_grid.reset_episode();
}

// ---- Congestion --------------------------------------------------------

static void apply_route(CongestionMap& cmap, const ObjectivePath& path,
                         int start_time, float speed_mps,
                         const GeoBox& geo_box, int delta) {
    int t = start_time;
    const int w = std::max(1, cmap.params.load_per_agent);
    for (const auto& way_id : path.edges) {
        auto it = geo_box.data.ways.find(way_id);
        if (it == geo_box.data.ways.end()) continue;
        const float dist   = it->second.distance_meters;
        const int transit  = (speed_mps > 0.0f)
                             ? std::max(1, static_cast<int>(std::ceil(dist / speed_mps)))
                             : 1;
        if (delta > 0) cmap.add_agent   (way_id, t, t + transit, w);
        else           cmap.remove_agent(way_id, t, t + transit, w);
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
