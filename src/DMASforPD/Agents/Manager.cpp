#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Structures/AgentSolution.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

// ---- Construction -------------------------------------------------------

PDPGlobalMemory::PDPGlobalMemory(GeoBox& box,
                                 const CongestionParams& cparams,
                                 const TaskAgentParams& taparams)
    : geo_box(box), server_memory(box),
      congestion_map(cparams), node_events(box, 1e30f), task_agent(0, taparams) {
    server_memory.initialize_from_geobox();
    region_grid.init(box);
}

//  RegionStatsGrid implementation

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

    // Node temporal layer: the task is present at its pickup/delivery from
    // creation to end of episode.
    const float t_end  = total_steps > 0 ? static_cast<float>(total_steps) : 1e9f;
    const float span   = t_end - static_cast<float>(current_time_);
    for (osmium::object_id_type nid : { pickup.id, delivery.id })
        node_events.node_chain(nid).insert(t_end, span)->objective_id.insert(id);
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

const DeliveryAgent* PDPGlobalMemory::get_delivery_agent(int agent_id) const {
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

// Build a TimedPath from a cached ObjectivePath at a given speed, ORIENTED so
TimedPath make_timed_path(const ObjectivePath& path,
                          osmium::object_id_type from,
                          float speed_mps,
                          const MyData& data, const CongestionMap& congestion,
                          int leg_start_step) {
    TimedPath tp;
    if (path.nodes.size() < 2) {
        tp.from_node = path.node_a;
        tp.to_node   = path.node_b;
        return tp;
    }
    const bool forward = (path.nodes.front() == from) || (path.nodes.back() != from);

    tp.total_distance = path.cost;
    const std::size_t ne = path.edges.size();
    tp.nodes.reserve(path.nodes.size());
    tp.edge_ids.reserve(ne);
    tp.edge_steps.reserve(ne);
    if (forward) tp.nodes = path.nodes;
    else         tp.nodes.assign(path.nodes.rbegin(), path.nodes.rend());
    tp.from_node = tp.nodes.front();
    tp.to_node   = tp.nodes.back();

    for (std::size_t k = 0; k < ne; ++k) {
        const osmium::object_id_type way_id =
            forward ? path.edges[k] : path.edges[ne - 1 - k];
        auto it = data.ways.find(way_id);
        const float dist = (it != data.ways.end()) ? it->second.distance_meters : 0.0f;
        const int entry  = leg_start_step + tp.total_steps;
        const int steps  = congestion.traversal_steps(way_id, dist, entry, speed_mps);
        tp.edge_ids.push_back(way_id);
        tp.edge_steps.push_back(steps);
        tp.total_steps += steps;
    }
    tp.seal();
    return tp;
}

}  // namespace

void PDPGlobalMemory::unregister_committed_plan(int agent_id) {
    auto it = committed_loads_.find(agent_id);
    if (it == committed_loads_.end()) return;
    // Exact removal of what was added: the ledger holds the (way, t_lo, t_hi,
    for (const LoadWindow& lw : it->second)
        congestion_map.remove_agent(lw.way, lw.t_lo, lw.t_hi, lw.weight);
    committed_loads_.erase(it);
}

void PDPGlobalMemory::register_committed_plan(int agent_id, float speed_mps) {
    DeliveryAgent* agent = get_delivery_agent(agent_id);
    if (!agent) return;

    AgentSolution& sol = agent->solution;
    if (!sol.valid() || sol.empty()) return;

    std::vector<LoadWindow>& ledger = committed_loads_[agent_id];
    ledger.clear();

    int t = current_time_;
    const int w  = std::max(1, congestion_map.params.load_per_agent);
    const int lo = congestion_map.window_lo();
    const int hi = congestion_map.window_hi();

    if (record_plan_congestion) agent->local_memory.plan_cong.clear();

    auto register_leg = [&](osmium::object_id_type from, const ObjectiveNode& to)
        -> int /* arrival step, or -1 if no path */ {
        const ObjectivePath* path = get_or_compute_path(from, to.id, to.group_id);
        if (!path || !path->valid()) return -1;
        TimedPath tp = make_timed_path(*path, from, speed_mps,
                                       geo_box.data, congestion_map, t);
        for (std::size_t i = 0; i < tp.edge_ids.size(); ++i) {
            if (record_plan_congestion) {
                auto wit = geo_box.data.ways.find(tp.edge_ids[i]);
                const float dist = (wit != geo_box.data.ways.end())
                    ? wit->second.distance_meters : 0.f;
                const float cap = congestion_map.edge_capacity(dist);
                const float x   = congestion_map.get_load(
                    tp.edge_ids[i], tp.abs_entry(i, t)) / std::max(1.f, cap);
                agent->local_memory.plan_cong[tp.edge_ids[i]] = x / (1.f + x);
            }
            const int e_lo = std::max(lo, tp.abs_entry(i, t));
            const int e_hi = std::min(hi, tp.abs_exit (i, t));
            if (e_lo > e_hi) continue;          // outside the registrable window
            congestion_map.add_agent(tp.edge_ids[i], e_lo, e_hi, w, agent_id);
            ledger.push_back({tp.edge_ids[i], e_lo, e_hi, w});
        }
        return t + tp.total_steps;
    };

    // First leg: current node → sequence[0], then sequence[k] → sequence[k+1].
    osmium::object_id_type from = *sol.current_position;
    for (std::size_t k = 0; k < sol.sequence.size(); ++k) {
        const int arrival = register_leg(from, sol.sequence[k].node);
        if (arrival < 0) break;
        sol.sequence[k].estimated_arrival = arrival;
        t    = arrival;
        from = sol.sequence[k].node.id;
    }
}

void PDPGlobalMemory::commit_plan(int agent_id, float speed_mps) {
    unregister_committed_plan(agent_id);
    register_committed_plan  (agent_id, speed_mps);
}

// ---- Congestion-push reroute -------------------------------------------

void PDPGlobalMemory::push_rerouted_path(int agent_id, float speed_mps,
                                         RerouteOutcome* out) {
    DeliveryAgent* agent = get_delivery_agent(agent_id);
    if (!agent) return;

    const AgentSolution& sol = agent->solution;
    if (sol.empty()) return;
    // Works both mid-traversal (has edge_cursor) and at leg boundaries

    // The agent is heading from current_node to sequence[0].
    const osmium::object_id_type from = agent->current_node;
    const ObjectiveNode&         to   = sol.sequence[0].node;
    if (from == to.id) return;

    // Remaining travel time of the CURRENT route, BPR-replayed from the
    // agent's position (time units: steps — same units as TD-A* below).
    const ObjectivePath* cur = agent->local_memory.current_path;
    if (!cur || !cur->valid()) return;
    const int cur_steps = path_travel_steps(*cur, from, current_time_, speed_mps);
    if (cur_steps == std::numeric_limits<int>::max()) return;

    // Congestion-aware alternative.
    TDAStarResult tda = time_dependent_astar(from, to.id, current_time_, speed_mps);
    if (!tda.valid() || tda.nodes.size() < 2) return;

    if (out) {
        out->attempted = true;
        out->cur_steps = cur_steps;
        out->tda_steps = tda.total_time;
    }

    // Reroute only on a meaningful (>5%) travel-time improvement.
    if (static_cast<float>(tda.total_time)
        >= static_cast<float>(cur_steps) / 1.05f)
        return;

    if (out) out->adopted = true;

    // Adopt the TD-A* GEOMETRY (not just its cost): copy it into the agent's
    // owned reroute slot so current_path can outlive the cache entry.
    float static_dist = 0.f;
    for (auto eid : tda.edges) {
        auto wit = geo_box.data.ways.find(eid);
        if (wit != geo_box.data.ways.end())
            static_dist += wit->second.distance_meters;
    }
    ObjectivePath& rp = agent->local_memory.reroute_path;
    rp.node_a       = std::min(from, to.id);
    rp.node_b       = std::max(from, to.id);
    rp.nodes        = std::move(tda.nodes);
    rp.edges        = std::move(tda.edges);
    rp.cost         = static_dist;
    rp.dynamic_cost = static_cast<float>(tda.total_time);
    rp.dynamic_step = current_time_;
    rp.intermediate_objectives.clear();

    agent->push_updated_path(&rp);
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

float PDPGlobalMemory::bpr_path_cost(
    osmium::object_id_type from, osmium::object_id_type to,
    int group_id, int depart_step, float speed_mps, int self_weight
) {
    const ObjectivePath* p = get_or_compute_path(from, to, group_id);
    if (!p || !p->valid()) return std::numeric_limits<float>::max();
    if (p->edges.empty()) return p->cost;          // same node / trivial hop

    // Orient the replay: the cache stores one direction only, but the BPR
    const bool forward =
        p->nodes.size() < 2 || p->nodes.front() == from || p->nodes.back() != from;

    const auto& ways = geo_box.data.ways;
    const float spd  = (speed_mps > 0.f) ? speed_mps : 1.f;
    const std::size_t ne = p->edges.size();
    float acc = 0.f;                                // adjusted distance-equiv (meters)
    float t   = static_cast<float>(depart_step);    // running predicted arrival time
    for (std::size_t k = 0; k < ne; ++k) {
        const osmium::object_id_type wid = forward ? p->edges[k] : p->edges[ne - 1 - k];
        auto it = ways.find(wid);
        const float d = (it != ways.end()) ? it->second.distance_meters : 0.f;
        if (d <= 0.f) continue;
        const float adj = congestion_map.adjusted_cost(wid, d, d,
                                                       static_cast<int>(t), self_weight);
        acc += adj;
        t   += adj / spd;                           // advance time along the path
    }
    return acc;
}

int PDPGlobalMemory::path_travel_steps(
    const ObjectivePath& path,
    osmium::object_id_type from,
    int depart_step, float speed_mps
) const {
    constexpr int kUnreachable = std::numeric_limits<int>::max();
    if (!path.valid() || path.nodes.size() < 2) return kUnreachable;

    // Locate the start within the path, oriented so the walk begins at `from`.
    const auto& nodes = path.nodes;
    const std::size_t nn = nodes.size();
    const std::size_t ne = path.edges.size();
    const bool forward = (nodes.front() == from) || (nodes.back() != from);

    std::size_t start_edge = nn;   // sentinel = not found
    for (std::size_t i = 0; i < nn; ++i) {
        const std::size_t oi = forward ? i : (nn - 1 - i);
        if (nodes[oi] == from) { start_edge = i; break; }
    }
    if (start_edge >= nn) return kUnreachable;   // `from` not on this path
    if (start_edge >= ne) return 0;              // already at the far endpoint

    const auto& ways = geo_box.data.ways;
    int t = depart_step;
    for (std::size_t k = start_edge; k < ne; ++k) {
        const osmium::object_id_type wid =
            forward ? path.edges[k] : path.edges[ne - 1 - k];
        auto it = ways.find(wid);
        const float d = (it != ways.end()) ? it->second.distance_meters : 0.f;
        t += congestion_map.traversal_steps(wid, d, t, speed_mps);
    }
    return t - depart_step;
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

    // Free the memoised geometry of delivered/picked objective nodes. Deferred
    if (kRetiredPurgeSteps > 0 && (t_now % kRetiredPurgeSteps) == 0)
        purge_retired_paths();
}

void PDPGlobalMemory::purge_retired_paths() {
    std::unordered_set<const ObjectivePath*> live;
    live.reserve(delivery_agents_.size() * 2);
    for (const auto& [aid, agent] : delivery_agents_) {
        (void)aid;
        if (!agent) continue;
        if (agent->local_memory.current_path) live.insert(agent->local_memory.current_path);
        if (agent->local_memory.next_path)    live.insert(agent->local_memory.next_path);
    }
    server_memory.purge_retired_paths(live);
}

// ---- Episode reset -----------------------------------------------------

void PDPGlobalMemory::reset_episode() {
    // Drop the committed-plan ledger. No need to subtract the loads one by
    // one: congestion_map.reset() below wipes every entry anyway.
    committed_loads_.clear();

    tasks_.clear();
    available_tasks.clear();
    allocated_tasks.clear();
    finished_tasks.clear();
    node_to_task_id_.clear();

    // Reset the congestion map: drops all occupancy intervals and t_now_ to 0.
    congestion_map.reset();
    node_events.reset();

    // Reset clock to 0 so advance_time() accepts the new episode's steps.
    current_time_ = 0;

    // Clear dynamic objective registrations (task nodes from the just-finished episode).
    // Preserves paths_ so A* results are reused across episodes.
    server_memory.reset_objectives(1);

    // Wipe per-episode heatmap stats (keeps precomputed cell→edge sampling).
    region_grid.reset_episode();
}

// ---- Congestion --------------------------------------------------------

float PDPGlobalMemory::congestion_cost(
    osmium::object_id_type way_id, float base_cost, float distance_meters, int t
) const {
    return congestion_map.adjusted_cost(way_id, base_cost, distance_meters, t);
}