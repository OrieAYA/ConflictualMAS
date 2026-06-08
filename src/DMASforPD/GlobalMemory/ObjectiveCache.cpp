#include "ObjectiveCache.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>

// ── Path-compute timing accumulator (single-threaded simulation) ────────────
namespace { long long g_path_compute_time_us = 0; }
long long PDPServerMemory::path_compute_time_us()        { return g_path_compute_time_us; }
void      PDPServerMemory::reset_path_compute_time()     { g_path_compute_time_us = 0; }

// ---- ObjPairHash -------------------------------------------------------

std::size_t ObjPairHash::operator()(
    const std::pair<osmium::object_id_type, osmium::object_id_type>& p
) const noexcept {
    std::size_t h = std::hash<osmium::object_id_type>{}(p.first);
    h ^= std::hash<osmium::object_id_type>{}(p.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

// ---- ObjectiveGroupCache -----------------------------------------------

ObjectiveGroupCache::PathKey ObjectiveGroupCache::make_key(
    osmium::object_id_type a, osmium::object_id_type b
) {
    return {std::min(a, b), std::max(a, b)};
}

void ObjectiveGroupCache::build_objective_set() {
    objective_ids_.clear();
    for (const auto& on : objective_nodes)
        objective_ids_.insert(on.id);
}

void ObjectiveGroupCache::register_in_set(osmium::object_id_type node_id) {
    objective_ids_.insert(node_id);
}

void ObjectiveGroupCache::episode_reset() {
    objective_nodes.clear();
    objective_ids_.clear();
    obj_pair_count_ = 0;
    search_states_.clear();
    // search_states_td_ MUST also be cleared — otherwise the TD-aware Dijkstra
    // states accumulate across episodes (one entry per agent × task origin),
    // growing unboundedly and ultimately triggering bad_alloc after ~10 rounds.
    search_states_td_.clear();
    // paths_ intentionally kept: the static geometry (nodes/edges/cost) is
    // permanent road-network data, safe to reuse across episodes AND methods.
    // BUT the per-path DYNAMIC cache (congestion-adjusted TD travel time) is
    // episode-specific: it was computed against the previous episode's clock and
    // congestion map. The clock resets to 0 each episode, so a leftover
    // dynamic_step (e.g. 1500) would wrongly satisfy the
    // "dynamic_step >= current_step" freshness guard in refresh_dynamic_cost and
    // leak the previous episode's congestion costs into this one (consumed by
    // push_rerouted_path at every leg boundary, and by the CongestionAware /
    // FaithfulCongestionAware branches). Invalidate it so each episode recomputes
    // its own dynamic costs from a clean slate.
    for (auto& [key, path] : paths_) {
        (void)key;
        path.dynamic_cost = std::numeric_limits<float>::max();
        path.dynamic_step = -1;
    }
}

bool ObjectiveGroupCache::is_complete() const {
    const std::size_t n = objective_nodes.size();
    return static_cast<std::size_t>(obj_pair_count_) >= n * (n - 1) / 2;
}

bool ObjectiveGroupCache::has_path(osmium::object_id_type a, osmium::object_id_type b) const {
    return paths_.count(make_key(a, b)) > 0;
}

const ObjectivePath* ObjectiveGroupCache::get_path(
    osmium::object_id_type a, osmium::object_id_type b
) const {
    auto it = paths_.find(make_key(a, b));
    return it != paths_.end() ? &it->second : nullptr;
}

ObjectivePath* ObjectiveGroupCache::get_path_mutable(
    osmium::object_id_type a, osmium::object_id_type b
) {
    auto it = paths_.find(make_key(a, b));
    return it != paths_.end() ? &it->second : nullptr;
}

void ObjectiveGroupCache::store_path(const ObjectivePath& path) {
    if (objective_ids_.count(path.node_a) && objective_ids_.count(path.node_b))
        ++obj_pair_count_;
    paths_.emplace(make_key(path.node_a, path.node_b), path);
}

bool ObjectiveGroupCache::contains_objective(osmium::object_id_type node_id) const {
    return objective_ids_.count(node_id) > 0;
}

void ObjectiveGroupCache::remove_node(osmium::object_id_type node_id) {
    objective_ids_.erase(node_id);
    objective_nodes.erase(
        std::remove_if(objective_nodes.begin(), objective_nodes.end(),
            [node_id](const ObjectiveNode& n){ return n.id == node_id; }),
        objective_nodes.end()
    );
    // Update obj_pair_count_ for paths that had both endpoints as objectives.
    // Do NOT erase paths: they are permanently memoized A* results on the road
    // network, which never changes. Erasing them forces A* to rerun for the same
    // node pairs in subsequent episodes, which is the main source of training slowness.
    for (const auto& [key, _] : paths_) {
        if (key.first == node_id || key.second == node_id) {
            const osmium::object_id_type other =
                (key.first == node_id) ? key.second : key.first;
            if (objective_ids_.count(other))
                --obj_pair_count_;
        }
    }
    // Invalidate the Dijkstra search state originating from this node only.
    search_states_.erase(node_id);
}

std::vector<const ObjectivePath*> ObjectiveGroupCache::paths_from(
    osmium::object_id_type node_id
) const {
    std::vector<const ObjectivePath*> result;
    for (const auto& [key, path] : paths_)
        if (path.node_a == node_id || path.node_b == node_id)
            result.push_back(&path);
    return result;
}

ObjectivePath ObjectiveGroupCache::reconstruct_path(
    const SearchState& state,
    osmium::object_id_type from,
    osmium::object_id_type to,
    float cost
) const {
    ObjectivePath result;
    result.node_a = std::min(from, to);
    result.node_b = std::max(from, to);
    result.cost   = cost;

    std::vector<osmium::object_id_type> nodes_rev, edges;
    osmium::object_id_type cur = to;
    nodes_rev.push_back(cur);
    while (cur != from) {
        auto it = state.came_from.find(cur);
        if (it == state.came_from.end()) break;
        auto [prev, way_id] = it->second;
        edges.push_back(way_id);
        cur = prev;
        nodes_rev.push_back(cur);
    }
    std::reverse(nodes_rev.begin(), nodes_rev.end());
    std::reverse(edges.begin(), edges.end());
    result.nodes = std::move(nodes_rev);
    result.edges = std::move(edges);

    for (const auto& nid : result.nodes) {
        if (nid == result.node_a || nid == result.node_b) continue;
        if (objective_ids_.count(nid))
            result.intermediate_objectives.push_back(nid);
    }
    return result;
}

// Core incremental Dijkstra shared by discover_step and discover_next_path.
ObjectiveGroupCache::DiscoveryStep ObjectiveGroupCache::discover_step(
    osmium::object_id_type from,
    const MyData&          data,
    const std::unordered_set<osmium::object_id_type>& agent_positions,
    float                  max_cost
) {
    // When searching for agents (agent_positions non-empty), skip the
    // is_complete() gate: all pairwise objective paths may be cached, but
    // agents are not objectives — the Dijkstra must continue to find them.
    // is_complete() still applies for pure path-discovery calls (agent_positions empty).
    if (agent_positions.empty() && is_complete()) return {};

    auto& state = search_states_[from];
    if (state.exhausted) return {};

    if (state.g_score.empty()) {
        state.g_score[from] = 0.0f;
        state.open.push_back({0.0f, from});
    }

    using OpenEntry = SearchState::OpenEntry;
    auto cmp = [](const OpenEntry& a, const OpenEntry& b){ return a.first > b.first; };

    while (!state.open.empty()) {
        std::pop_heap(state.open.begin(), state.open.end(), cmp);
        auto [cost, current] = state.open.back();
        state.open.pop_back();

        if (state.closed.count(current)) continue;

        // Spatial pruning: if the cheapest open node exceeds the search budget,
        // all remaining nodes also exceed it (min-heap). Push current back so a
        // later call with a larger max_cost can resume past this frontier
        // (recall mechanism). state.exhausted stays false → transient stop.
        if (cost > max_cost) {
            state.open.push_back({cost, current});
            std::push_heap(state.open.begin(), state.open.end(), cmp);
            return {};
        }

        state.closed.insert(current);

        // Expand neighbors before checking stops (ensures paths through this node are usable later).
        auto node_it = data.nodes.find(current);
        if (node_it != data.nodes.end()) {
            for (const auto& way_id : node_it->second.incident_ways) {
                auto way_it = data.ways.find(way_id);
                if (way_it == data.ways.end()) continue;
                const auto& way = way_it->second;
                osmium::object_id_type nb =
                    (way.node1_id == current) ? way.node2_id : way.node1_id;
                if (state.closed.count(nb)) continue;
                float nc = cost + way.distance_meters;
                if (nc > max_cost) continue;            // skip heap push beyond budget
                auto g_it = state.g_score.find(nb);
                if (g_it == state.g_score.end() || nc < g_it->second) {
                    state.g_score[nb]   = nc;
                    state.came_from[nb] = {current, way_id};
                    state.open.push_back({nc, nb});
                    std::push_heap(state.open.begin(), state.open.end(), cmp);
                }
            }
        }

        // Check: idle agent position (only relevant from delivery side per TAM spec).
        if (current != from && !agent_positions.empty() && agent_positions.count(current)) {
            DiscoveryStep step;
            step.type       = DiscoveryStep::Type::AgentPosition;
            step.agent_node = current;
            step.cost       = cost;
            return step;
        }

        // Check: new uncached objective node.
        if (current != from && objective_ids_.count(current) && !has_path(from, current)) {
            ObjectivePath path = reconstruct_path(state, from, current, cost);
            store_path(path);
            DiscoveryStep step;
            step.type = DiscoveryStep::Type::Path;
            step.path = get_path(from, current);
            return step;
        }
    }

    state.exhausted = true;
    return {};  // type == Exhausted
}

// ── Time-Dependent variant of discover_step ────────────────────────────────
// Cost accumulator is in TIME units (steps); each edge contributes the
// BPR-adjusted travel time evaluated at the predicted arrival step at the
// edge's tail (start_step + ceil(g_score_so_far)). Reconstructed paths
// store both the static distance (path.cost) and the TD-time (path.dynamic_cost).
ObjectiveGroupCache::DiscoveryStep ObjectiveGroupCache::discover_step_td(
    osmium::object_id_type from,
    const MyData&          data,
    int                    start_step,
    float                  speed_mps,
    const CongestionMap&   congestion,
    const std::unordered_set<osmium::object_id_type>& agent_positions,
    float                  max_cost
) {
    if (agent_positions.empty() && is_complete()) return {};
    if (speed_mps <= 0.f) speed_mps = 1.f;

    auto& state = search_states_td_[from];
    if (state.exhausted) return {};

    if (state.g_score.empty()) {
        state.g_score[from] = 0.0f;          // g_score = TD-time in steps
        state.open.push_back({0.0f, from});
    }

    using OpenEntry = SearchState::OpenEntry;
    auto cmp = [](const OpenEntry& a, const OpenEntry& b){ return a.first > b.first; };

    while (!state.open.empty()) {
        std::pop_heap(state.open.begin(), state.open.end(), cmp);
        auto [cost, current] = state.open.back();
        state.open.pop_back();

        if (state.closed.count(current)) continue;

        // Time-budget pruning (same recall semantics as the legacy variant).
        if (cost > max_cost) {
            state.open.push_back({cost, current});
            std::push_heap(state.open.begin(), state.open.end(), cmp);
            return {};
        }

        state.closed.insert(current);

        auto node_it = data.nodes.find(current);
        if (node_it != data.nodes.end()) {
            const int t_enter = start_step
                              + static_cast<int>(std::ceil(cost));
            for (const auto& way_id : node_it->second.incident_ways) {
                auto way_it = data.ways.find(way_id);
                if (way_it == data.ways.end()) continue;
                const auto& way = way_it->second;
                osmium::object_id_type nb =
                    (way.node1_id == current) ? way.node2_id : way.node1_id;
                if (state.closed.count(nb)) continue;
                const float length_m = way.distance_meters;
                if (length_m <= 0.f) continue;
                const float base_time = length_m / speed_mps;
                const float adj_time  = congestion.adjusted_cost(
                                            way_id, base_time, length_m, t_enter);
                float nc = cost + adj_time;
                if (nc > max_cost) continue;
                auto g_it = state.g_score.find(nb);
                if (g_it == state.g_score.end() || nc < g_it->second) {
                    state.g_score[nb]   = nc;
                    state.came_from[nb] = {current, way_id};
                    state.open.push_back({nc, nb});
                    std::push_heap(state.open.begin(), state.open.end(), cmp);
                }
            }
        }

        if (current != from && !agent_positions.empty() && agent_positions.count(current)) {
            DiscoveryStep step;
            step.type       = DiscoveryStep::Type::AgentPosition;
            step.agent_node = current;
            step.cost       = cost;             // TD-time at this agent's node
            return step;
        }

        if (current != from && objective_ids_.count(current) && !has_path(from, current)) {
            // Reconstruct path; recompute static distance separately so
            // path.cost retains its legacy "free-flow distance" semantics.
            ObjectivePath path = reconstruct_path(state, from, current, cost);
            // Override path.cost with static distance, store TD-time in dynamic_cost.
            float static_dist = 0.f;
            for (auto eid : path.edges) {
                auto wit = data.ways.find(eid);
                if (wit != data.ways.end()) static_dist += wit->second.distance_meters;
            }
            path.cost = static_dist;
            path.dynamic_cost = cost;
            path.dynamic_step = start_step;
            store_path(path);
            DiscoveryStep step;
            step.type = DiscoveryStep::Type::Path;
            step.path = get_path(from, current);
            step.cost = cost;                   // TD-time
            return step;
        }
    }

    state.exhausted = true;
    return {};
}

const ObjectivePath* ObjectiveGroupCache::discover_next_path(
    osmium::object_id_type from, const MyData& data
) {
    auto step = discover_step(from, data);
    return (step.type == DiscoveryStep::Type::Path) ? step.path : nullptr;
}

// ---- PDPServerMemory ---------------------------------------------------

PDPServerMemory::PDPServerMemory(GeoBox& box, Pathfinder& pf)
    : geo_box(box), pathfinder(pf) {}

void PDPServerMemory::initialize_from_geobox() {
    group_caches_.clear();
    for (const auto& [gid, group] : geo_box.data.objective_groups) {
        auto& cache    = group_caches_[gid];
        cache.group_id = gid;
        cache.objective_nodes.reserve(group.node_ids.size());
        for (const auto& nid : group.node_ids)
            cache.objective_nodes.push_back({nid, gid});
        cache.build_objective_set();
    }
    std::cout << "[PDPServerMemory] Initialized " << group_caches_.size() << " group caches\n";
}

void PDPServerMemory::ensure_group(int group_id) {
    if (!group_caches_.count(group_id))
        group_caches_[group_id].group_id = group_id;
}

const ObjectivePath* PDPServerMemory::get_or_compute_path(
    osmium::object_id_type from, osmium::object_id_type to, int group_id
) {
    const auto t0 = std::chrono::steady_clock::now();
    auto it = group_caches_.find(group_id);
    const ObjectivePath* result = nullptr;
    if (it != group_caches_.end()) {
        auto& cache = it->second;
        if (!cache.has_path(from, to))
            cache.store_path(compute_path(from, to, group_id));
        result = cache.get_path(from, to);
    }
    const auto t1 = std::chrono::steady_clock::now();
    g_path_compute_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return result;
}

const ObjectivePath* PDPServerMemory::discover_next_path(
    osmium::object_id_type from, int group_id
) {
    auto it = group_caches_.find(group_id);
    if (it == group_caches_.end()) return nullptr;
    return it->second.discover_next_path(from, geo_box.data);
}

ObjectiveGroupCache::DiscoveryStep PDPServerMemory::discover_step(
    osmium::object_id_type from,
    int group_id,
    const std::unordered_set<osmium::object_id_type>& agent_positions,
    float max_cost
) {
    auto it = group_caches_.find(group_id);
    if (it == group_caches_.end()) return {};
    return it->second.discover_step(from, geo_box.data, agent_positions, max_cost);
}

ObjectiveGroupCache::DiscoveryStep PDPServerMemory::discover_step_td(
    osmium::object_id_type from,
    int group_id,
    int start_step,
    float speed_mps,
    const CongestionMap& congestion,
    const std::unordered_set<osmium::object_id_type>& agent_positions,
    float max_cost
) {
    auto it = group_caches_.find(group_id);
    if (it == group_caches_.end()) return {};
    return it->second.discover_step_td(
        from, geo_box.data, start_step, speed_mps, congestion,
        agent_positions, max_cost);
}

ObjectivePath* PDPServerMemory::refresh_dynamic_cost(
    osmium::object_id_type from, osmium::object_id_type to,
    int group_id, float speed_mps,
    const CongestionMap& congestion, int current_step
) {
    auto it = group_caches_.find(group_id);
    if (it == group_caches_.end()) return nullptr;
    ObjectivePath* path = it->second.get_path_mutable(from, to);
    if (!path || !path->valid()) return nullptr;
    if (path->dynamic_step >= current_step) return path;  // already fresh

    TDAStarResult tda = time_dependent_astar(from, to, current_step, speed_mps, congestion);
    if (tda.valid()) {
        path->dynamic_cost = static_cast<float>(tda.total_time);
        path->dynamic_step = current_step;
    }
    return path;
}

void PDPServerMemory::store_path_in_group(int group_id, const ObjectivePath& path) {
    auto it = group_caches_.find(group_id);
    if (it != group_caches_.end())
        it->second.store_path(path);
}

void PDPServerMemory::reset_agent_search(osmium::object_id_type from, int group_id) {
    auto it = group_caches_.find(group_id);
    if (it != group_caches_.end())
        it->second.reset_search_state(from);
}

void PDPServerMemory::reset_objectives(int group_id) {
    auto it = group_caches_.find(group_id);
    if (it == group_caches_.end()) return;
    it->second.episode_reset();
}

void PDPServerMemory::add_objective_node(osmium::object_id_type node_id, int group_id) {
    ensure_group(group_id);
    auto& cache = group_caches_.at(group_id);
    if (cache.contains_objective(node_id)) return;
    cache.objective_nodes.push_back({node_id, group_id});
    cache.register_in_set(node_id);
}

void PDPServerMemory::remove_objective_node(osmium::object_id_type node_id, int group_id) {
    auto it = group_caches_.find(group_id);
    if (it != group_caches_.end())
        it->second.remove_node(node_id);
}

bool PDPServerMemory::is_objective_node(osmium::object_id_type node_id, int group_id) const {
    auto it = group_caches_.find(group_id);
    return it != group_caches_.end() && it->second.contains_objective(node_id);
}

bool PDPServerMemory::is_group_complete(int group_id) const {
    auto it = group_caches_.find(group_id);
    return it != group_caches_.end() && it->second.is_complete();
}

std::vector<const ObjectivePath*> PDPServerMemory::get_paths_from(
    osmium::object_id_type node_id, int group_id
) const {
    auto it = group_caches_.find(group_id);
    if (it == group_caches_.end()) return {};
    return it->second.paths_from(node_id);
}

ObjectivePath PDPServerMemory::compute_path(
    osmium::object_id_type from, osmium::object_id_type to, int group_id
) {
    ObjectivePath result;
    result.node_a = std::min(from, to);
    result.node_b = std::max(from, to);

    // Use TD-A* with no congestion (speed=1 m/s, start_time=0) for static distance.
    static const CongestionMap empty_congestion;
    TDAStarResult tda = time_dependent_astar(from, to, 0, 1.0f, empty_congestion);
    if (!tda.valid() || tda.edges.empty()) return result;

    result.edges = std::move(tda.edges);
    result.nodes = std::move(tda.nodes);
    result.cost  = tda.total_distance;
    tag_intermediate_objectives(result, group_id);
    return result;
}

void PDPServerMemory::tag_intermediate_objectives(ObjectivePath& path, int group_id) {
    auto cache_it = group_caches_.find(group_id);
    if (cache_it == group_caches_.end()) return;
    for (const auto& nid : path.nodes) {
        if (nid == path.node_a || nid == path.node_b) continue;
        if (cache_it->second.contains_objective(nid))
            path.intermediate_objectives.push_back(nid);
    }
}

// ---- Time-Dependent A* -------------------------------------------------

namespace {

double haversine_m(double lat1, double lon1, double lat2, double lon2) {
    constexpr double R    = 6371000.0;
    constexpr double kPi  = 3.14159265358979323846;
    constexpr double kDeg = kPi / 180.0;
    const double phi1  = lat1 * kDeg, phi2 = lat2 * kDeg;
    const double dphi  = (lat2 - lat1) * kDeg, dlam = (lon2 - lon1) * kDeg;
    const double a = std::sin(dphi/2)*std::sin(dphi/2)
                   + std::cos(phi1)*std::cos(phi2)*std::sin(dlam/2)*std::sin(dlam/2);
    return R * 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
}

}  // namespace

TDAStarResult PDPServerMemory::time_dependent_astar(
    osmium::object_id_type from,
    osmium::object_id_type to,
    int                    start_time,
    float                  agent_speed,
    const CongestionMap&   congestion
) const {
    TDAStarResult result;

    if (from == to) {
        result.nodes = {from}; result.total_time = 0; result.total_distance = 0.0f;
        return result;
    }

    auto goal_it = geo_box.data.nodes.find(to);
    if (goal_it == geo_box.data.nodes.end()) return result;

    const double goal_lat  = goal_it->second.lat;
    const double goal_lon  = goal_it->second.lon;
    const float  inv_speed = (agent_speed > 0.0f) ? 1.0f / agent_speed : 0.0f;

    auto heuristic = [&](osmium::object_id_type n) -> float {
        auto it = geo_box.data.nodes.find(n);
        if (it == geo_box.data.nodes.end()) return 0.0f;
        return static_cast<float>(
            haversine_m(it->second.lat, it->second.lon, goal_lat, goal_lon)
        ) * inv_speed;
    };

    std::unordered_map<osmium::object_id_type, float> g;
    std::unordered_map<osmium::object_id_type,
        std::pair<osmium::object_id_type, osmium::object_id_type>> came_from;
    std::unordered_set<osmium::object_id_type> closed;
    using Entry = std::pair<float, osmium::object_id_type>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;

    g[from] = 0.0f;
    open.push({heuristic(from), from});

    while (!open.empty()) {
        auto [f, current] = open.top(); open.pop();
        if (closed.count(current)) continue;
        closed.insert(current);

        if (current == to) {
            std::vector<osmium::object_id_type> nodes_rev, edges;
            float dist = 0.0f;
            osmium::object_id_type cur = to;
            nodes_rev.push_back(cur);
            while (cur != from) {
                auto it = came_from.find(cur);
                if (it == came_from.end()) break;
                auto [prev, way_id] = it->second;
                edges.push_back(way_id);
                cur = prev;
                nodes_rev.push_back(cur);
                auto way_it = geo_box.data.ways.find(way_id);
                if (way_it != geo_box.data.ways.end()) dist += way_it->second.distance_meters;
            }
            std::reverse(nodes_rev.begin(), nodes_rev.end());
            std::reverse(edges.begin(), edges.end());
            result.nodes          = std::move(nodes_rev);
            result.edges          = std::move(edges);
            result.total_time     = static_cast<int>(std::round(g.at(to)));
            result.total_distance = dist;
            return result;
        }

        auto node_it = geo_box.data.nodes.find(current);
        if (node_it == geo_box.data.nodes.end()) continue;

        const int t_arrival = start_time + static_cast<int>(g.at(current));

        for (const auto& way_id : node_it->second.incident_ways) {
            auto way_it = geo_box.data.ways.find(way_id);
            if (way_it == geo_box.data.ways.end()) continue;
            const auto& way = way_it->second;
            const osmium::object_id_type nb =
                (way.node1_id == current) ? way.node2_id : way.node1_id;
            if (closed.count(nb)) continue;

            const float base_steps = way.distance_meters * inv_speed;
            const float edge_time  = congestion.adjusted_cost(
                way_id, base_steps, way.distance_meters, t_arrival);
            const float new_g = g.at(current) + edge_time;
            auto g_it = g.find(nb);
            if (g_it == g.end() || new_g < g_it->second) {
                g[nb]         = new_g;
                came_from[nb] = {current, way_id};
                open.push({new_g + heuristic(nb), nb});
            }
        }
    }
    return result;
}
