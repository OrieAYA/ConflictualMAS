#include "Environment/Simulation/CongestionMap.hpp"
#include <algorithm>

CongestionMap::CongestionMap(const CongestionParams& p) : params(p) {}

int CongestionMap::current_step() const { return t_now_; }

float CongestionMap::edge_capacity(float distance_meters) const {
    return std::max(1.0f, distance_meters * params.capacity_per_meter);
}

void CongestionMap::update_load(
    osmium::object_id_type way_id, int t_start, int t_end, int delta
) {
    for (int t = t_start; t <= t_end; ++t) {
        if (t < t_now_ || t > t_now_ + params.horizon) continue;
        auto& count = load_[way_id][t];
        count = std::max(0, count + delta);
        if (count == 0) load_[way_id].erase(t);
    }
    if (auto it = load_.find(way_id); it != load_.end() && it->second.empty())
        load_.erase(it);
}

void CongestionMap::add_agent(
    osmium::object_id_type way_id, int t_enter, int t_exit, int weight
) {
    update_load(way_id, t_enter, t_exit, +weight);
}

void CongestionMap::remove_agent(
    osmium::object_id_type way_id, int t_enter, int t_exit, int weight
) {
    update_load(way_id, t_enter, t_exit, -weight);
}

void CongestionMap::add_ghost_load(
    osmium::object_id_type way_id, int t_enter, int t_exit, int weight
) {
    update_load(way_id, t_enter, t_exit, +weight);
}

void CongestionMap::remove_ghost_load(
    osmium::object_id_type way_id, int t_enter, int t_exit, int weight
) {
    update_load(way_id, t_enter, t_exit, -weight);
}

int CongestionMap::get_load(osmium::object_id_type way_id, int t) const {
    auto it = load_.find(way_id);
    if (it == load_.end()) return 0;
    auto jt = it->second.find(t);
    return jt != it->second.end() ? jt->second : 0;
}

float CongestionMap::adjusted_cost(
    osmium::object_id_type way_id,
    float base_cost,
    float distance_meters,
    int t,
    int self_weight
) const {
    int load_i = get_load(way_id, t) - self_weight;   // exclude own contribution
    if (load_i <= 0) return base_cost;            // free flow, common case — skip pow()
    const float load  = static_cast<float>(load_i);
    const float cap   = edge_capacity(distance_meters);
    const float ratio = load / cap;
    // Hot-path specialisation: β=4 (the configured default) → r^4 = (r²)².
    // std::pow is ~20× slower than two multiplications and dominates A*/Dijkstra
    // expansion cost when many edges carry load.
    float power;
    if (params.bpr_beta == 4.0f) {
        const float r2 = ratio * ratio;
        power = r2 * r2;
    } else if (params.bpr_beta == 2.0f) {
        power = ratio * ratio;
    } else {
        power = std::pow(ratio, params.bpr_beta);
    }
    return base_cost * (1.0f + params.bpr_alpha * power);
}

int CongestionMap::traversal_steps(
    osmium::object_id_type way_id,
    float distance_meters,
    int   t_enter,
    float speed_mps,
    int   self_weight
) const {
    if (distance_meters <= 0.f) return 1;
    const float spd = (speed_mps > 0.f) ? speed_mps : 1.f;
    const float eff = adjusted_cost(way_id, distance_meters, distance_meters,
                                    t_enter, self_weight);
    return std::max(1, static_cast<int>(std::ceil(eff / spd)));
}

void CongestionMap::advance(int t_now) {
    if (t_now <= t_now_) return;

    for (auto it = load_.begin(); it != load_.end(); ) {
        auto& steps = it->second;
        for (auto jt = steps.begin(); jt != steps.end(); )
            jt = (jt->first < t_now) ? steps.erase(jt) : std::next(jt);
        it = steps.empty() ? load_.erase(it) : std::next(it);
    }

    t_now_ = t_now;
}

void CongestionMap::reset() {
    load_.clear();
    t_now_ = 0;
}

float CongestionMap::mean_load_now() const {
    if (load_.empty()) return 0.f;
    int   sum_load = 0;
    int   n_edges  = 0;
    for (const auto& [way_id, steps] : load_) {
        auto jt = steps.find(t_now_);
        if (jt != steps.end()) { sum_load += jt->second; ++n_edges; }
    }
    return (n_edges > 0)
        ? static_cast<float>(sum_load) / static_cast<float>(n_edges)
        : 0.f;
}

int CongestionMap::peak_load_now() const {
    int peak = 0;
    for (const auto& [way_id, steps] : load_) {
        auto jt = steps.find(t_now_);
        if (jt != steps.end() && jt->second > peak) peak = jt->second;
    }
    return peak;
}

CongestionMap::LoadSample CongestionMap::load_sample_now() const {
    LoadSample s{0.f, 0};
    if (load_.empty()) return s;
    int sum_load = 0, n_edges = 0, peak = 0;
    for (const auto& [way_id, steps] : load_) {
        auto jt = steps.find(t_now_);
        if (jt != steps.end()) {
            sum_load += jt->second;
            ++n_edges;
            if (jt->second > peak) peak = jt->second;
        }
    }
    if (n_edges > 0)
        s.mean = static_cast<float>(sum_load) / static_cast<float>(n_edges);
    s.peak = peak;
    return s;
}

int CongestionMap::n_edges_load_ge(int threshold) const {
    int n = 0;
    for (const auto& [way_id, steps] : load_) {
        auto jt = steps.find(t_now_);
        if (jt != steps.end() && jt->second >= threshold) ++n;
    }
    return n;
}
