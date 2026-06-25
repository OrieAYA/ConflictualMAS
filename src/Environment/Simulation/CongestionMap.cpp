#include "Environment/Simulation/CongestionMap.hpp"
#include <algorithm>

CongestionMap::CongestionMap(const CongestionParams& p) : params(p) {}

int CongestionMap::current_step() const { return t_now_; }

float CongestionMap::edge_capacity(float distance_meters) const {
    return std::max(1.0f, distance_meters * params.capacity_per_meter);
}

const TemporalChainList* CongestionMap::chain(osmium::object_id_type way_id) const {
    auto it = chains_.find(way_id);
    return it == chains_.end() ? nullptr : &it->second;
}

TemporalChainList& CongestionMap::chain_mut(osmium::object_id_type way_id) {
    return chains_.try_emplace(way_id).first->second;
}

// ── Occupancy registration (intervals; no clamping → add/remove symmetric) ──

void CongestionMap::add_agent(
    osmium::object_id_type way_id, int t_enter, int t_exit, int weight, int agent_id
) {
    if (weight <= 0 || t_exit < t_enter || t_exit < t_now_) return;
    chain_mut(way_id).insert(static_cast<float>(t_exit),
                             static_cast<float>(t_exit - t_enter), weight, agent_id);
}

void CongestionMap::remove_agent(
    osmium::object_id_type way_id, int t_enter, int t_exit, int weight
) {
    auto it = chains_.find(way_id);
    if (it == chains_.end()) return;
    it->second.remove_interval(static_cast<float>(t_exit),
                               static_cast<float>(t_exit - t_enter), weight);
    if (it->second.size() == 0) chains_.erase(it);
}

void CongestionMap::add_ghost_load(
    osmium::object_id_type way_id, int t_enter, int t_exit, int weight
) { add_agent(way_id, t_enter, t_exit, weight); }

void CongestionMap::remove_ghost_load(
    osmium::object_id_type way_id, int t_enter, int t_exit, int weight
) { remove_agent(way_id, t_enter, t_exit, weight); }

int CongestionMap::get_load(osmium::object_id_type way_id, int t) const {
    const TemporalChainList* c = chain(way_id);
    return c ? c->load_at(static_cast<float>(t)) : 0;
}

// ── BPR cost ────────────────────────────────────────────────────────────────

float CongestionMap::adjusted_cost(
    osmium::object_id_type way_id,
    float base_cost,
    float distance_meters,
    int t,
    int self_weight
) const {
    int load_i = get_load(way_id, t) - self_weight;   // exclude own contribution
    if (load_i <= 0) return base_cost;                 // free flow, common case
    const float load  = static_cast<float>(load_i);
    const float cap   = edge_capacity(distance_meters);
    const float ratio = load / cap;
    // Hot-path specialisation: β=4 (the configured default) → r^4 = (r²)².
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

// ── Time advance / reset ─────────────────────────────────────────────────────

void CongestionMap::advance(int t_now) {
    if (t_now <= t_now_) return;
    const float tf = static_cast<float>(t_now);
    for (auto it = chains_.begin(); it != chains_.end(); ) {
        it->second.purge_expired(tf);
        it = (it->second.size() == 0) ? chains_.erase(it) : std::next(it);
    }
    t_now_ = t_now;
}

void CongestionMap::reset() {
    chains_.clear();
    t_now_ = 0;
}

// ── Current-step instrumentation ─────────────────────────────────────────────

float CongestionMap::mean_load_now() const {
    int sum_load = 0, n_edges = 0;
    for (const auto& [way_id, c] : chains_) {
        (void)way_id;
        const int l = c.load_at(static_cast<float>(t_now_));
        if (l > 0) { sum_load += l; ++n_edges; }
    }
    return n_edges > 0 ? static_cast<float>(sum_load) / static_cast<float>(n_edges) : 0.f;
}

int CongestionMap::peak_load_now() const {
    int peak = 0;
    for (const auto& [way_id, c] : chains_) {
        (void)way_id;
        peak = std::max(peak, c.load_at(static_cast<float>(t_now_)));
    }
    return peak;
}

CongestionMap::LoadSample CongestionMap::load_sample_now() const {
    LoadSample s{0.f, 0};
    int sum_load = 0, n_edges = 0, peak = 0;
    for (const auto& [way_id, c] : chains_) {
        (void)way_id;
        const int l = c.load_at(static_cast<float>(t_now_));
        if (l > 0) { sum_load += l; ++n_edges; if (l > peak) peak = l; }
    }
    if (n_edges > 0) s.mean = static_cast<float>(sum_load) / static_cast<float>(n_edges);
    s.peak = peak;
    return s;
}

int CongestionMap::n_edges_load_ge(int threshold) const {
    int n = 0;
    for (const auto& [way_id, c] : chains_) {
        (void)way_id;
        if (c.load_at(static_cast<float>(t_now_)) >= threshold) ++n;
    }
    return n;
}
