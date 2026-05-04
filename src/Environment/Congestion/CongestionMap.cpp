#include "CongestionMap.hpp"
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
    osmium::object_id_type way_id, int t_enter, int t_exit
) {
    update_load(way_id, t_enter, t_exit, +1);
}

void CongestionMap::remove_agent(
    osmium::object_id_type way_id, int t_enter, int t_exit
) {
    update_load(way_id, t_enter, t_exit, -1);
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
    int t
) const {
    const float load  = static_cast<float>(get_load(way_id, t));
    const float cap   = edge_capacity(distance_meters);
    const float ratio = load / cap;
    const float bpr   = 1.0f + params.bpr_alpha * std::pow(ratio, params.bpr_beta);
    return base_cost * bpr;
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
