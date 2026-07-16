#include "Environment/Simulation/MovingGhosts.hpp"
#include <algorithm>
#include <cmath>

void MovingGhostTraffic::reset(const GeoBox& geo_box, Config cfg,
                               uint32_t seed) {
    purge();
    cfg_ = cfg;
    rng_.seed(seed);

    const auto& ways  = geo_box.data.ways;
    const auto& nodes = geo_box.data.nodes;
    if (ways.empty() || nodes.empty()
        || cfg_.n_ghosts <= 0 || cfg_.total_steps <= 0)
        return;

    // Hot-way pool: same φh partial shuffle as GhostTrafficController.
    std::vector<osmium::object_id_type> all_ways;
    all_ways.reserve(ways.size());
    for (const auto& [wid, _] : ways) all_ways.push_back(wid);
    const float frac = std::clamp(cfg_.hot_way_fraction, 0.001f, 1.0f);
    const size_t take = std::min(
        std::max<size_t>(8, static_cast<size_t>(all_ways.size() * frac)),
        all_ways.size());
    for (size_t i = 0; i < take; ++i) {
        std::uniform_int_distribution<size_t> u(i, all_ways.size() - 1);
        std::swap(all_ways[i], all_ways[u(rng_)]);
    }
    hot_ways_.assign(all_ways.begin(), all_ways.begin() + take);

    std::vector<osmium::object_id_type> road_nodes;
    road_nodes.reserve(nodes.size());
    for (const auto& [nid, pt] : nodes)
        if (!pt.incident_ways.empty()) road_nodes.push_back(nid);
    if (road_nodes.empty()) return;

    // Spawn-step density ∝ profile(t/T) (same inverse-CDF as the task stream).
    const int T = cfg_.total_steps;
    std::vector<float> w(static_cast<size_t>(T), 0.f);
    double wsum = 0.0;
    for (int s = 0; s < T; ++s) {
        const float f = std::max(0.f, temporal_profile_value(
            cfg_.profile, (static_cast<float>(s) + 0.5f) / T));
        w[s] = f; wsum += f;
    }
    if (wsum <= 0.0) std::fill(w.begin(), w.end(), 1.f);

    std::discrete_distribution<int>       step_dist(w.begin(), w.end());
    std::uniform_int_distribution<size_t> pick_node(0, road_nodes.size() - 1);
    std::uniform_int_distribution<size_t> pick_hot(0, hot_ways_.size() - 1);
    std::uniform_int_distribution<int>    load_draw(event_tuning::kGhostLoadMin,
                                                    event_tuning::kGhostLoadMax);
    std::uniform_int_distribution<int>    len_draw(
        std::max(1, cfg_.route_edges_min),
        std::max(cfg_.route_edges_min, cfg_.route_edges_max));
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    const float speed = std::max(0.1f, cfg_.speed_mps);
    std::vector<int> d_active(static_cast<size_t>(T) + 1, 0);
    long long active_span = 0;

    for (int gid = 0; gid < cfg_.n_ghosts; ++gid) {
        osmium::object_id_type cur = 0;
        if (u01(rng_) < cfg_.hot_spawn_bias && !hot_ways_.empty()) {
            const auto& hw = ways.at(hot_ways_[pick_hot(rng_)]);
            cur = (u01(rng_) < 0.5f) ? hw.node1_id : hw.node2_id;
        }
        auto nit = nodes.find(cur);
        if (nit == nodes.end() || nit->second.incident_ways.empty()) {
            cur = road_nodes[pick_node(rng_)];
            nit = nodes.find(cur);
        }

        const int load    = load_draw(rng_);
        const int n_edges = len_draw(rng_);
        int t = step_dist(rng_);
        const int t_spawn = t;

        osmium::object_id_type prev_edge = 0;
        for (int k = 0; k < n_edges && t < T; ++k) {
            const auto& inc = nit->second.incident_ways;
            if (inc.empty()) break;

            osmium::object_id_type wid = 0;
            std::uniform_int_distribution<size_t> pick_edge(0, inc.size() - 1);
            for (int tries = 0; tries < 4; ++tries) {
                wid = inc[pick_edge(rng_)];
                if (wid != prev_edge || inc.size() == 1) break;
            }
            auto wit = ways.find(wid);
            if (wit == ways.end()) break;

            const float dist = std::max(1.f, wit->second.distance_meters);
            const int   dur  = std::max(1, static_cast<int>(
                                   std::ceil(dist / speed)));
            const int t_exit = std::min(T, t + dur);
            transits_.push_back({ gid, wid, t, t_exit, load });

            t         = t + dur;
            prev_edge = wid;
            const osmium::object_id_type next =
                (wit->second.node1_id == cur) ? wit->second.node2_id
                                              : wit->second.node1_id;
            nit = nodes.find(next);
            if (nit == nodes.end()) break;
            cur = next;
        }

        const int t_end = std::min(T, t);
        if (t_end > t_spawn) {
            d_active[t_spawn] += 1;
            d_active[t_end]   -= 1;
            active_span       += t_end - t_spawn;
            ++n_ghosts_;
        }
    }

    std::sort(transits_.begin(), transits_.end(),
              [](const GhostTransit& a, const GhostTransit& b) {
                  return a.t_entry < b.t_entry;
              });
    for (int i = 0; i < static_cast<int>(transits_.size()); ++i)
        by_edge_[transits_[i].edge].push_back(i);

    active_per_step_.assign(static_cast<size_t>(T), 0);
    int running = 0;
    for (int s = 0; s < T; ++s) {
        running += d_active[s];
        active_per_step_[s] = running;
    }
    mean_active_ = static_cast<float>(active_span) / static_cast<float>(T);
}

void MovingGhostTraffic::step(int current_step, CongestionMap& cmap) {
    while (next_reveal_ < transits_.size()
           && transits_[next_reveal_].t_entry <= current_step) {
        const GhostTransit& tr = transits_[next_reveal_];
        if (reveal_to_map)
            cmap.add_ghost_load(tr.edge, tr.t_entry, tr.t_exit, tr.load);
        ++next_reveal_;
    }
}

int MovingGhostTraffic::truth_load(osmium::object_id_type edge, int t) const {
    auto it = by_edge_.find(edge);
    if (it == by_edge_.end()) return 0;
    int total = 0;
    for (int idx : it->second) {
        const GhostTransit& tr = transits_[idx];
        if (tr.t_entry > t) break;              // indices sorted by t_entry
        if (t < tr.t_exit) total += tr.load;
    }
    return total;
}

void MovingGhostTraffic::observe_incident(const MyData& data,
                                          osmium::object_id_type node, int t,
                                          std::vector<Observation>& out) const {
    auto nit = data.nodes.find(node);
    if (nit == data.nodes.end()) return;
    for (auto wid : nit->second.incident_ways) {
        const int l = truth_load(wid, t);
        if (l > 0) out.push_back({ wid, l });
    }
}

int MovingGhostTraffic::n_active_now(int step) const {
    if (step < 0 || step >= static_cast<int>(active_per_step_.size())) return 0;
    return active_per_step_[step];
}

void MovingGhostTraffic::purge() {
    hot_ways_.clear();
    transits_.clear();
    by_edge_.clear();
    active_per_step_.clear();
    next_reveal_ = 0;
    n_ghosts_    = 0;
    mean_active_ = 0.f;
}
