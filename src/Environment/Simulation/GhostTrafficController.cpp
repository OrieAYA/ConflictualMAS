#include "Environment/Simulation/GhostTrafficController.hpp"
#include <algorithm>
#include <cmath>

void GhostTrafficController::reset(
    const GeoBox& geo_box, CongestionMap& cmap, Config cfg, uint32_t seed)
{
    purge();
    cfg_ = cfg;
    rng_.seed(seed);
    hot_ways_.clear();

    // Sample the hot-way pool: a random φh-subset of edges; the rest stay
    // congestion-free so real agents always have low-cost routes (feasibility).
    const auto& ways = geo_box.data.ways;
    if (ways.empty() || (cfg_.hot_way_fraction <= 0.f && cfg_.hot_way_count <= 0))
        return;

    std::vector<osmium::object_id_type> all_ways;
    all_ways.reserve(ways.size());
    for (const auto& [wid, _] : ways) all_ways.push_back(wid);

    size_t n_hot;
    if (cfg_.hot_way_count > 0) {
        n_hot = std::min<size_t>(static_cast<size_t>(cfg_.hot_way_count), all_ways.size());
    } else {
        const float frac = std::clamp(cfg_.hot_way_fraction, 0.001f, 1.0f);
        n_hot = std::max<size_t>(8, static_cast<size_t>(all_ways.size() * frac));
    }
    const size_t take = std::min(n_hot, all_ways.size());
    for (size_t i = 0; i < take; ++i) {
        std::uniform_int_distribution<size_t> u(i, all_ways.size() - 1);
        std::swap(all_ways[i], all_ways[u(rng_)]);
    }
    hot_ways_.assign(all_ways.begin(), all_ways.begin() + take);

    generate_events(cfg_.profile);

    // Inject the WHOLE congestion stream at episode start: each event is one
    // interval on its edge's temporal chain (+load at `step`, −load at
    // `until`). Planning therefore sees the full announced congestion.
    for (const CongestionEvent& e : events_)
        cmap.add_ghost_load(e.edge, e.step, e.until, e.load);
}

// Pre-generate the full ghost sequence: n_events ghosts, each appearing at a
// step drawn with density ∝ profile(t/T) (inverse-CDF sampling, same as the
// task generator), occupying one random hot way over [step, step+window] with
// an integer load drawn in [kGhostLoadMin, kGhostLoadMax].
void GhostTrafficController::generate_events(TemporalProfile profile) {
    events_.clear();
    mean_active_ = 0.f;
    if (hot_ways_.empty() || cfg_.n_events <= 0 || cfg_.total_steps <= 0) return;

    const int T      = cfg_.total_steps;
    const int window = std::max(1, cfg_.window_steps);

    std::vector<float> w(static_cast<size_t>(T), 0.f);
    double wsum = 0.0;
    for (int s = 0; s < T; ++s) {
        const float f = std::max(0.f, temporal_profile_value(
            profile, (static_cast<float>(s) + 0.5f) / T));
        w[s] = f; wsum += f;
    }
    if (wsum <= 0.0) std::fill(w.begin(), w.end(), 1.f);

    std::discrete_distribution<int>       step_dist(w.begin(), w.end());
    std::uniform_int_distribution<size_t> pick(0, hot_ways_.size() - 1);
    std::uniform_int_distribution<int>    load_draw(event_tuning::kGhostLoadMin,
                                                    event_tuning::kGhostLoadMax);

    events_.reserve(static_cast<size_t>(cfg_.n_events));
    long long active_span = 0;
    for (int i = 0; i < cfg_.n_events; ++i) {
        const int step  = step_dist(rng_);
        const int until = std::min(T, step + window);
        const int load  = load_draw(rng_);
        events_.push_back({ step, hot_ways_[pick(rng_)], until, load });
        active_span += (until - step);
    }
    std::sort(events_.begin(), events_.end(),
              [](const CongestionEvent& a, const CongestionEvent& b){
                  return a.step < b.step;
              });
    mean_active_ = static_cast<float>(active_span) / static_cast<float>(T);
}

void GhostTrafficController::step(int current_step) {
    // Loads are already on the CongestionMap (injected at reset); this only
    // tracks how many ghosts are live for instrumentation (n_active_now).
    while (event_idx_ < events_.size() && events_[event_idx_].step <= current_step) {
        active_until_.push_back(events_[event_idx_].until);
        ++event_idx_;
    }
    active_until_.erase(std::remove_if(active_until_.begin(), active_until_.end(),
                        [&](int u){ return u < current_step; }), active_until_.end());
}

void GhostTrafficController::purge() {
    events_.clear();
    event_idx_ = 0;
    active_until_.clear();
    mean_active_ = 0.f;
}
