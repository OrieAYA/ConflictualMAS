#include "Environment/Simulation/GhostTrafficController.hpp"
#include <algorithm>
#include <cmath>

const char* congestion_profile_label(CongestionProfile p) {
    switch (p) {
        case CongestionProfile::Flat:       return "flat";
        case CongestionProfile::RampUpDown: return "ramp_up_down";
        case CongestionProfile::ShockBurst: return "shock_burst";
        case CongestionProfile::BuildingUp: return "building_up";
        case CongestionProfile::Wave:       return "wave";
    }
    return "unknown";
}

void GhostTrafficController::reset(
    const GeoBox& geo_box, CongestionMap& cmap, Config cfg, uint32_t seed)
{
    purge();
    cfg_  = cfg;
    cmap_ = &cmap;
    rng_.seed(seed);
    hot_ways_.clear();

    // Sample hot ways: a random subset of edges; the rest stay congestion-free
    // so real agents always have low-cost routes (feasibility).
    const auto& ways = geo_box.data.ways;
    if (ways.empty() || cfg_.hot_way_fraction <= 0.f) return;

    std::vector<osmium::object_id_type> all_ways;
    all_ways.reserve(ways.size());
    for (const auto& [wid, _] : ways) all_ways.push_back(wid);

    size_t n_hot;
    if (cfg_.hot_way_count > 0) {
        n_hot = std::min<size_t>(static_cast<size_t>(cfg_.hot_way_count), all_ways.size());
    } else {
        const float frac = std::clamp(cfg_.hot_way_fraction, 0.001f, 0.60f);
        n_hot = std::max<size_t>(8, static_cast<size_t>(all_ways.size() * frac));
    }
    // density_per_hot_way > 0 → derive n_max so per-hot-way density is uniform across cities.
    if (cfg_.density_per_hot_way > 0.f)
        cfg_.n_max = std::max(1, static_cast<int>(
            std::ceil(cfg_.density_per_hot_way * static_cast<float>(n_hot))));

    const size_t take = std::min(n_hot, all_ways.size());
    for (size_t i = 0; i < take; ++i) {
        std::uniform_int_distribution<size_t> u(i, all_ways.size() - 1);
        std::swap(all_ways[i], all_ways[u(rng_)]);
    }
    hot_ways_.assign(all_ways.begin(), all_ways.begin() + take);

    generate_events();
}

// Pre-generate the full ghost-injection sequence: per step, expire finished
// ghosts then top up to target_count, recording each spawn as an event. Uses
// the same RNG progression as a per-step procedural spawn would, so the
// resulting congestion is identical — only it is now an explicit sequence.
void GhostTrafficController::generate_events() {
    events_.clear();
    mean_active_ = 0.f;
    if (hot_ways_.empty() || cfg_.n_max <= 0 || cfg_.total_steps <= 0) return;

    const int w      = std::max(1, cfg_.load_per_ghost);
    const int window = std::max(1, cfg_.window_steps);
    std::vector<int> live;          // until-times of currently-active ghosts
    long long active_sum = 0;

    for (int step = 0; step < cfg_.total_steps; ++step) {
        live.erase(std::remove_if(live.begin(), live.end(),
                   [&](int ex){ return ex < step; }), live.end());
        const int target_entries = (target_count(step) + w - 1) / w;
        std::uniform_int_distribution<size_t> pick(0, hot_ways_.size() - 1);
        while (static_cast<int>(live.size()) < target_entries) {
            const int until = step + window;
            events_.push_back({ step, hot_ways_[pick(rng_)], until, w });
            live.push_back(until);
        }
        active_sum += static_cast<long long>(live.size());
    }
    mean_active_ = static_cast<float>(active_sum) / static_cast<float>(cfg_.total_steps);
}

void GhostTrafficController::step(int current_step) {
    while (event_idx_ < events_.size() && events_[event_idx_].step <= current_step) {
        const CongestionEvent& e = events_[event_idx_++];
        if (cmap_) cmap_->add_ghost_load(e.edge, e.step, e.until, e.load);
        active_until_.push_back(e.until);
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

int GhostTrafficController::target_count(int step) const {
    if (cfg_.total_steps <= 0) return 0;
    const float t = std::clamp(
        static_cast<float>(step) / static_cast<float>(cfg_.total_steps), 0.f, 1.f);
    const float n_max = static_cast<float>(cfg_.n_max);

    float f = 0.f;
    switch (cfg_.profile) {
        case CongestionProfile::Flat: f = 0.10f; break;
        case CongestionProfile::RampUpDown:
            if      (t < 0.25f) f = (t / 0.25f) * 0.30f;
            else if (t < 0.50f) f = 0.30f - ((t - 0.25f) / 0.25f) * 0.20f;
            else if (t < 0.75f) f = 0.10f + ((t - 0.50f) / 0.25f) * 0.40f;
            else                f = 0.50f + ((t - 0.75f) / 0.25f) * 0.50f;
            break;
        case CongestionProfile::ShockBurst:
            f = (t < 0.60f) ? 0.05f : (t < 0.80f ? 1.00f : 0.30f);
            break;
        case CongestionProfile::BuildingUp: f = t; break;
        case CongestionProfile::Wave:       f = std::sin(t * 3.14159265f); break;
    }
    return static_cast<int>(std::max(0.f, f) * n_max);
}
