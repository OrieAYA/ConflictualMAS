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

// ── Lifecycle ───────────────────────────────────────────────────────────────

void GhostTrafficController::reset(
    const GeoBox& geo_box, CongestionMap& cmap,
    Config cfg, uint32_t seed)
{
    purge();   // drop any leftover from a previous episode

    cfg_   = cfg;
    cmap_  = &cmap;
    rng_.seed(seed);

    active_sum_   = 0;
    active_steps_ = 0;
    active_.clear();
    hot_ways_.clear();

    // Sample hot ways: take `hot_way_fraction` of all ways at random. The rest
    // stay congestion-free so real agents have low-cost routes available
    // (feasibility guarantee).
    const auto& ways = geo_box.data.ways;
    if (ways.empty() || cfg_.hot_way_fraction <= 0.f) return;

    std::vector<osmium::object_id_type> all_ways;
    all_ways.reserve(ways.size());
    for (const auto& [wid, _] : ways) all_ways.push_back(wid);

    // Hot-way pool size: absolute count takes priority when > 0 (used by
    // strong-congestion eval to force ghost concentration), otherwise the
    // fractional-of-ways heuristic. Floor lifted to 0.001 so per-city
    // proportional configs (e.g. 10% of ways per city for uniform agent
    // exposure across scales) are honoured rather than clamped to 5%.
    size_t n_hot;
    if (cfg_.hot_way_count > 0) {
        n_hot = std::min<size_t>(static_cast<size_t>(cfg_.hot_way_count),
                                 all_ways.size());
    } else {
        const float frac = std::clamp(cfg_.hot_way_fraction, 0.001f, 0.60f);
        n_hot = std::max<size_t>(8, static_cast<size_t>(all_ways.size() * frac));
    }

    // Density override: when density_per_hot_way > 0, derive n_max from the
    // chosen hot pool size so the AVERAGE ghosts-per-hot-way is identical
    // across cities. This is what gives uniform per-edge BPR profile
    // regardless of graph size; without it, the same n_max diluted over
    // a larger hot pool produces weaker per-edge congestion.
    if (cfg_.density_per_hot_way > 0.f) {
        cfg_.n_max = std::max(1, static_cast<int>(
            std::ceil(cfg_.density_per_hot_way * static_cast<float>(n_hot))));
    }

    // Partial shuffle: first n_hot picks are random distinct ways.
    const size_t take = std::min(n_hot, all_ways.size());
    for (size_t i = 0; i < take; ++i) {
        std::uniform_int_distribution<size_t> u(i, all_ways.size() - 1);
        size_t j = u(rng_);
        std::swap(all_ways[i], all_ways[j]);
    }
    hot_ways_.assign(all_ways.begin(), all_ways.begin() + take);
}

void GhostTrafficController::step(int current_step) {
    if (!cmap_ || hot_ways_.empty() || cfg_.n_max <= 0) return;

    const int w = std::max(1, cfg_.load_per_ghost);

    // 1. Expire ghosts whose t_exit < current_step.
    for (size_t i = 0; i < active_.size(); ) {
        if (active_[i].t_exit < current_step) {
            cmap_->remove_ghost_load(active_[i].way_id,
                                     active_[i].t_enter, active_[i].t_exit, w);
            active_[i] = active_.back();
            active_.pop_back();
        } else {
            ++i;
        }
    }

    // 2. Top up to target. With load_per_ghost=K, each ghost entry contributes
    //    K load units, so we need ceil(target_load / K) entries to reach the
    //    requested target load. Same congestion intensity for K× fewer entries.
    const int target_load    = target_count(current_step);
    const int target_entries = (target_load + w - 1) / w;
    std::uniform_int_distribution<size_t> pick(0, hot_ways_.size() - 1);
    while (static_cast<int>(active_.size()) < target_entries) {
        GhostLoad g;
        g.way_id  = hot_ways_[pick(rng_)];
        g.t_enter = current_step;
        g.t_exit  = current_step + std::max(1, cfg_.window_steps);
        cmap_->add_ghost_load(g.way_id, g.t_enter, g.t_exit, w);
        active_.push_back(g);
    }

    active_sum_   += static_cast<long long>(active_.size());
    active_steps_ += 1;
}

void GhostTrafficController::purge() {
    if (cmap_) {
        const int w = std::max(1, cfg_.load_per_ghost);
        for (const auto& g : active_)
            cmap_->remove_ghost_load(g.way_id, g.t_enter, g.t_exit, w);
    }
    active_.clear();
}

float GhostTrafficController::mean_active() const {
    return (active_steps_ > 0)
        ? static_cast<float>(active_sum_) / static_cast<float>(active_steps_)
        : 0.f;
}

// ── Profile shapes ──────────────────────────────────────────────────────────

int GhostTrafficController::target_count(int step) const {
    if (cfg_.total_steps <= 0) return 0;
    const float t = std::clamp(
        static_cast<float>(step) / static_cast<float>(cfg_.total_steps),
        0.f, 1.f);
    const float n_max = static_cast<float>(cfg_.n_max);

    float f = 0.f;
    switch (cfg_.profile) {
        case CongestionProfile::Flat:
            f = 0.10f;
            break;

        case CongestionProfile::RampUpDown: {
            // 4 quarter-phases: 0→30%, 30→10%, 10→50%, 50→100%.
            if (t < 0.25f)
                f = (t / 0.25f) * 0.30f;
            else if (t < 0.50f)
                f = 0.30f - ((t - 0.25f) / 0.25f) * 0.20f;
            else if (t < 0.75f)
                f = 0.10f + ((t - 0.50f) / 0.25f) * 0.40f;
            else
                f = 0.50f + ((t - 0.75f) / 0.25f) * 0.50f;
            break;
        }

        case CongestionProfile::ShockBurst: {
            if (t < 0.60f)      f = 0.05f;
            else if (t < 0.80f) f = 1.00f;
            else                f = 0.30f;
            break;
        }

        case CongestionProfile::BuildingUp:
            f = t;
            break;

        case CongestionProfile::Wave: {
            constexpr float kPi = 3.14159265f;
            f = std::sin(t * kPi);
            break;
        }
    }
    return static_cast<int>(std::max(0.f, f) * n_max);
}
