#ifndef GHOST_TRAFFIC_CONTROLLER_HPP
#define GHOST_TRAFFIC_CONTROLLER_HPP

#include "../GeoBox/Box.hpp"
#include "CongestionMap.hpp"
#include <random>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// CongestionProfile — temporal shape of injected background traffic
// ════════════════════════════════════════════════════════════════════════════
//
// Controls target_count(t) = N_max * f_profile(t / total_steps) where
// t ∈ [0, total_steps]. The controller calls target_count() every step and
// adds/removes ghost windows on CongestionMap to track the curve.
//
//   Flat        — constant low (10% of N_max). Baseline / sanity scenario.
//   RampUpDown  — 4-quarter pattern:
//                   0.00-0.25 :  0% → 30%        (slow rise)
//                   0.25-0.50 : 30% → 10%        (relax)
//                   0.50-0.75 : 10% → 50%        (rebuild)
//                   0.75-1.00 : 50% → 100%       (peak)
//                 Matches the user's "début peu, augmente, redescend, monte
//                 beaucoup" description.
//   ShockBurst  — 60% calm @ 5%, then 20% peak @ 100%, then 20% tail @ 30%.
//                 Models incident / event-driven congestion shocks.
//   BuildingUp  — linear 0% → 100% across the episode. Continuous growth.
//   Wave        — sine bell sin(π·t) → smooth rise/peak/fall.
enum class CongestionProfile {
    Flat,
    RampUpDown,
    ShockBurst,
    BuildingUp,
    Wave
};

const char* congestion_profile_label(CongestionProfile p);

// ════════════════════════════════════════════════════════════════════════════
// GhostTrafficController
// ════════════════════════════════════════════════════════════════════════════
//
// Injects synthetic "background" agents into a CongestionMap to create
// spatial/temporal congestion heterogeneity that real delivery agents must
// navigate around. Has no concept of tasks, planning, or rewards — it only
// registers (way_id, t_enter, t_exit) windows.
//
// Spatial heterogeneity:
//   At reset(), the controller samples a subset of way_ids ("hot ways") from
//   the GeoBox. All ghost loads are placed on hot ways, leaving the remaining
//   ways (~70% of the graph by default) congestion-free so episodes stay
//   feasible — real agents can route around hot zones.
//
// Temporal heterogeneity:
//   At each step(), target_count is recomputed from the active profile. New
//   ghosts are spawned to meet target; old ghosts expire naturally as their
//   t_exit passes the current step. Each ghost occupies a way for
//   window_steps steps (= "slow-moving" external vehicle).
//
// Reproducibility:
//   The internal RNG is seeded per-episode in reset(seed); ghost placement is
//   fully reproducible given the same seed and config.
//
// Lifecycle:
//   reset()  at episode start.
//   step(t)  once per simulation step (after CongestionMap::advance).
//   purge()  at episode end to unregister any still-active ghosts.
class GhostTrafficController {
public:
    struct Config {
        int   n_max          = 40;      // peak simultaneous ghost loads
        int   total_steps    = 3600;    // episode length
        int   window_steps   = 5;       // ghost occupies a way for this many steps
        float hot_way_fraction = 0.30f; // fraction of geo_box ways used as hot pool
        // When > 0, overrides hot_way_fraction with an absolute hot-way count.
        // Used by Option Y strong-congestion eval to force ghost concentration
        // onto a small set of arteries so collisions create real BPR jams.
        int   hot_way_count    = 0;
        // When > 0, n_max is derived at reset time as ceil(density × n_hot)
        // so the average ghost density per hot way is identical across cities.
        // Set to 0 to keep the n_max value as-is (legacy behaviour).
        float density_per_hot_way = 0.0f;
        // Load amplification per ghost. With load_per_ghost=K, the controller
        // spawns ceil(target_load / K) ghost ENTRIES that each contribute K
        // units of load to the CongestionMap. Same BPR profile, K× fewer hash
        // ops per spawn/expire cycle — the dominant cost driver at high n_max.
        // Quantises load resolution by K; BPR being polynomial degree β, the
        // numerical impact stays <1% up to K=5 in typical configs.
        int   load_per_ghost      = 1;
        CongestionProfile profile = CongestionProfile::Flat;
    };

    GhostTrafficController() = default;

    // Set up for a new episode: sample hot ways from geo_box, store the
    // CongestionMap pointer, seed the RNG. Idempotent — call once per episode.
    void reset(const GeoBox& geo_box, CongestionMap& cmap,
               Config cfg, uint32_t seed);

    // Step the controller: expire old ghosts, then top up to target_count
    // by spawning new ghosts on random hot ways with windows [t, t + window].
    // Cheap: O(n_max) per call typically.
    void step(int current_step);

    // Unregister any active ghosts. Idempotent.
    void purge();

    // ── Metrics ───────────────────────────────────────────────────────────
    int   n_active_now() const { return static_cast<int>(active_.size()); }
    float mean_active() const;
    int   n_hot_ways() const { return static_cast<int>(hot_ways_.size()); }
    int   n_max() const { return cfg_.n_max; }   // derived after reset()
    CongestionProfile profile() const { return cfg_.profile; }

private:
    Config         cfg_;
    CongestionMap* cmap_ = nullptr;
    std::mt19937   rng_;

    std::vector<osmium::object_id_type> hot_ways_;

    struct GhostLoad {
        osmium::object_id_type way_id;
        int t_enter;
        int t_exit;
    };
    std::vector<GhostLoad> active_;

    long long active_sum_   = 0;
    int       active_steps_ = 0;

    int target_count(int step) const;
};

#endif // GHOST_TRAFFIC_CONTROLLER_HPP
