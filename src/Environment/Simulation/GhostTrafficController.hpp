#ifndef GHOST_TRAFFIC_CONTROLLER_HPP
#define GHOST_TRAFFIC_CONTROLLER_HPP

#include "../GeoBox/Box.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "Environment/Structure/EventStream.hpp"   // TemporalProfile + event_tuning
#include <random>
#include <vector>

// One congestion event of the episode stream: a ghost occupying `edge` with an
// integer load over [step, until]. The pair (appearance at `step`, removal at
// `until`) is encoded as one interval — exactly what the per-edge temporal
// chain (CongestionMap / TemporalChainList) stores.
struct CongestionEvent {
    int                    step;    // ghost appears (load added to the chain)
    osmium::object_id_type edge;
    int                    until;   // ghost expires (same load removed)
    int                    load;    // integer weight ∈ [kGhostLoadMin, kGhostLoadMax]
};

// Congestion-event generator. At episode start (reset) it samples the hot-way
// pool, pre-generates the FULL sequence of n_events ghost events for the episode
// — each appearance step drawn with density ∝ profile(t/T), the same time
// function the task generator takes — and injects every event into the
// CongestionMap right away (the event stream is complete from step 0; nothing
// is generated online). step() only maintains the live-ghost window for
// instrumentation.
class GhostTrafficController {
public:
    struct Config {
        // Total ghost EVENT count. Resolved by the caller — either an explicit
        // override (tests) or event_tuning::derived_ghost_count(SCE, RM).
        int   n_events     = 0;
        int   total_steps  = 3600;
        int   window_steps = 5;       // x: a ghost occupies its way for this long
        // φh — fraction of ways eligible as hot pool (event_tuning constant).
        float hot_way_fraction = event_tuning::kHotWayFraction;
        int   hot_way_count    = 0;   // > 0 overrides the fraction (absolute)
        TemporalProfile profile = TemporalProfile::Flat;
    };

    GhostTrafficController() = default;

    // Episode start: sample hot ways, pre-generate the full event sequence
    // from `profile`, and register EVERY event on the CongestionMap.
    void reset(const GeoBox& geo_box, CongestionMap& cmap,
               Config cfg, uint32_t seed);

    // Advance the live-ghost window (instrumentation only — the loads were
    // already injected at reset()).
    void step(int current_step);

    // Drop the event stream / live tracking (e.g. ghost-off episode).
    void purge();

    const std::vector<CongestionEvent>& events() const { return events_; }

    int   n_active_now() const { return static_cast<int>(active_until_.size()); }
    float mean_active()  const { return mean_active_; }
    int   n_hot_ways()   const { return static_cast<int>(hot_ways_.size()); }
    int   n_events()     const { return cfg_.n_events; }
    TemporalProfile profile() const { return cfg_.profile; }

private:
    Config         cfg_;
    std::mt19937   rng_;

    std::vector<osmium::object_id_type> hot_ways_;
    std::vector<CongestionEvent>        events_;       // full pre-generated stream
    std::size_t                         event_idx_ = 0;
    std::vector<int>                    active_until_; // until-times of live ghosts
    float                               mean_active_ = 0.f;

    void generate_events(TemporalProfile profile);   // fills events_ + mean_active_
};

#endif // GHOST_TRAFFIC_CONTROLLER_HPP
