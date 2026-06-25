#ifndef GHOST_TRAFFIC_CONTROLLER_HPP
#define GHOST_TRAFFIC_CONTROLLER_HPP

#include "../GeoBox/Box.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include <random>
#include <vector>

// Temporal shape of the injected background traffic: target_count(t) =
// n_max · f_profile(t / total_steps).
//   Flat — constant low · RampUpDown — rise/relax/rebuild/peak · ShockBurst —
//   calm then spike then tail · BuildingUp — linear 0→100% · Wave — sine bell.
enum class CongestionProfile { Flat, RampUpDown, ShockBurst, BuildingUp, Wave };

const char* congestion_profile_label(CongestionProfile p);

// One congestion-injection event: `load` ghost agents occupy `edge` over
// [step, until]. This is what a congestion profile returns.
struct CongestionEvent {
    int                    step;
    osmium::object_id_type edge;
    int                    until;
    int                    load;
};

// Congestion profile: from a seed it pre-generates (reset) the full sequence of
// ghost-injection events over the episode, then materialises them ONLINE in
// step(t) — an event is registered on the CongestionMap only once its time is
// reached, so agents cannot see future congestion. Ghosts live only on a sampled
// subset of "hot ways" so real agents can always route around them.
class GhostTrafficController {
public:
    struct Config {
        int   n_max          = 40;      // peak simultaneous ghost loads
        int   total_steps    = 3600;
        int   window_steps   = 5;       // a ghost occupies a way for this long
        float hot_way_fraction = 0.30f; // fraction of ways used as hot pool
        int   hot_way_count    = 0;     // > 0 overrides the fraction (absolute)
        float density_per_hot_way = 0.0f; // > 0 derives n_max = ceil(density·n_hot)
        int   load_per_ghost      = 1;  // load units per ghost entry
        CongestionProfile profile = CongestionProfile::Flat;
    };

    GhostTrafficController() = default;

    // Episode start: sample hot ways, pre-generate the event sequence.
    void reset(const GeoBox& geo_box, CongestionMap& cmap,
               Config cfg, uint32_t seed);

    // Materialise every event due at `t` onto the CongestionMap (online).
    void step(int current_step);

    // Stop materialising / drop the live tracking (e.g. ghost-off episode).
    void purge();

    const std::vector<CongestionEvent>& events() const { return events_; }

    int   n_active_now() const { return static_cast<int>(active_until_.size()); }
    float mean_active()  const { return mean_active_; }
    int   n_hot_ways()   const { return static_cast<int>(hot_ways_.size()); }
    int   n_max()        const { return cfg_.n_max; }
    CongestionProfile profile() const { return cfg_.profile; }

private:
    Config         cfg_;
    CongestionMap* cmap_ = nullptr;
    std::mt19937   rng_;

    std::vector<osmium::object_id_type> hot_ways_;
    std::vector<CongestionEvent>        events_;       // pre-generated sequence
    std::size_t                         event_idx_ = 0;
    std::vector<int>                    active_until_; // until-times of live ghosts
    float                               mean_active_ = 0.f;

    int  target_count(int step) const;
    void generate_events();   // fills events_ + mean_active_ from the profile
};

#endif // GHOST_TRAFFIC_CONTROLLER_HPP
