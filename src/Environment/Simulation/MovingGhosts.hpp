#ifndef MOVING_GHOSTS_HPP
#define MOVING_GHOSTS_HPP

#include "Environment/GeoBox/Box.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "Environment/Structure/EventStream.hpp"
#include <random>
#include <unordered_map>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// MovingGhostTraffic — exogenous vehicles that MOVE like delivery agents.
//
// Each ghost spawns at a node (density ∝ temporal profile), walks the road
// graph edge by edge at free-flow speed, then leaves the system. The whole
// trajectory set is a pure function of the seed (deterministic replay), but —
// unlike delivery agents whose plans are committed ahead — a ghost transit is
// only observable AT THE MOMENT IT HAPPENS:
//   reveal_to_map = true  : step() pushes each transit into the CongestionMap
//                           when its entry step is reached (online in time,
//                           global in space — the manager never sees a ghost's
//                           future).
//   reveal_to_map = false : nothing is written to the map; the truth lives
//                           only here (truth_load / observe_incident) and the
//                           manager must be fed by what delivery agents SEE.
//
// ── NOT WIRED YET — wiring plan ──────────────────────────────────────────────
// 1. Revealed mode (drop-in for GhostTrafficController):
//    EpisodeRunner::setup_ghost_traffic → when cfg.ghost_moving, reset() this
//    controller with the SharedEpisodeSetup ghost seed (volume parity:
//    n_ghosts ≈ n_events · window_steps / mean_route_duration); main loop →
//    replace ghost_traffic_.step(step) by step(step, memory_.congestion_map).
//    Planning/TD-A* then only see ghost load already revealed.
// 2. Online mode (post-evaluation switch): reveal_to_map = false; movement
//    physics (schedule_next_edge) adds truth_load(edge, step) on top of the
//    manager map so traversals feel the real traffic; on every
//    arrive_at_node, the runner calls observe_incident(current_node) and
//    ingests each observation into the manager map as a short-TTL ghost load
//    (add_ghost_load(edge, t, t+ttl, load)) — the manager's belief becomes
//    exactly what the fleet has seen. LSM input g[4] switches to the count
//    of freshly observed edges.
// ════════════════════════════════════════════════════════════════════════════

struct GhostTransit {
    int                    ghost_id;
    osmium::object_id_type edge;
    int                    t_entry;
    int                    t_exit;
    int                    load;
};

class MovingGhostTraffic {
public:
    struct Config {
        int   n_ghosts    = 0;      // fleet size (resolved by the caller)
        int   total_steps = 3600;
        float speed_mps   = 5.f;    // free-flow speed (m/step)
        int   route_edges_min = 8;
        int   route_edges_max = 40;
        float hot_spawn_bias  = 0.6f;   // P(spawn on a hot-way endpoint)
        float hot_way_fraction = event_tuning::kHotWayFraction;
        TemporalProfile profile = TemporalProfile::Flat;
    };

    struct Observation {
        osmium::object_id_type edge;
        int                    load;
    };

    MovingGhostTraffic() = default;

    void reset(const GeoBox& geo_box, Config cfg, uint32_t seed);

    // Reveal every transit whose entry step is reached (interval
    // [t_entry, t_exit] on the edge's temporal chain) when reveal_to_map.
    void step(int current_step, CongestionMap& cmap);

    // Ground truth: total ghost load physically on `edge` at time t.
    int truth_load(osmium::object_id_type edge, int t) const;

    // What an entity standing at `node` sees at time t: ghost load on each
    // incident edge (future online mode: delivery agents report these).
    void observe_incident(const MyData& data, osmium::object_id_type node,
                          int t, std::vector<Observation>& out) const;

    void purge();

    bool reveal_to_map = true;

    const std::vector<GhostTransit>& transits() const { return transits_; }
    int   n_ghosts()     const { return n_ghosts_; }
    int   n_transits()   const { return static_cast<int>(transits_.size()); }
    int   n_active_now(int step) const;
    float mean_active()  const { return mean_active_; }
    int   n_hot_ways()   const { return static_cast<int>(hot_ways_.size()); }

private:
    Config       cfg_;
    std::mt19937 rng_;

    std::vector<osmium::object_id_type> hot_ways_;
    std::vector<GhostTransit>           transits_;      // sorted by t_entry
    std::unordered_map<osmium::object_id_type, std::vector<int>> by_edge_;
    std::vector<int>                    active_per_step_;
    std::size_t                         next_reveal_ = 0;
    int                                 n_ghosts_    = 0;
    float                               mean_active_ = 0.f;
};

#endif // MOVING_GHOSTS_HPP
