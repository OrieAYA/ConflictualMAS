#ifndef CONGESTION_MAP_HPP
#define CONGESTION_MAP_HPP

#include "../GeoBox/Box.hpp"
#include "Environment/Simulation/TemporalChainList.hpp"
#include <unordered_map>
#include <cmath>

struct CongestionParams {
    // Predictive lookahead (steps). Planned movements are registered up to
    // t_now + horizon; BPR queries beyond it see free flow. Must cover the
    // typical plan length, otherwise time-aware planning degenerates to
    // static cost past the horizon.
    int   horizon            = 1800;
    float bpr_alpha          = 0.15f;   // BPR function α
    float bpr_beta           = 4.0f;    // BPR function β
    float capacity_per_meter = 0.05f;   // agent slots per meter of edge length
    // Load amplification per REAL agent's path edge. Symmetric to
    // GhostTrafficController::Config::load_per_ghost: setting K>1 lets each
    // real agent register K load units per edge, so a fleet of N agents
    // produces the congestion footprint of N×K agents. Use to shrink the fleet
    // (faster A*/Dijkstra, fewer policy decisions) while preserving the
    // congestion intensity of a larger fleet.
    int   load_per_agent     = 1;
};

// Time-dependent edge-load model.
//
// Each edge keeps a TemporalChainList of the occupancy intervals
// [t_enter, t_exit] currently registered on it; the load at a step t is the sum
// of the weights of the intervals covering t (TemporalChainList::load_at). This
// replaces the old per-step materialisation (one hash entry per (edge, step)):
// the in-window load values are identical, but add/remove are O(1 interval)
// instead of O(duration), and there is nothing to "fill in" between t_enter and
// t_exit.
//
// Cost model (BPR):
//   adjusted_cost = base_cost * (1 + α * (load / capacity)^β)
// where capacity = distance_meters * capacity_per_meter.
class CongestionMap {
public:
    CongestionParams params;

    explicit CongestionMap(const CongestionParams& p = {});

    int   current_step() const;
    float edge_capacity(float distance_meters) const;

    // Register / unregister an agent occupying edge `way_id` during [t_enter, t_exit].
    // `weight` lets a single registration count for >1 unit of load — used to
    // simulate K real agents with one entry. remove_* must be called with the
    // SAME (t_enter, t_exit, weight) the matching add_* used.
    void add_agent   (osmium::object_id_type way_id, int t_enter, int t_exit, int weight = 1, int agent_id = -1);
    void remove_agent(osmium::object_id_type way_id, int t_enter, int t_exit, int weight = 1);

    // Ghost-load API — synthetic background traffic (GhostTrafficController).
    // Identical mechanics to add_agent/remove_agent, separate name for auditing.
    void add_ghost_load   (osmium::object_id_type way_id, int t_enter, int t_exit, int weight = 1);
    void remove_ghost_load(osmium::object_id_type way_id, int t_enter, int t_exit, int weight = 1);

    int get_load(osmium::object_id_type way_id, int t) const;

    // BPR cost. self_weight is removed from the load first: an agent timing its
    // OWN traversal passes load_per_agent so it sees n others, not n+1.
    float adjusted_cost(osmium::object_id_type way_id,
                        float base_cost,
                        float distance_meters,
                        int   t,
                        int   self_weight = 0) const;

    // ceil(adjusted_cost / speed), >= 1. Shared by planning, commit and movement.
    int traversal_steps(osmium::object_id_type way_id,
                        float distance_meters,
                        int   t_enter,
                        float speed_mps,
                        int   self_weight = 0) const;

    // Currently registrable time window [window_lo, window_hi].
    int window_lo() const { return t_now_; }
    int window_hi() const { return t_now_ + params.horizon; }

    // Advance current step to t_now and purge every fully-past interval.
    void advance(int t_now);

    // Hard reset: drop all intervals and set t_now back to 0 (new episode).
    void reset();

    // Mean / peak / conflict-count of agent load at the current step.
    float mean_load_now() const;
    int   peak_load_now() const;

    struct LoadSample { float mean; int peak; };
    LoadSample load_sample_now() const;

    // Number of edges with load >= threshold at the current step (conflict density).
    int n_edges_load_ge(int threshold) const;

private:
    std::unordered_map<osmium::object_id_type, TemporalChainList> chains_;  // edge → occupancy intervals
    int t_now_ = 0;

    const TemporalChainList* chain(osmium::object_id_type way_id) const;
    TemporalChainList&       chain_mut(osmium::object_id_type way_id);
};

#endif // CONGESTION_MAP_HPP
