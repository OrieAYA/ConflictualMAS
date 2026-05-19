#ifndef CONGESTION_MAP_HPP
#define CONGESTION_MAP_HPP

#include "../GeoBox/Box.hpp"
#include <unordered_map>
#include <cmath>

struct CongestionParams {
    int   horizon            = 100;     // max future steps retained
    float bpr_alpha          = 0.15f;   // BPR function α
    float bpr_beta           = 4.0f;    // BPR function β
    float capacity_per_meter = 0.05f;   // agent slots per meter of edge length
};

// Sparse temporal edge-load map.
//
// Time is a discrete integer step t in {0, 1, ..., n}.
// For each edge e and step t, load_(e, t) counts the number of agents
// currently scheduled to occupy that edge at step t.
//
// Only steps in [t_now, t_now + horizon] are retained; older ones are
// purged on advance(). The structure is sparse: edges with zero load are
// not stored.
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
    // Steps outside [t_now, t_now + horizon] are silently ignored.
    void add_agent   (osmium::object_id_type way_id, int t_enter, int t_exit);
    void remove_agent(osmium::object_id_type way_id, int t_enter, int t_exit);

    int get_load(osmium::object_id_type way_id, int t) const;

    // BPR-adjusted cost: base_cost * (1 + α * (load/capacity)^β).
    float adjusted_cost(osmium::object_id_type way_id,
                        float base_cost,
                        float distance_meters,
                        int t) const;

    // Advance current step to t_now and purge all steps older than t_now.
    void advance(int t_now);

    // Hard reset: clear all load entries and set t_now back to 0.
    // Call at the start of each new episode so steps 0..N are accepted again.
    void reset();

    // Mean agent load at the current step across all tracked edges.
    // Returns 0 if no edge currently has any load. Used by EpisodeRunner to
    // populate GlobalState::congestion so the centralised critic sees a real
    // congestion signal (was a 0-padding placeholder before).
    float mean_load_now() const;

private:
    std::unordered_map<osmium::object_id_type,
        std::unordered_map<int, int>> load_;   // way_id → {step_t → agent_count}
    int t_now_ = 0;

    void update_load(osmium::object_id_type way_id, int t_start, int t_end, int delta);
};

#endif // CONGESTION_MAP_HPP
