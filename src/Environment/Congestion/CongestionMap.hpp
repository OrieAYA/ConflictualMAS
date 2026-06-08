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
    // Load amplification per REAL agent's path edge. Symmetric to
    // GhostTrafficController::Config::load_per_ghost: setting K>1 lets each
    // real agent register K load units per edge, so a fleet of N agents
    // produces the congestion footprint of N×K agents. Use to shrink the fleet
    // (faster A*/Dijkstra, fewer policy decisions) while preserving the
    // congestion intensity of a larger fleet.
    int   load_per_agent     = 1;
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
    // `weight` lets a single registration count for >1 unit of load — used to
    // simulate K real agents with one bookkeeping entry (K× fewer hash ops at
    // the price of quantised load resolution; BPR is polynomial degree β so
    // small K has negligible numerical impact).
    void add_agent   (osmium::object_id_type way_id, int t_enter, int t_exit, int weight = 1);
    void remove_agent(osmium::object_id_type way_id, int t_enter, int t_exit, int weight = 1);

    // Ghost-load API — for synthetic background traffic injected by
    // GhostTrafficController (Option M). Conceptually identical to add_agent /
    // remove_agent, but kept under a separate name so future code can audit
    // real-agent vs ghost-agent contributions if needed.
    void add_ghost_load   (osmium::object_id_type way_id, int t_enter, int t_exit, int weight = 1);
    void remove_ghost_load(osmium::object_id_type way_id, int t_enter, int t_exit, int weight = 1);

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

    // Maximum load on any single edge at the current step.
    // Useful to detect peak congestion (vs the mean).
    int peak_load_now() const;

    // Combined mean + peak in a SINGLE pass over load_ — avoids the 2× cost
    // of calling mean_load_now() and peak_load_now() back-to-back from per-
    // step instrumentation (3600 calls/episode × N_edges iteration each).
    struct LoadSample { float mean; int peak; };
    LoadSample load_sample_now() const;

    // Number of edges with load >= threshold at the current step.
    // n_edges_load_ge(2) counts overlap incidents (≥2 entities sharing an
    // edge in the same step). Used to quantify network conflict density.
    int n_edges_load_ge(int threshold) const;

private:
    std::unordered_map<osmium::object_id_type,
        std::unordered_map<int, int>> load_;   // way_id → {step_t → agent_count}
    int t_now_ = 0;

    void update_load(osmium::object_id_type way_id, int t_start, int t_end, int delta);
};

#endif // CONGESTION_MAP_HPP
