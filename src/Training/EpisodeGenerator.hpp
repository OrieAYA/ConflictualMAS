#ifndef TRAINING_EPISODE_GENERATOR_HPP
#define TRAINING_EPISODE_GENERATOR_HPP

#include "EpisodeConfig.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <osmium/osm/types.hpp>
#include <random>
#include <vector>

// ── One task in the episode timeline ─────────────────────────────────────────
struct ScheduledTask {
    int                    arrival_step;      // simulation step when task appears
    osmium::object_id_type pickup_node_id;    // raw OSM node (pickup)
    osmium::object_id_type delivery_node_id;  // raw OSM node (delivery)
    float                  reward;            // base reward (≈ pickup→delivery distance / kCostScale)
    float                  importance;        // urgency drawn from [0.5, 2.0]
    bool                   is_clustered;      // for metrics: was this task in a hot zone?
};

// ── Phase schedule with step boundaries ──────────────────────────────────────
struct PhaseInfo {
    int   step_begin;       // inclusive
    int   step_end;         // exclusive
    float lambda;           // Poisson rate (tasks/step); may be > 1.0
    int   n_agents_start;
    int   n_agents_end;
    float label;
    int   n_hot_zones;      // hot zones active for this phase

    // Linearly interpolate fleet size between start and end of phase.
    int n_agents_at(int step) const {
        if (n_agents_start == n_agents_end || step_end <= step_begin)
            return n_agents_start;
        float t = float(step - step_begin) / float(step_end - step_begin);
        t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
        return static_cast<int>(n_agents_start + t * (n_agents_end - n_agents_start) + 0.5f);
    }
};

// ── Episode Generator ─────────────────────────────────────────────────────────
//
// Generates a deterministic (or seeded) task stream for one episode on a
// pre-loaded GeoBox.  The generator has NO side-effects on GlobalMemory —
// it only produces ScheduledTask records.  The episode runner is responsible
// for converting them into PDPTask objects and registering them with the system.
//
// Spatial patterns:
//   Uniform  — pickup and delivery sampled independently from all valid road nodes.
//   Clustered — with probability cluster_prob, a node is sampled within one of
//               n_hot_zones randomly chosen centres (hot zones persist per episode,
//               matching real-world demand clustering: commercial districts, hubs).
//
// Arrival process:
//   Each step, the number of arriving tasks is drawn from Poisson(lambda).
//   Lambda is computed automatically as tasks_per_agent * avg_agents / steps,
//   which can exceed 1.0 under high density (Amazon-scale: 250 tasks/agent).
class EpisodeGenerator {
public:
    explicit EpisodeGenerator(const EpisodeConfig& cfg,
                              const GeoBox&        geo_box,
                              uint32_t             seed = 42);

    // Generate the full task stream for one episode (sorted by arrival_step).
    // Each call re-samples hot zones → different spatial pattern per episode.
    std::vector<ScheduledTask> generate();

    // Build phase boundary table from EpisodeConfig.
    std::vector<PhaseInfo> build_phase_table() const;

    // Sample one random valid road node (uniform over the road graph).
    osmium::object_id_type sample_node_uniform();

    // Sample a node within radius_m metres of center_node.
    // Falls back to uniform if no node found within radius.
    osmium::object_id_type sample_node_near(osmium::object_id_type center_node,
                                             float radius_m);

    // Number of valid road nodes available.
    int node_count() const { return static_cast<int>(valid_nodes_.size()); }

private:
    const EpisodeConfig& cfg_;
    const GeoBox&        geo_box_;
    std::mt19937         rng_;

    // Flat index of valid OSM node IDs (connected to at least one way).
    std::vector<osmium::object_id_type> valid_nodes_;

    // Hot-zone centres for the current episode (resampled by generate()).
    std::vector<osmium::object_id_type> hot_zones_;

    // Delivery node of the last generated task.
    // Used to implement same_origin_prob: next task's pickup = this delivery.
    osmium::object_id_type last_delivery_ = 0;

    void build_valid_index();
    void resample_hot_zones(int n_zones);

    // Pick pickup/delivery pair respecting same_origin_prob.
    osmium::object_id_type sample_pickup (bool clustered);
    osmium::object_id_type sample_delivery(osmium::object_id_type pickup, bool clustered);

    // Haversine distance (metres) between two OSM node IDs.
    float haversine_m(osmium::object_id_type a, osmium::object_id_type b) const;

    // Estimate a base reward from geographical distance between pickup and delivery.
    float estimate_reward(osmium::object_id_type pickup,
                          osmium::object_id_type delivery) const;
};

#endif // TRAINING_EPISODE_GENERATOR_HPP
