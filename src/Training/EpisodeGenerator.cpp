#include "EpisodeGenerator.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// ── Construction ──────────────────────────────────────────────────────────────
EpisodeGenerator::EpisodeGenerator(const EpisodeConfig& cfg,
                                   const GeoBox&        geo_box,
                                   uint32_t             seed)
    : cfg_(cfg), geo_box_(geo_box), rng_(seed)
{
    if (!geo_box_.is_valid)
        throw std::runtime_error("EpisodeGenerator: GeoBox is not valid");
    build_valid_index();
}

// ── Valid node index ──────────────────────────────────────────────────────────
void EpisodeGenerator::build_valid_index() {
    valid_nodes_.clear();
    valid_nodes_.reserve(geo_box_.data.nodes.size());
    for (const auto& [id, pt] : geo_box_.data.nodes) {
        if (!pt.incident_ways.empty())
            valid_nodes_.push_back(id);
    }
    if (valid_nodes_.empty())
        throw std::runtime_error("EpisodeGenerator: no valid road nodes in GeoBox");
    std::sort(valid_nodes_.begin(), valid_nodes_.end()); // deterministic ordering
}

// ── Hot zone re-sampling ──────────────────────────────────────────────────────
void EpisodeGenerator::resample_hot_zones() {
    hot_zones_.clear();
    int n = cfg_.n_hot_zones;
    if (n <= 0 || cfg_.cluster_prob <= 0.f) return;

    hot_zones_.reserve(n);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(valid_nodes_.size()) - 1);
    for (int i = 0; i < n; ++i)
        hot_zones_.push_back(valid_nodes_[pick(rng_)]);
}

// ── Phase table ───────────────────────────────────────────────────────────────
std::vector<PhaseInfo> EpisodeGenerator::build_phase_table() const {
    std::vector<PhaseInfo> table;
    int step = 0;
    for (const auto& p : cfg_.phases) {
        table.push_back({ step, step + p.steps, p.lambda, p.n_agents, p.label });
        step += p.steps;
    }
    return table;
}

// ── Episode generation ────────────────────────────────────────────────────────
std::vector<ScheduledTask> EpisodeGenerator::generate() {
    resample_hot_zones();
    last_delivery_ = 0;   // reset chain for this episode
    auto phases = build_phase_table();
    std::vector<ScheduledTask> stream;

    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> imp_dist(0.5f, 2.0f);

    for (const auto& ph : phases) {
        // Bernoulli arrival with probability = lambda (valid for lambda <= 0.2)
        float lam = std::min(ph.lambda, 1.0f);

        for (int step = ph.step_begin; step < ph.step_end; ++step) {
            if (unit(rng_) > lam) continue; // no task this step

            bool clustered = (!hot_zones_.empty()) && (unit(rng_) < cfg_.cluster_prob);

            auto pu  = sample_pickup (clustered);
            auto del = sample_delivery(pu, clustered);
            if (pu == 0 || del == 0 || pu == del) continue; // degenerate

            ScheduledTask t;
            t.arrival_step     = step;
            t.pickup_node_id   = pu;
            t.delivery_node_id = del;
            t.reward           = estimate_reward(pu, del);
            t.importance       = imp_dist(rng_);
            t.is_clustered     = clustered;
            stream.push_back(t);
            last_delivery_ = del;  // enable same_origin_prob for the next task
        }
    }
    return stream;
}

// ── Node sampling ─────────────────────────────────────────────────────────────
osmium::object_id_type EpisodeGenerator::sample_node_uniform() {
    if (valid_nodes_.empty()) return 0;
    std::uniform_int_distribution<int> d(0, static_cast<int>(valid_nodes_.size()) - 1);
    return valid_nodes_[d(rng_)];
}

osmium::object_id_type EpisodeGenerator::sample_node_near(
    osmium::object_id_type center, float radius_m)
{
    auto it_c = geo_box_.data.nodes.find(center);
    if (it_c == geo_box_.data.nodes.end()) return sample_node_uniform();

    double clat = it_c->second.lat;
    double clon = it_c->second.lon;

    // Collect candidates within radius (linear scan — acceptable for training).
    std::vector<osmium::object_id_type> candidates;
    for (auto id : valid_nodes_) {
        auto it = geo_box_.data.nodes.find(id);
        if (it == geo_box_.data.nodes.end()) continue;
        double d = calculate_haversine_distance(clat, clon,
                                                it->second.lat, it->second.lon);
        if (d <= radius_m) candidates.push_back(id);
    }
    if (candidates.empty()) return sample_node_uniform();
    std::uniform_int_distribution<int> pick(0, static_cast<int>(candidates.size()) - 1);
    return candidates[pick(rng_)];
}

// ── Private helpers ───────────────────────────────────────────────────────────
osmium::object_id_type EpisodeGenerator::sample_pickup(bool clustered) {
    // same_origin_prob: model return-trips / lifelong locality by starting a new
    // task from the previous task's delivery node (chaining tasks spatially).
    if (cfg_.same_origin_prob > 0.f && last_delivery_ != 0) {
        std::bernoulli_distribution coin(cfg_.same_origin_prob);
        if (coin(rng_)) return last_delivery_;
    }
    if (!clustered || hot_zones_.empty()) return sample_node_uniform();
    std::uniform_int_distribution<int> zp(0, static_cast<int>(hot_zones_.size()) - 1);
    return sample_node_near(hot_zones_[zp(rng_)], cfg_.hot_zone_radius);
}

osmium::object_id_type EpisodeGenerator::sample_delivery(
    osmium::object_id_type pickup, bool clustered)
{
    // same_origin_prob: delivery = previous pickup (lifelong reuse scenario)
    // For now, always sample independently; same_origin_prob is reserved.
    osmium::object_id_type del = 0;
    for (int tries = 0; tries < 5; ++tries) {
        if (clustered && !hot_zones_.empty()) {
            std::uniform_int_distribution<int> zd(0, static_cast<int>(hot_zones_.size()) - 1);
            del = sample_node_near(hot_zones_[zd(rng_)], cfg_.hot_zone_radius);
        } else {
            del = sample_node_uniform();
        }
        if (del != pickup) break;
    }
    return del;
}

float EpisodeGenerator::haversine_m(osmium::object_id_type a,
                                     osmium::object_id_type b) const
{
    auto ia = geo_box_.data.nodes.find(a);
    auto ib = geo_box_.data.nodes.find(b);
    if (ia == geo_box_.data.nodes.end() || ib == geo_box_.data.nodes.end())
        return 0.f;
    return static_cast<float>(
        calculate_haversine_distance(ia->second.lat, ia->second.lon,
                                     ib->second.lat, ib->second.lon));
}

float EpisodeGenerator::estimate_reward(osmium::object_id_type pickup,
                                         osmium::object_id_type delivery) const
{
    float dist = haversine_m(pickup, delivery);
    // Reward proportional to task distance: longer task → higher reward.
    // Normalised so that a 2-km task gives reward ≈ 1.0.
    return std::clamp(dist / 2000.f, 0.1f, 5.0f);
}
