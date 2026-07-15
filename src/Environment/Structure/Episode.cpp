#include "Environment/Structure/Episode.hpp"
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
    build_spatial_grid();
}

// ── Spatial grid (cell ≈ hot_zone_radius) ─────────────────────────────────────
void EpisodeGenerator::build_spatial_grid() {
    spatial_grid_.clear();
    if (valid_nodes_.empty()) return;

    double min_lat =  90.0, max_lat = -90.0;
    double min_lon = 180.0, max_lon = -180.0;
    for (auto id : valid_nodes_) {
        auto it = geo_box_.data.nodes.find(id);
        if (it == geo_box_.data.nodes.end()) continue;
        min_lat = std::min(min_lat, it->second.lat);
        max_lat = std::max(max_lat, it->second.lat);
        min_lon = std::min(min_lon, it->second.lon);
        max_lon = std::max(max_lon, it->second.lon);
    }
    grid_min_lat_ = min_lat;
    grid_min_lon_ = min_lon;

    const double r = std::max(cfg_.hot_zone_radius, 200.f);  // metres
    const double mid_lat_rad = ((min_lat + max_lat) * 0.5) * 3.14159265358979323846 / 180.0;
    cell_lat_deg_ = r / 111000.0;
    cell_lon_deg_ = r / (111000.0 * std::max(std::cos(mid_lat_rad), 0.01));

    for (auto id : valid_nodes_) {
        auto it = geo_box_.data.nodes.find(id);
        if (it == geo_box_.data.nodes.end()) continue;
        spatial_grid_[cell_of(it->second.lat, it->second.lon)].push_back(id);
    }
}

std::pair<int,int> EpisodeGenerator::cell_of(double lat, double lon) const {
    return { static_cast<int>(std::floor((lat - grid_min_lat_) / cell_lat_deg_)),
             static_cast<int>(std::floor((lon - grid_min_lon_) / cell_lon_deg_)) };
}

// ── Hot zone re-sampling ──────────────────────────────────────────────────────
void EpisodeGenerator::resample_hot_zones(int n_zones) {
    hot_zones_.clear();
    if (n_zones <= 0 || cfg_.cluster_prob <= 0.f || valid_nodes_.empty()) return;
    hot_zones_.reserve(n_zones);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(valid_nodes_.size()) - 1);
    for (int i = 0; i < n_zones; ++i)
        hot_zones_.push_back(valid_nodes_[pick(rng_)]);
}

// ── Phase table ───────────────────────────────────────────────────────────────
std::vector<PhaseInfo> EpisodeGenerator::build_phase_table() const {
    std::vector<PhaseInfo> table;
    int step = 0;
    for (const auto& p : cfg_.phases) {
        int nz = (p.n_hot_zones >= 0) ? p.n_hot_zones : cfg_.n_hot_zones;
        table.push_back({ step, step + p.steps,
                          p.n_agents_start, p.n_agents_end, p.label, nz });
        step += p.steps;
    }
    return table;
}

// ── Episode generation ────────────────────────────────────────────────────────
std::vector<ScheduledTask> EpisodeGenerator::generate(TemporalProfile profile) {
    last_delivery_ = 0;
    auto phases = build_phase_table();
    std::vector<ScheduledTask> stream;

    std::uniform_real_distribution<float> unit(0.f, 1.f);
    // Importance distribution: widened range when task-value heterogeneity is on
    // (Phase 3). Default [0.5, 2.0] keeps the existing behaviour for all other
    // options.
    const float imp_lo = cfg_.enable_task_value_heterogeneity
        ? std::max(0.05f, cfg_.task_imp_min) : 0.5f;
    const float imp_hi = cfg_.enable_task_value_heterogeneity
        ? std::max(imp_lo + 0.01f, cfg_.task_imp_max) : 2.0f;
    std::uniform_real_distribution<float> imp_dist(imp_lo, imp_hi);

    // Per-task reward multiplier — independent of distance — gated by the
    // task-value-heterogeneity flag. Decouples task VALUE from task EFFORT,
    // so two same-distance tasks can have very different rewards.
    const float val_lo = cfg_.enable_task_value_heterogeneity
        ? std::max(0.1f, cfg_.task_value_mul_min) : 1.0f;
    const float val_hi = cfg_.enable_task_value_heterogeneity
        ? std::max(val_lo + 0.01f, cfg_.task_value_mul_max) : 1.0f;
    std::uniform_real_distribution<float> val_mul_dist(val_lo, val_hi);

    // Episode horizon — used for the feasibility filter below.
    const int total_episode_steps = cfg_.total_steps();
    const float speed = std::max(cfg_.speed_mps, 0.1f);

    // Feasibility-margin multiplier on the minimum delivery time. Accounts for
    //   (i) the pickup leg (agent → pickup), unknown at generation time but
    //       comparable in scale to the delivery leg
    //   (ii) the road-vs-haversine factor (real road distance is typically
    //        1.3–1.6× the great-circle distance)
    // Tasks that need more than `feasibility_margin × haversine_steps` of
    // remaining episode time are physically un-deliverable; we skip them at
    // generation rather than letting them pollute the buffer with un-finishable
    // accept decisions. A logical pickup-then-delivery task must be doable.
    constexpr float kFeasibilityMargin = 2.2f;

    if (total_episode_steps <= 0) return stream;

    // ── Event count = round(100 · SCE · RM); fd only shapes WHEN they occur ─
    const int n_events = event_tuning::derived_task_count(cfg_.env_scale,
                                                          cfg_.ratio_mult);

    // ── Arrival steps: density ∝ profile(t/T) (inverse-CDF sampling) ───────
    std::vector<float> weights(static_cast<size_t>(total_episode_steps), 0.f);
    double weight_sum = 0.0;
    for (int s = 0; s < total_episode_steps; ++s) {
        const float f = std::max(0.f, temporal_profile_value(
            profile, (static_cast<float>(s) + 0.5f) / total_episode_steps));
        weights[s]  = f;
        weight_sum += f;
    }
    if (weight_sum <= 0.0)                       // degenerate profile → uniform
        std::fill(weights.begin(), weights.end(), 1.f);

    std::discrete_distribution<int> step_dist(weights.begin(), weights.end());
    std::vector<int> arrival_steps(static_cast<size_t>(n_events));
    for (int i = 0; i < n_events; ++i) arrival_steps[i] = step_dist(rng_);
    std::sort(arrival_steps.begin(), arrival_steps.end());

    // ── Per-task draws (walk arrivals in order; hot zones follow phases) ───
    size_t phase_idx = 0;
    resample_hot_zones(phases.empty() ? cfg_.n_hot_zones
                                      : phases.front().n_hot_zones);
    stream.reserve(arrival_steps.size());
    for (int step : arrival_steps) {
        // Resample hot zones at each phase boundary so spatial clusters
        // shift with demand (commercial morning peak → residential evening).
        while (phase_idx + 1 < phases.size() && step >= phases[phase_idx].step_end) {
            ++phase_idx;
            resample_hot_zones(phases[phase_idx].n_hot_zones);
        }

        const int steps_remaining = total_episode_steps - step;
        bool clustered = (!hot_zones_.empty()) && (unit(rng_) < cfg_.cluster_prob);

        auto pu  = sample_pickup(clustered);
        auto del = sample_delivery(pu, clustered);
        if (pu == 0 || del == 0 || pu == del) continue;

        // Feasibility filter: drop tasks that cannot logically finish
        // before the episode ends. Computes the lower-bound delivery
        // time from the great-circle distance and rejects when even
        // the most generous margin overshoots the remaining horizon.
        const float dist_m       = haversine_m(pu, del);
        const float min_steps    = dist_m / speed;
        const float needed_steps = min_steps * kFeasibilityMargin;
        if (needed_steps > static_cast<float>(steps_remaining)) {
            // Try one fallback: resample a closer delivery once. If
            // still infeasible, drop the task entirely (rather than
            // ship an impossible pickup-then-delivery).
            auto alt = sample_delivery(pu, false);
            if (alt == 0 || alt == pu) continue;
            const float alt_dist = haversine_m(pu, alt);
            if ((alt_dist / speed) * kFeasibilityMargin >
                static_cast<float>(steps_remaining)) continue;
            del = alt;
        }

        ScheduledTask t;
        t.arrival_step     = step;
        t.pickup_node_id   = pu;
        t.delivery_node_id = del;
        // When task-value heterogeneity is enabled, multiply the
        // distance-based reward by a per-task random factor so the
        // policy faces a real "value vs effort" tradeoff. Otherwise
        // val_mul_dist is identity (1.0 always) — legacy behaviour.
        t.reward           = estimate_reward(pu, del) * val_mul_dist(rng_);
        t.importance       = imp_dist(rng_);
        t.is_clustered     = clustered;
        stream.push_back(t);
        last_delivery_ = del;
    }
    return stream;
}

// ── Scenario density scaling (canonical sub/supersample) ─────────────────────
void apply_density_mult(std::vector<ScheduledTask>& stream,
                        float density_mult, uint32_t seed)
{
    if (density_mult == 1.0f || stream.empty()) return;

    std::mt19937 rng(seed);
    const int target = std::max<int>(
        1, static_cast<int>(std::round(stream.size() * density_mult)));
    auto by_arrival = [](const ScheduledTask& a, const ScheduledTask& b) {
        return a.arrival_step < b.arrival_step;
    };
    if (density_mult <= 1.0f) {
        std::shuffle(stream.begin(), stream.end(), rng);
        stream.resize(static_cast<size_t>(target));
        std::sort(stream.begin(), stream.end(), by_arrival);
    } else {
        std::vector<ScheduledTask> scaled;
        scaled.reserve(static_cast<size_t>(target));
        const int max_step = stream.back().arrival_step + 1;
        std::uniform_int_distribution<size_t> pick(0, stream.size() - 1);
        std::uniform_int_distribution<int>    step_pick(0, std::max(1, max_step - 1));
        for (int i = 0; i < target; ++i) {
            ScheduledTask t = stream[pick(rng)];
            t.arrival_step = step_pick(rng);
            scaled.push_back(t);
        }
        std::sort(scaled.begin(), scaled.end(), by_arrival);
        stream = std::move(scaled);
    }
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

    const double clat = it_c->second.lat;
    const double clon = it_c->second.lon;

    // Scan only cells whose bounding box overlaps the radius.
    const int span_lat = std::max(1, (int)std::ceil(radius_m / 111000.0 / cell_lat_deg_));
    const int span_lon = std::max(1, (int)std::ceil(radius_m / 111000.0 / cell_lon_deg_));
    const auto [ci, cj] = cell_of(clat, clon);

    std::vector<osmium::object_id_type> candidates;
    for (int di = -span_lat; di <= span_lat; ++di) {
        for (int dj = -span_lon; dj <= span_lon; ++dj) {
            auto it = spatial_grid_.find({ci + di, cj + dj});
            if (it == spatial_grid_.end()) continue;
            for (auto id : it->second) {
                auto nit = geo_box_.data.nodes.find(id);
                if (nit == geo_box_.data.nodes.end()) continue;
                double d = calculate_haversine_distance(
                    clat, clon, nit->second.lat, nit->second.lon);
                if (d <= radius_m) candidates.push_back(id);
            }
        }
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
    const float min_d = cfg_.min_task_dist_m;
    const float max_d = cfg_.max_task_dist_m;

    osmium::object_id_type del = 0;
    for (int tries = 0; tries < 12; ++tries) {
        osmium::object_id_type candidate;
        if (clustered && !hot_zones_.empty()) {
            std::uniform_int_distribution<int> zd(0, static_cast<int>(hot_zones_.size()) - 1);
            candidate = sample_node_near(hot_zones_[zd(rng_)], cfg_.hot_zone_radius);
        } else {
            candidate = sample_node_uniform();
        }
        if (candidate == pickup) continue;
        float d = haversine_m(pickup, candidate);
        if (d >= min_d && d <= max_d) { del = candidate; break; }
    }
    // Fallback: relax distance constraint rather than produce a degenerate task.
    if (del == 0 || del == pickup)
        del = sample_node_uniform();
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
    return std::clamp(dist / 2000.f, event_tuning::kTaskRewardClampMin,
                      event_tuning::kTaskRewardClampMax);
}
