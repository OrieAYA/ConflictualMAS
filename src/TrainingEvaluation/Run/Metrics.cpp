#include "TrainingEvaluation/Run/Metrics.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

// ════════════════════════════════════════════════════════════════════════════
// Part 1 — collection hooks
// ════════════════════════════════════════════════════════════════════════════

void EpisodeMetricsCollector::on_offer_sample(
    int agents_offered, int recall_rounds, int candidates_scored,
    long long alloc_us, long long pure_alloc_us, int tam_dijkstra_steps)
{
    tam_agents_offered_sum    += agents_offered;
    tam_recall_rounds_sum     += recall_rounds;
    tam_candidates_scored_sum += candidates_scored;
    tam_offer_samples         += 1;
    allocation_time_us_sum    += alloc_us;
    pure_alloc_time_us_sum    += pure_alloc_us;
    allocation_time_count     += 1;
    if (tam_dijkstra_steps > 0) {
        tam_dijkstra_steps_sum += tam_dijkstra_steps;
        tam_dijkstra_count     += 1;
    }
}

void EpisodeMetricsCollector::on_accept(float importance, bool high_cong) {
    ++n_accepted;
    imp_accepted_sum += importance;
    imp_accepted_n   += 1;
    if (high_cong) ++accepts_high_cong; else ++accepts_low_cong;
}

void EpisodeMetricsCollector::on_allocation_choice(
    const std::vector<float>& pre_marginal_costs,
    int chosen_agent_index, bool rmca_mode)
{
    if (chosen_agent_index < 0 ||
        static_cast<size_t>(chosen_agent_index) >= pre_marginal_costs.size())
        return;

    // Oracle ratio: chosen agent's marginal cost vs the fleet-wide cheapest.
    const float chosen = pre_marginal_costs[chosen_agent_index];
    float oracle = std::numeric_limits<float>::max();
    for (float c : pre_marginal_costs) if (c < oracle) oracle = c;
    if (oracle > 0.f && std::isfinite(oracle) && std::isfinite(chosen)) {
        marginal_ratio_sum   += static_cast<double>(chosen) / oracle;
        marginal_ratio_count += 1;
    }

    // RMCA(r) relative regret [Chen et al. 2021, eq.13/14/16]. mc(k1) is
    // floored at 1 m to avoid division blow-ups on near-coincident pickups,
    // and the ratio is clamped to [1, 50] to keep the episode mean robust.
    // A single eligible candidate ⇒ no second-best ⇒ regret = 1.
    if (!rmca_mode) return;
    constexpr float kInfeasible = 1e30f;   // capacity-infeasible sentinel
    float mc1 = std::numeric_limits<float>::max();   // cheapest        (k1)
    float mc2 = std::numeric_limits<float>::max();   // second cheapest (k2)
    for (float c : pre_marginal_costs) {
        if (!std::isfinite(c) || c < 0.f || c > kInfeasible) continue;
        if      (c < mc1) { mc2 = mc1; mc1 = c; }
        else if (c < mc2) { mc2 = c; }
    }
    if (mc1 > kInfeasible) return;                   // no feasible agent at all

    const bool  has_k2 = (mc2 <= kInfeasible);
    const float mc1f   = std::max(mc1, 1.0f);
    const float mc2f   = has_k2 ? mc2 : mc1;
    const float regret = std::clamp(mc2f / mc1f, 1.0f, 50.0f);

    rmca_mc_k1_sum    += static_cast<double>(mc1);
    rmca_mc_k2_sum    += static_cast<double>(mc2f);
    rmca_regret_sum   += static_cast<double>(regret);
    rmca_regret_count += 1;
}

// ════════════════════════════════════════════════════════════════════════════
// Part 2 — end-of-episode aggregation
// ════════════════════════════════════════════════════════════════════════════

ComparisonMetrics finalize_episode_metrics(
    const EpisodeMetricsCollector&                     c,
    const PDPGlobalMemory&                             memory,
    const std::vector<std::unique_ptr<DeliveryAgent>>& agents,
    const EpisodeConfig&                               cfg,
    int                                                total_steps,
    int                                                episode_fleet_size,
    bool                                               ghost_on,
    float                                              ghost_mean_active,
    const char*                                        congestion_profile)
{
    ComparisonMetrics metrics;
    metrics.method = "DMAS-MAPPO";

    // ── Throughput / latency / utilisation ─────────────────────────────────
    const int tasks_appeared = c.n_accepted + c.n_refused + c.n_no_candidate;
    metrics.tasks_appeared  = tasks_appeared;
    metrics.tasks_completed = c.latency_count;
    const int n_decisions   = c.n_accepted + c.n_refused;
    metrics.accept_rate     = n_decisions > 0
        ? static_cast<float>(c.n_accepted) / n_decisions : 0.f;
    metrics.refuse_rate     = 1.f - metrics.accept_rate;
    metrics.throughput_rate = tasks_appeared > 0
        ? static_cast<float>(c.latency_count) / tasks_appeared : 0.f;
    if (c.latency_count > 0)
        metrics.latency_mean = static_cast<float>(c.latency_sum) / c.latency_count;
    const int mean_active = (c.active_steps > 0)
        ? (c.active_sum / c.active_steps) : 1;
    if (mean_active > 0)
        metrics.latency_per_agent = metrics.latency_mean / mean_active;
    // Utilisation against the EFFECTIVE provisioned fleet (scenario-scaled).
    const int util_fleet = std::max(1, episode_fleet_size);
    metrics.agent_utilisation = (c.active_steps > 0 && c.n_accepted > 0)
        ? static_cast<float>(c.active_sum) / c.active_steps / util_fleet : 0.f;
    metrics.n_agents_max = episode_fleet_size;
    metrics.total_steps  = total_steps;

    metrics.mean_congestion = (c.congestion_steps > 0)
        ? static_cast<float>(c.congestion_sum / c.congestion_steps) : 0.f;
    metrics.mean_trip_steps = (c.trip_count > 0)
        ? static_cast<float>(c.trip_sum) / c.trip_count : 0.f;
    metrics.mean_wait_steps = (c.wait_count > 0)
        ? static_cast<float>(c.wait_sum) / c.wait_count : 0.f;
    metrics.mean_road_pd_m  = (c.road_pd_count > 0)
        ? static_cast<float>(c.road_pd_sum / c.road_pd_count) : 0.f;

    // ── Selectivity diagnostics ────────────────────────────────────────────
    metrics.completion_per_accepted = (c.n_accepted > 0)
        ? static_cast<float>(c.latency_count) / static_cast<float>(c.n_accepted) : 0.f;
    metrics.unfinished_accept_rate  = (c.n_accepted > 0)
        ? 1.f - metrics.completion_per_accepted : 0.f;
    metrics.mean_congestion_at_decision = (c.congestion_at_decision_count > 0)
        ? static_cast<float>(c.congestion_at_decision_sum
                             / c.congestion_at_decision_count) : 0.f;
    metrics.n_ghost_active_mean      = ghost_on ? ghost_mean_active : 0.f;
    metrics.congestion_profile_label = ghost_on
        ? std::string(congestion_profile ? congestion_profile : "")
        : std::string();

    // ── TAM efficiency means (one sample per offer call) ───────────────────
    if (c.tam_offer_samples > 0) {
        metrics.mean_agents_offered_per_task = static_cast<float>(
            c.tam_agents_offered_sum) / c.tam_offer_samples;
        metrics.mean_recall_rounds_per_task = static_cast<float>(
            c.tam_recall_rounds_sum)  / c.tam_offer_samples;
        metrics.mean_candidates_scored_per_task = static_cast<float>(
            c.tam_candidates_scored_sum) / c.tam_offer_samples;
    }

    // ── Selection intelligence (delivery quality, not just count) ──────────
    metrics.value_throughput_rate = (c.value_appeared_sum > 0.0)
        ? static_cast<float>(c.value_delivered_sum / c.value_appeared_sum) : 0.f;
    metrics.mean_completion_value = (c.latency_count > 0)
        ? static_cast<float>(c.value_delivered_sum / c.latency_count) : 0.f;
    metrics.value_loss_to_refusal = static_cast<float>(c.value_refused_sum);

    // ── Real impact on edge traversal (BPR factors actually paid) ──────────
    metrics.mean_bpr_along_route = (c.bpr_along_route_count > 0)
        ? static_cast<float>(c.bpr_along_route_sum / c.bpr_along_route_count)
        : 1.f;
    metrics.time_lost_to_congestion_steps =
        static_cast<float>(c.time_lost_to_congestion_sum);
    metrics.n_traversals_in_jam = c.n_traversals_in_jam;

    // ── Allocation optimality vs full-scan cheapest-insertion oracle ──────────────────────
    metrics.marginal_cost_ratio_vs_oracle = (c.marginal_ratio_count > 0)
        ? static_cast<float>(c.marginal_ratio_sum / c.marginal_ratio_count)
        : 1.f;

    // ── RMCA(r) regret diagnostics (RMCA mode only) ────────────────────────
    metrics.rmca_relative_regret  = (c.rmca_regret_count > 0)
        ? static_cast<float>(c.rmca_regret_sum / c.rmca_regret_count) : 0.f;
    metrics.rmca_marginal_cost_k1 = (c.rmca_regret_count > 0)
        ? static_cast<float>(c.rmca_mc_k1_sum / c.rmca_regret_count) : 0.f;
    metrics.rmca_marginal_cost_k2 = (c.rmca_regret_count > 0)
        ? static_cast<float>(c.rmca_mc_k2_sum / c.rmca_regret_count) : 0.f;

    // ── Temporal complexity (allocation-only wallclock + TAM Dijkstra) ─────
    metrics.mean_allocation_time_us = (c.allocation_time_count > 0)
        ? static_cast<float>(c.allocation_time_us_sum) / c.allocation_time_count
        : 0.f;
    metrics.mean_tam_dijkstra_steps = (c.tam_dijkstra_count > 0)
        ? static_cast<float>(c.tam_dijkstra_steps_sum) / c.tam_dijkstra_count
        : 0.f;

    // ── Path-compute breakdown (ms units) ──────────────────────────────────
    metrics.path_compute_time_ms =
        static_cast<float>(c.path_compute_time_us_ep) / 1000.f;
    metrics.mean_pure_alloc_time_ms = (c.allocation_time_count > 0)
        ? (static_cast<float>(c.pure_alloc_time_us_sum)
           / c.allocation_time_count) / 1000.f
        : 0.f;

    // ── Multi-axis performance diagnostics ─────────────────────────────────
    metrics.mean_imp_accepted = (c.imp_accepted_n > 0)
        ? static_cast<float>(c.imp_accepted_sum / c.imp_accepted_n) : 0.f;
    metrics.mean_imp_refused  = (c.imp_refused_n > 0)
        ? static_cast<float>(c.imp_refused_sum  / c.imp_refused_n)  : 0.f;
    metrics.accept_rate_high_cong = (c.accepts_high_cong > 0) ? 1.f : 0.f;
    metrics.accept_rate_low_cong  = (c.accepts_low_cong  > 0) ? 1.f : 0.f;

    // ── Per-agent load balance over delivered tasks ─────────────────────────
    // Counts include only agents that received at least one allocation during
    // the episode (Gini / CoV over the active fleet, not the oversized pool).
    {
        std::vector<int> per_agent_completed(agents.size(), 0);
        std::vector<int> per_agent_allocated(agents.size(), 0);
        for (const PDPTask* t : memory.finished_tasks)
            if (t && t->agent_id >= 0 && t->agent_id < (int)agents.size())
                ++per_agent_completed[t->agent_id];
        for (const PDPTask* t : memory.allocated_tasks)
            if (t && t->agent_id >= 0 && t->agent_id < (int)agents.size())
                ++per_agent_allocated[t->agent_id];

        std::vector<int> active_completed;
        active_completed.reserve(per_agent_completed.size());
        int mx = 0, mn = std::numeric_limits<int>::max();
        bool any = false;
        for (size_t i = 0; i < per_agent_completed.size(); ++i) {
            if (per_agent_completed[i] > 0 || per_agent_allocated[i] > 0) {
                active_completed.push_back(per_agent_completed[i]);
                mx = std::max(mx, per_agent_completed[i]);
                mn = std::min(mn, per_agent_completed[i]);
                any = true;
            }
        }
        metrics.max_agent_completed = any ? mx : 0;
        metrics.min_agent_completed = any ? mn : 0;

        if (!active_completed.empty()) {
            std::sort(active_completed.begin(), active_completed.end());
            const size_t n = active_completed.size();
            double sum = 0.0;
            for (int v : active_completed) sum += v;
            const double mean = sum / static_cast<double>(n);

            // Gini (sorted ascending: sum_i (2*i - n + 1) * x_i) / (n * sum)
            double gnum = 0.0;
            for (size_t i = 0; i < n; ++i)
                gnum += (2.0 * static_cast<double>(i + 1)
                         - static_cast<double>(n) - 1.0)
                        * static_cast<double>(active_completed[i]);
            metrics.agent_completed_gini = (sum > 0.0)
                ? static_cast<float>(gnum / (static_cast<double>(n) * sum))
                : 0.f;

            // Coefficient of variation (std / mean).
            double var = 0.0;
            for (int v : active_completed) {
                const double d = static_cast<double>(v) - mean;
                var += d * d;
            }
            var /= static_cast<double>(n);
            metrics.agent_completed_std = (mean > 0.0)
                ? static_cast<float>(std::sqrt(var) / mean) : 0.f;
        }
    }

    // Total fleet distance = metres actually driven by the whole fleet. Was
    // Σ P→D road distance over completed tasks, which is a different quantity
    // from the identically-named SoTA column (real traversed distance) — the
    // two were not comparable despite sharing the name.
    metrics.total_fleet_distance_m = static_cast<float>(c.fleet_distance_m);

    // ── Route efficiency + detour magnitude ─────────────────────────────────
    {
        const float ideal_steps = (cfg.speed_mps > 0.f && metrics.mean_road_pd_m > 0.f)
            ? metrics.mean_road_pd_m / cfg.speed_mps : 0.f;
        metrics.mean_extra_steps_per_task =
            (metrics.mean_trip_steps > 0.f && ideal_steps > 0.f)
            ? std::max(0.f, metrics.mean_trip_steps - ideal_steps) : 0.f;
        metrics.delivery_route_efficiency =
            (metrics.mean_trip_steps > 0.f && ideal_steps > 0.f)
            ? std::min(1.f, ideal_steps / metrics.mean_trip_steps) : 0.f;
    }

    // ── Network-level congestion impact ─────────────────────────────────────
    metrics.peak_congestion = c.peak_load_episode;
    metrics.mean_overlap_edges = (c.overlap_steps > 0)
        ? static_cast<float>(c.overlap_edges_sum) / c.overlap_steps : 0.f;
    if (c.load_now_n > 1) {
        const double m   = c.load_now_sum_lin / c.load_now_n;
        const double var = std::max(0.0, c.load_now_sum_sq / c.load_now_n - m * m);
        metrics.congestion_variance = static_cast<float>(std::sqrt(var));
    } else {
        metrics.congestion_variance = 0.f;
    }
    metrics.route_congestion_exposure = (c.route_exposure_n > 0)
        ? static_cast<float>(c.route_exposure_sum / c.route_exposure_n) : 0.f;

    // ── Spatial complexity over served tasks ────────────────────────────────
    // Walk all accepted tasks (allocated + finished): bbox area, convex hull
    // area, mean P→D distance, mean nearest-neighbour pickup distance.
    {
        struct PtLL { double lat, lon; };
        std::vector<PtLL> pus, dels;
        auto add_task_points = [&](const PDPTask* t) {
            if (!t) return;
            auto itp = memory.geo_box.data.nodes.find(t->pickup.id);
            auto itd = memory.geo_box.data.nodes.find(t->delivery.id);
            if (itp == memory.geo_box.data.nodes.end() ||
                itd == memory.geo_box.data.nodes.end()) return;
            pus .push_back({itp->second.lat, itp->second.lon});
            dels.push_back({itd->second.lat, itd->second.lon});
        };
        for (const PDPTask* t : memory.finished_tasks)  add_task_points(t);
        for (const PDPTask* t : memory.allocated_tasks) add_task_points(t);

        constexpr double kPi = 3.14159265358979323846;
        auto hav = [&](double la1, double lo1, double la2, double lo2) {
            constexpr double R = 6371000.0;
            const double dlat = (la2 - la1) * kPi / 180.0;
            const double dlon = (lo2 - lo1) * kPi / 180.0;
            const double a1 = la1 * kPi / 180.0;
            const double a2 = la2 * kPi / 180.0;
            const double a = std::sin(dlat/2)*std::sin(dlat/2)
                           + std::cos(a1)*std::cos(a2)*std::sin(dlon/2)*std::sin(dlon/2);
            return R * 2.0 * std::atan2(std::sqrt(a), std::sqrt(1-a));
        };

        if (!pus.empty()) {
            double mn_lat=+1e9, mx_lat=-1e9, mn_lon=+1e9, mx_lon=-1e9;
            for (size_t i = 0; i < pus.size(); ++i) {
                mn_lat = std::min({mn_lat, pus[i].lat, dels[i].lat});
                mx_lat = std::max({mx_lat, pus[i].lat, dels[i].lat});
                mn_lon = std::min({mn_lon, pus[i].lon, dels[i].lon});
                mx_lon = std::max({mx_lon, pus[i].lon, dels[i].lon});
            }
            const double w = hav(mn_lat, mn_lon, mn_lat, mx_lon);
            const double h = hav(mn_lat, mn_lon, mx_lat, mn_lon);
            metrics.bbox_area_km2 = static_cast<float>((w * h) / 1e6);

            // Convex hull (Andrew's monotone chain) on local-flat projection.
            const double cos_lat = std::cos((mn_lat + mx_lat) * 0.5 * kPi / 180.0);
            struct P2 { double x, y; };
            std::vector<P2> pts;
            pts.reserve(pus.size() * 2);
            for (size_t i = 0; i < pus.size(); ++i) {
                pts.push_back({pus [i].lon * 111320.0 * cos_lat, pus [i].lat * 111320.0});
                pts.push_back({dels[i].lon * 111320.0 * cos_lat, dels[i].lat * 111320.0});
            }
            std::sort(pts.begin(), pts.end(),
                      [](const P2& a, const P2& b){
                          return (a.x != b.x) ? a.x < b.x : a.y < b.y;
                      });
            auto crossp = [](const P2& O, const P2& A, const P2& B){
                return (A.x-O.x)*(B.y-O.y) - (A.y-O.y)*(B.x-O.x);
            };
            const int nP = static_cast<int>(pts.size());
            std::vector<P2> hull(2 * nP);
            int k = 0;
            for (int i = 0; i < nP; ++i) {
                while (k >= 2 && crossp(hull[k-2], hull[k-1], pts[i]) <= 0) --k;
                hull[k++] = pts[i];
            }
            for (int i = nP-2, tlim = k+1; i >= 0; --i) {
                while (k >= tlim && crossp(hull[k-2], hull[k-1], pts[i]) <= 0) --k;
                hull[k++] = pts[i];
            }
            hull.resize(k - 1);
            double sarea = 0;
            for (size_t i = 0; i < hull.size(); ++i) {
                const auto& A = hull[i];
                const auto& B = hull[(i+1) % hull.size()];
                sarea += A.x * B.y - B.x * A.y;
            }
            metrics.convex_hull_area_km2 =
                static_cast<float>(std::abs(sarea) * 0.5 / 1e6);

            // Mean P→D direct distance.
            double sum_pd = 0.0;
            for (size_t i = 0; i < pus.size(); ++i)
                sum_pd += hav(pus[i].lat, pus[i].lon, dels[i].lat, dels[i].lon);
            metrics.mean_pd_distance_m = static_cast<float>(sum_pd / pus.size());

            // Mean nearest-neighbour pickup distance (O(N²), N is small here).
            double sum_nn = 0.0; int n_nn = 0;
            for (size_t i = 0; i < pus.size(); ++i) {
                double best = 1e18;
                for (size_t j = 0; j < pus.size(); ++j) {
                    if (i == j) continue;
                    const double d = hav(pus[i].lat, pus[i].lon,
                                          pus[j].lat, pus[j].lon);
                    if (d < best) best = d;
                }
                if (best < 1e17) { sum_nn += best; ++n_nn; }
            }
            metrics.mean_nn_pickup_m = (n_nn > 0)
                ? static_cast<float>(sum_nn / n_nn) : 0.f;
        }
    }

    // ── Validity audit over the served tasks (must remain 0) ────────────────
    {
        const int max_carry = std::max(
            1, memory.task_agent.params.max_tasks_per_agent);
        for (const PDPTask* t : memory.finished_tasks) {
            if (!t) continue;
            if (t->timeline.delivered_step >= 0 &&
                t->timeline.picked_step    >= 0 &&
                t->timeline.picked_step >= t->timeline.delivered_step)
                ++metrics.pairing_violations_runtime;
            if (t->timeline.picked_step    >= 0 &&
                t->timeline.created_step   >= 0 &&
                t->timeline.picked_step < t->timeline.created_step)
                ++metrics.pairing_violations_runtime;
        }
        // Capacity: replay per-agent timeline.
        struct Ev { int step; int delta; };
        std::unordered_map<int, std::vector<Ev>> by_agent;
        auto push_task = [&](const PDPTask* t){
            if (!t || t->agent_id < 0) return;
            if (t->timeline.picked_step    >= 0)
                by_agent[t->agent_id].push_back({t->timeline.picked_step,    +1});
            if (t->timeline.delivered_step >= 0)
                by_agent[t->agent_id].push_back({t->timeline.delivered_step, -1});
        };
        for (const PDPTask* t : memory.finished_tasks)  push_task(t);
        for (const PDPTask* t : memory.allocated_tasks) push_task(t);
        for (auto& [aid, evs] : by_agent) {
            std::sort(evs.begin(), evs.end(),
                      [](const Ev& a, const Ev& b){ return a.step < b.step; });
            int load = 0, peak = 0;
            for (const auto& e : evs) { load += e.delta; if (load > peak) peak = load; }
            // Use the agent's OWN capacity, not the global TAM ceiling (the
            // ceiling equals hetero_capacity_max under a heterogeneous fleet).
            const DeliveryAgent* a = memory.get_delivery_agent(aid);
            const int agent_cap = (a && a->max_capacity > 0) ? a->max_capacity
                                                             : max_carry;
            if (peak > agent_cap)
                metrics.capacity_violations_runtime += peak - agent_cap;
        }
    }

    return metrics;
}
