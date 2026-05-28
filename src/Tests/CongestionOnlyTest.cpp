#include "CongestionOnlyTest.hpp"
#include "Training/CityConfig.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Environment/Congestion/CongestionMap.hpp"
#include "Environment/Congestion/GhostTrafficController.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Mirror customize_episode_for_city's ghost_n_max tiering. Kept local to avoid
// pulling MultiCityTrainer just for one helper.
int ghost_n_max_for(const CityConfig& cc) {
    const double a = cc.area_km2;
    if (a < 70.0)  return 20;
    if (a < 300.0) return 40;
    return 80;
}

struct ScenarioSpec {
    const char*       label;
    CongestionProfile profile;
    int               n_max_override; // -1 = use per-city, else override (over_fleet)
};

// Snapshot of CongestionMap state at one sampling step.
struct Sample {
    int   step;
    int   peak_load;
    float mean_load;
    int   n_overlap;   // edges with load >= 2
    int   n_jam;       // edges with load >= 5
    int   n_heavy;     // edges with load >= 10
    int   n_active_ghosts;
};

struct ScenarioRun {
    std::string scenario;
    int         n_max;
    int         n_hot_ways;
    int         peak_load_episode;
    int         peak_step;
    float       mean_load_episode;
    int         n_overlap_at_peak;
    int         n_jam_at_peak;
    int         n_heavy_at_peak;
    int         max_overlap_episode;
    long long   ms;
    // BPR cost factor at peak load on a typical 100m road (cap = 5).
    // factor = 1 + 0.15 x (peak/5)^4. Reported as percentage cost increase.
    float       bpr_factor_typical;
};

float bpr_factor(int load, float distance_m, const CongestionParams& p) {
    const float cap   = std::max(1.f, distance_m * p.capacity_per_meter);
    const float ratio = static_cast<float>(load) / cap;
    return 1.f + p.bpr_alpha * std::pow(ratio, p.bpr_beta);
}

ScenarioRun simulate_scenario(const GeoBox& geo_box,
                              const ScenarioSpec& spec,
                              int /*n_max_city*/,
                              int total_steps)
{
    ScenarioRun out{};
    out.scenario = spec.label;
    // n_max is now derived from density × hot_count INSIDE the controller's
    // reset(). We pass 100 as a placeholder; the controller will overwrite
    // it from the density formula. Read back after reset for reporting.
    out.n_max    = (spec.n_max_override > 0) ? spec.n_max_override : 100;

    CongestionMap cmap;   // default params (BPR α=0.15, β=4, cap=0.05/m)
    GhostTrafficController ghost;

    GhostTrafficController::Config gcfg;
    gcfg.n_max               = 100;    // ignored (density supersedes)
    gcfg.total_steps         = total_steps;
    gcfg.window_steps        = 8;
    // Per-city PROPORTIONAL ghost setup — matches Option Y/Q strong-eval.
    // 10% hot ways per city → uniform agent route coverage across scales.
    // density 2.0 → n_max derived per city, preserves per-edge BPR profile.
    gcfg.hot_way_count       = 0;      // disable absolute count
    gcfg.hot_way_fraction    = 0.05f;  // 5% — matches Y/Q proportional config
    gcfg.density_per_hot_way = 1.4f;   // ↓ from 2.0 (Option O reduction, −30%)
    gcfg.profile             = spec.profile;

    // Same seed across scenarios so the same hot-way SAMPLES are reused -- the
    // only difference scenario-to-scenario is the temporal target_count shape.
    ghost.reset(geo_box, cmap, gcfg, 4242u);
    out.n_hot_ways = ghost.n_hot_ways();
    out.n_max      = ghost.n_max();   // read back density-derived value

    const auto t0 = std::chrono::steady_clock::now();
    // Sample every 50 steps to keep output bounded.
    constexpr int kSampleStride = 50;
    std::vector<Sample> samples;
    samples.reserve(total_steps / kSampleStride + 1);

    out.peak_load_episode   = 0;
    out.peak_step           = 0;
    out.max_overlap_episode = 0;
    double mean_acc = 0.0;
    int    n_acc    = 0;

    for (int step = 0; step < total_steps; ++step) {
        cmap.advance(step);
        ghost.step(step);

        if (step % kSampleStride == 0) {
            Sample s;
            s.step             = step;
            s.peak_load        = cmap.peak_load_now();
            s.mean_load        = cmap.mean_load_now();
            s.n_overlap        = cmap.n_edges_load_ge(2);
            s.n_jam            = cmap.n_edges_load_ge(5);
            s.n_heavy          = cmap.n_edges_load_ge(10);
            s.n_active_ghosts  = ghost.n_active_now();
            samples.push_back(s);
            if (s.peak_load > out.peak_load_episode) {
                out.peak_load_episode = s.peak_load;
                out.peak_step         = s.step;
                out.n_overlap_at_peak = s.n_overlap;
                out.n_jam_at_peak     = s.n_jam;
                out.n_heavy_at_peak   = s.n_heavy;
            }
            if (s.n_overlap > out.max_overlap_episode)
                out.max_overlap_episode = s.n_overlap;
            mean_acc += s.mean_load;
            ++n_acc;
        }
    }
    out.mean_load_episode  = (n_acc > 0) ? static_cast<float>(mean_acc / n_acc) : 0.f;
    out.bpr_factor_typical = bpr_factor(out.peak_load_episode, 100.f, cmap.params);

    const auto t1 = std::chrono::steady_clock::now();
    out.ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return out;
}

void print_header() {
    std::cout << "  "
              << std::left
              << std::setw(16) << "scenario"
              << std::right
              << std::setw(7)  << "n_max"
              << std::setw(9)  << "hot_w"
              << std::setw(7)  << "peak"
              << std::setw(11) << "peak_step"
              << std::setw(10) << "mean"
              << std::setw(10) << "overlap"
              << std::setw(8)  << "jam"
              << std::setw(8)  << "heavy"
              << std::setw(12) << "BPR@peak"
              << std::setw(8)  << "ms"
              << "\n";
}

void print_row(const ScenarioRun& r) {
    std::cout << "  "
              << std::left
              << std::setw(16) << r.scenario
              << std::right
              << std::setw(7)  << r.n_max
              << std::setw(9)  << r.n_hot_ways
              << std::setw(7)  << r.peak_load_episode
              << std::setw(11) << r.peak_step
              << std::setw(10) << std::fixed << std::setprecision(4) << r.mean_load_episode
              << std::setw(10) << r.n_overlap_at_peak
              << std::setw(8)  << r.n_jam_at_peak
              << std::setw(8)  << r.n_heavy_at_peak
              << "    x" << std::fixed << std::setprecision(3) << r.bpr_factor_typical
              << std::setw(8)  << r.ms
              << "\n";
}

} // namespace

bool run_congestion_only_test(const std::string& osm_root,
                              const std::string& cache_root)
{
    std::cout << "\n=== Congestion-only Test (no agents, ghost traffic only) ===\n";

    CityRegistry::set_osm_root(osm_root);

    // 6 cities = Option Y eval scope (train + held-out, London dropped).
    const std::vector<std::string> city_names = {
        "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small",
        "Tokyo_Large", "Paris", "NewYork"
    };

    // Mirror Option Y's eval_scenarios shape (over_fleet uses the same profile
    // as normal_wave for congestion -- the difference is at fleet level, which
    // this test does not exercise -- so we still report it as a check).
    const std::vector<ScenarioSpec> scenarios = {
        {"normal_wave",    CongestionProfile::Wave,        -1},
        {"normal_ramp",    CongestionProfile::RampUpDown,  -1},
        {"stress_shock",   CongestionProfile::ShockBurst,  -1},
        {"stress_buildup", CongestionProfile::BuildingUp,  -1},
        {"over_fleet",     CongestionProfile::Wave,        -1},
    };

    constexpr int kTotalSteps = 3600;

    long long total_ms = 0;
    int       total_runs = 0;

    // Aggregate worst-case across cities for the final analysis section.
    int   global_max_peak_load = 0;
    int   global_max_jam       = 0;
    float global_min_bpr       = 1.f;
    float global_max_bpr       = 1.f;

    for (const auto& name : city_names) {
        const CityConfig* cc = nullptr;
        for (const auto& c : CityRegistry::all())
            if (c.name == name) { cc = &c; break; }
        if (!cc) {
            std::cout << "  [skip] " << name << " - not in CityRegistry\n";
            continue;
        }

        const std::string cache_path = cache_root + "/" + name + ".json";
        if (!GeoBoxManager::cache_exists(cache_path)) {
            std::cout << "  [skip] " << name << " -- cache not found at " << cache_path << "\n";
            continue;
        }
        GeoBox gb = GeoBoxManager::load_geobox(cache_path);
        if (!gb.is_valid || gb.data.nodes.empty() || gb.data.ways.empty()) {
            std::cout << "  [skip] " << name << " -- invalid GeoBox\n";
            continue;
        }

        const int n_max_city = ghost_n_max_for(*cc);
        std::cout << "\n[" << name << "  area=" << cc->area_km2
                  << " km2  nodes=" << gb.data.nodes.size()
                  << "  ways=" << gb.data.ways.size()
                  << "  n_max=" << n_max_city << "]\n";
        print_header();

        for (const auto& sc : scenarios) {
            ScenarioRun r = simulate_scenario(gb, sc, n_max_city, kTotalSteps);
            print_row(r);
            total_ms += r.ms;
            ++total_runs;
            if (r.peak_load_episode > global_max_peak_load)
                global_max_peak_load = r.peak_load_episode;
            if (r.n_jam_at_peak > global_max_jam) global_max_jam = r.n_jam_at_peak;
            if (r.bpr_factor_typical > global_max_bpr) global_max_bpr = r.bpr_factor_typical;
            if (r.bpr_factor_typical < global_min_bpr) global_min_bpr = r.bpr_factor_typical;
        }
    }

    std::cout << "\n=== Summary across " << total_runs << " (city x scenario) runs ===\n";
    std::cout << "  Global peak load on any single edge   : " << global_max_peak_load << "\n";
    std::cout << "  Worst single-step jam (edges ≥5 load) : " << global_max_jam << "\n";
    std::cout << "  BPR factor range at peak (100m way)   : x" << std::fixed
              << std::setprecision(3) << global_min_bpr << "  ->  x"
              << global_max_bpr << "\n";
    std::cout << "  Wallclock for all " << total_runs
              << " ghost simulations            : " << total_ms << " ms\n";
    std::cout << "\n  Interpretation:\n";
    std::cout << "    BPRx1.000-x1.010   -> congestion is cosmetic, no real cost impact\n";
    std::cout << "    BPRx1.010-x1.100   -> mild slowdown, marginal differentiation\n";
    std::cout << "    BPRx1.100-x2.000   -> meaningful slowdown, baselines should differentiate\n";
    std::cout << "    BPRx2.000+         -> strong jam, dynamic-cost baselines should clearly win\n";
    std::cout << "\n=== DONE ===\n\n";
    return true;
}