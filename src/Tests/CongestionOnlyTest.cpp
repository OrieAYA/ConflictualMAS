#include "Tests/CongestionOnlyTest.hpp"
#include "TrainingEvaluation/StructuresParam/CityConfig.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "Environment/Simulation/GhostTrafficController.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

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
    int         n_events;
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
    float       bpr_factor_typical;
    // Proportional (network-wide) congestion, averaged over the episode:
    //   net_bpr  = mean over ALL ways of BPR(load, length)  → whole-network slowdown
    //   jam_bpr  = mean over CONGESTED ways of BPR          → severity where it bites
    //   cong_pct = share of ways with load >= 1             → spatial extent
    float       net_bpr;
    float       jam_bpr;
    float       cong_pct;
};

float bpr_factor(int load, float distance_m, const CongestionParams& p) {
    const float cap   = std::max(1.f, distance_m * p.capacity_per_meter);
    const float ratio = static_cast<float>(load) / cap;
    return 1.f + p.bpr_alpha * std::pow(ratio, p.bpr_beta);
}

ScenarioRun simulate_scenario(const GeoBox& geo_box,
                              const std::string& label,
                              TemporalProfile profile,
                              int   n_events,
                              float hot_frac,
                              int   window,
                              int   total_steps)
{
    ScenarioRun out{};
    out.scenario = label;
    out.n_events = n_events;

    CongestionMap cmap;   // default params (BPR α=0.15, β=4, cap=0.05/m)
    GhostTrafficController ghost;

    GhostTrafficController::Config gcfg;
    gcfg.n_events         = out.n_events;
    gcfg.total_steps      = total_steps;
    gcfg.window_steps     = window;
    gcfg.hot_way_count    = 0;
    gcfg.hot_way_fraction = hot_frac;
    gcfg.profile          = profile;

    // Same seed across scenarios so the same hot-way SAMPLES are reused -- the
    // only difference scenario-to-scenario is the temporal fd(x) shape.
    ghost.reset(geo_box, cmap, gcfg, 4242u);
    out.n_hot_ways = ghost.n_hot_ways();
    out.n_events   = ghost.n_events();

    const auto t0 = std::chrono::steady_clock::now();
    // Sample every 50 steps to keep output bounded.
    constexpr int kSampleStride = 300;   // ~12 samples: cheap network-wide BPR scan
    std::vector<Sample> samples;
    samples.reserve(total_steps / kSampleStride + 1);

    out.peak_load_episode   = 0;
    out.peak_step           = 0;
    out.max_overlap_episode = 0;
    double mean_acc = 0.0;
    int    n_acc    = 0;
    double net_bpr_acc = 0.0, jam_bpr_acc = 0.0, cong_pct_acc = 0.0;
    const double n_ways = std::max<size_t>(1, geo_box.data.ways.size());

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

            // Proportional BPR over the whole network at this step.
            double bpr_sum = 0.0; int cong = 0;
            for (const auto& [wid, way] : geo_box.data.ways) {
                const int L = cmap.get_load(wid, step);
                if (L <= 0) continue;
                bpr_sum += bpr_factor(L, way.distance_meters, cmap.params);
                ++cong;
            }
            // Free ways contribute BPR = 1 each to the network mean.
            net_bpr_acc  += (bpr_sum + (n_ways - cong)) / n_ways;
            jam_bpr_acc  += (cong > 0) ? (bpr_sum / cong) : 1.0;
            cong_pct_acc += static_cast<double>(cong) / n_ways;
            ++n_acc;
        }
    }
    out.mean_load_episode  = (n_acc > 0) ? static_cast<float>(mean_acc / n_acc) : 0.f;
    out.bpr_factor_typical = bpr_factor(out.peak_load_episode, 100.f, cmap.params);
    out.net_bpr  = (n_acc > 0) ? static_cast<float>(net_bpr_acc  / n_acc) : 1.f;
    out.jam_bpr  = (n_acc > 0) ? static_cast<float>(jam_bpr_acc  / n_acc) : 1.f;
    out.cong_pct = (n_acc > 0) ? static_cast<float>(cong_pct_acc / n_acc) * 100.f : 0.f;

    const auto t1 = std::chrono::steady_clock::now();
    out.ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return out;
}

void print_header() {
    std::cout << "  "
              << std::left
              << std::setw(16) << "scenario"
              << std::right
              << std::setw(9)  << "n_ev"
              << std::setw(9)  << "hot_w"
              << std::setw(7)  << "peak"
              << std::setw(10) << "netBPR"
              << std::setw(10) << "jamBPR"
              << std::setw(9)  << "cong%"
              << std::setw(8)  << "jam"
              << std::setw(11) << "BPR@peak"
              << std::setw(8)  << "ms"
              << "\n";
}

void print_row(const ScenarioRun& r) {
    std::cout << "  "
              << std::left
              << std::setw(16) << r.scenario
              << std::right
              << std::setw(9)  << r.n_events
              << std::setw(9)  << r.n_hot_ways
              << std::setw(7)  << r.peak_load_episode
              << "   x" << std::fixed << std::setprecision(3) << std::setw(6) << r.net_bpr
              << "   x" << std::setw(6) << r.jam_bpr
              << std::setw(8) << std::setprecision(2) << r.cong_pct << "%"
              << std::setw(8) << r.n_jam_at_peak
              << "     x" << std::setprecision(3) << r.bpr_factor_typical
              << std::setw(8)  << r.ms
              << "\n";
}

} // namespace

bool run_congestion_only_test(const std::string& osm_root,
                              const std::string& cache_root)
{
    std::cout << "\n=== Congestion calibration sweep (count x window, phi_h = 0.40 fixed) ===\n"
              << "Goal: pick (n_events, window) giving proportional BPR at feasible cost.\n"
              << std::flush;

    CityRegistry::set_osm_root(osm_root);

    // Small (SCE 1) + Medium (SCE 2) after the 2×-area rebuild. Per city the
    // ghost count = base · SCE, so we test the production formula directly:
    // consistent netBPR across sizes means count scales with the graph.
    const std::vector<std::string> cities = { "Tokyo_Small", "Tokyo_Medium" };
    constexpr float kPhiH       = event_tuning::kHotWayFraction;   // 0.40 (fixed)
    constexpr int   kTotalSteps = 3600;
    const std::vector<int> bases   = { 15000, 25000, 35000 };  // ghost base (× SCE)
    const int              window  = 250;

    auto sce_of = [](double area) { return area < 40.0 ? 1.f : (area < 90.0 ? 2.f : 5.f); };

    for (const auto& name : cities) {
        const CityConfig* cc = nullptr;
        for (const auto& c : CityRegistry::all())
            if (c.name == name) { cc = &c; break; }
        if (!cc) { std::cout << "  [skip] " << name << " not in registry\n" << std::flush; continue; }

        const std::string cache_path = cache_root + "/" + name + ".json";
        GeoBox gb;
        if (GeoBoxManager::cache_exists(cache_path)) {
            gb = GeoBoxManager::load_geobox(cache_path);
        } else {
            std::cout << "  [build] " << name << " from OSM (" << cc->osm_path << ")...\n" << std::flush;
            gb = create_geo_box(cc->osm_path, cc->bbox.min_lon, cc->bbox.min_lat,
                                 cc->bbox.max_lon, cc->bbox.max_lat);
            if (gb.is_valid) GeoBoxManager::save_geobox(gb, cache_path);
        }
        if (!gb.is_valid || gb.data.ways.empty()) {
            std::cout << "  [skip] " << name << " invalid GeoBox\n" << std::flush; continue;
        }
        const float sce = sce_of(cc->area_km2);
        std::cout << "\n[" << name << "  area=" << cc->area_km2 << " SCE=" << sce
                  << "  ways=" << gb.data.ways.size()
                  << "  phi_h=" << std::fixed << std::setprecision(2) << kPhiH
                  << "  (~" << static_cast<int>(gb.data.ways.size() * kPhiH)
                  << " hot)  window=" << window << "]\n" << std::flush;
        print_header();
        for (int base : bases) {
            const int n = static_cast<int>(base * sce);
            ScenarioRun r = simulate_scenario(
                gb, "base" + std::to_string(base / 1000) + "k", TemporalProfile::Wave,
                n, kPhiH, window, kTotalSteps);
            print_row(r);
            std::cout << std::flush;
        }
    }

    std::cout << "\n=== Reading the sweep ===\n"
              << "  n_ev   = base · SCE (the production ghost count for RM=1).\n"
              << "  netBPR = whole-network mean BPR over time (proportional slowdown).\n"
              << "  Consistent netBPR across Small/Medium at the SAME base = SCE scaling works.\n"
              << "  Pick the base whose netBPR sits in x1.5-2.0 → kCongBaseQtt.\n"
              << "\n=== DONE ===\n\n" << std::flush;
    return true;
}