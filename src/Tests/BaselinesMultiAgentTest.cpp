#include "BaselinesMultiAgentTest.hpp"
#include "Training/EpisodeConfig.hpp"
#include "Training/EpisodeRunner.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Environment/Congestion/GhostTrafficController.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// ── Soft-check infrastructure (same pattern as TamMcTest) ──────────────────
static bool g_any_fail = false;
#define SOFT_CHECK(cond, msg)                                                   \
    do { if (!(cond)) { std::cout << "  [FAIL] " msg "\n"; g_any_fail = true; } } while (0)
#define HARD_CHECK(cond, msg)                                                   \
    do { if (!(cond)) { std::cout << "  [FAIL] " msg "\n"; return false; } } while (0)

static bool finite01(float v) { return std::isfinite(v) && v >= 0.f && v <= 1.f; }

namespace {

const char* mode_name(PolicyMode m) {
    switch (m) {
        case PolicyMode::Greedy:          return "Greedy";
        case PolicyMode::InsertionGreedy: return "InsertionGreedy";
        case PolicyMode::LaCAM:           return "LaCAM";
        case PolicyMode::PIBT:            return "PIBT";
        case PolicyMode::CongestionAware: return "CongestionAware";
        case PolicyMode::MCA:             return "MCA";
        case PolicyMode::TrafficFlow:     return "TrafficFlow";
        case PolicyMode::TokenPassing:    return "TokenPassing";
        case PolicyMode::DoubleHorizon:   return "DoubleHorizon";
        default:                          return "Unknown";
    }
}

struct OneRun {
    bool  built       = false;
    bool  ran         = false;
    int   appeared    = 0;
    int   completed   = 0;
    float accept_rate = 0.f;
    float throughput  = 0.f;
    int   cap_viol    = 0;
    int   pair_viol   = 0;
    int   peak_cong   = 0;
    float ghost_act   = 0.f;
    long long ms      = 0;
};

// Build an EpisodeConfig sized for a small fast multi-agent capacitated
// lifelong scenario, with ghost congestion ON. Same shape across every
// baseline so the comparison is apples-to-apples (homogeneity guarantee).
EpisodeConfig make_cfg(bool heterogeneous_capacity) {
    EpisodeConfig cfg;
    cfg.speed_mps          = 5.f;
    cfg.min_task_dist_m    = 80.f;
    cfg.max_task_dist_m    = 3000.f;
    cfg.cluster_prob       = 0.35f;
    cfg.n_hot_zones        = 3;
    cfg.hot_zone_radius    = 400.f;
    cfg.max_tasks_per_agent = 3;   // capacitated multi-package agent

    cfg.phases = {
        { 200, 8.f, 4, 5, 0.0f, 2 },
        { 200, 10.f, 5, 6, 1.0f, 3 },
    };

    // Congestion: Wave profile so we hit non-trivial dynamic costs for
    // CongestionAware/TrafficFlow during the same windows in every baseline.
    cfg.enable_ghost_traffic = true;
    cfg.ghost_n_max          = 20;
    cfg.ghost_window_steps   = 5;
    cfg.ghost_hot_way_frac   = 0.30f;

    // No depot — agents stay where they delivered (default).
    cfg.use_depot = false;

    if (heterogeneous_capacity) {
        cfg.enable_heterogeneous_capacity = true;
        cfg.hetero_capacity_min = 2;
        cfg.hetero_capacity_max = 4;
    }
    return cfg;
}

OneRun run_one(EpisodeRunner& runner, PolicyMode mode, uint32_t ep_seed,
               EpisodeScenario sc) {
    OneRun out;
    out.built = true;
    runner.train_mode  = false;
    runner.policy_mode = mode;
    try {
        // Clear ALL learning buffers — we are eval-only and pollution from
        // a previous run on the same singleton must not bias accept/refuse
        // counters here. (EpisodeRunner::run() already clears them when
        // train_mode=false, but be explicit for posterity.)
        ObjectiveDMPolicy::shared().clear_buffer();
        RunResult r = runner.run(0, 1, sc, ep_seed);
        out.ran         = true;
        out.appeared    = r.metrics.tasks_appeared;
        out.completed   = r.metrics.tasks_completed;
        out.accept_rate = r.metrics.accept_rate;
        out.throughput  = r.metrics.throughput_rate;
        out.cap_viol    = r.metrics.capacity_violations_runtime;
        out.pair_viol   = r.metrics.pairing_violations_runtime;
        out.peak_cong   = r.metrics.peak_congestion;
        out.ghost_act   = r.metrics.n_ghost_active_mean;
        out.ms          = r.wallclock_ms;
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] run_one(" << mode_name(mode)
                  << ") threw: " << e.what() << "\n";
        g_any_fail = true;
    }
    return out;
}

void print_row(PolicyMode m, const OneRun& r) {
    std::cout << "    "
              << std::setw(16) << std::left  << mode_name(m)
              << "  appeared=" << std::setw(3) << r.appeared
              << "  completed=" << std::setw(3) << r.completed
              << "  acc="       << std::fixed << std::setprecision(2) << r.accept_rate
              << "  thr="       << std::fixed << std::setprecision(2) << r.throughput
              << "  peak_c="    << std::setw(2) << r.peak_cong
              << "  ghost~"     << std::fixed << std::setprecision(1) << r.ghost_act
              << "  cap_v="     << r.cap_viol
              << "  pair_v="    << r.pair_viol
              << "  ms="        << r.ms
              << "\n";
}

void check_metrics(PolicyMode m, const OneRun& r) {
    const char* n = mode_name(m);
    SOFT_CHECK(r.ran,                       "run did not complete");
    SOFT_CHECK(finite01(r.accept_rate),     "accept_rate out of [0,1] / NaN");
    SOFT_CHECK(finite01(r.throughput),      "throughput out of [0,1] / NaN");
    SOFT_CHECK(r.appeared > 0,              "tasks_appeared = 0 (scenario vacuous)");
    SOFT_CHECK(r.cap_viol  == 0,            "capacity_violations_runtime > 0");
    SOFT_CHECK(r.pair_viol == 0,            "pairing_violations_runtime > 0");
    (void)n;
}

} // namespace

bool run_baselines_multi_agent_test(const std::string& osm_file,
                                    const std::string& cache_dir) {
    std::cout << "\n=== Baselines Multi-Agent (LGPDP) Test ===\n";
    g_any_fail = false;

    // ── 1. GeoBox: reuse the smoke_test cache (small central Tokyo). ──────
    const std::string cache_path = cache_dir + "/smoke_test.json";
    GeoBox geo_box;
    if (GeoBoxManager::cache_exists(cache_path)) {
        std::cout << "  [1] GeoBox <- cache\n";
        geo_box = GeoBoxManager::load_geobox(cache_path);
    } else {
        std::cout << "  [1] GeoBox <- OSM (first run; will be cached)\n";
        geo_box = create_geo_box(osm_file,
                                 139.695, 35.670,
                                 139.740, 35.710);
        if (geo_box.is_valid)
            GeoBoxManager::save_geobox(geo_box, cache_path);
    }
    HARD_CHECK(geo_box.is_valid,            "GeoBox invalid");
    HARD_CHECK(!geo_box.data.nodes.empty(), "GeoBox has no nodes");
    std::cout << "  [1] OK — " << geo_box.data.nodes.size() << " nodes, "
              << geo_box.data.ways.size() << " ways\n";

    Pathfinder pathfinder(geo_box);

    // Modes we want paper-fidelity coverage on, plus the supporting baselines
    // that share the same offer pipeline (so any pipeline regression breaks
    // them all together).
    const std::vector<PolicyMode> modes = {
        PolicyMode::Greedy,
        PolicyMode::InsertionGreedy,
        PolicyMode::PIBT,
        PolicyMode::LaCAM,
        PolicyMode::CongestionAware,
        PolicyMode::TokenPassing,
        PolicyMode::MCA,
        PolicyMode::TrafficFlow,
        PolicyMode::DoubleHorizon,
    };

    // ── 2. Homogeneous capacity, normal scenario, deterministic seed. ─────
    //
    // Every mode uses the SAME (cfg, seed, scenario) — gen_.reset_seed +
    // ghost-traffic episode_seed are derived from rng_ + scenario.label inside
    // EpisodeRunner::run(), so all modes face IDENTICAL task streams,
    // identical ghost loads, identical hetero-capacity draws. This is the
    // homogeneity guarantee the user asked for: any divergence in metrics
    // below is attributable to the policy, not the environment.
    std::cout << "\n  [2] Homogeneous capacity (3), Wave congestion, seed=4242\n";
    {
        EpisodeConfig cfg = make_cfg(/*hetero*/false);
        EpisodeScenario sc;
        sc.density_mult       = 1.0f;
        sc.agents_mult        = 1.0f;
        sc.label              = "normal";
        sc.congestion_profile = CongestionProfile::Wave;

        std::map<PolicyMode, OneRun> results;
        for (PolicyMode m : modes) {
            // Fresh runner per mode so the agent pool, RNG seed, and per-cfg
            // capacity bookkeeping start from a known state. This is what
            // MultiCityTrainer also does between rounds.
            std::unique_ptr<EpisodeRunner> runner;
            try {
                runner = std::make_unique<EpisodeRunner>(cfg, geo_box, pathfinder, 4242u);
            } catch (const std::exception& e) {
                std::cout << "    [FAIL] " << mode_name(m)
                          << " runner ctor threw: " << e.what() << "\n";
                g_any_fail = true;
                continue;
            }
            OneRun r = run_one(*runner, m, 4242u, sc);
            print_row(m, r);
            check_metrics(m, r);
            results[m] = r;
        }

        // ── Sanity: distinct baselines must produce DIFFERENT decisions on
        // the same environment. If two modes are byte-identical it likely
        // means a copy-paste regression made them collapse onto one branch.
        auto signature = [](const OneRun& r) {
            // (appeared, completed, accept×1000) — captures the allocation
            // and outcome shape without being so strict that benign variance
            // matters. Two same-family baselines (MCA / DoubleHorizon) may
            // legitimately match on a given seed; we only require that the
            // FULL set of 9 baselines doesn't collapse to <3 distinct
            // signatures.
            return std::make_tuple(
                r.appeared, r.completed,
                static_cast<int>(std::round(r.accept_rate * 1000.f)));
        };
        std::map<std::tuple<int,int,int>, int> sig_count;
        for (const auto& [m, r] : results)
            if (r.ran) sig_count[signature(r)]++;
        SOFT_CHECK(sig_count.size() >= 3,
                   "Baselines collapsed onto <3 distinct (appeared, completed, accept) "
                   "signatures — fidelity regression suspected");

        // ── Fidelity-specific contrasts on the SAME episode seed: ─────────
        // - TrafficFlow should not be byte-identical to PIBT (TF now ignores
        //   the load tie-break — that was the bug).
        // - TokenPassing should not be byte-identical to LaCAM (TP uses h
        //   only, LaCAM uses c_pu + c_del).
        // - MCA should not be byte-identical to InsertionGreedy (MCA scans
        //   ALL agents; IG short-circuits on the first profitable bid).
        auto cmp_signatures = [&](PolicyMode a, PolicyMode b, const char* why) {
            auto ia = results.find(a);
            auto ib = results.find(b);
            if (ia == results.end() || ib == results.end()) return;
            if (!ia->second.ran || !ib->second.ran)         return;
            const bool same = signature(ia->second) == signature(ib->second);
            if (same) {
                std::cout << "  [FAIL] " << mode_name(a) << " == " << mode_name(b)
                          << " signature — " << why << "\n";
                g_any_fail = true;
            }
        };
        cmp_signatures(PolicyMode::TrafficFlow, PolicyMode::PIBT,
                       "TrafficFlow now drops the load tie-break, must differ from pure PIBT");
        cmp_signatures(PolicyMode::TokenPassing, PolicyMode::LaCAM,
                       "TP uses h only; LaCAM uses c_pu + c_del — must differ");
        cmp_signatures(PolicyMode::MCA, PolicyMode::InsertionGreedy,
                       "MCA scans all agents; InsertionGreedy short-circuits — must differ");
    }

    // ── 3. Heterogeneous capacity sweep. Verifies the cap-fix in
    //      compute_marginal_cost / compute_double_horizon_cost: with
    //      hetero_capacity ∈ [2,4] and TAM-ceiling 4, MCA and DH must not
    //      pick insertions that overshoot the per-agent cap. The
    //      capacity_violations_runtime counter catches any overshoot.
    std::cout << "\n  [3] Heterogeneous capacity [2..4], Wave congestion, seed=9001\n";
    {
        EpisodeConfig cfg = make_cfg(/*hetero*/true);
        EpisodeScenario sc;
        sc.density_mult       = 1.2f;
        sc.agents_mult        = 1.0f;
        sc.label              = "normal";
        sc.congestion_profile = CongestionProfile::Wave;

        // We only stress the capacity-aware baselines here.
        const std::vector<PolicyMode> cap_modes = {
            PolicyMode::MCA, PolicyMode::DoubleHorizon, PolicyMode::TrafficFlow,
            PolicyMode::CongestionAware, PolicyMode::TokenPassing,
        };
        for (PolicyMode m : cap_modes) {
            std::unique_ptr<EpisodeRunner> runner;
            try {
                runner = std::make_unique<EpisodeRunner>(cfg, geo_box, pathfinder, 9001u);
            } catch (const std::exception& e) {
                std::cout << "    [FAIL] " << mode_name(m)
                          << " runner ctor threw (hetero): " << e.what() << "\n";
                g_any_fail = true;
                continue;
            }
            OneRun r = run_one(*runner, m, 9001u, sc);
            print_row(m, r);
            check_metrics(m, r);
        }
    }

    // ── 4. Stress scenario (over-saturation) — homogeneous cap. ──────────
    //      Verifies baselines don't crash when the system is structurally
    //      over-loaded. throughput < 1.0 is expected; we only check liveness.
    std::cout << "\n  [4] Stress scenario (density×2.0, agents×0.6), ShockBurst, seed=1337\n";
    {
        EpisodeConfig cfg = make_cfg(/*hetero*/false);
        EpisodeScenario sc;
        sc.density_mult       = 2.0f;
        sc.agents_mult        = 0.6f;
        sc.label              = "stress_heavy";
        sc.congestion_profile = CongestionProfile::ShockBurst;

        const std::vector<PolicyMode> stress_modes = {
            PolicyMode::MCA, PolicyMode::TrafficFlow, PolicyMode::TokenPassing,
        };
        for (PolicyMode m : stress_modes) {
            std::unique_ptr<EpisodeRunner> runner;
            try {
                runner = std::make_unique<EpisodeRunner>(cfg, geo_box, pathfinder, 1337u);
            } catch (const std::exception& e) {
                std::cout << "    [FAIL] " << mode_name(m)
                          << " runner ctor threw (stress): " << e.what() << "\n";
                g_any_fail = true;
                continue;
            }
            OneRun r = run_one(*runner, m, 1337u, sc);
            print_row(m, r);
            SOFT_CHECK(r.ran,                   "stress run did not complete");
            SOFT_CHECK(finite01(r.accept_rate), "stress accept_rate out of [0,1]");
            SOFT_CHECK(r.cap_viol == 0,         "stress capacity_violations > 0");
            SOFT_CHECK(r.pair_viol == 0,        "stress pairing_violations > 0");
        }
    }

    if (g_any_fail) {
        std::cout << "\n=== FAIL ===\n\n";
        return false;
    }
    std::cout << "\n=== PASS ===\n\n";
    return true;
}