#include "Tests/RegressionTests.hpp"
#include "Tests/TestSupport.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/StructuresParam/ScenarioConfig.hpp"
#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "SoTA/SolverFramework.hpp"
#include "SoTA/Standalone/CA.hpp"
#include "SoTA/Standalone/HAPC.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <cmath>
#include <iostream>

// Golden values are pinned for the current build (fixed config + seed). They
// MUST be updated deliberately whenever a real behaviour change lands; an
// unintended drift in the simulation / structures / workflows fails here.
static EpisodeConfig regression_config() {
    EpisodeConfig cfg;
    cfg.speed_mps           = 12.f;
    cfg.min_task_dist_m     = 50.f;
    cfg.max_task_dist_m     = 600.f;
    cfg.cluster_prob        = 0.3f;
    cfg.n_hot_zones         = 2;
    cfg.hot_zone_radius     = 400.f;
    cfg.max_tasks_per_agent = 3;
    cfg.phases = { { 300, 15.f, 3, 3, 0.0f, 2 } };
    return cfg;
}

static constexpr uint32_t kSeed = 20240622u;
static bool close(float a, float b) { return std::fabs(a - b) < 1e-3f; }

static ComparisonMetrics run_rl(GeoBox& gb, Pathfinder& pf, PolicyMode m) {
    EpisodeConfig cfg = regression_config();
    bid_policy(BidPolicyKind::MAPPO).clear_buffers();
    EpisodeRunner runner(cfg, gb, pf, 7u);
    runner.train_mode  = false;
    runner.policy_mode = m;
    ComparisonMetrics r = runner.run(0, 1, {}, kSeed).metrics;
    bid_policy(BidPolicyKind::MAPPO).clear_buffers();
    return r;
}

bool run_regression_tests(const std::string& osm_file, const std::string& cache_dir)
{
    std::cout << "\n=== R. Regression (golden values) ===\n";
    GeoBox gb = load_smoke_geobox(osm_file, cache_dir);
    Pathfinder pf(gb);

    // ── [1] Structure ─────────────────────────────────────────────────────────
    CHECK(gb.data.nodes.size() == 33179, "smoke GeoBox node count drifted");
    CHECK(gb.data.ways.size()  == 39877, "smoke GeoBox way count drifted");
    std::cout << "  [1] Structure OK — 33179 nodes / 39877 ways\n";

    // ── [2] Scenario grid ─────────────────────────────────────────────────────
    const auto grid = make_scenario_grid();
    CHECK(grid.size() == 9, "scenario grid size drifted");
    std::cout << "  [2] Scenario grid OK — 9 combos\n";

    // ── [3] RL pipeline (deterministic episode → golden metrics) ──────────────
    auto check_rl = [&](const char* label, PolicyMode m,
                        int g_app, int g_comp, float g_thr, float g_cong) {
        const auto a = run_rl(gb, pf, m);
        const auto b = run_rl(gb, pf, m);
        CHECK(a.tasks_appeared == b.tasks_appeared && a.tasks_completed == b.tasks_completed
           && close(a.throughput_rate, b.throughput_rate),
              "RL run not deterministic across two identical runs");
        CHECK(a.pairing_violations_runtime == 0 && a.capacity_violations_runtime == 0,
              "RL run produced PDP/capacity violations");
        CHECK(a.tasks_appeared == g_app,   "tasks_appeared drifted");
        CHECK(a.tasks_completed == g_comp, "tasks_completed drifted");
        CHECK(close(a.throughput_rate, g_thr),  "throughput drifted");
        CHECK(close(a.mean_congestion, g_cong), "mean_congestion drifted");
        std::cout << "  [3] RL " << label << " OK — appeared=" << a.tasks_appeared
                  << " completed=" << a.tasks_completed << " thr=" << a.throughput_rate << "\n";
        return true;
    };
    CHECK(check_rl("TAA ", PolicyMode::TamAlwaysAccept, 35, 4, 0.114286f, 0.980607f), "TAA regression");
    CHECK(check_rl("RMCA", PolicyMode::RMCA,            35, 4, 0.114286f, 0.980607f), "RMCA regression");

    // ── [4] SoTA pipeline (deterministic SolverRunner → golden metrics) ───────
    auto check_sota = [&](const char* label, ISolver& s, int g_app, int g_comp, float g_thr) {
        EpisodeConfig cfg = regression_config();
        SolverRunner r1(cfg, gb, pf, {}, kSeed); const auto a = r1.run(s);
        SolverRunner r2(cfg, gb, pf, {}, kSeed); const auto b = r2.run(s);
        CHECK(a.tasks_appeared == b.tasks_appeared && a.tasks_completed == b.tasks_completed,
              "SoTA run not deterministic across two identical runs");
        CHECK(a.capacity_violations == 0 && a.pairing_violations == 0,
              "SoTA run produced capacity/pairing violations");
        CHECK(a.tasks_appeared == g_app,   "SoTA tasks_appeared drifted");
        CHECK(a.tasks_completed == g_comp, "SoTA tasks_completed drifted");
        CHECK(close(a.throughput_rate, g_thr), "SoTA throughput drifted");
        std::cout << "  [4] SoTA " << label << " OK — appeared=" << a.tasks_appeared
                  << " completed=" << a.tasks_completed << "\n";
        return true;
    };
    { FaithfulCASolver               s; CHECK(check_sota("CA  ", s, 35, 0, 0.f),        "CA regression"); }
    { HybridAdaptivePredictiveSolver s; CHECK(check_sota("HAPC", s, 35, 4, 0.114286f), "HAPC regression"); }

    std::cout << "=== R PASS ===\n";
    return true;
}
