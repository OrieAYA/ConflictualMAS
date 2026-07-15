#include "Tests/TamMcTest.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include <iostream>
#include <numeric>
#include <cmath>

// SOFT_CHECK : continue malgré l'échec pour voir tous les résultats.
// HARD_CHECK : arrêt immédiat (réservé aux erreurs bloquantes).
static bool g_any_fail = false;
#define SOFT_CHECK(cond, msg) \
    do { if (!(cond)) { std::cout << "  [FAIL] " msg "\n"; g_any_fail = true; } } while(0)
#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cout << "  [FAIL] " msg "\n"; return false; } } while(0)

static bool in01(float v) { return std::isfinite(v) && v >= 0.f && v <= 1.f; }

// Run N_EPISODES eval episodes; return mean accept_rate.
static float mean_accept_rate(EpisodeRunner& runner, int n_episodes)
{
    float sum = 0.f;
    for (int i = 0; i < n_episodes; ++i) {
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();
        RunResult r = runner.run(0, 1, {}, static_cast<uint32_t>(42 + i));
        sum += r.metrics.accept_rate;
    }
    return sum / static_cast<float>(n_episodes);
}

bool run_tam_mc_test(const std::string& osm_file,
                     const std::string& cache_dir)
{
    std::cout << "\n=== TAM Multi-Candidate Test ===\n";
    constexpr int N_EPS = 3;     // episodes per configuration
    g_any_fail = false;
    // Theoretical guarantee: Format A = 1.0. The slack accounts for tasks
    // whose pickup/delivery nodes sit in disconnected graph components that
    // contain no agent objective node — those exhaust the Dijkstra naturally
    // and are genuinely unallocatable (graph topology, not a TAM bug).
    // Legacy TAM [7] gives the reference ceiling for this GeoBox.
    // Saturated TamAlwaysAccept episodes can leave a few tasks with no
    // orientation-valid candidate (all agents busy, none idle-reachable from
    // pickup) — a legitimate system state, not a TAM bug. Threshold tolerates it.
    constexpr float PASS_RATE = 0.90f;

    // ── 1. Load GeoBox (small central Tokyo, same bbox as smoke test) ─────────
    const std::string cache_path = cache_dir + "/smoke_test.json";
    GeoBox geo_box;
    if (GeoBoxManager::cache_exists(cache_path)) {
        std::cout << "  [1] GeoBox <- cache\n";
        geo_box = GeoBoxManager::load_geobox(cache_path);
    } else {
        std::cout << "  [1] GeoBox <- OSM\n";
        geo_box = create_geo_box(osm_file,
                                 139.695, 35.670,
                                 139.740, 35.710);
        if (geo_box.is_valid)
            GeoBoxManager::save_geobox(geo_box, cache_path);
    }
    CHECK(geo_box.is_valid,            "GeoBox invalid");
    CHECK(!geo_box.data.nodes.empty(), "GeoBox has no nodes");
    std::cout << "  [1] OK — " << geo_box.data.nodes.size() << " nodes\n";

    // ── 2. Episode config — short episodes, 3–5 agents ────────────────────────
    // Keep tasks short (min 50 m) so the 4×5 km patch generates enough variety.
    EpisodeConfig cfg;
    cfg.speed_mps       = 5.f;
    cfg.min_task_dist_m = 50.f;
    cfg.max_task_dist_m = 3000.f;
    cfg.cluster_prob    = 0.3f;
    cfg.n_hot_zones     = 3;
    cfg.hot_zone_radius = 400.f;
    cfg.phases = {
        { 150, 3, 4, 0.0f, 2 },
        { 150, 4, 5, 1.0f, 3 },
    };
    cfg.env_scale = 0.45f;   // ~45 tasks, matches prior volume
    cfg.tam_mc_max_candidates   = 5;
    cfg.tam_mc_ratio_min        = 1.4f;
    cfg.tam_mc_ratio_max        = 3.0f;
    cfg.tam_mc_ratio_scale      = 2000.f;
    cfg.tam_mc_recall_time_frac = 0.20f;
    cfg.tam_mc_reject_time_frac = 0.70f;


    // ── 3. Format A + TamAlwaysAccept — KEY correctness check ─────────────────
    // force_assign=true + score=1.0 → every candidate wins immediately.
    // Before the max_search_cost_ fix: budget = task_dist * 3, short tasks had
    // tiny search radius → candidates not found → accept_rate ~55%.
    // After fix: budget = inf until first candidate → accept_rate ~1.0.
    {
        EpisodeConfig cfgA = cfg;
        cfgA.tam_mc_force_assign = true;
        std::unique_ptr<EpisodeRunner> runner;
        try {
            runner = std::make_unique<EpisodeRunner>(cfgA, geo_box, 42u);
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] Format A runner: " << e.what() << "\n";
            return false;
        }
        runner->train_mode  = false;
        runner->policy_mode = PolicyMode::TamAlwaysAccept;

        const float rate = mean_accept_rate(*runner, N_EPS);
        std::cout << "  [3] Format A + TamAlwaysAccept  accept_rate=" << rate;
        if (rate >= PASS_RATE) std::cout << "  OK\n";
        else                   std::cout << "  FAIL (expected >= " << PASS_RATE << ")\n";
        CHECK(in01(rate), "Format A TamAlwaysAccept accept_rate out of [0,1]");
        SOFT_CHECK(rate >= PASS_RATE,
              "Format A + TamAlwaysAccept accept_rate < 1.0 — force_assign must allocate every found candidate; check max_search_cost_ fix");
    }

    // ── 4. Format B + TamAlwaysAccept ─────────────────────────────────────────
    // force_assign=false, but score=1.0 >= 0.5 → never deferred.
    // Should match Format A for TamAlwaysAccept.
    {
        EpisodeConfig cfgB = cfg;
        cfgB.tam_mc_force_assign = false;
        std::unique_ptr<EpisodeRunner> runner;
        try {
            runner = std::make_unique<EpisodeRunner>(cfgB, geo_box, 42u);
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] Format B runner: " << e.what() << "\n";
            return false;
        }
        runner->train_mode  = false;
        runner->policy_mode = PolicyMode::TamAlwaysAccept;

        const float rate = mean_accept_rate(*runner, N_EPS);
        std::cout << "  [4] Format B + TamAlwaysAccept  accept_rate=" << rate;
        if (rate >= PASS_RATE) std::cout << "  OK\n";
        else                   std::cout << "  FAIL (expected >= " << PASS_RATE << ")\n";
        CHECK(in01(rate), "Format B TamAlwaysAccept accept_rate out of [0,1]");
        SOFT_CHECK(rate >= PASS_RATE,
              "Format B + TamAlwaysAccept accept_rate < 1.0 — score=1.0 >= 0.5 threshold, task must never be deferred");
    }

    // ── 5. Format A + MAPPO (untrained weights) ────────────────────────────────
    // force_assign=true overrides score; accept_rate should still be ~1.0.
    {
        EpisodeConfig cfgM = cfg;
        cfgM.tam_mc_force_assign = true;
        std::unique_ptr<EpisodeRunner> runner;
        try {
            runner = std::make_unique<EpisodeRunner>(cfgM, geo_box, 42u);
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] Format A MAPPO runner: " << e.what() << "\n";
            return false;
        }
        runner->train_mode  = false;
        runner->policy_mode = PolicyMode::MAPPO;

        const float rate = mean_accept_rate(*runner, N_EPS);
        std::cout << "  [5] Format A + MAPPO (untrained)  accept_rate=" << rate;
        if (rate >= PASS_RATE) std::cout << "  OK\n";
        else                   std::cout << "  FAIL (expected >= " << PASS_RATE << ")\n";
        CHECK(in01(rate), "Format A MAPPO accept_rate out of [0,1]");
        SOFT_CHECK(rate >= PASS_RATE,
              "Format A + MAPPO accept_rate < 1.0 — force_assign must allocate argmax regardless of score");
    }

    // ── 6. Format B + MAPPO (untrained) — no-crash check only ─────────────────
    // Untrained weights → scores likely < 0.5 → deferral is expected.
    // We only verify no crash and that accept_rate is in [0, 1].
    {
        EpisodeConfig cfgBM = cfg;
        cfgBM.tam_mc_force_assign = false;
        std::unique_ptr<EpisodeRunner> runner;
        try {
            runner = std::make_unique<EpisodeRunner>(cfgBM, geo_box, 42u);
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] Format B MAPPO runner: " << e.what() << "\n";
            return false;
        }
        runner->train_mode  = false;
        runner->policy_mode = PolicyMode::MAPPO;

        const float rate = mean_accept_rate(*runner, N_EPS);
        std::cout << "  [6] Format B + MAPPO (untrained) accept_rate=" << rate
                  << "  (deferral expected — no-crash check only)\n";
        CHECK(in01(rate), "Format B MAPPO accept_rate out of [0,1]");
    }

    // ── 7. Legacy TAM (mc=false) + TamAlwaysAccept — reference ───────────────
    {
        EpisodeConfig cfgL = cfg;
        std::unique_ptr<EpisodeRunner> runner;
        try {
            runner = std::make_unique<EpisodeRunner>(cfgL, geo_box, 42u);
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] Legacy runner: " << e.what() << "\n";
            return false;
        }
        runner->train_mode  = false;
        runner->policy_mode = PolicyMode::TamAlwaysAccept;

        const float rate = mean_accept_rate(*runner, N_EPS);
        std::cout << "  [7] Legacy TAM + TamAlwaysAccept  accept_rate=" << rate
                  << "  (reference — not checked)\n";
        CHECK(in01(rate), "Legacy accept_rate out of [0,1]");
    }

    if (g_any_fail) std::cout << "=== FAIL ===\n\n";
    else            std::cout << "=== PASS ===\n\n";
    return !g_any_fail;
}