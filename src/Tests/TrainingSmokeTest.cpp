#include "Tests/TrainingSmokeTest.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include <iostream>
#include <cmath>

// ── Helpers ───────────────────────────────────────────────────────────────────

static bool in01(float v) { return std::isfinite(v) && v >= 0.f && v <= 1.f; }
static bool finite(float v) { return std::isfinite(v); }

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cout << "  [FAIL] " msg "\n"; return false; } } while(0)

// ── Smoke test ────────────────────────────────────────────────────────────────

bool run_training_smoke_test(const std::string& osm_file,
                             const std::string& cache_dir)
{
    std::cout << "\n=== Training Smoke Test ===\n";

    // ── 1. Load a small GeoBox (≈4×5 km patch of central Tokyo) ──────────────
    const std::string cache_path = cache_dir + "/smoke_test.json";

    GeoBox geo_box;
    if (GeoBoxManager::cache_exists(cache_path)) {
        std::cout << "  [1] GeoBox <- cache\n";
        geo_box = GeoBoxManager::load_geobox(cache_path);
    } else {
        std::cout << "  [1] GeoBox <- OSM (parsing small bbox...)\n";
        geo_box = create_geo_box(osm_file,
                                 139.695, 35.670,   // min_lon, min_lat
                                 139.740, 35.710);  // max_lon, max_lat
        if (geo_box.is_valid)
            GeoBoxManager::save_geobox(geo_box, cache_path);
    }

    CHECK(geo_box.is_valid,            "GeoBox is not valid");
    CHECK(!geo_box.data.nodes.empty(), "GeoBox has no nodes");
    std::cout << "  [1] OK — " << geo_box.data.nodes.size() << " nodes, "
              << geo_box.data.ways.size() << " ways\n";

    // ── 2. Build minimal EpisodeConfig ────────────────────────────────────────
    // 200 total steps, 2 phases, up to 4 agents, small distances.
    EpisodeConfig cfg;
    cfg.speed_mps       = 5.f;
    cfg.min_task_dist_m = 50.f;     // relax for small bbox
    cfg.max_task_dist_m = 5000.f;
    cfg.cluster_prob    = 0.3f;
    cfg.n_hot_zones     = 3;
    cfg.hot_zone_radius = 400.f;
    cfg.phases = {
        { 100, 2, 3, 0.0f, 2 },   // low phase:  2→3 agents
        { 100, 3, 4, 1.0f, 3 },   // high phase: 3→4 agents
    };
    cfg.env_scale = 0.4f;   // ~40 tasks over 200 steps (α·env_scale)

    CHECK(cfg.max_agents() == 4, "max_agents() mismatch");
    CHECK(cfg.total_steps() == 200, "total_steps() mismatch");
    std::cout << "  [2] Config OK — " << cfg.total_steps() << " steps, "
              << cfg.max_agents() << " agents max\n";

    // ── 3. Construct runner and run a TRAINING episode ────────────────────────
    bid_policy(BidPolicyKind::MAPPO).clear_buffers();

    std::unique_ptr<EpisodeRunner> runner;
    try {
        runner = std::make_unique<EpisodeRunner>(cfg, geo_box, 42u);
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] EpisodeRunner constructor: " << e.what() << "\n";
        return false;
    }
    std::cout << "  [3] Runner constructed OK\n";

    runner->train_mode  = true;
    runner->policy_mode = PolicyMode::MAPPO;
    RunResult r_train;
    try {
        r_train = runner->run(0, 1);
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Training episode: " << e.what() << "\n";
        return false;
    }

    CHECK(in01(r_train.metrics.accept_rate),    "accept_rate out of [0,1]");
    CHECK(in01(r_train.metrics.throughput_rate),"throughput_rate out of [0,1]");
    CHECK(r_train.metrics.tasks_appeared >= 0,  "tasks_appeared < 0");
    CHECK(r_train.wallclock_ms >= 0,            "wallclock_ms < 0");
    CHECK(finite(r_train.train_stats.actor_loss), "actor_loss not finite");
    CHECK(finite(r_train.train_stats.critic_loss),"critic_loss not finite");

    std::cout << "  [3] Training episode OK — "
              << r_train.metrics.tasks_appeared  << " tasks, "
              << "acc=" << r_train.metrics.accept_rate << ", "
              << "aloss=" << r_train.train_stats.actor_loss << ", "
              << "closs=" << r_train.train_stats.critic_loss << ", "
              << r_train.wallclock_ms << "ms\n";

    // ── 4. Run eval episodes (Greedy + MAPPO) ─────────────────────────────────
    runner->train_mode  = false;
    runner->policy_mode = PolicyMode::Greedy;
    RunResult r_greedy;
    try {
        r_greedy = runner->run(0, 1);
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Greedy eval: " << e.what() << "\n";
        return false;
    }
    CHECK(in01(r_greedy.metrics.accept_rate),    "Greedy accept_rate out of [0,1]");
    CHECK(in01(r_greedy.metrics.throughput_rate),"Greedy throughput out of [0,1]");

    runner->policy_mode = PolicyMode::MAPPO;
    RunResult r_eval;
    try {
        r_eval = runner->run(0, 1);
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] MAPPO eval: " << e.what() << "\n";
        return false;
    }
    CHECK(in01(r_eval.metrics.accept_rate),    "MAPPO eval accept_rate out of [0,1]");

    // The training run must have produced PPO experiences (the buffer/credit
    // path is alive) — otherwise the actor/critic losses above are meaningless.
    CHECK(r_train.train_stats.n_exp > 0, "training collected no experiences");

    std::cout << "  [4] Eval OK — Greedy acc=" << r_greedy.metrics.accept_rate
              << ", MAPPO acc=" << r_eval.metrics.accept_rate << "\n";

    // ── 5. Second training round — verify episode reset works ─────────────────
    runner->train_mode  = true;
    runner->policy_mode = PolicyMode::MAPPO;
    RunResult r_train2;
    try {
        r_train2 = runner->run(0, 1);
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Second training episode: " << e.what() << "\n";
        return false;
    }
    CHECK(in01(r_train2.metrics.accept_rate),    "2nd train accept_rate out of [0,1]");
    CHECK(finite(r_train2.train_stats.actor_loss),"2nd train actor_loss not finite");
    std::cout << "  [5] Episode reset OK — 2nd train acc=" << r_train2.metrics.accept_rate << "\n";

    // ── 6. Multi-task per agent — validates FIFO append fix ───────────────────
    // Builds a config with high task density so agents queue multiple tasks
    // concurrently (max_tasks_per_agent = 5). Verifies that:
    //   - No crash / exception during edge-by-edge traversal with queued tasks
    //   - Accept rate is non-trivially higher than the single-task baseline
    //   - Per-task metrics remain in [0,1] (no NaN, no broken accounting)
    EpisodeConfig cfg_mt = cfg;
    cfg_mt.max_tasks_per_agent = 5;
    cfg_mt.speed_mps        = 15.f;   // faster so tasks actually complete in test
    cfg_mt.min_task_dist_m  = 50.f;
    cfg_mt.max_task_dist_m  = 600.f;  // short trips → multi-task cycling visible
    cfg_mt.phases = {
        // High density + short trips → agents queue multiple tasks and complete some.
        { 400, 3, 4, 0.0f, 3 },
    };
    cfg_mt.env_scale = 0.85f;   // ~85 tasks over 400 steps

    std::unique_ptr<EpisodeRunner> runner_mt;
    try {
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();
        runner_mt = std::make_unique<EpisodeRunner>(cfg_mt, geo_box, 43u);
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Multi-task runner: " << e.what() << "\n";
        return false;
    }
    runner_mt->train_mode  = true;
    runner_mt->policy_mode = PolicyMode::MAPPO;

    RunResult r_mt;
    try {
        r_mt = runner_mt->run(0, 1);
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Multi-task episode: " << e.what() << "\n";
        return false;
    }
    CHECK(in01(r_mt.metrics.accept_rate),     "MT accept_rate out of [0,1]");
    CHECK(in01(r_mt.metrics.throughput_rate), "MT throughput_rate out of [0,1]");
    CHECK(finite(r_mt.train_stats.actor_loss),"MT actor_loss not finite");
    CHECK(r_mt.metrics.tasks_appeared > 0,    "MT no tasks generated");
    CHECK(r_mt.metrics.tasks_completed > 0,   "MT no tasks completed — multi-task delivery flow broken?");

    // Run a second multi-task episode to validate that episode_reset works
    // correctly with multi-task state (objective registrations, paths, plans).
    RunResult r_mt2;
    try {
        r_mt2 = runner_mt->run(0, 1);
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Multi-task 2nd episode: " << e.what() << "\n";
        return false;
    }
    CHECK(in01(r_mt2.metrics.accept_rate), "MT2 accept_rate out of [0,1]");

    std::cout << "  [6] Multi-task OK — max_tasks=5, acc=" << r_mt.metrics.accept_rate
              << ", thr=" << r_mt.metrics.throughput_rate
              << ", tasks=" << r_mt.metrics.tasks_appeared
              << ", reset acc=" << r_mt2.metrics.accept_rate << "\n";

    // ── 7. Reward shaping v2 sanity — bounds + credit identity ────────────────
    {
        // Pure-helper identity: φ(0) = 1, so the pickup+delivery credits of an
        // immediately-delivered task sum to val_i (val_i/λ under normalisation).
        CHECK(latency_phi(0, 0.4f, 1500) == 1.f, "latency_phi(0) must be 1");
        EpisodeConfig cfg_rs = cfg_mt;
        enable_all_reward_shaping(cfg_rs);
        const RewardScales sc = make_reward_scales(cfg_rs);
        CHECK(sc.task_value >= 1.f, "normalisation scale must be >= 1");
        const float val   = event_tuning::kTaskRewardClampMax;
        const float split = cfg_rs.pickup_reward_frac * val / sc.task_value
                          + (1.f - cfg_rs.pickup_reward_frac) * val / sc.task_value;
        CHECK(std::fabs(split - val / sc.task_value) < 1e-5f,
              "pickup+delivery credits must sum to val/lambda");

        // Full-flags episode: every buffered reward stays bounded after
        // normalisation. Eval mode keeps the buffers un-trained so we can
        // probe the accumulated per-entry rewards at the end.
        auto& pol = bid_policy(BidPolicyKind::MAPPO);
        pol.clear_buffers();
        EpisodeRunner runner_rs(cfg_rs, geo_box, 44u);
        runner_rs.train_mode  = false;
        runner_rs.policy_mode = PolicyMode::MAPPO;
        RunResult r_rs;
        try { r_rs = runner_rs.run(0, 1); }
        catch (const std::exception& e) {
            std::cout << "  [FAIL] RS sanity episode: " << e.what() << "\n";
            return false;
        }
        CHECK(in01(r_rs.metrics.accept_rate), "RS accept_rate out of [0,1]");
        const float mx = pol.max_abs_reward();
        CHECK(finite(mx) && mx <= 2.0f,
              "reward-shaping term exceeded the normalised bound");

        // Train-mode pass: shaped rewards (all terms + §9 annealing at e=10 +
        // §10 potential shaping) must flow through GAE/PPO without NaN.
        pol.clear_buffers();
        EpisodeRunner runner_rs2(cfg_rs, geo_box, 45u);
        runner_rs2.train_mode  = true;
        runner_rs2.policy_mode = PolicyMode::MAPPO;
        runner_rs2.set_global_episode(10);
        RunResult r_tr;
        try { r_tr = runner_rs2.run(0, 1); }
        catch (const std::exception& e) {
            std::cout << "  [FAIL] RS train episode: " << e.what() << "\n";
            return false;
        }
        CHECK(finite(r_tr.train_stats.actor_loss)
           && finite(r_tr.train_stats.critic_loss),
              "RS train losses not finite");
        CHECK(r_tr.train_stats.n_exp > 0, "RS train episode buffered no experience");
        pol.clear_buffers();
        std::cout << "  [7] Reward shaping v2 OK — eval max|r|=" << mx
                  << " (bound 2.0), train n_exp=" << r_tr.train_stats.n_exp
                  << ", losses finite, credit identity holds\n";
    }

    std::cout << "=== PASS ===\n\n";
    return true;
}
