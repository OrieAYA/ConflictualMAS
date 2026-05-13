#include "MultiCityTrainer.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include <filesystem>
#include <iostream>
#include <random>

namespace fs = std::filesystem;

// ── City loading ──────────────────────────────────────────────────────────────

// Scale per-city episode params so task duration is short relative to episode
// length — needed so agents don't stay saturated and the policy gets enough
// experience. Adapted from city area (km²).
static void customize_episode_for_city(EpisodeConfig& ep, const CityConfig& cc) {
    const double a = cc.area_km2;
    // Tier thresholds match Tokyo_Small / Tokyo_Medium / Tokyo_Large areas.
    // Multi-task FIFO queue per agent — safe after the receive_task() reorder
    // fix. Bigger graph ⇒ deeper queue so agents stay busy across long trips.
    if (a < 50.0) {
        // Small (~25 km²): short trips, dense flow.
        ep.phases = {
            { 1000,  60.f, 4,  5, 0.0f, 3 },
            { 1500, 100.f, 5,  7, 0.5f, 4 },
            { 1100, 140.f, 7,  8, 1.0f, 5 },
        };
        ep.min_task_dist_m = 150.f;
        ep.max_task_dist_m = 1500.f;
        ep.hot_zone_radius = 300.f;
        ep.max_tasks_per_agent = 5;
    } else if (a < 300.0) {
        // Medium (~144 km²): mixed scale.
        ep.phases = {
            { 1000, 100.f,  6,  8, 0.0f, 4 },
            { 1500, 150.f,  8, 10, 0.5f, 6 },
            { 1100, 200.f, 10, 12, 1.0f, 7 },
        };
        ep.min_task_dist_m = 250.f;
        ep.max_task_dist_m = 3500.f;
        ep.hot_zone_radius = 500.f;
        ep.max_tasks_per_agent = 6;
    } else {
        // Large (≥300 km²): Amazon-scale density, long task queues.
        ep.phases = {
            { 1000, 120.f,  8, 10, 0.0f, 4 },
            { 1500, 200.f, 10, 13, 0.5f, 6 },
            { 1100, 280.f, 13, 15, 1.0f, 8 },
        };
        ep.min_task_dist_m = 400.f;
        ep.max_task_dist_m = 5000.f;
        ep.hot_zone_radius = 800.f;
        ep.max_tasks_per_agent = 8;
    }
}

std::unique_ptr<CityAssets> MultiCityTrainer::load_city(
    const CityConfig& cc, int idx, EpisodeConfig ep,
    const std::string& cache_root)
{
    ep.city = &cc;
    customize_episode_for_city(ep, cc);

    GeoBox gb;
    const std::string cache_path = cache_root + "/" + cc.name + ".json";

    if (GeoBoxManager::cache_exists(cache_path)) {
        std::cout << "  [Load] " << cc.name << " <- cache\n";
        gb = GeoBoxManager::load_geobox(cache_path);
    } else {
        std::cout << "  [Load] " << cc.name << " <- OSM (" << cc.osm_path << ")\n";
        gb = create_geo_box(cc.osm_path,
                            cc.bbox.min_lon, cc.bbox.min_lat,
                            cc.bbox.max_lon, cc.bbox.max_lat);
        if (gb.is_valid) {
            fs::create_directories(cache_root);
            GeoBoxManager::save_geobox(gb, cache_path);
        }
    }

    if (!gb.is_valid)
        throw std::runtime_error("MultiCityTrainer: failed to load city " + cc.name);

    return std::make_unique<CityAssets>(&cc, idx, std::move(ep), std::move(gb));
}

// ── Evaluation ────────────────────────────────────────────────────────────────

int MultiCityTrainer::run_eval(
    const TrainingConfig& cfg,
    const std::vector<std::unique_ptr<CityAssets>>& assets,
    std::vector<std::unique_ptr<EpisodeRunner>>& runners,
    int global_ep, int seed,
    TrainingLogger& logger)
{
    const int num_cities = static_cast<int>(assets.size());

    static constexpr std::array<PolicyMode, 3> k_modes = {
        PolicyMode::MAPPO, PolicyMode::Greedy, PolicyMode::Random };
    static constexpr std::array<const char*, 3> k_mode_strs = {
        "MAPPO", "Greedy", "Random" };

    for (int ci = 0; ci < num_cities; ++ci) {
        const CityAssets& ca     = *assets[ci];
        EpisodeRunner&    runner = *runners[ci];

        runner.train_mode = false;

        for (int mi = 0; mi < 3; ++mi) {
            runner.policy_mode = k_modes[mi];
            for (int e = 0; e < cfg.n_eval_episodes; ++e) {
                RunResult res = runner.run(ca.index, num_cities);
                EpisodeRecord rec = make_record(
                    res, seed, global_ep++,
                    ca.config->name, "eval", k_mode_strs[mi],
                    ca.ep_cfg.max_agents());
                logger.push(rec);
            }
        }
    }

    return global_ep;
}

// ── Main training loop ────────────────────────────────────────────────────────

void MultiCityTrainer::train(const TrainingConfig& cfg) {
    fs::create_directories(cfg.output_dir);

    // ── 1. Load train cities only ─────────────────────────────────────────
    const auto train_ptrs = CityRegistry::train_cities();
    const int num_cities  = static_cast<int>(train_ptrs.size());

    std::cout << "Loading " << num_cities << " train cities from " << cfg.cache_root << "\n";
    std::vector<std::unique_ptr<CityAssets>> assets;
    assets.reserve(num_cities);
    for (int i = 0; i < num_cities; ++i)
        assets.push_back(load_city(*train_ptrs[i], i, cfg.episode_cfg, cfg.cache_root));
    std::cout << "All cities loaded.\n\n";

    // ── 2. All loaded cities are train cities ─────────────────────────────
    std::vector<int> train_indices;
    for (int i = 0; i < num_cities; ++i)
        train_indices.push_back(i);

    // ── 3. Multi-seed loop ────────────────────────────────────────────────
    const std::string summary_path = cfg.output_dir + "/summary.csv";

    for (int s = 0; s < cfg.n_seeds; ++s) {
        const int seed = s + 42;
        std::cout << "══════════════════════════════════════\n"
                  << " Seed " << s << "  (rng=" << seed << ")\n"
                  << "══════════════════════════════════════\n";

        // Re-initialise MAPPO weights for this seed.
        std::mt19937 init_rng(static_cast<uint32_t>(seed));
        auto& policy = ObjectiveDMPolicy::shared();
        policy.actor.init_xavier(init_rng);
        policy.critic.init_xavier(init_rng);
        policy.clear_buffer();

        if (cfg.load_policy && !cfg.policy_path.empty()) {
            if (!policy.load(cfg.policy_path))
                std::cerr << "[Warn] Could not load policy from " << cfg.policy_path << "\n";
            else
                std::cout << "[Policy] Loaded from " << cfg.policy_path << "\n";
        }

        // Create per-city runners. Reused across rounds so the A* cache warms up.
        std::vector<std::unique_ptr<EpisodeRunner>> runners;
        runners.reserve(num_cities);
        for (int i = 0; i < num_cities; ++i) {
            CityAssets& ca = *assets[i];
            runners.push_back(std::make_unique<EpisodeRunner>(
                ca.ep_cfg, ca.geo_box, ca.pathfinder,
                static_cast<uint32_t>(seed)));
        }

        TrainingLogger logger(cfg.output_dir, seed);
        int global_ep = 0;

        // ── 4. Training rounds ────────────────────────────────────────────
        for (int round = 0; round < cfg.n_rounds; ++round) {

            for (int ci : train_indices) {
                CityAssets&    ca     = *assets[ci];
                EpisodeRunner& runner = *runners[ci];
                runner.train_mode  = true;
                runner.policy_mode = PolicyMode::MAPPO;

                RunResult     res = runner.run(ca.index, num_cities);
                EpisodeRecord rec = make_record(
                    res, seed, global_ep++,
                    ca.config->name, "train", "MAPPO",
                    ca.ep_cfg.max_agents());
                logger.push(rec);

                if (cfg.verbose && global_ep % cfg.log_every == 0) {
                    std::cout << "  [s" << s << " r" << round
                              << " " << ca.config->name << "]"
                              << "  thr="   << res.metrics.throughput_rate
                              << "  acc="   << res.metrics.accept_rate
                              << "  aloss=" << res.train_stats.actor_loss
                              << "  closs=" << res.train_stats.critic_loss
                              << "  ent="   << res.train_stats.entropy
                              << "  kl="    << res.train_stats.kl_approx
                              << "  cf="    << res.train_stats.clip_frac
                              << "  ep="    << res.train_stats.n_epochs
                                            << "/" << policy.hparams.epochs
                              << "  n="     << res.train_stats.n_exp
                              << "  "       << res.wallclock_ms << "ms\n";
                }
            }

            // Periodic eval (after completing the full round).
            if ((round + 1) % cfg.eval_every == 0) {
                std::cout << "  -- Eval @ round " << (round + 1) << " --\n";
                global_ep = run_eval(cfg, assets, runners, global_ep, seed, logger);
                logger.flush();
            }
        }

        // ── 5. Final evaluation ───────────────────────────────────────────
        std::cout << "  -- Final Eval --\n";
        global_ep = run_eval(cfg, assets, runners, global_ep, seed, logger);

        // ── 6. Policy checkpoint ──────────────────────────────────────────
        if (cfg.save_policy) {
            const std::string ckpt = cfg.output_dir + "/policy_seed"
                                   + std::to_string(seed) + ".bin";
            policy.save(ckpt);
            std::cout << "  [Checkpoint] saved to " << ckpt << "\n";
        }

        // ── 7. Per-seed summary ───────────────────────────────────────────
        logger.flush();
        TrainingLogger::write_summary(summary_path, logger.records(), seed);

        std::cout << "Seed " << s << " done — " << global_ep << " episodes.\n\n";
    }

    std::cout << "Training complete. Results in " << cfg.output_dir << "\n";
}
