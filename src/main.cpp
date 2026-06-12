#include "Legacy/Tests/LegacyTests.hpp"
#include "DMASforPD/Tests/PDPTests.hpp"
#include "AmazonDataset/AmazonTest.hpp"
#include "Comparisons/DbVNS_vs_LKH/ComparisonTest.hpp"
#include "Training/MultiCityTrainer.hpp"
#include "Training/CityConfig.hpp"
#include "Training/TrainingConfig.hpp"
#include "Training/SharedEpisodeSetup.hpp"
#include "Training/TrainingSmokeTest.hpp"
#include "Tests/PlanningComparisonTest.hpp"
#include "Tests/TamMcTest.hpp"
#include "Tests/GeoBoxConnectivityTest.hpp"
#include "Tests/CongestionOnlyTest.hpp"

// System-level SoTA baselines (paper: CA = Asadi+2025, HAPC = Cortés+2009).
#include "SoTA/SolverRunner.hpp"
#include "SoTA/SolverCSVLogger.hpp"
#include "SoTA/FaithfulCongestionAware/FaithfulCASolver.hpp"
#include "SoTA/HybridAdaptivePredictive/HybridAdaptivePredictiveSolver.hpp"

#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// Forward-declares the actual main body so a thrown std::runtime_error never
// terminates the process silently.
static int run_main();

int main()
{
    try {
        return run_main();
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] " << e.what() << "\n" << std::flush;
        return 1;
    } catch (...) {
        std::cerr << "\n[FATAL] unknown exception\n" << std::flush;
        return 1;
    }
}

// ── Shared experiment constants ───────────────────────────────────────────────

static const std::string kOsmRoot   = "C:\\ConflictualMAS\\src\\maps";
static const std::string kCacheRoot = "C:\\ConflictualMAS\\data\\cache";
static const std::string kOutputDir = "C:\\ConflictualMAS\\results";

// Evaluation scenarios — paper Table 4. Profiles: Wave = sin bell,
// RampUpDown = "pulse" quarters, ShockBurst = spike, BuildingUp = climbing.
static std::vector<EpisodeScenario> paper_eval_scenarios() {
    return {
        EpisodeScenario{ 1.0f,  1.0f, "wave",           CongestionProfile::Wave       },
        EpisodeScenario{ 1.0f,  1.0f, "pulse",          CongestionProfile::RampUpDown },
        EpisodeScenario{ 1.5f,  0.8f, "stress_shock",   CongestionProfile::ShockBurst },
        EpisodeScenario{ 2.0f,  0.6f, "stress_buildup", CongestionProfile::BuildingUp },
        EpisodeScenario{ 0.5f, 10.0f, "over_fleet",     CongestionProfile::Wave       },
    };
}

// Environment knobs shared by training (T) and evaluation (Y) so there is no
// train/eval distribution shift — paper §5.5.1: DbVNS planning, proportional
// ghost congestion (5% hot ways, density 2/way), heterogeneous capacity [3,7],
// latency shaping + task value heterogeneity, TAM K=5 force-assign,
// fleet ÷2 with load_per_agent=2.
static void apply_paper_environment(EpisodeConfig& ep) {
    ep.use_dbvns_planning            = true;

    ep.enable_ghost_traffic          = true;
    ep.ghost_hot_way_count           = 0;
    ep.ghost_hot_way_frac            = 0.05f;
    ep.ghost_density_per_hot_way     = 2.0f;
    ep.ghost_window_steps            = 8;
    ep.ghost_n_max                   = 100;     // superseded by density scaling
    ep.ghost_n_max_user_set          = false;

    ep.enable_heterogeneous_capacity = true;
    ep.hetero_capacity_min           = 3;
    ep.hetero_capacity_max           = 7;

    ep.enable_latency_shaping        = true;
    ep.latency_shaping_lambda        = 0.4f;
    ep.latency_shaping_max_steps     = 1500;

    ep.enable_task_value_heterogeneity = true;
    ep.task_value_mul_min = 0.5f;
    ep.task_value_mul_max = 2.0f;
    ep.task_imp_min       = 0.3f;
    ep.task_imp_max       = 3.0f;

    ep.tam_mc_force_assign   = true;     // Format A
    ep.tam_mc_max_candidates = 5;
    ep.tam_mc_ratio_min      = 1.4f;
    ep.tam_mc_ratio_max      = 3.0f;
    ep.tam_mc_ratio_scale    = 2000.f;

    ep.fleet_size_divisor    = 2;
    ep.fleet_load_per_agent  = 2;
}

static int run_main()
{
    const std::string osm_file   = "C:\\ConflictualMAS\\src\\maps\\Tokyo.osm.pbf";
    const std::string cache_dir  = "C:\\ConflictualMAS\\src\\geobox_cache_folder";
    const std::string amazon_dir = "C:\\ConflictualMAS\\src\\AmazonDataset\\~\\.rc-cli\\data";

    std::cout << "\n=== ConflictualMAS ===\n\n";
    std::cout << "[Legacy]\n";
    std::cout << "  G  Global Solution Constructor\n";
    std::cout << "  V  VNS Orienteering\n";
    std::cout << "  PSO  PSO MTTDS\n";
    std::cout << "  E  Create GeoBox\n";
    std::cout << "  I  Initialize POI (Flickr)\n";
    std::cout << "  R  Render map\n\n";
    std::cout << "[DMASforPD]\n";
    std::cout << "  D  PDP System test\n";
    std::cout << "  C  Create PDP GeoBox\n\n";
    std::cout << "[Amazon Last Mile]\n";
    std::cout << "  A  DbVNS on Amazon routes (13 routes)\n";
    std::cout << "  B  DbVNS vs LKH benchmark (side-by-side)\n\n";
    std::cout << "[Tests]\n";
    std::cout << "  S  Training smoke test (small bbox, fast)\n";
    std::cout << "  K  GeoBox connectivity check (all caches)\n";
    std::cout << "  J  TAM multi-candidate correctness test\n";
    std::cout << "  Z  Congestion-only test (ghost traffic impact)\n\n";
    std::cout << "[Paper protocol]\n";
    std::cout << "  T  Train MAPPO + IPPO + MAPPER (3 Small cities, 50 rounds)\n";
    std::cout << "  Y  Evaluate — 4 comparison levels (RL+RMCA / TP / CA+HAPC)\n";
    std::cout << "  P  Planning comparison (DbVNS vs ALNS vs DoubleHorizon)\n\n";
    std::cout << "Choice: ";

    std::string rep;
    std::cin >> rep;

    // ── Legacy / PDP / Amazon (unchanged pipelines) ────────────────────────
    if      (rep == "G" || rep == "g") legacy_run_global(cache_dir);
    else if (rep == "V" || rep == "v") legacy_run_vns(cache_dir);
    else if (rep == "PSO" || rep == "pso") legacy_run_pso(cache_dir);
    else if (rep == "E" || rep == "e") legacy_create_geobox(osm_file, cache_dir);
    else if (rep == "I" || rep == "i") legacy_init_poi(cache_dir);
    else if (rep == "R" || rep == "r") legacy_render(cache_dir);
    else if (rep == "D" || rep == "d") test_pdp_system(cache_dir);
    else if (rep == "C" || rep == "c") pdp_create_geobox(osm_file, cache_dir);
    else if (rep == "A" || rep == "a") test_amazon_routes(amazon_dir);
    else if (rep == "B" || rep == "b") test_comparison(amazon_dir);

    // ── Tests ──────────────────────────────────────────────────────────────
    else if (rep == "S" || rep == "s") {
        run_training_smoke_test(osm_file, cache_dir);
    }
    else if (rep == "K" || rep == "k") {
        std::vector<std::string> caches = { cache_dir + "/smoke_test.json" };
        for (const auto& entry : std::filesystem::directory_iterator(kCacheRoot))
            if (entry.path().extension() == ".json")
                caches.push_back(entry.path().string());
        run_geobox_connectivity_test(caches, "all caches");
    }
    else if (rep == "J" || rep == "j") {
        run_tam_mc_test(osm_file, cache_dir);
    }
    else if (rep == "Z" || rep == "z") {
        run_congestion_only_test(kOsmRoot, kCacheRoot);
    }

    // ── T — Training (paper §5.5.1) ────────────────────────────────────────
    // 50 rounds × 3 Small cities × {MAPPO, IPPO, MAPPER}, single seed 42,
    // 3-regime scenario sampler, checkpoints every 10 rounds. Evaluation is
    // a separate run (option Y) on the saved checkpoints.
    else if (rep == "T" || rep == "t") {
        CityRegistry::set_osm_root(kOsmRoot);

        TrainingConfig cfg;
        cfg.cache_root  = kCacheRoot;
        cfg.output_dir  = kOutputDir;
        cfg.n_rounds    = 50;
        cfg.n_seeds     = 1;
        cfg.start_seed  = 42;
        cfg.save_policy = true;
        cfg.verbose     = true;

        cfg.train_only            = true;
        cfg.enable_generalization = false;
        cfg.disable_slack_regime  = true;
        cfg.checkpoint_every_rounds = 10;

        cfg.train_city_filter = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };
        cfg.train_modes = {
            PolicyMode::MAPPO, PolicyMode::IPPO, PolicyMode::MAPPER
        };
        cfg.load_policy = false;     // fresh init per seed

        apply_paper_environment(cfg.episode_cfg);

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }

    // ── Y — Evaluation at the paper's four comparison levels ──────────────
    //
    //   Phase A (EpisodeRunner pipeline — same TAM + DbVNS for everyone):
    //     policy level     : MAPPO, IPPO, MAPPER, Hybrid, RMCA
    //     allocation level : TokenPassing (replaces TAM + policy)
    //     ablation         : TamAlwaysAccept
    //   Phase B (standalone full pipelines via SolverRunner):
    //     system level     : CA [Asadi+2025], HAPC [Cortés+2009]
    //
    //   Both phases consume the SAME SharedEpisodeSetup per (city, scenario,
    //   episode): identical task streams, agent start nodes, capacities and
    //   ghost seeds. ep_seed formula matches MultiCityTrainer::run_eval.
    //
    //   Cities: 3 train Smalls + 2 held-out Smalls (NewYork, Paris).
    //   Scenarios: paper Table 4. Episodes: 2 per (city, scenario).
    else if (rep == "Y" || rep == "y") {
        std::cout << "Seed (42/43/44): ";
        int seed = 42;
        std::cin >> seed;

        CityRegistry::set_osm_root(kOsmRoot);

        TrainingConfig cfg;
        cfg.cache_root      = kCacheRoot;
        cfg.output_dir      = kOutputDir + "\\paper_eval";
        cfg.n_seeds         = 1;
        cfg.start_seed      = seed;
        cfg.n_eval_episodes = 2;
        cfg.verbose         = true;

        cfg.eval_only            = true;
        cfg.skip_stress_eval     = true;     // scenarios already cover stress
        cfg.enable_generalization = true;
        cfg.use_shared_episode_setup = true;

        cfg.train_city_filter = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };
        cfg.gen_city_filter   = { "NewYork_Small", "Paris_Small" };

        cfg.eval_modes = {
            PolicyMode::MAPPO,
            PolicyMode::IPPO,
            PolicyMode::MAPPER,
            PolicyMode::Hybrid,
            PolicyMode::RMCA,
            PolicyMode::TokenPassing,
            PolicyMode::TamAlwaysAccept,
        };
        cfg.eval_scenarios = paper_eval_scenarios();

        apply_paper_environment(cfg.episode_cfg);
        cfg.episode_cfg.agent_pool_multiplier = 10.0f;   // over_fleet head-room

        cfg.load_policy        = true;
        cfg.policy_path        = kOutputDir + "\\mappo\\policy_seed42.bin";
        cfg.ippo_policy_path   = kOutputDir + "\\ippo_faithful\\ippo_seed42.bin";
        cfg.mapper_policy_path = kOutputDir + "\\mapper\\mapper_seed42.bin";

        std::filesystem::create_directories(cfg.output_dir);

        // ── Phase A — RL + RMCA + TP + ablation on the shared pipeline ────
        std::cout << "\n--- Phase A: EpisodeRunner pipeline (7 modes) ---\n";
        {
            MultiCityTrainer trainer;
            trainer.train(cfg);
        }

        // ── Phase B — system-level standalone solvers (CA, HAPC) ──────────
        std::cout << "\n--- Phase B: standalone CA + HAPC ---\n";
        const std::string sota_dir = cfg.output_dir + "\\sota_standalone";
        std::filesystem::create_directories(sota_dir);
        const std::string sota_csv = sota_dir + "\\sota_seed"
                                   + std::to_string(seed) + ".csv";
        SolverCSVLogger logger(sota_csv, /*append=*/false);
        logger.write_header();

        struct CityToRun {
            const CityConfig* cc;
            int               ca_index;     // index within its phase list
        };
        const auto& all_cities = CityRegistry::all();
        auto collect = [&](const std::vector<std::string>& names) {
            std::vector<CityToRun> out;
            for (const auto& want : names)
                for (const auto& cc : all_cities)
                    if (cc.name == want) {
                        out.push_back({ &cc, static_cast<int>(out.size()) });
                        break;
                    }
            return out;
        };

        auto eval_phase = [&](const std::vector<CityToRun>& cities) {
            for (const auto& entry : cities) {
                const CityConfig& cc = *entry.cc;
                std::cout << "\n  City: " << cc.name << "\n";

                const std::string cache_path = kCacheRoot + "/" + cc.name + ".json";
                GeoBox geo_box;
                if (GeoBoxManager::cache_exists(cache_path)) {
                    geo_box = GeoBoxManager::load_geobox(cache_path);
                } else {
                    geo_box = create_geo_box(cc.osm_path,
                                             cc.bbox.min_lon, cc.bbox.min_lat,
                                             cc.bbox.max_lon, cc.bbox.max_lat);
                    if (geo_box.is_valid)
                        GeoBoxManager::save_geobox(geo_box, cache_path);
                }
                if (!geo_box.is_valid) {
                    std::cerr << "    [Skip] GeoBox invalid for " << cc.name << "\n";
                    continue;
                }
                Pathfinder pathfinder(geo_box);

                EpisodeConfig per_city_ep = cfg.episode_cfg;
                per_city_ep.city = &cc;
                MultiCityTrainer::customize_episode_for_city(per_city_ep, cc);

                for (size_t si = 0; si < cfg.eval_scenarios.size(); ++si) {
                    const EpisodeScenario& sc = cfg.eval_scenarios[si];
                    for (int e = 0; e < cfg.n_eval_episodes; ++e) {
                        // EXACT MultiCityTrainer::run_eval seed formula —
                        // Phase A and Phase B see the same task streams.
                        const uint32_t ep_seed = static_cast<uint32_t>(
                            1u + e
                            + 101u * (static_cast<int>(si) + 1)
                            + 10007u * (entry.ca_index + 1)
                            + 1000003u * static_cast<uint32_t>(seed));

                        SolverRunner runner(per_city_ep, geo_box, pathfinder,
                                            sc, ep_seed);
                        const SharedEpisodeSetup setup =
                            build_shared_episode_setup(
                                ep_seed, cc, sc, per_city_ep, geo_box);

                        auto run_and_log = [&](ISolver& s) {
                            SolverMetrics m = runner.run(s, &setup);
                            m.city_label = cc.name;
                            m.episode    = e;
                            logger.write_row(m);
                            std::cout << "    " << cc.name << " | " << sc.label
                                      << " | ep" << e << " | " << m.solver_name
                                      << "  thr=" << m.throughput_rate
                                      << "  lat=" << m.latency_mean
                                      << "  (" << m.wallclock_ms << "ms)\n";
                        };
                        { FaithfulCASolver               s; run_and_log(s); }
                        { HybridAdaptivePredictiveSolver s; run_and_log(s); }
                    }
                }
            }
        };

        eval_phase(collect(cfg.train_city_filter));
        eval_phase(collect(cfg.gen_city_filter));
        logger.close();

        std::cout << "\n=== Evaluation done ===\n"
                  << "  Phase A CSV: " << cfg.output_dir << "\\episodes_seed"
                  << seed << ".csv\n"
                  << "  Phase B CSV: " << sota_csv << "\n"
                  << "  Join keys  : (city, scenario, episode)\n";
    }

    // ── P — Planning-level comparison (paper §6.2) ─────────────────────────
    // Single-agent lifelong PDP: DbVNS vs ALNS vs DoubleHorizon (+MCA-LNS),
    // identical allocation (MCA argmin), identical task draws.
    else if (rep == "P" || rep == "p") {
        run_planning_comparison_test(
            /*base_seed=*/42,
            /*max_tasks=*/0,
            kOsmRoot, kCacheRoot,
            kOutputDir + "\\planning_comparison",
            /*n_seeds=*/100,
            /*detail_every=*/10);
    }

    else {
        std::cout << "Unknown option.\n";
    }

    return 0;
}
