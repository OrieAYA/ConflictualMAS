#include "Tests/LegacyTests.hpp"
#include "Tests/PDPTests.hpp"
#include "TrainingEvaluation/Run/Trainer.hpp"
#include "TrainingEvaluation/StructuresParam/CityConfig.hpp"
#include "TrainingEvaluation/General/TrainingConfig.hpp"
#include "Environment/Structure/EpisodeManager.hpp"
#include "Tests/StructureTests.hpp"          // A
#include "Tests/MechanicsTests.hpp"          // B
#include "Tests/ModuleTests.hpp"             // C
#include "Tests/EndToEndTests.hpp"           // D
#include "Tests/RegressionTests.hpp"         // R
#include "Tests/CongestionOnlyTest.hpp"      // G — diagnostic
#include "Tests/PlanningComparisonTest.hpp"  // P — diagnostic

#include "SoTA/SolverFramework.hpp"
#include "SoTA/Standalone/CA.hpp"
#include "SoTA/Standalone/HAPC.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"

#include <cctype>
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

// 9 scenarios = 3 task levels × 3 congestion levels (make_scenario_grid()).
// Edit the level tables in EpisodeRunner.cpp to retune. Training and eval
// share the same grid.

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

    std::cout << "\n=== ConflictualMAS ===\n\n"
                 "  1  Legacy      (orienteering / GeoBox / render / PDP utils)\n"
                 "  2  Tests       (submenu: A structure B mechanics C modules D e2e E all)\n"
                 "  3  Training    (MAPPO + IPPO + MAPPER, paper protocol)\n"
                 "  4  Evaluation  (RL + RMCA + TP + CA + HAPC)\n\n"
                 "Choice: ";
    std::string rep;
    std::cin >> rep;

    // ── 1. Legacy submenu ───────────────────────────────────────────────────
    if (rep == "1" || rep == "legacy" || rep == "L" || rep == "l") {
        std::cout << "\n[Legacy]  G Global  V VNS  PSO  E CreateGeoBox  I InitPOI"
                     "  R Render  D PDPtest  C PDPGeoBox\nChoice: ";
        std::string sub; std::cin >> sub;
        if      (sub == "G" || sub == "g")     legacy_run_global(cache_dir);
        else if (sub == "V" || sub == "v")     legacy_run_vns(cache_dir);
        else if (sub == "PSO" || sub == "pso") legacy_run_pso(cache_dir);
        else if (sub == "E" || sub == "e")     legacy_create_geobox(osm_file, cache_dir);
        else if (sub == "I" || sub == "i")     legacy_init_poi(cache_dir);
        else if (sub == "R" || sub == "r")     legacy_render(cache_dir);
        else if (sub == "D" || sub == "d")     test_pdp_system(cache_dir);
        else if (sub == "C" || sub == "c")     pdp_create_geobox(osm_file, cache_dir);
        else std::cout << "Unknown legacy option.\n";
    }

    // ── 2. Test batteries (hand-picked from a submenu) ───────────────────────
    //   A Structure  B Mechanics  C Modules  D EndToEnd  E All
    //   G CongestionDiag  P PlanningCompare  (heavier diagnostics)
    else if (rep == "2" || rep == "test" || rep == "tests") {
        std::cout << "\n[Tests]  A Structure  B Mechanics  C Modules  D EndToEnd  R Regression  E All"
                     "   |   G CongestionDiag  P PlanningCompare\nChoice: ";
        std::string sub; std::cin >> sub;
        const char c = sub.empty() ? ' ' : static_cast<char>(std::toupper(sub[0]));

        auto run_all = [&]() {
            bool ok = true;
            ok &= run_structure_tests (osm_file, cache_dir);
            ok &= run_mechanics_tests (osm_file, cache_dir);
            ok &= run_module_tests    (osm_file, cache_dir);
            ok &= run_end_to_end_tests(osm_file, cache_dir);
            ok &= run_regression_tests(osm_file, cache_dir);
            std::cout << (ok ? "\n=== ALL BATTERIES PASS ===\n"
                             : "\n=== SOME BATTERIES FAILED ===\n");
        };

        switch (c) {
            case 'A': run_structure_tests (osm_file, cache_dir); break;
            case 'B': run_mechanics_tests (osm_file, cache_dir); break;
            case 'C': run_module_tests    (osm_file, cache_dir); break;
            case 'D': run_end_to_end_tests(osm_file, cache_dir); break;
            case 'R': run_regression_tests(osm_file, cache_dir); break;
            case 'E': run_all();                                 break;
            case 'G': run_congestion_only_test(kOsmRoot, kCacheRoot); break;
            case 'P': run_planning_comparison_test(42u, 25, kOsmRoot, kCacheRoot, kOutputDir); break;
            default:  std::cout << "Unknown test option.\n";
        }
    }

    // ── T — Training (paper §5.5.1) ────────────────────────────────────────
    // 50 rounds × 3 Small cities × {MAPPO, IPPO, MAPPER}, single seed 42,
    // 3-regime scenario sampler, checkpoints every 10 rounds. Evaluation is
    // a separate run (option Y) on the saved checkpoints.
    else if (rep == "3" || rep == "train" || rep == "training") {
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
    else if (rep == "4" || rep == "eval" || rep == "evaluation") {
        std::cout << "Seed (42/43/44): ";
        int seed = 42;
        std::cin >> seed;

        CityRegistry::set_osm_root(kOsmRoot);

        TrainingConfig cfg;
        cfg.cache_root      = kCacheRoot;
        cfg.output_dir      = kOutputDir + "\\paper_eval";
        cfg.n_seeds         = 1;
        cfg.start_seed      = seed;
        cfg.n_eval_episodes = 1;     // 1 episode per (method, city, scenario)
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
        cfg.eval_scenarios = make_scenario_grid();   // 9 task×congestion combos

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


    else {
        std::cout << "Unknown option.\n";
    }

    return 0;
}
