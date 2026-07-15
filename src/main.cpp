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

#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Environment/GeoBox/Box.hpp"

#include <cctype>
#include <chrono>
#include <cmath>
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

// Scenario grid = task × congestion × fleet regimes (ScenarioConfig.cpp,
// paper_* tables). Training and evaluation share the same grid.

// Environment knobs shared by training and evaluation so there is no
// train/eval distribution shift: DbVNS planning, ghost congestion driven by
// the EventStream tuning constants (φh = kHotWayFraction, volume = β-quantity
// formula), heterogeneous capacity [3,7], latency shaping + task value
// heterogeneity, TAM K=5 force-assign, fleet ÷2 with load_per_agent=2.
static void apply_paper_environment(EpisodeConfig& ep) {
    ep.use_dbvns_planning            = true;

    ep.enable_ghost_traffic          = true;
    ep.ghost_hot_way_count           = 0;
    ep.ghost_hot_way_frac            = event_tuning::kHotWayFraction;   // φh
    ep.ghost_window_steps            = 250;  // long residence → fewer events for same congestion

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

    // Agent pool ≥ max fleet: fleet = round(10·SCE·AM), AM up to 2.5.
    ep.agent_pool_multiplier = 3.0f;

    // Reward shaping v2 (paper Table 2 revision) — full package ON for the
    // paper T/Y runs; ablations flip individual rs_* flags back off.
    enable_all_reward_shaping(ep);
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

    // ── 3 — Training: 6 cities × 3 seeds × 27 scenarios = 486 ep / policy,
    //    one independent policy per (type, seed); checkpoints
    //    results/{mappo,ippo,mapper}/*_seed{seed}.bin rewritten every episode.
    else if (rep == "3" || rep == "train" || rep == "training") {
        CityRegistry::set_osm_root(kOsmRoot);

        TrainingConfig cfg;
        cfg.cache_root  = kCacheRoot;
        cfg.output_dir  = kOutputDir;
        cfg.start_seed  = 42;
        cfg.n_seeds     = 3;
        cfg.save_policy = true;
        cfg.verbose     = true;
        cfg.log_every   = 1;

        cfg.train_city_filter = {
            "Tokyo_Small",      "Tokyo_Medium",
            "LosAngeles_Small", "LosAngeles_Medium",
            "Paris_Small",      "Paris_Medium",
        };
        cfg.train_modes = {
            PolicyMode::MAPPO, PolicyMode::IPPO, PolicyMode::MAPPER
        };
        cfg.train_scenarios = build_scenarios(paper_task_regimes(),
                                              paper_congestion_regimes(),
                                              paper_fleet_regimes());

        apply_paper_environment(cfg.episode_cfg);

        MultiCityTrainer trainer;
        trainer.train_grid(cfg);
    }

    // ── S — Smoke training: Tokyo Small+Medium, 1 seed. Times a few episodes
    //    with the real paper environment; kill early after enough ep lines.
    else if (rep == "S" || rep == "smoke") {
        CityRegistry::set_osm_root(kOsmRoot);

        TrainingConfig cfg;
        cfg.cache_root  = kCacheRoot;
        cfg.output_dir  = kOutputDir + "\\smoke";
        cfg.start_seed  = 42;
        cfg.n_seeds     = 1;
        cfg.save_policy = true;
        cfg.resume      = false;
        cfg.verbose     = true;
        cfg.log_every   = 1;

        cfg.train_city_filter = { "Tokyo_Small", "Tokyo_Medium" };
        cfg.train_modes = {
            PolicyMode::MAPPO, PolicyMode::IPPO, PolicyMode::MAPPER
        };
        cfg.train_scenarios = build_scenarios(paper_task_regimes(),
                                              paper_congestion_regimes(),
                                              paper_fleet_regimes());

        apply_paper_environment(cfg.episode_cfg);

        MultiCityTrainer trainer;
        trainer.train_grid(cfg);
    }

    // ── 4 — Evaluation: Phase A (RL + RMCA + TP + ablation on the shared
    //    TAM/DbVNS pipeline) then Phase B (standalone CA + HAPC). Both phases
    //    consume the same SharedEpisodeSetup per (city, scenario, episode).
    //    Loads the per-seed checkpoints produced by option 3.
    else if (rep == "4" || rep == "eval" || rep == "evaluation") {
        // Sweep de charge : un niveau = (seed, RM). RM = ratio_mult des events
        // (tasks = round(100·SCE·RM), ghosts = round(25000·SCE·RM)) ; la
        // flotte (10·SCE·AM) ne change pas. RM avance de kEvalRatioStep par
        // niveau ; seed = kEvalFirstSeed + (RM−1)/step — fonction de RM, donc
        // un même niveau produit les mêmes épisodes quel que soit le
        // découpage des lancements. Checkpoints d'entraînement FIXES
        // (kEvalPolicySeed). Parallélisme : un terminal par groupe de villes,
        // sorties séparées par le suffixe _g{groupe}.
        const int   kEvalPolicySeed = 42;    // checkpoint train (42 / 43 / 44)
        const int   kEvalFirstSeed  = 42;    // seed d'éval du niveau RM=1.0
        const float kEvalRatioStep  = 0.5f;

        std::cout << "Groupe de villes (1=Tokyo+Kyoto  2=LosAngeles+NewYork"
                     "  3=Paris  0=toutes) : ";
        int grp = 0;
        std::cin >> grp;
        std::cout << "RM min max (ex: 1.0 2.0) : ";
        float rm_min = 1.f, rm_max = 1.f;
        std::cin >> rm_min >> rm_max;

        CityRegistry::set_osm_root(kOsmRoot);

        const std::vector<std::vector<std::string>> city_groups = {
            { "Tokyo_Small",      "Tokyo_Medium",
              "Kyoto_Small",      "Kyoto_Medium",
              "LosAngeles_Small", "LosAngeles_Medium",
              "NewYork_Small",    "NewYork_Medium",
              "Paris_Small",      "Paris_Medium" },
            { "Tokyo_Small",      "Tokyo_Medium",
              "Kyoto_Small",      "Kyoto_Medium" },
            { "LosAngeles_Small", "LosAngeles_Medium",
              "NewYork_Small",    "NewYork_Medium" },
            { "Paris_Small",      "Paris_Medium" },
        };
        const std::vector<std::string>& eval_cities =
            city_groups[(grp >= 1 && grp <= 3) ? grp : 0];

        const std::string seed_tag =
            "_seed" + std::to_string(kEvalPolicySeed) + ".bin";

        const int n_levels = static_cast<int>(
            std::round((rm_max - rm_min) / kEvalRatioStep)) + 1;
        for (int lvl = 0; lvl < n_levels; ++lvl) {
        const float rm     = rm_min + lvl * kEvalRatioStep;
        const int   seed   = kEvalFirstSeed + static_cast<int>(
            std::round((rm - 1.f) / kEvalRatioStep));
        const int   tenths = static_cast<int>(std::round(rm * 10.f));
        const std::string rm_tag =
            std::to_string(tenths / 10) + "." + std::to_string(tenths % 10);
        std::cout << "\n════════ Niveau de charge RM=" << rm_tag
                  << " (seed=" << seed << ", groupe " << grp
                  << ", policy seed " << kEvalPolicySeed << ") ════════\n";

        TrainingConfig cfg;
        cfg.cache_root      = kCacheRoot;
        cfg.output_dir      = kOutputDir + "\\paper_eval\\pol"
                            + std::to_string(kEvalPolicySeed)
                            + "_rm" + rm_tag
                            + "_g" + std::to_string(grp);
        cfg.n_seeds         = 1;
        cfg.start_seed      = seed;
        cfg.n_eval_episodes = 1;
        cfg.verbose         = true;

        cfg.train_city_filter = eval_cities;

        cfg.eval_modes = {
            PolicyMode::MAPPO,
            PolicyMode::IPPO,
            PolicyMode::MAPPER,
            PolicyMode::Hybrid,
            PolicyMode::RMCA,
            PolicyMode::TokenPassing,
        };
        cfg.eval_scenarios = make_scenario_grid();

        apply_paper_environment(cfg.episode_cfg);
        cfg.episode_cfg.ratio_mult = rm;

        cfg.policy_path        = kOutputDir + "\\mappo\\policy" + seed_tag;
        cfg.ippo_policy_path   = kOutputDir + "\\ippo\\ippo" + seed_tag;
        cfg.mapper_policy_path = kOutputDir + "\\mapper\\mapper" + seed_tag;

        std::filesystem::create_directories(cfg.output_dir);

        // Episode-major : chaque slot (ville, scénario, épisode) est généré
        // une fois puis rejoué par les 7 modes et par CA/HAPC standalone.
        {
            MultiCityTrainer trainer;
            trainer.evaluate(cfg);
        }

        std::cout << "\n=== Niveau RM=" << rm_tag << " termine ===\n"
                  << "  Modes CSV: " << cfg.output_dir << "\\episodes_seed"
                  << seed << ".csv\n"
                  << "  SoTA  CSV: " << cfg.output_dir
                  << "\\sota_standalone\\sota_seed" << seed << ".csv\n"
                  << "  Join keys: (city, scenario, episode)\n";
        }   // fin du niveau RM
    }


    else {
        std::cout << "Unknown option.\n";
    }

    return 0;
}
