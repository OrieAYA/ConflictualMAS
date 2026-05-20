#include "Legacy/Tests/LegacyTests.hpp"
#include "DMASforPD/Tests/PDPTests.hpp"
#include "AmazonDataset/AmazonTest.hpp"
#include "Comparisons/DbVNS_vs_LKH/ComparisonTest.hpp"
#include "Training/MultiCityTrainer.hpp"
#include "Training/CityConfig.hpp"
#include "Training/TrainingConfig.hpp"
#include "Training/TrainingSmokeTest.hpp"
#include "Tests/PlanningComparisonTest.hpp"
#include <iostream>
#include <string>

int main()
{
    const std::string osm_file  = "C:\\ConflictualMAS\\src\\maps\\Tokyo.osm.pbf";
    const std::string cache_dir = "C:\\ConflictualMAS\\src\\geobox_cache_folder";
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
    std::cout << "  A  DbVNS on Amazon routes (new dataset, 13 routes)\n";
    std::cout << "  B  DbVNS vs LKH benchmark (same routes, side-by-side)\n\n";
    std::cout << "[MAPPO Training]\n";
    std::cout << "  T  Multi-city MAPPO training (Tokyo/Kyoto/LosAngeles × Small/Medium)\n";
    std::cout << "  S  Training smoke test (uses kanto OSM, no extra files needed)\n";
    std::cout << "  M  Multi-city MAPPER-only training (faster, no IPPO/MAPPO trained)\n";
    std::cout << "  N  MAPPO DH training (re-train MAPPO under Double-Horizon planning)\n";
    std::cout << "  X  Eval-only (load saved checkpoints, 5-mode: MAPPO+MAPPER+FM+Hybrid+TAM-AA)\n";
    std::cout << "  P  Planning comparison test (Tokyo_Small, MCA vs Double-Horizon, SVG)\n";
    std::cout << "  Y  SoTA architecture comparison (RL + 8 SoTA baselines, full sweep)\n\n";
    std::cout << "Choice: ";

    std::string rep;
    std::cin >> rep;

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
    else if (rep == "T" || rep == "t") {
        // ── MAPPO multi-city training ─────────────────────────────────────
        // OSM files expected at:  <osm_root>/<CityName>.osm.pbf
        // GeoBox JSON caches at:  <cache_root>/<CityName>.json  (auto-created)
        // Results (CSV + policy) at:  <output_dir>/
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results";

        // set_osm_root must be called BEFORE CityRegistry::all() (lazy static).
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root       = cache_root;
        cfg.output_dir       = output_dir;
        cfg.n_rounds         = 60;
        cfg.n_seeds          = 1;
        cfg.n_eval_episodes  = 3;
        cfg.eval_every       = 20;
        cfg.save_policy      = true;
        cfg.verbose          = true;

        // ── Lightweight periodic eval ────────────────────────────────────────
        // The default eval array (12 modes × 6 cities × 3 eps = 216 eps per
        // periodic eval) was taking 1-2h per eval window. With 3 eval windows
        // per seed + final + stress + generalize, that's 6-8h of pure eval.
        // The lightweight subset below (5 modes) drops periodic eval cost
        // by 2.4× while keeping the comparison meaningful for monitoring.
        // The FINAL exhaustive comparison should be done via option X.
        cfg.eval_modes = {
            PolicyMode::MAPPO,
            PolicyMode::IPPO,
            PolicyMode::MAPPER,
            PolicyMode::TamAlwaysAccept,
            PolicyMode::Greedy
        };
        // Periodic stress eval is expensive (same 5 modes × 6 cities); enable
        // only at the end if needed. Generalize eval kept at end of each seed.
        cfg.skip_stress_eval     = true;
        cfg.skip_generalize_eval = false;

        // Train on Tokyo/Kyoto/LA × Small/Medium only (6 cities). Generalisation
        // eval on Tokyo_Large/Fukuoka/NewYork/Paris/London is disabled because
        // the policy structure changed (lr_actor ×10, adv ×2) and the old
        // generalization protocol added 3+ h that doesn't fit the overnight budget.

        // To extend from a saved checkpoint (add more seeds / rounds):
        //   cfg.load_policy  = true;
        //   cfg.policy_path  = output_dir + "/policy_seed42.bin";

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else if (rep == "M" || rep == "m") {
        // ── MAPPER-only training, MAY 17 SETUP IDENTICAL except eval modes ──
        //
        // SAME as May 17 baseline run:
        //   - 60 rounds × 1 seed × 6 train cities (Tokyo/Kyoto/LosAngeles × S/M)
        //   - Scenario sampler active (slack/normal/stress_light/stress_heavy)
        //   - Reward shaping unchanged
        //   - 3 eval windows (rounds 20, 40, 60) with stress eval per window
        //   - Generalize eval on held-out cities at end of seed
        //
        // DIFFERENT from May 17:
        //   - Trains ONLY MAPPER (no IPPO, no MAPPO retrained)
        //   - 4 eval modes instead of 8 — drops LaCAM, PIBT, CongestionAware,
        //     Random, InsertionGreedy. Keeps the 4 the user specified:
        //     MAPPER + MAPPO + TamAlwaysAccept + Greedy.
        //
        // For MAPPO comparison during eval, loads the May 17 checkpoint
        // (policy_seed42_mappo_working_2026-05-17.bin from results/accepted/).
        // For the 12-feature run, do this on the mapper-12d-train worktree
        // with the 13→12 patch applied — the May 17 checkpoint is 12-d.
        //
        // Estimated wallclock: ~7-9h.
        // May 17 reference (MAPPO + 8 eval modes) = 14.7h. The 4-mode eval
        // halves eval time; MAPPER training is faster per episode than MAPPO
        // (fewer experiences per buffer flush).
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results";  // change for 12-d
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root      = cache_root;
        cfg.output_dir      = output_dir;
        cfg.n_rounds        = 60;
        cfg.n_seeds         = 1;
        cfg.save_policy     = true;
        cfg.verbose         = true;

        // ── Pure training run — all eval phases disabled ───────────────────
        // train/eval are fully decoupled: Option M produces three RL
        // checkpoints (MAPPER-enhanced, MAPPER-faithful, IPPO-faithful),
        // Option X/Y do the full comparison eval afterwards on saved weights.
        // Skips periodic, final, stress, and generalize evals.
        cfg.train_only = true;

        // ── Planning method = forward DbVNS-PDP (lifelong global replan) ────
        // All RL policies train under DbVNS planning at the routing level:
        // every accept triggers a global forward DbVNS-PDP replan of the
        // remaining sequence. This matches the architectural separation
        //   TAM → policy (MAPPO/MAPPER/Hybrid) decides accept/refuse
        //   DbVNS handles routing exclusively
        // and gives the policy a higher-quality route than DH insertion.
        cfg.episode_cfg.use_dbvns_planning = true;

        // ── Two MAPPER variants trained simultaneously, head-to-head ───────
        //   MAPPER         — enhanced variant (elite + Gaussian mutation N(0, 0.02²))
        //   FaithfulMAPPER — paper-faithful (probabilistic copy of best, no mutation)
        // MAPPO is intentionally NOT retrained — we compare against the May 17
        // MAPPO checkpoint (results/accepted/policy_seed42_mappo_working_2026-05-17.bin)
        // during Option X eval. IPPO was dropped from this run (too slow); its
        // paper-faithful refactor remains in the codebase for later use.
        // All three share kPolicySz = 12 features.
        // Each round runs one episode PER training mode PER train city: 2 modes
        // × 6 cities = 12 episodes per round.
        cfg.train_modes = {
            PolicyMode::MAPPER,
            PolicyMode::FaithfulMAPPER
        };

        // Match May 17 scenario distribution exactly: 3-regime sampler (no slack)
        // Verified from log: 0/360 slack samples → slack regime was absent on May 17.
        cfg.disable_slack_regime = true;

        // No checkpoint loading needed: all three RL policies start from Xavier
        // init, and no eval phase compares against MAPPO during training.
        cfg.load_policy = false;

        // Output subdirectories (created automatically at save time):
        //   results/mapper_enhanced/mapper_seed42.bin
        //   results/mapper_faithful/mapper_seed42.bin
        // MAPPO May 17 checkpoint at results/accepted/ remains untouched.
        // IPPO is not in train_modes → no checkpoint produced this run.

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else if (rep == "N" || rep == "n") {
        // ── MAPPO re-training under DbVNS planning ───────────────────────────
        //
        // WHY: The May 17 MAPPO was trained under MCA planning (cheapest
        // insertion). Option X now evaluates everything under DbVNS — re-
        // training MAPPO directly under DbVNS eliminates the distribution
        // shift and gives a fair MAPPO baseline that matches the
        // architectural separation: TAM → policy → DbVNS planner.
        //
        // WHAT CHANGES vs May 17:
        //   - use_dbvns_planning = true            (was MCA)
        //   - train_modes = {MAPPO}                (MAPPER/IPPO/FaithfulMAPPER skipped)
        //   - save dir: results/mappo_dh/mappo/policy_seed42.bin
        //
        // SAME AS MAY 17:
        //   - 60 rounds × 1 seed × 6 train cities
        //   - 3-regime scenario sampler (no slack, matches May 17 distribution)
        //   - train_only = true (eval via Option X after)
        //
        // AFTER THIS RUN:
        //   Update Option X: cfg.policy_path = ".../mappo_dh/mappo/policy_seed42.bin"
        //   to compare MAPPO-DH instead of MAPPO-May17 against MAPPER + Hybrid.
        //
        // Estimated wallclock: ~5-7h (MAPPO only, 60 rounds, 6 cities).
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results\\mappo_dh";
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root      = cache_root;
        cfg.output_dir      = output_dir;
        cfg.n_rounds        = 60;
        cfg.n_seeds         = 1;
        cfg.save_policy     = true;
        cfg.verbose         = true;

        cfg.train_only   = true;   // eval via Option X after, not during
        cfg.load_policy  = false;  // fresh Xavier init

        cfg.episode_cfg.use_dbvns_planning = true;

        cfg.train_modes = { PolicyMode::MAPPO };

        // Match May 17 scenario distribution: 3-regime, no slack.
        cfg.disable_slack_regime = true;

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else if (rep == "Y" || rep == "y") {
        // ── SoTA-ONLY benchmark (no training, no RL) ────────────────────────
        // Companion to option X (which evaluates our RL policies). Together
        // X + Y produce the complete data table for the thesis:
        //   X.csv → MAPPO, MAPPER, Hybrid metrics
        //   Y.csv → 7 SoTA architectures metrics
        //   merge → "our architecture vs state-of-the-art" comparison.
        //
        // SoTA modes (7) — strictly published state-of-the-art baselines:
        //   - MCA            [Chen+2021 ICRA]            Marginal-cost assignment
        //   - LaCAM          [Okumura+2022 spirit]       Min full-trip cost
        //   - PIBT           [Okumura+2022 adapted]      Load-balanced priority
        //   - CongestionAware [Liu, Saha+]               Pickup-leg dynamic cost
        //   - TrafficFlow    [Chen+2024 AAAI GP-PIBT]    Full-trip dynamic cost
        //   - TokenPassing   [Ma+2017 AAMAS]             Decoupled MAPD min h
        //   - DoubleHorizon  [Mitrovic-Minic+2004]       Horizon-aware insertion
        //
        // NO TRAINING — eval_only = true; no checkpoint loading needed because
        // SoTA baselines are deterministic / heuristic (no weights). The TAM is
        // still used to route offers (these modes opt out of policy decisions).
        //
        // Scenarios (4) — IDENTICAL to option X's planned coverage:
        //   slack         : density 0.5 × agents 1.2
        //   normal        : density 1.0 × agents 1.0
        //   stress_light  : density 1.5 × agents 0.8
        //   stress_heavy  : density 2.0 × agents 0.6
        //
        // Cities: 6 train (Tokyo/Kyoto/LA × Small/Medium) + 5 held-out
        //         (Tokyo_Large/Fukuoka/NewYork/Paris/London) = 11 cities.
        //
        // Coverage: 7 modes × 11 cities × 4 scenarios × 3 eps = 924 episodes.
        // Estimated wallclock: ~8-10h on a single seed (no RL training).
        //
        // Metrics (same set as option X — captured in episodes_seedXX.csv):
        //   throughput, agent_utilisation, latency, congestion (objective),
        //   bbox / convex_hull / mean_pd / mean_nn_pickup (spatial),
        //   wallclock / per_task / per_decision (temporal),
        //   pairing_violations / capacity_violations (validity).
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results";
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root       = cache_root;
        cfg.output_dir       = output_dir;
        cfg.n_seeds          = 1;
        cfg.n_eval_episodes  = 3;
        cfg.verbose          = true;

        cfg.eval_only            = true;   // NO training
        cfg.skip_stress_eval     = true;   // stress is covered via eval_scenarios
        cfg.skip_generalize_eval = false;  // include held-out cities

        // SoTA-only line-up: NO RL policies (MAPPO/MAPPER/Hybrid/IPPO excluded),
        // NO trivial ablations (Greedy/Random/InsertionGreedy/TamAlwaysAccept
        // also excluded — they live in option X). Strictly published SoTA.
        cfg.eval_modes = {
            PolicyMode::CongestionAware, // [Liu, Saha+]
            PolicyMode::TrafficFlow,     // [Chen+2024]
            PolicyMode::TokenPassing,    // [Ma+2017]
        };

        // 4-scenario sweep aligned with option X coverage.
        cfg.eval_scenarios = {
            EpisodeScenario{0.5f, 1.2f, "slack"},
            EpisodeScenario{1.0f, 1.0f, "normal"},
            EpisodeScenario{1.5f, 0.8f, "stress_light"},
            EpisodeScenario{2.0f, 0.6f, "stress_heavy"},
        };

        // No checkpoint loading — SoTA baselines carry no learned weights.
        // (load_policy stays false by default; safety check passes because
        //  none of the cfg.eval_modes entries are RL policies.)

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else if (rep == "P" || rep == "p") {
        // ── Planning comparison batch test (mono-agent, lifelong GPDP) ──
        // 100 seeds × {Tokyo_Small, Kyoto_Small, LosAngeles_Small}.
        // Each seed is stratified into one of three saturation regimes
        // (low / medium / high) so we observe the planning algorithms
        // both when all tasks are completable and when the agent is
        // oversaturated. max_tasks and density_mult are determined per-seed
        // from the regime — the explicit args below are kept as fallback.
        // For every seed all three modes (MCA / DoubleHorizon / DbVNS)
        // are evaluated.
        // Outputs:
        //   - planning_summary.csv      (one row per seed × mode)
        //   - details/seed_NNNN_*.csv   (every 10 seeds: task + agent traces)
        //   - details/seed_NNNN_*.svg   (every 10 seeds: render)
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results\\planning_test";
        run_planning_comparison_test(
            /*base_seed*/0,            // 0 → random from std::random_device
            /*max_tasks*/25,           // ignored — regime overrides per seed
            osm_root, cache_root, output_dir,
            /*n_seeds*/100,
            /*detail_every*/10,
            /*cities*/{"Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"},
            /*density_min*/0.5f,       // ignored — regime overrides per seed
            /*density_max*/2.5f);
    }
    else if (rep == "S" || rep == "s") run_training_smoke_test(osm_file, cache_dir);
    else if (rep == "X" || rep == "x") {
        // ── Eval-only: 5-mode benchmark across all 7 cities ──────────────────
        // The FINAL publication-grade comparison. Evaluates 5 policies on every
        // available city (train + held-out = 7) with both normal and stress
        // scenarios. Uses saved checkpoints, no training.
        //
        // Modes:
        //   - MAPPO           (May17 baseline, MCA-trained, evaluated under DH)
        //   - MAPPER          (enhanced variant: elite + Gaussian mutation, DH)
        //   - FaithfulMAPPER  (paper-faithful Liu+2020: copy-no-mutation, DH)
        //   - Hybrid          (frozen MAPPO May17 base + per-agent REINFORCE residual)
        //                     No training needed: base set from MAPPO checkpoint,
        //                     residuals start at zero (= pure MAPPO hot-start)
        //                     and online-adapt during each eval episode.
        //   - TamAlwaysAccept (TAM routing only, always-accept ablation)
        //
        // NOTE: After Option N completes, update cfg.policy_path below to
        //   ".../mappo_dh/mappo/policy_seed42.bin" for a MAPPO-DH vs Hybrid-DH
        //   comparison where the distribution shift is eliminated.
        //
        // Coverage: 5 modes × 7 cities × 2 phases (normal + stress) × 5 eps
        //         = 350 eval episodes per seed.
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results";
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root       = cache_root;
        cfg.output_dir       = output_dir;
        cfg.n_seeds          = 1;
        cfg.n_eval_episodes  = 5;
        cfg.verbose          = true;

        // Eval-only flags — full coverage: train + stress + generalize
        cfg.eval_only            = true;
        cfg.skip_stress_eval     = false;  // include over-saturated scenarios
        cfg.skip_generalize_eval = false;  // include held-out cities

        // 5-mode comparison.
        //
        //   MAPPO              — May 17 baseline, 12-d, MCA planning at train time
        //   MAPPER             — enhanced variant (elite + Gaussian mutation), DH
        //   FaithfulMAPPER     — paper-faithful (copy-no-mutation, Liu+2020), DH
        //   Hybrid             — frozen MAPPO base + per-agent online REINFORCE residual
        //                        (14 params/agent, rollback safety, hot-starts as MAPPO)
        //   TamAlwaysAccept    — TAM routing only (always-accept ablation)
        //
        // Greedy dropped: 100-700s per episode on Medium cities, dominates wallclock.
        // Hybrid added: Hybrid base is auto-wired from MAPPO checkpoint below.
        cfg.eval_modes = {
            PolicyMode::MAPPO,
            PolicyMode::MAPPER,
            PolicyMode::FaithfulMAPPER,
            PolicyMode::Hybrid,
            PolicyMode::TamAlwaysAccept
        };

        // ── Planning: forward DbVNS-PDP at eval (architectural separation).
        // TAM → policy (MAPPO/MAPPER/Hybrid) decides accept/refuse, then
        // DbVNS replans the full remaining sequence on every acceptance.
        // The MAPPO checkpoint trained under MCA suffers a distribution shift
        // (planner is stronger than at train time); MAPPER/Hybrid baselines
        // were retrained under the same DbVNS planner so the comparison
        // ("policy ↔ planner pair held fixed") is fair for them.
        cfg.episode_cfg.use_dbvns_planning = true;

        // Load trained policies from disk.
        //   policy_path                  → MAPPO May 17 baseline (accepted dir)
        //   mapper_policy_path           → MAPPER enhanced (mapper_enhanced/)
        //   faithful_mapper_policy_path  → MAPPER paper-faithful (mapper_faithful/)
        //   ippo_policy_path             → empty (IPPO not in eval_modes)
        cfg.load_policy                 = true;
        cfg.policy_path                 = "C:\\ConflictualMAS\\results\\acceptedMAPPPO\\"
                                          "policy_seed42_mappo_working_2026-05-17.bin";
        cfg.mapper_policy_path          = "C:\\ConflictualMAS\\results\\mapper_enhanced\\"
                                          "mapper_seed42.bin";
        cfg.faithful_mapper_policy_path = "C:\\ConflictualMAS\\results\\mapper_faithful\\"
                                          "mapper_seed42.bin";

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else std::cout << "Unknown option.\n";

    return 0;
}

// =============================================================================
// OPTION S — Training Smoke Test
// =============================================================================
//
// OVERVIEW
//   Quick end-to-end sanity check for the MAPPO training pipeline.
//   Uses the kanto OSM file (already present at src/maps/kanto-latest.osm.pbf)
//   with a small 4×5 km central Tokyo bbox — no extra downloads needed.
//
// HOW TO RUN
//   1. Build  : cmake --build "C:/ConflictualMAS/build" --config Release 2>&1 | grep -iE "error|main.vcxproj ->" | tail -10
//   2. Launch : C:\ConflictualMAS\build\Release\main.exe
//
// WHAT IT CHECKS
//   1. GeoBox loads and contains valid road nodes.
//   2. EpisodeRunner constructs without crash (agents, path cache, TaskAgent).
//   3. Training episode (MAPPO, 200 steps, 4 agents): metrics in [0,1],
//      actor_loss and critic_loss finite.
//   4. Greedy + MAPPO eval episodes complete without crash.
//   5. Second training episode: episode reset is clean (tasks, congestion, clock).
//
// EXPECTED OUTPUT
//   [1] GeoBox <- cache / <- OSM   (parses once, cached to geobox_cache_folder/)
//   [2] Config OK — 200 steps, 4 agents max
//   [3] Runner constructed OK
//   [3] Training episode OK — N tasks, acc=X, aloss=Y, closs=Z, Nms
//   [4] Eval OK — Greedy acc=X, MAPPO acc=Y
//   [5] Episode reset OK — 2nd train acc=X
//   === PASS ===
//
// EXPECTED TIME   < 30 seconds (first run may take 2-3 min for OSM parsing).
//
// =============================================================================

// =============================================================================
// OPTION T - MAPPO Multi-City Training (publishable protocol)
// =============================================================================
//
// OVERVIEW
//   Trains a shared MAPPO policy on the Lifelong General Pickup-and-Delivery
//   Problem (LGPDP) over 3 Tokyo scales (curriculum), then evaluates on unseen
//   cities for out-of-distribution generalisation.
//
//   Architecture (CTDE):
//     Actor  (shared)      :  MLP  12 -> 64 -> 64 -> 1  (sigmoid, Bernoulli)
//     Critic (centralised) :  MLP  20 -> 64 -> 64 -> 1  (linear, V(global_state))
//
//   Allocation: real Task Allocation Module (incremental Dijkstra from pickup
//   and delivery; agents discovered via objective nodes of their assigned tasks;
//   plan-order check before offer; max 3 recalls with importance boost).
//
// =============================================================================
// STEP-BY-STEP TUTORIAL
// =============================================================================
//
// STEP 1 -- OSM file (only Tokyo is required to train)
//   Required for training: src/maps/Tokyo.osm.pbf
//     All 3 train scales (Small/Medium/Large) use the same Tokyo.osm.pbf with
//     different bboxes (5x5 km / 12x12 km / 30x28 km).
//
//   Optional for the generalisation phase (loaded with graceful skip):
//     src/maps/Kyoto.osm.pbf       Kyoto       (~217 km²)
//     src/maps/Fukuoka.osm.pbf     Fukuoka     (~340 km²)
//     src/maps/LosAngeles.osm.pbf  Los Angeles (~1300 km²)
//     src/maps/NewYork.osm.pbf     New York    (~783 km²)
//     src/maps/Paris.osm.pbf       Paris       (~105 km²)
//     src/maps/London.osm.pbf      London      (~1572 km²)
//
//   Any missing OSM file is skipped at load time (printed as "[Skip] <city>").
//
// STEP 2 -- Build
//   cmake --build C:\ConflictualMAS\build --config Debug
//
// STEP 3 -- Smoke test (sanity check, ~30 s after first OSM parse)
//   build\Debug\main.exe  ->  type S
//   Expected: === PASS ===
//
// STEP 4 -- Launch training
//   build\Debug\main.exe  ->  type T
//
//   First run: each city bbox is parsed from Tokyo.osm.pbf and cached as JSON
//   in C:\ConflictualMAS\data\cache\ (Tokyo_Small.json, etc.). Subsequent
//   runs skip parsing.
//
//   Expected console flow:
//     Loading 3 train cities from C:\ConflictualMAS\data\cache
//       [Load] Tokyo_Small  <- OSM (parsing...)   <- first run
//       [Load] Tokyo_Small  <- cache              <- next runs
//       ...
//     All train cities loaded.
//
//     Loading generalisation cities (skip if missing)...
//       [Skip] Kyoto -- failed to load city Kyoto    <- if Kyoto.osm.pbf absent
//       [Load] LosAngeles  <- cache
//       ...
//
//     Seed 0  (rng=42)
//       [s0 r0 Tokyo_Small]   thr=...  acc=...  aloss=...  cf=...  ms=...
//       [s0 r0 Tokyo_Medium]  ...
//       [s0 r0 Tokyo_Large]   ...
//       -- Eval @ round 25 --        <- 3 cities x 5 modes x 3 eps = 45 episodes
//       ...
//       -- Eval @ round 100 --
//       -- Generalisation Eval (N cities) --   <- 3 modes x N x 3 eps
//       [Checkpoint] saved to .../policy_seed42.bin
//     Seed 0 done.
//
//     Seed 1  (rng=43)  ...
//
// STEP 5 -- Check results
//   C:\ConflictualMAS\results\
//     seed42.csv         -- per-episode metrics for seed 0 (rng=42)
//     seed43.csv         -- per-episode metrics for seed 1 (rng=43)
//     summary.csv        -- per-seed aggregates
//     policy_seed42.bin  -- trained weights for seed 0
//     policy_seed43.bin  -- trained weights for seed 1
//
//   Key comparisons (filter split=eval or split=generalize):
//     MAPPO vs TamAlwaysAccept   -- isolates the value of policy LEARNING
//                                   (both use the same TAM routing)
//     TamAlwaysAccept vs Greedy  -- isolates the value of TAM routing
//     MAPPO vs InsertionGreedy   -- vs the strongest non-learning baseline
//
//   Convergence to watch on split=train (per round, per city):
//     throughput_rate  -- rises with rounds
//     accept_rate      -- settles ~0.4-0.8 (NOT pinned at 0.99 or 0.01)
//     clip_fraction    -- > 0.05 means PPO updates are non-trivial
//
// =============================================================================
// TRAINING PROTOCOL
// =============================================================================
//
//   Seeds             :  2  (rng = 42, 43)
//   Rounds per seed   :  100  (one MAPPO episode per train city per round)
//   Train cities      :  3  (Tokyo_Small, Tokyo_Medium, Tokyo_Large)
//   Train episodes    :  100 x 3            =  300 / seed
//   Eval checkpoints  :  rounds 25, 50, 75, 100   = 4 / seed
//   Eval modes        :  8  (MAPPO, TamAlwaysAccept, Greedy, Random, InsertionGreedy,
//                            LaCAM, PIBT, CongestionAware)
//                       — see related works mapping in EpisodeRunner.hpp
//   Eval episodes     :  3 x 6 x 8 x 3      =  432 / seed
//   Generalise modes  :  4  (MAPPO, TamAlwaysAccept, LaCAM, CongestionAware)
//   Generalise eps    :  N_gen x 3 x 3      (per seed, N_gen = available comp. cities)
//   Total per seed    :  ~480 + 9*N_gen episodes
//
//   Timing (after Tokyo.osm.pbf is cached, Debug build):
//     Per seed   :  ~1.5-2 h   (large Tokyo dominates)
//     2 seeds    :  ~3-4 h
//
// =============================================================================
// EPISODE STRUCTURE (3 600 steps, three-phase curriculum)
// =============================================================================
//
// Per-city overrides applied by MultiCityTrainer::customize_episode_for_city:
//
//   Tokyo_Small  (~25 km²)   max_tasks_per_agent=3
//     Low   1000 steps  4-5 agents   3 tasks/agent   3 hot zones
//     Med   1500 steps  5-7 agents   5 tasks/agent   4 hot zones
//     High  1100 steps  7-8 agents   7 tasks/agent   5 hot zones
//
//   Tokyo_Medium (~144 km²)  max_tasks_per_agent=3
//     Low   1000 steps  6-8 agents   4 tasks/agent   4 hot zones
//     Med   1500 steps  8-10         6 tasks/agent   6 hot zones
//     High  1100 steps  10-12        7 tasks/agent   7 hot zones
//
//   Tokyo_Large  (>=300 km²) max_tasks_per_agent=4
//     Low   1000 steps  8-10         4 tasks/agent   4 hot zones
//     Med   1500 steps  10-13        6 tasks/agent   6 hot zones
//     High  1100 steps  13-15        8 tasks/agent   8 hot zones
//
// Poisson task arrivals; fleet ramps linearly within each phase. Hot zones
// resampled at phase boundaries.
//
// =============================================================================
// MAPPO HYPERPARAMETERS
// =============================================================================
//
//   PPO clip eps      = 0.20
//   gamma             = 0.99
//   lambda (GAE)      = 0.95
//   epochs / episode  = 10  (with KL early stop at 0.01)
//   Grad-norm clipping (max_norm=0.5), mini-batch SGD.
//
// Policy feature vector (12 floats, see ObjectiveDMPolicy::PolicyFeatures):
//   Rentability : profit_rate, insertion_cost_norm
//   Status      : current_load, queue_duration_norm
//   Social      : call_rank_norm, importance_norm, recall_round_norm
//   Relativity  : n_active_norm, allocated_ratio, available_ratio
//   Temporal    : time_remaining
//
// =============================================================================
// CONVERGENCE MONITORING
// =============================================================================
//
//   Verbose log (one line per training episode):
//     [s0 r5 Tokyo_Small]  thr=0.41  acc=0.68  aloss=0.12  closs=0.08
//                          ent=0.61  kl=0.007  cf=0.18  ep=10/10  n=312  420ms
//
//   Healthy signals:
//     cf  > 0.05   PPO updates are non-trivial (was 0 before fixes -> degenerate)
//     acc in [0.4, 0.85]  policy actually refuses some tasks
//     thr rises across rounds
//     ent decreases gradually
//     kl < 0.02 most steps (else lr too high)
//
//   ep < 10/10 means KL early stop triggered (normal and healthy).
//
// =============================================================================
// TUNING
// =============================================================================
//
//   n_rounds, n_seeds, eval_every, n_eval_episodes
//     -> in the TrainingConfig block below
//
//   tasks_per_agent, n_agents per phase, hot zones
//     -> src/Training/MultiCityTrainer.cpp :: customize_episode_for_city
//
//   PPO hyperparameters (lr_actor, lr_critic, epochs, clip_eps, ...)
//     -> src/DMASforPD/Policy/ObjectiveDMPolicy.hpp
//
//   Resume / extend from checkpoint:
//     cfg.load_policy = true;
//     cfg.policy_path = output_dir + "/policy_seed42.bin";
//
// =============================================================================

// =============================================================================
// OPTION A — Amazon Last Mile Single-Agent DbVNS Test
// =============================================================================
//
// OVERVIEW
//   Runs the DbVNS-PDP solver on 10 randomly-sampled routes from the
//   Amazon Last Mile Challenge 2021 dataset and compares results against
//   the ground-truth driver sequences.
//
// DATASET (required files)
//   src/AmazonDataset/~/.rc-cli/data/
//     model_apply_inputs/new_travel_times.json    — inter-stop travel times (s)
//     model_score_inputs/new_actual_sequences.json — driver visit order
//
//   The dataset contains 13 routes. Each route has:
//     - One depot  (stop at sequence position 0)
//     - 130–190 delivery stops with a complete travel-time matrix
//
// HOW TO RUN
//   1. Build  : cmake --build C:\ConflictualMAS\build --config Debug
//              (or: cd build && cmake --build . --config Debug)
//   2. Launch : build\Debug\main.exe  ->  type A
//
// WHAT IT DOES
//   For each route:
//     1. Identifies the depot (sequence position 0).
//     2. Pairs the remaining delivery stops consecutively into artificial
//        PDP tasks: (stop_0, stop_1), (stop_2, stop_3), ...
//        This lets DbVNS enforce the pickup-before-delivery constraint.
//     3. Builds an OperableEnvironment directly from the travel-time matrix
//        (no OSM graph, no GlobalMemory needed).
//     4. Runs one LocalSolutionAgent per sampled anchor (≤15), picks the
//        best sequence, and measures execution time.
//     5. Computes the full trip cost (depot → seq → last stop) and compares
//        it with the ground-truth driver cost.
//
// OUTPUT INTERPRETATION
//   ratio = DbVNS time / ground-truth time
//     < 1.0  — DbVNS beats the Amazon driver
//     = 1.0  — equal
//     > 1.0  — driver is better  (expected ~1.30–1.50 with default params,
//               because the PDP pairing is arbitrary and not task-optimal)
//
//   Architecture validation section verifies:
//     - PDP constraint : pickup always visited before its delivery  (must be 0)
//     - Unreachable edges : no sentinel costs in the solution        (must be 0)
//     - Sequence completeness : all stops covered exactly once
//
//   Execution scaling section measures ms/stop and the Pearson correlation
//   between route size and execution time (expected strong positive for O(N²)).
//
// TUNING  (src/AmazonDataset/AmazonTest.cpp → test_amazon_routes)
//   params.max_iterations     — VNS iterations per anchor  (default 15)
//   params.max_decompositions — branches explored per shake (default 2)
//   params.k_max              — shake depth factor          (default 3)
//   max_anchors               — delivery nodes tried as anchors (default 15)
//   Increasing these improves solution quality at the cost of execution time.
// =============================================================================
