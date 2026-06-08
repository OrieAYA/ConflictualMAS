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
#include "Tests/BaselinesMultiAgentTest.hpp"
#include "Tests/CongestionOnlyTest.hpp"

// SoTA standalone solvers (Option U)
#include "SoTA/SolverRunner.hpp"
#include "SoTA/SolverCSVLogger.hpp"
#include "SoTA/TokenPassing/TokenPassingSolver.hpp"
#include "SoTA/CongestionAware/CongestionAwareSolver.hpp"
#include "SoTA/FaithfulCongestionAware/FaithfulCASolver.hpp"
#include "SoTA/FaithfulCongestionAwareCapacited/FaithfulCACapacitedSolver.hpp"
#include "SoTA/TrafficFlow/TrafficFlowSolver.hpp"
#include "SoTA/RHCR/RHCRSolver.hpp"
#include "SoTA/HybridAdaptivePredictive/HybridAdaptivePredictiveSolver.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <filesystem>

#include <exception>
#include <iostream>
#include <string>

// Forward-declares the actual main body so we can wrap it in a try/catch
// without indenting every existing line. Without this guard, a thrown
// std::runtime_error (notably from MultiCityTrainer::try_load when a
// checkpoint is missing in eval_only mode) silently terminates the process
// with no stderr output — extremely hard to debug.
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

static int run_main()
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
    std::cout << "  X  Eval-only (load saved checkpoints, 4-mode: MAPPO+FM+Hybrid+TAM-AA)\n";
    std::cout << "  P  Planning comparison test (Tokyo_Small, MCA vs Double-Horizon, SVG)\n";
    std::cout << "  Y  SoTA architecture comparison (RL + 6 SoTA baselines, full sweep)\n";
    std::cout << "  F  Eval — TAM multi-candidate Format A (force-assign argmax)\n";
    std::cout << "  H  Eval — TAM multi-candidate Format B (threshold 0.5 + retry)\n";
    std::cout << "  J  TAM multi-candidate correctness test (Format A/B, small bbox)\n";
    std::cout << "  K  GeoBox connectivity check (smoke_test + tous les caches)\n";
    std::cout << "  L  Baselines multi-agent LGPDP test (MCA/TP/TF/CA/PIBT/...)\n";
    std::cout << "  U  SoTA standalone benchmark (full-pipeline TP/CA/FCA/TF/RHCR, src/SoTA/)\n";
    std::cout << "  O  Combined Y + SoTA standalone evaluation (4 RL + 5 SoTA full-pipeline)\n";
    std::cout << "  Z  Congestion-only test (ghost traffic impact on 6 cities × 5 scenarios)\n";
    std::cout << "  Q  TAM + congestion validation (4 cities × 4 modes × 3 scenarios, ~2-4h)\n\n";
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
        // ── Option T — Unified training : MAPPO + IPPO + FaithfulMAPPER ──────
        //
        // PURPOSE
        //   Single training run that produces the 3 checkpoints needed by the
        //   Q / F evaluation. Environment matched to Q so train and eval see
        //   the same congestion regime / planning back-end (no train-test shift).
        //
        // WHAT THIS CONFIG DOES
        //   - Trains MAPPO, IPPO, and FaithfulMAPPER from scratch (fresh Xavier).
        //   - Saves each checkpoint at the path the eval options expect.
        //   - 3 Small training cities (Tokyo / Kyoto / LosAngeles) — Medium
        //     and Large are NOT trained on (DbVNS pathology on Medium ; Large
        //     is held out for generalization).
        //   - Pure training: no in-loop eval. Run Q after T for evaluation.
        //
        // ENVIRONMENT (matches Y and Q exactly)
        //   - DbVNS planning back-end (same as the eval pipeline).
        //   - Per-city proportional ghost congestion: 5% of ways are hot, with
        //     an average ghost density of 2 per hot way. Produces the same
        //     peak BPR (×2.5 – ×6) across cities regardless of graph size.
        //   - Heterogeneous agent capacity ∈ [3, 7].
        //   - Latency shaping + task value heterogeneity (Option M Phase 2+3).
        //   - Fleet shrink by 2 with load_per_agent=2 → same congestion
        //     footprint, ~2× faster compute.
        //
        // ROOM FOR ITERATION
        //   The input vector enrichment (Batch A/B features), the scenario
        //   sampling tweaks, and the reward shaping fixes we discussed will be
        //   layered on top of this config. Nothing here locks them out.
        //
        // ESTIMATED WALLCLOCK
        //   ~6-10 h on the new optimised stack (3 policies × 60 rounds × 3
        //   Small cities). FaithfulMAPPER is the slowest contributor (~12
        //   per-agent networks updated independently).
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results";
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root      = cache_root;
        cfg.output_dir      = output_dir;
        cfg.n_rounds        = 50;         // reduced from 60 for shorter wallclock
        cfg.n_seeds         = 1;          // seed 42 only (single training run)
                                          // raise to 2-3 for multi-seed retraining
                                          // (start_seed=42 default, so 42 / 42+43 / 42+43+44).
        cfg.save_policy     = true;
        cfg.verbose         = true;

        // Pure training. Q does the evaluation.
        cfg.train_only            = true;
        cfg.enable_generalization = false;

        // 5 training cities — 3 Small + 2 Large (Tokyo_Large + London) for
        // cross-scale generalisation. The policy sees both the short-trip
        // regime (Small) and the long-trip / dense-graph regime (Large)
        // BEFORE Q's held-out eval, so generalisation becomes a real test
        // of robustness rather than an out-of-distribution surprise.
        cfg.train_city_filter = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };

        // ── Planning back-end : same as eval ────────────────────────────────
        cfg.episode_cfg.use_dbvns_planning = true;

        // ── Scenario sampler : 3-regime distribution (slack removed) ────────
        // The slack regime (low density × many agents) carried a weak learning
        // signal — accept_rate trivially saturates to 1. Eval's `over_fleet`
        // scenario (agents×10) is already a held-out generalisation test, so
        // dropping slack does not introduce a train/eval shift on the
        // in-distribution evaluation regimes (normal, stress_light, stress_heavy).
        cfg.disable_slack_regime = true;

        // ── Congestion : matches Y / Q (per-city proportional) ──────────────
        cfg.episode_cfg.enable_ghost_traffic       = true;
        cfg.episode_cfg.ghost_hot_way_count        = 0;     // use fraction
        cfg.episode_cfg.ghost_hot_way_frac         = 0.05f; // 5% hot ways per city
        cfg.episode_cfg.ghost_density_per_hot_way  = 2.0f;  // n_max = 2 × n_hot
        cfg.episode_cfg.ghost_window_steps         = 8;
        cfg.episode_cfg.ghost_n_max                = 100;   // ignored (density)
        cfg.episode_cfg.ghost_n_max_user_set       = false;

        // ── Compute-amortisation : fleet ÷2 with load_per_agent=2 ───────────
        cfg.episode_cfg.fleet_size_divisor   = 2;
        cfg.episode_cfg.fleet_load_per_agent = 2;

        // ── Heterogeneous capacity + latency shaping + task value het ───────
        // Kept from Option M Phase 2/3 — these are what made the selectivity
        // signal trainable in the first place. To be revisited when the new
        // reward shaping lands.
        cfg.episode_cfg.enable_heterogeneous_capacity = true;
        cfg.episode_cfg.hetero_capacity_min           = 3;
        cfg.episode_cfg.hetero_capacity_max           = 7;

        cfg.episode_cfg.enable_latency_shaping     = true;
        cfg.episode_cfg.latency_shaping_lambda     = 0.4f;
        cfg.episode_cfg.latency_shaping_max_steps  = 1500;

        cfg.episode_cfg.enable_task_value_heterogeneity = true;
        cfg.episode_cfg.task_value_mul_min = 0.5f;
        cfg.episode_cfg.task_value_mul_max = 2.0f;
        cfg.episode_cfg.task_imp_min       = 0.3f;
        cfg.episode_cfg.task_imp_max       = 3.0f;

        // ── MC TAM Format A (matches Q) ─────────────────────────────────────
        // Train under the EXACT TAM regime Q uses so the policy learns the
        // right signal: be the argmax candidate among K bidders, not just
        // score > 0.5. Without this, T → Q is a train/eval distribution shift.
        cfg.episode_cfg.tam_multi_candidate     = true;
        cfg.episode_cfg.tam_mc_force_assign     = true;     // Format A
        cfg.episode_cfg.tam_mc_max_candidates   = 5;
        cfg.episode_cfg.tam_mc_ratio_min        = 1.4f;
        cfg.episode_cfg.tam_mc_ratio_max        = 3.0f;
        cfg.episode_cfg.tam_mc_ratio_scale      = 2000.f;

        // ── Checkpoint every 10 rounds (insurance against crash) ────────────
        cfg.checkpoint_every_rounds = 10;

        // ── Train MAPPO + IPPO + FaithfulMAPPER ─────────────────────────────
        cfg.train_modes = {
            PolicyMode::MAPPO,
            PolicyMode::IPPO,
            PolicyMode::FaithfulMAPPER
        };

        // Fresh Xavier init. Q picks up the produced checkpoints.
        cfg.load_policy = false;

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else if (rep == "M" || rep == "m") {
        // ── Option M — Diversified-scenario training (MAPPO + FaithfulMAPPER)
        //
        // PURPOSE
        //   Train MAPPO and FaithfulMAPPER from scratch under DbVNS planning
        //   with a richer scenario distribution than the May 17 baseline:
        //   the same 4-regime density/agents sampler, NOW PAIRED with a
        //   per-episode CongestionProfile that injects spatially heterogeneous
        //   "ghost" background traffic (CongestionMap::add_ghost_load).
        //
        //   Goal: create the conditions where selective accept/refuse choices
        //   start mattering — without temporally varying network conditions,
        //   throughput is largely planner-dominated and the RL signal is
        //   washed out.
        //
        // WHAT'S NEW vs Option T / Option N
        //   - cfg.episode_cfg.enable_ghost_traffic = true
        //       → GhostTrafficController active; n_max scaled per city size
        //         (Tokyo_Small≈20, Medium≈40, Large≈80) via
        //         customize_episode_for_city.
        //   - Each scenario draw is paired with a CongestionProfile from
        //     {Flat, Wave, RampUpDown, ShockBurst, BuildingUp} via the
        //     regime-aware sampler in MultiCityTrainer.
        //   - Trains MAPPO + FaithfulMAPPER (MAPPER-enhanced is skipped; the
        //     paper-faithful Liu+2020 variant is the comparison axis).
        //
        // WHAT STAYS THE SAME
        //   - 60 rounds × 1 seed × 6 train cities (Tokyo/Kyoto/LosAngeles × S/M)
        //   - DbVNS planning at the routing level (priority 1 — eliminates
        //     the train/test mismatch we identified).
        //   - REWARD SHAPING UNCHANGED — Phase 2 only if Phase 1 needs it.
        //   - train_only (no in-loop eval); Option X handles eval afterward.
        //
        // OUTPUTS
        //   results/policy_seedXX.bin          (MAPPO)
        //   results/mapper_faithful/...bin     (FaithfulMAPPER)
        //   results/episodes_seedXX.csv         — new columns:
        //     completion_per_accepted, unfinished_accept_rate,
        //     mean_congestion_at_decision, n_ghost_active_mean,
        //     congestion_profile.
        //
        // ESTIMATED WALLCLOCK
        //   ~6-8 h (MAPPO + FaithfulMAPPER × 60 rounds × 6 cities). Ghost
        //   controller is O(n_max) per step → negligible overhead vs the
        //   per-step A* and policy work.
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results";
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root      = cache_root;
        cfg.output_dir      = output_dir;
        cfg.n_rounds        = 80;   // bumped from 60 — extra rounds for convergence
        cfg.n_seeds         = 2;    // 2 seeds (rng=42, 43) → mean ± std for analysis
        cfg.save_policy     = true;
        cfg.verbose         = true;

        // Pure training run — all eval phases disabled.
        cfg.train_only = true;
        // Skip loading held-out cities (Tokyo_Large/Fukuoka/NewYork/Paris/London).
        // train_only already skips the eval phases, but the trainer still LOADS
        // these multi-GB OSM graphs by default — wasteful when no eval runs
        // against them. Saves ~5-10 min of startup + several GB of RAM.
        cfg.enable_generalization = false;

        // Drop Medium-sized training cities (Tokyo_Medium, Kyoto_Medium,
        // LosAngeles_Medium). The diagnostic K run showed DbVNS+TD-A* is
        // pathological on these graphs (~3-5 min/episode for <10% throughput),
        // which would push Option M wallclock to 24-30h. Small cities give
        // the same architectural signal in ~6-8h.
        cfg.train_city_filter = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };

        // Planning = forward DbVNS-PDP (priority 1 from the analysis).
        cfg.episode_cfg.use_dbvns_planning = true;

        // ── Diversified scenarios ──────────────────────────────────────────
        // 4-regime density/agents sampler PLUS ghost-traffic congestion mix.
        // disable_slack_regime stays false so we get the full 5-profile
        // distribution including slack/wave (where the policy learns to be
        // picky about task quality, not just to refuse under saturation).
        cfg.disable_slack_regime = false;
        cfg.episode_cfg.enable_ghost_traffic = true;
        // Per-city ghost_n_max is set inside customize_episode_for_city; the
        // value below is the default for any unknown-sized graph and is
        // overwritten before the episode runs.
        cfg.episode_cfg.ghost_n_max          = 40;
        cfg.episode_cfg.ghost_window_steps   = 5;
        cfg.episode_cfg.ghost_hot_way_frac   = 0.30f;

        // ── Phase 2: latency-aware delivery shaping ────────────────────────
        // Makes selectivity profitable: a task that ends up taking forever
        // pays less reward. Pickup credit (0.3 × imp × reward) and penalties
        // (refuse, unfinished) are NOT touched, so the indifference structure
        // is preserved. Conservative defaults: lambda=0.4, max=1500 steps.
        cfg.episode_cfg.enable_latency_shaping  = true;
        cfg.episode_cfg.latency_shaping_lambda  = 0.4f;
        cfg.episode_cfg.latency_shaping_max_steps = 1500;

        // ── Heterogeneous per-agent capacity (Option M diversification) ────
        // Each agent gets a uniform draw in [3, 7] at the start of each
        // episode. Fleet mixes light couriers (capa 3) with heavy carriers
        // (capa 7) — the policy must learn to discriminate based on the
        // winning agent's free slots, not just its idle/active status.
        cfg.episode_cfg.enable_heterogeneous_capacity = true;
        cfg.episode_cfg.hetero_capacity_min = 3;
        cfg.episode_cfg.hetero_capacity_max = 7;

        // ── Phase 3: task-value heterogeneity ──────────────────────────────
        // Decouple task VALUE from task EFFORT so the policy faces a real
        // "good task vs bad task" tradeoff (in addition to "fast vs slow"
        // already injected by Phase 2 latency shaping). Per-task reward
        // multiplier ∈ [0.5, 2.0] independent of distance, importance ∈
        // [0.3, 3.0]. Effective reward × imp range ≈ 40× between worst and
        // best task → strong discrimination signal.
        cfg.episode_cfg.enable_task_value_heterogeneity = true;
        cfg.episode_cfg.task_value_mul_min = 0.5f;
        cfg.episode_cfg.task_value_mul_max = 2.0f;
        cfg.episode_cfg.task_imp_min = 0.3f;
        cfg.episode_cfg.task_imp_max = 3.0f;

        // ── Intermediate checkpoint every 20 rounds ────────────────────────
        // Insurance against crash / interrupt during the ~2h training run.
        // Each snapshot overwrites the same MAPPO + FaithfulMAPPER subdirs,
        // so disk usage stays flat (only latest is kept).
        cfg.checkpoint_every_rounds = 20;

        // ── Train modes: MAPPO + FaithfulMAPPER ────────────────────────────
        //   MAPPO          — centralised critic, shared actor (CTDE).
        //   FaithfulMAPPER — paper-faithful Liu+2020 (probabilistic copy of
        //                    best, no mutation). Article-aligned baseline.
        // MAPPER-enhanced and IPPO are intentionally excluded.
        cfg.train_modes = {
            PolicyMode::MAPPO,
            PolicyMode::FaithfulMAPPER
        };

        // Fresh Xavier init; Option X picks up the resulting checkpoints.
        cfg.load_policy = false;

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
        // ── Full comparative benchmark — RL policies vs SoTA baselines ────────
        // Reformulated from the legacy "SoTA-only" Y to a comprehensive eval
        // including the trained RL policies side-by-side with published
        // baselines. Single CSV, single seed sequence, deterministic
        // per-(city, scenario, ep) seeding → every mode plays exactly the
        // same task streams + ghost traffic + hetero capacity for fair
        // policy-vs-policy comparison.
        //
        // MODES (10) :
        //   ── Our RL policies (with checkpoints from Option M Phase 2+3) ──
        //   - MAPPO            [our work]   centralised critic + shared actor
        //   - FaithfulMAPPER   [our work]   per-agent evolutionary, paper-faithful
        //   - Hybrid           [our work]   frozen MAPPO + online REINFORCE residual
        //   - TamAlwaysAccept  [ablation]   TAM routing only, no policy
        //
        //   ── Published SoTA baselines (heuristic / hand-crafted) ─────────
        //   - MCA              [Chen+2021 ICRA]      Marginal-cost assignment
        //   - CongestionAware  [Liu, Saha+]          Pickup-leg dynamic cost
        //   - TrafficFlow      [Chen+2024 AAAI]      Full-trip dynamic cost
        //   - TokenPassing     [Ma+2017 AAMAS]       Decoupled MAPD min h
        //   - PIBT             [Okumura+2022]        Load-balanced priority
        //   - DoubleHorizon    [Mitrovic-Minic+2004] Horizon-aware insertion
        //
        // ENVIRONMENT (matched to Option M training) :
        //   - DbVNS planning
        //   - Ghost traffic ON (5 profiles, regime-aware sampling)
        //   - Heterogeneous per-agent capacity [3, 7]
        //
        // SCENARIOS (4) :
        //   slack         : density 0.5 × agents 1.2
        //   normal        : density 1.0 × agents 1.0
        //   stress_light  : density 1.5 × agents 0.8
        //   stress_heavy  : density 2.0 × agents 0.6
        //
        // CITIES (7) — same as Option X for direct comparability :
        //   Train scope  : Tokyo_Small, Kyoto_Small, LosAngeles_Small
        //   Held-out     : Tokyo_Large, Paris, NewYork  (London dropped: ~3-5h saved per mode)
        //
        // Coverage: 10 modes × 7 cities × 4 scenarios × 3 eps = 840 episodes.
        // Estimated wallclock: ~12-20h on a single seed (MAPPO pathology
        // on large held-out cities is the dominant cost).
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

        cfg.eval_only            = true;
        cfg.skip_stress_eval     = true;   // the 4-scenario sweep already covers stress
        cfg.skip_generalize_eval = false;

        // Our RL policies + ablation + 6 published SoTA baselines.
        cfg.eval_modes = {
            PolicyMode::MAPPO,
            PolicyMode::FaithfulMAPPER,
            PolicyMode::IPPO,
            PolicyMode::Hybrid,
            PolicyMode::CongestionAware,
            PolicyMode::TrafficFlow,
            PolicyMode::TokenPassing
        };

        // ── Y scenario sweep — congestion-profile × fleet-density diversity ──
        //
        // Goal: stress-test every architecture across the FULL congestion
        // landscape (rush-hour, multi-peak, incident, monotone growth, calm)
        // AND across the FULL fleet/density landscape (under-provisioned →
        // over-provisioned by ~10×). This is the publication-grade sweep
        // that gives reviewer-credible evidence on robustness.
        //
        // Per-scenario fields (density_mult, agents_mult, label, profile):
        //   density_mult : multiplies the per-phase Poisson task arrival rate.
        //   agents_mult  : multiplies the per-phase active-agent count. Pool
        //                  is sized via cfg.episode_cfg.agent_pool_multiplier
        //                  (bumped to 10 below) so values up to 10× are honoured.
        //   profile      : temporal shape of background ghost congestion
        //                  (see CongestionProfile in GhostTrafficController.hpp).
        //
        // Reading the matrix:
        //   normal_wave         — baseline: nominal fleet, single-peak congestion.
        //   normal_ramp         — multi-peak (matin/midi/après-midi/soir).
        //   stress_shock        — incident: brief 100% traffic spike mid-episode.
        //   stress_buildup      — saturation: linear 0 → 100% congestion buildup.
        //   over_fleet          — over-provisioned fleet (10× agents) at low
        //                          density: tests selectivity, free-agent
        //                          discrimination (TokenPassing), and load
        //                          balancing (PIBT/TrafficFlow) under abundance.
        //
        // Coverage: 5 scenarios × 6 modes × 6 cities × 3 eps = 540 eval episodes.
        cfg.eval_scenarios = {
            EpisodeScenario{ 1.0f,  1.0f, "normal_wave",    CongestionProfile::Wave        },
            EpisodeScenario{ 1.0f,  1.0f, "normal_ramp",    CongestionProfile::RampUpDown  },
            EpisodeScenario{ 1.5f,  0.8f, "stress_shock",   CongestionProfile::ShockBurst  },
            EpisodeScenario{ 2.0f,  0.6f, "stress_buildup", CongestionProfile::BuildingUp  },
            EpisodeScenario{ 0.5f, 10.0f, "over_fleet",     CongestionProfile::Wave        },
        };

        // Match Option X environment exactly so RL policies are evaluated
        // under the same conditions they saw during training, and SoTA
        // baselines face the same congestion challenges.
        cfg.episode_cfg.use_dbvns_planning            = true;
        cfg.episode_cfg.enable_ghost_traffic          = true;

        // ── Meaningful-slowdown ghost parameters (publication-grade Y eval) ──
        //
        // Diagnostic Option Z proved the legacy params (n_max=20-80 dispersed
        // over 11k-360k hot ways) produce peak load 1-2 and BPR ×1.000-×1.004
        // — essentially no impact on edge costs, so dynamic-cost baselines
        // cannot differentiate themselves from static-cost ones.
        //
        // The two changes below force ghosts to PILE UP on a small set of
        // arteries so collisions create real BPR jams:
        //
        //   ghost_hot_way_count  = 150  (absolute, overrides hot_way_frac)
        //   ghost_n_max          = 300  (same across cities)
        //
        // Expected behaviour at Wave/Ramp/Shock peaks (λ ≈ n_max / hot_count = 2.0):
        //   - avg ghosts per hot way ≈ 300/150 = 2.0
        //   - Poisson tail: ~150 × 5%   = ~8 edges with load ≥ 5 → BPR ×1.15-×1.66
        //   - Peak load typical: 5-7 → BPR ×1.15 to ×1.66
        //   - Rare edges at load 8+ → BPR ×2 (heaviest local jam)
        //
        // This sits in the "meaningful slowdown" band (BPR ×1.1-×2.0) per Option Z's
        // interpretation legend — strong enough that dynamic-cost baselines can
        // differentiate themselves, mild enough that the simulation remains
        // realistic (no ×10 pathological jams that would dominate every route).
        //
        // ghost_n_max_user_set=true tells customize_episode_for_city NOT to
        // reset ghost_n_max to its per-city tier value (20/40/80), so the
        // 300 above is respected across all eval cities.
        //
        // ghost_window_steps bumped to 8 so each ghost lingers a bit longer,
        // stabilising the average load profile (less Poisson churn).
        // ── Option O — REDUCED congestion + REDUCED fleet (vs Y) ────────────
        // Goal: keep meaningful congestion for evaluation BUT speed up the
        // sweep by ~30-50% compared to Y's full-density regime.
        //
        // Changes vs Y:
        //   ghost_density_per_hot_way : 2.0 → 1.4   (-30% ghost load per hot way)
        //   fleet_size_divisor        : 2   → 3     (33% fewer real agents)
        //   fleet_load_per_agent      : 2   → 2     (unchanged)
        //
        // Expected congestion impact (vs Y):
        //   Per-edge ghost BPR factor : -30%
        //   Real-agent fleet count    : -33% (less compute, less real-agent congestion)
        //   Net: ~25-35% lower congestion footprint, ~33% faster compute
        //
        // Validation: ran Option Z post-change to confirm meaningful BPR
        // remains (target range ×1.05–×1.35, i.e. above the "no impact"
        // floor of ×1.00–×1.05 but below the saturated ×1.5+ regime).
        cfg.episode_cfg.ghost_hot_way_count           = 0;     // use fraction
        cfg.episode_cfg.ghost_hot_way_frac            = 0.05f; // unchanged (route coverage)
        cfg.episode_cfg.ghost_density_per_hot_way     = 1.4f;  // ↓ from 2.0 (−30%)
        cfg.episode_cfg.ghost_window_steps            = 8;
        cfg.episode_cfg.ghost_n_max                   = 100;
        cfg.episode_cfg.ghost_n_max_user_set          = false;

        cfg.episode_cfg.enable_heterogeneous_capacity = true;
        cfg.episode_cfg.hetero_capacity_min           = 3;
        cfg.episode_cfg.hetero_capacity_max           = 7;

        // Fewer real agents (faster compute). fleet_load_per_agent stays at 2,
        // so each real agent still contributes 2 BPR units — congestion is
        // 33% lower (not 50%) because per-agent multiplier is unchanged.
        cfg.episode_cfg.fleet_size_divisor            = 3;     // ↑ from 2
        cfg.episode_cfg.fleet_load_per_agent          = 2;     // unchanged

        // Bump the agent pool so the over_fleet scenario (agents_mult = 10)
        // is actually instantiated. Default 1.5 clamps to 1.5× phase max,
        // which would silently collapse over_fleet to ~standard fleet size.
        cfg.episode_cfg.agent_pool_multiplier         = 10.0f;

        // Load RL checkpoints from Option M Phase 2+3 training.
        cfg.load_policy                 = true;
        cfg.policy_path                 = "C:\\ConflictualMAS\\results\\mappo\\"
                                          "policy_seed42.bin";
        cfg.mapper_policy_path          = "C:\\ConflictualMAS\\results\\mapper_enhanced\\"
                                          "mapper_seed42.bin";
        cfg.faithful_mapper_policy_path = "C:\\ConflictualMAS\\results\\mapper_faithful\\"
                                          "mapper_seed42.bin";

        // Same city scope as Option X for direct comparability.
        cfg.train_city_filter     = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };
        cfg.enable_generalization = true;
        cfg.gen_city_filter       = {
            "Tokyo_Medium", "Paris", "NewYork"
        };

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
        // MAPPER dropped: its checkpoint comes from a legacy training run with
        // different conditions (no Phase 2, no hetero capa) — mixing it in
        // would muddy the comparison. The 4-way scope is the focused user ask
        // (MAPPO vs FaithfulMAPPER vs TAA) plus Hybrid as a MAPPO + online
        // adaptation control.
        cfg.eval_modes = {
            PolicyMode::MAPPO,
            PolicyMode::FaithfulMAPPER,
            PolicyMode::TamAlwaysAccept,
            PolicyMode::CongestionAware,
            PolicyMode::TrafficFlow,
            PolicyMode::TokenPassing
        };

        // ── Eval conditions matched to Option M training ──────────────────────
        // DbVNS planning, ghost traffic, heterogeneous capacity — all ON so
        // the policies are evaluated under the same environment they saw
        // during training. Latency reward shaping has NO effect at eval time
        // (train_mode=false → no buffer writes), so we leave it off.
        cfg.episode_cfg.use_dbvns_planning            = true;
        cfg.episode_cfg.enable_ghost_traffic          = true;
        cfg.episode_cfg.ghost_n_max                   = 40;
        cfg.episode_cfg.ghost_window_steps            = 5;
        cfg.episode_cfg.ghost_hot_way_frac            = 0.30f;
        cfg.episode_cfg.enable_heterogeneous_capacity = true;
        cfg.episode_cfg.hetero_capacity_min           = 3;
        cfg.episode_cfg.hetero_capacity_max           = 7;

        // Load checkpoints from the latest Option M training (Phase 2 +
        // hetero capa, 2 seeds, 80 rounds). MAPPER-enhanced is loaded from
        // a legacy run for reference but is excluded from eval_modes below.
        //   policy_path                  → MAPPO (Option M Phase 2)
        //   mapper_policy_path           → legacy (not in eval_modes)
        //   faithful_mapper_policy_path  → FaithfulMAPPER (Option M Phase 2)
        cfg.load_policy                 = true;
        cfg.policy_path                 = "C:\\ConflictualMAS\\results\\mappo\\"
                                          "policy_seed42.bin";
        cfg.mapper_policy_path          = "C:\\ConflictualMAS\\results\\mapper_enhanced\\"
                                          "mapper_seed42.bin";
        cfg.faithful_mapper_policy_path = "C:\\ConflictualMAS\\results\\mapper_faithful\\"
                                          "mapper_seed42.bin";

        // Eval scope: 3 train cities (all Small) + 4 held-out cities.
        // Excluded: the Medium training cities (Tokyo/Kyoto/LA Medium) — never
        // loaded due to DbVNS pathology (~3-5 min/episode for <10% throughput).
        // Held-out includes Tokyo_Large to test generalisation to a larger
        // scale of the same city family.
        //
        // Expected wallclock (4 modes × 5 eps × 2 scenarios per city for train,
        // 4 modes × 5 eps × 1 scenario per city for held-out):
        //   Tokyo_Small       ~10 min
        //   Kyoto_Small       ~50 min
        //   LosAngeles_Small  ~35 min
        //   Tokyo_Large       ~1-3 h  (large graph, DbVNS slowness risk)
        //   Paris             ~20 min
        //   NewYork           ~1-3 h
        //   London            ~3-5 h (largest graph)
        // Total: ~7-13 h. Run overnight.
        cfg.train_city_filter     = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };
        cfg.enable_generalization = true;
        cfg.gen_city_filter       = {
            "Tokyo_Large", "Paris", "NewYork"
        };

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else if (rep == "F" || rep == "f" || rep == "H" || rep == "h") {
        // ── Eval — TAM multi-candidate mode (Format A or Format B) ───────────
        //
        // Identical to Option X (same 4 modes, 7 cities, environment, scope,
        // checkpoints, deterministic per-(city,scenario,ep) seeding) EXCEPT
        // the TAM runs in multi-candidate mode instead of legacy first-fit:
        //
        //   1. Dijkstra expands until the first valid candidate at
        //      x = max(pickup_cost, delivery_cost).
        //   2. Expansion continues to budget = x * ratio(x) (ratio decreasing
        //      3.0 → 1.4 with x), or until 5 candidates, or both exhaust.
        //   3. All candidates scored by the policy (same try_accept_task call
        //      path as legacy); the argmax wins.
        //
        //   Option F — Format A: always allocate to the argmax.
        //   Option H — Format B: if every score < 0.5, the task is DEFERRED
        //              and re-offered after 20% of the episode; past 70% of
        //              the episode it is rejected outright.
        //
        // Outputs go to a dedicated subdir so Option X / Y CSVs are untouched:
        //   Option F → results/mc_format_A/episodes_seed42.csv
        //   Option H → results/mc_format_B/episodes_seed42.csv
        //
        // Recovery: the legacy TAM is restored by simply not running F/H
        // (tam_multi_candidate defaults to false for every other option).
        const bool format_a = (rep == "F" || rep == "f");

        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = format_a
            ? "C:\\ConflictualMAS\\results\\mc_format_A"
            : "C:\\ConflictualMAS\\results\\mc_format_B";
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root       = cache_root;
        cfg.output_dir       = output_dir;
        cfg.n_seeds          = 1;
        cfg.n_eval_episodes  = 5;
        cfg.verbose          = true;

        cfg.eval_only            = true;
        cfg.skip_stress_eval     = false;
        cfg.skip_generalize_eval = false;

        // Same 4-mode line-up as Option X — all TAM-driven, so all are
        // affected by the multi-candidate switch. (SoTA baselines bypass the
        // TAM entirely; compare against Option Y for those.)
        cfg.eval_modes = {
            PolicyMode::MAPPO,
            PolicyMode::FaithfulMAPPER,
            PolicyMode::TamAlwaysAccept,
            PolicyMode::CongestionAware,
            PolicyMode::TrafficFlow,
            PolicyMode::TokenPassing
        };

        // Environment matched to Option M training / Option X eval.
        cfg.episode_cfg.use_dbvns_planning            = true;
        cfg.episode_cfg.enable_ghost_traffic          = true;
        cfg.episode_cfg.ghost_n_max                   = 40;
        cfg.episode_cfg.ghost_window_steps            = 5;
        cfg.episode_cfg.ghost_hot_way_frac            = 0.30f;
        cfg.episode_cfg.enable_heterogeneous_capacity = true;
        cfg.episode_cfg.hetero_capacity_min           = 3;
        cfg.episode_cfg.hetero_capacity_max           = 7;

        // Compute-amortisation (F/H): shrink fleet by 2, double per-agent
        // load. ~2× speedup on A* / policy with preserved congestion footprint.
        cfg.episode_cfg.fleet_size_divisor            = 2;
        cfg.episode_cfg.fleet_load_per_agent          = 2;

        // ── TAM multi-candidate switch ──────────────────────────────────────
        cfg.episode_cfg.tam_multi_candidate     = true;
        cfg.episode_cfg.tam_mc_force_assign     = format_a;  // F = true, H = false
        cfg.episode_cfg.tam_mc_max_candidates   = 5;
        cfg.episode_cfg.tam_mc_ratio_min        = 1.4f;
        cfg.episode_cfg.tam_mc_ratio_max        = 3.0f;
        cfg.episode_cfg.tam_mc_ratio_scale      = 2000.0f;
        cfg.episode_cfg.tam_mc_recall_time_frac = 0.20f;     // Format B retry delay
        cfg.episode_cfg.tam_mc_reject_time_frac = 0.70f;     // Format B reject cutoff

        // Checkpoints — same as Option X (Option M Phase 2+3).
        cfg.load_policy                 = true;
        cfg.policy_path                 = "C:\\ConflictualMAS\\results\\mappo\\"
                                          "policy_seed42.bin";
        cfg.mapper_policy_path          = "C:\\ConflictualMAS\\results\\mapper_enhanced\\"
                                          "mapper_seed42.bin";
        cfg.faithful_mapper_policy_path = "C:\\ConflictualMAS\\results\\mapper_faithful\\"
                                          "mapper_seed42.bin";

        // Same city scope as Option X for direct comparability.
        cfg.train_city_filter     = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };
        cfg.enable_generalization = true;
        cfg.gen_city_filter       = {
            "Tokyo_Large", "Paris", "NewYork"
        };

        std::cout << "[Option " << (format_a ? "F" : "H")
                  << "] TAM multi-candidate, Format " << (format_a ? "A" : "B")
                  << "\n";

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else if (rep == "J" || rep == "j") {
        // ── TAM multi-candidate correctness test ──────────────────────────────
        // Small central Tokyo bbox (same GeoBox as training smoke test S).
        // Reuses cache_dir so smoke_test.json is shared between S and J.
        run_tam_mc_test(osm_file, cache_dir);
    }
    else if (rep == "K" || rep == "k") {
        // GeoBox connectivity check
        const std::string smoke   = cache_dir + "/smoke_test.json";
        const std::string data_dir = "C:\\ConflictualMAS\\data\\cache";
        run_geobox_connectivity_test({ smoke }, "smoke_test");
        run_geobox_connectivity_test({
            data_dir + "/Tokyo_Small.json",
            data_dir + "/Tokyo_Medium.json",
            data_dir + "/Tokyo_Large.json",
            data_dir + "/Kyoto_Small.json",
            data_dir + "/Kyoto_Medium.json",
            data_dir + "/Kyoto.json",
            data_dir + "/LosAngeles_Small.json",
            data_dir + "/LosAngeles_Medium.json",
            data_dir + "/LosAngeles.json",
            data_dir + "/NewYork.json",
            data_dir + "/Paris.json",
            data_dir + "/London.json",
            data_dir + "/Fukuoka.json"
        }, "production caches");
    }
    else if (rep == "Z" || rep == "z") {
        // Congestion-only test: measure REAL impact of ghost traffic on edge
        // costs, across all 6 Y-eval cities × 5 scenarios. No agents involved.
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        run_congestion_only_test(osm_root, cache_root);
    }
    else if (rep == "Q" || rep == "q") {
        // ── Option Q — Focused TAM + congestion validation (~2-4h) ─────────────
        //
        // Purpose
        //   Validate, in a single overnight-friendly run, that:
        //     (a) the TAM remains efficient under the new meaningful-slowdown
        //         congestion regime (mean_agents_offered_per_task << n_active);
        //     (b) the ghost-traffic injection actually impacts the agents that
        //         traverse the network (mean_congestion_at_decision > 0,
        //         route_congestion_exposure > 0);
        //     (c) the 3 TAM-driven modes (MAPPO / FaithfulMAPPER / TamAlwaysAccept)
        //         + 1 SoTA reference (CongestionAware) produce coherent metrics
        //         on which to base the next-iteration policy changes.
        //
        // Coverage
        //   - 4 cities  : 3 train (Tokyo_Small, Kyoto_Small, LosAngeles_Small)
        //                 + 1 held-out (Tokyo_Large)
        //   - 4 modes   : MAPPO, FaithfulMAPPER, TamAlwaysAccept, CongestionAware
        //   - 3 scenarios : normal_wave, normal_ramp, stress_shock
        //   - 3 episodes per (city, mode, scenario)
        //   Total : 4 × 4 × 3 × 3 = 144 episodes (≈2-4h on the new strong-
        //   congestion config; Tokyo_Large is the dominant cost).
        //
        // Output
        //   results/tam_validation/episodes_seed42.csv
        //   (new columns include mean_agents_offered_per_task,
        //    mean_recall_rounds_per_task, mean_candidates_scored_per_task)
        //
        // Recovery
        //   Skipping Option Q has zero impact on F/H/X/Y artifacts (separate
        //   output_dir). Re-running it overwrites only its own subdir.
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results\\tam_validation";
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root       = cache_root;
        cfg.output_dir       = output_dir;
        cfg.n_seeds          = 1;
        cfg.n_eval_episodes  = 3;
        cfg.verbose          = true;

        cfg.eval_only            = true;
        cfg.skip_stress_eval     = true;   // 3-scenario sweep covers stress
        cfg.skip_generalize_eval = false;  // Tokyo_Large is held-out

        // The 3 RL policies trained by Option T + 1 SoTA reference
        // (CongestionAware). MAPPO and IPPO probe the centralised-vs-local
        // critic question, FaithfulMAPPER the per-agent population question.
        cfg.eval_modes = {
            PolicyMode::MAPPO,
            PolicyMode::IPPO,
            PolicyMode::FaithfulMAPPER,
            PolicyMode::CongestionAware
        };

        // 3 scenarios covering the meaningful congestion regime.
        cfg.eval_scenarios = {
            EpisodeScenario{ 1.0f, 1.0f, "normal_wave",  CongestionProfile::Wave        },
            EpisodeScenario{ 1.0f, 1.0f, "normal_ramp",  CongestionProfile::RampUpDown  },
            EpisodeScenario{ 1.5f, 0.8f, "stress_shock", CongestionProfile::ShockBurst  },
        };

        // ── Environment: per-city proportional congestion (Y + Q matched) ─────
        // Same per-edge BPR profile across cities (uniform density), but the
        // hot pool covers a proportional FRACTION of each city's network →
        // every city's agents have a similar probability of routing through
        // a hot way. This is what makes cross-city generalization comparisons
        // fair (Tokyo_Small vs Tokyo_Large).
        //
        //   ghost_hot_way_frac        = 0.10    → 10% of city ways are hot.
        //                                          Route coverage > 95% on every
        //                                          city (prob a 30-edge route
        //                                          crosses a hot way = 1-0.9^30).
        //   ghost_density_per_hot_way = 2.0     → avg 2 ghosts/hot way.
        //                                          n_max derived per city:
        //                                          Tokyo_Small ~14k, Tokyo_Large
        //                                          ~240k. Preserves the BPR
        //                                          profile observed in Z
        //                                          (peak load 5-9, BPR ×1.5-×2.6).
        //   ghost_window_steps        = 8       → each ghost lingers 8 steps.
        cfg.episode_cfg.use_dbvns_planning            = true;
        cfg.episode_cfg.enable_ghost_traffic          = true;
        cfg.episode_cfg.ghost_hot_way_count           = 0;     // use fraction
        cfg.episode_cfg.ghost_hot_way_frac            = 0.05f; // 5% — proportional coverage
        cfg.episode_cfg.ghost_density_per_hot_way     = 2.0f;
        cfg.episode_cfg.ghost_window_steps            = 8;
        cfg.episode_cfg.ghost_n_max                   = 100;   // ignored (density-derived)
        cfg.episode_cfg.ghost_n_max_user_set          = false; // tier-override fine, density supersedes
        cfg.episode_cfg.enable_heterogeneous_capacity = true;
        cfg.episode_cfg.hetero_capacity_min           = 3;
        cfg.episode_cfg.hetero_capacity_max           = 7;
        // Default pool multiplier (no over_fleet scenario here).

        // Compute-amortisation (F/H): shrink fleet by 2, double per-agent load
        // so the congestion footprint is preserved. ~2× speedup on A* / policy.
        cfg.episode_cfg.fleet_size_divisor            = 2;
        cfg.episode_cfg.fleet_load_per_agent          = 2;

        // ── TAM multi-candidate Format A (matches Option F) ─────────────────
        // Force-assign argmax: the TAM scores K candidates and the best wins
        // unconditionally. Empirically the strongest mode for MAPPO/FM
        // (F eval: MAPPO 0.549 vs H Format B 0.489). Gives the upper-bound
        // policy performance and keeps the comparison apples-to-apples with F.
        cfg.episode_cfg.tam_multi_candidate     = true;
        cfg.episode_cfg.tam_mc_force_assign     = true;     // Format A
        cfg.episode_cfg.tam_mc_max_candidates   = 5;
        cfg.episode_cfg.tam_mc_ratio_min        = 1.4f;
        cfg.episode_cfg.tam_mc_ratio_max        = 3.0f;
        cfg.episode_cfg.tam_mc_ratio_scale      = 2000.0f;

        // Checkpoints — paths match Option T's save layout.
        cfg.load_policy                 = true;
        cfg.policy_path                 = "C:\\ConflictualMAS\\results\\mappo\\"
                                          "policy_seed42.bin";
        cfg.ippo_policy_path            = "C:\\ConflictualMAS\\results\\ippo_faithful\\"
                                          "ippo_seed42.bin";
        cfg.faithful_mapper_policy_path = "C:\\ConflictualMAS\\results\\mapper_faithful\\"
                                          "mapper_seed42.bin";

        // City scope: 3 train Small + 1 held-out Large for the TAM under load.
        cfg.train_city_filter     = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };
        cfg.enable_generalization = true;
        cfg.gen_city_filter       = { "Tokyo_Large" };

        std::cout << "[Option Q] TAM + congestion validation\n";

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else if (rep == "L" || rep == "l") {
        // Baselines multi-agent LGPDP test — exercises every published baseline
        // (MCA, TokenPassing, TrafficFlow, CongestionAware, PIBT, LaCAM,
        // DoubleHorizon, InsertionGreedy, Greedy) in our actual evaluation
        // context: multi-agent, capacitated (homogeneous + heterogeneous),
        // ghost-traffic congestion, no predefined depot. Uses the smoke_test
        // cache so first run reuses the option-S OSM parse.
        run_baselines_multi_agent_test(osm_file, cache_dir);
    }
    else if (rep == "U" || rep == "u") {
        // ── Full-pipeline SoTA standalone benchmark ─────────────────────────
        // Runs the 5 standalone solvers from src/SoTA/ (TokenPassing,
        // CongestionAware, FaithfulCongestionAware, TrafficFlow, RHCR) on
        // a (city × scenario × episode) grid, each solver in its OWN full
        // pipeline (own allocation + own path planning + own internal logic)
        // sharing only the simulation infrastructure (task stream, ghost
        // traffic, congestion map, agent starting positions).
        //
        // This is DISTINCT from Option Y, which uses the allocation-only
        // branches in EpisodeRunner sharing DbVNS planning. Option U is the
        // fair full-pipeline comparison: each method uses its own algorithms.
        //
        // Outputs results/sota_standalone/sota_standalone.csv with one row
        // per (solver, city, scenario, episode).
        std::cout << "\n=== SoTA Standalone Benchmark (Option U) ===\n";

        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string output_dir = "C:\\ConflictualMAS\\results\\sota_standalone";
        std::filesystem::create_directories(output_dir);
        const std::string csv_path   = output_dir + "\\sota_standalone.csv";

        CityRegistry::set_osm_root(osm_root);

        // Use the smoke_test bbox for now — keep the runtime tractable.
        // The first full sweep can extend to all Option Y cities later.
        const std::string cache_path = cache_dir + "/smoke_test.json";
        GeoBox geo_box;
        if (GeoBoxManager::cache_exists(cache_path)) {
            std::cout << "  GeoBox <- cache (" << cache_path << ")\n";
            geo_box = GeoBoxManager::load_geobox(cache_path);
        } else {
            std::cout << "  GeoBox <- OSM (will be cached)\n";
            geo_box = create_geo_box(osm_file, 139.695, 35.670, 139.740, 35.710);
            if (geo_box.is_valid) GeoBoxManager::save_geobox(geo_box, cache_path);
        }
        if (!geo_box.is_valid) {
            std::cerr << "  GeoBox invalid; aborting.\n";
            return 0;
        }
        Pathfinder pathfinder(geo_box);

        // Episode config — same shape as BaselinesMultiAgentTest so the
        // comparison is comparable to that smoke test.
        EpisodeConfig cfg;
        cfg.speed_mps          = 5.f;
        cfg.min_task_dist_m    = 80.f;
        cfg.max_task_dist_m    = 3000.f;
        cfg.max_tasks_per_agent = 3;
        cfg.phases = {
            { 200, 8.f,  4, 5, 0.0f, 2 },
            { 200, 10.f, 5, 6, 1.0f, 3 },
        };
        cfg.enable_ghost_traffic = true;
        cfg.ghost_n_max          = 20;
        cfg.ghost_window_steps   = 5;
        cfg.ghost_hot_way_frac   = 0.30f;

        // The scenarios sampled in this sweep (4 here, matching Option Y).
        struct ScenarioSpec {
            float       density_mult;
            float       agents_mult;
            const char* label;
            CongestionProfile profile;
        };
        const std::vector<ScenarioSpec> scenarios = {
            { 1.0f, 1.0f, "normal",      CongestionProfile::Wave        },
            { 1.5f, 0.8f, "stress",      CongestionProfile::ShockBurst  },
            { 0.5f, 1.2f, "slack",       CongestionProfile::RampUpDown  },
        };

        const int n_episodes_per_scenario = 3;

        // Open the CSV.
        SolverCSVLogger logger(csv_path, /*append=*/false);
        logger.write_header();

        // Make the city label match the smoke_test bbox identifier.
        const std::string city_label = "smoke_test";

        for (const auto& sc_spec : scenarios) {
            for (int ep = 0; ep < n_episodes_per_scenario; ++ep) {
                EpisodeScenario sc;
                sc.density_mult       = sc_spec.density_mult;
                sc.agents_mult        = sc_spec.agents_mult;
                sc.label              = sc_spec.label;
                sc.congestion_profile = sc_spec.profile;

                const uint32_t ep_seed =
                    static_cast<uint32_t>(0xC0DE0000 + ep * 7919);

                std::cout << "\n[" << sc.label << " ep=" << ep
                          << " seed=" << ep_seed << "]\n";

                // One SolverRunner per (scenario, ep). Each run() call
                // resets the shared congestion map and regenerates the
                // exact same task stream, so all 5 solvers face byte-
                // identical environments.
                SolverRunner runner(cfg, geo_box, pathfinder, sc, ep_seed);

                // Run each solver in turn. Order does not matter (each
                // gets a fresh CongestionMap).
                auto run_and_log = [&](ISolver& s) {
                    SolverMetrics m = runner.run(s);
                    m.city_label    = city_label;
                    m.episode       = ep;
                    logger.write_row(m);
                    std::cout << "  "
                              << m.solver_name << "  thr=" << m.throughput_rate
                              << "  lat=" << m.latency_mean
                              << "  ms="  << m.wallclock_ms
                              << "  alloc/ms="
                              << (m.compute_time_per_decision_us / 1000.0)
                              << "\n";
                };

                {  TokenPassingSolver        s; run_and_log(s); }
                {  CongestionAwareSolver     s; run_and_log(s); }
                {  FaithfulCASolver          s; run_and_log(s); }
                {  TrafficFlowSolver         s; run_and_log(s); }
                {  RHCRSolver                s; run_and_log(s); }
                {  HybridAdaptivePredictiveSolver s; run_and_log(s); }
            }
        }

        logger.close();
        std::cout << "\n=== Option U done. CSV: " << csv_path << " ===\n";
    }
    else if (rep == "O" || rep == "o") {
        // ══════════════════════════════════════════════════════════════════════
        // OPTION O — Combined evaluation (Y RL policies + SoTA standalone)
        // ══════════════════════════════════════════════════════════════════════
        //
        // CRITICAL: this option uses the EXACT SAME parameters as Option Y
        // (training config, scenario sweep, ghost traffic, capacity hetero,
        // fleet sizing, agent pool) so the RL-side metrics are directly
        // comparable to a real Y run. The novelty is that SoTA methods are
        // pulled from src/SoTA/* (full-pipeline standalone solvers, with their
        // own allocation + planning) instead of the allocation-only branches
        // in EpisodeRunner.
        //
        // Two phases:
        //   Phase A — RL policies via MultiCityTrainer (same code path as Y).
        //             Produces results/<output_dir>/summary.csv with the 4 RL
        //             policies (MAPPO, FaithfulMAPPER, IPPO, Hybrid).
        //   Phase B — SoTA standalone solvers via SolverRunner. Produces a
        //             SECOND CSV results/sota_standalone_O/sota_standalone.csv
        //             with the 5 standalone solvers (TP, CA, FCA, TF, RHCR).
        //
        // The two CSVs share keys (city, scenario, episode_index, seed) so
        // they can be JOINED post-hoc for full cross-method analysis.
        //
        // Coverage: 4 RL × 6 cities × 5 scenarios × 3 eps = 360 RL episodes
        //         + 5 SoTA × 6 cities × 5 scenarios × 3 eps = 450 SoTA episodes
        //         = 810 total runs. Wallclock estimate 14-30h on one seed
        //           (RHCR dominates the SoTA cost).
        std::cout << "\n=== Option O — Combined Y + SoTA standalone (publication-grade, 2 seeds) ===\n";

        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        // Option O — RL + TP allocation on the 5 Small cities (Tokyo, Kyoto,
        // LA, NewYork, Paris — all _Small ~25 km²). HAPC is split into a
        // dedicated Option OP (same env, parallel run).
        // Each option has its own top-level results folder so the two runs
        // can't overwrite each other when run in parallel.
        const std::string output_dir = "C:\\ConflictualMAS\\results\\Option_O_results";
        const std::string sota_dir   = output_dir + "\\sota_standalone_O";
        std::filesystem::create_directories(sota_dir);
        CityRegistry::set_osm_root(osm_root);

        // ── Build the SAME TrainingConfig as Option Y ───────────────────────
        // (verbatim copy — DO NOT change parameters; we want RL-side metrics
        // to match Y exactly when the same seed is used).
        TrainingConfig cfg;
        cfg.cache_root       = cache_root;
        cfg.output_dir       = output_dir;
        cfg.n_seeds          = 1;          // we externally loop over seeds for per-seed checkpoint
        cfg.n_eval_episodes  = 2;          // 2 eps × 2 seeds = 4 measurements per slot — fits ~8-10h/seed
        cfg.verbose          = true;

        cfg.eval_only            = true;
        cfg.skip_stress_eval     = true;
        cfg.skip_generalize_eval = false;

        // ═══════════════════════════════════════════════════════════════
        // Allocation-only comparison (Option Y style, publication-grade)
        // ═══════════════════════════════════════════════════════════════
        // All modes share DbVNS-PDP planning (capacity-aware) and TAM Format
        // A allocation interface. Each mode differs ONLY in its ALLOCATION
        // RULE (who gets the next task).
        //
        // PLANNING is a SEPARATE evaluation handled by Option P (DbVNS vs
        // MCA-plan vs DoubleHorizon-plan vs ALNS-plan, under MCA allocation).
        //
        // ┌─────────────────────────────────────────────────────────────┐
        // │ OPTION O — RL + TP on the 5 Small cities                     │
        // │  Tokyo_Small + Kyoto_Small + LosAngeles_Small +              │
        // │  NewYork_Small + Paris_Small  (all ~25 km², Small tier)      │
        // │                                                              │
        // │ HAPC has been split off into Option OP (same env, runs in    │
        // │ parallel — separate output_dir).                             │
        // │                                                              │
        // │ Phase A runs MAPPO, FaithfulMAPPER, IPPO, Hybrid (4 RL) +    │
        // │ TokenPassing (allocation only). All 5 share the SAME         │
        // │ DbVNS-PDP planner — publication-grade A/B test of allocation.│
        // │                                                              │
        // │ Both Option O and Option OP target the same 5 cities so the  │
        // │ joined CSVs cover (city × method × scenario × episode)       │
        // │ uniformly. The ep_seed formula is shared verbatim, so for    │
        // │ each (city, scenario, episode_index) both options see the    │
        // │ SAME task_stream / agent_start_nodes / ghost_seed (byte-     │
        // │ identical SharedEpisodeSetup).                               │
        // │                                                              │
        // │ FINAL MERGE (manual, 2-way):                                 │
        // │   Option_O_results\episodes_seed43.csv      (RL + TP)        │
        // │ + Option_OP_results\sota_standalone_O\                       │
        // │     sota_standalone_seed43.csv              (HAPC standalone)│
        // └─────────────────────────────────────────────────────────────┘
        // RMCA(r) SoTA baseline only [Chen et al. 2021] — the RL / Hybrid / TP
        // evaluations are already on disk (episodes_seed4{2,3,4}.csv), so this
        // run appends ONLY the RMCA(r) rows (same env / scenarios / shared setup
        // → directly comparable). Restore the full set below to re-run everything.
        cfg.eval_modes = {
            PolicyMode::RMCA,                     // marginal-cost insertion + regret (Chen+2021)
            PolicyMode::MAPPO,
            PolicyMode::FaithfulMAPPER,
            PolicyMode::IPPO,
            PolicyMode::Hybrid,
            PolicyMode::TokenPassing,          // task allocation only (Ma+2017)
        };

        // Exact same 5 scenarios as Y.
        cfg.eval_scenarios = {
            EpisodeScenario{ 1.0f,  1.0f, "normal_wave",    CongestionProfile::Wave        },
            EpisodeScenario{ 1.0f,  1.0f, "normal_ramp",    CongestionProfile::RampUpDown  },
            EpisodeScenario{ 1.5f,  0.8f, "stress_shock",   CongestionProfile::ShockBurst  },
            EpisodeScenario{ 2.0f,  0.6f, "stress_buildup", CongestionProfile::BuildingUp  },
            EpisodeScenario{ 0.5f, 10.0f, "over_fleet",     CongestionProfile::Wave        },
        };

        // Exact same environment as Y (verbatim).
        cfg.episode_cfg.use_dbvns_planning            = true;   // DbVNS-PDP planning ON
        cfg.episode_cfg.enable_ghost_traffic          = true;
        cfg.episode_cfg.ghost_hot_way_count           = 0;
        cfg.episode_cfg.ghost_hot_way_frac            = 0.05f;
        cfg.episode_cfg.ghost_density_per_hot_way     = 2.0f;
        cfg.episode_cfg.ghost_window_steps            = 8;
        cfg.episode_cfg.ghost_n_max                   = 100;
        cfg.episode_cfg.ghost_n_max_user_set          = false;
        cfg.episode_cfg.enable_heterogeneous_capacity = true;
        cfg.episode_cfg.hetero_capacity_min           = 3;
        cfg.episode_cfg.hetero_capacity_max           = 7;
        cfg.episode_cfg.fleet_size_divisor            = 2;
        cfg.episode_cfg.fleet_load_per_agent          = 2;
        cfg.episode_cfg.agent_pool_multiplier         = 10.0f;

        // ── TAM multi-candidate Format A (force-assign argmax) ─────────────
        // CRITICAL: matches Option Y's TAM config. Without this, RL refuses
        // ~10-15% of tasks (first-fit threshold 0.5), creating a structural
        // disadvantage vs SoTA standalone solvers (which always allocate via
        // argmin insertion-cost). Format A force-assign → acc ≈ 1.0 for RL,
        // matching SoTA's "always allocate" behaviour. Comparison is now fair.
        cfg.episode_cfg.tam_multi_candidate     = true;
        cfg.episode_cfg.tam_mc_force_assign     = true;        // Format A
        cfg.episode_cfg.tam_mc_max_candidates   = 5;
        cfg.episode_cfg.tam_mc_ratio_min        = 1.4f;
        cfg.episode_cfg.tam_mc_ratio_max        = 3.0f;
        cfg.episode_cfg.tam_mc_ratio_scale      = 2000.f;

        // Publication-grade cross-method homogeneity: build a canonical
        // SharedEpisodeSetup per (city, scenario, episode) and pass it to BOTH
        // EpisodeRunner (Phase A, RL) and SolverRunner (Phase B, SoTA). This
        // guarantees byte-identical task_stream, agent_start_nodes,
        // per_agent_capacity, total_steps and ghost_seed across all methods —
        // any throughput/latency gap between RL and SoTA is now a pure method
        // effect, not a confound of environment-derivation differences.
        cfg.use_shared_episode_setup = true;

        cfg.load_policy                 = true;
        cfg.policy_path                 = "C:\\ConflictualMAS\\results\\mappo\\"
                                          "policy_seed42.bin";
        cfg.ippo_policy_path            = "C:\\ConflictualMAS\\results\\ippo_faithful\\"
                                          "ippo_seed42.bin";
        cfg.mapper_policy_path          = "C:\\ConflictualMAS\\results\\mapper_enhanced\\"
                                          "mapper_seed42.bin";
        cfg.faithful_mapper_policy_path = "C:\\ConflictualMAS\\results\\mapper_faithful\\"
                                          "mapper_seed42.bin";

        // ── City split (CRITICAL for homogeneity with Option OP) ──────────
        // MultiCityTrainer's run_generalize_eval ONLY loads ComparisonOnly
        // cities, so Tokyo_Small / Kyoto_Small / LosAngeles_Small (registered
        // as TrainAndApply) MUST go in train_city_filter. Paris_Small and
        // NewYork_Small are ComparisonOnly so they go in gen_city_filter.
        //
        // All 5 cities have area=25 km² → customize_episode_for_city applies
        // the SAME Small tier (n_agents 6-12, ghost_n_max=30, max_task_dist
        // 1500m, max_tasks_per_agent=3) so the per-city environment is
        // uniform regardless of the train/gen label. The CSV phase column
        // will say "eval_X" vs "generalize_X" but the underlying simulation
        // is on the same scale.
        //
        // The IDENTICAL split is mirrored in Option OP so ca.index matches
        // for the same city in both options → byte-identical ep_seed and
        // SharedEpisodeSetup.
        cfg.train_city_filter     = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };
        cfg.enable_generalization = true;
        cfg.gen_city_filter       = {
            "NewYork_Small", "Paris_Small"
        };

        // ══════════════════════════════════════════════════════════════════
        // ┌──────────────────────────────────────────────────────────────┐
        // │  MANUAL CONFIG — edit these 2 constants between runs         │
        // ├──────────────────────────────────────────────────────────────┤
        // │  Run 1 :  CURRENT_SEED = 42, RUN_PHASE_A = false              │
        // │           (Phase B only on seed 42 — Phase A already on disk)│
        // │  Run 2 :  CURRENT_SEED = 43, RUN_PHASE_A = true               │
        // │  Run 3 :  CURRENT_SEED = 44, RUN_PHASE_A = true               │
        // │  Run 4 :  CURRENT_SEED = 45, RUN_PHASE_A = true               │
        // │                                                              │
        // │  Each invocation runs ONE seed (Phase A optional + Phase B). │
        // │  This avoids the MultiCityTrainer destructor crash because   │
        // │  the trainer is created/destroyed exactly once per process.  │
        // └──────────────────────────────────────────────────────────────┘
        
        std::cout << "Seed: ";

        int seed;
        std::cin >> seed;

        const int  CURRENT_SEED = seed;     // seeds 42, 43 already on disk → use 44 for fresh eval
        const bool RUN_PHASE_A  = true;   // 4 RL + TP allocation on the 5 Small cities
        const bool RUN_PHASE_B  = false;  // HAPC standalone runs in Option OP (parallel)
        // ══════════════════════════════════════════════════════════════════

        const std::vector<int> seeds_to_run = { CURRENT_SEED };
        const auto t_global_start = std::chrono::steady_clock::now();

        std::cout << "\n\n╔════════════════════════════════════════════════════════════╗\n"
                  <<     "║  Option O — SEED " << CURRENT_SEED
                  <<     "  Phase A=" << (RUN_PHASE_A ? "YES" : "skip (on disk)")
                  <<     "  Phase B=" << (RUN_PHASE_B ? "YES" : "skip") << "\n"
                  <<     "╚════════════════════════════════════════════════════════════╝\n";

        // ── PHASE A : optional, only when RUN_PHASE_A=true ───────────────
        // MultiCityTrainer is invoked at most ONCE per process invocation,
        // ensuring its destructor runs exactly once and the static-singleton
        // policy state (ObjectiveDMPolicy, IPPOPolicy, MapperPolicy,
        // FaithfulMapperPolicy, HybridPolicy) is not stressed by repeated
        // train()/destroy cycles — which was the seed-loop crash root cause.
        if (RUN_PHASE_A) {
            std::cout << "\n--- PHASE A (seed=" << CURRENT_SEED
                      << ") : 4 RL policies (MAPPO/FaithfulMAPPER/IPPO/Hybrid) ---\n";
            cfg.start_seed = CURRENT_SEED;
            cfg.n_seeds    = 1;
            MultiCityTrainer trainer;
            trainer.train(cfg);
            std::cout << "\n--- PHASE A seed " << CURRENT_SEED
                      << " done. CSV: " << output_dir
                      << "\\episodes_seed" << CURRENT_SEED << ".csv ---\n";
        } else {
            std::cout << "\n--- PHASE A skipped — reusing " << output_dir
                      << "\\episodes_seed" << CURRENT_SEED << ".csv on disk ---\n";
        }

        // ── PHASE B : optional, only when RUN_PHASE_B=true ───────────────
        if (RUN_PHASE_B)
        for (size_t si = 0; si < seeds_to_run.size(); ++si) {
            const int seed = seeds_to_run[si];
            const auto t_seed_start = std::chrono::steady_clock::now();

            std::cout << "\n\n╔════════════════════════════════════════════════════════════╗\n"
                      <<     "║  PHASE B seed " << seed << "\n"
                      <<     "╚════════════════════════════════════════════════════════════╝\n";

            // Phase A CSV is expected on disk for cross-join purposes.
            const std::string ep_csv = output_dir
                + "\\episodes_seed" + std::to_string(seed) + ".csv";

            // ── PHASE B : SoTA standalone solvers ─────────────────────────────
        // We replicate the city × scenario × episode iteration that Phase A
        // just performed, but using the SoTA standalone solvers via
        // SolverRunner. Each iteration uses a SEED derived deterministically
        // from (city_index, scenario_index, episode_index) — matching the
        // scheme used by MultiCityTrainer::run_eval so the task streams of
        // Phase A and Phase B coincide.
        std::cout << "\n--- PHASE B (seed=" << seed
                  << ") : HAPC standalone smoke test (Cortés+2009) ---\n";

        // ── City lists — train_cities and gen_cities are kept SEPARATE so
        // each one resets ca_index to 0 (mirrors MultiCityTrainer's
        // run_eval / run_generalize loops, where ca.index is per-phase).
        struct CityToRun {
            const CityConfig* cc;
            std::string       cache_path;
            int               ca_index;   // index within its phase (train OR gen)
            const char*       phase_label; // "train" or "gen"
        };
        std::vector<CityToRun> train_cities_to_run, gen_cities_to_run;
        const auto& all_cities = CityRegistry::all();
        auto add_city = [&](std::vector<CityToRun>& dst, const std::string& want,
                             const char* phase_label) {
            for (const auto& cc : all_cities) {
                if (cc.name == want) {
                    dst.push_back({ &cc,
                                    cache_root + "/" + cc.name + ".json",
                                    static_cast<int>(dst.size()),
                                    phase_label });
                    return;
                }
            }
            std::cerr << "  [Warn] city \"" << want << "\" not found in registry.\n";
        };
        for (const auto& n : cfg.train_city_filter) add_city(train_cities_to_run, n, "train");
        if (cfg.enable_generalization)
            for (const auto& n : cfg.gen_city_filter) add_city(gen_cities_to_run, n, "gen");

        // Per-seed CSV: sota_standalone_seed{seed}.csv (mirrors episodes_seed{seed}.csv)
        const std::string sota_csv = sota_dir + "\\sota_standalone_seed"
            + std::to_string(seed) + ".csv";
        SolverCSVLogger logger(sota_csv, /*append=*/false);
        logger.write_header();

        const int n_eps = cfg.n_eval_episodes;
        const uint32_t base_seed = static_cast<uint32_t>(seed);  // matches Phase A's ep_seed formula

        // ── EXACT MultiCityTrainer seed formula (mirrors run_eval / run_generalize)
        //    ep_seed = 1u + e + 101u*(sc_idx+1) + 10007u*(ca.index+1) + 1000003u*seed
        // This guarantees Phase A and Phase B see the SAME task stream, ghost
        // traffic, and hetero-capacity draws for the same (city, scenario,
        // episode) slot — the only difference is the solver/policy.
        auto eval_phase = [&](std::vector<CityToRun>& cities) {
            for (auto& entry : cities) {
                const CityConfig& cc = *entry.cc;
                std::cout << "\n  [" << entry.phase_label << " ca.index="
                          << entry.ca_index << "] City: " << cc.name << "\n";

                GeoBox geo_box;
                if (GeoBoxManager::cache_exists(entry.cache_path)) {
                    geo_box = GeoBoxManager::load_geobox(entry.cache_path);
                } else {
                    geo_box = create_geo_box(cc.osm_path,
                                             cc.bbox.min_lon, cc.bbox.min_lat,
                                             cc.bbox.max_lon, cc.bbox.max_lat);
                    if (geo_box.is_valid)
                        GeoBoxManager::save_geobox(geo_box, entry.cache_path);
                }
                if (!geo_box.is_valid) {
                    std::cerr << "    [Skip] GeoBox invalid for " << cc.name << "\n";
                    continue;
                }
                Pathfinder pathfinder(geo_box);

                // CRITICAL: same per-city EpisodeConfig customisation that
                // MultiCityTrainer applies internally (phases, task distance,
                // ghost_n_max tier). Without this, Phase B uses default phases
                // and would face a DIFFERENT episode shape than Phase A.
                EpisodeConfig per_city_ep = cfg.episode_cfg;
                per_city_ep.city = &cc;
                MultiCityTrainer::customize_episode_for_city(per_city_ep, cc);

                for (size_t si = 0; si < cfg.eval_scenarios.size(); ++si) {
                    const EpisodeScenario& sc = cfg.eval_scenarios[si];
                    const int sc_idx = static_cast<int>(si);
                    for (int e = 0; e < n_eps; ++e) {
                        // EXACT formula from MultiCityTrainer.cpp line 314-318.
                        const uint32_t ep_seed = static_cast<uint32_t>(
                            1u + e
                            + 101u * (sc_idx + 1)
                            + 10007u * (entry.ca_index + 1)
                            + 1000003u * base_seed);

                        std::cout << "    " << cc.name << " | " << sc.label
                                  << " | ep=" << e << " seed=" << ep_seed << "\n";

                        SolverRunner runner(per_city_ep, geo_box, pathfinder,
                                             sc, ep_seed);

                        // Publication-grade: SAME SharedEpisodeSetup as Phase A
                        // builds for this (ep_seed, city, scenario, cfg). Both
                        // phases see byte-identical environment derivations.
                        const SharedEpisodeSetup setup =
                            build_shared_episode_setup(
                                ep_seed, cc, sc, per_city_ep, geo_box);

                        auto run_and_log = [&](ISolver& s) {
                            SolverMetrics m = runner.run(s, &setup);
                            m.city_label = cc.name;
                            m.episode    = e;
                            logger.write_row(m);
                            std::cout << "      " << m.solver_name
                                      << "  thr=" << m.throughput_rate
                                      << "  lat=" << m.latency_mean
                                      << "  ms="  << m.wallclock_ms
                                      << "\n";
                        };

                        // Option O Phase B is DISABLED by default (RUN_PHASE_B
                        // = false above). HAPC has been split off into
                        // Option G to run in parallel. If Phase B is ever
                        // re-enabled here, comment in the solvers you want.
                        // { TokenPassingSolver    s; run_and_log(s); }
                        // { CongestionAwareSolver s; run_and_log(s); }
                        // { FaithfulCASolver      s; run_and_log(s); }
                        // { TrafficFlowSolver     s; run_and_log(s); }
                        // { RHCRSolver            s; run_and_log(s); }
                        // HAPC lives in Option G:
                        // { HybridAdaptivePredictiveSolver s; run_and_log(s); }
                    }
                }
            }
        };

        std::cout << "\n  -- Train cities --\n";
        eval_phase(train_cities_to_run);
        std::cout << "\n  -- Gen cities --\n";
        eval_phase(gen_cities_to_run);

        logger.close();

        const auto t_seed_end = std::chrono::steady_clock::now();
        const long long seed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                                       t_seed_end - t_seed_start).count();
        std::cout << "\n+++ SEED " << (si + 1) << "/" << seeds_to_run.size()
                  << " (rng=" << seed << ") COMPLETE in "
                  << (seed_sec / 60) << " min "
                  << (seed_sec % 60) << " s +++\n";
        std::cout << "    RL CSV   : " << ep_csv   << "\n";
        std::cout << "    SoTA CSV : " << sota_csv << "\n";
        }  // end seed loop

        const auto t_global_end = std::chrono::steady_clock::now();
        const long long total_sec = std::chrono::duration_cast<std::chrono::seconds>(
                                        t_global_end - t_global_start).count();
        std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
        std::cout << "=== Option O ALL SEEDS DONE — wallclock " << (total_sec / 3600)
                  << "h " << ((total_sec % 3600) / 60) << "min\n";
        for (int s : seeds_to_run) {
            std::cout << "  seed " << s << ":\n";
            std::cout << "    RL   : " << output_dir << "\\episodes_seed" << s << ".csv\n";
            std::cout << "    SoTA : " << sota_dir   << "\\sota_standalone_seed" << s << ".csv\n";
        }
        std::cout << "  Cross-join keys: (seed, city, scenario, episode)\n";
    }
    else if (rep == "OP" || rep == "op") {
        // ══════════════════════════════════════════════════════════════════════
        // OPTION OP — HAPC standalone (parallel sibling of Option O)
        // ══════════════════════════════════════════════════════════════════════
        //
        // EXACT same environment as Option O: the 5 Small cities (Tokyo_Small,
        // Kyoto_Small, LosAngeles_Small, NewYork_Small, Paris_Small), same
        // scenarios, same SharedEpisodeSetup, same ghost traffic, same hetero
        // capacity, same ep_seed formula, same CURRENT_SEED. The ONLY
        // difference: this option runs HAPC (Cortés+2009) standalone via
        // SolverRunner, while Option O runs RL + TP allocation via
        // MultiCityTrainer.
        //
        // Designed to run IN PARALLEL with Option O — they write to DIFFERENT
        // output directories so there's no file contention. Combined, they
        // produce the full publication comparison on the 5 Small cities ×
        // 5 scenarios × 2 episodes × (4 RL + TP + HAPC).
        //
        // HOMOGENEITY GUARANTEE: for every (city, scenario, episode_index),
        // both options' SharedEpisodeSetup yields byte-identical task streams,
        // agent_start_nodes, per_agent_capacity, and ghost_seed. The merged
        // CSV is therefore directly comparable cell-by-cell across methods.
        //
        // CSV output:
        //   results\Option_OP_results\sota_standalone_O\
        //     sota_standalone_seed{CURRENT_SEED}.csv
        // Joinable with Option O's CSV on (city, scenario, episode).
        std::cout << "\n=== Option OP — FCA + HAPC on GEN cities (NewYork, Paris) — seeds 42, 43, 44 ===\n";

        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        // Separate top-level results folder — keeps Option OP's outputs apart
        // from Option O's so the two runs can execute in parallel.
        const std::string output_dir = "C:\\ConflictualMAS\\results\\Option_OP_results";
        // Gen-cities FCA+HAPC re-run lands in its OWN subdir so it does NOT
        // overwrite any prior fca_standalone / sota_standalone_O outputs — the
        // existing merged unified data is preserved. Only NewYork/Paris rows
        // (FCA + HAPC) are produced here, for a later re-merge into the unified.
        const std::string sota_dir   = output_dir + "\\fca_gencities";
        std::filesystem::create_directories(sota_dir);
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root       = cache_root;
        cfg.output_dir       = output_dir;
        cfg.n_seeds          = 1;
        cfg.n_eval_episodes  = 2;
        cfg.verbose          = true;
        cfg.eval_only            = true;
        cfg.skip_stress_eval     = true;
        cfg.skip_generalize_eval = false;

        // Same scenarios as Option O — must match for cross-join.
        cfg.eval_scenarios = {
            EpisodeScenario{ 1.0f,  1.0f, "normal_wave",    CongestionProfile::Wave        },
            EpisodeScenario{ 1.0f,  1.0f, "normal_ramp",    CongestionProfile::RampUpDown  },
            EpisodeScenario{ 1.5f,  0.8f, "stress_shock",   CongestionProfile::ShockBurst  },
            EpisodeScenario{ 2.0f,  0.6f, "stress_buildup", CongestionProfile::BuildingUp  },
            EpisodeScenario{ 0.5f, 10.0f, "over_fleet",     CongestionProfile::Wave        },
        };

        // Same environment as Option O (verbatim).
        cfg.episode_cfg.use_dbvns_planning            = true;
        cfg.episode_cfg.enable_ghost_traffic          = true;
        cfg.episode_cfg.ghost_hot_way_count           = 0;
        cfg.episode_cfg.ghost_hot_way_frac            = 0.05f;
        cfg.episode_cfg.ghost_density_per_hot_way     = 2.0f;
        cfg.episode_cfg.ghost_window_steps            = 8;
        cfg.episode_cfg.ghost_n_max                   = 100;
        cfg.episode_cfg.ghost_n_max_user_set          = false;
        cfg.episode_cfg.enable_heterogeneous_capacity = true;
        cfg.episode_cfg.hetero_capacity_min           = 3;
        cfg.episode_cfg.hetero_capacity_max           = 7;
        cfg.episode_cfg.fleet_size_divisor            = 2;
        cfg.episode_cfg.fleet_load_per_agent          = 2;
        cfg.episode_cfg.agent_pool_multiplier         = 10.0f;
        cfg.episode_cfg.tam_multi_candidate     = true;
        cfg.episode_cfg.tam_mc_force_assign     = true;
        cfg.episode_cfg.tam_mc_max_candidates   = 5;
        cfg.episode_cfg.tam_mc_ratio_min        = 1.4f;
        cfg.episode_cfg.tam_mc_ratio_max        = 3.0f;
        cfg.episode_cfg.tam_mc_ratio_scale      = 2000.f;
        cfg.use_shared_episode_setup = true;

        // ── Same train/gen split as Option O (CRITICAL for homogeneity) ──
        // ca.index in the ep_seed formula is per-phase:
        //   train_cities_to_run : Tokyo=0, Kyoto=1, LosAngeles=2
        //   gen_cities_to_run   : NewYork=0, Paris=1
        // Option O's MultiCityTrainer assigns these same indices, so for
        // any (city, scenario, episode) both options produce the SAME
        // ep_seed → SharedEpisodeSetup byte-identical → task_stream,
        // agent_start_nodes, ghost_seed all match.
        cfg.train_city_filter     = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };
        cfg.enable_generalization = true;
        cfg.gen_city_filter       = {
            "NewYork_Small", "Paris_Small"
        };

        // ── SEEDS 42, 43, 44 RUN — FCA standalone across all seeds ───────
        // Reuses Option O's per-seed SharedEpisodeSetup verbatim (byte-
        // identical task streams) so each new CSV cross-joins with the
        // existing episodes_seed{42,43,44} RL CSVs.
        // FCA-Cap is currently disabled (see solver call site below) —
        // re-enable by uncommenting the FaithfulCACapacitedSolver line if
        // a load-aware insertion variant is implemented in the future.
        const std::vector<int> seeds_to_run = { 42, 43, 44 };
        const auto t_global_start = std::chrono::steady_clock::now();

        std::cout << "\n\n╔════════════════════════════════════════════════════════════╗\n"
                  <<     "║  Option OP — FCA standalone — SEEDS";
        for (int s : seeds_to_run) std::cout << " " << s;
        std::cout <<     "\n╚════════════════════════════════════════════════════════════╝\n";

        for (size_t si = 0; si < seeds_to_run.size(); ++si) {
            const int seed = seeds_to_run[si];
            const auto t_seed_start = std::chrono::steady_clock::now();

            std::cout << "\n--- FCA standalone (seed=" << seed
                      << ", " << (si + 1) << "/" << seeds_to_run.size() << ") ---\n";

            struct CityToRun {
                const CityConfig* cc;
                std::string       cache_path;
                int               ca_index;
                const char*       phase_label;
            };
            std::vector<CityToRun> train_cities_to_run, gen_cities_to_run;
            const auto& all_cities = CityRegistry::all();
            auto add_city = [&](std::vector<CityToRun>& dst, const std::string& want,
                                 const char* phase_label) {
                for (const auto& cc : all_cities) {
                    if (cc.name == want) {
                        dst.push_back({ &cc,
                                        cache_root + "/" + cc.name + ".json",
                                        static_cast<int>(dst.size()),
                                        phase_label });
                        return;
                    }
                }
                std::cerr << "  [Warn] city \"" << want << "\" not found in registry.\n";
            };
            for (const auto& n : cfg.train_city_filter) add_city(train_cities_to_run, n, "train");
            if (cfg.enable_generalization)
                for (const auto& n : cfg.gen_city_filter) add_city(gen_cities_to_run, n, "gen");

            const std::string sota_csv = sota_dir + "\\sota_standalone_seed"
                + std::to_string(seed) + ".csv";
            SolverCSVLogger logger(sota_csv, /*append=*/false);
            logger.write_header();

            const int n_eps = cfg.n_eval_episodes;
            const uint32_t base_seed = static_cast<uint32_t>(seed);

            auto eval_phase = [&](std::vector<CityToRun>& cities) {
                for (auto& entry : cities) {
                    const CityConfig& cc = *entry.cc;
                    std::cout << "\n  [" << entry.phase_label << " ca.index="
                              << entry.ca_index << "] City: " << cc.name << "\n";

                    GeoBox geo_box;
                    if (GeoBoxManager::cache_exists(entry.cache_path)) {
                        geo_box = GeoBoxManager::load_geobox(entry.cache_path);
                    } else {
                        geo_box = create_geo_box(cc.osm_path,
                                                 cc.bbox.min_lon, cc.bbox.min_lat,
                                                 cc.bbox.max_lon, cc.bbox.max_lat);
                        if (geo_box.is_valid)
                            GeoBoxManager::save_geobox(geo_box, entry.cache_path);
                    }
                    if (!geo_box.is_valid) {
                        std::cerr << "    [Skip] GeoBox invalid for " << cc.name << "\n";
                        continue;
                    }
                    Pathfinder pathfinder(geo_box);

                    EpisodeConfig per_city_ep = cfg.episode_cfg;
                    per_city_ep.city = &cc;
                    MultiCityTrainer::customize_episode_for_city(per_city_ep, cc);

                    for (size_t sci = 0; sci < cfg.eval_scenarios.size(); ++sci) {
                        const EpisodeScenario& sc = cfg.eval_scenarios[sci];
                        const int sc_idx = static_cast<int>(sci);
                        for (int e = 0; e < n_eps; ++e) {
                            // EXACT same ep_seed formula as Option O — guarantees
                            // byte-identical task_stream, agents, ghost_seed.
                            const uint32_t ep_seed = static_cast<uint32_t>(
                                1u + e
                                + 101u * (sc_idx + 1)
                                + 10007u * (entry.ca_index + 1)
                                + 1000003u * base_seed);

                            std::cout << "    " << cc.name << " | " << sc.label
                                      << " | ep=" << e << " seed=" << ep_seed << "\n";

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
                                std::cout << "      " << m.solver_name
                                          << "  thr=" << m.throughput_rate
                                          << "  lat=" << m.latency_mean
                                          << "  ms="  << m.wallclock_ms
                                          << "\n";
                            };

                            // FCA standalone — the capacitated extension FCA-Cap
                            // was disabled after seed=42 experiments showed it
                            // consistently underperforms single-task FCA across
                            // all tested regimes (mean throughput drop ~30-50%,
                            // collapses on over_fleet). Documented as negative
                            // result in the paper; the call site is preserved
                            // commented out for traceability and to allow
                            // re-enabling with a load-aware insertion objective.
                            { FaithfulCASolver               s; run_and_log(s); }
                            // { HybridAdaptivePredictiveSolver s; run_and_log(s); }
                            // { FaithfulCACapacitedSolver  s; run_and_log(s); }
                        }
                    }
                }
            };

            // ── ALL CITIES (train + gen) ─────────────────────────────────────
            // Run on the full 5-city set. Per-phase ca_index (train: Tokyo=0,
            // Kyoto=1, LosAngeles=2 ; gen: NewYork=0, Paris=1) matches Option O's
            // MultiCityTrainer indices, so ep_seed — hence the SharedEpisodeSetup
            // (task stream, agent starts, ghost seed) — is byte-identical to
            // Option O on every city, giving homogeneous baselines across all 5.
            std::cout << "\n  -- Train cities --\n";
            eval_phase(train_cities_to_run);
            std::cout << "\n  -- Gen cities --\n";
            eval_phase(gen_cities_to_run);

            logger.close();

            const auto t_seed_end = std::chrono::steady_clock::now();
            const long long seed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                                           t_seed_end - t_seed_start).count();
            std::cout << "\n+++ Option OP SEED " << seed << " COMPLETE in "
                      << (seed_sec / 60) << " min "
                      << (seed_sec % 60) << " s +++\n";
            std::cout << "    FCA CSV : " << sota_csv << "\n";
        }

        const auto t_global_end = std::chrono::steady_clock::now();
        const long long total_sec = std::chrono::duration_cast<std::chrono::seconds>(
                                        t_global_end - t_global_start).count();
        std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
        std::cout << "=== Option OP DONE — wallclock " << (total_sec / 3600)
                  << "h " << ((total_sec % 3600) / 60) << "min\n";
        for (int s : seeds_to_run) {
            std::cout << "  FCA seed " << s << ": " << sota_dir
                      << "\\sota_standalone_seed" << s << ".csv\n";
        }
        std::cout << "  Cross-join with Option O on: (city, scenario, episode)\n";
    }

        else if (rep == "OQ" || rep == "oq") {
        // ══════════════════════════════════════════════════════════════════════
        // OPTION OP — HAPC standalone (parallel sibling of Option O)
        // ══════════════════════════════════════════════════════════════════════
        //
        // EXACT same environment as Option O: the 5 Small cities (Tokyo_Small,
        // Kyoto_Small, LosAngeles_Small, NewYork_Small, Paris_Small), same
        // scenarios, same SharedEpisodeSetup, same ghost traffic, same hetero
        // capacity, same ep_seed formula, same CURRENT_SEED. The ONLY
        // difference: this option runs HAPC (Cortés+2009) standalone via
        // SolverRunner, while Option O runs RL + TP allocation via
        // MultiCityTrainer.
        //
        // Designed to run IN PARALLEL with Option O — they write to DIFFERENT
        // output directories so there's no file contention. Combined, they
        // produce the full publication comparison on the 5 Small cities ×
        // 5 scenarios × 2 episodes × (4 RL + TP + HAPC).
        //
        // HOMOGENEITY GUARANTEE: for every (city, scenario, episode_index),
        // both options' SharedEpisodeSetup yields byte-identical task streams,
        // agent_start_nodes, per_agent_capacity, and ghost_seed. The merged
        // CSV is therefore directly comparable cell-by-cell across methods.
        //
        // CSV output:
        //   results\Option_OP_results\sota_standalone_O\
        //     sota_standalone_seed{CURRENT_SEED}.csv
        // Joinable with Option O's CSV on (city, scenario, episode).
        std::cout << "\n=== Option OP — HAPC on GEN cities (NewYork, Paris) — seeds 42, 43, 44 ===\n";

        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        // Separate top-level results folder — keeps Option OP's outputs apart
        // from Option O's so the two runs can execute in parallel.
        const std::string output_dir = "C:\\ConflictualMAS\\results\\Option_OP_results";
        // Gen-cities FCA+HAPC re-run lands in its OWN subdir so it does NOT
        // overwrite any prior fca_standalone / sota_standalone_O outputs — the
        // existing merged unified data is preserved. Only NewYork/Paris rows
        // (FCA + HAPC) are produced here, for a later re-merge into the unified.
        const std::string sota_dir   = output_dir + "\\hapc_gencities";
        std::filesystem::create_directories(sota_dir);
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root       = cache_root;
        cfg.output_dir       = output_dir;
        cfg.n_seeds          = 1;
        cfg.n_eval_episodes  = 2;
        cfg.verbose          = true;
        cfg.eval_only            = true;
        cfg.skip_stress_eval     = true;
        cfg.skip_generalize_eval = false;

        // Same scenarios as Option O — must match for cross-join.
        cfg.eval_scenarios = {
            EpisodeScenario{ 1.0f,  1.0f, "normal_wave",    CongestionProfile::Wave        },
            EpisodeScenario{ 1.0f,  1.0f, "normal_ramp",    CongestionProfile::RampUpDown  },
            EpisodeScenario{ 1.5f,  0.8f, "stress_shock",   CongestionProfile::ShockBurst  },
            EpisodeScenario{ 2.0f,  0.6f, "stress_buildup", CongestionProfile::BuildingUp  },
            EpisodeScenario{ 0.5f, 10.0f, "over_fleet",     CongestionProfile::Wave        },
        };

        // Same environment as Option O (verbatim).
        cfg.episode_cfg.use_dbvns_planning            = true;
        cfg.episode_cfg.enable_ghost_traffic          = true;
        cfg.episode_cfg.ghost_hot_way_count           = 0;
        cfg.episode_cfg.ghost_hot_way_frac            = 0.05f;
        cfg.episode_cfg.ghost_density_per_hot_way     = 2.0f;
        cfg.episode_cfg.ghost_window_steps            = 8;
        cfg.episode_cfg.ghost_n_max                   = 100;
        cfg.episode_cfg.ghost_n_max_user_set          = false;
        cfg.episode_cfg.enable_heterogeneous_capacity = true;
        cfg.episode_cfg.hetero_capacity_min           = 3;
        cfg.episode_cfg.hetero_capacity_max           = 7;
        cfg.episode_cfg.fleet_size_divisor            = 2;
        cfg.episode_cfg.fleet_load_per_agent          = 2;
        cfg.episode_cfg.agent_pool_multiplier         = 10.0f;
        cfg.episode_cfg.tam_multi_candidate     = true;
        cfg.episode_cfg.tam_mc_force_assign     = true;
        cfg.episode_cfg.tam_mc_max_candidates   = 5;
        cfg.episode_cfg.tam_mc_ratio_min        = 1.4f;
        cfg.episode_cfg.tam_mc_ratio_max        = 3.0f;
        cfg.episode_cfg.tam_mc_ratio_scale      = 2000.f;
        cfg.use_shared_episode_setup = true;

        // ── Same train/gen split as Option O (CRITICAL for homogeneity) ──
        // ca.index in the ep_seed formula is per-phase:
        //   train_cities_to_run : Tokyo=0, Kyoto=1, LosAngeles=2
        //   gen_cities_to_run   : NewYork=0, Paris=1
        // Option O's MultiCityTrainer assigns these same indices, so for
        // any (city, scenario, episode) both options produce the SAME
        // ep_seed → SharedEpisodeSetup byte-identical → task_stream,
        // agent_start_nodes, ghost_seed all match.
        cfg.train_city_filter     = {
            "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small"
        };
        cfg.enable_generalization = true;
        cfg.gen_city_filter       = {
            "NewYork_Small", "Paris_Small"
        };

        // ── SEEDS 42, 43, 44 RUN — FCA standalone across all seeds ───────
        // Reuses Option O's per-seed SharedEpisodeSetup verbatim (byte-
        // identical task streams) so each new CSV cross-joins with the
        // existing episodes_seed{42,43,44} RL CSVs.
        // FCA-Cap is currently disabled (see solver call site below) —
        // re-enable by uncommenting the FaithfulCACapacitedSolver line if
        // a load-aware insertion variant is implemented in the future.
        const std::vector<int> seeds_to_run = { 42, 43, 44 };
        const auto t_global_start = std::chrono::steady_clock::now();

        std::cout << "\n\n╔════════════════════════════════════════════════════════════╗\n"
                  <<     "║  Option OP — FCA standalone — SEEDS";
        for (int s : seeds_to_run) std::cout << " " << s;
        std::cout <<     "\n╚════════════════════════════════════════════════════════════╝\n";

        for (size_t si = 0; si < seeds_to_run.size(); ++si) {
            const int seed = seeds_to_run[si];
            const auto t_seed_start = std::chrono::steady_clock::now();

            std::cout << "\n--- FCA standalone (seed=" << seed
                      << ", " << (si + 1) << "/" << seeds_to_run.size() << ") ---\n";

            struct CityToRun {
                const CityConfig* cc;
                std::string       cache_path;
                int               ca_index;
                const char*       phase_label;
            };
            std::vector<CityToRun> train_cities_to_run, gen_cities_to_run;
            const auto& all_cities = CityRegistry::all();
            auto add_city = [&](std::vector<CityToRun>& dst, const std::string& want,
                                 const char* phase_label) {
                for (const auto& cc : all_cities) {
                    if (cc.name == want) {
                        dst.push_back({ &cc,
                                        cache_root + "/" + cc.name + ".json",
                                        static_cast<int>(dst.size()),
                                        phase_label });
                        return;
                    }
                }
                std::cerr << "  [Warn] city \"" << want << "\" not found in registry.\n";
            };
            for (const auto& n : cfg.train_city_filter) add_city(train_cities_to_run, n, "train");
            if (cfg.enable_generalization)
                for (const auto& n : cfg.gen_city_filter) add_city(gen_cities_to_run, n, "gen");

            const std::string sota_csv = sota_dir + "\\sota_standalone_seed"
                + std::to_string(seed) + ".csv";
            SolverCSVLogger logger(sota_csv, /*append=*/false);
            logger.write_header();

            const int n_eps = cfg.n_eval_episodes;
            const uint32_t base_seed = static_cast<uint32_t>(seed);

            auto eval_phase = [&](std::vector<CityToRun>& cities) {
                for (auto& entry : cities) {
                    const CityConfig& cc = *entry.cc;
                    std::cout << "\n  [" << entry.phase_label << " ca.index="
                              << entry.ca_index << "] City: " << cc.name << "\n";

                    GeoBox geo_box;
                    if (GeoBoxManager::cache_exists(entry.cache_path)) {
                        geo_box = GeoBoxManager::load_geobox(entry.cache_path);
                    } else {
                        geo_box = create_geo_box(cc.osm_path,
                                                 cc.bbox.min_lon, cc.bbox.min_lat,
                                                 cc.bbox.max_lon, cc.bbox.max_lat);
                        if (geo_box.is_valid)
                            GeoBoxManager::save_geobox(geo_box, entry.cache_path);
                    }
                    if (!geo_box.is_valid) {
                        std::cerr << "    [Skip] GeoBox invalid for " << cc.name << "\n";
                        continue;
                    }
                    Pathfinder pathfinder(geo_box);

                    EpisodeConfig per_city_ep = cfg.episode_cfg;
                    per_city_ep.city = &cc;
                    MultiCityTrainer::customize_episode_for_city(per_city_ep, cc);

                    for (size_t sci = 0; sci < cfg.eval_scenarios.size(); ++sci) {
                        const EpisodeScenario& sc = cfg.eval_scenarios[sci];
                        const int sc_idx = static_cast<int>(sci);
                        for (int e = 0; e < n_eps; ++e) {
                            // EXACT same ep_seed formula as Option O — guarantees
                            // byte-identical task_stream, agents, ghost_seed.
                            const uint32_t ep_seed = static_cast<uint32_t>(
                                1u + e
                                + 101u * (sc_idx + 1)
                                + 10007u * (entry.ca_index + 1)
                                + 1000003u * base_seed);

                            std::cout << "    " << cc.name << " | " << sc.label
                                      << " | ep=" << e << " seed=" << ep_seed << "\n";

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
                                std::cout << "      " << m.solver_name
                                          << "  thr=" << m.throughput_rate
                                          << "  lat=" << m.latency_mean
                                          << "  ms="  << m.wallclock_ms
                                          << "\n";
                            };

                            // FCA standalone — the capacitated extension FCA-Cap
                            // was disabled after seed=42 experiments showed it
                            // consistently underperforms single-task FCA across
                            // all tested regimes (mean throughput drop ~30-50%,
                            // collapses on over_fleet). Documented as negative
                            // result in the paper; the call site is preserved
                            // commented out for traceability and to allow
                            // re-enabling with a load-aware insertion objective.
                            // { FaithfulCASolver               s; run_and_log(s); }
                            { HybridAdaptivePredictiveSolver s; run_and_log(s); }
                            // { FaithfulCACapacitedSolver  s; run_and_log(s); }
                        }
                    }
                }
            };

            // ── ALL CITIES (train + gen) ─────────────────────────────────────
            // Run on the full 5-city set. Per-phase ca_index (train: Tokyo=0,
            // Kyoto=1, LosAngeles=2 ; gen: NewYork=0, Paris=1) matches Option O's
            // MultiCityTrainer indices, so ep_seed — hence the SharedEpisodeSetup
            // (task stream, agent starts, ghost seed) — is byte-identical to
            // Option O on every city, giving homogeneous baselines across all 5.
            std::cout << "\n  -- Train cities --\n";
            eval_phase(train_cities_to_run);
            std::cout << "\n  -- Gen cities --\n";
            eval_phase(gen_cities_to_run);

            logger.close();

            const auto t_seed_end = std::chrono::steady_clock::now();
            const long long seed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                                           t_seed_end - t_seed_start).count();
            std::cout << "\n+++ Option OP SEED " << seed << " COMPLETE in "
                      << (seed_sec / 60) << " min "
                      << (seed_sec % 60) << " s +++\n";
            std::cout << "    FCA CSV : " << sota_csv << "\n";
        }

        const auto t_global_end = std::chrono::steady_clock::now();
        const long long total_sec = std::chrono::duration_cast<std::chrono::seconds>(
                                        t_global_end - t_global_start).count();
        std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
        std::cout << "=== Option OP DONE — wallclock " << (total_sec / 3600)
                  << "h " << ((total_sec % 3600) / 60) << "min\n";
        for (int s : seeds_to_run) {
            std::cout << "  FCA seed " << s << ": " << sota_dir
                      << "\\sota_standalone_seed" << s << ".csv\n";
        }
        std::cout << "  Cross-join with Option O on: (city, scenario, episode)\n";
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
//   PPO clip eps      = 0.10  (Yu+2022 §5.3 MARL recommendation)
//   gamma             = 0.99
//   lambda (GAE)      = 0.95
//   lr (actor/critic) = 5e-4 → 5e-5 (linear anneal, Yu+2022 default)
//   entropy bonus     = 0.01 → 0.001 (linear anneal)
//   epochs / episode  = 10  (with KL early stop at 0.01)
//   Grad-norm clipping (max_norm=0.5), full-batch SGD (Yu+2022 §5.3).
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
