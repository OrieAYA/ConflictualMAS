#include "TrainingEvaluation/Run/Trainer.hpp"
#include "Environment/Structure/EpisodeManager.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "DMASforPD/Policy/MAPPO.hpp"
#include "DMASforPD/Policy/IPPO.hpp"
#include "DMASforPD/Policy/MAPPERInspired.hpp"
#include "DMASforPD/Policy/Hybrid.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;

// ── City loading ──────────────────────────────────────────────────────────────

// Scale per-city episode params so task duration is short relative to episode
// length — needed so agents don't stay saturated and the policy gets enough
// experience. Adapted from city area (km²).
void MultiCityTrainer::customize_episode_for_city(EpisodeConfig& ep, const CityConfig& cc) {
    const double a = cc.area_km2;
    // Per-city tier override: skipped when the caller has set
    // ghost_n_max_user_set, so strong-congestion eval (Option Y) keeps a
    // SAME-density ghost regime across cities (otherwise the tier would
    // silently downgrade Tokyo_Large from a user-supplied 500 to 80).
    const bool override_ghost = !ep.ghost_n_max_user_set;
    // Fleet shrink factor: caller-controlled (Y/F/Q opt-in). Combined with
    // fleet_load_per_agent on the CongestionMap to keep the same per-edge
    // congestion footprint while paying K× less A* / policy compute.
    const int fleet_div = std::max(1, ep.fleet_size_divisor);
    auto shrink = [fleet_div](int n) {
        return std::max(2, (n + fleet_div - 1) / fleet_div);
    };
    // Tier thresholds match Tokyo_Small / Tokyo_Medium / Tokyo_Large areas.
    // Multi-task FIFO queue per agent — safe after the receive_task() reorder
    // fix. Bigger graph ⇒ deeper queue so agents stay busy across long trips.
    if (a < 50.0) {
        // Small (~25 km²): ~150m–1500m trips.
        ep.phases = {
            { 1000,  6.f,  shrink( 6), shrink( 8), 0.0f, 3 },
            { 1500,  8.f,  shrink( 8), shrink(11), 0.5f, 4 },
            { 1100, 10.f,  shrink(11), shrink(12), 1.0f, 5 },
        };
        ep.min_task_dist_m = 150.f;
        ep.max_task_dist_m = 1500.f;
        ep.hot_zone_radius = 300.f;
        ep.max_tasks_per_agent = 3;
        if (override_ghost) ep.ghost_n_max = 30;
    } else if (a < 300.0) {
        // Medium (~144 km²): ~250m–3500m trips.
        ep.phases = {
            { 1000, 3.f, shrink(10), shrink(14), 0.0f, 4 },
            { 1500, 4.f, shrink(14), shrink(18), 0.5f, 6 },
            { 1100, 5.f, shrink(18), shrink(22), 1.0f, 7 },
        };
        ep.min_task_dist_m = 250.f;
        ep.max_task_dist_m = 3500.f;
        ep.hot_zone_radius = 500.f;
        ep.max_tasks_per_agent = 3;
        if (override_ghost) ep.ghost_n_max = 72;
    } else {
        // Large (>=300 km²): ~400m–5000m trips.
        ep.phases = {
            { 1000, 2.f, shrink(16), shrink(20), 0.0f, 4 },
            { 1500, 3.f, shrink(20), shrink(26), 0.5f, 6 },
            { 1100, 4.f, shrink(26), shrink(30), 1.0f, 8 },
        };
        ep.min_task_dist_m = 400.f;
        ep.max_task_dist_m = 5000.f;
        ep.hot_zone_radius = 800.f;
        ep.max_tasks_per_agent = 4;
        if (override_ghost) ep.ghost_n_max = 160;
    }
}

std::unique_ptr<CityAssets> MultiCityTrainer::load_city(
    const CityConfig& cc, int idx, EpisodeConfig ep,
    const std::string& cache_root)
{
    ep.city = &cc;
    MultiCityTrainer::customize_episode_for_city(ep, cc);

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

// Fixed over-saturation scenario for the optional stress-eval phase.
static const EpisodeScenario kStressScenario{ 2.0f, 0.6f, "stress" };

// ── PolicyMode <-> string helper for flexible eval mode subsets ───────────────
static const char* policy_mode_label(PolicyMode m) {
    switch (m) {
        case PolicyMode::MAPPO:           return "MAPPO";
        case PolicyMode::IPPO:            return "IPPO";
        case PolicyMode::MAPPER:          return "MAPPER";
        case PolicyMode::FaithfulMAPPER:  return "FaithfulMAPPER";
        case PolicyMode::Hybrid:          return "Hybrid";
        case PolicyMode::TamAlwaysAccept: return "TamAlwaysAccept";
        case PolicyMode::Greedy:          return "Greedy";
        case PolicyMode::Random:          return "Random";
        case PolicyMode::InsertionGreedy: return "InsertionGreedy";
        case PolicyMode::PIBT:            return "PIBT";
        case PolicyMode::MCA:             return "MCA";
        case PolicyMode::TokenPassing:    return "TokenPassing";
        case PolicyMode::DoubleHorizon:   return "DoubleHorizon";
        case PolicyMode::RMCA:            return "RMCA";
    }
    return "Unknown";
}

// Resolve which modes to evaluate: cfg.eval_modes if non-empty, else `default_modes`.
static std::vector<PolicyMode> resolve_eval_modes(
    const TrainingConfig& cfg, std::initializer_list<PolicyMode> default_modes)
{
    if (!cfg.eval_modes.empty()) return cfg.eval_modes;
    return std::vector<PolicyMode>(default_modes);
}

// ── Evaluation (train cities) ─────────────────────────────────────────────────
//
// 5 eval modes per city:
//   MAPPO            – full system (learned policy + TAM routing)
//   TamAlwaysAccept  – TAM routing only, no learned policy (ablation)
//   Greedy           – sequential scan, first idle agent
//   Random           – random accept/reject baseline
//   InsertionGreedy  – cost-aware heuristic (strongest non-learning baseline)
//
// The MAPPO vs TamAlwaysAccept comparison isolates the policy learning benefit.
// The TamAlwaysAccept vs Greedy comparison isolates the TAM routing benefit.

int MultiCityTrainer::run_eval(
    const TrainingConfig& cfg,
    const std::vector<std::unique_ptr<CityAssets>>& assets,
    std::vector<std::unique_ptr<EpisodeRunner>>& runners,
    int global_ep, int seed,
    TrainingLogger& logger)
{
    const int num_cities = static_cast<int>(assets.size());

    // Eval line-up: full default = 3 RL baselines + 9 non-learning, sweeps the
    // design axes (centralised critic, decentralised actor, evolutionary RL).
    // Override via cfg.eval_modes for a lighter targeted comparison.
    const std::vector<PolicyMode> modes = resolve_eval_modes(cfg, {
        PolicyMode::MAPPO, PolicyMode::IPPO, PolicyMode::MAPPER,
        PolicyMode::RMCA,
        PolicyMode::TamAlwaysAccept,
        PolicyMode::Greedy, PolicyMode::Random,
        PolicyMode::TokenPassing });

    // Scenario sweep: if cfg.eval_scenarios is non-empty, iterate over each
    // and tag the resulting record with phase = "eval_<label>". Otherwise
    // fall back to single default scenario {1, 1, "normal"} → phase = "eval".
    std::vector<EpisodeScenario> scenarios = cfg.eval_scenarios;
    const bool single_scenario = scenarios.empty();
    if (single_scenario)
        scenarios.push_back(EpisodeScenario{1.f, 1.f, "normal"});

    for (int ci = 0; ci < num_cities; ++ci) {
        const CityAssets& ca     = *assets[ci];
        EpisodeRunner&    runner = *runners[ci];

        runner.train_mode = false;

        for (PolicyMode m : modes) {
            const char* name = policy_mode_label(m);
            runner.policy_mode = m;
            int sc_idx = 0;
            for (const EpisodeScenario& sc : scenarios) {
                const std::string phase = single_scenario
                    ? "eval" : std::string("eval_") + sc.label;
                for (int e = 0; e < cfg.n_eval_episodes; ++e) {
                    // Deterministic per-(city, scenario, episode) seed so that
                    // every policy mode sees the SAME task stream, ghost
                    // profile, and hetero-capacity draws for the same eval
                    // slot — the only difference across modes is the policy.
                    const uint32_t ep_seed = static_cast<uint32_t>(
                        1u + e
                        + 101u * (sc_idx + 1)
                        + 10007u * (ca.index + 1)
                        + 1000003u * static_cast<uint32_t>(seed));
                    // Publication-grade: build the canonical SharedEpisodeSetup
                    // so every policy (and every SoTA solver run on the same
                    // setup) sees a byte-identical environment. Opt-in via
                    // cfg.use_shared_episode_setup so training remains unchanged.
                    std::unique_ptr<SharedEpisodeSetup> setup;
                    if (cfg.use_shared_episode_setup) {
                        setup = std::make_unique<SharedEpisodeSetup>(
                            build_shared_episode_setup(
                                ep_seed, *ca.config, sc, ca.ep_cfg, ca.geo_box));
                    }
                    RunResult res = runner.run(ca.index, num_cities, sc,
                                                ep_seed, setup.get());
                    EpisodeRecord rec = make_record(
                        res, seed, global_ep++,
                        ca.config->name, phase, name,
                        res.metrics.n_agents_max);
                    logger.push(rec);
                    std::cout << "    [" << phase
                              << " " << ca.config->name << "/" << name
                              << "/" << (e + 1) << "/" << cfg.n_eval_episodes
                              << "]  thr=" << res.metrics.throughput_rate
                              << "  acc=" << res.metrics.accept_rate
                              << "  " << res.wallclock_ms << "ms\n";
                }
                ++sc_idx;
            }
        }
    }

    return global_ep;
}

// ── Generalisation evaluation (unseen cities) ─────────────────────────────────
//
// Runs MAPPO + TamAlwaysAccept + InsertionGreedy on cities held out from
// training. Called once per seed after the final train-city eval.
// Only 3 modes (skip Greedy/Random — already characterised on train cities).

int MultiCityTrainer::run_generalize_eval(
    const TrainingConfig& cfg,
    const std::vector<std::unique_ptr<CityAssets>>& gen_assets,
    std::vector<std::unique_ptr<EpisodeRunner>>& gen_runners,
    int global_ep, int seed,
    TrainingLogger& logger)
{
    if (gen_assets.empty()) return global_ep;

    const int num_gen = static_cast<int>(gen_assets.size());

    // Default generalisation modes: 3 RL + 5 strongest non-learning references.
    // Override via cfg.eval_modes.
    const std::vector<PolicyMode> modes = resolve_eval_modes(cfg, {
        PolicyMode::MAPPO, PolicyMode::IPPO, PolicyMode::MAPPER,
        PolicyMode::RMCA,
        PolicyMode::TamAlwaysAccept, PolicyMode::TokenPassing });

    // Scenario sweep on held-out cities (mirrors run_eval logic).
    std::vector<EpisodeScenario> scenarios = cfg.eval_scenarios;
    const bool single_scenario = scenarios.empty();
    if (single_scenario)
        scenarios.push_back(EpisodeScenario{1.f, 1.f, "normal"});

    std::cout << "  -- Generalisation Eval (" << num_gen << " cities) --\n";
    for (int ci = 0; ci < num_gen; ++ci) {
        const CityAssets& ca     = *gen_assets[ci];
        EpisodeRunner&    runner = *gen_runners[ci];

        runner.train_mode = false;

        for (PolicyMode m : modes) {
            const char* name = policy_mode_label(m);
            runner.policy_mode = m;
            int sc_idx = 0;
            for (const EpisodeScenario& sc : scenarios) {
                const std::string phase = single_scenario
                    ? "generalize" : std::string("generalize_") + sc.label;
                for (int e = 0; e < cfg.n_eval_episodes; ++e) {
                    // Deterministic seed: identical task stream per
                    // (gen_city, scenario, ep) across all modes — see run_eval.
                    const uint32_t ep_seed = static_cast<uint32_t>(
                        1u + e
                        + 101u * (sc_idx + 1)
                        + 10007u * (ca.index + 1)
                        + 1000003u * static_cast<uint32_t>(seed));
                    std::unique_ptr<SharedEpisodeSetup> setup;
                    if (cfg.use_shared_episode_setup) {
                        setup = std::make_unique<SharedEpisodeSetup>(
                            build_shared_episode_setup(
                                ep_seed, *ca.config, sc, ca.ep_cfg, ca.geo_box));
                    }
                    RunResult res = runner.run(ca.index, num_gen, sc,
                                                ep_seed, setup.get());
                    EpisodeRecord rec = make_record(
                        res, seed, global_ep++,
                        ca.config->name, phase, name,
                        res.metrics.n_agents_max);
                    logger.push(rec);
                    std::cout << "    [" << phase
                              << " " << ca.config->name << "/" << name
                              << "/" << (e + 1) << "/" << cfg.n_eval_episodes
                              << "]  thr=" << res.metrics.throughput_rate
                              << "  acc=" << res.metrics.accept_rate
                              << "  " << res.wallclock_ms << "ms\n";
                }
                ++sc_idx;
            }
        }
    }

    return global_ep;
}

// ── Stress evaluation (oversaturated scenarios) ───────────────────────────────
//
// Runs all 8 modes on each train city with the kStressScenario applied
// (density_mult=2.0, agents_mult=0.6). With ~4× over-saturation the system
// cannot deliver every task; the policy must learn to be selective.
//
// Logged with phase = "stress" so the analysis script can separate normal-
// load vs over-saturated performance.

int MultiCityTrainer::run_stress_eval(
    const TrainingConfig& cfg,
    const std::vector<std::unique_ptr<CityAssets>>& assets,
    std::vector<std::unique_ptr<EpisodeRunner>>& runners,
    int global_ep, int seed,
    TrainingLogger& logger)
{
    const int num_cities = static_cast<int>(assets.size());

    const std::vector<PolicyMode> modes = resolve_eval_modes(cfg, {
        PolicyMode::MAPPO, PolicyMode::IPPO, PolicyMode::MAPPER,
        PolicyMode::RMCA,
        PolicyMode::TamAlwaysAccept,
        PolicyMode::Greedy, PolicyMode::Random,
        PolicyMode::TokenPassing });

    std::cout << "  -- Stress Eval (density="
              << kStressScenario.density_mult << " agents="
              << kStressScenario.agents_mult << ") --\n";

    for (int ci = 0; ci < num_cities; ++ci) {
        const CityAssets& ca     = *assets[ci];
        EpisodeRunner&    runner = *runners[ci];
        runner.train_mode = false;

        for (PolicyMode m : modes) {
            const char* name = policy_mode_label(m);
            runner.policy_mode = m;
            for (int e = 0; e < cfg.n_eval_episodes; ++e) {
                // Deterministic stress seed (single scenario = sc_idx 0).
                const uint32_t ep_seed = static_cast<uint32_t>(
                    1u + e
                    + 101u
                    + 10007u * (ca.index + 1)
                    + 1000003u * static_cast<uint32_t>(seed));
                RunResult res = runner.run(ca.index, num_cities, kStressScenario, ep_seed);
                EpisodeRecord rec = make_record(
                    res, seed, global_ep++,
                    ca.config->name, "stress", name,
                    res.metrics.n_agents_max);
                logger.push(rec);
                std::cout << "    [stress " << ca.config->name << "/" << name
                          << "/" << (e + 1) << "/" << cfg.n_eval_episodes
                          << "]  thr=" << res.metrics.throughput_rate
                          << "  acc=" << res.metrics.accept_rate
                          << "  " << res.wallclock_ms << "ms\n";
            }
        }
    }

    return global_ep;
}

// ── Main training loop ────────────────────────────────────────────────────────

void MultiCityTrainer::train(const TrainingConfig& cfg) {
    fs::create_directories(cfg.output_dir);

    // ── 1. Load train cities ──────────────────────────────────────────────
    // When no filter is set, fall back to the registry's default "train" set
    // (TrainAndApply role). When a filter IS set, take the user's list at
    // face value — they may include ComparisonOnly cities (Tokyo_Large,
    // London, ...) intentionally to expand the training distribution.
    std::vector<const CityConfig*> train_ptrs;
    if (cfg.train_city_filter.empty()) {
        train_ptrs = CityRegistry::train_cities();
    } else {
        const auto& all_cities = CityRegistry::all();
        for (const auto& want : cfg.train_city_filter) {
            bool matched = false;
            for (const auto& cc : all_cities) {
                if (cc.name == want) {
                    train_ptrs.push_back(&cc);
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                std::cout << "  [Warn] train_city_filter entry \"" << want
                          << "\" not found in CityRegistry — skipped.\n";
            }
        }
    }
    const int num_cities  = static_cast<int>(train_ptrs.size());

    std::cout << "Loading " << num_cities << " train cities from " << cfg.cache_root << "\n";
    std::vector<std::unique_ptr<CityAssets>> assets;
    assets.reserve(num_cities);
    for (int i = 0; i < num_cities; ++i)
        assets.push_back(load_city(*train_ptrs[i], i, cfg.episode_cfg, cfg.cache_root));
    std::cout << "All train cities loaded.\n\n";

    // ── 2. Load generalisation cities (skip gracefully if OSM missing) ────
    std::vector<std::unique_ptr<CityAssets>> gen_assets;
    if (cfg.enable_generalization) {
        auto gen_ptrs_all = CityRegistry::comparison_cities();
        std::vector<const CityConfig*> gen_ptrs;
        if (cfg.gen_city_filter.empty()) {
            gen_ptrs = gen_ptrs_all;
        } else {
            for (const auto* cc : gen_ptrs_all) {
                for (const auto& want : cfg.gen_city_filter) {
                    if (cc->name == want) { gen_ptrs.push_back(cc); break; }
                }
            }
        }
        if (!gen_ptrs.empty()) {
            std::cout << "Loading generalisation cities (skip if missing)...\n";
            for (int i = 0; i < (int)gen_ptrs.size(); ++i) {
                try {
                    gen_assets.push_back(
                        load_city(*gen_ptrs[i], num_cities + i, cfg.episode_cfg, cfg.cache_root));
                } catch (const std::exception& e) {
                    std::cout << "  [Skip] " << gen_ptrs[i]->name
                              << " — " << e.what() << "\n";
                }
            }
            std::cout << gen_assets.size() << "/" << gen_ptrs.size()
                      << " generalisation cities loaded.\n\n";
        }
    } else {
        std::cout << "Generalisation eval disabled (Tokyo-only run).\n\n";
    }

    std::vector<int> train_indices;
    for (int i = 0; i < num_cities; ++i)
        train_indices.push_back(i);

    // ── 3. Multi-seed loop ────────────────────────────────────────────────
    const std::string summary_path = cfg.output_dir + "/summary.csv";

    for (int s = 0; s < cfg.n_seeds; ++s) {
        const int seed = s + cfg.start_seed;
        std::cout << "══════════════════════════════════════\n"
                  << " Seed " << s << "  (rng=" << seed << ")\n"
                  << "══════════════════════════════════════\n";

        // Fresh networks for this seed.
        auto& policy = mappo_policy();
        policy.reinit(static_cast<uint32_t>(seed));

        int learning_pool = 0;
        for (int i = 0; i < num_cities; ++i)
            learning_pool = std::max(learning_pool,
                static_cast<int>(std::ceil(assets[i]->ep_cfg.max_agents() * 1.5)));

        auto& ippo = ippo_policy();
        ippo.reinit(static_cast<uint32_t>(seed) ^ 0x1Au);
        ippo.ensure_agents(learning_pool);

        auto& mapper = mapper_policy();
        mapper.reinit(static_cast<uint32_t>(seed) ^ 0x2Bu);
        mapper.ensure_agents(learning_pool);

        if (cfg.load_policy) {
            auto try_load = [&](const char* name, const std::string& path,
                                auto loader) {
                if (path.empty()) return;
                if (!loader(path)) {
                    const std::string msg = std::string("Could not load ") + name
                                          + " from " + path;
                    if (cfg.eval_only)
                        throw std::runtime_error("[eval_only] " + msg
                            + " — aborting to avoid evaluating Xavier-init weights");
                    std::cerr << "[Warn] " << msg << "\n";
                } else {
                    std::cout << "[Policy] Loaded " << name << " from " << path << "\n";
                }
            };
            try_load("MAPPO   ", cfg.policy_path,
                     [&](const std::string& p){ return policy.load(p); });
            try_load("IPPO    ", cfg.ippo_policy_path,
                     [&](const std::string& p){ return ippo.load(p); });
            try_load("MAPPER  ", cfg.mapper_policy_path,
                     [&](const std::string& p){ return mapper.load(p); });
        }

        // ── Hybrid base initialisation ──────────────────────────────────────
        // Hybrid uses MAPPO's actor as its frozen base. If MAPPO was loaded
        // from a checkpoint just above, the base is the trained policy;
        // otherwise it inherits the fresh-init MAPPO.
        auto& hybrid = hybrid_policy();
        hybrid.reinit(static_cast<uint32_t>(seed) ^ 0x3Cu);
        hybrid.set_base_actor(policy.actor());
        hybrid.ensure_agents(learning_pool);
        std::cout << "[Policy] Hybrid base set from MAPPO actor "
                  << "(" << learning_pool << " agent slots)\n";

        // ── Safety check for eval_only mode ─────────────────────────────────
        // If a policy is in eval_modes but its checkpoint path is empty, the
        // eval would silently run on Xavier-init weights — invalidating the
        // comparison. Detect and abort.
        if (cfg.eval_only && !cfg.eval_modes.empty()) {
            auto has_mode = [&](PolicyMode m){
                return std::find(cfg.eval_modes.begin(),
                                  cfg.eval_modes.end(), m)
                       != cfg.eval_modes.end();
            };
            auto check = [&](const char* name, PolicyMode m,
                             const std::string& p) {
                if (has_mode(m) && p.empty()) {
                    throw std::runtime_error(
                        std::string("[eval_only] ") + name
                        + " is in eval_modes but its checkpoint path is empty"
                        + " — set the corresponding *_policy_path in cfg");
                }
            };
            check("MAPPO",          PolicyMode::MAPPO,          cfg.policy_path);
            check("IPPO",           PolicyMode::IPPO,           cfg.ippo_policy_path);
            check("MAPPER",         PolicyMode::MAPPER,         cfg.mapper_policy_path);
            // Hybrid base derives from MAPPO, so check MAPPO path for it too.
            if (has_mode(PolicyMode::Hybrid) && cfg.policy_path.empty()) {
                throw std::runtime_error(
                    "[eval_only] Hybrid is in eval_modes but its base (MAPPO) "
                    "path is empty — set cfg.policy_path");
            }
        }

        // Per-city runners for train cities (reused across rounds — A* cache warms up).
        std::vector<std::unique_ptr<EpisodeRunner>> runners;
        runners.reserve(num_cities);
        for (int i = 0; i < num_cities; ++i) {
            CityAssets& ca = *assets[i];
            runners.push_back(std::make_unique<EpisodeRunner>(
                ca.ep_cfg, ca.geo_box, ca.pathfinder,
                static_cast<uint32_t>(seed)));
        }

        // Per-city runners for generalisation cities (cold path — no training).
        std::vector<std::unique_ptr<EpisodeRunner>> gen_runners;
        gen_runners.reserve(gen_assets.size());
        for (auto& ga : gen_assets)
            gen_runners.push_back(std::make_unique<EpisodeRunner>(
                ga->ep_cfg, ga->geo_box, ga->pathfinder,
                static_cast<uint32_t>(seed)));

        TrainingLogger logger(cfg.output_dir, seed);
        int global_ep = 0;

        const std::vector<EpisodeScenario> scenario_grid =
            cfg.train_scenarios.empty() ? make_scenario_grid() : cfg.train_scenarios;
        size_t scenario_cursor = 0;

        // ── 4. Training rounds ────────────────────────────────────────────
        // In eval_only mode the training loop is skipped entirely: we drop
        // straight to the post-training eval phase with the loaded weights.
        // This is the protocol for the final "general eval on 7 cities with
        // the best policy" — re-run with eval_only=true after a training run.
        const int n_train_rounds = cfg.eval_only ? 0 : cfg.n_rounds;
        for (int round = 0; round < n_train_rounds; ++round) {
            // Linear anneal: lr_actor, lr_critic, ent_w decay toward their
            // *_min values over the seed's training horizon. SoTA PPO/MAPPO
            // recipe: high lr + high entropy early for exploration, both
            // taper off so the policy can commit to a strategy.
            const float progress = (cfg.n_rounds > 1)
                ? static_cast<float>(round) / (cfg.n_rounds - 1) : 0.f;
            policy.set_progress(progress);
            ippo.set_progress(progress);
            mapper.set_progress(progress);

            // Helper to log one training episode's stats line uniformly.
            auto log_train = [&](const RunResult& res, const char* tag,
                                  int max_ep_h) {
                if (!cfg.verbose) return;
                std::cout << "  [s" << s << " r" << round
                          << " " << (cfg.verbose ? "" : "")
                          << " " << tag << "]"
                          << "  thr="   << res.metrics.throughput_rate
                          << "  acc="   << res.metrics.accept_rate
                          << "  aloss=" << res.train_stats.actor_loss
                          << "  closs=" << res.train_stats.critic_loss
                          << "  ent="   << res.train_stats.entropy
                          << "  kl="    << res.train_stats.kl_approx
                          << "  cf="    << res.train_stats.clip_frac
                          << "  ep="    << res.train_stats.n_epochs
                                        << "/" << max_ep_h
                          << "  n="     << res.train_stats.n_exp
                          << "  "       << res.wallclock_ms << "ms\n";
            };

            // Resolve which policies to train this round.
            auto train_mode_active = [&](PolicyMode m){
                if (cfg.train_modes.empty()) return true;  // default = train all
                return std::find(cfg.train_modes.begin(),
                                  cfg.train_modes.end(), m)
                       != cfg.train_modes.end();
            };
            const bool train_ippo     = train_mode_active(PolicyMode::IPPO);
            // FaithfulMAPPER is a legacy alias of MAPPER (single paper-faithful
            // implementation) — either entry trains the same policy.
            const bool train_mapper   = train_mode_active(PolicyMode::MAPPER)
                                     || train_mode_active(PolicyMode::FaithfulMAPPER);
            const bool train_mappo    = train_mode_active(PolicyMode::MAPPO);

            for (int ci : train_indices) {
                CityAssets&    ca     = *assets[ci];
                EpisodeRunner& runner = *runners[ci];

                // Deterministic walk over the 9 task×congestion combinations;
                // all policies in this (round, city) slot share the same one.
                const EpisodeScenario sc = scenario_grid[scenario_cursor % scenario_grid.size()];
                ++scenario_cursor;

                // ── IPPO training episode (skipped if not in train_modes) ───
                if (train_ippo) {
                    runner.train_mode  = true;
                    runner.policy_mode = PolicyMode::IPPO;
                    RunResult res = runner.run(ca.index, num_cities, sc);
                    logger.push(make_record(
                        res, seed, global_ep++,
                        ca.config->name, "train", "IPPO",
                        res.metrics.n_agents_max));
                    if (global_ep % cfg.log_every == 0)
                        log_train(res, "IPPO", ippo.hparams.epochs);
                }

                // ── MAPPER training episode (skipped if not in train_modes) ─
                if (train_mapper) {
                    runner.train_mode  = true;
                    runner.policy_mode = PolicyMode::MAPPER;
                    RunResult res = runner.run(ca.index, num_cities, sc);
                    logger.push(make_record(
                        res, seed, global_ep++,
                        ca.config->name, "train", "MAPPER",
                        res.metrics.n_agents_max));
                    if (global_ep % cfg.log_every == 0)
                        log_train(res, "MAPPER", mapper.hparams.epochs);
                }

                // ── MAPPO training episode (skipped if not in train_modes) ──
                if (train_mappo) {
                    runner.train_mode  = true;
                    runner.policy_mode = PolicyMode::MAPPO;
                    RunResult res = runner.run(ca.index, num_cities, sc);
                    logger.push(make_record(
                        res, seed, global_ep++,
                        ca.config->name, "train", "MAPPO",
                        res.metrics.n_agents_max));
                    if (global_ep % cfg.log_every == 0)
                        log_train(res, "MAPPO", policy.hparams.epochs);
                }
            }

            // ── MAPPER evolutionary selection (paper-faithful) ────────────
            // on_round_end() self-gates on ev_params.period_rounds and runs
            //   p_i = 1 − exp(η·R̄_i)/exp(η·R̄_best); Θ_i ← Θ_best (exact copy).
            if (train_mapper) {
                mapper.on_round_end();
                const int n_rep = mapper.last_evolution_replacements();
                if (n_rep > 0 && cfg.verbose
                    && (round + 1) % std::max(1, mapper.ev_params.period_rounds) == 0) {
                    std::cout << "  [MAPPER-Ev @ round " << (round + 1)
                              << "] replaced " << n_rep << " policies"
                              << " (exact copy of best, no mutation)\n";
                }
            }

            // Periodic eval on train cities (normal + stress scenarios).
            if (!cfg.train_only && (round + 1) % cfg.eval_every == 0) {
                std::cout << "  -- Eval @ round " << (round + 1) << " --\n";
                global_ep = run_eval(cfg, assets, runners, global_ep, seed, logger);
                if (!cfg.skip_stress_eval)
                    global_ep = run_stress_eval(cfg, assets, runners, global_ep, seed, logger);
                logger.flush();
            }

            // ── Per-round CSV flush ───────────────────────────────────────────
            // Force the OS buffer to disk so the user can monitor progress live
            // (default ofstream buffering is ~4 KB → ~10+ rows held internally
            // until full). This is cheap (single fsync per round) and gives
            // immediate feedback in long training runs.
            logger.flush();

            // ── Optional periodic checkpoint ──────────────────────────────────
            // Insurance against crash / interrupt. Overwrites the same paths as
            // the final end-of-seed checkpoint, so only the latest snapshot is
            // kept on disk. Off by default (checkpoint_every_rounds = 0).
            if (cfg.save_policy && !cfg.eval_only
                && cfg.checkpoint_every_rounds > 0
                && (round + 1) % cfg.checkpoint_every_rounds == 0
                && (round + 1) < n_train_rounds)
            {
                auto should_save = [&](PolicyMode m){
                    if (cfg.train_modes.empty()) return true;
                    return std::find(cfg.train_modes.begin(),
                                      cfg.train_modes.end(), m)
                           != cfg.train_modes.end();
                };
                auto ensure_dir = [](const std::string& p) {
                    std::error_code ec; fs::create_directories(p, ec);
                };
                if (should_save(PolicyMode::MAPPO)) {
                    const std::string subdir = cfg.output_dir + "/mappo";
                    ensure_dir(subdir);
                    policy.save(subdir + "/policy_seed" + std::to_string(seed) + ".bin");
                }
                if (should_save(PolicyMode::IPPO)) {
                    const std::string subdir = cfg.output_dir + "/ippo_faithful";
                    ensure_dir(subdir);
                    ippo.save(subdir + "/ippo_seed" + std::to_string(seed) + ".bin");
                }
                if (should_save(PolicyMode::MAPPER)
                    || should_save(PolicyMode::FaithfulMAPPER)) {
                    const std::string subdir = cfg.output_dir + "/mapper";
                    ensure_dir(subdir);
                    mapper.save(subdir + "/mapper_seed" + std::to_string(seed) + ".bin");
                }
                if (cfg.verbose)
                    std::cout << "  [Checkpoint @ round " << (round + 1)
                              << "] intermediate snapshot written\n";
            }
        }

        // ── 5. Final evaluation ───────────────────────────────────────────
        // In eval_only mode we ALWAYS run the final eval (no training rounds
        // happened, so no periodic eval triggered). In training mode we run
        // it only if the last round did not already eval (avoid duplicates).
        // train_only short-circuits this entirely.
        if (!cfg.train_only &&
            (cfg.eval_only || cfg.n_rounds == 0 ||
             cfg.n_rounds % cfg.eval_every != 0)) {
            std::cout << "  -- Final Eval --\n";
            global_ep = run_eval(cfg, assets, runners, global_ep, seed, logger);
            if (!cfg.skip_stress_eval)
                global_ep = run_stress_eval(cfg, assets, runners, global_ep, seed, logger);
        }

        // ── 6. Generalisation evaluation (unseen cities, end of seed) ─────
        if (!cfg.train_only && !cfg.skip_generalize_eval)
            global_ep = run_generalize_eval(
                cfg, gen_assets, gen_runners, global_ep, seed, logger);
        logger.flush();

        // ── 7. Policy checkpoints ─────────────────────────────────────────
        // Save only the policies that were actually trained this run — saving
        // a Xavier-init policy that was never updated would silently overwrite
        // a previously-trained checkpoint with random weights. Default (empty
        // train_modes) preserves the old behaviour (save all four).
        //
        // Each method goes into its OWN subdirectory under cfg.output_dir,
        // so checkpoints from different training runs (different methods, same
        // seed) don't overwrite each other. The directories are created on the
        // fly if they don't exist yet.
        if (cfg.save_policy && !cfg.eval_only) {
            auto should_save = [&](PolicyMode m){
                if (cfg.train_modes.empty()) return true;   // default = save all
                return std::find(cfg.train_modes.begin(),
                                  cfg.train_modes.end(), m)
                       != cfg.train_modes.end();
            };
            auto ensure_dir = [](const std::string& p) {
                std::error_code ec;
                fs::create_directories(p, ec);
            };

            if (should_save(PolicyMode::MAPPO)) {
                const std::string subdir   = cfg.output_dir + "/mappo";
                ensure_dir(subdir);
                const std::string ckpt = subdir + "/policy_seed"
                                       + std::to_string(seed) + ".bin";
                policy.save(ckpt);
                std::cout << "  [Checkpoint] MAPPO          saved to " << ckpt << "\n";
            } else {
                std::cout << "  [Checkpoint] MAPPO          skipped (not in train_modes)\n";
            }
            if (should_save(PolicyMode::IPPO)) {
                const std::string subdir = cfg.output_dir + "/ippo_faithful";
                ensure_dir(subdir);
                const std::string ckpt = subdir + "/ippo_seed"
                                       + std::to_string(seed) + ".bin";
                ippo.save(ckpt);
                std::cout << "  [Checkpoint] IPPO           saved to " << ckpt
                          << "  (shared actor + shared critic)\n";
            } else {
                std::cout << "  [Checkpoint] IPPO           skipped (not in train_modes)\n";
            }
            if (should_save(PolicyMode::MAPPER)
                || should_save(PolicyMode::FaithfulMAPPER)) {
                const std::string subdir = cfg.output_dir + "/mapper";
                ensure_dir(subdir);
                const std::string ckpt = subdir + "/mapper_seed"
                                       + std::to_string(seed) + ".bin";
                mapper.save(ckpt);
                std::cout << "  [Checkpoint] MAPPER         saved to " << ckpt
                          << "  (paper-faithful evolution)\n";
            } else {
                std::cout << "  [Checkpoint] MAPPER         skipped (not in train_modes)\n";
            }
        }

        // ── 8. Per-seed summary ───────────────────────────────────────────
        TrainingLogger::write_summary(summary_path, logger.records(), seed);

        std::cout << "Seed " << s << " done — " << global_ep << " episodes.\n\n";
    }

    std::cout << "Training complete. Results in " << cfg.output_dir << "\n";
}

// ===== TrainingLogger.cpp =====
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

// ── Construction ─────────────────────────────────────────────────────────────

TrainingLogger::TrainingLogger(const std::string& output_dir, int seed) {
    fs::create_directories(output_dir);
    std::string path = output_dir + "/episodes_seed" + std::to_string(seed) + ".csv";
    file_.open(path);
    if (!file_.is_open())
        throw std::runtime_error("TrainingLogger: cannot open " + path);
    write_header();
}

TrainingLogger::~TrainingLogger() {
    if (file_.is_open()) file_.close();
}

// ── Header ───────────────────────────────────────────────────────────────────

void TrainingLogger::write_header() {
    file_ << "seed,global_episode,city,phase,policy_mode,"
          << "total_steps,n_agents_max,"
          // Objective: throughput + agent efficiency
          << "tasks_appeared,tasks_completed,throughput_rate,"
          << "accept_rate,refuse_rate,"
          << "latency_mean,latency_per_agent,agent_utilisation,"
          << "mean_congestion,mean_trip_steps,mean_wait_steps,"
          << "mean_road_pd_m,delivery_route_efficiency,"
          // Spatial complexity (haversine-based geometry of served task locations)
          << "bbox_area_km2,convex_hull_km2,mean_pd_hav_m,mean_nn_pickup_m,"
          // Temporal complexity (derived from wallclock)
          << "compute_time_per_task_ms,compute_time_per_decision_us,"
          // Validity
          << "pairing_violations,capacity_violations,"
          // RL stats
          << "actor_loss,critic_loss,entropy,adv_std,n_experiences,"
          // Selectivity diagnostics + scenario context (Option M)
          << "completion_per_accepted,unfinished_accept_rate,"
          << "mean_congestion_at_decision,n_ghost_active_mean,"
          << "congestion_profile,"
          // Multi-axis performance diagnostics (Option X)
          << "agent_completed_gini,agent_completed_std,"
          << "mean_imp_accepted,mean_imp_refused,"
          << "accept_rate_high_cong,accept_rate_low_cong,"
          << "mean_extra_steps_per_task,"
          // Network-level congestion impact
          << "peak_congestion,mean_overlap_edges,congestion_variance,"
          << "route_congestion_exposure,"
          << "max_agent_completed,min_agent_completed,total_fleet_distance_m,"
          // TAM efficiency (paper's "minimize comm overhead" claim)
          << "mean_agents_offered_per_task,mean_recall_rounds_per_task,"
          << "mean_candidates_scored_per_task,"
          // Selection intelligence (delivery quality)
          << "value_throughput_rate,mean_completion_value,value_loss_to_refusal,"
          // Real impact on edge traversal (BPR factors paid)
          << "mean_bpr_along_route,time_lost_to_congestion_steps,"
          << "n_traversals_in_jam,"
          // Allocation optimality vs MCA full-scan oracle
          << "marginal_cost_ratio_vs_oracle,"
          // RMCA(r) regret diagnostics [Chen et al. 2021] (RMCA mode only)
          << "rmca_relative_regret,rmca_marginal_cost_k1,rmca_marginal_cost_k2,"
          // Temporal complexity (allocation cost only)
          << "mean_allocation_time_us,mean_tam_dijkstra_steps,"
          << "path_compute_time_ms,mean_pure_alloc_time_ms,"
          << "wallclock_ms\n";
    header_written_ = true;
}

// ── Row write ────────────────────────────────────────────────────────────────

void TrainingLogger::write_row(const EpisodeRecord& r) {
    file_ << r.seed             << ','
          << r.global_episode   << ','
          << r.city             << ','
          << r.phase            << ','
          << r.policy_mode      << ','
          << r.total_steps      << ','
          << r.n_agents_max     << ','
          << r.tasks_appeared   << ','
          << r.tasks_completed  << ','
          << std::fixed << std::setprecision(4)
          << r.throughput_rate  << ','
          << r.accept_rate      << ','
          << r.refuse_rate      << ','
          << r.latency_mean     << ','
          << r.latency_per_agent<< ','
          << r.agent_utilisation<< ','
          << r.mean_congestion             << ','
          << r.mean_trip_steps             << ','
          << r.mean_wait_steps             << ','
          << r.mean_road_pd_m              << ','
          << r.delivery_route_efficiency   << ','
          << r.bbox_area_km2    << ','
          << r.convex_hull_area_km2 << ','
          << r.mean_pd_distance_m   << ','
          << r.mean_nn_pickup_m     << ','
          << r.compute_time_per_task_ms      << ','
          << r.compute_time_per_decision_us  << ','
          << r.pairing_violations_runtime    << ','
          << r.capacity_violations_runtime   << ','
          << r.actor_loss       << ','
          << r.critic_loss      << ','
          << r.entropy          << ','
          << r.adv_std          << ','
          << r.n_experiences    << ','
          << r.completion_per_accepted     << ','
          << r.unfinished_accept_rate      << ','
          << r.mean_congestion_at_decision << ','
          << r.n_ghost_active_mean         << ','
          << r.congestion_profile_label    << ','
          << r.agent_completed_gini        << ','
          << r.agent_completed_std         << ','
          << r.mean_imp_accepted           << ','
          << r.mean_imp_refused            << ','
          << r.accept_rate_high_cong       << ','
          << r.accept_rate_low_cong        << ','
          << r.mean_extra_steps_per_task   << ','
          << r.peak_congestion             << ','
          << r.mean_overlap_edges          << ','
          << r.congestion_variance         << ','
          << r.route_congestion_exposure   << ','
          << r.max_agent_completed         << ','
          << r.min_agent_completed         << ','
          << r.total_fleet_distance_m      << ','
          << r.mean_agents_offered_per_task    << ','
          << r.mean_recall_rounds_per_task     << ','
          << r.mean_candidates_scored_per_task << ','
          << r.value_throughput_rate           << ','
          << r.mean_completion_value           << ','
          << r.value_loss_to_refusal           << ','
          << r.mean_bpr_along_route            << ','
          << r.time_lost_to_congestion_steps   << ','
          << r.n_traversals_in_jam             << ','
          << r.marginal_cost_ratio_vs_oracle   << ','
          << r.rmca_relative_regret            << ','
          << r.rmca_marginal_cost_k1           << ','
          << r.rmca_marginal_cost_k2           << ','
          << r.mean_allocation_time_us         << ','
          << r.mean_tam_dijkstra_steps         << ','
          << r.path_compute_time_ms            << ','
          << r.mean_pure_alloc_time_ms         << ','
          << r.wallclock_ms     << '\n';
}

// ── Public interface ─────────────────────────────────────────────────────────

void TrainingLogger::push(const EpisodeRecord& r) {
    records_.push_back(r);
    write_row(r);
}

void TrainingLogger::flush() {
    file_.flush();
}

// ── Cross-seed summary ───────────────────────────────────────────────────────

void TrainingLogger::write_summary(const std::string& path,
                                   const std::vector<EpisodeRecord>& records,
                                   int seed) {
    static constexpr const char* kHeader =
        "seed,city,phase,policy_mode,"
        "n_episodes,"
        "throughput_mean,throughput_std,"
        "accept_rate_mean,accept_rate_std,"
        "latency_mean_mean,latency_mean_std,"
        "utilisation_mean,utilisation_std,"
        "mean_congestion_mean,mean_congestion_std,"
        "mean_trip_steps_mean,mean_trip_steps_std,"
        "mean_wait_steps_mean,mean_wait_steps_std,"
        "mean_road_pd_m_mean,mean_road_pd_m_std,"
        "delivery_route_efficiency_mean,delivery_route_efficiency_std,"
        "actor_loss_mean,critic_loss_mean,entropy_mean";

    // Group by (city, phase, policy_mode) and compute stats.
    auto key_of = [](const EpisodeRecord& r) {
        return r.city + "|" + r.phase + "|" + r.policy_mode;
    };

    std::vector<std::string> seen_keys;
    for (const auto& r : records) {
        std::string k = key_of(r);
        if (std::find(seen_keys.begin(), seen_keys.end(), k) == seen_keys.end())
            seen_keys.push_back(k);
    }

    // Build the set of (seed,city,phase,mode) CSV-key prefixes we're about to
    // write — any existing row with the same prefix is stale and must be
    // dropped to keep summary.csv idempotent across re-runs.
    auto csv_key = [seed](const std::string& city,
                          const std::string& phase,
                          const std::string& mode) {
        return std::to_string(seed) + "," + city + "," + phase + "," + mode + ",";
    };
    std::vector<std::string> new_keys;
    new_keys.reserve(seen_keys.size());
    for (const auto& k : seen_keys) {
        // unpack city|phase|mode
        auto p1 = k.find('|');
        auto p2 = k.find('|', p1 + 1);
        new_keys.push_back(csv_key(k.substr(0, p1),
                                   k.substr(p1 + 1, p2 - p1 - 1),
                                   k.substr(p2 + 1)));
    }

    // Read surviving rows from the existing file (drop matching keys).
    std::vector<std::string> surviving;
    if (fs::exists(path)) {
        std::ifstream in(path);
        std::string line;
        bool first = true;
        while (std::getline(in, line)) {
            if (first) { first = false; continue; }   // skip header
            if (line.empty()) continue;
            bool drop = false;
            for (const auto& nk : new_keys) {
                if (line.rfind(nk, 0) == 0) { drop = true; break; }
            }
            if (!drop) surviving.push_back(line);
        }
    }

    // Truncate + rewrite: header, surviving rows, then new rows below.
    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) return;
    f << kHeader << '\n';
    for (const auto& line : surviving) f << line << '\n';

    for (const auto& k : seen_keys) {
        std::vector<const EpisodeRecord*> group;
        for (const auto& r : records)
            if (key_of(r) == k) group.push_back(&r);
        if (group.empty()) continue;

        auto mean_of = [&](auto fn) {
            float s = 0.f;
            for (const auto* r : group) s += fn(*r);
            return s / group.size();
        };
        auto std_of = [&](auto fn, float mu) {
            float s = 0.f;
            for (const auto* r : group) { float d = fn(*r) - mu; s += d * d; }
            return std::sqrt(s / group.size());
        };

        float thr_m  = mean_of([](const EpisodeRecord& r){ return r.throughput_rate; });
        float acc_m  = mean_of([](const EpisodeRecord& r){ return r.accept_rate; });
        float lat_m  = mean_of([](const EpisodeRecord& r){ return r.latency_mean; });
        float uti_m  = mean_of([](const EpisodeRecord& r){ return r.agent_utilisation; });
        float cng_m  = mean_of([](const EpisodeRecord& r){ return r.mean_congestion; });
        float trp_m  = mean_of([](const EpisodeRecord& r){ return r.mean_trip_steps; });
        float wt_m   = mean_of([](const EpisodeRecord& r){ return r.mean_wait_steps; });
        float rpd_m  = mean_of([](const EpisodeRecord& r){ return r.mean_road_pd_m; });
        float eff_m  = mean_of([](const EpisodeRecord& r){ return r.delivery_route_efficiency; });

        f << seed << ','
          << group[0]->city << ',' << group[0]->phase << ',' << group[0]->policy_mode << ','
          << group.size()   << ','
          << std::fixed << std::setprecision(4)
          << thr_m  << ',' << std_of([](const EpisodeRecord& r){ return r.throughput_rate; }, thr_m) << ','
          << acc_m  << ',' << std_of([](const EpisodeRecord& r){ return r.accept_rate; },    acc_m) << ','
          << lat_m  << ',' << std_of([](const EpisodeRecord& r){ return r.latency_mean; },   lat_m) << ','
          << uti_m  << ',' << std_of([](const EpisodeRecord& r){ return r.agent_utilisation; },uti_m) << ','
          << cng_m  << ',' << std_of([](const EpisodeRecord& r){ return r.mean_congestion; }, cng_m) << ','
          << trp_m  << ',' << std_of([](const EpisodeRecord& r){ return r.mean_trip_steps; }, trp_m) << ','
          << wt_m   << ',' << std_of([](const EpisodeRecord& r){ return r.mean_wait_steps; },  wt_m)  << ','
          << rpd_m  << ',' << std_of([](const EpisodeRecord& r){ return r.mean_road_pd_m; },   rpd_m) << ','
          << eff_m  << ',' << std_of([](const EpisodeRecord& r){ return r.delivery_route_efficiency; }, eff_m) << ','
          << mean_of([](const EpisodeRecord& r){ return r.actor_loss; })  << ','
          << mean_of([](const EpisodeRecord& r){ return r.critic_loss; }) << ','
          << mean_of([](const EpisodeRecord& r){ return r.entropy; })     << '\n';
    }
}

// ── Helper ───────────────────────────────────────────────────────────────────

EpisodeRecord make_record(const RunResult& result,
                          int seed, int global_episode,
                          const std::string& city,
                          const std::string& phase,
                          const std::string& policy_mode_str,
                          int n_agents_max) {
    EpisodeRecord r;
    r.seed              = seed;
    r.global_episode    = global_episode;
    r.city              = city;
    r.phase             = phase;
    r.policy_mode       = policy_mode_str;
    r.n_agents_max      = n_agents_max;

    const auto& m = result.metrics;
    r.total_steps       = m.total_steps;
    r.tasks_appeared    = m.tasks_appeared;
    r.tasks_completed   = m.tasks_completed;
    r.throughput_rate   = m.throughput_rate;
    r.accept_rate       = m.accept_rate;
    r.refuse_rate       = m.refuse_rate;
    r.latency_mean      = m.latency_mean;
    r.latency_per_agent = m.latency_per_agent;
    r.agent_utilisation = m.agent_utilisation;
    r.mean_congestion            = m.mean_congestion;
    r.mean_trip_steps            = m.mean_trip_steps;
    r.mean_wait_steps            = m.mean_wait_steps;
    r.mean_road_pd_m             = m.mean_road_pd_m;
    r.delivery_route_efficiency  = m.delivery_route_efficiency;
    r.bbox_area_km2          = m.bbox_area_km2;
    r.convex_hull_area_km2   = m.convex_hull_area_km2;
    r.mean_pd_distance_m     = m.mean_pd_distance_m;
    r.mean_nn_pickup_m       = m.mean_nn_pickup_m;
    r.compute_time_per_task_ms      = m.compute_time_per_task_ms;
    r.compute_time_per_decision_us  = m.compute_time_per_decision_us;
    r.pairing_violations_runtime    = m.pairing_violations_runtime;
    r.capacity_violations_runtime   = m.capacity_violations_runtime;
    r.completion_per_accepted       = m.completion_per_accepted;
    r.unfinished_accept_rate        = m.unfinished_accept_rate;
    r.mean_congestion_at_decision   = m.mean_congestion_at_decision;
    r.n_ghost_active_mean           = m.n_ghost_active_mean;
    r.congestion_profile_label      = m.congestion_profile_label;
    r.agent_completed_gini          = m.agent_completed_gini;
    r.agent_completed_std           = m.agent_completed_std;
    r.mean_imp_accepted             = m.mean_imp_accepted;
    r.mean_imp_refused              = m.mean_imp_refused;
    r.accept_rate_high_cong         = m.accept_rate_high_cong;
    r.accept_rate_low_cong          = m.accept_rate_low_cong;
    r.mean_extra_steps_per_task     = m.mean_extra_steps_per_task;
    r.peak_congestion               = m.peak_congestion;
    r.mean_overlap_edges            = m.mean_overlap_edges;
    r.congestion_variance           = m.congestion_variance;
    r.route_congestion_exposure     = m.route_congestion_exposure;
    r.mean_agents_offered_per_task    = m.mean_agents_offered_per_task;
    r.mean_recall_rounds_per_task     = m.mean_recall_rounds_per_task;
    r.mean_candidates_scored_per_task = m.mean_candidates_scored_per_task;
    r.value_throughput_rate           = m.value_throughput_rate;
    r.mean_completion_value           = m.mean_completion_value;
    r.value_loss_to_refusal           = m.value_loss_to_refusal;
    r.mean_bpr_along_route            = m.mean_bpr_along_route;
    r.time_lost_to_congestion_steps   = m.time_lost_to_congestion_steps;
    r.n_traversals_in_jam             = m.n_traversals_in_jam;
    r.marginal_cost_ratio_vs_oracle   = m.marginal_cost_ratio_vs_oracle;
    r.rmca_relative_regret            = m.rmca_relative_regret;
    r.rmca_marginal_cost_k1           = m.rmca_marginal_cost_k1;
    r.rmca_marginal_cost_k2           = m.rmca_marginal_cost_k2;
    r.mean_allocation_time_us         = m.mean_allocation_time_us;
    r.mean_tam_dijkstra_steps         = m.mean_tam_dijkstra_steps;
    r.path_compute_time_ms            = m.path_compute_time_ms;
    r.mean_pure_alloc_time_ms         = m.mean_pure_alloc_time_ms;
    r.max_agent_completed           = m.max_agent_completed;
    r.min_agent_completed           = m.min_agent_completed;
    r.total_fleet_distance_m        = m.total_fleet_distance_m;

    const auto& ts = result.train_stats;
    r.actor_loss        = ts.actor_loss;
    r.critic_loss       = ts.critic_loss;
    r.entropy           = ts.entropy;
    r.adv_std           = ts.adv_std;
    r.n_experiences     = ts.n_exp;

    r.wallclock_ms      = result.wallclock_ms;
    return r;
}
