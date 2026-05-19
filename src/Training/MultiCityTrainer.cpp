#include "MultiCityTrainer.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"
#include "DMASforPD/Policy/IPPOPolicy.hpp"
#include "DMASforPD/Policy/MapperPolicy.hpp"
#include "DMASforPD/Policy/FaithfulMapperPolicy.hpp"
#include "DMASforPD/Policy/HybridPolicy.hpp"
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
static void customize_episode_for_city(EpisodeConfig& ep, const CityConfig& cc) {
    const double a = cc.area_km2;
    // Tier thresholds match Tokyo_Small / Tokyo_Medium / Tokyo_Large areas.
    // Multi-task FIFO queue per agent — safe after the receive_task() reorder
    // fix. Bigger graph ⇒ deeper queue so agents stay busy across long trips.
    if (a < 50.0) {
        // Small (~25 km²): ~150m–1500m trips, ~150 tasks/episode.
        ep.phases = {
            { 1000,  6.f, 4, 5, 0.0f, 3 },
            { 1500,  8.f, 5, 7, 0.5f, 4 },
            { 1100, 10.f, 7, 8, 1.0f, 5 },
        };
        ep.min_task_dist_m = 150.f;
        ep.max_task_dist_m = 1500.f;
        ep.hot_zone_radius = 300.f;
        ep.max_tasks_per_agent = 3;
    } else if (a < 300.0) {
        // Medium (~144 km²): ~250m–3500m trips, ~90 tasks/episode.
        ep.phases = {
            { 1000, 3.f,  6,  8, 0.0f, 4 },
            { 1500, 4.f,  8, 10, 0.5f, 6 },
            { 1100, 5.f, 10, 12, 1.0f, 7 },
        };
        ep.min_task_dist_m = 250.f;
        ep.max_task_dist_m = 3500.f;
        ep.hot_zone_radius = 500.f;
        ep.max_tasks_per_agent = 3;
    } else {
        // Large (>=300 km²): ~400m–5000m trips, ~80 tasks/episode.
        ep.phases = {
            { 1000, 2.f,  8, 10, 0.0f, 4 },
            { 1500, 3.f, 10, 13, 0.5f, 6 },
            { 1100, 4.f, 13, 15, 1.0f, 8 },
        };
        ep.min_task_dist_m = 400.f;
        ep.max_task_dist_m = 5000.f;
        ep.hot_zone_radius = 800.f;
        ep.max_tasks_per_agent = 4;
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

// ── Per-episode scenario sampler ──────────────────────────────────────────────
//
// Returns an EpisodeScenario with random density × agent multipliers. Compared
// to the previous version, the distribution now spans four regimes and applies
// the multipliers more broadly (agents may also exceed 1.0). This gives the
// policy gradient signal across the full saturation spectrum:
//
//   15 % slack         : density ~ U(0.3, 0.6),  agents ~ U(1.0, 1.4)
//                        Plenty of agents, few tasks. Policy learns when to
//                        be picky about task quality (low-profit refusals).
//   35 % normal        : density ~ U(0.7, 1.3),  agents ~ U(0.8, 1.2)
//                        Balanced regime — most realistic.
//   30 % stress-light  : density ~ U(1.3, 1.8),  agents ~ U(0.6, 0.9)
//                        Demand exceeds supply lightly — selective acceptance
//                        starts mattering.
//   20 % stress-heavy  : density ~ U(1.8, 2.8),  agents ~ U(0.4, 0.7)
//                        Strong over-saturation — refusal becomes essential
//                        and the deliverability feature decides which tasks
//                        are worth committing to.
//
// All multipliers stack on top of the city-specific phase config produced by
// customize_episode_for_city, so a "slack" episode on Tokyo_Small (4-8 agents
// nominal) and the same on LosAngeles_Medium (8-15 nominal) both scale
// proportionally to the city's problem size.
static EpisodeScenario sample_scenario(std::mt19937& rng, bool disable_slack = false) {
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    const float p = u01(rng);
    auto sample_uniform = [&](float a, float b){
        return a + u01(rng) * (b - a);
    };
    EpisodeScenario s;
    if (disable_slack) {
        // 3-regime sampler — replicates May 17 baseline (no slack regime).
        // Observed distribution from May 17 log: ~53% normal, 29% stress_light,
        // 18% stress_heavy. Thresholds picked to match those proportions.
        if (p < 0.53f) {
            s.density_mult = sample_uniform(0.7f, 1.3f);
            s.agents_mult  = sample_uniform(0.8f, 1.2f);
            s.label        = "normal";
        } else if (p < 0.82f) {
            s.density_mult = sample_uniform(1.3f, 1.8f);
            s.agents_mult  = sample_uniform(0.6f, 0.9f);
            s.label        = "stress_light";
        } else {
            s.density_mult = sample_uniform(1.8f, 2.8f);
            s.agents_mult  = sample_uniform(0.4f, 0.7f);
            s.label        = "stress_heavy";
        }
        return s;
    }
    // Default 4-regime sampler (current production).
    if (p < 0.15f) {
        s.density_mult = sample_uniform(0.3f, 0.6f);
        s.agents_mult  = sample_uniform(1.0f, 1.4f);
        s.label        = "slack";
    } else if (p < 0.50f) {
        s.density_mult = sample_uniform(0.7f, 1.3f);
        s.agents_mult  = sample_uniform(0.8f, 1.2f);
        s.label        = "normal";
    } else if (p < 0.80f) {
        s.density_mult = sample_uniform(1.3f, 1.8f);
        s.agents_mult  = sample_uniform(0.6f, 0.9f);
        s.label        = "stress_light";
    } else {
        s.density_mult = sample_uniform(1.8f, 2.8f);
        s.agents_mult  = sample_uniform(0.4f, 0.7f);
        s.label        = "stress_heavy";
    }
    return s;
}

// Fixed stress scenario for stress evaluation: ~4× over-saturation.
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
        case PolicyMode::LaCAM:           return "LaCAM";
        case PolicyMode::PIBT:            return "PIBT";
        case PolicyMode::CongestionAware: return "CongestionAware";
        case PolicyMode::MCA:             return "MCA";
        case PolicyMode::TrafficFlow:     return "TrafficFlow";
        case PolicyMode::TokenPassing:    return "TokenPassing";
        case PolicyMode::DoubleHorizon:   return "DoubleHorizon";
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
        PolicyMode::TamAlwaysAccept,
        PolicyMode::Greedy, PolicyMode::Random, PolicyMode::InsertionGreedy,
        PolicyMode::PIBT, PolicyMode::CongestionAware,
        PolicyMode::MCA, PolicyMode::TrafficFlow, PolicyMode::TokenPassing });

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
            for (const EpisodeScenario& sc : scenarios) {
                const std::string phase = single_scenario
                    ? "eval" : std::string("eval_") + sc.label;
                for (int e = 0; e < cfg.n_eval_episodes; ++e) {
                    RunResult res = runner.run(ca.index, num_cities, sc);
                    EpisodeRecord rec = make_record(
                        res, seed, global_ep++,
                        ca.config->name, phase, name,
                        ca.ep_cfg.max_agents());
                    logger.push(rec);
                    std::cout << "    [" << phase
                              << " " << ca.config->name << "/" << name
                              << "/" << (e + 1) << "/" << cfg.n_eval_episodes
                              << "]  thr=" << res.metrics.throughput_rate
                              << "  acc=" << res.metrics.accept_rate
                              << "  " << res.wallclock_ms << "ms\n";
                }
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
        PolicyMode::TamAlwaysAccept, PolicyMode::CongestionAware,
        PolicyMode::MCA, PolicyMode::TrafficFlow, PolicyMode::TokenPassing });

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
            for (const EpisodeScenario& sc : scenarios) {
                const std::string phase = single_scenario
                    ? "generalize" : std::string("generalize_") + sc.label;
                for (int e = 0; e < cfg.n_eval_episodes; ++e) {
                    RunResult res = runner.run(ca.index, num_gen, sc);
                    EpisodeRecord rec = make_record(
                        res, seed, global_ep++,
                        ca.config->name, phase, name,
                        ca.ep_cfg.max_agents());
                    logger.push(rec);
                    std::cout << "    [" << phase
                              << " " << ca.config->name << "/" << name
                              << "/" << (e + 1) << "/" << cfg.n_eval_episodes
                              << "]  thr=" << res.metrics.throughput_rate
                              << "  acc=" << res.metrics.accept_rate
                              << "  " << res.wallclock_ms << "ms\n";
                }
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
        PolicyMode::TamAlwaysAccept,
        PolicyMode::Greedy, PolicyMode::Random, PolicyMode::InsertionGreedy,
        PolicyMode::PIBT, PolicyMode::CongestionAware,
        PolicyMode::MCA, PolicyMode::TrafficFlow, PolicyMode::TokenPassing });

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
                RunResult res = runner.run(ca.index, num_cities, kStressScenario);
                EpisodeRecord rec = make_record(
                    res, seed, global_ep++,
                    ca.config->name, "stress", name,
                    ca.ep_cfg.max_agents());
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
    const auto train_ptrs = CityRegistry::train_cities();
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
        const auto gen_ptrs = CityRegistry::comparison_cities();
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

        // Re-initialise IPPO (shared actor + per-agent local critics) for this seed.
        auto& ippo = IPPOPolicy::shared();
        int learning_pool = 0;
        for (int i = 0; i < num_cities; ++i)
            learning_pool = std::max(learning_pool,
                static_cast<int>(std::ceil(assets[i]->ep_cfg.max_agents() * 1.5)));
        ippo.ensure_agents(learning_pool);
        ippo.init_xavier(init_rng);
        ippo.clear_buffer_all();

        // Re-initialise MAPPER (per-agent decentralised + enhanced Ev) for this seed.
        // Same pool size — slots are mapped to the same agent ids used by IPPO.
        auto& mapper = MapperPolicy::shared();
        mapper.ensure_agents(learning_pool);
        mapper.init_xavier_all(init_rng);
        mapper.clear_buffer_all();

        // Re-initialise FaithfulMAPPER (per-agent decentralised + paper-faithful Ev:
        // probabilistic replacement with EXACT copy of best agent, no mutation).
        // Trained side-by-side with MAPPER on the same scenario draws to allow a
        // direct head-to-head comparison of the two evolution mechanisms.
        auto& faithful = FaithfulMapperPolicy::shared();
        faithful.ensure_agents(learning_pool);
        faithful.init_xavier_all(init_rng);
        faithful.clear_buffer_all();

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
            try_load("FMAPPER ", cfg.faithful_mapper_policy_path,
                     [&](const std::string& p){ return faithful.load(p); });
        }

        // ── Hybrid base initialisation ──────────────────────────────────────
        // Hybrid uses MAPPO's actor as its frozen base. Wire it up here so that
        // any episode running with policy_mode = Hybrid sees the correct base.
        // If MAPPO was loaded from checkpoint just above, the base is now the
        // trained policy. Otherwise it inherits the Xavier-init MAPPO.
        HybridPolicy::shared().set_base_from(policy);
        HybridPolicy::shared().ensure_agents(learning_pool);
        HybridPolicy::shared().clear_buffer_all();
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
            check("FaithfulMAPPER", PolicyMode::FaithfulMAPPER, cfg.faithful_mapper_policy_path);
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

        // Independent RNG for scenario sampling so per-episode difficulty
        // randomisation does not perturb policy weight init reproducibility.
        std::mt19937 scenario_rng(static_cast<uint32_t>(seed) ^ 0x9E3779B9u);

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
            faithful.set_progress(progress);

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
            const bool train_mapper   = train_mode_active(PolicyMode::MAPPER);
            const bool train_faithful = train_mode_active(PolicyMode::FaithfulMAPPER);
            const bool train_mappo    = train_mode_active(PolicyMode::MAPPO);

            for (int ci : train_indices) {
                CityAssets&    ca     = *assets[ci];
                EpisodeRunner& runner = *runners[ci];

                // The three learning baselines see the SAME scenario draw so
                // their training conditions are directly comparable. The
                // EpisodeRunner resets GlobalMemory state in run(), so the
                // three back-to-back episodes are independent except for the
                // persistent A* path cache.
                //
                // Order: IPPO first (warms A* cache), then MAPPER, then MAPPO.
                // MAPPO's centralised critic is the most expensive to train;
                // running it last lets it benefit from a fully warmed cache,
                // balancing wall-clock time across the three policies.
                const EpisodeScenario sc = sample_scenario(
                    scenario_rng, cfg.disable_slack_regime);

                // ── IPPO training episode (skipped if not in train_modes) ───
                if (train_ippo) {
                    runner.train_mode  = true;
                    runner.policy_mode = PolicyMode::IPPO;
                    RunResult res = runner.run(ca.index, num_cities, sc);
                    logger.push(make_record(
                        res, seed, global_ep++,
                        ca.config->name, "train", "IPPO",
                        ca.ep_cfg.max_agents()));
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
                        ca.ep_cfg.max_agents()));
                    if (global_ep % cfg.log_every == 0)
                        log_train(res, "MAPPER", mapper.hparams.epochs);
                }

                // ── FaithfulMAPPER training episode (skipped if not in train_modes) ─
                // Same per-agent architecture as MAPPER but paper-faithful Ev:
                // probabilistic replacement of weights with exact copies of the
                // best agent (no mutation noise). Runs on the SAME scenario draw
                // as MAPPER above for head-to-head comparison.
                if (train_faithful) {
                    runner.train_mode  = true;
                    runner.policy_mode = PolicyMode::FaithfulMAPPER;
                    RunResult res = runner.run(ca.index, num_cities, sc);
                    logger.push(make_record(
                        res, seed, global_ep++,
                        ca.config->name, "train", "FaithfulMAPPER",
                        ca.ep_cfg.max_agents()));
                    if (global_ep % cfg.log_every == 0)
                        log_train(res, "FMAPPER", faithful.hparams.epochs);
                }

                // ── MAPPO training episode (skipped if not in train_modes) ──
                if (train_mappo) {
                    runner.train_mode  = true;
                    runner.policy_mode = PolicyMode::MAPPO;
                    RunResult res = runner.run(ca.index, num_cities, sc);
                    logger.push(make_record(
                        res, seed, global_ep++,
                        ca.config->name, "train", "MAPPO",
                        ca.ep_cfg.max_agents()));
                    if (global_ep % cfg.log_every == 0)
                        log_train(res, "MAPPO", policy.hparams.epochs);
                }
            }

            // ── MAPPER Evolutionary RL step (enhanced variant) ────────────
            // Every `ev_params.period_rounds` rounds, rank decentralised
            // policies by their rolling fitness and replace the bottom worst_frac
            // with Gaussian-mutated copies of a random elite (Liu et al.
            // IROS 2020, augmented with mutation noise N(0, mutation_std²)).
            // Combines gradient-based PPO updates with genetic-style
            // exploration; runs on a multi-episode cadence (single-episode
            // fitness is too noisy).
            const int ev_period = std::max(1, mapper.ev_params.period_rounds);
            if (train_mapper && (round + 1) % ev_period == 0) {
                const int n_mut = mapper.evolutionary_step();
                if (n_mut > 0 && cfg.verbose) {
                    std::cout << "  [MAPPER-Ev @ round " << (round + 1)
                              << "] mutated " << n_mut << "/"
                              << mapper.n_agents() << " policies\n";
                }
            }

            // ── FaithfulMAPPER Evolutionary RL step (paper-faithful) ──────
            // Implements Algorithm 1 of Liu et al. exactly:
            //   p_i = 1 - exp(η·R̄_i)/exp(η·R̄_best)
            //   if uniform < p_i: Θ_i ← Θ_best   (exact copy, no mutation)
            // Reuses the same period_rounds cadence as MAPPER for fairness.
            const int faithful_ev_period =
                std::max(1, faithful.ev_params.period_rounds);
            if (train_faithful && (round + 1) % faithful_ev_period == 0) {
                const int n_rep = faithful.evolutionary_step();
                if (n_rep > 0 && cfg.verbose) {
                    std::cout << "  [FaithfulMAPPER-Ev @ round " << (round + 1)
                              << "] replaced " << n_rep << "/"
                              << faithful.n_agents() << " policies (no mutation)\n";
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
            if (should_save(PolicyMode::MAPPER)) {
                const std::string subdir = cfg.output_dir + "/mapper_enhanced";
                ensure_dir(subdir);
                const std::string ckpt = subdir + "/mapper_seed"
                                       + std::to_string(seed) + ".bin";
                mapper.save(ckpt);
                std::cout << "  [Checkpoint] MAPPER         saved to " << ckpt
                          << "  (" << mapper.n_agents()
                          << " agents, enhanced Ev with mutation)\n";
            } else {
                std::cout << "  [Checkpoint] MAPPER         skipped (not in train_modes)\n";
            }
            if (should_save(PolicyMode::FaithfulMAPPER)) {
                const std::string subdir = cfg.output_dir + "/mapper_faithful";
                ensure_dir(subdir);
                const std::string ckpt = subdir + "/mapper_seed"
                                       + std::to_string(seed) + ".bin";
                faithful.save(ckpt);
                std::cout << "  [Checkpoint] FaithfulMAPPER saved to " << ckpt
                          << "  (" << faithful.n_agents()
                          << " agents, paper-faithful Ev: copy without mutation)\n";
            } else {
                std::cout << "  [Checkpoint] FaithfulMAPPER skipped (not in train_modes)\n";
            }
        }

        // ── 8. Per-seed summary ───────────────────────────────────────────
        TrainingLogger::write_summary(summary_path, logger.records(), seed);

        std::cout << "Seed " << s << " done — " << global_ep << " episodes.\n\n";
    }

    std::cout << "Training complete. Results in " << cfg.output_dir << "\n";
}
