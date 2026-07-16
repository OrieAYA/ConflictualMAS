#include "TrainingEvaluation/Run/Trainer.hpp"
#include "Environment/Structure/EpisodeManager.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "DMASforPD/Policy/MAPPO.hpp"
#include "DMASforPD/Policy/IPPO.hpp"
#include "DMASforPD/Policy/MAPPERInspired.hpp"
#include "DMASforPD/Policy/Hybrid.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "SoTA/SolverFramework.hpp"
#include "SoTA/Standalone/CA.hpp"
#include "SoTA/Standalone/HAPC.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>
static long long process_commit_mb() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)))
        return static_cast<long long>(pmc.PrivateUsage) / (1024 * 1024);
    return -1;
}
#else
static long long process_commit_mb() { return -1; }
#endif

namespace fs = std::filesystem;

// ── City loading ──────────────────────────────────────────────────────────────

// Set the per-city size scalar SCE and its derived scale. SCE fixes the event
// counts (tasks/ghosts) via env_scale; the fleet base is round(10·SCE) here and
// gets ×AM per scenario at run time. Phases are a constant-fleet 3600-step
// timeline carrying only the density label and the hot-zone count.
void MultiCityTrainer::customize_episode_for_city(EpisodeConfig& ep, const CityConfig& cc) {
    // Area tracks SCE (Small 25 / Medium 50 / Large 75 km²), so SCE also
    // reflects graph size → consistent congestion density across sizes.
    const double a = cc.area_km2;
    float sce;
    if      (a < 40.0) sce = 1.f;   // Small  (~25 km²)
    else if (a < 65.0) sce = 2.f;   // Medium (~50 km²)
    else               sce = 3.f;   // Large  (~75 km²)
    ep.env_scale = sce;

    const int fleet = event_tuning::derived_fleet_size(sce, 1.f);   // 10·SCE (×AM at run time)
    ep.phases = {
        { 1000, fleet, fleet, 0.0f, 4 },
        { 1500, fleet, fleet, 0.5f, 6 },
        { 1100, fleet, fleet, 1.0f, 8 },
    };
    ep.max_tasks_per_agent = 3;

    if (sce <= 1.f) {
        ep.min_task_dist_m = 150.f;  ep.max_task_dist_m = 1500.f;  ep.hot_zone_radius = 300.f;
    } else if (sce <= 2.f) {
        ep.min_task_dist_m = 250.f;  ep.max_task_dist_m = 3500.f;  ep.hot_zone_radius = 500.f;
    } else {
        ep.min_task_dist_m = 400.f;  ep.max_task_dist_m = 5000.f;  ep.hot_zone_radius = 800.f;
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

// ── PolicyMode <-> string helper for flexible eval mode subsets ───────────────
static const char* policy_mode_label(PolicyMode m) {
    switch (m) {
        case PolicyMode::MAPPO:           return "MAPPO";
        case PolicyMode::IPPO:            return "IPPO";
        case PolicyMode::MAPPER:          return "MAPPER";
        case PolicyMode::Hybrid:          return "Hybrid";
        case PolicyMode::TamAlwaysAccept: return "TamAlwaysAccept";
        case PolicyMode::Greedy:          return "Greedy";
        case PolicyMode::Random:          return "Random";
        case PolicyMode::InsertionGreedy: return "InsertionGreedy";
        case PolicyMode::TokenPassing:    return "TokenPassing";
        case PolicyMode::DoubleHorizon:   return "DoubleHorizon";
        case PolicyMode::DbVNS:           return "DbVNS";
        case PolicyMode::ALNS:            return "ALNS";
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

// ── Evaluation ────────────────────────────────────────────────────────────────
//
// Episode-major protocol: one initial state (SharedEpisodeSetup) per
// (city, scenario, episode) slot, built ONCE and replayed by every method —
// the eval modes on the shared TAM/DbVNS pipeline, then the standalone SoTA
// solvers (CA, HAPC) on the same setup. prepare_run() restores the initial
// state and clears the previous method's buffers before each run; the slot's
// path caches are released once every method has passed.

int MultiCityTrainer::run_eval(
    const TrainingConfig& cfg,
    const std::vector<std::unique_ptr<CityAssets>>& assets,
    std::vector<std::unique_ptr<EpisodeRunner>>& runners,
    int global_ep, int seed,
    TrainingLogger& logger,
    SolverCSVLogger* sota)
{
    const int num_cities = static_cast<int>(assets.size());

    const std::vector<PolicyMode> modes = resolve_eval_modes(cfg, {
        PolicyMode::MAPPO, PolicyMode::IPPO, PolicyMode::MAPPER,
        PolicyMode::RMCA,
        PolicyMode::TamAlwaysAccept,
        PolicyMode::Greedy, PolicyMode::Random,
        PolicyMode::TokenPassing });

    std::vector<EpisodeScenario> scenarios = cfg.eval_scenarios;
    const bool single_scenario = scenarios.empty();
    if (single_scenario)
        scenarios.push_back(EpisodeScenario{1.f, "normal"});

    for (int ci = 0; ci < num_cities; ++ci) {
        CityAssets&    ca     = *assets[ci];
        EpisodeRunner& runner = *runners[ci];
        runner.train_mode = false;

        int sc_idx = 0;
        for (const EpisodeScenario& sc : scenarios) {
            const std::string phase = single_scenario
                ? "eval" : std::string("eval_") + sc.label;
            for (int e = 0; e < cfg.n_eval_episodes; ++e) {
                // Deterministic slot seed: every method sees the SAME task
                // stream, ghost profile and hetero-capacity draws.
                const uint32_t ep_seed = static_cast<uint32_t>(
                    1u + e
                    + 101u * (sc_idx + 1)
                    + 10007u * (ca.index + 1)
                    + 1000003u * static_cast<uint32_t>(seed));
                const SharedEpisodeSetup setup = build_shared_episode_setup(
                    ep_seed, *ca.config, sc, ca.ep_cfg, ca.geo_box);

                for (PolicyMode m : modes) {
                    const char* name = policy_mode_label(m);
                    runner.policy_mode = m;
                    RunResult res = runner.run(ca.index, num_cities, sc,
                                                ep_seed, &setup);
                    logger.push(make_record(
                        res, seed, global_ep++,
                        ca.config->name, phase, name,
                        res.metrics.n_agents_max));
                    const auto& M = res.metrics;
                    std::cout << "    [" << phase
                              << " " << ca.config->name << " " << sc.label
                              << " " << name
                              << " ep" << (e + 1) << "/" << cfg.n_eval_episodes
                              << "]  thr=" << M.throughput_rate
                              << " done=" << M.tasks_completed << "/" << M.tasks_appeared
                              << " cong=x" << M.mean_congestion
                              << " lat=" << M.latency_mean
                              << "  " << res.wallclock_ms << "ms\n";
                }

                if (sota) {
                    SolverRunner srunner(ca.ep_cfg, ca.geo_box, sc, ep_seed);
                    auto run_solver = [&](ISolver& s) {
                        SolverMetrics m = srunner.run(s, &setup);
                        m.city_label = ca.config->name;
                        m.episode    = e;
                        sota->write_row(m);
                        std::cout << "    [" << phase
                                  << " " << ca.config->name << " " << sc.label
                                  << " " << m.solver_name
                                  << " ep" << (e + 1) << "/" << cfg.n_eval_episodes
                                  << "]  thr=" << m.throughput_rate
                                  << " lat=" << m.latency_mean
                                  << "  " << m.wallclock_ms << "ms\n";
                    };
                    { FaithfulCASolver               s; run_solver(s); }
                    { HybridAdaptivePredictiveSolver s; run_solver(s); }
                }

                runner.release_episode_memory();
            }
            ++sc_idx;
        }
    }

    return global_ep;
}

// ── Evaluation protocol ───────────────────────────────────────────────────────
// Loads the per-seed checkpoints produced by train_grid, then runs the
// episode-major sweep (run_eval) on the eval cities. No training happens here.

void MultiCityTrainer::evaluate(const TrainingConfig& cfg) {
    fs::create_directories(cfg.output_dir);

    // ── 1. Load eval cities ───────────────────────────────────────────────
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

    std::cout << "Loading " << num_cities << " cities from " << cfg.cache_root << "\n";
    std::vector<std::unique_ptr<CityAssets>> assets;
    assets.reserve(num_cities);
    for (int i = 0; i < num_cities; ++i)
        assets.push_back(load_city(*train_ptrs[i], i, cfg.episode_cfg, cfg.cache_root));
    std::cout << "All cities loaded.\n\n";

    // ── 2. Multi-seed loop ────────────────────────────────────────────────
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

        auto try_load = [&](const char* name, const std::string& path,
                            auto loader) {
            if (path.empty()) return;
            if (!loader(path))
                throw std::runtime_error(std::string("Could not load ") + name
                    + " from " + path
                    + " — aborting to avoid evaluating fresh-init weights");
            std::cout << "[Policy] Loaded " << name << " from " << path << "\n";
        };
        try_load("MAPPO   ", cfg.policy_path,
                 [&](const std::string& p){ return policy.load(p); });
        try_load("IPPO    ", cfg.ippo_policy_path,
                 [&](const std::string& p){ return ippo.load(p); });
        try_load("MAPPER  ", cfg.mapper_policy_path,
                 [&](const std::string& p){ return mapper.load(p); });

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

        // Any RL mode in eval_modes without a checkpoint path would silently
        // run on fresh-init weights — detect and abort.
        if (!cfg.eval_modes.empty()) {
            auto has_mode = [&](PolicyMode m){
                return std::find(cfg.eval_modes.begin(),
                                  cfg.eval_modes.end(), m)
                       != cfg.eval_modes.end();
            };
            auto check = [&](const char* name, PolicyMode m,
                             const std::string& p) {
                if (has_mode(m) && p.empty()) {
                    throw std::runtime_error(
                        std::string(name)
                        + " is in eval_modes but its checkpoint path is empty"
                        + " — set the corresponding *_policy_path in cfg");
                }
            };
            check("MAPPO",  PolicyMode::MAPPO,  cfg.policy_path);
            check("IPPO",   PolicyMode::IPPO,   cfg.ippo_policy_path);
            check("MAPPER", PolicyMode::MAPPER, cfg.mapper_policy_path);
            // Hybrid base derives from MAPPO.
            if (has_mode(PolicyMode::Hybrid) && cfg.policy_path.empty()) {
                throw std::runtime_error(
                    "Hybrid is in eval_modes but its base (MAPPO) "
                    "path is empty — set cfg.policy_path");
            }
        }

        // Per-city runners (reused across seeds' episodes — A* cache warms up).
        std::vector<std::unique_ptr<EpisodeRunner>> runners;
        runners.reserve(num_cities);
        for (int i = 0; i < num_cities; ++i) {
            CityAssets& ca = *assets[i];
            runners.push_back(std::make_unique<EpisodeRunner>(
                ca.ep_cfg, ca.geo_box,
                static_cast<uint32_t>(seed)));
        }

        TrainingLogger logger(cfg.output_dir, seed);
        const std::string sota_dir = cfg.output_dir + "/sota_standalone";
        fs::create_directories(sota_dir);
        SolverCSVLogger sota(sota_dir + "/sota_seed" + std::to_string(seed) + ".csv");
        sota.write_header();
        int global_ep = 0;

        std::cout << "  -- Eval --\n";
        global_ep = run_eval(cfg, assets, runners, global_ep, seed, logger, &sota);
        sota.close();
        logger.flush();

        TrainingLogger::write_summary(summary_path, logger.records(), seed);

        std::cout << "Seed " << s << " done — " << global_ep << " episodes.\n\n";
    }

    std::cout << "Evaluation complete. Results in " << cfg.output_dir << "\n";
}

// Completed grid points per seed in an existing training CSV. Rows are flushed
// once per grid point, so a point is complete when every selected policy
// logged it. Rows of incomplete points (crash mid-write) are pruned from the
// file — the resumed run re-runs those points, keeping one clean dataset.
static std::unordered_map<int, int> completed_grid_points(
    const std::string& csv_path, int n_policies)
{
    std::unordered_map<int, int> per_seed;
    std::ifstream in(csv_path);
    std::string header, line;
    if (!in.is_open() || !std::getline(in, header)) return per_seed;

    std::vector<std::pair<long long, std::string>> data;   // (seed<<32|ep, raw)
    std::unordered_map<long long, int> rows;
    while (std::getline(in, line)) {
        long long key = -1;
        const size_t c1 = line.find(',');
        const size_t c2 = (c1 == std::string::npos)
            ? std::string::npos : line.find(',', c1 + 1);
        if (c2 != std::string::npos) {
            try {
                const long long s = std::stoi(line.substr(0, c1));
                const long long e = std::stoi(line.substr(c1 + 1, c2 - c1 - 1));
                key = (s << 32) | e;
            } catch (...) {}
        }
        if (key >= 0) ++rows[key];
        data.emplace_back(key, line);
    }
    in.close();

    for (const auto& [key, n] : rows)
        if (n >= n_policies) ++per_seed[static_cast<int>(key >> 32)];

    const auto complete = [&](const std::pair<long long, std::string>& d) {
        return d.first >= 0 && rows[d.first] >= n_policies;
    };
    if (!std::all_of(data.begin(), data.end(), complete)) {
        const std::string tmp = csv_path + ".tmp";
        std::ofstream out(tmp, std::ios::trunc);
        if (out.is_open()) {
            out << header << '\n';
            for (const auto& d : data)
                if (complete(d)) out << d.second << '\n';
            out.close();
            std::error_code ec;
            fs::rename(tmp, csv_path, ec);
        }
    }
    return per_seed;
}

void MultiCityTrainer::train_grid(const TrainingConfig& cfg_in) {
    fs::create_directories(cfg_in.output_dir);

    const std::vector<EpisodeScenario> scenarios =
        cfg_in.train_scenarios.empty() ? make_scenario_grid()
                                       : cfg_in.train_scenarios;

    std::vector<const CityConfig*> train_ptrs;
    {
        const auto& all_cities = CityRegistry::all();
        if (cfg_in.train_city_filter.empty()) {
            train_ptrs = CityRegistry::train_cities();
        } else {
            for (const auto& want : cfg_in.train_city_filter) {
                bool matched = false;
                for (const auto& cc : all_cities)
                    if (cc.name == want) { train_ptrs.push_back(&cc); matched = true; break; }
                if (!matched)
                    std::cout << "  [Warn] train city \"" << want << "\" not found — skipped.\n";
            }
        }
    }
    const int num_cities   = static_cast<int>(train_ptrs.size());
    const int n_seeds      = std::max(1, cfg_in.n_seeds);
    const int eps_per_seed = num_cities * static_cast<int>(scenarios.size());
    const int total_eps    = eps_per_seed * n_seeds;
    if (num_cities == 0 || scenarios.empty()) {
        std::cout << "  [train_grid] nothing to train (no cities/scenarios).\n";
        return;
    }

    TrainingConfig cfg = cfg_in;
    cfg.episode_cfg.rs_e_anneal = std::max(1, eps_per_seed);

    std::cout << "Grid training: " << num_cities << " cities × " << n_seeds
              << " seeds × " << scenarios.size() << " scenarios = "
              << total_eps << " episodes / policy ("
              << eps_per_seed << " per seed).\n";

    std::vector<std::unique_ptr<CityAssets>> assets;
    assets.reserve(num_cities);
    for (int i = 0; i < num_cities; ++i)
        assets.push_back(load_city(*train_ptrs[i], i, cfg.episode_cfg, cfg.cache_root));

    int learning_pool = 0;
    for (int i = 0; i < num_cities; ++i)
        learning_pool = std::max(learning_pool,
            static_cast<int>(std::ceil(assets[i]->ep_cfg.max_agents() * 1.5)));

    std::vector<std::unique_ptr<EpisodeRunner>> runners;
    runners.reserve(num_cities);
    for (int i = 0; i < num_cities; ++i)
        runners.push_back(std::make_unique<EpisodeRunner>(
            assets[i]->ep_cfg, assets[i]->geo_box,
            static_cast<uint32_t>(cfg.start_seed)));

    auto want = [&](PolicyMode m) {
        if (cfg.train_modes.empty()) return true;
        return std::find(cfg.train_modes.begin(), cfg.train_modes.end(), m)
               != cfg.train_modes.end();
    };
    const bool do_mappo  = want(PolicyMode::MAPPO);
    const bool do_ippo   = want(PolicyMode::IPPO);
    const bool do_mapper = want(PolicyMode::MAPPER);

    auto& mappo  = mappo_policy();
    auto& ippo   = ippo_policy();
    auto& mapper = mapper_policy();

    auto save_checkpoints = [&](int seed) {
        if (!cfg.save_policy) return;
        auto dir = [&](const char* sub) {
            const std::string d = cfg.output_dir + "/" + sub;
            std::error_code ec; fs::create_directories(d, ec);
            return d;
        };
        const std::string tag = "_seed" + std::to_string(seed) + ".bin";
        if (do_mappo)  mappo.save (dir("mappo")  + "/policy" + tag);
        if (do_ippo)   ippo.save  (dir("ippo")   + "/ippo"   + tag);
        if (do_mapper) mapper.save(dir("mapper") + "/mapper" + tag);
    };

    const int n_sel = (do_mappo ? 1 : 0) + (do_ippo ? 1 : 0) + (do_mapper ? 1 : 0);
    std::unordered_map<int, int> done_pts;
    if (cfg.resume)
        done_pts = completed_grid_points(
            cfg.output_dir + "/episodes_train.csv", n_sel);

    TrainingLogger logger(cfg.output_dir, std::string("train"), !done_pts.empty());
    const std::string summary_path = cfg.output_dir + "/summary.csv";

    auto reinit_all = [&](int seed) {
        if (do_mappo)  mappo.reinit(static_cast<uint32_t>(seed));
        if (do_ippo)  { ippo.reinit(static_cast<uint32_t>(seed) ^ 0x1Au);
                        ippo.ensure_agents(learning_pool); }
        if (do_mapper) { mapper.reinit(static_cast<uint32_t>(seed) ^ 0x2Bu);
                         mapper.ensure_agents(learning_pool); }
    };

    int    ep             = 0;
    size_t seed_rec_begin = 0;
    for (int si = 0; si < n_seeds; ++si) {
        const int seed = cfg.start_seed + si;

        int done = 0;
        if (auto it = done_pts.find(seed); it != done_pts.end())
            done = std::min(it->second, eps_per_seed);
        if (done >= eps_per_seed) {
            ep += eps_per_seed;
            std::cout << "\n════ seed " << (si + 1) << "/" << n_seeds
                      << " (rng=" << seed << ") — complete, skipped ════\n";
            continue;
        }

        std::cout << "\n════ seed " << (si + 1) << "/" << n_seeds
                  << " (rng=" << seed << ") ════\n";

        reinit_all(seed);
        if (done > 0) {
            const std::string tag = "_seed" + std::to_string(seed) + ".bin";
            bool ok = true;
            if (do_mappo)  ok = ok && mappo.load (cfg.output_dir + "/mappo/policy" + tag);
            if (do_ippo)   ok = ok && ippo.load  (cfg.output_dir + "/ippo/ippo"    + tag);
            if (do_mapper) ok = ok && mapper.load(cfg.output_dir + "/mapper/mapper"+ tag);
            if (ok) {
                if (do_ippo)   ippo.ensure_agents(learning_pool);
                if (do_mapper) mapper.ensure_agents(learning_pool);
                std::cout << "  Resume: " << done << "/" << eps_per_seed
                          << " episodes done — continuing at episode "
                          << (done + 1) << ".\n";
            } else {
                std::cout << "  [Warn] checkpoint load failed — seed restarts from 0.\n";
                done = 0;
                reinit_all(seed);
            }
        }

        int ep_in_seed = 0;
        for (size_t sc = 0; sc < scenarios.size(); ++sc) {
            const EpisodeScenario& scen = scenarios[sc];
            std::cout << "  ── scenario " << (sc + 1) << "/" << scenarios.size()
                      << "  " << scen.label
                      << "  (AM=" << scen.agents_mult << ", "
                      << num_cities << " cities × " << (do_mappo + do_ippo + do_mapper)
                      << " policies) ──\n" << std::flush;
            for (int ci = 0; ci < num_cities; ++ci) {
                if (ep_in_seed < done) { ++ep; ++ep_in_seed; continue; }
                CityAssets&    ca     = *assets[ci];
                EpisodeRunner& runner = *runners[ci];

                const float progress = (eps_per_seed > 1)
                    ? static_cast<float>(ep_in_seed) / (eps_per_seed - 1) : 0.f;
                if (do_mappo)  mappo.set_progress(progress);
                if (do_ippo)   ippo.set_progress(progress);
                if (do_mapper) mapper.set_progress(progress);

                runner.set_global_episode(ep_in_seed);

                uint32_t ep_seed =
                      (static_cast<uint32_t>(seed)   * 2654435761u)
                    ^ (static_cast<uint32_t>(ci + 1) *      40503u)
                    ^ (static_cast<uint32_t>(sc + 1) * 2246822519u);
                ep_seed |= 1u;

                // One generated episode (tasks, capacities, starts, ghost
                // seed) shared by every policy at this grid point.
                const SharedEpisodeSetup setup = build_shared_episode_setup(
                    ep_seed, *ca.config, scen, ca.ep_cfg, ca.geo_box);

                auto run_one = [&](PolicyMode m, const char* tag) {
                    runner.train_mode  = true;
                    runner.policy_mode = m;
                    RunResult res = runner.run(ci, num_cities, scen, ep_seed, &setup);
                    logger.push(make_record(res, seed, ep,
                                            ca.config->name, "train", tag,
                                            res.metrics.n_agents_max));
                    const auto& M = res.metrics;
                    if (cfg.verbose && ep % std::max(1, cfg.log_every) == 0)
                        std::cout << "  [ep " << (ep_in_seed + 1) << "/" << eps_per_seed
                                  << " s" << seed << " " << ca.config->name
                                  << " " << scen.label << " " << tag << "]"
                                  << "  thr="  << M.throughput_rate
                                  << " done="  << M.tasks_completed << "/" << M.tasks_appeared
                                  << " acc="   << M.accept_rate
                                  << " cong=x" << M.mean_congestion
                                  << " lat="   << M.latency_mean
                                  << " agents="<< M.n_agents_max
                                  << " | aloss=" << res.train_stats.actor_loss
                                  << " closs=" << res.train_stats.critic_loss
                                  << " n="     << res.train_stats.n_exp
                                  << "  "      << res.wallclock_ms << "ms"
                                  << "  mem="  << process_commit_mb() << "MB\n"
                                  << std::flush;
                };

                if (do_ippo)   run_one(PolicyMode::IPPO,   "IPPO");
                if (do_mapper) run_one(PolicyMode::MAPPER, "MAPPER");
                if (do_mappo)  run_one(PolicyMode::MAPPO,  "MAPPO");

                ++ep;
                ++ep_in_seed;
                logger.flush();
                save_checkpoints(seed);
                runner.release_episode_memory();
            }
            if (do_mapper && ep_in_seed > done) {
                mapper.on_round_end();
                const int n_rep = mapper.last_evolution_replacements();
                if (n_rep > 0 && cfg.verbose)
                    std::cout << "  [MAPPER-Ev] replaced " << n_rep << " policies\n";
            }
        }

        save_checkpoints(seed);
        {
            const auto& all = logger.records();
            std::vector<EpisodeRecord> seed_records(
                all.begin() + seed_rec_begin, all.end());
            seed_rec_begin = all.size();
            TrainingLogger::write_summary(summary_path, seed_records, seed);
        }
        std::cout << "  Seed " << seed << " done — checkpoints in "
                  << cfg.output_dir << "/{mappo,ippo,mapper}/*_seed"
                  << seed << ".bin\n";
    }

    logger.flush();
    std::cout << "Grid training complete — " << total_eps
              << " episodes / policy. Results in " << cfg.output_dir << "\n";
}

// ── Movement-policy training ──────────────────────────────────────────────────

void MultiCityTrainer::train_movement(const TrainingConfig& cfg) {
    fs::create_directories(cfg.output_dir);

    if (!cfg.episode_cfg.use_movement_policy || !cfg.episode_cfg.movement_train) {
        std::cout << "  [train_movement] episode_cfg.use_movement_policy / "
                     "movement_train must both be set.\n";
        return;
    }

    const std::vector<EpisodeScenario> scenarios =
        cfg.train_scenarios.empty() ? make_scenario_grid()
                                    : cfg.train_scenarios;

    std::vector<const CityConfig*> train_ptrs;
    for (const auto& want : cfg.train_city_filter)
        for (const auto& cc : CityRegistry::all())
            if (cc.name == want) { train_ptrs.push_back(&cc); break; }
    const int num_cities   = static_cast<int>(train_ptrs.size());
    const int n_seeds      = std::max(1, cfg.n_seeds);
    const int eps_per_seed = num_cities * static_cast<int>(scenarios.size());
    if (num_cities == 0 || scenarios.empty()) {
        std::cout << "  [train_movement] nothing to train (no cities/scenarios).\n";
        return;
    }

    std::cout << "Movement training: " << num_cities << " cities × " << n_seeds
              << " seeds × " << scenarios.size() << " scenarios = "
              << eps_per_seed * n_seeds << " episodes.\n"
              << "  Frozen bid policy: MAPPO <- " << cfg.policy_path << "\n";

    std::vector<std::unique_ptr<CityAssets>> assets;
    assets.reserve(num_cities);
    for (int i = 0; i < num_cities; ++i)
        assets.push_back(load_city(*train_ptrs[i], i, cfg.episode_cfg, cfg.cache_root));

    std::vector<std::unique_ptr<EpisodeRunner>> runners;
    runners.reserve(num_cities);
    for (int i = 0; i < num_cities; ++i)
        runners.push_back(std::make_unique<EpisodeRunner>(
            assets[i]->ep_cfg, assets[i]->geo_box,
            static_cast<uint32_t>(cfg.start_seed)));

    auto& mp    = movement_policy();
    auto& mappo = mappo_policy();

    const std::string mv_dir = cfg.output_dir + "/movement";
    fs::create_directories(mv_dir);
    std::ofstream csv(cfg.output_dir + "/movement_train.csv", std::ios::trunc);
    csv << "seed,episode,city,scenario,decisions,replans,adopted,gain_mean,"
           "aloss,closs,entropy,kl,n_exp,n_epochs,"
           "tasks_completed,tasks_appeared,throughput,wallclock_ms\n";

    for (int si = 0; si < n_seeds; ++si) {
        const int seed = cfg.start_seed + si;
        std::cout << "\n════ movement seed " << (si + 1) << "/" << n_seeds
                  << " (rng=" << seed << ") ════\n";

        mappo.reinit(static_cast<uint32_t>(seed));
        if (!mappo.load(cfg.policy_path)) {
            std::cout << "  [Abort] MAPPO checkpoint load failed: "
                      << cfg.policy_path << "\n";
            return;
        }
        mp.reinit(static_cast<uint32_t>(seed) ^ 0x3Cu);
        if (!cfg.lsm_path.empty()) {
            if (lsm_module().load(cfg.lsm_path))
                std::cout << "  Frozen LSM <- " << cfg.lsm_path << "\n";
            else
                std::cout << "  [Warn] LSM checkpoint load failed ("
                          << cfg.lsm_path << ") — f4 stays 0.\n";
        }

        int ep_in_seed = 0;
        for (size_t sc = 0; sc < scenarios.size(); ++sc) {
            const EpisodeScenario& scen = scenarios[sc];
            for (int ci = 0; ci < num_cities; ++ci) {
                CityAssets&    ca     = *assets[ci];
                EpisodeRunner& runner = *runners[ci];

                const float progress = (eps_per_seed > 1)
                    ? static_cast<float>(ep_in_seed) / (eps_per_seed - 1) : 0.f;
                mp.set_progress(progress);
                runner.set_global_episode(ep_in_seed);

                uint32_t ep_seed =
                      (static_cast<uint32_t>(seed)   * 2654435761u)
                    ^ (static_cast<uint32_t>(ci + 1) *      40503u)
                    ^ (static_cast<uint32_t>(sc + 1) * 2246822519u);
                ep_seed |= 1u;

                const SharedEpisodeSetup setup = build_shared_episode_setup(
                    ep_seed, *ca.config, scen, ca.ep_cfg, ca.geo_box);

                runner.train_mode  = false;   // bid side frozen (no exploration)
                runner.policy_mode = PolicyMode::MAPPO;
                RunResult res = runner.run(ci, num_cities, scen, ep_seed, &setup);

                const auto& mv = runner.movement_stats();
                const float gain_mean = (mv.n_replans > 0)
                    ? mv.gain_sum / mv.n_replans : 0.f;
                csv << seed << ',' << ep_in_seed << ',' << ca.config->name << ','
                    << scen.label << ',' << mv.n_decisions << ','
                    << mv.n_replans << ',' << mv.n_adopted << ','
                    << gain_mean << ',' << mv.train.actor_loss << ','
                    << mv.train.critic_loss << ',' << mv.train.entropy << ','
                    << mv.train.kl_approx << ',' << mv.train.n_exp << ','
                    << mv.train.n_epochs << ','
                    << res.metrics.tasks_completed << ','
                    << res.metrics.tasks_appeared << ','
                    << res.metrics.throughput_rate << ','
                    << res.wallclock_ms << '\n';
                csv.flush();

                if (cfg.verbose)
                    std::cout << "  [mv ep " << (ep_in_seed + 1) << "/" << eps_per_seed
                              << " s" << seed << " " << ca.config->name
                              << " " << scen.label << "]"
                              << "  dec=" << mv.n_decisions
                              << " rep=" << mv.n_replans
                              << " adopt=" << mv.n_adopted
                              << " gain=" << gain_mean
                              << " | aloss=" << mv.train.actor_loss
                              << " ent=" << mv.train.entropy
                              << " n=" << mv.train.n_exp
                              << "  thr=" << res.metrics.throughput_rate
                              << "  " << res.wallclock_ms << "ms"
                              << "  mem=" << process_commit_mb() << "MB\n"
                              << std::flush;

                if (cfg.save_policy)
                    mp.save(mv_dir + "/movement_seed" + std::to_string(seed) + ".bin");
                ++ep_in_seed;
                runner.release_episode_memory();
            }
        }
        std::cout << "  Movement seed " << seed << " done — checkpoint "
                  << mv_dir << "/movement_seed" << seed << ".bin\n";
    }
    std::cout << "Movement training complete. Results in "
              << cfg.output_dir << "\n";
}

// ── LSM readout pretraining ───────────────────────────────────────────────────

void MultiCityTrainer::pretrain_lsm(const TrainingConfig& cfg) {
    fs::create_directories(cfg.output_dir);

    if (!cfg.episode_cfg.use_lsm || !cfg.episode_cfg.lsm_train) {
        std::cout << "  [pretrain_lsm] episode_cfg.use_lsm / lsm_train "
                     "must both be set.\n";
        return;
    }

    const std::vector<EpisodeScenario> scenarios =
        cfg.train_scenarios.empty() ? make_scenario_grid()
                                    : cfg.train_scenarios;

    std::vector<const CityConfig*> train_ptrs;
    for (const auto& want : cfg.train_city_filter)
        for (const auto& cc : CityRegistry::all())
            if (cc.name == want) { train_ptrs.push_back(&cc); break; }
    const int num_cities   = static_cast<int>(train_ptrs.size());
    const int n_seeds      = std::max(1, cfg.n_seeds);
    const int eps_per_seed = num_cities * static_cast<int>(scenarios.size());
    if (num_cities == 0 || scenarios.empty()) {
        std::cout << "  [pretrain_lsm] nothing to train (no cities/scenarios).\n";
        return;
    }

    std::cout << "LSM pretraining: " << num_cities << " cities × " << n_seeds
              << " seeds × " << scenarios.size() << " scenarios = "
              << eps_per_seed * n_seeds << " episodes.\n"
              << "  Frozen bid policy: MAPPO <- " << cfg.policy_path << "\n";

    std::vector<std::unique_ptr<CityAssets>> assets;
    assets.reserve(num_cities);
    for (int i = 0; i < num_cities; ++i)
        assets.push_back(load_city(*train_ptrs[i], i, cfg.episode_cfg, cfg.cache_root));

    std::vector<std::unique_ptr<EpisodeRunner>> runners;
    runners.reserve(num_cities);
    for (int i = 0; i < num_cities; ++i)
        runners.push_back(std::make_unique<EpisodeRunner>(
            assets[i]->ep_cfg, assets[i]->geo_box,
            static_cast<uint32_t>(cfg.start_seed)));

    auto& lsm   = lsm_module();
    auto& mappo = mappo_policy();

    const std::string lsm_dir = cfg.output_dir + "/lsm";
    fs::create_directories(lsm_dir);
    std::ofstream csv(cfg.output_dir + "/lsm_train.csv", std::ios::trunc);
    csv << "seed,episode,city,scenario,n_ticks,n_learn,mse,"
           "alert_cells_mean,agent_alerts_mean,"
           "tasks_completed,tasks_appeared,throughput,wallclock_ms\n";

    for (int si = 0; si < n_seeds; ++si) {
        const int seed = cfg.start_seed + si;
        std::cout << "\n════ lsm seed " << (si + 1) << "/" << n_seeds
                  << " (rng=" << seed << ") ════\n";

        mappo.reinit(static_cast<uint32_t>(seed));
        if (!mappo.load(cfg.policy_path)) {
            std::cout << "  [Abort] MAPPO checkpoint load failed: "
                      << cfg.policy_path << "\n";
            return;
        }
        lsm.reinit(static_cast<uint32_t>(seed) ^ 0x4Du);

        int ep_in_seed = 0;
        for (size_t sc = 0; sc < scenarios.size(); ++sc) {
            const EpisodeScenario& scen = scenarios[sc];
            for (int ci = 0; ci < num_cities; ++ci) {
                CityAssets&    ca     = *assets[ci];
                EpisodeRunner& runner = *runners[ci];

                runner.set_global_episode(ep_in_seed);

                uint32_t ep_seed =
                      (static_cast<uint32_t>(seed)   * 2654435761u)
                    ^ (static_cast<uint32_t>(ci + 1) *      40503u)
                    ^ (static_cast<uint32_t>(sc + 1) * 2246822519u);
                ep_seed |= 1u;

                const SharedEpisodeSetup setup = build_shared_episode_setup(
                    ep_seed, *ca.config, scen, ca.ep_cfg, ca.geo_box);

                runner.train_mode  = false;
                runner.policy_mode = PolicyMode::MAPPO;
                RunResult res = runner.run(ci, num_cities, scen, ep_seed, &setup);

                const auto& st = runner.lsm_stats();
                csv << seed << ',' << ep_in_seed << ',' << ca.config->name << ','
                    << scen.label << ',' << st.n_ticks << ',' << st.n_learn << ','
                    << st.mse_mean() << ',' << st.alert_cells_mean() << ','
                    << st.agent_alerts_mean() << ','
                    << res.metrics.tasks_completed << ','
                    << res.metrics.tasks_appeared << ','
                    << res.metrics.throughput_rate << ','
                    << res.wallclock_ms << '\n';
                csv.flush();

                if (cfg.verbose)
                    std::cout << "  [lsm ep " << (ep_in_seed + 1) << "/" << eps_per_seed
                              << " s" << seed << " " << ca.config->name
                              << " " << scen.label << "]"
                              << "  mse=" << st.mse_mean()
                              << " learn=" << st.n_learn
                              << " alert_cells=" << st.alert_cells_mean()
                              << " agent_alerts=" << st.agent_alerts_mean()
                              << "  thr=" << res.metrics.throughput_rate
                              << "  " << res.wallclock_ms << "ms"
                              << "  mem=" << process_commit_mb() << "MB\n"
                              << std::flush;

                if (cfg.save_policy)
                    lsm.save(lsm_dir + "/lsm_seed" + std::to_string(seed) + ".bin");
                ++ep_in_seed;
                runner.release_episode_memory();
            }
        }
        std::cout << "  LSM seed " << seed << " done — checkpoint "
                  << lsm_dir << "/lsm_seed" << seed << ".bin\n";
    }
    std::cout << "LSM pretraining complete. Results in "
              << cfg.output_dir << "\n";
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

TrainingLogger::TrainingLogger(const std::string& output_dir,
                               const std::string& filename_tag, bool append) {
    fs::create_directories(output_dir);
    std::string path = output_dir + "/episodes_" + filename_tag + ".csv";
    std::error_code ec;
    const bool fresh = !append || !fs::exists(path, ec)
                       || fs::file_size(path, ec) == 0;
    file_.open(path, fresh ? std::ios::out : std::ios::app);
    if (!file_.is_open())
        throw std::runtime_error("TrainingLogger: cannot open " + path);
    if (fresh) write_header();
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
          // Allocation optimality vs full-scan cheapest-insertion oracle
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
