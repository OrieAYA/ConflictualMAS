#include "SolverRunner.hpp"
#include "Training/SharedEpisodeSetup.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

// ── SolverContext helper ─────────────────────────────────────────────────────

osmium::object_id_type SolverContext::sample_valid_node(std::mt19937& rng) const {
    if (!geo_box || geo_box->data.nodes.empty()) return 0;
    // Collect any node that is incident to at least one way (= traversable).
    // Cheap because called rarely (only by solvers that need parking nodes).
    std::vector<osmium::object_id_type> candidates;
    candidates.reserve(geo_box->data.nodes.size() / 4);
    for (const auto& [id, pt] : geo_box->data.nodes) {
        if (!pt.incident_ways.empty()) candidates.push_back(id);
    }
    if (candidates.empty()) return 0;
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    return candidates[dist(rng)];
}

// ── SolverRunner ─────────────────────────────────────────────────────────────

SolverRunner::SolverRunner(const EpisodeConfig& cfg,
                           GeoBox&              geo_box,
                           Pathfinder&          pathfinder,
                           EpisodeScenario      scenario,
                           uint32_t             episode_seed)
    : cfg_(cfg),
      scenario_(scenario),
      episode_seed_(episode_seed),
      gen_(cfg, geo_box, episode_seed)
{
    // Initial context wiring; task_stream is populated by prepare_episode().
    ctx_.geo_box        = &geo_box;
    ctx_.pathfinder     = &pathfinder;
    ctx_.congestion_map = &congestion_map_;
    ctx_.episode_config = &cfg_;
    ctx_.episode_seed   = episode_seed_;
    ctx_.speed_mps      = cfg_.speed_mps;

    // Configure the shared CongestionMap to match EpisodeConfig (so BPR
    // behaviour matches the legacy pipeline).
    congestion_map_.params.load_per_agent =
        std::max(1, cfg_.fleet_load_per_agent);

    // Optional ghost-traffic controller. Created when enabled so the
    // SolverContext exposes it to solvers that want to react to it (most
    // solvers just observe its effect via the CongestionMap and don't need
    // direct access).
    if (cfg_.enable_ghost_traffic) {
        ghost_ = std::make_unique<GhostTrafficController>();
        ctx_.ghost = ghost_.get();
    }

    prepare_episode();
}

void SolverRunner::prepare_episode() {
    // Reset the task generator's RNG so every prepare_episode() call yields
    // the SAME task stream (fair comparison guarantee).
    gen_.reset_seed(episode_seed_);

    // Apply scenario scaling — density_mult multiplies the lambda of each
    // phase; agents_mult scales the active agent count. We mutate a LOCAL
    // copy of the config (gen_ holds a const ref to cfg_, but generate()
    // reads phase data freshly). For now we apply the scaling by rebuilding
    // the task stream and scaling timestamps would be invasive; instead we
    // generate at lambda × density_mult by transparently multiplying the
    // number of tasks accepted from generator output. This is a placeholder
    // for the Phase-1 generator integration; if you need scenario_mult
    // support, extend EpisodeGenerator to accept a scale factor.
    ctx_.task_stream = gen_.generate();

    // Apply density_mult: drop or duplicate tasks proportionally. Simple
    // deterministic stride: keep ceil(N × density_mult) tasks via random
    // sampling (with replacement when >1.0).
    if (scenario_.density_mult != 1.0f && !ctx_.task_stream.empty()) {
        std::mt19937 rng(episode_seed_ ^ 0xA17EFEEDu);
        std::vector<ScheduledTask> scaled;
        const int target = std::max<int>(
            1, static_cast<int>(std::round(ctx_.task_stream.size() *
                                            scenario_.density_mult)));
        if (scenario_.density_mult <= 1.0f) {
            // Subsample — keep tasks with probability density_mult.
            std::shuffle(ctx_.task_stream.begin(), ctx_.task_stream.end(), rng);
            ctx_.task_stream.resize(static_cast<size_t>(target));
            std::sort(ctx_.task_stream.begin(), ctx_.task_stream.end(),
                      [](const ScheduledTask& a, const ScheduledTask& b) {
                          return a.arrival_step < b.arrival_step;
                      });
        } else {
            // Supersample by uniform draws (duplicates pickup/delivery pairs
            // at fresh random arrival steps within the same range).
            scaled.reserve(static_cast<size_t>(target));
            const int max_step = ctx_.task_stream.back().arrival_step + 1;
            std::uniform_int_distribution<size_t> pick(
                0, ctx_.task_stream.size() - 1);
            std::uniform_int_distribution<int> step_pick(0, std::max(1, max_step - 1));
            for (int i = 0; i < target; ++i) {
                ScheduledTask t = ctx_.task_stream[pick(rng)];
                t.arrival_step = step_pick(rng);
                scaled.push_back(t);
            }
            std::sort(scaled.begin(), scaled.end(),
                      [](const ScheduledTask& a, const ScheduledTask& b) {
                          return a.arrival_step < b.arrival_step;
                      });
            ctx_.task_stream = std::move(scaled);
        }
    }

    // Fleet sizing — pull the maximum requested agent count from the phase
    // table, then apply scenario.agents_mult.
    const auto phases = gen_.build_phase_table();
    int peak_agents = 1;
    int total_steps = 0;
    for (const auto& p : phases) {
        peak_agents = std::max(peak_agents,
                               std::max(p.n_agents_start, p.n_agents_end));
        total_steps = std::max(total_steps, p.step_end);
    }
    ctx_.n_active_agents = std::max(
        1, static_cast<int>(std::round(peak_agents * scenario_.agents_mult)));
    ctx_.total_steps  = total_steps;

    // Capacity (homogeneous by default).
    ctx_.max_capacity_per_agent = std::max(1, cfg_.max_tasks_per_agent);

    // Heterogeneous capacity — sample per-agent capacities when enabled.
    ctx_.per_agent_capacity.clear();
    if (cfg_.enable_heterogeneous_capacity) {
        std::mt19937 rng(episode_seed_ ^ 0xC0FFEE01u);
        std::uniform_int_distribution<int> dist(
            cfg_.hetero_capacity_min, cfg_.hetero_capacity_max);
        ctx_.per_agent_capacity.resize(static_cast<size_t>(ctx_.n_active_agents));
        for (int i = 0; i < ctx_.n_active_agents; ++i)
            ctx_.per_agent_capacity[i] = dist(rng);
    }

    // Initial agent positions — deterministic from the seed. Same agents
    // start at the same nodes across solvers for fair comparison.
    {
        std::mt19937 rng(episode_seed_ ^ 0xA9E47514u);
        ctx_.agent_start_nodes.clear();
        ctx_.agent_start_nodes.reserve(static_cast<size_t>(ctx_.n_active_agents));
        for (int i = 0; i < ctx_.n_active_agents; ++i)
            ctx_.agent_start_nodes.push_back(ctx_.sample_valid_node(rng));
    }
}

SolverMetrics SolverRunner::run(ISolver& solver,
                                 const SharedEpisodeSetup* setup) {
    const auto t0 = std::chrono::steady_clock::now();

    // Reset the shared CongestionMap so each solver starts clean.
    congestion_map_.reset();

    if (setup) {
        // ── Publication-grade path: consume the canonical SharedEpisodeSetup
        // verbatim. RL and SoTA paths see byte-identical environment.
        ctx_.task_stream            = setup->task_stream;
        ctx_.agent_start_nodes      = setup->agent_start_nodes;
        ctx_.per_agent_capacity     = setup->per_agent_capacity;
        ctx_.n_active_agents        = setup->n_active_agents;
        ctx_.total_steps            = setup->total_steps;
        ctx_.max_capacity_per_agent = setup->max_capacity_per_agent;
        ctx_.episode_seed           = setup->ep_seed;
    } else {
        // Legacy path: regenerate task_stream + sample positions + sample
        // capacities from the internal RNG seeded at construction.
        prepare_episode();
    }

    if (ghost_) {
        GhostTrafficController::Config gconf;
        gconf.n_max              = cfg_.ghost_n_max;
        gconf.total_steps        = ctx_.total_steps;
        gconf.window_steps       = cfg_.ghost_window_steps;
        gconf.hot_way_fraction   = cfg_.ghost_hot_way_frac;
        gconf.hot_way_count      = cfg_.ghost_hot_way_count;
        gconf.density_per_hot_way = cfg_.ghost_density_per_hot_way;
        gconf.load_per_ghost     = std::max(1, cfg_.ghost_load_per_unit);
        gconf.profile            = scenario_.congestion_profile;
        const uint32_t gseed = setup ? setup->ghost_seed
                                      : (episode_seed_ ^ 0xBADCAFEEu);
        ghost_->reset(*ctx_.geo_box, congestion_map_, gconf, gseed);
    }

    // Solver lifecycle: init → (per-step ingest + step) → finalize.
    solver.init(ctx_);

    // Index the task stream by arrival_step for O(1) per-step lookup.
    // The stream is already sorted by arrival_step.
    size_t next_task = 0;
    for (int t = 0; t < ctx_.total_steps; ++t) {
        // Advance ghost traffic AFTER the CongestionMap's own clock so
        // solvers querying load see ghost contributions for step t.
        congestion_map_.advance(t);
        if (ghost_) ghost_->step(t);

        // Inject all tasks arriving at step t.
        while (next_task < ctx_.task_stream.size() &&
               ctx_.task_stream[next_task].arrival_step <= t) {
            solver.inject_task(ctx_.task_stream[next_task], t);
            ++next_task;
        }

        solver.step(t);
    }

    SolverMetrics m = solver.finalize();
    m.solver_name      = solver.name();
    m.total_steps      = ctx_.total_steps;
    m.n_active_agents  = ctx_.n_active_agents;
    m.scenario_label   = scenario_.label ? scenario_.label : "";
    // city_label and episode are set by the caller (main option S) before
    // logging — the runner doesn't know the city by name.

    const auto t1 = std::chrono::steady_clock::now();
    m.wallclock_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Clean up shared state so subsequent run() calls start fresh.
    congestion_map_.reset();
    if (ghost_) ghost_->purge();

    return m;
}