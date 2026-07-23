#ifndef SOLVER_FRAMEWORK_HPP
#define SOLVER_FRAMEWORK_HPP

// ===== SolverMetrics.hpp =====
#include <string>

// one metric record per (solver, episode)
struct SolverMetrics {
    // ── Identity / context (filled by SolverRunner) ─────────────────────────
    std::string solver_name;
    std::string city_label;
    std::string scenario_label;
    int         episode      = 0;
    int         total_steps  = 0;
    int         n_active_agents = 0;

    // ── Headline LGPDP metrics ──────────────────────────────────────────────
    int    tasks_appeared  = 0;
    int    tasks_completed = 0;
    int    tasks_refused   = 0;
    float  throughput_rate = 0.f;
    float  accept_rate     = 0.f;

    // ── Time metrics (mean over completed tasks) ────────────────────────────
    double latency_mean       = 0.;
    double latency_per_agent  = 0.;
    double mean_wait_steps    = 0.;
    double mean_trip_steps    = 0.;
    double mean_road_pd_m     = 0.;

    // ── Routing diagnostics ─────────────────────────────────────────────────
    double mean_extra_steps_per_task     = 0.;
    double delivery_route_efficiency     = 0.;
    double total_fleet_distance_m        = 0.;

    // ── Agent metrics ───────────────────────────────────────────────────────
    float  agent_utilisation             = 0.f;
    int    agent_completed_max           = 0;
    int    agent_completed_min           = 0;
    float  agent_completed_gini          = 0.f;
    float  agent_completed_std           = 0.f;   // coefficient of variation

    // ── Network-level congestion ───────────────────────────────────────────
    float  mean_congestion               = 0.f;   // mean(load_now) over all steps
    float  congestion_variance           = 0.f;
    int    peak_load                     = 0;     // max load_now across episode
    float  n_ghost_active_mean           = 0.f;   // SolverRunner fills this

    // ── Congestion exposure (along agent traversals) ────────────────────────
    float  mean_congestion_at_decision   = 0.f;
    double mean_bpr_along_route          = 1.;
    double time_lost_to_congestion       = 0.;
    int    n_traversals_in_jam           = 0;
    float  route_congestion_exposure     = 0.f;

    // ── Compute complexity ──────────────────────────────────────────────────
    long long wallclock_ms               = 0;
    long long n_allocation_calls         = 0;     // # try_allocate / batch calls
    double    compute_time_per_task_ms   = 0.;
    double    compute_time_per_decision_us = 0.;

    // ── Solver-specific bookkeeping ─────────────────────────────────────────
    int    n_replans                     = 0;
    int    n_collisions_avoided          = 0;

    // ── Validity audit — must remain zero ───────────────────────────────────
    int    capacity_violations           = 0;
    int    pairing_violations            = 0;
};

// ===== SolverContext.hpp =====
#include "Environment/GeoBox/Box.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "Environment/Simulation/GhostTrafficController.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "Environment/Structure/Episode.hpp"
#include <cstdint>
#include <osmium/osm/types.hpp>
#include <vector>

// shared surface: identical across every compared method

struct SolverContext {
    // ── World (read-only references) ────────────────────────────────────────
    const GeoBox*    geo_box    = nullptr;   // road graph (owned by caller)

    // ── Mutable shared state ────────────────�
    CongestionMap*        congestion_map = nullptr;
    GhostTrafficController* ghost         = nullptr;   // may be nullptr

    // ── Episode definition ──────────────────────────────────────────────────
    const EpisodeConfig* episode_config = nullptr;   // user-configured params

    // deterministic task stream, replayed identically by every solver
    // solver replays the exact same arrivals. The solver does NOT mutate this.
    std::vector<ScheduledTask> task_stream;

    // Fleet sizing for this episode (post agents_mult scaling).
    int  n_active_agents       = 0;
    int  max_capacity_per_agent = 1;

    // initial agent positions
    // agent_start_nodes[i].
    std::vector<osmium::object_id_type> agent_start_nodes;

    // ── Per-agent capacity heterogeneity (paper Y, optional) ──────
    std::vector<int> per_agent_capacity;

    // ── Sim parameters (already scaled to this episode's scenario) ──────────
    int   total_steps  = 0;
    float speed_mps    = 5.f;
    int   city_index   = 0;
    int   num_cities   = 1;

    // ── RNG seed (stable for this episode) ────────────
    uint32_t episode_seed = 42;

    // ── Helpers ─────────────────────
    osmium::object_id_type sample_valid_node(std::mt19937& rng) const;
};

// ===== SolverCongestion.hpp =====
#include <algorithm>
#include <cmath>
#include <tuple>

// congestion footprint, mirrors commit_plan

// committed edge occupancy (edge, t_enter, t_exit), per agent
// the previous registration can be removed before re-committing.
using CommittedOcc = std::vector<std::tuple<osmium::object_id_type, int, int>>;

// republish the remaining route with BPR windows; returns the plan ETA
template <typename PathFn>
inline int commit_agent_route(
    CongestionMap& cmap, const GeoBox& geo_box, float speed_mps,
    osmium::object_id_type from,
    const std::vector<osmium::object_id_type>& stops,
    int start_step, CommittedOcc& committed, PathFn&& path_fn)
{
    const int w = std::max(1, cmap.params.load_per_agent);

    // 1. Unregister the previous footprint.
    for (const auto& occ : committed)
        cmap.remove_agent(std::get<0>(occ), std::get<1>(occ), std::get<2>(occ), w);
    committed.clear();

    // 2. Register the full remaining route with BPR-adjusted windows.
    const auto& ways = geo_box.data.ways;
    const float sp   = std::max(0.1f, speed_mps);
    osmium::object_id_type cur = from;
    int t = start_step;
    for (osmium::object_id_type target : stops) {
        if (target == 0)      continue;
        if (target == cur)    { cur = target; continue; }
        const auto edges = path_fn(cur, target, t);
        for (osmium::object_id_type eid : edges) {
            auto it = ways.find(eid);
            if (it == ways.end()) continue;
            const int steps =
                cmap.traversal_steps(eid, it->second.distance_meters, t, sp, 0);
            cmap.add_agent(eid, t, t + steps, w);
            committed.emplace_back(eid, t, t + steps);
            t += steps;
        }
        cur = target;
    }
    return t;
}

// ===== SolverInstrumentation.hpp =====
#include <chrono>

// secondary metrics: Gini, congestion, fleet distance, compute time
class SolverInstrumentation {
public:
    // Per-agent completion counters (size = n_agents).
    std::vector<int> per_agent_completed;

    // metres actually driven by the fleet
    // in meters).
    double total_fleet_distance_m = 0.0;

    // per-step congestion, Welford
    // online-Welford for variance.
    double cong_mean = 0.0;
    double cong_m2   = 0.0;     // sum of squared deviations
    long   cong_n    = 0;
    int    peak_load = 0;

    // BPR slowdown actually paid on traversed edges
    double bpr_sum = 0.0;
    long   bpr_n   = 0;

    // congestion exposure, same definitions as EpisodeRunner
    double time_lost_steps     = 0.0;
    int    n_traversals_in_jam = 0;
    double exposure_sum        = 0.0;
    long   exposure_n          = 0;

    // speed, for the ideal trip time
    // used by delivery_route_efficiency / mean_extra_steps_per_task.
    float  speed_mps = 5.f;

    // Ghost active sampling.
    double ghost_mean = 0.0;
    long   ghost_n    = 0;

    // Compute timing.
    std::chrono::steady_clock::time_point t_start;
    long long wallclock_ms      = 0;
    long long n_allocation_calls = 0;
    long long alloc_time_ns_sum = 0;

    // Initialise sizing — call from ISolver::init.
    void init(int n_agents, float speed = 5.f) {
        per_agent_completed.assign(static_cast<size_t>(n_agents), 0);
        total_fleet_distance_m = 0.0;
        cong_mean = 0.0; cong_m2 = 0.0; cong_n = 0;
        peak_load = 0;
        bpr_sum = 0.0; bpr_n = 0;
        time_lost_steps = 0.0; n_traversals_in_jam = 0;
        exposure_sum = 0.0; exposure_n = 0;
        speed_mps = (speed > 0.f) ? speed : 5.f;
        ghost_mean = 0.0; ghost_n = 0;
        wallclock_ms = 0;
        n_allocation_calls = 0;
        alloc_time_ns_sum = 0;
        t_start = std::chrono::steady_clock::now();
    }

    // one call per delivery
    // valid index into per_agent_completed.
    void record_delivery(int agent_id) {
        if (agent_id < 0 ||
            agent_id >= static_cast<int>(per_agent_completed.size())) return;
        ++per_agent_completed[agent_id];
    }

    // one call per edge traversal
    // fleet-wide distance counter.
    void record_edge_traversal(float edge_length_m) {
        if (edge_length_m > 0.f) total_fleet_distance_m += edge_length_m;
    }

    // one call per edge entry: free-flow vs adjusted time, load at entry
    void record_edge_entry(float base_time, float eff_time, int load_at_entry) {
        if (base_time > 0.f) {
            bpr_sum += static_cast<double>(eff_time / base_time);
            ++bpr_n;
            time_lost_steps += static_cast<double>(std::max(0.f, eff_time - base_time));
        }
        if (load_at_entry >= 5) ++n_traversals_in_jam;
    }

    // one call per in-transit agent per step
    // per step per in-transit agent (RL analogue: Runner.cpp route_exposure).
    void sample_route_exposure(int load_on_edge) {
        exposure_sum += static_cast<double>(load_on_edge);
        ++exposure_n;
    }

    // Sample the CongestionMap state — call ONCE per simulation step, AFTER
    void sample_congestion(const CongestionMap* cmap) {
        if (!cmap) return;
        const auto s = cmap->load_sample_now();
        if (s.peak > peak_load) peak_load = s.peak;
        // Welford online update.
        ++cong_n;
        const double dx = static_cast<double>(s.mean) - cong_mean;
        cong_mean += dx / cong_n;
        cong_m2   += dx * (static_cast<double>(s.mean) - cong_mean);
    }

    // Sample ghost traffic — call once per step.
    void sample_ghost(int ghost_active_now) {
        ++ghost_n;
        ghost_mean += (static_cast<double>(ghost_active_now) - ghost_mean) / ghost_n;
    }

    // Bracket an allocation call. Records both wallclock and counts.
    template <typename F>
    auto time_allocation(F&& f) -> decltype(f()) {
        const auto t0 = std::chrono::steady_clock::now();
        if constexpr (std::is_same_v<decltype(f()), void>) {
            f();
            const auto t1 = std::chrono::steady_clock::now();
            alloc_time_ns_sum +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            ++n_allocation_calls;
        } else {
            const auto r = f();
            const auto t1 = std::chrono::steady_clock::now();
            alloc_time_ns_sum +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            ++n_allocation_calls;
            return r;
        }
    }

    // Manual counter increment (when can't wrap a lambda).
    void incr_allocation_calls(long long ns) {
        alloc_time_ns_sum += ns;
        ++n_allocation_calls;
    }

    // fill the fields this helper owns; call after the headline metrics
    // ISolver::finalize after the solver has filled the headline metrics.
    void finalize_into(SolverMetrics& m) const {
        const auto t_end = std::chrono::steady_clock::now();
        m.wallclock_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_start).count();

        // Gini, max, min, std of per-agent completion counts.
        if (!per_agent_completed.empty()) {
            // active agents only
            // Gini of all-zero agents is degenerate.
            std::vector<int> active;
            active.reserve(per_agent_completed.size());
            for (int v : per_agent_completed)
                if (v > 0) active.push_back(v);

            if (!active.empty()) {
                std::sort(active.begin(), active.end());
                m.agent_completed_min = active.front();
                m.agent_completed_max = active.back();

                // Gini = 1 - (Σ (2k - N - 1) × x_k) / (N × Σ x_k) where
                const double N = static_cast<double>(active.size());
                double sum = 0.0, weighted = 0.0;
                for (size_t k = 0; k < active.size(); ++k) {
                    sum      += active[k];
                    weighted += static_cast<double>(active[k]) *
                                (2.0 * (k + 1) - N - 1.0);
                }
                m.agent_completed_gini = (sum > 0)
                    ? static_cast<float>(weighted / (N * sum))
                    : 0.f;

                // Coefficient of variation (std / mean).
                const double mean = sum / N;
                double var = 0.0;
                for (int v : active) {
                    const double d = v - mean;
                    var += d * d;
                }
                var /= N;
                m.agent_completed_std = (mean > 0)
                    ? static_cast<float>(std::sqrt(var) / mean)
                    : 0.f;
            }
        }

        m.total_fleet_distance_m = total_fleet_distance_m;

        m.mean_congestion = static_cast<float>(cong_mean);
        m.congestion_variance = (cong_n > 1)
            ? static_cast<float>(cong_m2 / (cong_n - 1))
            : 0.f;
        m.peak_load = peak_load;
        if (bpr_n > 0)
            m.mean_bpr_along_route = bpr_sum / static_cast<double>(bpr_n);
        m.time_lost_to_congestion   = time_lost_steps;
        m.n_traversals_in_jam       = n_traversals_in_jam;
        m.route_congestion_exposure = (exposure_n > 0)
            ? static_cast<float>(exposure_sum / exposure_n) : 0.f;
        m.n_ghost_active_mean = static_cast<float>(ghost_mean);

        // route efficiency, same formula as the RL pipeline
        const double ideal_steps = (speed_mps > 0.f && m.mean_road_pd_m > 0.)
            ? m.mean_road_pd_m / speed_mps : 0.;
        if (m.mean_trip_steps > 0. && ideal_steps > 0.) {
            m.mean_extra_steps_per_task =
                std::max(0.0, m.mean_trip_steps - ideal_steps);
            m.delivery_route_efficiency =
                std::min(1.0, ideal_steps / m.mean_trip_steps);
        }

        m.n_allocation_calls = n_allocation_calls;
        if (m.tasks_appeared > 0) {
            m.compute_time_per_task_ms =
                static_cast<double>(m.wallclock_ms) / m.tasks_appeared;
        }
        if (n_allocation_calls > 0) {
            m.compute_time_per_decision_us =
                static_cast<double>(alloc_time_ns_sum) /
                (1000.0 * n_allocation_calls);
        }
    }
};

// ===== PathHelper.hpp =====
#include <unordered_map>
#include <utility>

// shared static A* path cache
struct SimplePath {
    std::vector<osmium::object_id_type> nodes;     // nodes[0] = from, nodes.back() = to
    std::vector<osmium::object_id_type> edges;     // edges[i] connects nodes[i] and nodes[i+1]
    float                                cost = 0.f; // total length in metres
    bool                                 valid = false;
};

class PathHelper {
public:
    explicit PathHelper(const GeoBox& gb);

    // Get (or compute and cache) the shortest path from `from` to `to`.
    const SimplePath& get(osmium::object_id_type from,
                          osmium::object_id_type to);

    // Cache size diagnostic.
    int cache_size() const { return static_cast<int>(cache_.size()); }

    // haversine heuristic, metres
    float heuristic(osmium::object_id_type from,
                    osmium::object_id_type to) const;

private:
    const GeoBox& gb_;

    using Key = std::pair<osmium::object_id_type, osmium::object_id_type>;
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            const auto a = std::hash<osmium::object_id_type>{}(k.first);
            const auto b = std::hash<osmium::object_id_type>{}(k.second);
            return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
        }
    };

    std::unordered_map<Key, SimplePath, KeyHash> cache_;
    SimplePath                                    invalid_;  // returned when path fails
};

// ===== ISolver.hpp =====
#include "Environment/Structure/Episode.hpp"   // ScheduledTask

// interface for standalone SoTA solvers
class ISolver {
public:
    virtual ~ISolver() = default;

    // ── One-time setup ──────────────────�
    virtual void init(const SolverContext& ctx) = 0;

    // ── Task ingestion ──────────────────�
    virtual void inject_task(const ScheduledTask& task, int step) = 0;

    // ── Simulation tick ──────────────────�
    virtual void step(int timestep) = 0;

    // ── Wrap-up ─────────────────────
    virtual SolverMetrics finalize() = 0;

    // ── Identification ──────────────────�
    virtual const char* name() const = 0;
};

// ===== SolverCSVLogger.hpp =====
#include <fstream>

// ══════════════════════════
class SolverCSVLogger {
public:
    SolverCSVLogger(const std::string& path, bool append = false);
    ~SolverCSVLogger();

    // Writes the CSV header. Idempotent — only writes if the file is empty.
    void write_header();

    // Writes one row from a SolverMetrics record. Fields not set by the
    // solver are written as zero.
    void write_row(const SolverMetrics& m);

    // Flushes and closes the underlying stream.
    void close();

    bool is_open() const { return out_.is_open(); }

private:
    std::ofstream out_;
    bool          header_written_ = false;
};

// ===== SolverRunner.hpp =====
#include "TrainingEvaluation/Run/Runner.hpp"   // EpisodeScenario, PolicyMode
#include <memory>

// ══════════════════════════
class SolverRunner {
public:
    SolverRunner(const EpisodeConfig& cfg,
                 GeoBox&              geo_box,
                 EpisodeScenario      scenario      = {},
                 uint32_t             episode_seed  = 42);

    // Run ONE episode with the given solver.
    SolverMetrics run(ISolver& solver,
                       const struct SharedEpisodeSetup* setup = nullptr);

    // Pre-built shared context. Exposed for tests that want to inspect the
    // pre-generated task stream or share the context across multiple runs.
    const SolverContext& context() const { return ctx_; }

private:
    const EpisodeConfig& cfg_;
    EpisodeScenario      scenario_;
    uint32_t             episode_seed_;

    // Owned shared state — built once at SolverRunner construction.
    CongestionMap                            congestion_map_;
    std::unique_ptr<GhostTrafficController>  ghost_;
    EpisodeGenerator                         gen_;
    SolverContext                            ctx_;

    // Build the SolverContext for this episode. Called from the constructor
    void prepare_episode();
};

#endif // SOLVER_FRAMEWORK_HPP