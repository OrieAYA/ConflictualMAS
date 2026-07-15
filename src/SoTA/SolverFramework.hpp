#ifndef SOLVER_FRAMEWORK_HPP
#define SOLVER_FRAMEWORK_HPP

// ===== SolverMetrics.hpp =====
#include <string>

// Unified metric record produced by every standalone SoTA solver per episode.
// Mirrors ComparisonMetrics where it makes sense so SoTA and MAPPO/Hybrid runs
// compare apples-to-apples. Throughput (completed/appeared) is the LGPDP axis;
// the rest are secondary. Inapplicable metrics stay at their default 0.
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

// Everything that must be IDENTICAL across SoTA solvers for a fair comparison.
// SolverRunner builds one per episode and passes a const ref into ISolver::init().
// It owns the shared simulation surface (road graph, congestion model, task
// stream, scenario, fleet sizing) so two solvers face byte-identical arrivals,
// ghost injection and start positions. Each ISolver owns the rest itself
// (allocation, path planning, per-agent state, replanning, internal cost model).
// Lives for the whole run(); references inside stay stable for the episode.

struct SolverContext {
    // ── World (read-only references) ────────────────────────────────────────
    const GeoBox*    geo_box    = nullptr;   // road graph (owned by caller)

    // ── Mutable shared state ────────────────────────────────────────────────
    // Solvers WRITE to congestion_map (registering their agents' paths) so the
    // BPR feedback works across the system. Ghost traffic writes here too.
    CongestionMap*        congestion_map = nullptr;
    GhostTrafficController* ghost         = nullptr;   // may be nullptr

    // ── Episode definition ──────────────────────────────────────────────────
    const EpisodeConfig* episode_config = nullptr;   // user-configured params

    // The task stream. Pre-generated deterministically from the seed so every
    // solver replays the exact same arrivals. The solver does NOT mutate this.
    std::vector<ScheduledTask> task_stream;

    // Fleet sizing for this episode (post agents_mult scaling).
    int  n_active_agents       = 0;
    int  max_capacity_per_agent = 1;

    // Initial agent positions. solvers[i] should start agent i at
    // agent_start_nodes[i].
    std::vector<osmium::object_id_type> agent_start_nodes;

    // ── Per-agent capacity heterogeneity (paper Y, optional) ────────────────
    // When non-empty, capacities[i] is the carrying capacity of agent i. Empty
    // = use max_capacity_per_agent uniformly.
    std::vector<int> per_agent_capacity;

    // ── Sim parameters (already scaled to this episode's scenario) ──────────
    int   total_steps  = 0;
    float speed_mps    = 5.f;
    int   city_index   = 0;
    int   num_cities   = 1;

    // ── RNG seed (stable for this episode) ──────────────────────────────────
    // Solvers MUST NOT call rand() / std::random_device directly. Any internal
    // randomness should be seeded from this value for reproducibility.
    uint32_t episode_seed = 42;

    // ── Helpers ─────────────────────────────────────────────────────────────
    // Sample one valid road node (uniform over the road graph). Useful for
    // solvers that need to park idle agents at non-task endpoints.
    osmium::object_id_type sample_valid_node(std::mt19937& rng) const;
};

// ===== SolverCongestion.hpp =====
#include <algorithm>
#include <cmath>
#include <tuple>

// Solver congestion footprint — replicates the DMAS pipeline's commit_plan.
// Like commit_plan, every agent publishes the occupancy of its ENTIRE remaining
// route on the shared CongestionMap (each edge registered for its free-flow
// window with weight = load_per_agent, previous registration removed on replan),
// so the BPR/throughput/latency axes are measured under the same dense,
// predictive congestion regime as the EpisodeRunner pipeline.

// One committed edge occupancy: (edge_id, t_enter, t_exit). Stored per agent so
// the previous registration can be removed before re-committing.
using CommittedOcc = std::vector<std::tuple<osmium::object_id_type, int, int>>;

// Register an agent's full remaining route on `cmap`, exactly like commit_plan:
//   1. remove the agent's previous occupancy (`committed`);
//   2. walk the ordered `stops` from `from`, leg by leg, accumulating FREE-FLOW
//      time (ceil(edge_length / speed)); add_agent each edge for [t, t+steps]
//      with weight = load_per_agent; remember it in `committed`.
// `path_fn(from, to, t)` returns the edge-id list of the planned leg path.
template <typename PathFn>
inline void commit_agent_route(
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

    // 2. Register the full remaining route with free-flow windows.
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
            const int steps = std::max(1, static_cast<int>(
                std::ceil(it->second.distance_meters / sp)));
            cmap.add_agent(eid, t, t + steps, w);
            committed.emplace_back(eid, t, t + steps);
            t += steps;
        }
        cur = target;
    }
}

// ===== SolverInstrumentation.hpp =====
#include <chrono>

// Shared helper for the SECONDARY metrics (Gini, mean/var congestion, fleet
// distance, compute time, …). Solvers own one instance, call its sample_*()
// methods, and finalise() fills the columns it owns. Headline metrics
// (throughput, latency, wait/trip, road_pd, violations) stay solver-owned.
class SolverInstrumentation {
public:
    // Per-agent completion counters (size = n_agents).
    std::vector<int> per_agent_completed;

    // Total fleet distance (sum of edge lengths traversed by ALL agents,
    // in meters).
    double total_fleet_distance_m = 0.0;

    // Congestion sampling (mean_load_now per step). Stored as
    // online-Welford for variance.
    double cong_mean = 0.0;
    double cong_m2   = 0.0;     // sum of squared deviations
    long   cong_n    = 0;
    int    peak_load = 0;

    // BPR slowdown factor (adjusted / free-flow) actually experienced along
    // executed edges; its mean over all traversals feeds mean_bpr_along_route
    // (otherwise that field keeps its default 1.0 for SoTA solvers).
    double bpr_sum = 0.0;
    long   bpr_n   = 0;

    // Ghost active sampling.
    double ghost_mean = 0.0;
    long   ghost_n    = 0;

    // Compute timing.
    std::chrono::steady_clock::time_point t_start;
    long long wallclock_ms      = 0;
    long long n_allocation_calls = 0;
    long long alloc_time_ns_sum = 0;

    // Initialise sizing — call from ISolver::init.
    void init(int n_agents) {
        per_agent_completed.assign(static_cast<size_t>(n_agents), 0);
        total_fleet_distance_m = 0.0;
        cong_mean = 0.0; cong_m2 = 0.0; cong_n = 0;
        peak_load = 0;
        bpr_sum = 0.0; bpr_n = 0;
        ghost_mean = 0.0; ghost_n = 0;
        wallclock_ms = 0;
        n_allocation_calls = 0;
        alloc_time_ns_sum = 0;
        t_start = std::chrono::steady_clock::now();
    }

    // Call this once per agent that DELIVERS a task. agent_id must be a
    // valid index into per_agent_completed.
    void record_delivery(int agent_id) {
        if (agent_id < 0 ||
            agent_id >= static_cast<int>(per_agent_completed.size())) return;
        ++per_agent_completed[agent_id];
    }

    // Call once per edge traversal: add the edge length (meters) to the
    // fleet-wide distance counter.
    void record_edge_traversal(float edge_length_m) {
        if (edge_length_m > 0.f) total_fleet_distance_m += edge_length_m;
    }

    // Record the BPR slowdown factor (adjusted / free-flow) paid on one
    // traversed edge at its entry step. Mean over traversals → mean_bpr_along_route.
    void record_edge_bpr(float factor) {
        if (factor > 0.f) { bpr_sum += static_cast<double>(factor); ++bpr_n; }
    }

    // Sample the CongestionMap state — call ONCE per simulation step, AFTER
    // congestion_map.advance(t). Uses load_sample_now() which combines mean +
    // peak in a single pass over load_ → 2× faster than the previous
    // mean_load_now() + peak_load_now() sequence.
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

    // Fill the SolverMetrics fields this helper owns. Call from
    // ISolver::finalize after the solver has filled the headline metrics.
    void finalize_into(SolverMetrics& m) const {
        const auto t_end = std::chrono::steady_clock::now();
        m.wallclock_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_start).count();

        // Gini, max, min, std of per-agent completion counts.
        if (!per_agent_completed.empty()) {
            // Only consider agents that received at least one task; the
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
                // x is sorted ascending. Range [0, 1-1/N], with 0 = perfect
                // equality, 1 = one agent does all.
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
        m.n_ghost_active_mean = static_cast<float>(ghost_mean);

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

// Shared A* path cache for SoTA solvers. Wraps graph_search::shortest_path_edges
// into a (nodes, edges, cost) triple cached per (from, to). Safe because static
// paths depend only on the shared GeoBox, not on any solver. No cost adjustment /
// congestion-aware / guide-path logic — that lives inside each solver.
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
    // Returns a const reference into the cache; valid for the lifetime of
    // this PathHelper. If no path exists, the returned SimplePath has
    // valid=false.
    const SimplePath& get(osmium::object_id_type from,
                          osmium::object_id_type to);

    // Cache size diagnostic.
    int cache_size() const { return static_cast<int>(cache_.size()); }

    // Compute the EUCLIDEAN heuristic cost between two nodes (haversine in
    // metres). Used by solvers needing a quick h-value when no path exists
    // yet or when an upper-bound estimate is enough.
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

// Abstract interface every standalone SoTA solver implements. The solver owns
// its full LGPDP pipeline (allocation, planning, replanning) and READS the
// SolverContext for the shared surface (graph, congestion, task stream, fleet).
// It must not modify the task stream or GeoBox; it MAY register its paths on the
// CongestionMap. Must be deterministic given context.episode_seed.
//
// Lifecycle (driven by SolverRunner):
//   init(ctx);
//   for t in [0, total_steps): inject_task(task, t) for arrivals; step(t);
//   metrics = finalize();
class ISolver {
public:
    virtual ~ISolver() = default;

    // ── One-time setup ──────────────────────────────────────────────────────
    // Called once at the start of an episode. Solver should allocate its
    // internal agent state, path caches, queues, etc. and store the context
    // reference if it needs it later. The context lives for the entire run.
    virtual void init(const SolverContext& ctx) = 0;

    // ── Task ingestion ──────────────────────────────────────────────────────
    // Called once for each task at its arrival_step, BEFORE step(t) runs for
    // that same step. Solver may allocate the task immediately or defer it
    // (e.g. RHCR's batch replanning).
    virtual void inject_task(const ScheduledTask& task, int step) = 0;

    // ── Simulation tick ─────────────────────────────────────────────────────
    // Called once per simulation step. Solver advances its agents along
    // their current paths, processes pickup/delivery events, may replan
    // and may register edge load on the congestion map.
    virtual void step(int timestep) = 0;

    // ── Wrap-up ─────────────────────────────────────────────────────────────
    // Called once at episode end. Solver returns its metrics; SolverRunner
    // augments them with wallclock and solver_name.
    virtual SolverMetrics finalize() = 0;

    // ── Identification ──────────────────────────────────────────────────────
    // Short solver name for logging and CSV columns. Matches the paper
    // identifier (e.g. "TokenPassing", "TrafficFlow", "RHCR").
    virtual const char* name() const = 0;
};

// ===== SolverCSVLogger.hpp =====
#include <fstream>

// ════════════════════════════════════════════════════════════════════════════
// SolverCSVLogger
// ════════════════════════════════════════════════════════════════════════════
//
// One CSV row per (solver, city, scenario, episode); the schema mirrors the
// useful ComparisonMetrics columns so SoTA and MAPPO/Hybrid runs merge for
// analysis. TRUNCATE by default; append=true extends a file (no header rewrite).
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

// ════════════════════════════════════════════════════════════════════════════
// SolverRunner
// ════════════════════════════════════════════════════════════════════════════
//
// Runs ONE episode for ONE ISolver under a FIXED shared SolverContext. Fairness:
// the task stream is pre-generated once from episode_seed and replayed to every
// solver in arrival order; ghost traffic is seeded deterministically and ticks
// per step; the CongestionMap is reset before and after each run(). Independent
// from EpisodeRunner (no shared state) — that stays the entry point for RL.
class SolverRunner {
public:
    SolverRunner(const EpisodeConfig& cfg,
                 GeoBox&              geo_box,
                 EpisodeScenario      scenario      = {},
                 uint32_t             episode_seed  = 42);

    // Run ONE episode with the given solver.
    // The runner resets its shared congestion map before the run and again
    // after, so consecutive calls with different solvers are mutually
    // isolated. Returns the solver's metrics + wallclock + name.
    //
    // When `setup` is non-null, the runner consumes its canonical task stream,
    // agent_start_nodes, per_agent_capacity, n_active_agents, total_steps and
    // ghost_seed verbatim — bypassing internal RNG-driven re-derivation. This
    // is the publication-grade path used by Option O, ensuring SoTA solvers
    // see byte-identical environment to RL policies running on the SAME setup.
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
    // and again at the start of each run() to reset task_stream + congestion
    // map. The task stream is regenerated with the SAME seed so identity is
    // preserved across run() calls.
    void prepare_episode();
};

#endif // SOLVER_FRAMEWORK_HPP