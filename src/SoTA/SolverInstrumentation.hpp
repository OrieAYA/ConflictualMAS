#ifndef SOTA_SOLVER_INSTRUMENTATION_HPP
#define SOTA_SOLVER_INSTRUMENTATION_HPP

#include "SolverMetrics.hpp"
#include "Environment/Congestion/CongestionMap.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// SolverInstrumentation
// ════════════════════════════════════════════════════════════════════════════
//
// SHARED helper used by all standalone SoTA solvers to track the secondary
// metrics (Gini, mean/var congestion, fleet distance, compute time, etc.).
// Solvers own one instance and call its sample_*() methods at the right
// points; finalise() fills the SolverMetrics fields the helper owns.
//
// We do NOT take over the headline metric fields (throughput, latency,
// wait/trip steps, road_pd, capacity/pairing violations) — those stay
// solver-owned because they need awareness of the solver's task lifecycle.
// This helper only fills the COMMON, MECHANICAL columns.

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

#endif // SOTA_SOLVER_INSTRUMENTATION_HPP