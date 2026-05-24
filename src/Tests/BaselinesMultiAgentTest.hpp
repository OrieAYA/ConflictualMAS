#ifndef BASELINES_MULTI_AGENT_TEST_HPP
#define BASELINES_MULTI_AGENT_TEST_HPP

#include <string>

// ════════════════════════════════════════════════════════════════════════════
// BaselinesMultiAgentTest
// ════════════════════════════════════════════════════════════════════════════
//
// Exercises every published / heuristic baseline (MCA, TokenPassing,
// TrafficFlow, CongestionAware, PIBT, LaCAM, DoubleHorizon, InsertionGreedy,
// Greedy) in the Lifelong General Pickup and Delivery context this project
// targets:
//
//   - Multi-agent (no predefined depot — agents start on arbitrary road nodes
//     and stay where they delivered).
//   - Capacitated (max_tasks_per_agent > 1 so multi-package carry is possible,
//     plus one heterogeneous-capacity sweep).
//   - Lifelong (tasks arrive throughout the episode; episode never "ends" by
//     emptying the task list).
//   - Congestion-aware environment: GhostTrafficController active so the
//     dynamic-cost baselines (CongestionAware, TrafficFlow) see real traffic.
//
// For each baseline we verify:
//   1. No crash / no exception during construction or run.
//   2. accept_rate ∈ [0, 1] and finite.
//   3. tasks_appeared > 0 (the scenario is not vacuous).
//   4. capacity_violations_runtime == 0 (no agent overshoots its cap).
//   5. pairing_violations_runtime  == 0 (pickup always before delivery).
//   6. throughput_rate ∈ [0, 1] and finite.
//   7. Distinct baselines produce non-identical (allocation, throughput)
//      summaries on the SAME deterministic episode seed — sanity check that
//      our adaptations are actually different policies, not the same code path.
//
// Designed to run in well under a minute on the cached smoke_test bbox.
// Returns true on success (all soft checks passed), false otherwise.
bool run_baselines_multi_agent_test(const std::string& osm_file,
                                    const std::string& cache_dir);

#endif // BASELINES_MULTI_AGENT_TEST_HPP