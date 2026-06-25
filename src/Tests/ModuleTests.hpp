#ifndef TESTS_MODULE_TESTS_HPP
#define TESTS_MODULE_TESTS_HPP

#include <string>

// Battery C — Modules (architectural / unit).
//
// Exercises each architectural module on its own:
//   1. GlobalMemory  — task add/get, lifecycle list transitions, ID refresh.
//   2. BPR           — capacity model + monotone congestion factor.
//   3. TAM           — multi-candidate allocation correctness (TamMcTest).
//   4. Planning      — DbVNS / MCA / DoubleHorizon route validly AND deliver.
//   5. Scoring       — RMCA delivers with 0 pairing/capacity violations.
//   6. Policies      — MAPPO / IPPO / MAPPER / Hybrid produce valid sims.
//   7. Determinism   — same (mode, episode_seed) ⇒ identical metrics.
//   8. SoTA solvers  — CA + HAPC run the full SolverRunner pipeline.
//   9. Scenario grid — make_scenario_grid() = 9 deterministic combos.
//  10. TD-A*        — time-dependent routing: jamming the path slows/reroutes.
//  11. TAM retrieval — candidate agents found for a task's objective nodes;
//      node-event invariant (current objective in the node chain at t*).
//
// Every episode-driven check also asserts pickup-before-delivery and capacity
// are never violated at runtime. Returns true (PASS) / false (FAIL).
bool run_module_tests(const std::string& osm_file, const std::string& cache_dir);

#endif // TESTS_MODULE_TESTS_HPP
