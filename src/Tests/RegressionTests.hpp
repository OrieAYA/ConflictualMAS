#ifndef TESTS_REGRESSION_TESTS_HPP
#define TESTS_REGRESSION_TESTS_HPP

#include <string>

// Battery R — Regression (golden values).
//
// Pins the EXACT deterministic behaviour of the architecture so any unintended
// drift (simulation, structures, workflows) fails the test:
//   1. Structure   — smoke GeoBox node/way counts.
//   2. Scenario grid — make_scenario_grid() size + first/last combos.
//   3. RL pipeline — TamAlwaysAccept & RMCA episodes (fixed seed) → golden metrics
//                    + same-build determinism (run twice, identical).
//   4. SoTA pipeline — CA & HAPC via SolverRunner (fixed seed) → golden metrics.
//
// Returns true (PASS) / false (FAIL).
bool run_regression_tests(const std::string& osm_file, const std::string& cache_dir);

#endif // TESTS_REGRESSION_TESTS_HPP
