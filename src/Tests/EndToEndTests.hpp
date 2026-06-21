#ifndef TESTS_END_TO_END_TESTS_HPP
#define TESTS_END_TO_END_TESTS_HPP

#include <string>

// Battery D — End-to-end & integration.
//
//   1. Functional training: full train + PPO update + episode reset + eval
//      (reuses the training smoke test).
//   2. Episode homogeneity: the SAME (seed, scenario) under two different
//      methods yields the byte-identical task stream — so any cross-method gap
//      is a method effect, not an environment confound.
//   3. Refresh between methods: task IDs / memory restart fresh per episode,
//      while the generated map (GeoBox) is left untouched across methods.
//
// Returns true (PASS) / false (FAIL).
bool run_end_to_end_tests(const std::string& osm_file, const std::string& cache_dir);

#endif // TESTS_END_TO_END_TESTS_HPP
