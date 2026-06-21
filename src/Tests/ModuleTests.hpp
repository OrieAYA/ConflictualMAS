#ifndef TESTS_MODULE_TESTS_HPP
#define TESTS_MODULE_TESTS_HPP

#include <string>

// Battery C — Modules (architectural / unit).
//
// Exercises each architectural module on its own:
//   1. GlobalMemory  — task add/get, lifecycle list transitions, ID refresh.
//   2. BPR           — capacity model + monotone congestion factor.
//   3. TAM           — multi-candidate allocation correctness (TamMcTest).
//   4. Planning      — DbVNS / MCA / DoubleHorizon run with 0 PDP violations.
//   5. Scoring       — RMCA baseline produces valid metrics.
//   6. Policies      — MAPPO / IPPO / MAPPER / Hybrid run an eval episode.
//
// Returns true (PASS) / false (FAIL).
bool run_module_tests(const std::string& osm_file, const std::string& cache_dir);

#endif // TESTS_MODULE_TESTS_HPP
