#ifndef TESTS_MECHANICS_TESTS_HPP
#define TESTS_MECHANICS_TESTS_HPP

#include <string>

// Battery B — Mechanics & Simulation.
//
// Verifies the moving parts of one episode:
//   1. Congestion → BPR edge weight → other agents' travel time
//      (pure CongestionMap: monotonic, formula-exact, self-exclusion).
//   2. Lifelong task creation at dispersed arrival steps (EpisodeGenerator).
//   3. Time progression (event stream), agent movement node→node along edges,
//      and task completion (pickup THEN delivery, then off the active lists).
//   4. Ghost congestion measurably raises network load at the system level
//      (same seed, ghosts off vs on) — the end-to-end congestion loop.
//   5. BPR is time-dependent: a windowed load only costs inside its window.
//
// Returns true (PASS) / false (FAIL).
bool run_mechanics_tests(const std::string& osm_file, const std::string& cache_dir);

#endif // TESTS_MECHANICS_TESTS_HPP
