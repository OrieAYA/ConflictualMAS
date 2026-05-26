#ifndef CONGESTION_ONLY_TEST_HPP
#define CONGESTION_ONLY_TEST_HPP

#include <string>

// ════════════════════════════════════════════════════════════════════════════
// CongestionOnlyTest
// ════════════════════════════════════════════════════════════════════════════
//
// Measures the REAL impact of the ghost-traffic congestion injection on edge
// traversal costs, independent of any delivery agent / TAM / policy machinery.
//
// For every (city, scenario) pair used by Option Y evaluation, runs ONLY the
// GhostTrafficController + CongestionMap simulation over a full episode
// (3600 steps), sampling the load distribution and computing the BPR cost
// multiplier the system would apply at the observed peak load.
//
// Per-city × scenario output:
//   - n_max (ghost peak count, scaled per-city like customize_episode_for_city)
//   - n_hot_ways (size of the spawn pool)
//   - peak_load_episode (max load any edge reached at any step)
//   - mean_load_episode (network-wide time-average)
//   - n_edges_load_ge_2 / _5 / _10 at peak step (overlap / jam / heavy jam)
//   - BPR factor at peak (worst-case cost multiplier on a 100m way)
//   - wallclock per scenario (basis for Y wallclock extrapolation)
//
// What this test answers:
//   1. Do ghosts actually CONGEST anything, or are they too sparse to matter?
//   2. Does each profile shape (Wave/Ramp/Shock/Build) produce its expected
//      temporal signature?
//   3. Is the wallclock budget realistic for the full Y eval?
//
// Cities tested: 6 (same as Y train + held-out, London dropped).
// Scenarios: 5 (Wave / Ramp / Shock / Build / over_fleet=Wave with same ghosts).
//
// Returns true if no exception was raised. Diagnostic information is printed
// regardless of "pass/fail" — the test's purpose is measurement, not assertion.
bool run_congestion_only_test(const std::string& osm_root,
                              const std::string& cache_root);

#endif // CONGESTION_ONLY_TEST_HPP