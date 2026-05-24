#ifndef TAM_MC_TEST_HPP
#define TAM_MC_TEST_HPP

#include <string>

// TAM multi-candidate mode correctness test.
//
// Verifies:
//   1. Format A (force_assign=true) + TamAlwaysAccept  → accept_rate ~= 1.0
//      The critical check: with the max_search_cost_ fix (initial budget = inf),
//      the search always finds at least one candidate; force_assign guarantees
//      allocation → every task must be assigned.
//   2. Format B (force_assign=false) + TamAlwaysAccept → accept_rate ~= 1.0
//      TamAlwaysAccept always scores 1.0 (>= 0.5 threshold), so no task is
//      ever deferred.
//   3. Format A + MAPPO (untrained)                    → accept_rate ~= 1.0
//      force_assign overrides score; even untrained weights must not prevent
//      allocation in Format A.
//   4. Legacy TAM (mc=false) + TamAlwaysAccept         → accept_rate for reference
//      Printed for comparison; not checked (legacy is the trusted baseline).
//
// The threshold used is 0.99 (not 1.0) to absorb the hard kMaxTamSteps=300
// cap in EpisodeRunner (external to the TAM). On a well-connected small graph
// the gap from 1.0 should be zero or negligible.
//
// Returns true (PASS) or false (FAIL). Prints one-line verdict to stdout.
bool run_tam_mc_test(const std::string& osm_file,
                     const std::string& cache_dir);

#endif // TAM_MC_TEST_HPP