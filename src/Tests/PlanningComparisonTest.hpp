#ifndef PLANNING_COMPARISON_TEST_HPP
#define PLANNING_COMPARISON_TEST_HPP

#include <cstdint>
#include <string>
#include <vector>

// ── Batch planning comparison test ─────────────────────────────────────────
//
// Runs N_SEEDS independent lifelong episodes across multiple small-map cities,
// each with a randomised density × agents scenario. For every seed both
// insertion strategies are evaluated:
//   - MCA           (cheapest insertion — paper c1)
//   - DoubleHorizon (Mitrovic-Minic+2004 c3 adapted, paper-faithful)
//
// Outputs:
//   - planning_summary.csv   : one row per (seed, city, scenario, mode) with
//                              all metrics → load into pandas / R for analysis.
//   - details/seed_NNN_*.csv : task & agent traces (every detail_every seeds).
//   - details/seed_NNN_*.svg : visual render (every detail_every seeds).
//   - planning_report.txt    : human-readable summary table.
//
// n_seeds       : number of independent random scenarios to run (default 100).
// detail_every  : save full per-seed details every N seeds (default 10).
// max_tasks     : cap on expected tasks per episode (default 25).
// cities        : list of city names from CityRegistry to cycle over.
//                 default = {Tokyo_Small, Kyoto_Small, LosAngeles_Small}.
// density_min   : min scenario density multiplier (default 0.5).
// density_max   : max scenario density multiplier (default 2.5).
//
// Returns 0 on success, non-zero on hard failure.
int run_planning_comparison_test(uint32_t base_seed,
                                  int max_tasks,
                                  const std::string& osm_root,
                                  const std::string& cache_root,
                                  const std::string& output_dir,
                                  int n_seeds = 100,
                                  int detail_every = 10,
                                  const std::vector<std::string>& cities = {},
                                  float density_min = 0.5f,
                                  float density_max = 2.5f);

#endif // PLANNING_COMPARISON_TEST_HPP
