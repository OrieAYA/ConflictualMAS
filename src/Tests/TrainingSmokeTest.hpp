#ifndef TRAINING_SMOKE_TEST_HPP
#define TRAINING_SMOKE_TEST_HPP

#include <string>

// Quick end-to-end sanity check for the training pipeline.
// Uses the kanto OSM file (already present in src/maps/) with a tiny bbox
// so no extra downloads are needed.
//
// What it verifies:
//   1. GeoBox loads and has valid road nodes.
//   2. EpisodeRunner constructs without crash.
//   3. A training episode produces metrics in [0,1] and a non-zero PPO update.
//   4. An eval episode (Greedy + MAPPO) runs without crash.
//   5. The shared policy buffer is correctly cleared between modes.
//
// Returns true (PASS) or false (FAIL). Prints a one-line verdict to stdout.
bool run_training_smoke_test(const std::string& osm_file,
                             const std::string& cache_dir);

#endif // TRAINING_SMOKE_TEST_HPP
