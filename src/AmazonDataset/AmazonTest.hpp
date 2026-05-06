#ifndef AMAZON_TEST_HPP
#define AMAZON_TEST_HPP

#include <string>

// Run DbVNS on Amazon Last Mile Challenge routes.
//
// Uses the "new" dataset files (small, ~4 MB total):
//   model_apply_inputs/new_route_data.json
//   model_apply_inputs/new_travel_times.json
//   model_score_inputs/new_actual_sequences.json
//
// For each route:
//   - Warehouse = stop at sequence position 0 (actual_sequences ground truth).
//   - Delivery stops are paired consecutively as PDP tasks.
//   - OperableEnvironment is fed directly from the travel-time matrix (no GeoBox/GlobalMemory).
//   - One LocalSolutionAgent per delivery node (mirrors DeliveryAgent::plan() logic).
//   - Metrics: DbVNS solution cost, ground-truth cost, ratio, execution time.
//
// data_dir should point to:
//   AmazonDataset/~/.rc-cli/data
void test_amazon_routes(const std::string& data_dir);

#endif // AMAZON_TEST_HPP
