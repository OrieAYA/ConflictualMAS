#ifndef COMPARISON_TEST_HPP
#define COMPARISON_TEST_HPP

#include <string>

// Run DbVNS vs LKH side-by-side on the Amazon Last Mile routes.
//
// Same dataset and route selection as test_amazon_routes (seed=42, 10 routes).
// For each route, both solvers receive the identical OperableEnvironment.
// Results and SVG renders are written to:
//   src/Comparisons/DbVNS_vs_LKH/results/
//     {short_id}_dbvns.svg
//     {short_id}_lkh.svg
//
// data_dir should point to:
//   AmazonDataset/~/.rc-cli/data
void test_comparison(const std::string& data_dir);

#endif // COMPARISON_TEST_HPP
