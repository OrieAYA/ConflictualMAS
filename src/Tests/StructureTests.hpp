#ifndef TESTS_STRUCTURE_TESTS_HPP
#define TESTS_STRUCTURE_TESTS_HPP

#include <string>

// Battery A — Construction & Structure.
//
// Verifies the environment is built correctly and its road-graph data structure
// is internally consistent:
//   1. GeoBox creation (valid, non-empty nodes & ways).
//   2. Edge integrity: every way's endpoints exist, length > 0, no self-loop.
//   3. Incidence symmetry: way endpoints ⇔ node.incident_ways (both ways).
//   4. Full connectivity (single connected component).
//   5. PDPGlobalMemory construction (RegionStatsGrid + congestion map).
//   6. Routing primitive (Dijkstra reaches the whole component).
//   7. Temporal chained list (event-stream segments: order/overlap/find/expiry).
//   8. Temporal graph (one chain per node/edge with correct incident_elements).
//
// Returns true (PASS) / false (FAIL).
bool run_structure_tests(const std::string& osm_file, const std::string& cache_dir);

#endif // TESTS_STRUCTURE_TESTS_HPP
