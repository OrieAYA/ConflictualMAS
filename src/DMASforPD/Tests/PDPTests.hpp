#ifndef PDP_TESTS_HPP
#define PDP_TESTS_HPP

#include <string>

// Interactive: asks for cache name, output name, number of agents/tasks, speed, steps.
// Loads GeoBox, inits PDPGlobalMemory, creates DeliveryAgents, runs TAM allocation,
// calls plan(), marks routes on the geobox, saves and renders the map.
void test_pdp_system(const std::string& cache_dir);

// Interactive: asks for location + objectives, creates a geobox with two objective
// groups (pickup group 1, delivery group 2) ready for test_pdp_system.
void pdp_create_geobox(const std::string& osm_file, const std::string& cache_dir);

#endif // PDP_TESTS_HPP
