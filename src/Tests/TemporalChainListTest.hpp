#ifndef TESTS_TEMPORAL_CHAIN_LIST_TEST_HPP
#define TESTS_TEMPORAL_CHAIN_LIST_TEST_HPP

struct GeoBox;

// Unit tests for the temporal chained list itself (sentinels, time ordering,
// overlap-certified present_agent, removal reversal, find, expiry purge).
bool run_temporal_chain_tests();

// Attachment test: TemporalGraph builds one chain per node/edge on demand with
// the correct static incident_elements and usable insert/find.
bool run_temporal_graph_test(const GeoBox& geo_box);

#endif // TESTS_TEMPORAL_CHAIN_LIST_TEST_HPP
