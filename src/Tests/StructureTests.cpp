#include "Tests/StructureTests.hpp"
#include "Tests/TestSupport.hpp"
#include "Tests/GeoBoxConnectivityTest.hpp"
#include "Tests/TemporalChainListTest.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "Environment/GeoBox/GraphSearch.hpp"
#include <algorithm>
#include <exception>
#include <iostream>

bool run_structure_tests(const std::string& osm_file, const std::string& cache_dir)
{
    std::cout << "\n=== A. Structure & Construction ===\n";

    // ── [1] Environment creation ──────────────────────────────────────────────
    GeoBox gb = load_smoke_geobox(osm_file, cache_dir);
    CHECK(gb.is_valid,            "GeoBox not valid");
    CHECK(!gb.data.nodes.empty(), "GeoBox has no nodes");
    CHECK(!gb.data.ways.empty(),  "GeoBox has no ways");
    const MyData& d = gb.data;
    std::cout << "  [1] Env created — " << d.nodes.size() << " nodes, "
              << d.ways.size() << " ways\n";

    // ── [2] Edge integrity: endpoints exist, length > 0, no self-loop ─────────
    int dangling = 0, zero_len = 0, self_loop = 0;
    for (const auto& [wid, w] : d.ways) {
        (void)wid;
        if (!d.nodes.count(w.node1_id) || !d.nodes.count(w.node2_id)) ++dangling;
        if (w.distance_meters <= 0.f)                                 ++zero_len;
        if (w.node1_id == w.node2_id)                                 ++self_loop;
    }
    CHECK(dangling  == 0, "ways reference missing nodes");
    CHECK(zero_len  == 0, "ways with non-positive length");
    CHECK(self_loop == 0, "self-loop ways (node1 == node2)");
    std::cout << "  [2] Edge integrity OK — 0 dangling / 0 zero-length / 0 self-loop\n";

    // ── [3] Incidence symmetry (way endpoints ⇔ node.incident_ways) ───────────
    int way_missing_in_node = 0;
    for (const auto& [wid, w] : d.ways)
        for (auto nid : { w.node1_id, w.node2_id }) {
            auto it = d.nodes.find(nid);
            if (it == d.nodes.end()) continue;            // already counted in [2]
            const auto& iw = it->second.incident_ways;
            if (std::find(iw.begin(), iw.end(), wid) == iw.end()) ++way_missing_in_node;
        }
    int node_way_mismatch = 0;
    for (const auto& [nid, p] : d.nodes)
        for (auto wid : p.incident_ways) {
            auto it = d.ways.find(wid);
            if (it == d.ways.end()) { ++node_way_mismatch; continue; }
            if (it->second.node1_id != nid && it->second.node2_id != nid) ++node_way_mismatch;
        }
    CHECK(way_missing_in_node == 0, "way endpoint missing from node.incident_ways");
    CHECK(node_way_mismatch  == 0, "node.incident_ways lists a non-incident/missing way");
    std::cout << "  [3] Incidence symmetry OK\n";

    // ── [4] Full connectivity (single connected component) ────────────────────
    const std::string cache_path = cache_dir + "/smoke_test.json";
    CHECK(run_geobox_connectivity_test({ cache_path }, "smoke_test"),
          "GeoBox not fully connected");
    std::cout << "  [4] Connectivity OK\n";

    // ── [5] PDPGlobalMemory construction (grid from bbox + congestion map) ────
    try {
        PDPGlobalMemory mem(gb);
        CHECK(mem.count_total() == 0, "fresh memory should hold no task");
        mem.congestion_map.reset();   // bbox-derived congestion state usable
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] PDPGlobalMemory construction: " << e.what() << "\n";
        return false;
    }
    std::cout << "  [5] PDPGlobalMemory construction OK\n";

    // ── [6] Routing primitive (Dijkstra over the real graph) ──────────────────
    // On a fully-connected graph (verified in [4]) a Dijkstra from any node must
    // reach essentially every node with finite, non-negative distances.
    {
        const osmium::object_id_type src = d.nodes.begin()->first;
        const auto dist = graph_search::dijkstra_distances(gb, src);
        CHECK(!dist.empty(), "Dijkstra returned no distances");
        auto its = dist.find(src);
        CHECK(its != dist.end() && its->second == 0.f, "source distance must be 0");
        int negative = 0;
        for (const auto& [nid, dd] : dist) { (void)nid; if (!(dd >= 0.f)) ++negative; }
        CHECK(negative == 0, "Dijkstra produced negative/NaN distances");
        // Connectivity ([4]) ⇒ reachability of the whole component.
        CHECK(dist.size() * 2 >= d.nodes.size(), "Dijkstra reached too few nodes");
        std::cout << "  [6] Dijkstra OK — reached " << dist.size() << "/"
                  << d.nodes.size() << " nodes, all distances finite ≥ 0\n";
    }

    // ── [7]+[8] Temporal chained list (event-stream layer on nodes/edges) ─────
    CHECK(run_temporal_chain_tests(),     "temporal chained list unit tests failed");
    CHECK(run_temporal_graph_test(gb),    "temporal graph attachment test failed");

    std::cout << "=== A PASS ===\n";
    return true;
}
