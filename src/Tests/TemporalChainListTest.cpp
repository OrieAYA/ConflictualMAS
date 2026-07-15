#include "Tests/TemporalChainListTest.hpp"
#include "Tests/TestSupport.hpp"
#include "Environment/Simulation/TemporalChainList.hpp"
#include "Environment/Simulation/TemporalGraph.hpp"
#include "Environment/GeoBox/Box.hpp"
#include <iostream>

bool run_temporal_chain_tests()
{
    // ── T1 sentinels ──────────────────────────────────────────────────────────
    {
        TemporalChainList L(100.f);
        CHECK(L.size() == 0,                  "fresh list must be empty");
        CHECK(L.start()->time_end == 0.f,     "start sentinel must sit at t=0");
        CHECK(L.end()->time_end   == 100.f,   "end sentinel must sit at episode_end");
        CHECK(L.start()->next == L.end() && L.end()->before == L.start(),
              "sentinels must be linked back-to-back");
    }

    // ── T2 insertion keeps ascending time_end order ───────────────────────────
    {
        TemporalChainList L(200.f);
        L.insert(50.f, 10.f); L.insert(20.f, 5.f); L.insert(80.f, 10.f);
        CHECK(L.size() == 3, "size after three inserts");
        float prev = -1.f; int n = 0;
        for (TemporalNode* c = L.start()->next; c != L.end(); c = c->next) {
            CHECK(c->time_end >= prev, "segments not ordered by time_end");
            prev = c->time_end; ++n;
        }
        CHECK(n == 3, "walk visited the wrong number of segments");
    }

    // ── T3 load_at sums overlapping occupancy ─────────────────────────────────
    {
        TemporalChainList L(200.f);
        TemporalNode* A = L.insert(30.f, 20.f);   // [10, 30]
        TemporalNode* B = L.insert(40.f, 20.f);   // [20, 40] overlaps A on [20,30]
        L.insert(100.f, 5.f);                     // [95,100] disjoint
        CHECK(L.load_at(25.f) == 2, "load_at must sum both overlapping segments");
        CHECK(L.load_at(15.f) == 1, "load_at outside the overlap counts one");
        CHECK(L.load_at(97.f) == 1, "disjoint segment contributes its own load");
        CHECK(L.load_at(50.f) == 0, "gap between segments has zero load");
        (void)A;

        // ── T4 remove drops that segment's load contribution ──────────────────
        L.remove(B);
        CHECK(L.load_at(25.f) == 1, "remove must drop the removed segment's load");
        CHECK(L.size() == 2,        "size after one removal");
    }

    // ── T5 find returns the segment covering t (queries non-decreasing) ───────
    {
        TemporalChainList L(200.f);
        L.insert(20.f, 10.f); L.insert(50.f, 10.f); L.insert(80.f, 10.f);
        CHECK(L.find(15.f)->time_end == 20.f, "find(15) should land on end=20");
        CHECK(L.find(45.f)->time_end == 50.f, "find(45) should land on end=50");
        CHECK(L.find(90.f) == L.end(),        "find(90) past all reals → end sentinel");
    }

    // ── T6 expiry purges the past prefix but never the start sentinel ─────────
    {
        TemporalChainList L(200.f);
        L.insert(20.f, 5.f); L.insert(50.f, 5.f); L.insert(80.f, 5.f);
        L.find(60.f);                             // t* = 60 → drop end<60 (20, 50)
        CHECK(L.size() == 1, "expiry should keep only the segment ending >= 60");
        CHECK(L.start()->time_end == 0.f && L.start()->next != nullptr,
              "start sentinel must survive purge");
    }

    std::cout << "  [7] TemporalChainList OK — sentinels / order / overlap / remove / find / expiry\n";
    return true;
}

bool run_temporal_graph_test(const GeoBox& geo_box)
{
    const MyData& d = geo_box.data;
    TemporalGraph tg(geo_box, 1000.f);
    CHECK(tg.n_node_chains() == 0, "temporal graph must start empty (lazy creation)");

    osmium::object_id_type nid = 0;
    for (const auto& [id, p] : d.nodes)
        if (!p.incident_ways.empty()) { nid = id; break; }
    CHECK(nid != 0, "no node with incident ways");
    TemporalChainList& nc = tg.node_chain(nid);
    CHECK(tg.n_node_chains() == 1, "node chain not created on demand");
    CHECK(nc.start()->incident_elements == d.nodes.at(nid).incident_ways,
          "node chain must list its incident edges");
    CHECK(nc.start()->time_end == 0.f && nc.end()->time_end == 1000.f,
          "node chain sentinels not at [0, episode_end]");

    nc.insert(120.f, 20.f);                         // [100, 120]
    CHECK(nc.load_at(110.f) >= 1, "load_at after insert on a node chain");
    CHECK(tg.has_node_chain(nid) && !tg.has_node_chain(nid + 999983),
          "has_node_chain lookup wrong");

    std::cout << "  [8] TemporalGraph OK — per-node chains, correct incident edges\n";
    return true;
}
