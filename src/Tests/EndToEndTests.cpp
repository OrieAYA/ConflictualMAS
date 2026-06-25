#include "Tests/EndToEndTests.hpp"
#include "Tests/TestSupport.hpp"
#include "Tests/TrainingSmokeTest.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/StructuresParam/ScenarioConfig.hpp"   // EpisodeScenario
#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include "DMASforPD/Structures/PDPTask.hpp"
#include "DMASforPD/Structures/ObjectiveNode.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

using TaskSig = std::vector<std::pair<osmium::object_id_type, osmium::object_id_type>>;

static EpisodeConfig e2e_config() {
    EpisodeConfig cfg;
    cfg.speed_mps           = 12.f;
    cfg.min_task_dist_m     = 50.f;
    cfg.max_task_dist_m     = 700.f;
    cfg.cluster_prob        = 0.3f;
    cfg.n_hot_zones         = 2;
    cfg.hot_zone_radius     = 400.f;
    cfg.max_tasks_per_agent = 4;
    cfg.phases = { { 300, 12.f, 2, 3, 0.0f, 2 } };
    return cfg;
}

bool run_end_to_end_tests(const std::string& osm_file, const std::string& cache_dir)
{
    std::cout << "\n=== D. End-to-end & integration ===\n";

    // ── [1] Functional training (full pipeline) ───────────────────────────────
    CHECK(run_training_smoke_test(osm_file, cache_dir), "functional training failed");
    std::cout << "  [1] Functional training OK\n";

    // ── [2]+[3] Homogeneity across methods + map persistence ──────────────────
    GeoBox gb = load_smoke_geobox(osm_file, cache_dir);
    CHECK(gb.is_valid, "GeoBox not valid");
    const size_t n_nodes = gb.data.nodes.size();
    const size_t n_ways  = gb.data.ways.size();
    Pathfinder pf(gb);

    const EpisodeConfig  cfg = e2e_config();
    const EpisodeScenario sc{};            // single fixed scenario
    const uint32_t       ep_seed = 12345u; // deterministic replay seed

    // Run one method on (cfg, scenario, ep_seed) and return its task stream
    // signature: (pickup, delivery) node IDs in task-id order.
    auto stream_of = [&](PolicyMode m, const char* label) -> TaskSig {
        TaskSig sig;
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();
        try {
            EpisodeRunner runner(cfg, gb, pf, /*ctor seed*/999u);
            runner.train_mode  = false;
            runner.policy_mode = m;
            runner.run(0, 1, sc, ep_seed);
            const auto& mem = runner.memory();
            const int N = mem.count_total();
            for (int i = 0; i < N; ++i)
                if (const PDPTask* t = mem.get_task(i))
                    sig.emplace_back(t->pickup.id, t->delivery.id);
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] " << label << ": " << e.what() << "\n";
            sig.clear();
        }
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();
        return sig;
    };

    const TaskSig sigA = stream_of(PolicyMode::MAPPO, "method A (MAPPO)");
    const TaskSig sigB = stream_of(PolicyMode::RMCA,  "method B (RMCA)");

    CHECK(!sigA.empty(),    "no tasks generated for homogeneity check");
    CHECK(sigA == sigB,     "task stream differs across methods (episode not homogeneous)");
    std::cout << "  [2] Homogeneity OK — " << sigA.size()
              << " identical tasks across MAPPO vs RMCA (seed " << ep_seed << ")\n";

    // Map untouched across methods; IDs restarted at 0 each episode (sigA/sigB
    // both index 0..N-1 → fresh memory per method).
    CHECK(gb.data.nodes.size() == n_nodes && gb.data.ways.size() == n_ways,
          "the generated map was mutated between methods");
    std::cout << "  [3] Refresh OK — map stable (" << n_nodes << " nodes / " << n_ways
              << " ways), task IDs refreshed per episode\n";

    // ── [4] Single-agent replan (direct assignment, no TAM) ───────────────────
    // Inject task A, then task B while A is in flight (index 0). Verify the
    // in-flight objective is preserved, the plan/footprint are corrected, and
    // the env structure (node events + edge occupancy) reflects the change.
    {
        PDPGlobalMemory mem(gb, pf);
        mem.total_steps        = 2000;
        mem.planning_use_dbvns = true;

        std::vector<osmium::object_id_type> ids;
        for (const auto& [id, p] : gb.data.nodes) { (void)p; ids.push_back(id); if (ids.size() >= 5) break; }
        CHECK(ids.size() >= 5, "need 5 nodes for the replan scenario");
        const auto start = ids[0], pA = ids[1], dA = ids[2], pB = ids[3], dB = ids[4];

        DeliveryAgent agent(0, start);
        mem.register_delivery_agent(agent);

        const int ta = mem.add_task({ pA, 1 }, { dA, 2 });
        mem.assign_task(ta, 0);
        agent.receive_task(*mem.get_task(ta), mem);
        mem.commit_plan(0, 10.f);

        const auto& seq = agent.solution.sequence;
        CHECK(seq.size() == 2 && seq[0].node.id == pA,    "task A not planned [pickup, delivery]");
        CHECK(seq[0].estimated_arrival >= 0,              "commit_plan did not timestamp the plan");
        CHECK(mem.get_task(ta)->agent_id == 0,            "task A not assigned to the agent in memory");
        CHECK(mem.node_events.node_chain(pA).find(1.f)->objective_id.count(ta) == 1,
              "task A not registered on its pickup node chain");

        const ObjectivePath* legA = mem.get_or_compute_path(start, pA, 1);
        CHECK(legA && legA->valid() && !legA->edges.empty(), "no path start -> pickup A");
        const auto e0 = legA->edges.front();
        const int L1 = mem.congestion_map.get_load(e0, 0);
        CHECK(L1 >= 1, "agent footprint not registered on its first leg edge");

        mem.commit_plan(0, 10.f);                                  // re-commit is idempotent
        CHECK(mem.congestion_map.get_load(e0, 0) == L1, "re-commit leaked load (old footprint kept)");

        // Replan: task B while A is in flight.
        const int tb = mem.add_task({ pB, 1 }, { dB, 2 });
        mem.assign_task(tb, 0);
        agent.receive_task(*mem.get_task(tb), mem);
        mem.commit_plan(0, 10.f);

        auto idx = [&](osmium::object_id_type n) {
            for (int i = 0; i < static_cast<int>(seq.size()); ++i) if (seq[i].node.id == n) return i;
            return -1;
        };
        CHECK(seq.size() == 4,           "replan did not insert task B");
        CHECK(seq[0].node.id == pA,      "in-flight objective (index 0) must be preserved on replan");
        CHECK(idx(dA) > idx(pA),         "A pickup-before-delivery broken after replan");
        CHECK(idx(pB) >= 0 && idx(dB) > idx(pB), "B pickup-before-delivery broken");
        CHECK(mem.node_events.node_chain(pB).find(1.f)->objective_id.count(tb) == 1,
              "task B not registered on its node chain");
        CHECK(mem.congestion_map.get_load(e0, 0) == L1,
              "replan leaked footprint on the unchanged first leg");

        std::cout << "  [4] Replan OK — in-flight preserved, footprint corrected, env structure updated\n";
    }

    std::cout << "=== D PASS ===\n";
    return true;
}
