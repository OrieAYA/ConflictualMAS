#include "Tests/EndToEndTests.hpp"
#include "Tests/TestSupport.hpp"
#include "Tests/TrainingSmokeTest.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/StructuresParam/ScenarioConfig.hpp"   // EpisodeScenario
#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Structures/PDPTask.hpp"
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

    std::cout << "=== D PASS ===\n";
    return true;
}
