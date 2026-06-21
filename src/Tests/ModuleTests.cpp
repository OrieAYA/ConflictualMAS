#include "Tests/ModuleTests.hpp"
#include "Tests/TestSupport.hpp"
#include "Tests/TamMcTest.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Structures/ObjectiveNode.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <cmath>
#include <exception>
#include <iostream>

// Light config: enough tasks to exercise allocation/planning, small enough to
// keep the whole battery fast even with the heavier DbVNS planner.
static EpisodeConfig module_config() {
    EpisodeConfig cfg;
    cfg.speed_mps           = 10.f;
    cfg.min_task_dist_m     = 50.f;
    cfg.max_task_dist_m     = 800.f;
    cfg.cluster_prob        = 0.3f;
    cfg.n_hot_zones         = 2;
    cfg.hot_zone_radius     = 400.f;
    cfg.max_tasks_per_agent = 3;
    cfg.phases = { { 200, 8.f, 2, 3, 0.0f, 2 } };
    return cfg;
}

bool run_module_tests(const std::string& osm_file, const std::string& cache_dir)
{
    std::cout << "\n=== C. Modules ===\n";

    GeoBox gb = load_smoke_geobox(osm_file, cache_dir);
    CHECK(gb.is_valid && gb.data.nodes.size() >= 2, "GeoBox invalid / too small");
    Pathfinder pf(gb);

    // ── [1] GlobalMemory — lifecycle + ID refresh ─────────────────────────────
    {
        PDPGlobalMemory mem(gb, pf);
        auto it = gb.data.nodes.begin();
        const osmium::object_id_type n1 = it->first; ++it;
        const osmium::object_id_type n2 = it->first;
        const ObjectiveNode P{ n1, 1 }, D{ n2, 2 };

        const int id1 = mem.add_task(P, D);
        CHECK(id1 == 0,                       "first task id should be 0");
        CHECK(mem.get_task(id1) != nullptr,   "get_task returns null for valid id");
        CHECK(mem.count_available() == 1,     "task not in available list");

        mem.assign_task(id1, /*agent_id=*/0);
        CHECK(mem.count_allocated() == 1 && mem.count_available() == 0, "assign transition wrong");
        mem.complete_task(id1);
        CHECK(mem.count_finished() == 1 && mem.count_allocated() == 0, "complete transition wrong");

        const int id2 = mem.add_task(P, D);
        CHECK(id2 == 1, "ids must be monotonic within an episode");

        mem.reset_episode();
        CHECK(mem.count_total() == 0, "reset_episode left residual tasks");
        const int id3 = mem.add_task(P, D);
        CHECK(id3 == 0, "ids must refresh (restart at 0) after reset_episode");
        std::cout << "  [1] GlobalMemory OK — lifecycle + id refresh (0,1 → reset → 0)\n";
    }

    // ── [2] BPR — capacity model + monotone factor ────────────────────────────
    {
        CongestionMap cm;
        CHECK(cm.edge_capacity(1.f)    >= 1.f,  "capacity must floor at 1");
        CHECK(cm.edge_capacity(1000.f) == 50.f, "capacity = 0.05 * length");
        const osmium::object_id_type w = 9;
        cm.add_agent(w, 0, 5, 3);
        const float lo = cm.adjusted_cost(w, 100.f, 200.f, 2);   // some load
        cm.add_agent(w, 0, 5, 3);
        const float hi = cm.adjusted_cost(w, 100.f, 200.f, 2);
        CHECK(hi > lo && lo >= 100.f, "BPR factor not monotone / below free flow");
        std::cout << "  [2] BPR OK — cap floor + monotone factor\n";
    }

    // ── [3] TAM — multi-candidate allocation correctness ──────────────────────
    CHECK(run_tam_mc_test(osm_file, cache_dir), "TAM multi-candidate test failed");
    std::cout << "  [3] TAM OK\n";

    // ── Shared episode runner helper for modes [4][5][6] ──────────────────────
    auto run_mode = [&](const char* label, PolicyMode m, bool train) -> bool {
        EpisodeConfig cfg = module_config();
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();
        RunResult r;
        try {
            EpisodeRunner runner(cfg, gb, pf, 42u);
            runner.train_mode  = train;
            runner.policy_mode = m;
            r = runner.run(0, 1);
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] " << label << ": " << e.what() << "\n";
            return false;
        }
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();
        const bool ok = in01(r.metrics.accept_rate)
                     && in01(r.metrics.throughput_rate)
                     && r.metrics.pdp_violations == 0;
        if (!ok) std::cout << "  [FAIL] " << label << ": invalid metrics / PDP violation\n";
        return ok;
    };

    // ── [4] Planning — DbVNS / MCA / DoubleHorizon ────────────────────────────
    CHECK(run_mode("DbVNS",        PolicyMode::DbVNS,        false), "DbVNS planning failed");
    CHECK(run_mode("MCA",          PolicyMode::MCA,          false), "MCA planning failed");
    CHECK(run_mode("DoubleHorizon",PolicyMode::DoubleHorizon,false), "DoubleHorizon planning failed");
    std::cout << "  [4] Planning OK — DbVNS / MCA / DoubleHorizon (0 violations)\n";

    // ── [5] Scoring — RMCA ────────────────────────────────────────────────────
    CHECK(run_mode("RMCA", PolicyMode::RMCA, false), "RMCA scoring failed");
    std::cout << "  [5] Scoring OK — RMCA\n";

    // ── [6] Policies — MAPPO / IPPO / MAPPER / Hybrid (eval) ──────────────────
    CHECK(run_mode("MAPPO",  PolicyMode::MAPPO,  false), "MAPPO policy failed");
    CHECK(run_mode("IPPO",   PolicyMode::IPPO,   false), "IPPO policy failed");
    CHECK(run_mode("MAPPER", PolicyMode::MAPPER, false), "MAPPER policy failed");
    CHECK(run_mode("Hybrid", PolicyMode::Hybrid, false), "Hybrid policy failed");
    std::cout << "  [6] Policies OK — MAPPO / IPPO / MAPPER / Hybrid\n";

    std::cout << "=== C PASS ===\n";
    return true;
}
