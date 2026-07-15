#include "Tests/ModuleTests.hpp"
#include "Tests/TestSupport.hpp"
#include "Tests/TamMcTest.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/StructuresParam/ScenarioConfig.hpp"
#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include "DMASforPD/Algorithms/TAM.hpp"
#include "DMASforPD/Structures/ObjectiveNode.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "SoTA/SolverFramework.hpp"
#include "SoTA/Standalone/CA.hpp"
#include "SoTA/Standalone/HAPC.hpp"
#include "DMASforPD/Algorithms/TDAStar.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <queue>
#include <unordered_set>

// Feasible config: short trips + enough steps so every reasonable planner/policy
// actually DELIVERS tasks within the horizon (lets us assert completion, not
// just "it ran").
static EpisodeConfig module_config() {
    EpisodeConfig cfg;
    cfg.speed_mps           = 15.f;
    cfg.min_task_dist_m     = 50.f;
    cfg.max_task_dist_m     = 500.f;
    cfg.cluster_prob        = 0.3f;
    cfg.n_hot_zones         = 2;
    cfg.hot_zone_radius     = 400.f;
    cfg.max_tasks_per_agent = 3;
    cfg.phases    = { { 400, 3, 4, 0.0f, 2 } };
    cfg.env_scale = 0.7f;   // ~70 tasks, matches prior volume
    return cfg;
}

bool run_module_tests(const std::string& osm_file, const std::string& cache_dir)
{
    std::cout << "\n=== C. Modules ===\n";

    GeoBox gb = load_smoke_geobox(osm_file, cache_dir);
    CHECK(gb.is_valid && gb.data.nodes.size() >= 2, "GeoBox invalid / too small");

    // ── [1] GlobalMemory — lifecycle + ID refresh ─────────────────────────────
    {
        PDPGlobalMemory mem(gb);
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
        const float lo = cm.adjusted_cost(w, 100.f, 200.f, 2);
        cm.add_agent(w, 0, 5, 3);
        const float hi = cm.adjusted_cost(w, 100.f, 200.f, 2);
        CHECK(hi > lo && lo >= 100.f, "BPR factor not monotone / below free flow");
        std::cout << "  [2] BPR OK — cap floor + monotone factor\n";
    }

    // ── [3] TAM — multi-candidate allocation correctness ──────────────────────
    CHECK(run_tam_mc_test(osm_file, cache_dir), "TAM multi-candidate test failed");
    std::cout << "  [3] TAM OK\n";

    // ── Episode-runner mode check. Verifies the mode RUNS and, crucially, that
    //    the simulation it produced is VALID: pickup-before-delivery on every
    //    served task (pairing_violations_runtime) and capacity never exceeded
    //    (capacity_violations_runtime). require_delivery asks the mode to also
    //    actually complete ≥ 1 task (used for the non-RL modes that should).
    auto run_mode = [&](const char* label, PolicyMode m, bool require_delivery) -> bool {
        EpisodeConfig cfg = module_config();
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();
        RunResult r;
        try {
            EpisodeRunner runner(cfg, gb, 42u);
            runner.train_mode  = false;
            runner.policy_mode = m;
            r = runner.run(0, 1);
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] " << label << ": " << e.what() << "\n";
            return false;
        }
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();
        const auto& mx = r.metrics;
        bool ok = true;
        if (!in01(mx.accept_rate) || !in01(mx.throughput_rate)) {
            std::cout << "  [FAIL] " << label << ": metrics out of [0,1]\n"; ok = false;
        }
        if (mx.pairing_violations_runtime != 0) {
            std::cout << "  [FAIL] " << label << ": " << mx.pairing_violations_runtime
                      << " pairing (P-before-D) violations\n"; ok = false;
        }
        if (mx.capacity_violations_runtime != 0) {
            std::cout << "  [FAIL] " << label << ": " << mx.capacity_violations_runtime
                      << " capacity violations\n"; ok = false;
        }
        if (require_delivery && mx.tasks_completed <= 0) {
            std::cout << "  [FAIL] " << label << ": delivered 0 tasks on a feasible config\n";
            ok = false;
        }
        return ok;
    };

    // ── [4] Planning — DbVNS / ALNS / DoubleHorizon (must route + deliver) ────
    CHECK(run_mode("DbVNS",         PolicyMode::DbVNS,         true), "DbVNS planning failed");
    CHECK(run_mode("ALNS",          PolicyMode::ALNS,          true), "ALNS planning failed");
    CHECK(run_mode("DoubleHorizon", PolicyMode::DoubleHorizon, true), "DoubleHorizon planning failed");
    std::cout << "  [4] Planning OK — DbVNS / ALNS / DoubleHorizon (valid routes + delivery)\n";

    // ── [5] Scoring — RMCA (deterministic marginal-cost scorer) ───────────────
    CHECK(run_mode("RMCA", PolicyMode::RMCA, true), "RMCA scoring failed");
    std::cout << "  [5] Scoring OK — RMCA\n";

    // ── [6] Policies — MAPPO / IPPO / MAPPER / Hybrid (untrained → valid only) ─
    CHECK(run_mode("MAPPO",  PolicyMode::MAPPO,  false), "MAPPO policy failed");
    CHECK(run_mode("IPPO",   PolicyMode::IPPO,   false), "IPPO policy failed");
    CHECK(run_mode("MAPPER", PolicyMode::MAPPER, false), "MAPPER policy failed");
    CHECK(run_mode("Hybrid", PolicyMode::Hybrid, false), "Hybrid policy failed");
    std::cout << "  [6] Policies OK — MAPPO / IPPO / MAPPER / Hybrid (valid sims)\n";

    // ── [7] Determinism — same (mode, episode_seed) ⇒ identical metrics ───────
    {
        EpisodeConfig cfg = module_config();
        auto run_seeded = [&]() {
            bid_policy(BidPolicyKind::MAPPO).clear_buffers();
            EpisodeRunner runner(cfg, gb, 1u);
            runner.train_mode = false;
            runner.policy_mode = PolicyMode::TamAlwaysAccept;
            RunResult r = runner.run(0, 1, /*scenario*/{}, /*episode_seed*/777u);
            bid_policy(BidPolicyKind::MAPPO).clear_buffers();
            return r.metrics;
        };
        const auto a = run_seeded();
        const auto b = run_seeded();
        CHECK(a.tasks_appeared == b.tasks_appeared
           && a.tasks_completed == b.tasks_completed
           && std::fabs(a.throughput_rate - b.throughput_rate) < 1e-6f,
              "same (mode, episode_seed) gave different metrics (non-deterministic)");
        std::cout << "  [7] Determinism OK — replay identical (appeared=" << a.tasks_appeared
                  << " completed=" << a.tasks_completed << ")\n";
    }

    // ── [8] SoTA standalone solvers — full SolverRunner pipeline (CA, HAPC) ────
    {
        EpisodeConfig cfg = module_config();
        auto run_solver = [&](const char* label, ISolver& s) -> bool {
            SolverRunner runner(cfg, gb, /*scenario*/{}, /*seed*/55u);
            SolverMetrics m;
            try { m = runner.run(s); }
            catch (const std::exception& e) {
                std::cout << "  [FAIL] " << label << ": " << e.what() << "\n"; return false;
            }
            bool ok = m.tasks_appeared > 0
                   && in01(m.throughput_rate)
                   && m.capacity_violations == 0
                   && m.pairing_violations == 0;
            if (!ok)
                std::cout << "  [FAIL] " << label << ": appeared=" << m.tasks_appeared
                          << " thr=" << m.throughput_rate << " capV=" << m.capacity_violations
                          << " pairV=" << m.pairing_violations << "\n";
            return ok;
        };
        { FaithfulCASolver               s; CHECK(run_solver("CA",   s), "CA solver failed"); }
        { HybridAdaptivePredictiveSolver s; CHECK(run_solver("HAPC", s), "HAPC solver failed"); }
        std::cout << "  [8] SoTA solvers OK — CA + HAPC (valid SolverRunner pipeline)\n";
    }

    // ── [9] Scenario grid — 27 deterministic task×congestion×fleet combos ─────
    {
        const auto g1 = make_scenario_grid();
        const auto g2 = make_scenario_grid();
        CHECK(g1.size() == 27, "scenario grid must be 3 task × 3 congestion × 3 fleet = 27");
        bool same = g1.size() == g2.size();
        for (size_t i = 0; same && i < g1.size(); ++i)
            same = g1[i].task_profile == g2[i].task_profile
                && g1[i].congestion_profile == g2[i].congestion_profile
                && g1[i].agents_mult  == g2[i].agents_mult
                && g1[i].label == g2[i].label;
        CHECK(same, "scenario grid is not deterministic");
        std::cout << "  [9] Scenario grid OK — 27 deterministic combos\n";
    }

    // ── [10] TD-A* is time-dependent: jamming the path slows it or reroutes ───
    {
        // Pick A and a node ~12 hops away (bounded path, reroute possible).
        std::vector<osmium::object_id_type> ids;
        ids.reserve(gb.data.nodes.size());
        for (const auto& [id, p] : gb.data.nodes) { (void)p; ids.push_back(id); }
        const osmium::object_id_type A = ids.front();
        osmium::object_id_type B = A;
        {
            std::unordered_set<osmium::object_id_type> seen{ A };
            std::queue<std::pair<osmium::object_id_type,int>> q; q.push({ A, 0 });
            while (!q.empty()) {
                auto [n, dep] = q.front(); q.pop(); B = n;
                if (dep >= 12) break;
                auto nit = gb.data.nodes.find(n);
                if (nit == gb.data.nodes.end()) continue;
                for (auto wid : nit->second.incident_ways) {
                    auto wit = gb.data.ways.find(wid);
                    if (wit == gb.data.ways.end()) continue;
                    for (auto nb : { wit->second.node1_id, wit->second.node2_id })
                        if (!seen.count(nb)) { seen.insert(nb); q.push({ nb, dep + 1 }); }
                }
            }
        }
        CHECK(B != A, "could not find a distinct target node");

        CongestionMap empty;
        const TDAStarResult freep = td_astar(gb, A, B, 0, 10.f, empty);
        CHECK(freep.valid(), "TD-A* found no free-flow path");

        CongestionMap jam;                                    // jam the whole free path
        for (auto e : freep.edges) jam.add_agent(e, 0, 5000, 50);
        const TDAStarResult jammed = td_astar(gb, A, B, 0, 10.f, jam);
        CHECK(jammed.valid(), "TD-A* found no path under congestion");
        CHECK(jammed.total_time > freep.total_time || jammed.edges != freep.edges,
              "TD-A* ignored time-dependent congestion (same path, same time)");
        std::cout << "  [10] TD-A* OK — time " << freep.total_time << " -> " << jammed.total_time
                  << (jammed.edges != freep.edges ? " (rerouted)" : "") << "\n";
    }

    // ── [11] TAM retrieves the candidate agents for a task's objective nodes ──
    // Static (distance) candidate discovery: the agent standing on the pickup
    // must be retrieved, the task allocated, and memory updated with the winner.
    // Also checks the node-event invariant: the current objective sits in the
    // node's chain at t* (the first segment, not the start sentinel).
    {
        PDPGlobalMemory mem(gb);
        mem.total_steps = 2000;

        std::vector<osmium::object_id_type> ids;
        for (const auto& [id, p] : gb.data.nodes) { (void)p; ids.push_back(id); if (ids.size() >= 3) break; }
        CHECK(ids.size() >= 3, "need 3 nodes for the TAM scenario");

        // One step away from `n` along an incident edge, avoiding `avoid`.
        auto step_from = [&](osmium::object_id_type n, osmium::object_id_type avoid) {
            for (auto wid : gb.data.nodes.at(n).incident_ways) {
                const auto& w = gb.data.ways.at(wid);
                const auto nb = (w.node1_id == n) ? w.node2_id : w.node1_id;
                if (nb != avoid) return nb;
            }
            return n;
        };
        const auto pickup = ids[1], delivery = ids[2];
        const auto posNear = step_from(pickup, delivery);   // 1 hop from the pickup

        DeliveryAgent a0(0, pickup);     // standing ON the pickup (source-node case)
        DeliveryAgent a1(1, posNear);    // one hop from the pickup
        a0.max_capacity = 3; a1.max_capacity = 3;
        mem.register_delivery_agent(a0);
        mem.register_delivery_agent(a1);

        const int t = mem.add_task({ pickup, 1 }, { delivery, 2 });
        CHECK(mem.node_events.node_chain(pickup).find(0.f)->objective_id.count(t) == 1,
              "current objective not present in the pickup node chain at t*");

        TaskAllocationModule::Params p;
        p.always_accept = true;          // deterministic: every candidate bids μ=1
        TaskAllocationModule tam(*mem.get_task(t), p);
        int guard = 0;
        while (!tam.step(mem, 10.f) && ++guard < 100000) {}

        const auto& cands = tam.candidates_in_order();
        CHECK(!cands.empty(), "TAM retrieved no candidate for the objective nodes");
        for (int c : cands) CHECK(c == 0 || c == 1, "TAM returned an unregistered agent");
        CHECK(std::find(cands.begin(), cands.end(), 0) != cands.end(),
              "the agent standing on the pickup was not retrieved as a candidate");
        CHECK(tam.is_allocated(), "TAM did not allocate the task");
        CHECK(tam.winner_agent() == 0 || tam.winner_agent() == 1, "winner is not a registered agent");
        CHECK(mem.get_task(t)->agent_id == tam.winner_agent(), "memory not updated with the TAM winner");

        std::cout << "  [11] TAM retrieval OK — " << cands.size()
                  << " candidate(s), winner=" << tam.winner_agent() << "\n";
    }

    std::cout << "=== C PASS ===\n";
    return true;
}
