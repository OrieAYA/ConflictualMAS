#include "Tests/MechanicsTests.hpp"
#include "Tests/TestSupport.hpp"
#include "Environment/Simulation/CongestionMap.hpp"
#include "Environment/Structure/Episode.hpp"          // EpisodeGenerator, ScheduledTask
#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/Run/Runner.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include "DMASforPD/Structures/PDPTask.hpp"           // TaskTimeline
#include "DMASforPD/Policy/BidPolicy.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>

// Relative-tolerant float compare for the BPR formula checks.
static bool approx(float a, float b, float tol = 1e-2f) {
    return std::fabs(a - b) <= tol * std::max(1.f, std::fabs(b));
}

// High-density, short-trip config that reliably completes tasks within the
// horizon (same shape as the training smoke multi-task case).
static EpisodeConfig mechanics_config() {
    EpisodeConfig cfg;
    cfg.speed_mps           = 15.f;
    cfg.min_task_dist_m     = 50.f;
    cfg.max_task_dist_m     = 600.f;
    cfg.cluster_prob        = 0.3f;
    cfg.n_hot_zones         = 3;
    cfg.hot_zone_radius     = 400.f;
    cfg.max_tasks_per_agent = 5;
    cfg.phases    = { { 400, 3, 4, 0.0f, 3 } };
    cfg.env_scale = 0.85f;   // ~85 tasks (α·env_scale), matches prior volume
    return cfg;
}

bool run_mechanics_tests(const std::string& osm_file, const std::string& cache_dir)
{
    std::cout << "\n=== B. Mechanics & Simulation ===\n";

    // ── [1] Congestion → BPR edge weight → other agents' travel time ──────────
    {
        CongestionMap cm;                              // α=0.15, β=4, cap=0.05/m
        const osmium::object_id_type way = 1;
        const float dist = 100.f, base = 100.f;
        const float cap  = cm.edge_capacity(dist);     // max(1, 100*0.05) = 5

        CHECK(approx(cm.adjusted_cost(way, base, dist, 5), base),
              "free-flow cost must equal base when load = 0");

        cm.add_agent(way, 0, 10, /*weight=*/5);        // load 5 over [0,10]
        CHECK(cm.get_load(way, 5) == 5, "load bookkeeping wrong");
        const float f1 = cm.adjusted_cost(way, base, dist, 5);
        CHECK(approx(f1, base * (1.f + 0.15f * std::pow(5.f / cap, 4.f))),
              "BPR factor mismatch at load = 5");

        cm.add_agent(way, 0, 10, /*weight=*/5);        // load 10
        const float f2 = cm.adjusted_cost(way, base, dist, 5);
        CHECK(f2 > f1, "cost must rise monotonically with load");
        CHECK(approx(f2, base * (1.f + 0.15f * std::pow(10.f / cap, 4.f))),
              "BPR factor mismatch at load = 10");

        // An agent never congests itself: excluding its own weight → free flow.
        CHECK(approx(cm.adjusted_cost(way, base, dist, 5, /*self_weight=*/10), base),
              "self_weight exclusion broken");

        // Travel time of OTHER agents grows under load.
        const int t_free = CongestionMap{}.traversal_steps(way, dist, 0, 15.f);
        const int t_jam  = cm.traversal_steps(way, dist, 5, 15.f);
        CHECK(t_jam >= t_free, "traversal time must not drop under congestion");
        std::cout << "  [1] Congestion/BPR OK — f(5)=" << f1 << " f(10)=" << f2
                  << " steps " << t_free << "->" << t_jam << "\n";
    }

    GeoBox gb = load_smoke_geobox(osm_file, cache_dir);
    CHECK(gb.is_valid, "GeoBox not valid");

    // ── [2] Lifelong task creation at dispersed arrival steps ─────────────────
    {
        EpisodeConfig cfg = mechanics_config();
        EpisodeGenerator gen(cfg, gb, 7u);
        const auto stream = gen.generate(TemporalProfile::Uniform);
        CHECK(!stream.empty(), "no task generated");
        int lo = stream.front().arrival_step, hi = lo;
        for (const auto& s : stream) {
            lo = std::min(lo, s.arrival_step);
            hi = std::max(hi, s.arrival_step);
        }
        CHECK(hi > lo, "all tasks arrive at the same step (not lifelong)");
        CHECK(hi - lo >= cfg.total_steps() / 4, "task arrivals not spread over the episode");
        std::cout << "  [2] Lifelong tasks OK — " << stream.size()
                  << " tasks, arrivals in [" << lo << "," << hi << "] of "
                  << cfg.total_steps() << " steps\n";
    }

    // ── [3] Time progression + agent movement + task completion ───────────────
    {
        EpisodeConfig cfg = mechanics_config();
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();

        std::unique_ptr<EpisodeRunner> runner;
        try {
            runner = std::make_unique<EpisodeRunner>(cfg, gb, 43u);
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] runner ctor: " << e.what() << "\n"; return false;
        }
        runner->train_mode  = false;
        runner->policy_mode = PolicyMode::TamAlwaysAccept;   // deterministic, reliable completion

        RunResult r;
        try { r = runner->run(0, 1, {}, /*episode_seed*/4242u); }
        catch (const std::exception& e) {
            std::cout << "  [FAIL] episode: " << e.what() << "\n"; return false;
        }
        bid_policy(BidPolicyKind::MAPPO).clear_buffers();

        CHECK(r.metrics.tasks_appeared  > 0, "no task injected over time (event stream dead)");
        CHECK(r.metrics.tasks_completed > 0, "no task completed (pickup→delivery flow broken)");
        CHECK(r.metrics.pdp_violations == 0, "PDP precedence/capacity violated");

        const auto& mem = runner->memory();
        CHECK(mem.count_finished() == r.metrics.tasks_completed,
              "finished list size ≠ completed count");

        // Completion mechanics: every finished task had pickup THEN delivery
        // satisfied (and is therefore off the active lists).
        for (const PDPTask* t : mem.finished_tasks) {
            CHECK(t->timeline.picked_step    >= 0, "finished task never picked");
            CHECK(t->timeline.delivered_step >= 0, "finished task never delivered");
            CHECK(t->timeline.delivered_step >= t->timeline.picked_step,
                  "delivery recorded before pickup");
        }

        // Movement: every agent ended on a real graph node (moved along edges).
        for (const auto& a : runner->agents())
            CHECK(gb.data.nodes.count(a->current_node) > 0,
                  "agent left the road graph (off-node position)");

        std::cout << "  [3] Time/movement/completion OK — appeared="
                  << r.metrics.tasks_appeared << " completed=" << r.metrics.tasks_completed
                  << " finished_list=" << mem.count_finished() << "\n";
    }

    // ── [4] Ghost congestion measurably raises network load (system-level) ────
    // Same episode_seed ⇒ identical tasks/agents; the only difference is the
    // injected ghost traffic. Closes the loop the BPR unit check in [1] opens:
    // ghosts → higher edge load → the cost model the agents actually pay.
    {
        EpisodeConfig off = mechanics_config();           // no ghosts (default)
        EpisodeConfig on  = mechanics_config();
        on.enable_ghost_traffic    = true;
        on.ghost_n_events          = 400;
        on.ghost_n_events_user_set = true;                // fixed count for this test
        on.ghost_window_steps      = 8;
        on.ghost_hot_way_count     = 15;                  // concentrate → real pileups

        auto cong_of = [&](const EpisodeConfig& cfg) -> float {
            bid_policy(BidPolicyKind::MAPPO).clear_buffers();
            EpisodeRunner runner(cfg, gb, 5u);
            runner.train_mode  = false;
            runner.policy_mode = PolicyMode::TamAlwaysAccept;
            const float c = runner.run(0, 1, {}, /*episode_seed*/2024u).metrics.mean_congestion;
            bid_policy(BidPolicyKind::MAPPO).clear_buffers();
            return c;
        };
        const float c_off = cong_of(off);
        const float c_on  = cong_of(on);
        CHECK(isfin(c_off) && isfin(c_on), "congestion metric not finite");
        CHECK(c_on > c_off, "ghost traffic did not raise network congestion");
        std::cout << "  [4] Congestion impact OK — mean load " << c_off
                  << " (off) -> " << c_on << " (ghost on)\n";
    }

    // ── [5] BPR is TIME-dependent: a windowed load only costs inside the window ─
    // The defining dynamic property — the cost of an edge depends on WHEN it is
    // traversed, not just how loaded it is overall.
    {
        CongestionMap cm;
        const osmium::object_id_type w = 7;
        const float dist = 100.f, base = 100.f;
        cm.add_agent(w, 100, 200, 8);                   // load 8 only during [100,200]
        const float before = cm.adjusted_cost(w, base, dist, 50);
        const float during = cm.adjusted_cost(w, base, dist, 150);
        const float after  = cm.adjusted_cost(w, base, dist, 250);
        CHECK(approx(before, base) && approx(after, base),
              "cost must be free-flow outside the load's time window");
        CHECK(during > base, "cost must rise inside the load window (time-dependence)");
        std::cout << "  [5] BPR time-dependence OK — cost " << before << "/" << during
                  << "/" << after << " (before/in/after window)\n";
    }

    std::cout << "=== B PASS ===\n";
    return true;
}
