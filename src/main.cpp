#include "Legacy/Tests/LegacyTests.hpp"
#include "DMASforPD/Tests/PDPTests.hpp"
#include "AmazonDataset/AmazonTest.hpp"
#include "Comparisons/DbVNS_vs_LKH/ComparisonTest.hpp"
#include "Training/MultiCityTrainer.hpp"
#include "Training/CityConfig.hpp"
#include "Training/TrainingConfig.hpp"
#include "Training/TrainingSmokeTest.hpp"
#include <iostream>
#include <string>

int main()
{
    const std::string osm_file  = "C:\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    const std::string cache_dir = "C:\\ConflictualMAS\\src\\geobox_cache_folder";
    const std::string amazon_dir = "C:\\ConflictualMAS\\src\\AmazonDataset\\~\\.rc-cli\\data";

    std::cout << "\n=== ConflictualMAS ===\n\n";
    std::cout << "[Legacy]\n";
    std::cout << "  G  Global Solution Constructor\n";
    std::cout << "  V  VNS Orienteering\n";
    std::cout << "  P  PSO MTTDS\n";
    std::cout << "  E  Create GeoBox\n";
    std::cout << "  I  Initialize POI (Flickr)\n";
    std::cout << "  R  Render map\n\n";
    std::cout << "[DMASforPD]\n";
    std::cout << "  D  PDP System test\n";
    std::cout << "  C  Create PDP GeoBox\n\n";
    std::cout << "[Amazon Last Mile]\n";
    std::cout << "  A  DbVNS on Amazon routes (new dataset, 13 routes)\n";
    std::cout << "  B  DbVNS vs LKH benchmark (same routes, side-by-side)\n\n";
    std::cout << "[MAPPO Training]\n";
    std::cout << "  T  Multi-city MAPPO training (Tokyo / Kyoto / LosAngeles)\n";
    std::cout << "  S  Training smoke test (uses kanto OSM, no extra files needed)\n\n";
    std::cout << "Choice: ";

    std::string rep;
    std::cin >> rep;

    if      (rep == "G" || rep == "g") legacy_run_global(cache_dir);
    else if (rep == "V" || rep == "v") legacy_run_vns(cache_dir);
    else if (rep == "P" || rep == "p") legacy_run_pso(cache_dir);
    else if (rep == "E" || rep == "e") legacy_create_geobox(osm_file, cache_dir);
    else if (rep == "I" || rep == "i") legacy_init_poi(cache_dir);
    else if (rep == "R" || rep == "r") legacy_render(cache_dir);
    else if (rep == "D" || rep == "d") test_pdp_system(cache_dir);
    else if (rep == "C" || rep == "c") pdp_create_geobox(osm_file, cache_dir);
    else if (rep == "A" || rep == "a") test_amazon_routes(amazon_dir);
    else if (rep == "B" || rep == "b") test_comparison(amazon_dir);
    else if (rep == "T" || rep == "t") {
        // ── MAPPO multi-city training ─────────────────────────────────────
        // OSM files expected at:  <osm_root>/<CityName>.osm.pbf
        // GeoBox JSON caches at:  <cache_root>/<CityName>.json  (auto-created)
        // Results (CSV + policy) at:  <output_dir>/
        const std::string osm_root   = "C:\\ConflictualMAS\\src\\maps";
        const std::string cache_root = "C:\\ConflictualMAS\\data\\cache";
        const std::string output_dir = "C:\\ConflictualMAS\\results";

        // set_osm_root must be called BEFORE CityRegistry::all() (lazy static).
        CityRegistry::set_osm_root(osm_root);

        TrainingConfig cfg;
        cfg.cache_root  = cache_root;
        cfg.output_dir  = output_dir;
        cfg.n_rounds    = 50;
        cfg.n_seeds     = 3;
        cfg.eval_every  = 10;
        cfg.save_policy = true;
        cfg.verbose     = true;

        MultiCityTrainer trainer;
        trainer.train(cfg);
    }
    else if (rep == "S" || rep == "s") run_training_smoke_test(osm_file, cache_dir);
    else std::cout << "Unknown option.\n";

    return 0;
}

// =============================================================================
// OPTION S — Training Smoke Test
// =============================================================================
//
// OVERVIEW
//   Quick end-to-end sanity check for the MAPPO training pipeline.
//   Uses the kanto OSM file (already present at src/maps/kanto-latest.osm.pbf)
//   with a small 4×5 km central Tokyo bbox — no extra downloads needed.
//
// HOW TO RUN
//   1. Build  : cmake --build C:\ConflictualMAS\build --config Debug
//   2. Launch : build\Debug\main.exe  →  type S
//
// WHAT IT CHECKS
//   1. GeoBox loads and contains valid road nodes.
//   2. EpisodeRunner constructs without crash (agents, path cache, TaskAgent).
//   3. Training episode (MAPPO, 200 steps, 4 agents): metrics in [0,1],
//      actor_loss and critic_loss finite.
//   4. Greedy + MAPPO eval episodes complete without crash.
//   5. Second training episode: episode reset is clean (tasks, congestion, clock).
//
// EXPECTED OUTPUT
//   [1] GeoBox <- cache / <- OSM   (parses once, cached to geobox_cache_folder/)
//   [2] Config OK — 200 steps, 4 agents max
//   [3] Runner constructed OK
//   [3] Training episode OK — N tasks, acc=X, aloss=Y, closs=Z, Nms
//   [4] Eval OK — Greedy acc=X, MAPPO acc=Y
//   [5] Episode reset OK — 2nd train acc=X
//   === PASS ===
//
// EXPECTED TIME   < 30 seconds (first run may take 2-3 min for OSM parsing).
//
// =============================================================================

// =============================================================================
// OPTION T — MAPPO Multi-City Training
// =============================================================================
//
// OVERVIEW
//   Trains a shared MAPPO policy (actor + centralized critic) on the Lifelong
//   General Pickup-and-Delivery Problem across multiple real city road graphs.
//   The policy learns when delivery agents should accept or refuse task offers
//   under dynamic, multi-phase demand.
//
// ── REQUIRED OSM FILES ───────────────────────────────────────────────────────
//
//   Place at C:\ConflictualMAS\src\maps\<Name>.osm.pbf :
//
//   TRAIN cities (mandatory):
//     Tokyo.osm.pbf        —  extract: osmium extract --bbox=139.60,35.55,139.90,35.80 kanto-latest.osm.pbf
//     Kyoto.osm.pbf        —  extract: osmium extract --bbox=135.65,34.95,135.85,35.10 kanto-latest.osm.pbf
//     LosAngeles.osm.pbf   —  download: geofabrik.de → North America → us-california
//
//   EVAL cities (optional — trainer throws if missing; comment out or set n_eval_episodes=0):
//     NewYork.osm.pbf      —  geofabrik.de → North America → us-northeast
//     Paris.osm.pbf        —  geofabrik.de → Europe → france/ile-de-france
//     London.osm.pbf       —  geofabrik.de → Europe → great-britain
//     Fukuoka.osm.pbf      —  geofabrik.de → Asia → japan/kyushu
//
//   GeoBox JSON caches are auto-created at C:\ConflictualMAS\data\cache\ on
//   first run and reused on subsequent runs (much faster load).
//
// ── TRAINING PROTOCOL ────────────────────────────────────────────────────────
//
//   Seeds             :  3 independent runs (Xavier re-init each seed)
//   Rounds per seed   :  50  (one episode per train city per round)
//   Train episodes    :  50 × 3 cities = 150 per seed  →  450 total
//   Eval checkpoints  :  every 10 rounds + final  =  6 per seed
//   Eval episodes     :  6 × 7 cities × 3 modes × 10 eps  =  1 260 per seed
//   Total episodes    :  1 410 per seed  →  4 230 across all seeds
//
// ── EPISODE STRUCTURE (pseudo-day, 3 600 steps) ──────────────────────────────
//
//   Phase       Steps   Agents   Tasks/agent   Lambda     Hot zones
//   Low          1000    8→10       ~100        ≈ 0.90        4
//   Medium       1500   10→13       ~200        ≈ 1.53        6
//   High         1100   13→15       ~250        ≈ 3.18        8
//
//   Lambda > 1 is legal (Poisson generator — multiple tasks per step possible).
//   Fleet size ramps linearly within each phase.
//   Hot zones are resampled at each phase boundary (morning peak → evening peak).
//   Task distance: 300 m – 8 000 m haversine (pickup → delivery).
//
// ── NETWORK ARCHITECTURE ─────────────────────────────────────────────────────
//
//   Actor  (shared)     :  MLP  10 → 64 → 64 → 1  (sigmoid, Bernoulli policy)
//   Critic (centralized):  MLP  20 → 64 → 64 → 1  (linear, V(global_state))
//   PPO clip ε = 0.20   |  γ = 0.99  |  λ_GAE = 0.95  |  4 epochs per episode
//   GAE computed per-agent trajectory (avoids cross-agent bootstrap error).
//   Reward deferred: recorded as 0 at accept time, updated to task value on delivery.
//
// ── EXPECTED TIMING ──────────────────────────────────────────────────────────
//
//   OSM → GeoBox cache (one-time per city)
//     Tokyo / Kyoto      :  2 – 5 min each
//     LosAngeles         :  5 – 10 min
//     Comparison cities  :  3 – 8 min each
//
//   Training episode (warm A* cache after round 3–5)
//     ~15 – 40 s per episode (depends on city size and episode task volume)
//
//   Eval episode (mostly cached paths)
//     ~5 – 20 s per episode
//
//   Estimated wall-clock per seed    :  3 – 6 hours
//   Estimated wall-clock total (×3)  :  9 – 18 hours
//
//   To speed up a first run: reduce n_rounds to 10 and n_eval_episodes to 1
//   (edit the TrainingConfig block below). Full 3-seed run is for final results.
//
// ── OUTPUT FILES ─────────────────────────────────────────────────────────────
//
//   C:\ConflictualMAS\results\
//     seed0.csv, seed1.csv, seed2.csv   — per-episode metrics (train + eval)
//     summary.csv                       — cross-seed aggregated statistics
//     policy_seed0.bin, …               — serialized actor + critic weights
//
//   CSV columns: seed, episode, city, split (train/eval), method (MAPPO/Greedy/Random),
//     n_agents, tasks_appeared, tasks_completed, throughput_rate, latency_mean,
//     latency_per_agent, agent_utilisation, accept_rate, refuse_rate,
//     actor_loss, critic_loss, entropy, n_exp, wallclock_ms.
//
// ── PROGRESS MONITORING ──────────────────────────────────────────────────────
//
//   With verbose=true (default), one line is printed per training episode:
//     [sS rR CityName]  thr=0.52  acc=0.71  aloss=0.14  closs=0.09  ent=0.63  n=124  340ms
//
//   Healthy signs during training:
//     thr (throughput)  — should rise from ~0.2 to >0.5 over 50 rounds
//     acc (accept rate) — starts ~0.73 (bias), adapts to ≈ 0.5 – 0.8
//     aloss             — fluctuates; positive trend = policy improving
//     closs             — should decrease toward 0 over rounds
//     ent               — should slowly decrease (policy becomes less random)
//     n                 — number of PPO experiences; higher = more gradient signal
//
// ── TUNING ────────────────────────────────────────────────────────────────────
//
//   TrainingConfig (below):
//     n_rounds          — total rounds of multi-city training (default 50)
//     n_seeds           — independent statistical seeds (default 3)
//     eval_every        — eval checkpoint period in rounds (default 10)
//     n_eval_episodes   — eval episodes per city per mode (default 10)
//
//   EpisodeConfig (src/Training/EpisodeConfig.hpp):
//     phases            — adjust tasks_per_agent / n_agents / steps per phase
//     min_task_dist_m   — minimum pickup→delivery haversine distance (default 300 m)
//     speed_mps         — agent travel speed (default 5 m/s ≈ 18 km/h)
//
//   PPOParams (src/DMASforPD/Policy/ObjectiveDMPolicy.hpp):
//     lr_actor / lr_critic — learning rates (default 3e-4 / 1e-3)
//     epochs               — PPO gradient epochs per episode (default 4)
//
// =============================================================================

// =============================================================================
// OPTION A — Amazon Last Mile Single-Agent DbVNS Test
// =============================================================================
//
// OVERVIEW
//   Runs the DbVNS-PDP solver on 10 randomly-sampled routes from the
//   Amazon Last Mile Challenge 2021 dataset and compares results against
//   the ground-truth driver sequences.
//
// DATASET (required files)
//   src/AmazonDataset/~/.rc-cli/data/
//     model_apply_inputs/new_travel_times.json    — inter-stop travel times (s)
//     model_score_inputs/new_actual_sequences.json — driver visit order
//
//   The dataset contains 13 routes. Each route has:
//     - One depot  (stop at sequence position 0)
//     - 130–190 delivery stops with a complete travel-time matrix
//
// HOW TO RUN
//   1. Build  : cmake --build C:\ConflictualMAS\build --config Debug
//              (or: cd build && cmake --build . --config Debug)
//   2. Launch : build\Debug\main.exe  ->  type A
//
// WHAT IT DOES
//   For each route:
//     1. Identifies the depot (sequence position 0).
//     2. Pairs the remaining delivery stops consecutively into artificial
//        PDP tasks: (stop_0, stop_1), (stop_2, stop_3), ...
//        This lets DbVNS enforce the pickup-before-delivery constraint.
//     3. Builds an OperableEnvironment directly from the travel-time matrix
//        (no OSM graph, no GlobalMemory needed).
//     4. Runs one LocalSolutionAgent per sampled anchor (≤15), picks the
//        best sequence, and measures execution time.
//     5. Computes the full trip cost (depot → seq → last stop) and compares
//        it with the ground-truth driver cost.
//
// OUTPUT INTERPRETATION
//   ratio = DbVNS time / ground-truth time
//     < 1.0  — DbVNS beats the Amazon driver
//     = 1.0  — equal
//     > 1.0  — driver is better  (expected ~1.30–1.50 with default params,
//               because the PDP pairing is arbitrary and not task-optimal)
//
//   Architecture validation section verifies:
//     - PDP constraint : pickup always visited before its delivery  (must be 0)
//     - Unreachable edges : no sentinel costs in the solution        (must be 0)
//     - Sequence completeness : all stops covered exactly once
//
//   Execution scaling section measures ms/stop and the Pearson correlation
//   between route size and execution time (expected strong positive for O(N²)).
//
// TUNING  (src/AmazonDataset/AmazonTest.cpp → test_amazon_routes)
//   params.max_iterations     — VNS iterations per anchor  (default 15)
//   params.max_decompositions — branches explored per shake (default 2)
//   params.k_max              — shake depth factor          (default 3)
//   max_anchors               — delivery nodes tried as anchors (default 15)
//   Increasing these improves solution quality at the cost of execution time.
// =============================================================================
