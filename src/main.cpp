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
        cfg.cache_root       = cache_root;
        cfg.output_dir       = output_dir;
        cfg.n_rounds         = 50;      // 7 cities × 50 rounds = 350 train eps  ≈ 3–5 h
        cfg.n_seeds          = 1;       // single seed fits in one session; add more via load_policy
        cfg.n_eval_episodes  = 5;       // per city × mode (3 checkpoints × 7 × 3 × 5 = 315 eps ≈ 1 h)
        cfg.eval_every       = 25;      // eval at round 25, 50 + final
        cfg.save_policy      = true;    // checkpoint saved → rerun with load_policy=true to extend
        cfg.verbose          = true;

        // To extend from a saved checkpoint (add more seeds / rounds):
        //   cfg.load_policy  = true;
        //   cfg.policy_path  = output_dir + "/policy_seed0.bin";

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
// OPTION T - MAPPO Multi-City Training
// =============================================================================
//
// OVERVIEW
//   Trains a shared MAPPO policy (actor + centralized critic) on the Lifelong
//   General Pickup-and-Delivery Problem across 7 real city road graphs.
//   Each agent independently decides to accept or refuse task offers using a
//   shared 10->64->64->1 MLP (sigmoid), trained via PPO with centralised critic.
//
// =============================================================================
// STEP-BY-STEP TUTORIAL
// =============================================================================
//
// STEP 1 -- Download OSM regional files (no extra tool needed)
//   The code uses libosmium (already linked) to parse .pbf files with a bbox
//   filter applied at runtime. No pre-extraction or osmium-tool required.
//
//   Download these 6 files from geofabrik.de and place them in:
//     C:\ConflictualMAS\src\maps\
//
//   File                          URL (geofabrik.de)                       Covers
//   kanto-latest.osm.pbf          /asia/japan/kanto                        Tokyo + Kyoto
//   kyushu-latest.osm.pbf         /asia/japan/kyushu                       Fukuoka
//   california-latest.osm.pbf     /north-america/us/california             Los Angeles
//   new-york-latest.osm.pbf       /north-america/us/new-york               New York
//   ile-de-france-latest.osm.pbf  /europe/france/ile-de-france             Paris
//   great-britain-latest.osm.pbf  /europe/great-britain                    London
//
//   Note: kanto-latest.osm.pbf is already at src/maps/ (used by smoke test).
//         Only 5 additional downloads are needed.
//
// STEP 2 -- Build
//   cmake --build C:\ConflictualMAS\build --config Debug
//
// STEP 3 -- Run the smoke test (sanity check before the full run, ~5 min)
//   build\Debug\main.exe  ->  type S
//   Expected: === PASS ===
//
// STEP 4 -- Launch training
//   build\Debug\main.exe  ->  type T
//
//   First run: each city .pbf is parsed and cached as a .json GeoBox in
//     C:\ConflictualMAS\data\cache\
//   Parsing is slow (~5-15 min per city, ~60-90 min total for all 7) because
//   libosmium reads the full regional file and filters by bbox.
//   All subsequent runs load from cache and skip parsing entirely.
//
//   Expected console flow:
//     Loading 7 cities from C:\ConflictualMAS\data\cache
//       [Load] Tokyo    <- OSM (kanto-latest.osm.pbf, parsing...)   <- first run
//       [Load] Tokyo    <- cache                                     <- next runs
//       ...
//     All cities loaded.
//
//     Seed 0  (rng=42)
//       [s0 r0 Tokyo]      thr=0.18  acc=0.73  aloss=...  kl=...  ep=.../10  n=...  ...ms
//       [s0 r0 Kyoto]      thr=0.21  ...
//       [s0 r0 Fukuoka]    ...
//       [s0 r0 LosAngeles] ...
//       [s0 r0 NewYork]    ...
//       [s0 r0 Paris]      ...
//       [s0 r0 London]     ...
//       -- Eval @ round 25 --
//       -- Eval @ round 50 --
//       -- Final Eval --
//       [Checkpoint] saved to C:\ConflictualMAS\results\policy_seed0.bin
//     Seed 0 done -- 665 episodes.
//     Training complete. Results in C:\ConflictualMAS\results
//
// STEP 5 -- Check results
//   C:\ConflictualMAS\results\
//     seed0.csv          -- all episode metrics (open in Excel / plot with Python)
//     summary.csv        -- aggregated stats
//     policy_seed0.bin   -- trained weights
//
//   Key columns to watch (split=train rows):
//     throughput_rate  -- rises from ~0.2 toward >0.5 as policy converges
//     accept_rate      -- moves away from 0.73 flat; settles ~0.5-0.8
//     critic_loss      -- should decrease toward 0 over rounds
//
// STEP 6 -- Extend training (optional, for publication-quality results)
//   Uncomment in the TrainingConfig block below:
//     cfg.load_policy = true;
//     cfg.policy_path = output_dir + "/policy_seed0.bin";
//   Run again -> type T.
//   Each additional 50-round seed takes ~3-4 h.  3 seeds recommended for stats.
//
// =============================================================================
// TECHNICAL REFERENCE
// =============================================================================
//
// -- TRAINING PROTOCOL (single 5-6 h session) ---------------------------------
//
//   Seeds             :  1  (add more via load_policy; see Step 6)
//   Rounds per seed   :  50  (one episode per city per round, round-robin)
//   Train cities      :  7  (all cities, shared policy)
//   Train episodes    :  50 x 7 = 350
//   Eval checkpoints  :  rounds 25 and 50 + final  =  3 per seed
//   Eval episodes     :  3 x 7 cities x 3 modes x 5 eps  =  315
//   Total episodes    :  665
//
//   Timing (1 seed, after first-run GeoBox cache is built):
//     Training  :  350 eps x ~25 s avg  ~  2.5 h
//     Eval      :  315 eps x ~10 s avg  ~  52 min
//     Total     :  ~3.5 h
//
// -- EPISODE STRUCTURE (pseudo-day, 3 600 steps) ------------------------------
//
//   Phase       Steps   Agents   Tasks/agent   Lambda     Hot zones
//   Low          1000    8-10       ~100        ~0.90         4
//   Medium       1500   10-13       ~200        ~1.53         6
//   High         1100   13-15       ~250        ~3.18         8
//
//   Poisson arrivals (lambda > 1 valid); fleet ramps linearly within each phase.
//   Hot zones resampled at phase boundaries. Task dist: 300 m to 8 000 m.
//
// -- NETWORK ARCHITECTURE -----------------------------------------------------
//
//   Actor  (shared)      :  MLP  10 -> 64 -> 64 -> 1  (sigmoid, Bernoulli policy)
//   Critic (centralised) :  MLP  20 -> 64 -> 64 -> 1  (linear, V(global_state))
//   PPO clip eps=0.20  |  gamma=0.99  |  lam_GAE=0.95  |  10 epochs / episode
//   Mini-batch SGD + grad norm clipping (max_norm=0.5) + KL early stop (0.01).
//
// -- CONVERGENCE MONITORING ---------------------------------------------------
//
//   Verbose log (one line per training episode):
//     [s0 r5 Tokyo]  thr=0.41  acc=0.68  aloss=0.12  closs=0.08  ent=0.61
//                    kl=0.007  cf=0.18  ep=10/10  n=312  420ms
//
//   thr rises, ent falls, closs falls, kl < 0.01 -> policy is converging.
//   ep < 10/10 means KL early stop triggered (normal and healthy).
//
// -- TUNING -------------------------------------------------------------------
//
//   n_rounds, eval_every, n_eval_episodes  -- in the TrainingConfig block below
//   phases, min_task_dist_m, speed_mps    -- src/Training/EpisodeConfig.hpp
//   lr_actor, lr_critic, epochs           -- src/DMASforPD/Policy/ObjectiveDMPolicy.hpp
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
