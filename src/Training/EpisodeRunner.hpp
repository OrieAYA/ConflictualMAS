#ifndef TRAINING_EPISODE_RUNNER_HPP
#define TRAINING_EPISODE_RUNNER_HPP

#include "EpisodeConfig.hpp"
#include "EpisodeGenerator.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include "DMASforPD/DeliveryAgent/DeliveryAgent.hpp"
#include "DMASforPD/Policy/ObjectiveDMPolicy.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <array>
#include <chrono>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// ── Result returned from a single episode run ─────────────────────────────────
struct RunResult {
    ComparisonMetrics                    metrics;
    ObjectiveDMPolicy::TrainingStats     train_stats; // zeroed when train_mode=false
    long long                            wallclock_ms = 0;
};

// ════════════════════════════════════════════════════════════════════════════
// EpisodeRunner
// ════════════════════════════════════════════════════════════════════════════
//
// Runs one training/evaluation episode end-to-end:
//
//   1. Generates a ScheduledTask stream via EpisodeGenerator.
//   2. Simulates step-by-step: inject tasks → offer to agents → move agents.
//   3. Collects MAPPO experiences via ObjectiveDMPolicy::shared().
//   4. Calls train_epoch() at episode end when train_mode is true.
//
// Movement model (single-hop for training efficiency):
//   Each agent leg (current → pickup, pickup → delivery) is a single
//   timed hop.  The agent arrives at the objective node at:
//     step + ceil(path_cost / speed_mps)
//   Intermediate road nodes are not simulated, keeping each episode O(steps).
//
// Allocation: direct offer loop (no TAM).  Each arriving task is offered
//   to idle agents in sequence; the first score >= 0.5 wins.
//   Refused tasks are counted but not re-offered.
//
// One EpisodeRunner instance = one episode.  Reuse of the GeoBox/Pathfinder
//   across episodes is the caller's responsibility.
// How the runner decides to accept/reject tasks.
// Random and Greedy do NOT call try_accept_task() and do NOT write to the
// training buffer — safe for use as baselines without contaminating MAPPO.
enum class PolicyMode {
    MAPPO,           // shared actor MLP (default; records experiences)
    Greedy,          // always accept first idle agent (upper-bound throughput proxy)
    Random,          // accept with p=0.5 (random baseline)
    InsertionGreedy  // SOTA-style: accept if reward / insertion_cost > threshold
                     // (myopic-optimal proxy, inspired by CBBA / threshold-bid auctions)
};

class EpisodeRunner {
public:
    bool       train_mode   = true;
    PolicyMode policy_mode  = PolicyMode::MAPPO;

    EpisodeRunner(const EpisodeConfig& cfg,
                  GeoBox&             geo_box,
                  Pathfinder&         pathfinder,
                  uint32_t            seed = 42);

    ~EpisodeRunner();

    // Run one complete episode.
    //   city_index  : index of this city in the full registry (for city_id_norm feature).
    //   num_cities  : total number of cities (for normalisation).
    RunResult run(int city_index = 0, int num_cities = 1);

private:
    const EpisodeConfig& cfg_;
    PDPGlobalMemory      memory_;
    EpisodeGenerator     gen_;

    std::vector<std::unique_ptr<DeliveryAgent>> all_agents_;

    std::vector<std::array<float, kGlobSz>> global_states_;
    std::mt19937                            rng_; // for Random policy mode

    // task_id → index in ObjectiveDMPolicy buffer for the accept decision.
    // Populated when a task is accepted under MAPPO; used at delivery to call
    // update_reward() with the actual completion reward.
    std::unordered_map<int, int> task_accept_buf_idx_;

    // ── Arrival queue ──────────────────────────────────────────────────────
    //
    // One entry per scheduled edge end.
    //   is_objective = false  → intermediate road node (advance cursor, next edge)
    //   is_objective = true   → objective node (pickup or delivery reached)
    struct ScheduledArrival {
        int  agent_id;
        int  task_id;
        bool is_pickup;
        bool is_objective;   // true = last edge of this leg
        int  arrival_step;
    };
    std::vector<ScheduledArrival> arrivals_;

    // Running accumulators reset in run().
    int  n_accepted_     = 0;
    int  n_refused_      = 0;
    long latency_sum_    = 0;
    int  latency_count_  = 0;
    int  active_sum_     = 0;
    int  active_steps_   = 0;

    // Offer task to the first available agent (Idle or Active with capacity).
    // Returns accepting agent_id, or -1 if refused by all.
    int offer_task(int task_id, float reward, float importance, int n_active,
                   const std::array<float, kGlobSz>& gs);

    // ── Edge-by-edge movement ──────────────────────────────────────────────

    // Begin a new leg: fetch path from→to, load into agent.edge_cursor,
    // set local_memory.current_path, pre-fetch next_path, schedule first edge.
    // Returns arrival_step of the first edge (or single hop if no cached path).
    int start_leg(int agent_id, int task_id, bool is_pickup,
                  osmium::object_id_type from,
                  osmium::object_id_type to,
                  int current_step);

    // Schedule the next edge in the agent's active cursor.
    // Must be called with a valid edge_cursor that still has_more_edges().
    // Returns arrival_step of that edge.
    int schedule_next_edge(int agent_id, int current_step);

    // Called when an agent arrives at an objective (pickup or delivery).
    // Fires the appropriate event, promotes next_path, starts the next leg.
    void on_objective_reached(int agent_id, int task_id, bool is_pickup, int current_step);

    // Haversine-based fallback cost (road-factor adjusted) when A* fails.
    float fallback_cost(osmium::object_id_type from,
                        osmium::object_id_type to) const;

    // Process all due arrivals (arrival_step <= current_step).
    void process_arrivals(int current_step);

    // Assemble the 20-feature GlobalState for the centralised critic.
    GlobalState build_global_state(int step, int total_steps,
                                   float phase_label, float lambda,
                                   float city_norm, int n_active) const;
};

#endif // TRAINING_EPISODE_RUNNER_HPP
