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

// ── Episode difficulty scenario ───────────────────────────────────────────────
//
// Scales the per-phase task-arrival lambda and active-agent count to expose
// the policy to a wide range of agent-to-task ratios within a single training
// session. The MAPPO refusal action is meaningful only when the system is at
// or beyond capacity — under-saturated episodes give no gradient signal for
// "refuse this task". Sampling scenarios per episode produces buffers with
// mixed-difficulty experiences, which the global advantage normaliser then
// brings to comparable scales.
//
//   density_mult ~ Uniform[0.5, 2.5]   →  task arrival rate × this
//   agents_mult  ~ Uniform[0.5, 1.0]   →  active-agent count × this  (rounded)
//
// Stress eval: density_mult=2.0, agents_mult=0.6  →  ~4× over-saturation,
// most tasks cannot be delivered, the policy must learn to refuse.
struct EpisodeScenario {
    float       density_mult = 1.0f;  // multiplies phase lambdas
    float       agents_mult  = 1.0f;  // multiplies phase n_agents
    const char* label        = "normal";
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
    MAPPO,            // full system: shared-actor + centralised critic MAPPO
                      // (CTDE: centralised training, decentralised execution)
    IPPO,             // Independent PPO [de Witt+2020]: shared actor +
                      // per-agent local critics. Ablation between MAPPO's
                      // centralised critic and MAPPER's decentralised actor.
    MAPPER,           // decentralised RL baseline [Liu+IROS2020 adapted]:
                      // per-agent actor + per-agent critic + Evolutionary RL
                      // (selection/mutation of per-agent policy weights).
    Hybrid,           // frozen MAPPO base + per-agent online linear residual.
                      // Combines MAPPO's hot-start for new agents with
                      // MAPPER's specialisation, plus an explicit rollback
                      // safety net (residual shrinks/resets when fitness
                      // drops below fleet mean − threshold).
    TamAlwaysAccept,  // ablation: TAM routing only, always accept (no policy learning)
                      // isolates TAM routing contribution from MAPPO learning
    Greedy,           // always accept first idle agent (sequential scan)
    Random,           // accept with p=0.5 (random baseline)
    InsertionGreedy,  // accept if reward / insertion_cost > threshold (cost-aware heuristic)

    // ── SoTA-adapted baselines (related-works comparisons) ────────────────
    // Mapping to the three directions surveyed in our related works:
    //
    //   MAS-based (PIBT) [Okumura+2022]:
    //     PIBT     — priority-based: least-loaded eligible agent wins.
    //                Captures PIBT priority-coordination spirit, adapted to PDP
    //                where "priority" = load-balancing instead of grid conflicts.
    //
    //   Adaptation/Prevention (congestion-aware MAPP) [Liu, Saha, ...]:
    //     CongestionAware — picks agent whose route to pickup has the lowest
    //                       congestion-adjusted (dynamic) cost. Reuses
    //                       PDPServerMemory::refresh_dynamic_cost / TD-A*.
    //
    //   MCA [Chen+2021, ICRA] — Marginal-Cost Assignment adapted for L-GPDP:
    //     Scans ALL eligible agents and assigns to the one with the minimum
    //     TRUE marginal insertion cost (cheapest capacity-aware insertion over
    //     all valid (pos_P, pos_D) positions in the agent's current route).
    //     Unlike LaCAM (which ignores the existing route) and InsertionGreedy
    //     (which stops at the first bid above a threshold), MCA gives the
    //     globally optimal single-task assignment under real route costs.
    //     RMCA (regret-based variant) is equivalent to MCA in the online
    //     single-task-at-a-time setting and is therefore not implemented
    //     separately.
    //
    //   TrafficFlow [Chen+2024, AAAI] — GP-PIBT routing adapted for L-GPDP:
    //     Extends CongestionAware by using the congestion-adjusted (dynamic)
    //     cost for BOTH the current→pickup leg AND the pickup→delivery leg.
    //     CongestionAware only congestion-weights the first leg; TrafficFlow
    //     captures the full planned trip cost under current traffic conditions,
    //     mapping to the "guide path" spirit of GP-PIBT without requiring a
    //     grid map or explicit flow-refinement iterations.
    //
    //   PIBT-Matsui [Matsui+2025] — NOT adapted: requires a biconnected grid
    //     graph with dead-end aisles and physical agent collisions, none of
    //     which apply to OSM road-network delivery. The load-balancing intent
    //     is already covered by the PIBT baseline above.
    //
    //   Token Passing [Ma+2017, AAMAS] — decoupled MAPD adapted for L-GPDP:
    //     In the original TP, agents take turns holding a shared token; the
    //     token holder picks the task with the smallest h(loc(a), pickup) from
    //     the set of unassigned tasks whose pickup/delivery is not blocked by
    //     another agent's planned path. In our online single-task arrival
    //     setting, this reduces to: for each arriving task, assign it to the
    //     agent with minimum static A* distance from current_node to pickup.
    //     The collision-filter T' from the paper is trivial in our model
    //     (no physical collisions on OSM road networks). The well-formed
    //     instance hypothesis (non-task endpoints for parking) is not needed
    //     either — capacity is enforced by receive_task() and lifelong
    //     idle-rest is handled implicitly (idle agents stay at their last
    //     delivered node). TPTS (Task Swaps) is omitted: in online single-
    //     task arrival, the swap reduces to "pick the agent with min h" —
    //     identical to TP unless we batch multiple arriving tasks, which we
    //     don't. TP differs from LaCAM (which uses c_pu + c_del) by using
    //     only the pickup-leg cost, matching the paper's h-value criterion.
    //
    // LaCAM was dropped from the comparison set in favour of MAPPER, which
    // is more relevant in the RL category and gives a meaningful axis of
    // comparison against MAPPO's centralised-critic design.
    LaCAM,            // kept in the enum for source compatibility (unused branch)
    PIBT,
    CongestionAware,
    MCA,              // [Chen+2021] true marginal-cost assignment over all agents
    TrafficFlow,      // [Chen+2024] GP-PIBT spirit: full-trip congestion-aware cost
    TokenPassing,     // [Ma+2017]   decoupled MAPD: min h(current, pickup) selection

    // ── Double-Horizon Insertion [Mitrovic-Minic, Krishnamurti, Laporte 2004] ──
    //
    // Adapted to our capacity-constrained lifelong context:
    //   - Short-term horizon: first H steps of the agent's planned route
    //   - Long-term horizon : remainder of the route
    //
    // Cost formula for inserting task at (pos_P, pos_D) in agent's sequence:
    //   - If both insertion legs fall in short-term horizon:
    //         cost = route_length_increase   (= MCA's c1)
    //   - If insertion extends into long-term horizon:
    //         cost = (1 - α)·route_length_increase − α·local_slack_gain
    //     where local_slack_gain = slack_time_around(pos_D) (preserves future flex)
    //
    // Difference vs MCA (Chen+2021):
    //   MCA optimises pure marginal route cost (= c1 only).
    //   DoubleHorizon trades off route cost vs preserved slack for distant insertions
    //   — better long-term flexibility for accepting future tasks at the cost of
    //   slightly suboptimal short-term routing.
    DoubleHorizon
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
    //   scenario    : optional difficulty multipliers (density × agents). Default = 1.0/1.0.
    RunResult run(int city_index = 0, int num_cities = 1,
                  EpisodeScenario scenario = {});

    // ── Trace accessors (for offline analysis / rendering) ──────────────────
    // Read-only view of the GlobalMemory after a run() call. Use these to
    // extract task traces (pickup/delivery node IDs, timeline, agent_id, etc.)
    // and agent positions for SVG/CSV export from PlanningComparisonTest.
    const PDPGlobalMemory& memory() const { return memory_; }
    const std::vector<std::unique_ptr<DeliveryAgent>>& agents() const { return all_agents_; }

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
    int   n_accepted_     = 0;
    int   n_refused_      = 0;
    long  latency_sum_    = 0;
    int   latency_count_  = 0;
    int   active_sum_     = 0;
    int   active_steps_   = 0;

    // Running mean of reward_original / insertion_cost (m) for delivered tasks.
    // Feeds GlobalState::avg_efficiency so the critic gets a real route-quality
    // signal instead of the 0-placeholder previously used.
    double efficiency_sum_   = 0.0;
    int    efficiency_count_ = 0;

    // Congestion level (mean edge load) accumulated per step — used to compute
    // mean_congestion for the episode-level metric logged to CSV.
    double congestion_sum_   = 0.0;
    int    congestion_steps_ = 0;

    // Pickup→delivery traversal time (steps) for completed tasks.
    // mean_trip_steps = trip_sum_ / trip_count_ at episode end.
    long   trip_sum_         = 0;
    int    trip_count_       = 0;

    // Result of offering a task.
    //   agent_id  : winning agent (>= 0) or -1 if refused
    //   tam_owned : true if the TAM already performed assign_task + receive_task
    //               + commit_plan inside offer_to_agent (MAPPO/TamAlwaysAccept path).
    //               The caller must skip those operations to avoid double-bookkeeping.
    struct OfferResult { int agent_id; bool tam_owned; };

    // Offer task following the policy_mode protocol:
    //   MAPPO/TamAlwaysAccept → drive the TAM (incremental Dijkstra from pickup
    //                            and delivery; agents discovered via objective
    //                            nodes of their assigned tasks).
    //   Greedy/Random/IG      → simple capacity-based sequential scan.
    OfferResult offer_task(int task_id, float reward, float importance, int n_active,
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

    // Dry-run cheapest insertion: returns the minimum route-cost delta incurred
    // by inserting task into agent a's current sequence, without modifying it.
    // Mirrors DeliveryAgent::receive_task() capacity-aware (pos_P, pos_D) search.
    // Used by PolicyMode::MCA to pick the globally best agent per arriving task.
    float compute_marginal_cost(const DeliveryAgent& a, const PDPTask& task);

    // Double-horizon insertion cost [Mitrovic-Minic+2004 adapted].
    // Same (pos_P, pos_D) search as compute_marginal_cost, but the cost is
    // weighted by slack-time preservation when the insertion falls in the
    // long-term horizon (= positions beyond `short_horizon_pos` in the sequence).
    //
    //   short_horizon_pos: index threshold splitting the agent's sequence into
    //                       short-term (positions <  threshold) and long-term
    //                       (positions >= threshold) segments.
    //   alpha            : convex-combination weight in [0, 1].
    //                       alpha=0  → pure cheapest insertion (= compute_marginal_cost)
    //                       alpha=1  → maximise slack only (degenerate)
    //                       0.25     → paper's recommended value
    float compute_double_horizon_cost(const DeliveryAgent& a, const PDPTask& task,
                                       int short_horizon_pos, float alpha);

    // Process all due arrivals (arrival_step <= current_step).
    void process_arrivals(int current_step);

    // Assemble the 20-feature GlobalState for the centralised critic.
    GlobalState build_global_state(int step, int total_steps,
                                   float phase_label, float lambda,
                                   float city_norm, int n_active) const;
};

#endif // TRAINING_EPISODE_RUNNER_HPP
