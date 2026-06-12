#ifndef TRAINING_EPISODE_RUNNER_HPP
#define TRAINING_EPISODE_RUNNER_HPP

#include "EpisodeConfig.hpp"
#include "EpisodeGenerator.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include "DMASforPD/DeliveryAgent/DeliveryAgent.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "Environment/Congestion/GhostTrafficController.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include <array>
#include <chrono>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// ── Result returned from a single episode run ─────────────────────────────────
struct RunResult {
    ComparisonMetrics metrics;
    TrainingStats     train_stats;   // zeroed when train_mode=false
    long long         wallclock_ms = 0;
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

    // ── Background traffic profile (Option M scenario diversification) ──────
    // When EpisodeConfig::enable_ghost_traffic is true, EpisodeRunner spins
    // up a GhostTrafficController seeded with this profile and the per-city
    // ghost_n_max budget. Default Flat = neutral / off-policy regimes that
    // do not want background congestion. Sampler in MultiCityTrainer picks
    // among the 5 profiles for diversified training.
    CongestionProfile congestion_profile = CongestionProfile::Flat;
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
    // ── Learning policies (paper §4.3.3) — TAM + DbVNS pipeline ───────────
    MAPPO,            // shared actor + centralised critic (CTDE)
    IPPO,             // shared actor + shared LOCAL critic [de Witt+2020]
    MAPPER,           // per-agent actor/critic + evolutionary selection [Liu+2020]
    FaithfulMAPPER,   // legacy alias of MAPPER (kept for old configs/CSV labels)
    Hybrid,           // frozen MAPPO base + bounded per-agent online residual

    // ── Policy-level baseline — same TAM + DbVNS pipeline, scoring swapped ─
    RMCA,             // marginal-cost insertion scorer [Chen+2021], deterministic

    // ── Ablations / sanity baselines ───────────────────────────────────────
    TamAlwaysAccept,  // TAM routing only, every candidate bids with mu=1
    Greedy,           // first capable agent (sequential scan)
    Random,           // accept with p=0.5
    InsertionGreedy,  // bid = reward*importance/insertion_cost > threshold

    // ── Allocation-level baselines — REPLACE the TAM + policy entirely ─────
    TokenPassing,     // argmin h(loc, pickup) on free agents [Ma+2017]
    PIBT,             // least-loaded priority allocation (contributor-owned
                      // adaptation — do not modify without coordinating)

    // ── Planning-level modes (planning comparison; allocation = MCA argmin) ─
    MCA,              // cheapest insertion + anytime LNS [Chen+2021 Alg.3]
    DoubleHorizon,    // slack-preserving insertion [Mitrovic-Minic+2004]
    DbVNS,            // forward DbVNS-PDP global replan (the paper's planner)
    ALNS              // destroy-repair ALNS [Ropke & Pisinger 2006]
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
    //   city_index   : index of this city in the full registry (for city_id_norm feature).
    //   num_cities   : total number of cities (for normalisation).
    //   scenario     : optional difficulty multipliers (density × agents). Default = 1.0/1.0.
    //   episode_seed : optional seed for deterministic episode replay. When > 0,
    //                  resets the internal RNGs (gen_, rng_, ghost controller seed)
    //                  so that two policies evaluated on the same (city,
    //                  episode_index, scenario) face identical task streams,
    //                  ghost traffic, and hetero-capacity draws — making the
    //                  policy-vs-policy comparison fair. When 0 (default),
    //                  RNG state advances normally (legacy behaviour).
    RunResult run(int city_index = 0, int num_cities = 1,
                  EpisodeScenario scenario = {},
                  uint32_t episode_seed = 0,
                  const struct SharedEpisodeSetup* setup = nullptr);

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

    std::mt19937                            rng_; // for Random policy mode

    // Active learning policy for the current run (null for non-policy modes
    // and for RMCA, which scores deterministically without a buffer).
    IBidPolicy* active_policy_ = nullptr;

    // Optional synthetic background traffic. Constructed once, reset per
    // episode if cfg_.enable_ghost_traffic. No-op when the flag is off so
    // existing options (T, N, X, Y) keep their pre-change behaviour exactly.
    GhostTrafficController ghost_traffic_;

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

    // Appearance→pickup arrival (steps) — response/wait time before first move.
    // mean_wait_steps = wait_sum_ / wait_count_ at episode end.
    long   wait_sum_         = 0;
    int    wait_count_       = 0;

    // A* road distance pickup→delivery for completed tasks (metres).
    // mean_road_pd_m = road_pd_sum_ / road_pd_count_ at episode end.
    double road_pd_sum_      = 0.0;
    int    road_pd_count_    = 0;

    // Congestion sampled ONLY at accept/refuse decision steps (one sample
    // per offered task). Lets us report what the policy actually "saw" at
    // decision time, distinct from mean_congestion (averaged across all
    // simulation steps).
    double congestion_at_decision_sum_   = 0.0;
    int    congestion_at_decision_count_ = 0;

    // Selectivity discrimination — track the importance distribution that
    // the policy chose to ACCEPT vs REFUSE. A real selective policy should
    // show mean_imp_accepted > mean_imp_refused.
    double imp_accepted_sum_ = 0.0;
    int    imp_accepted_n_   = 0;
    double imp_refused_sum_  = 0.0;
    int    imp_refused_n_    = 0;

    // Context-bucketed acceptance: split decisions on whether the network
    // was congested (mean_load_now >= 1.0) at decision time. accept_rate
    // delta across buckets quantifies congestion sensitivity.
    int    accepts_high_cong_ = 0;
    int    refuses_high_cong_ = 0;
    int    accepts_low_cong_  = 0;
    int    refuses_low_cong_  = 0;

    // TAM efficiency accumulators (one sample per offer_task call). For TAM
    // modes the values come from the TAM getters; for SoTA baselines they
    // default to n_active (uniform sentinel — the baseline scanned everyone).
    long   tam_agents_offered_sum_     = 0;
    long   tam_recall_rounds_sum_      = 0;
    long   tam_candidates_scored_sum_  = 0;
    int    tam_offer_samples_          = 0;

    // Filled by offer_task() on each call; read by the surrounding run loop
    // after the call to accumulate per-episode TAM efficiency stats.
    struct LastOfferStats {
        int       agents_offered    = 0;
        int       recall_rounds     = 0;
        int       candidates_scored = 0;
        int       tam_dijkstra_steps = 0;   // TAM step() loop iterations
        long long allocation_time_us = 0;   // wallclock of the offer_task call
        long long path_time_us       = 0;   // time spent in get_or_compute_path within this offer
        long long pure_alloc_time_us = 0;   // = allocation_time_us - path_time_us (TAM+policy only)
        // For oracle metric: marginal costs of every n_active agent for THIS
        // task, in the PRE-allocation state (TAM had not committed yet).
        std::vector<float> pre_marginal_costs;
    };
    LastOfferStats last_offer_stats_;

    // Effective active fleet for the current episode = the provisioned agent
    // pool scaled by scenario.agents_mult (matches SolverRunner's
    // setup->n_active_agents). Used as the agent_utilisation denominator AND
    // the logged n_agents — NOT cfg_.max_agents() (the nominal peak, which
    // ignored agents_mult and under-reported the fleet by 10× in over_fleet).
    int episode_fleet_size_ = 0;

    // Selection-intelligence accumulators (delivery quality, not just count).
    double value_appeared_sum_  = 0.0;
    double value_delivered_sum_ = 0.0;
    double value_refused_sum_   = 0.0;

    // Edge-traversal impact accumulators (real BPR impact on agents).
    double bpr_along_route_sum_           = 0.0;
    int    bpr_along_route_count_         = 0;
    double time_lost_to_congestion_sum_   = 0.0;   // in steps
    int    n_traversals_in_jam_           = 0;

    // Allocation-optimality accumulator (vs MCA full-scan oracle).
    double marginal_ratio_sum_   = 0.0;
    int    marginal_ratio_count_ = 0;

    // RMCA(r) regret accumulators [Chen et al. 2021, eq.13/14/16]. Populated
    // only when policy_mode == RMCA: per allocation we read mc(k1)=cheapest and
    // mc(k2)=2nd-cheapest insertion cost over the eligible agents (from
    // last_offer_stats_.pre_marginal_costs) and accumulate the relative regret
    // mc(k2)/mc(k1). Means are exported in finalize().
    double rmca_regret_sum_  = 0.0;   // Σ relative regret  (eq.16)
    double rmca_mc_k1_sum_   = 0.0;   // Σ cheapest insertion cost (m)
    double rmca_mc_k2_sum_   = 0.0;   // Σ 2nd-cheapest insertion cost (m)
    int    rmca_regret_count_ = 0;

    // Accumulate one RMCA(r) regret sample from the pre-allocation marginal
    // costs of every eligible agent. No-op unless policy_mode == RMCA.
    void accumulate_rmca_regret(const std::vector<float>& pre_marginal_costs);

    // Temporal-complexity accumulators (allocation cost only).
    long long allocation_time_us_sum_ = 0;
    int       allocation_time_count_  = 0;
    long long tam_dijkstra_steps_sum_ = 0;
    int       tam_dijkstra_count_     = 0;
    // Pure TAM+policy allocation time (excludes path-cache lookups), summed
    // across offers. Mean = sum / allocation_time_count_. Plus per-episode
    // total path-compute time (read once at finalize from PDPServerMemory).
    long long pure_alloc_time_us_sum_ = 0;
    long long path_compute_time_us_ep_ = 0;

    // Network-level congestion impact accumulators (per-step sampling).
    int     peak_load_episode_   = 0;          // max load_now seen this episode
    long    overlap_edges_sum_   = 0;          // sum of n_edges_load_ge(2) per step
    int     overlap_steps_       = 0;          // sample count for above
    double  load_now_sum_sq_     = 0.0;        // for variance: Σ load²
    double  load_now_sum_lin_    = 0.0;        //                  Σ load
    int     load_now_n_          = 0;          //                  N
    double  route_exposure_sum_  = 0.0;        // Σ load_now on in-transit agents' edges
    int     route_exposure_n_    = 0;          //   sample count

    // Result of offering a task.
    //   agent_id  : winning agent (>= 0) or -1 if refused / deferred
    //   tam_owned : true if the TAM already performed assign_task + receive_task
    //               + commit_plan inside offer_to_agent (MAPPO/TamAlwaysAccept path).
    //               The caller must skip those operations to avoid double-bookkeeping.
    //   deferred  : true only in TAM multi-candidate Format B when every
    //               candidate scored < 0.5. The task is neither accepted nor
    //               genuinely refused — the caller must re-offer it later
    //               instead of counting it. Always false otherwise.
    struct OfferResult { int agent_id; bool tam_owned; bool deferred = false; };

    // Deferred-task retry queue (TAM multi-candidate Format B only). Empty in
    // every other mode, so the retry loop is a no-op there.
    struct RetryEntry { int task_id; int retry_step; };
    std::vector<RetryEntry> retry_queue_;

    // Post-allocation bookkeeping shared by the new-task-arrival loop and the
    // retry loop: resolves was_idle, records the policy buffer index, and
    // starts the pickup leg if the agent was idle.
    void commit_accepted_task(int task_id, int winner, bool tam_owned,
                              osmium::object_id_type pickup_node, int step);

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
