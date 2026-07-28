#ifndef TRAINING_EPISODE_RUNNER_HPP
#define TRAINING_EPISODE_RUNNER_HPP

#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include "TrainingEvaluation/StructuresParam/ScenarioConfig.hpp"
#include "TrainingEvaluation/Run/Metrics.hpp"
#include "TrainingEvaluation/Run/RewardShaping.hpp"
#include "Environment/Structure/Episode.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include "DMASforPD/Policy/BidPolicy.hpp"
#include "DMASforPD/Policy/MovementPolicy.hpp"
#include "DMASforPD/Prediction/Lsm.hpp"
#include "Environment/Simulation/GhostTrafficController.hpp"
#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// Result of a single episode run.
struct RunResult {
    ComparisonMetrics metrics;
    TrainingStats     train_stats;   // zeroed when train_mode=false
    long long         wallclock_ms = 0;
};

// How an arriving task is accepted / scored.
enum class PolicyMode {
    // Learning policies (paper §4.3) — TAM + DbVNS pipeline
    MAPPO,            // shared actor + centralised critic [Yu+2022]
    IPPO,             // shared actor + shared local critic [de Witt+2020]
    MAPPER,           // per-agent actor/critic + evolutionary selection [Liu+2020]
    Hybrid,           // frozen MAPPO base + bounded per-agent online residual

    // Policy-level baseline — same TAM + DbVNS pipeline, scoring swapped
    RMCA,             // marginal-cost insertion scorer [Chen+2021], deterministic

    // Sanity baselines (tests only, not in the paper evaluation)
    TamAlwaysAccept,  // TAM routing only, every candidate bids with mu=1
    Greedy,           // first capable agent (sequential scan)

    // Allocation-level baseline — replaces the TAM + policy entirely
    TokenPassing,     // argmin h(loc, pickup) on free agents [Ma+2017]

    // Planning-level modes (allocation = cheapest marginal insertion)
    DoubleHorizon,    // slack-preserving insertion [Mitrovic-Minic+2004]
    DbVNS,            // forward DbVNS-PDP global replan (the paper's planner)
    ALNS              // destroy-repair ALNS [Ropke & Pisinger 2006]
};

// ════════════════════════════════════════════════════════════════════════════
// EpisodeRunner — runs one training/evaluation episode end-to-end.
// ════════════════════════════════════════════════════════════════════════════
//
// The ScheduledTask stream is generated up-front (EpisodeGenerator or
// SharedEpisodeSetup); each step injects due arrivals, offers them to agents
// (TAM + policy, or a baseline rule) and advances agents edge-by-edge along
// the road graph. In train_mode the active policy buffers experiences and
// train_round() runs at episode end. The caller owns and reuses the GeoBox.
//
// Implementation is split across three files:
//   Runner.cpp           — episode lifecycle (prepare, main loop, rewards)
//   RunnerAllocation.cpp — task offer protocol + insertion-cost helpers
//   RunnerMovement.cpp   — edge-by-edge movement + arrival processing
class EpisodeRunner {
public:
    bool       train_mode   = true;
    PolicyMode policy_mode  = PolicyMode::MAPPO;

    EpisodeRunner(const EpisodeConfig& cfg,
                  GeoBox&             geo_box,
                  uint32_t            seed = 42);

    ~EpisodeRunner();

    // Run one complete episode. episode_seed > 0 re-anchors every RNG so the
    // same (city, scenario, episode) slot replays identically across policy
    // modes; a SharedEpisodeSetup additionally makes the environment
    // byte-identical with the standalone SoTA solvers (CA, HAPC).
    RunResult run(int city_index = 0, int num_cities = 1,
                  EpisodeScenario scenario = {},
                  uint32_t episode_seed = 0,
                  const struct SharedEpisodeSetup* setup = nullptr);

    // Weight-annealing clock, published by the trainer before each run().
    // Never set (-1) => annealed weights stay at w0 (eval / tests unchanged).
    void set_global_episode(int e) { global_episode_ = e; }

    // Drop the memoised path + search caches. Called by the trainer once a
    // grid point is done so idle runners retain no episode-sized memory.
    void release_episode_memory();

    // Read-only views for offline analysis / rendering after run().
    const PDPGlobalMemory& memory() const { return memory_; }
    const std::vector<std::unique_ptr<DeliveryAgent>>& agents() const { return all_agents_; }

    // Movement-policy episode counters (cfg.use_movement_policy runs only).
    struct MovementEpisodeStats {
        int   n_decisions = 0;
        int   n_replans   = 0;
        int   n_adopted   = 0;
        float gain_sum    = 0.f;
        TrainingStats train;
    };
    const MovementEpisodeStats& movement_stats() const { return mv_stats_; }

    // LSM episode counters (cfg.use_lsm runs only).
    struct LsmEpisodeStats {
        int       n_ticks          = 0;
        int       n_learn          = 0;
        double    mse_sum          = 0.0;
        long long alert_cells_sum  = 0;
        long long agent_alerts_sum = 0;

        float mse_mean() const {
            return n_learn > 0 ? static_cast<float>(mse_sum / n_learn) : 0.f;
        }
        float alert_cells_mean() const {
            return n_ticks > 0
                ? static_cast<float>(alert_cells_sum) / n_ticks : 0.f;
        }
        float agent_alerts_mean() const {
            return n_ticks > 0
                ? static_cast<float>(agent_alerts_sum) / n_ticks : 0.f;
        }
    };
    const LsmEpisodeStats& lsm_stats() const { return lsm_stats_; }

private:
    const EpisodeConfig& cfg_;
    PDPGlobalMemory      memory_;
    EpisodeGenerator     gen_;

    std::vector<std::unique_ptr<DeliveryAgent>> all_agents_;

    std::mt19937 rng_;

    // Active learning policy for the current run (null for non-policy modes
    // and for RMCA, which scores deterministically without a buffer).
    IBidPolicy* active_policy_ = nullptr;

    // Synthetic background traffic; no-op when cfg_.enable_ghost_traffic is off.
    GhostTrafficController ghost_traffic_;

    // task_id -> policy buffer index of the accept decision. Used to credit
    // pickup/delivery rewards and the unfinished penalty to that entry.
    std::unordered_map<int, int> task_accept_buf_idx_;

    // One entry per scheduled edge end; is_objective marks the last edge of a
    // leg (pickup or delivery reached).
    struct ScheduledArrival {
        int  agent_id;
        int  task_id;
        bool is_pickup;
        bool is_objective;
        int  arrival_step;
    };
    std::vector<ScheduledArrival> arrivals_;

    // Per-episode metric accumulators (Metrics.hpp Part 1 — collection);
    // aggregated at episode end by finalize_episode_metrics() (Part 2).
    EpisodeMetricsCollector acc_;

    // ep_seed of the last prepared run — path cache purged when it changes.
    uint32_t last_episode_seed_ = 0;

    MovementEpisodeStats mv_stats_;

    // ── LSM prediction/signaling state (RunnerPrediction.cpp) ───────────────
    std::deque<std::pair<int, std::vector<float>>> lsm_hist_;
    std::unordered_map<int, float>                 lsm_alert_;
    LsmEpisodeStats                                lsm_stats_;

    void lsm_tick(int step, int total_steps, float lambda, int n_active);

    // Reward shaping v2 state (RewardShaping.hpp).
    RewardScales scales_;               // static scales (1 when flag off)
    int    global_episode_ = -1;        // annealing clock (-1 = off)
    double cins_sum_    = 0.0;          // running mean of accepted insertion costs
    int    cins_n_      = 0;

    float rs_w(float w0, float floor_frac) const {
        return annealed_w(w0, floor_frac, global_episode_,
                          cfg_.rs_e_anneal, cfg_.rs_weight_annealing);
    }

    // Filled by offer_task(); read back by the run loop for TAM stats.
    struct LastOfferStats {
        int       agents_offered    = 0;
        int       recall_rounds     = 0;
        int       candidates_scored = 0;
        int       tam_dijkstra_steps = 0;
        long long allocation_time_us = 0;   // wallclock of the offer_task call
        long long path_time_us       = 0;   // get_or_compute_path share
        long long pure_alloc_time_us = 0;   // allocation - path (TAM+policy only)
        // Marginal cost of every active agent for THIS task, sampled BEFORE
        // allocation (full-scan oracle metric).
        std::vector<float> pre_marginal_costs;
    };
    LastOfferStats last_offer_stats_;

    // Active fleet for this episode (provisioned pool × scenario.agents_mult,
    // same value as SolverRunner's n_active_agents). Logged as n_agents and
    // used as the agent_utilisation denominator.
    int episode_fleet_size_ = 0;

    // agent_id -1 = refused / deferred. tam_owned: the TAM already performed
    // assign_task + receive_task + commit_plan (the caller must skip those).
    // deferred: TAM Format B only — the task must be re-offered later.
    struct OfferResult { int agent_id; bool tam_owned; bool deferred = false; };

    // Deferred-task retry queue (TAM Format B only; empty in every other mode).
    struct RetryEntry { int task_id; int retry_step; };
    std::vector<RetryEntry> retry_queue_;

    // ── Episode lifecycle (Runner.cpp) ──────────────────────────────────────

    // Resolve the active policy, clear buffers and metrics, propagate TAM
    // parameters + planning flags, reset agents and fleet size.
    void prepare_run(const EpisodeScenario& scenario, uint32_t episode_seed,
                     const SharedEpisodeSetup* setup);

    void setup_ghost_traffic(const EpisodeScenario& scenario,
                             const SharedEpisodeSetup* setup,
                             int total_steps, int city_index, bool ghost_on);

    // Shared offer core (arrival + retry loops): drives offer_task, records
    // instrumentation + outcome metrics, commits on acceptance. Deferred
    // handling stays with the caller.
    OfferResult offer_with_metrics(PDPTask& task, int step, int n_active,
                                   const std::array<float, kGlobSz>& gs);

    void retry_deferred_tasks(int step, int total_steps, int n_active,
                              const std::array<float, kGlobSz>& gs);

    // Per-step sampling: network congestion, route exposure, fleet activity.
    void sample_step_state(int step, int n_active);

    // Assemble the 20-feature GlobalState for the centralised critic.
    GlobalState build_global_state(int step, int total_steps,
                                   float phase_label, float lambda,
                                   float city_norm, int n_active) const;

    // ── Allocation (RunnerAllocation.cpp) ───────────────────────────────────

    // Offer a task following the policy_mode protocol (TAM pipeline for
    // RL/RMCA/TamAlwaysAccept, direct selection rules for the baselines).
    OfferResult offer_task(int task_id, float reward, float importance, int n_active,
                           const std::array<float, kGlobSz>& gs);

    // Post-allocation bookkeeping: resolves was_idle, then starts the pickup
    // leg if the agent was idle.
    void commit_accepted_task(int task_id, int winner, bool tam_owned,
                              osmium::object_id_type pickup_node, int step);

    // Read-only cheapest-insertion delta of task into agent a's sequence
    // (mirrors DeliveryAgent::receive_task). Allocation rule for the
    // DbVNS/ALNS planning modes and source of the full-scan oracle metric.
    float compute_marginal_cost(const DeliveryAgent& a, const PDPTask& task);

    // Double-horizon insertion cost [Mitrovic-Minic+2004]: cheapest insertion
    // weighted by slack preservation for long-horizon positions (alpha=0.25).
    // short_horizon_pos is unused (kept for ABI); the split is time-based.
    float compute_double_horizon_cost(const DeliveryAgent& a, const PDPTask& task,
                                       int short_horizon_pos, float alpha);

    // ── Movement (RunnerMovement.cpp) ───────────────────────────────────────

    // Begin a new leg: load the path into the agent cursor, pre-fetch the
    // next leg, schedule the first edge. Returns its arrival step.
    int start_leg(int agent_id, int task_id, bool is_pickup,
                  osmium::object_id_type from,
                  osmium::object_id_type to,
                  int current_step);

    // Schedule the next edge of the active cursor; returns its arrival step.
    int schedule_next_edge(int agent_id, int current_step);

    // Pickup/delivery reached: fire rewards, promote the pre-fetched path,
    // start the next leg. came_from = edge just traversed (movement policy).
    void on_objective_reached(int agent_id, int task_id, bool is_pickup,
                              int current_step,
                              osmium::object_id_type came_from = 0);

    // Movement policy gate: one decision per node arrival; on "replan",
    // push_rerouted_path runs and the immediate reward is credited.
    void movement_decision(int agent_id, int current_step,
                           osmium::object_id_type next_edge,
                           osmium::object_id_type came_from);

    // Resolve task identity + pickup/delivery role of the agent's next
    // sequence objective from its own task states.
    void resolve_next_leg(const DeliveryAgent& agent,
                          const ObjectiveNode& next_obj,
                          int& next_task_id, bool& next_pickup);

    // Haversine-based fallback cost (road-factor adjusted) when A* fails.
    float fallback_cost(osmium::object_id_type from,
                        osmium::object_id_type to) const;

    // Process all due arrivals (arrival_step <= current_step).
    void process_arrivals(int current_step);
};

#endif // TRAINING_EPISODE_RUNNER_HPP
