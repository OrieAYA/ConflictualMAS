#ifndef HYBRID_POLICY_HPP
#define HYBRID_POLICY_HPP

#include "ObjectiveDMPolicy.hpp"  // reuses ActorMLP, PolicyFeatures, kPolicySz, kHid
#include <array>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// HybridPolicy — Frozen MAPPO base + per-agent online linear residual
// ════════════════════════════════════════════════════════════════════════════
//
// Architecture:
//
//   features (13d) ──┬─────► [MAPPO base (frozen)] ──► logit_base
//                    │
//                    └─────► [Residual_i: w_i · features + b_i] ──► delta_logit
//
//   μ_i = sigmoid(logit_base + delta_logit)
//
// The MAPPO base is FROZEN during HybridPolicy execution — copied from a
// trained ObjectiveDMPolicy and never modified. Each agent owns a small linear
// residual (14 parameters: w_i ∈ R^13, b_i ∈ R) that is updated online via
// REINFORCE on observed rewards. Residuals are initialised to zero, so every
// new agent starts as pure MAPPO (full hot-start), then drifts toward its own
// specialisation based on its experiences.
//
// Rollback safety net:
//   - Soft rollback (every check_period decisions): if the agent's rolling
//     fitness is below the fleet mean − rollback_threshold, shrink the
//     residual by rollback_shrink (default 0.5).
//   - Hard rollback (after K consecutive soft rollbacks): reset residual to
//     zero, the agent returns to pure MAPPO behaviour.
//
// Comparison to MAPPO / MAPPER:
//
//   ┌──────────────┬───────────┬─────────────────┬────────────────────────┐
//   │              │ MAPPO     │ MAPPER          │ Hybrid                 │
//   ├──────────────┼───────────┼─────────────────┼────────────────────────┤
//   │ Base actor   │ 1 shared  │ N independent   │ 1 shared (frozen)      │
//   │ Per-agent    │ none      │ full actor      │ 14-param residual      │
//   │ Hot-start    │ ✓         │ ✗ (Xavier init) │ ✓ (residual = 0)       │
//   │ Heterogeneity│ ✗         │ ✓ full          │ ✓ bounded by clip      │
//   │ Online adapt │ no        │ no              │ ✓ continuous           │
//   │ Rollback     │ no        │ via Ev step     │ ✓ explicit threshold   │
//   │ Params/agent │ shared 64k│ ~5k             │ 14 (1000× less)        │
//   │ Mode collapse│ sensitive │ resistant       │ resistant + reversible │
//   └──────────────┴───────────┴─────────────────┴────────────────────────┘

// ── Online residual hyperparameters ──────────────────────────────────────────
struct HybridHParams {
    // Online REINFORCE update
    float lr_residual          = 0.005f;   // online learning rate (REINFORCE)
    float weight_clip          = 0.5f;     // |w_i[k]| ≤ weight_clip
    float bias_clip            = 2.0f;     // |b_i| ≤ bias_clip

    // Fleet-relative rollback (legacy mechanism)
    int   check_period         = 50;       // decisions between rollback checks
    float rollback_threshold   = 0.1f;     // soft rollback if fitness drops below
                                           // fleet_mean − this value
    float rollback_shrink      = 0.5f;     // multiply residual by this on soft rollback
    int   hard_rollback_after  = 5;        // hard reset after K consecutive softs
    int   fitness_window       = 20;       // rolling window of recent rewards

    // ── Reward-divergence safety rollback ───────────────────────────────────
    // Tracks a per-agent EMA of episode reward as "baseline" while the residual
    // norm is small (~MAPPO behaviour). Once the residual grows, the agent's
    // current reward EMA is compared to baseline ± K × baseline_std.
    // If current_reward < baseline_reward - divergence_threshold × baseline_std
    // for `divergence_episodes` consecutive episodes → emergency hard rollback.
    //
    // This is a SAFETY MECHANISM, complementary to the fleet-relative one above.
    // It triggers even when the whole fleet performs poorly (e.g. stress eval)
    // as long as THIS agent's residual is making it worse than its own past.
    bool  enable_divergence_rollback = true;
    float divergence_threshold      = 2.0f;  // K in σ-units below baseline
    int   divergence_episodes       = 3;     // consecutive episodes below threshold
    float baseline_ema_alpha        = 0.05f; // EMA decay; ≈ 1/(1-α) effective window
    float baseline_lock_norm        = 0.1f;  // baseline updates while ||residual|| < this
};

class HybridPolicy {
public:
    // Per-agent residual state.
    struct AgentResidual {
        std::array<float, kPolicySz> w{};  // linear weights on features
        float                        b = 0.f;

        // Fleet-relative rollback bookkeeping
        int                          decisions_since_check = 0;
        int                          consecutive_soft_rollbacks = 0;
        std::vector<float>           recent_rewards;     // for previous-episodes baseline
        float                        cached_fitness = 0.f;

        // ── Divergence-based rollback bookkeeping ───────────────────────────
        // Baseline EMA + EMA-of-squared-deviation, updated only while
        // ||residual|| < hparams.baseline_lock_norm (i.e. agent is still
        // close to pure-MAPPO behaviour and produces a clean baseline).
        float baseline_reward_ema = 0.f;
        float baseline_var_ema    = 1.f;   // EMA of (reward - baseline)² → std²
        float current_reward_ema  = 0.f;
        bool  baseline_initialised = false;
        int   consecutive_divergent_episodes = 0;
        int   n_baseline_episodes = 0;     // for the warmup before divergence checks trigger
        int   n_hard_rollbacks    = 0;     // total hard rollbacks applied (for logging)
    };

    HybridHParams hparams;

    HybridPolicy();

    // Shared instance — call from every delivery agent.
    static HybridPolicy& shared();

    // ── Base setup ─────────────────────────────────────────────────────────
    // Copy weights from a trained MAPPO actor. Must be called before
    // any score() / record() calls. After this, the base is FROZEN.
    void set_base_from(const ObjectiveDMPolicy& mappo);

    // Direct copy of the base actor (for cases where the actor was loaded
    // from disk into a temporary ObjectiveDMPolicy instance).
    void set_base_actor(const ActorMLP& src);

    // ── Inference ──────────────────────────────────────────────────────────
    // Returns μ ∈ [0,1]. TAM threshold: μ ≥ 0.5 → accept.
    // Lazily creates a zero-residual entry for unseen agent_id (= hot-start).
    float score(int agent_id, const PolicyFeatures& features);

    // ── Data collection ────────────────────────────────────────────────────
    // Records a decision. Returns the buffer index for later reward updates.
    int record(int agent_id, const PolicyFeatures& obs, float action, float reward);

    // Reward shaping: same semantics as MAPPO / MAPPER.
    void update_reward(int agent_id, int buf_idx, float reward);
    void add_to_reward(int agent_id, int buf_idx, float delta);

    int  buffer_size(int agent_id) const;
    int  total_buffer_size() const;

    // ── Recent-records log (parallels MapperPolicy / IPPOPolicy) ─────────────
    int                n_recent_records() const;
    std::pair<int,int> recent_record(int i) const;
    void               clear_recent_records();

    // ── End-of-episode online update ───────────────────────────────────────
    // Walks the buffer of each agent, applies REINFORCE update to that
    // agent's residual using observed rewards. Then runs rollback checks.
    // Clears buffers and recent records at the end.
    using TrainingStats = ObjectiveDMPolicy::TrainingStats;
    TrainingStats train_epoch();

    void clear_buffer(int agent_id);
    void clear_buffer_all();

    // ── Persistence ─────────────────────────────────────────────────────────
    // Saves base actor + all residuals. Compatible with load().
    void save(const std::string& path) const;
    bool load(const std::string& path);

    // ── Pre-allocate residual state for agent ids [0, n_agents) ─────────────
    void ensure_agents(int n_agents);
    int  n_agents() const { return static_cast<int>(residuals_.size()); }

    // Current per-agent residual norm (||w||₂ + |b|) for monitoring / logging.
    float residual_norm(int agent_id) const;

    // Number of agents currently in soft / hard rollback state.
    int  n_in_rollback() const;

    // Total hard rollbacks triggered by divergence safety net (logging).
    int  total_divergence_rollbacks() const;

private:
    ActorMLP                                base_actor_;   // frozen MAPPO actor
    bool                                    base_set_ = false;
    std::unordered_map<int, AgentResidual>  residuals_;
    std::unordered_map<int, std::vector<Experience>> buffers_;
    std::vector<std::pair<int,int>>         recent_records_;
    std::mt19937                            rng_;

    AgentResidual& get_or_create_residual(int agent_id);

    // Compute delta_logit for this agent on these features.
    float delta_logit(const AgentResidual& r, const PolicyFeatures& f) const;

    // REINFORCE update on the agent's residual using one experience.
    void reinforce_update(AgentResidual& r, const Experience& e);

    // Check rollback condition for one agent. Returns true if a rollback
    // (soft or hard) was applied.
    bool check_and_apply_rollback(int agent_id, AgentResidual& r,
                                   float fleet_mean_fitness);

    // Update baseline EMAs and check divergence rollback (per-episode).
    // Returns true if a divergence-triggered hard rollback was applied.
    bool update_baseline_and_check_divergence(AgentResidual& r,
                                              float episode_mean_reward);

    void clip_residual(AgentResidual& r);

    static float sigmoid(float x);
    static float logit (float p);
};

#endif // HYBRID_POLICY_HPP
