#ifndef HYBRID_POLICY_HPP
#define HYBRID_POLICY_HPP

#include "BidPolicy.hpp"
#include <array>

// ════════════════════════════════════════════════════════════════════════════
// Hybrid (paper §4.3.3, eq. 8) — frozen MAPPO base + bounded per-agent
// online linear residual in logit space:
//
//   μ_i(x) = sigmoid( z_base(x) + w_i·x + b_i )
//
// The residual (w_i ∈ R^12, b_i) starts at zero (pure-MAPPO hot start) and is
// updated by online REINFORCE at the end of EVERY episode (train and eval —
// continual adaptation is the defining property). Two safety mechanisms:
//   - fleet-relative rollback: residual halved when the agent's rolling
//     fitness drops below fleet mean − threshold; reset to zero after K
//     consecutive halvings.
//   - divergence rollback: baseline reward EMA captured while the residual
//     is small; hard reset when the current reward EMA stays more than
//     div_threshold σ below the baseline for div_episodes episodes.
// Bounds: |w_i[k]| ≤ 0.5, |b_i| ≤ 2.
// ════════════════════════════════════════════════════════════════════════════
class HybridPolicy : public IBidPolicy {
public:
    struct HParams {
        float lr_residual        = 0.005f;
        float weight_clip        = 0.5f;
        float bias_clip          = 2.0f;

        float rollback_threshold = 0.1f;   // fleet mean − this → soft rollback
        float rollback_shrink    = 0.5f;
        int   hard_after         = 5;      // consecutive softs → hard reset
        int   fitness_window     = 20;     // rolling episode fitness window

        bool  divergence_enabled  = true;
        float divergence_sigma    = 2.0f;
        int   divergence_episodes = 3;
        float baseline_ema_alpha  = 0.05f;
        float baseline_lock_norm  = 0.1f;  // baseline learns while ||res|| < this
    };
    HParams hparams;

    HybridPolicy() = default;

    const char* name() const override { return "Hybrid"; }

    // Copy the (trained, frozen) MAPPO actor as the shared base. Must be
    // called before scoring; until then the base is a zero network (μ=0.5).
    void set_base_actor(const Mlp& base);
    bool has_base() const { return base_set_; }

    float score(int agent_id, const PolicyFeatures& f) override;

    // Online REINFORCE on every residual + rollback checks. Runs at the end
    // of EVERY episode (trains_online() = true).
    TrainingStats train_round() override;
    bool trains_online() const override { return true; }

    void ensure_agents(int n_agents) override;
    void reinit(uint32_t seed) override;       // clears residuals (base kept)

    void save(const std::string& path) const override;
    bool load(const std::string& path) override;

    float residual_norm(int agent_id) const;
    int   total_hard_rollbacks() const { return n_hard_rollbacks_; }

private:
    struct Residual {
        std::array<float, kPolicySz> w{};
        float b = 0.f;

        std::vector<float> fitness_history;    // rolling episode means
        int   consecutive_softs = 0;

        float baseline_ema  = 0.f;
        float baseline_var  = 1.f;
        float current_ema   = 0.f;
        bool  baseline_init = false;
        int   n_baseline_episodes    = 0;
        int   consecutive_divergent  = 0;
    };

    Mlp  base_;
    bool base_set_ = false;
    std::unordered_map<int, Residual> residuals_;
    int  n_hard_rollbacks_ = 0;

    Residual& get_or_create(int agent_id);
    float logit_of(const Residual& r, const float* x) const;
    void  clip_residual(Residual& r) const;
};

#endif // HYBRID_POLICY_HPP
