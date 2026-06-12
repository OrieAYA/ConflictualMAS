#ifndef POLICY_KIT_HPP
#define POLICY_KIT_HPP

#include "PolicyOptim.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// PolicyKit — the single shared implementation of the networks and the PPO
// machinery used by every learning policy (MAPPO / IPPO / MAPPER / Hybrid).
//
// Network geometry (paper Table 5): in → 64 → 64 → 1.
//   - actor : in = kPolicySz (12-d input vector), sigmoid head → μ ∈ [0,1]
//   - critic: in = kPolicySz (local, IPPO/MAPPER) or kCriticIn (32-d
//             Feature-Pruned global state, MAPPO), linear head → V(s)
//
// PPO (paper §4.3.3 / Table 5): clipping, value clipping, KL early stop,
// value normalisation (running mean/std of returns), Huber value loss,
// per-trajectory GAE, global advantage normalisation, Adam, linear LR and
// entropy annealing.
// ════════════════════════════════════════════════════════════════════════════

// ── Dimensions and normalisation constants ───────────────────────────────────
static constexpr int   kPolicySz   = 12;          // policy input vector length
static constexpr int   kHid        = 64;          // hidden width
static constexpr int   kGlobSz     = 20;          // system-only global state
static constexpr int   kCriticIn   = kGlobSz + kPolicySz;   // 32 (MAPPO FP state)
static constexpr float kCostScale  = 10'000.f;    // 10 km reference
static constexpr float kQueueScale = 50'000.f;    // 50 km reference
static constexpr float kMaxLoad    = 10.f;        // max tasks per agent
static constexpr float kMaxAgents  = 20.f;        // max fleet size
static constexpr float kImpMax     = 10.f;        // max task importance

// ── Policy input vector (paper Table 1) ──────────────────────────────────────
// Built at bid time from TaskOffer + Manager state + agent state.
// All entries ∈ [0,1].
struct PolicyFeatures {
    // Global context (4)
    float profit_rate       = 0.f;
    float n_agents_ratio    = 0.f;
    float n_alloc_ratio     = 0.f;
    float time_remaining    = 0.f;
    // Agent state (3)
    float queue_duration    = 0.f;
    float load_at_insertion = 0.f;
    float efficiency_loss   = 0.f;
    // Competition + impact (3)
    float marginal_cost_relative = 0.f;
    float fleet_pressure         = 0.f;
    float divergence_ratio       = 0.f;
    // Spatio-temporal (2)
    float congestion_delta_contribution = 0.f;
    float area_heat_pickup              = 0.f;

    void to_array(float* dst) const {
        dst[ 0] = profit_rate;            dst[ 1] = n_agents_ratio;
        dst[ 2] = n_alloc_ratio;          dst[ 3] = time_remaining;
        dst[ 4] = queue_duration;         dst[ 5] = load_at_insertion;
        dst[ 6] = efficiency_loss;        dst[ 7] = marginal_cost_relative;
        dst[ 8] = fleet_pressure;         dst[ 9] = divergence_ratio;
        dst[10] = congestion_delta_contribution;
        dst[11] = area_heat_pickup;
    }
};

// ── One recorded bid decision ─────────────────────────────────────────────────
struct Experience {
    std::array<float, kPolicySz> obs{};   // policy input at decision time
    std::array<float, kGlobSz>   gs{};    // system state at decision time
                                          // (consumed only by the MAPPO critic)
    int   agent_id  = -1;
    float action    = 0.f;   // 1 = bid, 0 = refuse
    float log_prob  = 0.f;   // log π_old(a|s) at decision time
    float reward    = 0.f;   // accumulated task outcome / shaping
    float value     = 0.f;   // V(s), filled during training
    float advantage = 0.f;   // GAE, filled during training
    float ret       = 0.f;   // λ-return, filled during training
};

// ── Running mean/std of returns (value normalisation, Welford) ───────────────
struct RunningMeanStd {
    double    mean  = 0.0;
    double    var   = 1.0;
    long long count = 0;

    void update(float x) {
        ++count;
        const double d1 = static_cast<double>(x) - mean;
        mean += d1 / static_cast<double>(count);
        const double d2 = static_cast<double>(x) - mean;
        var = (count > 1) ? var + (d1 * d2 - var) / static_cast<double>(count)
                          : 1.0;
    }
    float mean_f()  const { return static_cast<float>(mean); }
    float std_dev() const { return static_cast<float>(std::sqrt(std::max(1e-6, var))); }

    void save_to(std::FILE* f) const;
    bool load_from(std::FILE* f);
};

// ── PPO hyper-parameters (paper Table 5 defaults = MAPPO column) ─────────────
struct PPOParams {
    float lr_actor       = 5e-4f;   // current (annealed by set_progress)
    float lr_actor_init  = 5e-4f;
    float lr_actor_min   = 5e-5f;
    float lr_critic      = 5e-4f;
    float lr_critic_init = 5e-4f;
    float lr_critic_min  = 5e-5f;
    float clip_eps       = 0.1f;
    float val_clip_eps   = 0.1f;
    float max_grad_norm  = 10.0f;
    float target_kl      = 0.01f;
    float gamma          = 0.99f;
    float lam_gae        = 0.95f;
    float ent_w          = 0.01f;
    float ent_w_init     = 0.01f;
    float ent_w_min      = 0.001f;
    int   epochs         = 10;
    int   batch_sz       = 1 << 20; // full-batch (paper: avoid mini-batching)

    // Linear annealing of lr and entropy with training progress ∈ [0,1].
    void set_progress(float p) {
        p = std::clamp(p, 0.f, 1.f);
        auto lerp = [p](float a, float b) { return a + p * (b - a); };
        lr_actor  = lerp(lr_actor_init,  lr_actor_min);
        lr_critic = lerp(lr_critic_init, lr_critic_min);
        ent_w     = lerp(ent_w_init,     ent_w_min);
    }
};

// ── Training stats (one train_round call) ────────────────────────────────────
struct TrainingStats {
    float actor_loss  = 0.f;
    float critic_loss = 0.f;
    float entropy     = 0.f;
    float adv_mean    = 0.f;
    float adv_std     = 0.f;
    float kl_approx   = 0.f;
    float clip_frac   = 0.f;
    int   n_exp       = 0;
    int   n_epochs    = 0;
};

// ── 3-layer MLP, runtime input width ─────────────────────────────────────────
struct Mlp {
    enum class Init { Orthogonal, HeTruncated };

    int in_dim = 0;
    std::vector<float> W1, b1, W2, b2, W3;   // W1: kHid×in, W2: kHid×kHid, W3: kHid
    float b3 = 0.f;

    // out_gain scales the output layer after init (0.01 for policy heads so
    // the initial μ ≈ 0.5; 1.0 for value heads).
    void init(int input_dim, std::mt19937& rng, float out_gain,
              Init kind = Init::Orthogonal);

    // Forward pass; optional buffers (size kHid each) cache activations for
    // backprop. Returns the raw output (logit / value) — no sigmoid.
    float forward(const float* x,
                  float* h1 = nullptr, float* h2 = nullptr,
                  float* pa1 = nullptr, float* pa2 = nullptr) const;

    static float sigmoid(float z) { return 1.f / (1.f + std::exp(-z)); }

    void save_to(std::FILE* f) const;
    bool load_from(std::FILE* f);          // in_dim must already be set
};

// ── Gradient accumulator + Adam state for one Mlp ────────────────────────────
struct MlpGrads {
    std::vector<float> dW1, db1, dW2, db2, dW3;
    float db3 = 0.f;

    void resize_for(const Mlp& net);
    void zero();
    void clip_global_norm(float max_norm);
    void negate();                          // ascent → descent direction
};

struct MlpAdam {
    std::vector<float> mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2, mW3, vW3;
    policy_optim::AdamScalar b3;
    int t = 0;

    void resize_for(const Mlp& net);
    void apply(Mlp& net, const MlpGrads& g, float lr);
};

// Accumulate ∂out/∂θ · dz3 for one sample into `g` (x, h1, h2, pa1, pa2 are
// the cached forward activations).
void backprop_sample(const Mlp& net, const float* x,
                     const float* h1, const float* h2,
                     const float* pa1, const float* pa2,
                     float dz3, MlpGrads& g);

// ── GAE over ONE trajectory (a single agent's ordered decisions) ─────────────
// Fills advantage and ret from reward and value. Terminal bootstrap = 0
// (called at episode end).
void compute_gae(std::vector<Experience>& traj, float gamma, float lam);

// ── One full PPO update over a set of trajectories ───────────────────────────
//
// Shared by MAPPO (1 actor + 1 centralised critic, N trajectories), IPPO
// (1 shared actor + 1 shared local critic, N trajectories) and MAPPER
// (per-agent nets, 1 trajectory per call).
//
//   critic              may be null → advantages = raw λ-returns (no baseline).
//   build_critic_input  fills critic->in_dim floats from an Experience
//                       (MAPPO: gs ++ obs; IPPO/MAPPER: obs).
//
// Steps: (1) fill values via the frozen critic (denormalised through vrms),
// (2) per-trajectory GAE, (3) vrms update from raw returns, (4) global
// advantage normalisation, (5) epoch loop of full-batch actor + critic SGD
// with PPO clip / value clip / Huber / KL early stop. Buffers are NOT
// cleared — the caller owns their lifecycle.
TrainingStats ppo_train(
    Mlp& actor,  MlpAdam& actor_opt,
    Mlp* critic, MlpAdam* critic_opt,
    RunningMeanStd& vrms,
    PPOParams& hp,
    const std::vector<std::vector<Experience>*>& trajs,
    const std::function<void(const Experience&, float*)>& build_critic_input,
    std::mt19937& rng);

#endif // POLICY_KIT_HPP
