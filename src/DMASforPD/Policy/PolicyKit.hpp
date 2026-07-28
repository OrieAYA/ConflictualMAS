#ifndef POLICY_KIT_HPP
#define POLICY_KIT_HPP

#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

// ════════════════════════════════════════════════════════════════════════════
// PolicyOptim — shared optimisation utilities for PPO baselines
// ════════════════════════════════════════════════════════════════════════════
//
// Provides:
//   • Adam optimiser (β1=0.9, β2=0.999, ε=1e-5) — Yu+2022 Tab.7,
//     deWitt+2020 §4 (standard PPO default).
//   • Orthogonal weight initialisation — Yu+2022 Tab.7 "network init Orthogonal".
//   • He (variance-scaling, scale=2.0) initialisation — deWitt+2020 §4
//     "variance scaling initializer with truncated normal, scale=2.0".
//   • Huber loss helpers (δ=10.0) — Yu+2022 Tab.7 "value loss huber, δ=10".
//
// All routines are header-only inline so the four policy translation units
// (MappoPolicy, IppoPolicy, MapperPolicy, HybridPolicy) share them without an
// extra .cpp.

namespace policy_optim {

// ── Adam hyperparameters (paper-aligned defaults) ─────────────────────────
inline constexpr float kAdamBeta1 = 0.9f;
inline constexpr float kAdamBeta2 = 0.999f;
inline constexpr float kAdamEps   = 1e-5f;   // Yu+2022 Tab.7

// ── Huber loss δ ──────────────────────────────────────────────────────────
inline constexpr float kHuberDelta = 10.0f;  // Yu+2022 Tab.7

// ── Adam moment buffer ────────────────────────────────────────────────────
//
// Holds m and v (1st and 2nd moment EMAs) for a fixed-size parameter tensor.
// Stored alongside (not inside) the weights to keep the on-disk format of the
// policy networks stable: Adam state is NOT persisted; it resets to zero when
// a policy is loaded from a checkpoint.
template <int N>
struct AdamBuf {
    float m[N]{};
    float v[N]{};
};

// Specialisation for scalars (biases as single floats live elsewhere; this is
// used for the scalar bias b3 in MLPs).
struct AdamScalar {
    float m = 0.f;
    float v = 0.f;
};

// ── Adam update (parameter-tensor version) ────────────────────────────────
//
// One gradient step. `t` is the global step counter shared by every tensor in
// the same optimiser; pre-incremented by the caller so the first call passes
// t=1 (mandatory for bias correction).
inline void adam_apply(float* W, const float* dW,
                       float* m, float* v, int n,
                       float lr, int t,
                       float beta1 = kAdamBeta1,
                       float beta2 = kAdamBeta2,
                       float eps   = kAdamEps)
{
    const float bc1 = 1.f - std::pow(beta1, static_cast<float>(t));
    const float bc2 = 1.f - std::pow(beta2, static_cast<float>(t));
    for (int i = 0; i < n; ++i) {
        m[i] = beta1 * m[i] + (1.f - beta1) * dW[i];
        v[i] = beta2 * v[i] + (1.f - beta2) * dW[i] * dW[i];
        const float m_hat = m[i] / bc1;
        const float v_hat = v[i] / bc2;
        W[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

// Scalar Adam update (for a single-float parameter like b3).
inline void adam_apply_scalar(float& W, float dW,
                              AdamScalar& s, int t,
                              float lr,
                              float beta1 = kAdamBeta1,
                              float beta2 = kAdamBeta2,
                              float eps   = kAdamEps)
{
    const float bc1 = 1.f - std::pow(beta1, static_cast<float>(t));
    const float bc2 = 1.f - std::pow(beta2, static_cast<float>(t));
    s.m = beta1 * s.m + (1.f - beta1) * dW;
    s.v = beta2 * s.v + (1.f - beta2) * dW * dW;
    const float m_hat = s.m / bc1;
    const float v_hat = s.v / bc2;
    W -= lr * m_hat / (std::sqrt(v_hat) + eps);
}

// ── Orthogonal initialisation (Yu+2022 MAPPO default) ─────────────────────
//
// Generates an orthogonal weight matrix W of shape (rows × cols) row-major via
// QR decomposition of a random Gaussian matrix, then scales by `gain`.
// Standard MAPPO/CleanRL convention: gain = √2 for hidden ReLU layers,
// gain = 0.01 for the policy output, gain = 1.0 for the value head.
inline void init_orthogonal(float* W, int rows, int cols,
                            std::mt19937& rng, float gain = 1.4142135f)
{
    const int n = rows * cols;
    // Step 1: fill with standard normal.
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> A(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) A[i] = nd(rng);

    // Step 2: modified Gram-Schmidt on the row vectors (rows × cols).
    // Treat A as `rows` row-vectors of length `cols`; orthonormalise rows when
    // rows ≤ cols, otherwise orthonormalise columns. We pick the orientation
    // with the smaller dimension to be the one we orthonormalise.
    const bool by_row = (rows <= cols);
    const int  K = by_row ? rows : cols;
    const int  L = by_row ? cols : rows;

    auto at = [&](int k, int l) -> float& {
        return by_row ? A[k * cols + l] : A[l * cols + k];
    };

    for (int k = 0; k < K; ++k) {
        // Subtract projections onto previously processed orthonormal vectors.
        for (int p = 0; p < k; ++p) {
            float dot = 0.f;
            for (int l = 0; l < L; ++l) dot += at(p, l) * at(k, l);
            for (int l = 0; l < L; ++l) at(k, l) -= dot * at(p, l);
        }
        // Normalise vector k.
        float nrm = 0.f;
        for (int l = 0; l < L; ++l) nrm += at(k, l) * at(k, l);
        nrm = std::sqrt(nrm);
        if (nrm < 1e-8f) nrm = 1e-8f;
        const float inv = 1.f / nrm;
        for (int l = 0; l < L; ++l) at(k, l) *= inv;
    }

    // Step 3: copy back into W with gain scaling.
    for (int i = 0; i < n; ++i) W[i] = gain * A[i];
}

// ── He / variance-scaling init (deWitt+2020 §4: scale=2.0, truncated normal) ─
//
// Equivalent to PyTorch / Keras "VarianceScaling(scale=2.0, distribution=
// 'truncated_normal')" — std = √(2 / fan_in). The "truncated_normal" part
// caps the tails at ±2σ; we sample by rejection (very fast for ±2σ ≈ 95.4%).
inline void init_he_truncated(float* W, int fan_in, int fan_out,
                              std::mt19937& rng)
{
    const float sigma = std::sqrt(2.f / static_cast<float>(fan_in));
    std::normal_distribution<float> dist(0.f, sigma);
    for (int i = 0; i < fan_in * fan_out; ++i) {
        float w;
        do { w = dist(rng); } while (std::abs(w) > 2.f * sigma);  // ±2σ truncation
        W[i] = w;
    }
}

// ── Huber loss squared-error replacement ──────────────────────────────────
//
// MAPPO Yu+2022 trains the critic with Huber loss (δ=10): quadratic for small
// errors, linear past δ. We use the symbol-level form L(e) = e²/2 if |e|≤δ,
// δ(|e|−δ/2) otherwise. The MAX-trick used by the existing implementations
// (max(err_new², err_clip²)) generalises to Huber by computing both values
// and taking the larger one — which is what the value-clipping objective
// requires under PPO-style "pessimistic" critic updates.
inline float huber_value(float err, float delta = kHuberDelta) {
    const float a = std::abs(err);
    if (a <= delta) return 0.5f * err * err;
    return delta * (a - 0.5f * delta);
}

// Gradient of Huber w.r.t. err: clamp(err, -δ, +δ).
inline float huber_grad(float err, float delta = kHuberDelta) {
    if (err >  delta) return  delta;
    if (err < -delta) return -delta;
    return err;
}

}  // namespace policy_optim

#include <array>
#include <cstdio>
#include <functional>
#include <string>

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
