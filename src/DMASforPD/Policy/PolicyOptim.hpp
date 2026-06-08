#ifndef POLICY_OPTIM_HPP
#define POLICY_OPTIM_HPP

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
// (ObjectiveDMPolicy, IPPOPolicy, MapperPolicy, FaithfulMapperPolicy) compile
// without an extra .cpp.

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

#endif  // POLICY_OPTIM_HPP
