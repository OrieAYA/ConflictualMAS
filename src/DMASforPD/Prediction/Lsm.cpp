#include "DMASforPD/Prediction/Lsm.hpp"
#include <algorithm>
#include <cmath>

namespace {

constexpr float kVLeak      = 0.90f;
constexpr float kTraceLeak  = 0.90f;
constexpr float kThreshold  = 1.0f;
constexpr float kLambda2    = 4.0f;    // λ² of the distance kernel (λ=2, Maass 2002)
constexpr float kGainRes    = 1.2f;    // per-neuron Σ|w| after row normalisation
constexpr float kGainIn     = 0.8f;
constexpr float kInDensity  = 0.30f;
constexpr float kExcFrac    = 0.80f;

// Lattice 4×4×16.
inline void lattice_pos(int i, int& x, int& y, int& z) {
    x = i % 4;
    y = (i / 4) % 4;
    z = i / 16;
}

// Maass 2002 connection probabilities C by (pre, post) type.
inline float pair_c(bool pre_exc, bool post_exc) {
    if (pre_exc  && post_exc)  return 0.3f;
    if (pre_exc  && !post_exc) return 0.2f;
    if (!pre_exc && post_exc)  return 0.4f;
    return 0.1f;
}

void row_normalise(std::vector<float>& W, int rows, int cols, float gain) {
    for (int i = 0; i < rows; ++i) {
        float s = 0.f;
        float* row = W.data() + static_cast<size_t>(i) * cols;
        for (int j = 0; j < cols; ++j) s += std::fabs(row[j]);
        if (s <= 1e-9f) continue;
        const float k = gain / s;
        for (int j = 0; j < cols; ++j) row[j] *= k;
    }
}

}  // namespace

void LsmModule::reinit(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    std::vector<bool> exc(kLsmN);
    for (int i = 0; i < kLsmN; ++i) exc[i] = u01(rng) < kExcFrac;

    Wres_.assign(static_cast<size_t>(kLsmN) * kLsmN, 0.f);
    for (int pre = 0; pre < kLsmN; ++pre) {
        int x1, y1, z1;
        lattice_pos(pre, x1, y1, z1);
        for (int post = 0; post < kLsmN; ++post) {
            if (pre == post) continue;
            int x2, y2, z2;
            lattice_pos(post, x2, y2, z2);
            const float d2 = static_cast<float>(
                (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
                + (z1 - z2) * (z1 - z2));
            const float p = pair_c(exc[pre], exc[post])
                          * std::exp(-d2 / kLambda2);
            if (u01(rng) >= p) continue;
            const float w = u01(rng);
            Wres_[static_cast<size_t>(post) * kLsmN + pre] =
                exc[pre] ? w : -w;
        }
    }
    row_normalise(Wres_, kLsmN, kLsmN, kGainRes);

    Win_.assign(static_cast<size_t>(kLsmN) * kLsmIn, 0.f);
    for (int i = 0; i < kLsmN; ++i)
        for (int j = 0; j < kLsmIn; ++j)
            if (u01(rng) < kInDensity)
                Win_[static_cast<size_t>(i) * kLsmIn + j] =
                    (u01(rng) < 0.5f ? 1.f : -1.f) * u01(rng);
    row_normalise(Win_, kLsmN, kLsmIn, kGainIn);

    Wout_.assign(static_cast<size_t>(kLsmOut) * kLsmN, 0.f);
    reset_state();
}

void LsmModule::reset_state() {
    v_.assign(kLsmN, 0.f);
    spike_.assign(kLsmN, 0.f);
    trace_.assign(kLsmN, 0.f);
}

void LsmModule::step(const float* u) {
    std::vector<float> next(kLsmN, 0.f);
    for (int i = 0; i < kLsmN; ++i) {
        float s = kVLeak * v_[i];
        const float* wi = Win_.data() + static_cast<size_t>(i) * kLsmIn;
        for (int j = 0; j < kLsmIn; ++j) s += wi[j] * u[j];
        const float* wr = Wres_.data() + static_cast<size_t>(i) * kLsmN;
        for (int j = 0; j < kLsmN; ++j) s += wr[j] * spike_[j];
        if (s > kThreshold) {
            next[i] = 1.f;
            v_[i]   = 0.f;
        } else {
            v_[i] = std::max(s, -kThreshold);
        }
    }
    spike_ = std::move(next);
    for (int i = 0; i < kLsmN; ++i)
        trace_[i] = kTraceLeak * trace_[i] + (1.f - kTraceLeak) * spike_[i];
}

void LsmModule::predict_state(const std::vector<float>& x, float* y) const {
    for (int o = 0; o < kLsmOut; ++o) {
        float s = 0.f;
        const float* row = Wout_.data() + static_cast<size_t>(o) * kLsmN;
        for (int j = 0; j < kLsmN; ++j) s += row[j] * x[j];
        y[o] = s;
    }
}

float LsmModule::learn(const std::vector<float>& x_past, const float* y,
                       float lr) {
    float yhat[kLsmOut];
    predict_state(x_past, yhat);

    float nx = 1e-6f;
    for (int j = 0; j < kLsmN; ++j) nx += x_past[j] * x_past[j];
    const float k = lr / nx;

    float mse = 0.f;
    for (int o = 0; o < kLsmOut; ++o) {
        const float e = y[o] - yhat[o];
        mse += e * e;
        float* row = Wout_.data() + static_cast<size_t>(o) * kLsmN;
        for (int j = 0; j < kLsmN; ++j) row[j] += k * e * x_past[j];
    }
    return mse / kLsmOut;
}

// ── Persistence ──────────────────────────────────────────────────────────────

static constexpr uint32_t kMagicLsm = 0xDEA110D4u;

void LsmModule::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    const uint32_t dims[4] = { kMagicLsm, kLsmN, kLsmIn, kLsmOut };
    std::fwrite(dims, sizeof(uint32_t), 4, f);
    std::fwrite(Win_.data(),  sizeof(float), Win_.size(),  f);
    std::fwrite(Wres_.data(), sizeof(float), Wres_.size(), f);
    std::fwrite(Wout_.data(), sizeof(float), Wout_.size(), f);
    std::fclose(f);
}

bool LsmModule::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t dims[4] = {};
    bool ok = std::fread(dims, sizeof(uint32_t), 4, f) == 4
           && dims[0] == kMagicLsm && dims[1] == kLsmN
           && dims[2] == kLsmIn && dims[3] == kLsmOut;
    if (ok) {
        Win_.resize(static_cast<size_t>(kLsmN) * kLsmIn);
        Wres_.resize(static_cast<size_t>(kLsmN) * kLsmN);
        Wout_.resize(static_cast<size_t>(kLsmOut) * kLsmN);
        ok = std::fread(Win_.data(),  sizeof(float), Win_.size(),  f) == Win_.size()
          && std::fread(Wres_.data(), sizeof(float), Wres_.size(), f) == Wres_.size()
          && std::fread(Wout_.data(), sizeof(float), Wout_.size(), f) == Wout_.size();
    }
    std::fclose(f);
    if (ok) reset_state();
    return ok;
}

LsmModule& lsm_module() { static LsmModule m; return m; }
