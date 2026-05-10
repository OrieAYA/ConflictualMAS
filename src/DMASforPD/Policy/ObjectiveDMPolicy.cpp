#include "ObjectiveDMPolicy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {

// ── Xavier normal initialisation ──────────────────────────────────────────────
void xavier(float* W, int fan_in, int fan_out, std::mt19937& rng) {
    float s = std::sqrt(2.f / static_cast<float>(fan_in + fan_out));
    std::normal_distribution<float> dist(0.f, s);
    for (int i = 0; i < fan_in * fan_out; ++i) W[i] = dist(rng);
}

// ── Linear + ReLU layer ───────────────────────────────────────────────────────
// W[out_n × in_n], row-major. Writes pre-activation to pa (optional).
void linrelu(const float* W, const float* b,
             const float* x, int out_n, int in_n,
             float* y, float* pa)
{
    for (int i = 0; i < out_n; ++i) {
        float s = b[i];
        const float* row = W + i * in_n;
        for (int j = 0; j < in_n; ++j) s += row[j] * x[j];
        if (pa) pa[i] = s;
        y[i] = s > 0.f ? s : 0.f;
    }
}

// ── Linear output (no activation) ────────────────────────────────────────────
float linout(const float* W, float b, const float* x, int in_n) {
    float s = b;
    for (int j = 0; j < in_n; ++j) s += W[j] * x[j];
    return s;
}

} // namespace

// ── PolicyFeatures ────────────────────────────────────────────────────────────
void PolicyFeatures::to_array(float* dst) const {
    dst[0] = cost_diff;
    dst[1] = profit_rate;
    dst[2] = current_load;
    dst[3] = queue_duration;
    dst[4] = efficiency_loss;
    dst[5] = rank_in_call;
    dst[6] = task_importance;
    dst[7] = n_agents_ratio;
    dst[8] = n_alloc_ratio;
    dst[9] = n_avail_ratio;
}

// ── ActorMLP ──────────────────────────────────────────────────────────────────
void ActorMLP::init_xavier(std::mt19937& rng) {
    xavier(W1, kPolicySz, kHid, rng);
    xavier(W2, kHid, kHid, rng);
    xavier(W3, kHid, 1, rng);
    std::fill(b1, b1 + kHid, 0.f);
    std::fill(b2, b2 + kHid, 0.f);
    b3 = 1.0f; // → sigmoid(1.0) ≈ 0.73: prefer acceptance before training
}

float ActorMLP::forward(const float* x,
                        float* h1, float* h2,
                        float* pa1, float* pa2) const
{
    float _h1[kHid], _h2[kHid], _pa1[kHid], _pa2[kHid];
    if (!h1)  h1  = _h1;
    if (!h2)  h2  = _h2;
    if (!pa1) pa1 = _pa1;
    if (!pa2) pa2 = _pa2;

    linrelu(W1, b1, x,  kHid, kPolicySz, h1, pa1);
    linrelu(W2, b2, h1, kHid, kHid,      h2, pa2);
    float z3 = linout(W3, b3, h2, kHid);
    return 1.f / (1.f + std::expf(-z3));
}

// ── CriticMLP ─────────────────────────────────────────────────────────────────
void CriticMLP::init_xavier(std::mt19937& rng) {
    xavier(W1, kGlobSz, kHid, rng);
    xavier(W2, kHid, kHid, rng);
    xavier(W3, kHid, 1, rng);
    std::fill(b1, b1 + kHid, 0.f);
    std::fill(b2, b2 + kHid, 0.f);
    b3 = 0.f;
}

float CriticMLP::forward(const float* x,
                         float* h1, float* h2,
                         float* pa1, float* pa2) const
{
    float _h1[kHid], _h2[kHid], _pa1[kHid], _pa2[kHid];
    if (!h1)  h1  = _h1;
    if (!h2)  h2  = _h2;
    if (!pa1) pa1 = _pa1;
    if (!pa2) pa2 = _pa2;

    linrelu(W1, b1, x,  kHid, kGlobSz, h1, pa1);
    linrelu(W2, b2, h1, kHid, kHid,    h2, pa2);
    return linout(W3, b3, h2, kHid); // raw value, no sigmoid
}

// ── ObjectiveDMPolicy ─────────────────────────────────────────────────────────
ObjectiveDMPolicy::ObjectiveDMPolicy()
    : rng_(std::random_device{}())
{
    actor.init_xavier(rng_);
    critic.init_xavier(rng_);
}

ObjectiveDMPolicy& ObjectiveDMPolicy::shared() {
    static ObjectiveDMPolicy instance;
    return instance;
}

float ObjectiveDMPolicy::score(const PolicyFeatures& features) const {
    float x[kPolicySz];
    features.to_array(x);
    return actor.forward(x);
}

void ObjectiveDMPolicy::record(const PolicyFeatures& obs, float action,
                               float reward, int agent_id) {
    float x[kPolicySz];
    obs.to_array(x);
    const float eps = 1e-8f;
    float mu = actor.forward(x);
    float lp = (action > 0.5f)
        ? std::logf(mu + eps)
        : std::logf(1.f - mu + eps);

    Experience e;
    obs.to_array(e.obs.data());
    e.agent_id = agent_id;
    e.action   = action;
    e.log_prob = lp;
    e.reward   = reward;
    buffer_.push_back(e);
}

void ObjectiveDMPolicy::update_reward(int buffer_idx, float reward) {
    if (buffer_idx >= 0 && buffer_idx < static_cast<int>(buffer_.size()))
        buffer_[buffer_idx].reward = reward;
}

// ── GAE (Generalized Advantage Estimation) — per-agent trajectories ───────────
//
// The global buffer interleaves decisions from multiple agents. Running a single
// backward pass treats buffer[i+1] as the successor of buffer[i] even when they
// belong to different agents, injecting cross-agent bootstrap error into the
// advantages. This version groups buffer indices by agent_id and computes GAE
// independently for each agent's sub-trajectory, fixing that error.
void ObjectiveDMPolicy::compute_gae() {
    int n = static_cast<int>(buffer_.size());
    if (n == 0) return;

    // Collect buffer indices per agent in insertion order.
    std::unordered_map<int, std::vector<int>> by_agent;
    by_agent.reserve(32);
    for (int i = 0; i < n; ++i)
        by_agent[buffer_[i].agent_id].push_back(i);

    for (auto& [aid, idxs] : by_agent) {
        float gae = 0.f, next_val = 0.f;
        for (int k = static_cast<int>(idxs.size()) - 1; k >= 0; --k) {
            int i = idxs[k];
            float delta = buffer_[i].reward
                        + hparams.gamma * next_val
                        - buffer_[i].value;
            gae = delta + hparams.gamma * hparams.lam_gae * gae;
            buffer_[i].advantage = gae;
            buffer_[i].ret       = gae + buffer_[i].value;
            next_val             = buffer_[i].value;
        }
    }
}

// ── Actor PPO update (manual backprop, gradient ascent on L_clip + entropy) ───
//
// Bernoulli policy: π(a|s) = μ^a * (1−μ)^(1−a)
//   log π        = a·log(μ) + (1−a)·log(1−μ)
//   ∂log π/∂z3   = (a − μ)            [chain rule cancels μ(1−μ)]
//   ∂H/∂z3       = log((1−μ)/μ)·μ(1−μ) [entropy gradient through sigmoid]
//   ∂L_clip/∂z3  = A·r·(a−μ)  when unclipped, else 0
ObjectiveDMPolicy::EpochActorStats ObjectiveDMPolicy::update_actor() {
    int n = static_cast<int>(buffer_.size());
    if (n == 0) return {};

    float adv_mean = 0.f;
    for (const auto& e : buffer_) adv_mean += e.advantage;
    adv_mean /= n;
    float adv_var = 0.f;
    for (const auto& e : buffer_) {
        float d = e.advantage - adv_mean;
        adv_var += d * d;
    }
    adv_var /= n;
    float adv_std = std::sqrt(adv_var + 1e-8f);

    float dW1[kHid * kPolicySz]{};
    float db1[kHid]{};
    float dW2[kHid * kHid]{};
    float db2[kHid]{};
    float dW3[kHid]{};
    float db3 = 0.f;

    const float eps   = 1e-8f;
    const float inv_n = 1.f / n;

    float L_clip_acc = 0.f, entropy_acc = 0.f;

    for (const auto& e : buffer_) {
        float h1[kHid], h2[kHid], pa1[kHid], pa2[kHid];
        float mu = actor.forward(e.obs.data(), h1, h2, pa1, pa2);

        float lp_new = (e.action > 0.5f)
            ? std::logf(mu + eps)
            : std::logf(1.f - mu + eps);
        float r = std::expf(lp_new - e.log_prob);
        float A = (e.advantage - adv_mean) / adv_std;

        bool clipped = (r > 1.f + hparams.clip_eps && A > 0.f)
                    || (r < 1.f - hparams.clip_eps && A < 0.f);

        float clip_grad = 0.f;
        if (!clipped)
            clip_grad = A * r * (e.action - mu);

        float ent_grad = hparams.ent_w
                       * std::logf((1.f - mu + eps) / (mu + eps))
                       * mu * (1.f - mu);

        // Stats: L_clip value and policy entropy
        float L_clip = clipped
            ? std::clamp(r, 1.f - hparams.clip_eps, 1.f + hparams.clip_eps) * A
            : r * A;
        L_clip_acc  += L_clip;
        entropy_acc += -(mu * std::logf(mu + eps) + (1.f - mu) * std::logf(1.f - mu + eps));

        float dz3 = (clip_grad + ent_grad) * inv_n;

        // ── Backprop through layer 3 ──────────────────────────────────────
        db3 += dz3;
        float dh2[kHid]{};
        for (int j = 0; j < kHid; ++j) {
            dW3[j] += dz3 * h2[j];
            dh2[j]  = dz3 * actor.W3[j];
        }

        // ── Backprop through ReLU layer 2 ─────────────────────────────────
        float dh1[kHid]{};
        for (int i = 0; i < kHid; ++i) {
            float dz2_i = dh2[i] * (pa2[i] > 0.f ? 1.f : 0.f);
            db2[i] += dz2_i;
            const float* row = actor.W2 + i * kHid;
            for (int j = 0; j < kHid; ++j) {
                dW2[i * kHid + j] += dz2_i * h1[j];
                dh1[j]            += dz2_i * row[j];
            }
        }

        // ── Backprop through ReLU layer 1 ─────────────────────────────────
        for (int i = 0; i < kHid; ++i) {
            float dz1_i = dh1[i] * (pa1[i] > 0.f ? 1.f : 0.f);
            db1[i] += dz1_i;
            for (int j = 0; j < kPolicySz; ++j)
                dW1[i * kPolicySz + j] += dz1_i * e.obs[j];
        }
    }

    float lr = hparams.lr_actor;
    for (int i = 0; i < kHid * kPolicySz; ++i) actor.W1[i] += lr * dW1[i];
    for (int i = 0; i < kHid; ++i)              actor.b1[i] += lr * db1[i];
    for (int i = 0; i < kHid * kHid; ++i)       actor.W2[i] += lr * dW2[i];
    for (int i = 0; i < kHid; ++i)              actor.b2[i] += lr * db2[i];
    for (int i = 0; i < kHid; ++i)              actor.W3[i] += lr * dW3[i];
    actor.b3 += lr * db3;

    return { -(L_clip_acc / n), entropy_acc / n, adv_mean, adv_std };
}

// ── Critic MSE update (gradient descent on (V − ret)²) ───────────────────────
float ObjectiveDMPolicy::update_critic(
    const std::vector<std::array<float, kGlobSz>>& gs)
{
    int n  = static_cast<int>(buffer_.size());
    int ng = static_cast<int>(gs.size());
    if (n == 0 || ng == 0) return 0.f;
    float mse_acc = 0.f;

    float dW1[kHid * kGlobSz]{};
    float db1[kHid]{};
    float dW2[kHid * kHid]{};
    float db2[kHid]{};
    float dW3[kHid]{};
    float db3 = 0.f;

    const float inv_n = 1.f / std::min(n, ng);

    for (int idx = 0; idx < n && idx < ng; ++idx) {
        const float* gx = gs[idx].data();
        float h1[kHid], h2[kHid], pa1[kHid], pa2[kHid];
        float V = critic.forward(gx, h1, h2, pa1, pa2);

        float err = V - buffer_[idx].ret;
        mse_acc  += err * err;
        float dz3 = 2.f * err * inv_n;

        db3 += dz3;
        float dh2[kHid]{};
        for (int j = 0; j < kHid; ++j) {
            dW3[j] += dz3 * h2[j];
            dh2[j]  = dz3 * critic.W3[j];
        }

        float dh1[kHid]{};
        for (int i = 0; i < kHid; ++i) {
            float dz2_i = dh2[i] * (pa2[i] > 0.f ? 1.f : 0.f);
            db2[i] += dz2_i;
            const float* row = critic.W2 + i * kHid;
            for (int j = 0; j < kHid; ++j) {
                dW2[i * kHid + j] += dz2_i * h1[j];
                dh1[j]            += dz2_i * row[j];
            }
        }

        for (int i = 0; i < kHid; ++i) {
            float dz1_i = dh1[i] * (pa1[i] > 0.f ? 1.f : 0.f);
            db1[i] += dz1_i;
            for (int j = 0; j < kGlobSz; ++j)
                dW1[i * kGlobSz + j] += dz1_i * gx[j];
        }
    }

    // SGD gradient descent: W -= lr * ∂MSE/∂W
    float lr = hparams.lr_critic;
    for (int i = 0; i < kHid * kGlobSz; ++i) critic.W1[i] -= lr * dW1[i];
    for (int i = 0; i < kHid; ++i)            critic.b1[i] -= lr * db1[i];
    for (int i = 0; i < kHid * kHid; ++i)     critic.W2[i] -= lr * dW2[i];
    for (int i = 0; i < kHid; ++i)            critic.b2[i] -= lr * db2[i];
    for (int i = 0; i < kHid; ++i)            critic.W3[i] -= lr * dW3[i];
    critic.b3 -= lr * db3;
    return mse_acc / std::min(n, ng);
}

// ── train_epoch ───────────────────────────────────────────────────────────────
ObjectiveDMPolicy::TrainingStats ObjectiveDMPolicy::train_epoch(
    const std::vector<std::array<float, kGlobSz>>& global_states)
{
    int n  = static_cast<int>(buffer_.size());
    int ng = static_cast<int>(global_states.size());
    TrainingStats ts;
    ts.n_exp = n;
    if (n == 0) return ts;

    if (ng > 0) {
        for (int i = 0; i < n && i < ng; ++i)
            buffer_[i].value = critic.forward(global_states[i].data());
    }

    compute_gae();

    // Critic epochs (buffer order preserved for gs alignment).
    float closs_acc = 0.f;
    if (ng > 0) {
        for (int ep = 0; ep < hparams.epochs; ++ep)
            closs_acc += update_critic(global_states);
        ts.critic_loss = closs_acc / hparams.epochs;
    }

    // Actor epochs (shuffled). Take stats from the last epoch — most current weights.
    EpochActorStats as{};
    for (int ep = 0; ep < hparams.epochs; ++ep) {
        std::shuffle(buffer_.begin(), buffer_.end(), rng_);
        as = update_actor();
    }
    ts.actor_loss = as.actor_loss;
    ts.entropy    = as.entropy;
    ts.adv_mean   = as.adv_mean;
    ts.adv_std    = as.adv_std;

    buffer_.clear();
    return ts;
}

// ── Persistence ───────────────────────────────────────────────────────────────
static constexpr uint32_t kMagicDM = 0xDEA110C0u;

void ObjectiveDMPolicy::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicDM,  sizeof(kMagicDM), 1, f);
    // Actor weights
    std::fwrite(actor.W1, sizeof(float), kHid * kPolicySz, f);
    std::fwrite(actor.b1, sizeof(float), kHid,             f);
    std::fwrite(actor.W2, sizeof(float), kHid * kHid,      f);
    std::fwrite(actor.b2, sizeof(float), kHid,             f);
    std::fwrite(actor.W3, sizeof(float), kHid,             f);
    std::fwrite(&actor.b3, sizeof(float), 1,               f);
    // Critic weights
    std::fwrite(critic.W1, sizeof(float), kHid * kGlobSz, f);
    std::fwrite(critic.b1, sizeof(float), kHid,           f);
    std::fwrite(critic.W2, sizeof(float), kHid * kHid,    f);
    std::fwrite(critic.b2, sizeof(float), kHid,           f);
    std::fwrite(critic.W3, sizeof(float), kHid,           f);
    std::fwrite(&critic.b3, sizeof(float), 1,             f);
    std::fclose(f);
}

bool ObjectiveDMPolicy::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != kMagicDM) {
        std::fclose(f); return false;
    }
    bool ok = true;
    ok = ok && std::fread(actor.W1,  sizeof(float), kHid * kPolicySz, f) == static_cast<size_t>(kHid * kPolicySz);
    ok = ok && std::fread(actor.b1,  sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(actor.W2,  sizeof(float), kHid * kHid,      f) == static_cast<size_t>(kHid * kHid);
    ok = ok && std::fread(actor.b2,  sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(actor.W3,  sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(&actor.b3, sizeof(float), 1,                f) == 1;
    ok = ok && std::fread(critic.W1,  sizeof(float), kHid * kGlobSz,  f) == static_cast<size_t>(kHid * kGlobSz);
    ok = ok && std::fread(critic.b1,  sizeof(float), kHid,            f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(critic.W2,  sizeof(float), kHid * kHid,     f) == static_cast<size_t>(kHid * kHid);
    ok = ok && std::fread(critic.b2,  sizeof(float), kHid,            f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(critic.W3,  sizeof(float), kHid,            f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(&critic.b3, sizeof(float), 1,               f) == 1;
    std::fclose(f);
    return ok;
}
