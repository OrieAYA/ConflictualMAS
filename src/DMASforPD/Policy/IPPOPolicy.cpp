#include "IPPOPolicy.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>

namespace {

void clip_grad_norm(float* arrays[], const int sizes[],
                    int n_arrays, float max_norm) {
    float sq = 0.f;
    for (int a = 0; a < n_arrays; ++a)
        for (int i = 0; i < sizes[a]; ++i)
            sq += arrays[a][i] * arrays[a][i];
    float norm = std::sqrt(sq);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-6f);
        for (int a = 0; a < n_arrays; ++a)
            for (int i = 0; i < sizes[a]; ++i)
                arrays[a][i] *= scale;
    }
}

} // namespace

// ── IPPOPolicy ──────────────────────────────────────────────────────────────
IPPOPolicy::IPPOPolicy()
    : rng_(std::random_device{}()) {
    actor.init_xavier(rng_);
    critic.init_xavier(rng_);
}

IPPOPolicy& IPPOPolicy::shared() {
    static IPPOPolicy instance;
    return instance;
}

void IPPOPolicy::set_progress(float progress) {
    const float p = std::clamp(progress, 0.f, 1.f);
    auto lerp = [p](float a, float b) { return a + p * (b - a); };
    hparams.lr_actor  = lerp(hparams.lr_actor_init,  hparams.lr_actor_min);
    hparams.lr_critic = lerp(hparams.lr_critic_init, hparams.lr_critic_min);
    hparams.ent_w     = lerp(hparams.ent_w_init,     hparams.ent_w_min);
}

void IPPOPolicy::init_xavier(std::mt19937& rng) {
    actor.init_xavier(rng);
    critic.init_xavier(rng);
    for (auto& [aid, buf] : buffers_) buf.clear();
}

void IPPOPolicy::ensure_agents(int n_agents) {
    for (int i = 0; i < n_agents; ++i) (void)get_or_create_buffer(i);
}

std::vector<Experience>& IPPOPolicy::get_or_create_buffer(int agent_id) {
    auto it = buffers_.find(agent_id);
    if (it != buffers_.end()) return it->second;
    auto [ins, _] = buffers_.emplace(agent_id, std::vector<Experience>{});
    return ins->second;
}

float IPPOPolicy::score(const PolicyFeatures& features) const {
    float x[kPolicySz];
    features.to_array(x);
    return actor.forward(x);
}

int IPPOPolicy::record(int agent_id, const PolicyFeatures& obs,
                        float action, float reward) {
    auto& buf = get_or_create_buffer(agent_id);
    float x[kPolicySz];
    obs.to_array(x);
    const float eps = 1e-8f;
    float mu = actor.forward(x);  // shared actor
    float lp = (action > 0.5f)
        ? std::logf(mu + eps) : std::logf(1.f - mu + eps);

    Experience e;
    obs.to_array(e.obs.data());
    e.agent_id = agent_id;
    e.action   = action;
    e.log_prob = lp;
    e.reward   = reward;
    buf.push_back(e);
    const int idx = static_cast<int>(buf.size()) - 1;
    recent_records_.emplace_back(agent_id, idx);
    return idx;
}

void IPPOPolicy::update_reward(int agent_id, int buf_idx, float reward) {
    auto it = buffers_.find(agent_id);
    if (it == buffers_.end()) return;
    auto& buf = it->second;
    if (buf_idx >= 0 && buf_idx < static_cast<int>(buf.size()))
        buf[buf_idx].reward = reward;
}

void IPPOPolicy::add_to_reward(int agent_id, int buf_idx, float delta) {
    auto it = buffers_.find(agent_id);
    if (it == buffers_.end()) return;
    auto& buf = it->second;
    if (buf_idx >= 0 && buf_idx < static_cast<int>(buf.size()))
        buf[buf_idx].reward += delta;
}

int IPPOPolicy::buffer_size(int agent_id) const {
    auto it = buffers_.find(agent_id);
    return (it == buffers_.end()) ? 0
        : static_cast<int>(it->second.size());
}

int IPPOPolicy::total_buffer_size() const {
    int total = 0;
    for (const auto& [aid, buf] : buffers_)
        total += static_cast<int>(buf.size());
    return total;
}

int IPPOPolicy::n_recent_records() const {
    return static_cast<int>(recent_records_.size());
}

std::pair<int,int> IPPOPolicy::recent_record(int i) const {
    return recent_records_[i];
}

void IPPOPolicy::clear_recent_records() {
    recent_records_.clear();
}

void IPPOPolicy::clear_buffer(int agent_id) {
    auto it = buffers_.find(agent_id);
    if (it != buffers_.end()) it->second.clear();
}

void IPPOPolicy::clear_buffer_all() {
    for (auto& [aid, buf] : buffers_) buf.clear();
    recent_records_.clear();
}

// ── Per-trajectory GAE — single agent's buffer reverse sweep ────────────────
void IPPOPolicy::compute_gae(std::vector<Experience>& buf) {
    const int n = static_cast<int>(buf.size());
    if (n == 0) return;
    float gae = 0.f, next_val = 0.f;
    for (int i = n - 1; i >= 0; --i) {
        float delta = buf[i].reward
                    + hparams.gamma * next_val
                    - buf[i].value;
        gae = delta + hparams.gamma * hparams.lam_gae * gae;
        buf[i].advantage = gae;
        buf[i].ret       = gae + buf[i].value;
        next_val         = buf[i].value;
    }
}

// ── Shared-actor mini-batch update (PPO clip + entropy) ─────────────────────
//
// The batch is a permutation of (agent_id, buf_idx) pairs. Each entry's
// (obs, action, log_prob, advantage) is fetched from buffers_[aid][idx]
// and the SAME shared actor is updated against all of them — that is the
// parameter-sharing benefit of IPPO over MAPPER.
IPPOPolicy::MBStats IPPOPolicy::update_actor_mb(
    const std::vector<std::pair<int,int>>& perm, int start, int end)
{
    const int bsz = end - start;
    if (bsz <= 0) return {};

    float dW1[kHid * kPolicySz]{};
    float db1[kHid]{};
    float dW2[kHid * kHid]{};
    float db2[kHid]{};
    float dW3[kHid]{};
    float db3 = 0.f;

    const float eps   = 1e-8f;
    const float inv_n = 1.f / bsz;

    float L_acc = 0.f, ent_acc = 0.f, kl_acc = 0.f, clip_acc = 0.f;

    for (int pi = start; pi < end; ++pi) {
        const auto [aid, idx] = perm[pi];
        const Experience& e = buffers_.at(aid)[idx];

        float h1[kHid], h2[kHid], pa1[kHid], pa2[kHid];
        float mu = actor.forward(e.obs.data(), h1, h2, pa1, pa2);

        float lp_new = (e.action > 0.5f)
            ? std::logf(mu + eps) : std::logf(1.f - mu + eps);
        float r = std::expf(lp_new - e.log_prob);
        float A = e.advantage;

        kl_acc += e.log_prob - lp_new;

        bool clipped = (r > 1.f + hparams.clip_eps && A > 0.f)
                    || (r < 1.f - hparams.clip_eps && A < 0.f);
        if (clipped) clip_acc += 1.f;

        float clip_grad = clipped ? 0.f : A * r * (e.action - mu);
        float ent_grad  = hparams.ent_w
                        * std::logf((1.f - mu + eps) / (mu + eps))
                        * mu * (1.f - mu);

        float L_clip = clipped
            ? std::clamp(r, 1.f - hparams.clip_eps, 1.f + hparams.clip_eps) * A
            : r * A;
        L_acc   += L_clip;
        ent_acc += -(mu * std::logf(mu + eps)
                  + (1.f - mu) * std::logf(1.f - mu + eps));

        float dz3 = (clip_grad + ent_grad) * inv_n;

        db3 += dz3;
        float dh2[kHid]{};
        for (int j = 0; j < kHid; ++j) {
            dW3[j] += dz3 * h2[j];
            dh2[j]  = dz3 * actor.W3[j];
        }

        float dh1[kHid]{};
        for (int i = 0; i < kHid; ++i) {
            float dz2 = dh2[i] * (pa2[i] > 0.f ? 1.f : 0.f);
            db2[i] += dz2;
            const float* row = actor.W2 + i * kHid;
            for (int j = 0; j < kHid; ++j) {
                dW2[i*kHid+j] += dz2 * h1[j];
                dh1[j]        += dz2 * row[j];
            }
        }
        for (int i = 0; i < kHid; ++i) {
            float dz1 = dh1[i] * (pa1[i] > 0.f ? 1.f : 0.f);
            db1[i] += dz1;
            for (int j = 0; j < kPolicySz; ++j)
                dW1[i*kPolicySz+j] += dz1 * e.obs[j];
        }
    }

    float* ag[] = { dW1, db1, dW2, db2, dW3, &db3 };
    int    as[] = { kHid*kPolicySz, kHid, kHid*kHid, kHid, kHid, 1 };
    clip_grad_norm(ag, as, 6, hparams.max_grad_norm);

    float lr = hparams.lr_actor;
    for (int i = 0; i < kHid*kPolicySz; ++i) actor.W1[i] += lr * dW1[i];
    for (int i = 0; i < kHid; ++i)            actor.b1[i] += lr * db1[i];
    for (int i = 0; i < kHid*kHid; ++i)       actor.W2[i] += lr * dW2[i];
    for (int i = 0; i < kHid; ++i)            actor.b2[i] += lr * db2[i];
    for (int i = 0; i < kHid; ++i)            actor.W3[i] += lr * dW3[i];
    actor.b3 += lr * db3;

    return { -(L_acc * inv_n), ent_acc * inv_n, kl_acc * inv_n, clip_acc * inv_n };
}

// ── Shared-critic mini-batch update (local features, value clipping) ────────
//
// All agents' data feeds into the SAME critic update — this is the paper's
// "parameter sharing across critics" (de Witt 2020 §4): one set of φ weights
// trained on every agent's (z^a, V̂^a) pairs.
float IPPOPolicy::update_critic_mb(
    const std::vector<std::pair<int,int>>& perm, int start, int end)
{
    const int bsz = end - start;
    if (bsz <= 0) return 0.f;

    float dW1[kHid * kPolicySz]{};
    float db1[kHid]{};
    float dW2[kHid * kHid]{};
    float db2[kHid]{};
    float dW3[kHid]{};
    float db3 = 0.f;

    const float inv_n = 1.f / bsz;
    float mse_acc = 0.f;

    for (int pi = start; pi < end; ++pi) {
        const auto [aid, idx] = perm[pi];
        const Experience& e = buffers_.at(aid)[idx];

        float h1[kHid], h2[kHid], pa1[kHid], pa2[kHid];
        float V_new = critic.forward(e.obs.data(), h1, h2, pa1, pa2);
        float V_old = e.value;
        float ret   = e.ret;

        float V_clip   = V_old + std::clamp(V_new - V_old,
                                            -hparams.val_clip_eps,
                                             hparams.val_clip_eps);
        float err_new  = V_new  - ret;
        float err_clip = V_clip - ret;
        mse_acc += std::max(err_new * err_new, err_clip * err_clip);

        float dz3 = 2.f * err_new * inv_n;

        db3 += dz3;
        float dh2[kHid]{};
        for (int j = 0; j < kHid; ++j) {
            dW3[j] += dz3 * h2[j];
            dh2[j]  = dz3 * critic.W3[j];
        }

        float dh1[kHid]{};
        for (int i = 0; i < kHid; ++i) {
            float dz2 = dh2[i] * (pa2[i] > 0.f ? 1.f : 0.f);
            db2[i] += dz2;
            const float* row = critic.W2 + i * kHid;
            for (int j = 0; j < kHid; ++j) {
                dW2[i*kHid+j] += dz2 * h1[j];
                dh1[j]        += dz2 * row[j];
            }
        }
        for (int i = 0; i < kHid; ++i) {
            float dz1 = dh1[i] * (pa1[i] > 0.f ? 1.f : 0.f);
            db1[i] += dz1;
            for (int j = 0; j < kPolicySz; ++j)
                dW1[i*kPolicySz+j] += dz1 * e.obs[j];
        }
    }

    float* cg[] = { dW1, db1, dW2, db2, dW3, &db3 };
    int    cs[] = { kHid*kPolicySz, kHid, kHid*kHid, kHid, kHid, 1 };
    clip_grad_norm(cg, cs, 6, hparams.max_grad_norm);

    float lr = hparams.lr_critic;
    for (int i = 0; i < kHid*kPolicySz; ++i) critic.W1[i] -= lr * dW1[i];
    for (int i = 0; i < kHid; ++i)            critic.b1[i] -= lr * db1[i];
    for (int i = 0; i < kHid*kHid; ++i)       critic.W2[i] -= lr * dW2[i];
    for (int i = 0; i < kHid; ++i)            critic.b2[i] -= lr * db2[i];
    for (int i = 0; i < kHid; ++i)            critic.W3[i] -= lr * dW3[i];
    critic.b3 -= lr * db3;

    return mse_acc * inv_n;
}

// ── train_epoch — shared actor + shared critic PPO update ───────────────────
IPPOPolicy::TrainingStats IPPOPolicy::train_epoch() {
    TrainingStats ts;
    const int total = total_buffer_size();
    ts.n_exp = total;
    if (total == 0) return ts;

    // ── 1. Frozen V_old per buffer entry (shared critic, local obs). ───────
    for (auto& [aid, buf] : buffers_) {
        for (auto& e : buf)
            e.value = critic.forward(e.obs.data());
    }

    // ── 2. Per-agent GAE (per-trajectory reverse sweep). ───────────────────
    for (auto& [aid, buf] : buffers_) compute_gae(buf);

    // ── 3. Global advantage normalisation (shared actor + shared critic
    //      need consistent scales across agents). ─────────────────────────
    double adv_mean = 0.0;
    for (auto& [aid, buf] : buffers_)
        for (const auto& e : buf) adv_mean += e.advantage;
    adv_mean /= total;
    double adv_var = 0.0;
    for (auto& [aid, buf] : buffers_)
        for (const auto& e : buf) {
            double d = e.advantage - adv_mean;
            adv_var += d * d;
        }
    const float adv_std = std::sqrt(static_cast<float>(adv_var / total + 1e-8));
    constexpr float kAdvScale = 2.0f;
    for (auto& [aid, buf] : buffers_)
        for (auto& e : buf)
            e.advantage = kAdvScale
                        * (e.advantage - static_cast<float>(adv_mean))
                        / adv_std;

    ts.adv_mean = static_cast<float>(adv_mean);
    ts.adv_std  = adv_std;

    // ── 4. Build global permutation of (aid, buf_idx) pairs. ───────────────
    std::vector<std::pair<int,int>> perm;
    perm.reserve(total);
    for (const auto& [aid, buf] : buffers_)
        for (int i = 0; i < static_cast<int>(buf.size()); ++i)
            perm.emplace_back(aid, i);

    // ── 5. PPO mini-batch loop with KL early stopping. ─────────────────────
    const int bsz = std::max(1, std::min(hparams.batch_sz, total));
    float aloss_acc = 0.f, ent_acc = 0.f, closs_acc = 0.f;
    float kl_acc = 0.f, clip_acc = 0.f;
    int   n_actor_updates = 0, n_critic_updates = 0, epoch_run = 0;

    for (int ep = 0; ep < hparams.epochs; ++ep) {
        std::shuffle(perm.begin(), perm.end(), rng_);

        float kl_epoch = 0.f;
        int   n_batch  = 0;

        for (int s = 0; s < total; s += bsz) {
            const int e = std::min(s + bsz, total);

            // 5a. Shared-actor update on the full mini-batch.
            auto [al, ent, kl, cf] = update_actor_mb(perm, s, e);
            aloss_acc += al; ent_acc += ent;
            kl_epoch  += kl; clip_acc += cf;
            ++n_actor_updates;

            // 5b. Shared-critic update on the same mini-batch.
            closs_acc += update_critic_mb(perm, s, e);
            ++n_critic_updates;

            ++n_batch;
        }

        ++epoch_run;
        float mean_kl = kl_epoch / std::max(1, n_batch);
        kl_acc += mean_kl;

        if (mean_kl > hparams.target_kl) break;
    }

    if (n_actor_updates > 0) {
        ts.actor_loss = aloss_acc / n_actor_updates;
        ts.entropy    = ent_acc   / n_actor_updates;
        ts.clip_frac  = clip_acc  / n_actor_updates;
    }
    if (n_critic_updates > 0) {
        ts.critic_loss = closs_acc / n_critic_updates;
    }
    ts.kl_approx = (epoch_run > 0) ? kl_acc / epoch_run : 0.f;
    ts.n_epochs  = epoch_run;

    clear_buffer_all();
    return ts;
}

// ── Persistence ─────────────────────────────────────────────────────────────
// New magic for the paper-faithful (shared-critic) format. The old per-agent
// critics format is intentionally incompatible — loading an old checkpoint
// returns false rather than silently producing nonsense.
static constexpr uint32_t kMagicIPPOFaithful = 0xDEA110C3u;

void IPPOPolicy::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicIPPOFaithful, sizeof(kMagicIPPOFaithful), 1, f);

    // Shared actor.
    std::fwrite(actor.W1, sizeof(float), kHid * kPolicySz, f);
    std::fwrite(actor.b1, sizeof(float), kHid,             f);
    std::fwrite(actor.W2, sizeof(float), kHid * kHid,      f);
    std::fwrite(actor.b2, sizeof(float), kHid,             f);
    std::fwrite(actor.W3, sizeof(float), kHid,             f);
    std::fwrite(&actor.b3, sizeof(float), 1,               f);

    // Shared critic (single set of weights, local obs input).
    std::fwrite(critic.W1, sizeof(float), kHid * kPolicySz, f);
    std::fwrite(critic.b1, sizeof(float), kHid,             f);
    std::fwrite(critic.W2, sizeof(float), kHid * kHid,      f);
    std::fwrite(critic.b2, sizeof(float), kHid,             f);
    std::fwrite(critic.W3, sizeof(float), kHid,             f);
    std::fwrite(&critic.b3, sizeof(float), 1,               f);
    std::fclose(f);
}

bool IPPOPolicy::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1
        || magic != kMagicIPPOFaithful) {
        std::fclose(f); return false;
    }

    bool ok = true;
    ok = ok && std::fread(actor.W1, sizeof(float), kHid * kPolicySz, f) == static_cast<size_t>(kHid * kPolicySz);
    ok = ok && std::fread(actor.b1, sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(actor.W2, sizeof(float), kHid * kHid,      f) == static_cast<size_t>(kHid * kHid);
    ok = ok && std::fread(actor.b2, sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(actor.W3, sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(&actor.b3, sizeof(float), 1,               f) == 1;

    ok = ok && std::fread(critic.W1, sizeof(float), kHid * kPolicySz, f) == static_cast<size_t>(kHid * kPolicySz);
    ok = ok && std::fread(critic.b1, sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(critic.W2, sizeof(float), kHid * kHid,      f) == static_cast<size_t>(kHid * kHid);
    ok = ok && std::fread(critic.b2, sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(critic.W3, sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
    ok = ok && std::fread(&critic.b3, sizeof(float), 1,               f) == 1;

    std::fclose(f);
    return ok;
}
