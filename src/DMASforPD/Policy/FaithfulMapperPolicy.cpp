#include "FaithfulMapperPolicy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <numeric>

namespace {

// ── Xavier normal init (same as ObjectiveDMPolicy / MapperPolicy) ───────────
// Orthogonal init forwarder (Yu+2022 MAPPO recipe; FaithfulMAPPER inherits
// the same MAPPO hyperparameter set, so we use the same init scheme).
void xavier(float* W, int fan_in, int fan_out, std::mt19937& rng) {
    policy_optim::init_orthogonal(W, fan_out, fan_in, rng, std::sqrt(2.f));
}

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

// ── FaithfulMapperPolicy ────────────────────────────────────────────────────
FaithfulMapperPolicy::FaithfulMapperPolicy()
    : rng_(std::random_device{}()) {}

FaithfulMapperPolicy& FaithfulMapperPolicy::shared() {
    static FaithfulMapperPolicy instance;
    return instance;
}

void FaithfulMapperPolicy::set_progress(float progress) {
    const float p = std::clamp(progress, 0.f, 1.f);
    auto lerp = [p](float a, float b) { return a + p * (b - a); };
    hparams.lr_actor  = lerp(hparams.lr_actor_init,  hparams.lr_actor_min);
    hparams.lr_critic = lerp(hparams.lr_critic_init, hparams.lr_critic_min);
    hparams.ent_w     = lerp(hparams.ent_w_init,     hparams.ent_w_min);
}

void FaithfulMapperPolicy::init_xavier_all(std::mt19937& rng) {
    for (auto& [aid, state] : agents_) {
        state.actor.init_xavier(rng);
        state.critic.init_xavier(rng);
        state.buffer.clear();
    }
}

void FaithfulMapperPolicy::ensure_agents(int n_agents) {
    for (int i = 0; i < n_agents; ++i) (void)get_or_create(i);
}

FaithfulMapperPolicy::AgentState&
FaithfulMapperPolicy::get_or_create(int agent_id) {
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) return it->second;
    AgentState s;
    s.actor.init_xavier(rng_);
    s.critic.init_xavier(rng_);
    auto [ins, _] = agents_.emplace(agent_id, std::move(s));
    return ins->second;
}

float FaithfulMapperPolicy::score(int agent_id, const PolicyFeatures& features) {
    AgentState& a = get_or_create(agent_id);
    float x[kPolicySz];
    features.to_array(x);
    return a.actor.forward(x);
}

int FaithfulMapperPolicy::record(int agent_id, const PolicyFeatures& obs,
                                  float action, float reward) {
    AgentState& a = get_or_create(agent_id);
    float x[kPolicySz];
    obs.to_array(x);
    const float eps = 1e-8f;
    float mu = a.actor.forward(x);
    float lp = (action > 0.5f)
        ? std::logf(mu + eps) : std::logf(1.f - mu + eps);

    Experience e;
    obs.to_array(e.obs.data());
    e.agent_id = agent_id;
    e.action   = action;
    e.log_prob = lp;
    e.reward   = reward;
    a.buffer.push_back(e);
    const int idx = static_cast<int>(a.buffer.size()) - 1;
    recent_records_.emplace_back(agent_id, idx);
    return idx;
}

int FaithfulMapperPolicy::n_recent_records() const {
    return static_cast<int>(recent_records_.size());
}

std::pair<int,int> FaithfulMapperPolicy::recent_record(int i) const {
    return recent_records_[i];
}

void FaithfulMapperPolicy::clear_recent_records() {
    recent_records_.clear();
}

void FaithfulMapperPolicy::update_reward(int agent_id, int buf_idx, float reward) {
    auto it = agents_.find(agent_id);
    if (it == agents_.end()) return;
    auto& buf = it->second.buffer;
    if (buf_idx >= 0 && buf_idx < static_cast<int>(buf.size()))
        buf[buf_idx].reward = reward;
}

void FaithfulMapperPolicy::add_to_reward(int agent_id, int buf_idx, float delta) {
    auto it = agents_.find(agent_id);
    if (it == agents_.end()) return;
    auto& buf = it->second.buffer;
    if (buf_idx >= 0 && buf_idx < static_cast<int>(buf.size()))
        buf[buf_idx].reward += delta;
}

int FaithfulMapperPolicy::buffer_size(int agent_id) const {
    auto it = agents_.find(agent_id);
    return (it == agents_.end()) ? 0
        : static_cast<int>(it->second.buffer.size());
}

int FaithfulMapperPolicy::total_buffer_size() const {
    int total = 0;
    for (const auto& [aid, state] : agents_)
        total += static_cast<int>(state.buffer.size());
    return total;
}

void FaithfulMapperPolicy::clear_buffer(int agent_id) {
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) it->second.buffer.clear();
}

void FaithfulMapperPolicy::clear_buffer_all() {
    for (auto& [aid, state] : agents_) state.buffer.clear();
    recent_records_.clear();
}

// ── Per-agent GAE — single-trajectory reverse sweep ─────────────────────────
void FaithfulMapperPolicy::compute_gae(AgentState& a) {
    const int n = static_cast<int>(a.buffer.size());
    if (n == 0) return;
    float gae = 0.f, next_val = 0.f;
    for (int i = n - 1; i >= 0; --i) {
        float delta = a.buffer[i].reward
                    + hparams.gamma * next_val
                    - a.buffer[i].value;
        gae = delta + hparams.gamma * hparams.lam_gae * gae;
        a.buffer[i].advantage = gae;
        a.buffer[i].ret       = gae + a.buffer[i].value;
        next_val              = a.buffer[i].value;
    }
}

// ── Actor mini-batch step (PPO clip + entropy bonus) ─────────────────────────
FaithfulMapperPolicy::MBStats FaithfulMapperPolicy::update_actor_mb(
    AgentState& a,
    const std::vector<int>& perm, int start, int end)
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
        const Experience& e = a.buffer[perm[pi]];
        float h1[kHid], h2[kHid], pa1[kHid], pa2[kHid];
        float mu = a.actor.forward(e.obs.data(), h1, h2, pa1, pa2);

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
            dh2[j]  = dz3 * a.actor.W3[j];
        }

        float dh1[kHid]{};
        for (int i = 0; i < kHid; ++i) {
            float dz2 = dh2[i] * (pa2[i] > 0.f ? 1.f : 0.f);
            db2[i] += dz2;
            const float* row = a.actor.W2 + i * kHid;
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

    // Ascent → descent for Adam.
    for (int i = 0; i < kHid*kPolicySz; ++i) dW1[i] = -dW1[i];
    for (int i = 0; i < kHid; ++i)            db1[i] = -db1[i];
    for (int i = 0; i < kHid*kHid; ++i)       dW2[i] = -dW2[i];
    for (int i = 0; i < kHid; ++i)            db2[i] = -db2[i];
    for (int i = 0; i < kHid; ++i)            dW3[i] = -dW3[i];
    db3 = -db3;

    ++a.adam_t_actor;
    const float lr = hparams.lr_actor;
    policy_optim::adam_apply(a.actor.W1, dW1, a.a_W1_adam.m, a.a_W1_adam.v,
                             kHid*kPolicySz, lr, a.adam_t_actor);
    policy_optim::adam_apply(a.actor.b1, db1, a.a_b1_adam.m, a.a_b1_adam.v,
                             kHid,           lr, a.adam_t_actor);
    policy_optim::adam_apply(a.actor.W2, dW2, a.a_W2_adam.m, a.a_W2_adam.v,
                             kHid*kHid,      lr, a.adam_t_actor);
    policy_optim::adam_apply(a.actor.b2, db2, a.a_b2_adam.m, a.a_b2_adam.v,
                             kHid,           lr, a.adam_t_actor);
    policy_optim::adam_apply(a.actor.W3, dW3, a.a_W3_adam.m, a.a_W3_adam.v,
                             kHid,           lr, a.adam_t_actor);
    policy_optim::adam_apply_scalar(a.actor.b3, db3, a.a_b3_adam,
                                    a.adam_t_actor, lr);

    return { -(L_acc * inv_n), ent_acc * inv_n, kl_acc * inv_n, clip_acc * inv_n };
}

// ── Critic mini-batch step (local features, value clipping) ──────────────────
float FaithfulMapperPolicy::update_critic_mb(
    AgentState& a,
    const std::vector<int>& perm, int start, int end)
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

    const float v_mu      = a.value_rms.mean_f();
    const float v_std     = a.value_rms.std_dev();
    const float inv_v_std = 1.f / v_std;

    for (int pi = start; pi < end; ++pi) {
        int idx = perm[pi];
        const Experience& e = a.buffer[idx];

        float h1[kHid], h2[kHid], pa1[kHid], pa2[kHid];
        float V_new      = a.critic.forward(e.obs.data(), h1, h2, pa1, pa2);   // normalised
        float V_old_norm = (e.value - v_mu) * inv_v_std;
        float ret_norm   = (e.ret   - v_mu) * inv_v_std;

        float V_clip   = V_old_norm + std::clamp(V_new - V_old_norm,
                                                 -hparams.val_clip_eps,
                                                  hparams.val_clip_eps);
        float err_new  = V_new  - ret_norm;
        float err_clip = V_clip - ret_norm;

        // Pessimistic Huber loss (Yu+2022 Tab.7, δ=10).
        const float h_new  = policy_optim::huber_value(err_new);
        const float h_clip = policy_optim::huber_value(err_clip);
        mse_acc += std::max(h_new, h_clip);

        float dz3 = policy_optim::huber_grad(err_new) * inv_n;

        db3 += dz3;
        float dh2[kHid]{};
        for (int j = 0; j < kHid; ++j) {
            dW3[j] += dz3 * h2[j];
            dh2[j]  = dz3 * a.critic.W3[j];
        }

        float dh1[kHid]{};
        for (int i = 0; i < kHid; ++i) {
            float dz2 = dh2[i] * (pa2[i] > 0.f ? 1.f : 0.f);
            db2[i] += dz2;
            const float* row = a.critic.W2 + i * kHid;
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

    ++a.adam_t_critic;
    const float lr = hparams.lr_critic;
    policy_optim::adam_apply(a.critic.W1, dW1, a.c_W1_adam.m, a.c_W1_adam.v,
                             kHid*kPolicySz, lr, a.adam_t_critic);
    policy_optim::adam_apply(a.critic.b1, db1, a.c_b1_adam.m, a.c_b1_adam.v,
                             kHid,           lr, a.adam_t_critic);
    policy_optim::adam_apply(a.critic.W2, dW2, a.c_W2_adam.m, a.c_W2_adam.v,
                             kHid*kHid,      lr, a.adam_t_critic);
    policy_optim::adam_apply(a.critic.b2, db2, a.c_b2_adam.m, a.c_b2_adam.v,
                             kHid,           lr, a.adam_t_critic);
    policy_optim::adam_apply(a.critic.W3, dW3, a.c_W3_adam.m, a.c_W3_adam.v,
                             kHid,           lr, a.adam_t_critic);
    policy_optim::adam_apply_scalar(a.critic.b3, db3, a.c_b3_adam,
                                    a.adam_t_critic, lr);

    return mse_acc * inv_n;
}

// ── train_epoch — per-agent independent PPO update (same as MapperPolicy) ───
FaithfulMapperPolicy::TrainingStats FaithfulMapperPolicy::train_epoch() {
    TrainingStats ts;
    int total_exp = total_buffer_size();
    ts.n_exp = total_exp;
    if (total_exp == 0) return ts;

    double w_aloss = 0.0, w_closs = 0.0, w_ent = 0.0;
    double w_kl    = 0.0, w_cf    = 0.0;
    int    w_count = 0, w_epochs = 0;
    double w_amean = 0.0, w_astd  = 0.0;

    for (auto& [aid, state] : agents_) {
        const int n = static_cast<int>(state.buffer.size());
        if (n == 0) continue;

        // Frozen V_old — denormalise the critic's normalised output via the
        // rms snapshot it was last trained against.
        const float v_mu_old  = state.value_rms.mean_f();
        const float v_std_old = state.value_rms.std_dev();
        for (auto& e : state.buffer) {
            const float v_norm = state.critic.forward(e.obs.data());
            e.value = v_norm * v_std_old + v_mu_old;
        }

        compute_gae(state);

        // Update rms with the freshly-computed returns; this rms is used in
        // update_critic_mb below for normalised targets and next time as v_old.
        for (const auto& e : state.buffer) state.value_rms.update(e.ret);

        float adv_mean = 0.f;
        for (const auto& e : state.buffer) adv_mean += e.advantage;
        adv_mean /= n;
        float adv_var = 0.f;
        for (const auto& e : state.buffer) {
            float d = e.advantage - adv_mean;
            adv_var += d * d;
        }
        float adv_std = std::sqrt(adv_var / n + 1e-8f);
        constexpr float kAdvScale = 2.0f;
        for (auto& e : state.buffer)
            e.advantage = kAdvScale * (e.advantage - adv_mean) / adv_std;

        w_amean += adv_mean;
        w_astd  += adv_std;

        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        const int bsz = std::max(1, std::min(hparams.batch_sz, n));

        float aloss_acc = 0.f, ent_acc = 0.f, closs_acc = 0.f;
        float kl_acc = 0.f, clip_acc = 0.f;
        int   n_updates = 0, epoch_run = 0;

        for (int ep = 0; ep < hparams.epochs; ++ep) {
            std::shuffle(perm.begin(), perm.end(), rng_);

            float kl_epoch = 0.f;
            int   n_batch  = 0;

            for (int s = 0; s < n; s += bsz) {
                int e = std::min(s + bsz, n);
                auto [al, ent, kl, cf] = update_actor_mb(state, perm, s, e);
                float cl = update_critic_mb(state, perm, s, e);

                aloss_acc += al;  ent_acc   += ent;
                kl_epoch  += kl;  clip_acc  += cf;
                closs_acc += cl;
                ++n_batch;
            }

            ++epoch_run;
            n_updates += n_batch;
            float mean_kl = kl_epoch / std::max(1, n_batch);
            kl_acc += mean_kl;

            if (mean_kl > hparams.target_kl) break;
        }

        if (n_updates > 0) {
            w_aloss  += static_cast<double>(aloss_acc) / n_updates * n;
            w_ent    += static_cast<double>(ent_acc)   / n_updates * n;
            w_closs  += static_cast<double>(closs_acc) / n_updates * n;
            w_cf     += static_cast<double>(clip_acc)  / n_updates * n;
        }
        if (epoch_run > 0) {
            w_kl     += static_cast<double>(kl_acc) / epoch_run * n;
            w_epochs += epoch_run;
        }
        w_count += n;
    }

    if (w_count > 0) {
        ts.actor_loss  = static_cast<float>(w_aloss  / w_count);
        ts.critic_loss = static_cast<float>(w_closs  / w_count);
        ts.entropy     = static_cast<float>(w_ent    / w_count);
        ts.clip_frac   = static_cast<float>(w_cf     / w_count);
        ts.kl_approx   = static_cast<float>(w_kl     / w_count);
    }
    if (!agents_.empty()) {
        ts.adv_mean = static_cast<float>(w_amean / agents_.size());
        ts.adv_std  = static_cast<float>(w_astd  / agents_.size());
    }
    ts.n_epochs = (!agents_.empty())
        ? static_cast<int>(w_epochs / agents_.size()) : 0;

    record_episode_fitness();
    clear_buffer_all();
    return ts;
}

// ── Fitness tracking (same recipe as MapperPolicy) ──────────────────────────
void FaithfulMapperPolicy::record_episode_fitness() {
    const int W = std::max(1, ev_params.fitness_window);
    for (auto& [aid, state] : agents_) {
        if (state.buffer.empty()) continue;
        float sum = 0.f;
        for (const auto& e : state.buffer) sum += e.reward;
        float mean = sum / static_cast<float>(state.buffer.size());
        state.fitness_history.push_back(mean);
        while (static_cast<int>(state.fitness_history.size()) > W)
            state.fitness_history.erase(state.fitness_history.begin());
    }
}

float FaithfulMapperPolicy::current_fitness(int agent_id) const {
    auto it = agents_.find(agent_id);
    if (it == agents_.end() || it->second.fitness_history.empty())
        return -std::numeric_limits<float>::infinity();
    float sum = 0.f;
    for (float v : it->second.fitness_history) sum += v;
    return sum / static_cast<float>(it->second.fitness_history.size());
}

// ── Paper-faithful Evolutionary step (Liu et al. IROS 2020, Algorithm 1) ────
//
// Verbatim from the paper:
//   p_i = 1 - exp(η · R̄_i) / exp(η · R̄_best)
//   if m < p_i: Θ_i ← Θ_j        (j = argmax fitness)
//
// Notation:
//   R_i        = current_fitness(i)   (rolling-mean reward over the window)
//   R_best     = max_i R_i
//   R_min      = min_i R_i
//   R̄_i       = R_i / max(R_max - R_min, ε)        (normalised fitness)
//
// Crucially this differs from the enhanced MapperPolicy version in TWO ways:
//   • Single parent (the unique best agent), not a random pick from elites.
//   • No mutation noise — Θ_i ← Θ_j is an exact copy of the parent's weights.
int FaithfulMapperPolicy::evolutionary_step() {
    if (agents_.size() < 2) return 0;

    // Step 1 — gather (fitness, agent_id) pairs for agents with usable data.
    std::vector<std::pair<float, int>> stats;
    stats.reserve(agents_.size());
    for (const auto& [aid, state] : agents_) {
        if (state.fitness_history.empty()) continue;
        stats.emplace_back(current_fitness(aid), aid);
    }
    if (stats.size() < 2) return 0;

    // Step 2 — find best agent (argmax) and the (R_max - R_min) normaliser.
    auto best_it = std::max_element(
        stats.begin(), stats.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    const int   best_aid = best_it->second;
    const float R_best   = best_it->first;

    float R_min =  std::numeric_limits<float>::infinity();
    float R_max = -std::numeric_limits<float>::infinity();
    for (const auto& [r, _] : stats) {
        if (r < R_min) R_min = r;
        if (r > R_max) R_max = r;
    }
    const float denom = std::max(1e-8f, R_max - R_min);

    // Step 3 — probabilistic replacement (no mutation).
    const AgentState&             parent = agents_.at(best_aid);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    int n_replaced = 0;

    for (const auto& [r, aid] : stats) {
        if (aid == best_aid) continue;
        const float R_bar_i    = r      / denom;
        const float R_bar_best = R_best / denom;
        // p_i = 1 - exp(η·R̄_i - η·R̄_best). Since R̄_i ≤ R̄_best, the
        // exponent ≤ 0 → exp ≤ 1 → p_i ∈ [0, 1).
        const float p_i = 1.f - std::exp(
            ev_params.eta * (R_bar_i - R_bar_best));
        if (uni(rng_) < p_i) {
            AgentState& w = agents_.at(aid);
            // Exact copy — actor and critic weights only. The buffer is
            // already empty (cleared at end of train_epoch).
            w.actor  = parent.actor;
            w.critic = parent.critic;
            // Reset child's fitness window — its prior history belongs to
            // a different parameter regime now.
            w.fitness_history.clear();
            ++n_replaced;
        }
    }

    return n_replaced;
}

// ── Persistence (distinct magic from MapperPolicy) ──────────────────────────
// Magic bumped (0xDEA110C4 → 0xDEA110C5) — per-agent RunningMeanStd snapshot
// for value normalisation now persisted with each AgentState block.
static constexpr uint32_t kMagicMapperFaithful = 0xDEA110C5u;

void FaithfulMapperPolicy::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicMapperFaithful, sizeof(kMagicMapperFaithful), 1, f);

    int32_t n = static_cast<int32_t>(agents_.size());
    std::fwrite(&n, sizeof(n), 1, f);

    for (const auto& [aid, state] : agents_) {
        int32_t id = aid;
        std::fwrite(&id, sizeof(id), 1, f);
        std::fwrite(state.actor.W1, sizeof(float), kHid * kPolicySz, f);
        std::fwrite(state.actor.b1, sizeof(float), kHid,             f);
        std::fwrite(state.actor.W2, sizeof(float), kHid * kHid,      f);
        std::fwrite(state.actor.b2, sizeof(float), kHid,             f);
        std::fwrite(state.actor.W3, sizeof(float), kHid,             f);
        std::fwrite(&state.actor.b3, sizeof(float), 1,               f);
        std::fwrite(state.critic.W1, sizeof(float), kHid * kPolicySz, f);
        std::fwrite(state.critic.b1, sizeof(float), kHid,             f);
        std::fwrite(state.critic.W2, sizeof(float), kHid * kHid,      f);
        std::fwrite(state.critic.b2, sizeof(float), kHid,             f);
        std::fwrite(state.critic.W3, sizeof(float), kHid,             f);
        std::fwrite(&state.critic.b3, sizeof(float), 1,               f);
        state.value_rms.save_to(f);
    }
    std::fclose(f);
}

bool FaithfulMapperPolicy::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1
        || magic != kMagicMapperFaithful) {
        std::fclose(f); return false;
    }
    int32_t n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1) { std::fclose(f); return false; }

    agents_.clear();
    bool ok = true;
    for (int k = 0; k < n && ok; ++k) {
        int32_t id = 0;
        ok = ok && std::fread(&id, sizeof(id), 1, f) == 1;
        AgentState s;
        ok = ok && std::fread(s.actor.W1,  sizeof(float), kHid * kPolicySz, f) == static_cast<size_t>(kHid * kPolicySz);
        ok = ok && std::fread(s.actor.b1,  sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
        ok = ok && std::fread(s.actor.W2,  sizeof(float), kHid * kHid,      f) == static_cast<size_t>(kHid * kHid);
        ok = ok && std::fread(s.actor.b2,  sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
        ok = ok && std::fread(s.actor.W3,  sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
        ok = ok && std::fread(&s.actor.b3, sizeof(float), 1,                f) == 1;
        ok = ok && std::fread(s.critic.W1, sizeof(float), kHid * kPolicySz, f) == static_cast<size_t>(kHid * kPolicySz);
        ok = ok && std::fread(s.critic.b1, sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
        ok = ok && std::fread(s.critic.W2, sizeof(float), kHid * kHid,      f) == static_cast<size_t>(kHid * kHid);
        ok = ok && std::fread(s.critic.b2, sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
        ok = ok && std::fread(s.critic.W3, sizeof(float), kHid,             f) == static_cast<size_t>(kHid);
        ok = ok && std::fread(&s.critic.b3, sizeof(float), 1,               f) == 1;
        ok = ok && s.value_rms.load_from(f);
        if (ok) agents_.emplace(id, std::move(s));
    }
    std::fclose(f);
    return ok;
}
