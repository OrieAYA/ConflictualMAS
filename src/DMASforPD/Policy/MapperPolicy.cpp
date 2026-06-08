#include "MapperPolicy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <numeric>

namespace {

// ── Orthogonal init (Yu+2022 MAPPO Tab.7) ────────────────────────────────────
// MAPPER follows MAPPO's recipe; forwarder kept under the `xavier` name to keep
// existing call sites untouched.
void xavier(float* W, int fan_in, int fan_out, std::mt19937& rng) {
    policy_optim::init_orthogonal(W, fan_out, fan_in, rng, std::sqrt(2.f));
}

void linrelu(const float* W, const float* b,
             const float* x, int out_n, int in_n,
             float* y, float* pa) {
    for (int i = 0; i < out_n; ++i) {
        float s = b[i];
        const float* row = W + i * in_n;
        for (int j = 0; j < in_n; ++j) s += row[j] * x[j];
        if (pa) pa[i] = s;
        y[i] = s > 0.f ? s : 0.f;
    }
}

float linout(const float* W, float b, const float* x, int in_n) {
    float s = b;
    for (int j = 0; j < in_n; ++j) s += W[j] * x[j];
    return s;
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

// ── MapperCriticMLP ─────────────────────────────────────────────────────────
void MapperCriticMLP::init_xavier(std::mt19937& rng) {
    xavier(W1, kPolicySz, kHid, rng);
    xavier(W2, kHid, kHid, rng);
    xavier(W3, kHid, 1, rng);
    std::fill(b1, b1 + kHid, 0.f);
    std::fill(b2, b2 + kHid, 0.f);
    b3 = 0.f;
}

float MapperCriticMLP::forward(const float* x,
                                float* h1, float* h2,
                                float* pa1, float* pa2) const {
    float _h1[kHid], _h2[kHid], _pa1[kHid], _pa2[kHid];
    if (!h1)  h1  = _h1;
    if (!h2)  h2  = _h2;
    if (!pa1) pa1 = _pa1;
    if (!pa2) pa2 = _pa2;

    linrelu(W1, b1, x,  kHid, kPolicySz, h1, pa1);
    linrelu(W2, b2, h1, kHid, kHid,      h2, pa2);
    return linout(W3, b3, h2, kHid);
}

// ── MapperPolicy ────────────────────────────────────────────────────────────
MapperPolicy::MapperPolicy()
    : rng_(std::random_device{}()) {}

MapperPolicy& MapperPolicy::shared() {
    static MapperPolicy instance;
    return instance;
}

void MapperPolicy::set_progress(float progress) {
    const float p = std::clamp(progress, 0.f, 1.f);
    auto lerp = [p](float a, float b) { return a + p * (b - a); };
    hparams.lr_actor  = lerp(hparams.lr_actor_init,  hparams.lr_actor_min);
    hparams.lr_critic = lerp(hparams.lr_critic_init, hparams.lr_critic_min);
    hparams.ent_w     = lerp(hparams.ent_w_init,     hparams.ent_w_min);
}

void MapperPolicy::init_xavier_all(std::mt19937& rng) {
    for (auto& [aid, state] : agents_) {
        state.actor.init_xavier(rng);
        state.critic.init_xavier(rng);
        state.buffer.clear();
    }
}

void MapperPolicy::ensure_agents(int n_agents) {
    for (int i = 0; i < n_agents; ++i) (void)get_or_create(i);
}

MapperPolicy::AgentState& MapperPolicy::get_or_create(int agent_id) {
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) return it->second;
    AgentState s;
    s.actor.init_xavier(rng_);
    s.critic.init_xavier(rng_);
    auto [ins, _] = agents_.emplace(agent_id, std::move(s));
    return ins->second;
}

float MapperPolicy::score(int agent_id, const PolicyFeatures& features) {
    AgentState& a = get_or_create(agent_id);
    float x[kPolicySz];
    features.to_array(x);
    return a.actor.forward(x);
}

int MapperPolicy::record(int agent_id, const PolicyFeatures& obs,
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

int MapperPolicy::n_recent_records() const {
    return static_cast<int>(recent_records_.size());
}

std::pair<int,int> MapperPolicy::recent_record(int i) const {
    return recent_records_[i];
}

void MapperPolicy::clear_recent_records() {
    recent_records_.clear();
}

void MapperPolicy::update_reward(int agent_id, int buf_idx, float reward) {
    auto it = agents_.find(agent_id);
    if (it == agents_.end()) return;
    auto& buf = it->second.buffer;
    if (buf_idx >= 0 && buf_idx < static_cast<int>(buf.size()))
        buf[buf_idx].reward = reward;
}

void MapperPolicy::add_to_reward(int agent_id, int buf_idx, float delta) {
    auto it = agents_.find(agent_id);
    if (it == agents_.end()) return;
    auto& buf = it->second.buffer;
    if (buf_idx >= 0 && buf_idx < static_cast<int>(buf.size()))
        buf[buf_idx].reward += delta;
}

int MapperPolicy::buffer_size(int agent_id) const {
    auto it = agents_.find(agent_id);
    return (it == agents_.end()) ? 0
        : static_cast<int>(it->second.buffer.size());
}

int MapperPolicy::total_buffer_size() const {
    int total = 0;
    for (const auto& [aid, state] : agents_)
        total += static_cast<int>(state.buffer.size());
    return total;
}

void MapperPolicy::clear_buffer(int agent_id) {
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) it->second.buffer.clear();
}

void MapperPolicy::clear_buffer_all() {
    for (auto& [aid, state] : agents_) state.buffer.clear();
    recent_records_.clear();
}

// ── Per-agent GAE — single-trajectory reverse sweep ─────────────────────────
void MapperPolicy::compute_gae(AgentState& a) {
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
MapperPolicy::MBStats MapperPolicy::update_actor_mb(
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

    // Ascent → descent sign flip for Adam.
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
float MapperPolicy::update_critic_mb(
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

    for (int pi = start; pi < end; ++pi) {
        int idx = perm[pi];
        const Experience& e = a.buffer[idx];

        float h1[kHid], h2[kHid], pa1[kHid], pa2[kHid];
        float V_new = a.critic.forward(e.obs.data(), h1, h2, pa1, pa2);
        float V_old = e.value;   // snapshot taken at start of train_epoch
        float ret   = e.ret;

        float V_clip   = V_old + std::clamp(V_new - V_old,
                                            -hparams.val_clip_eps,
                                             hparams.val_clip_eps);
        float err_new  = V_new  - ret;
        float err_clip = V_clip - ret;

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

// ── train_epoch — per-agent independent PPO update ──────────────────────────
MapperPolicy::TrainingStats MapperPolicy::train_epoch() {
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

        // 1. Frozen V_old per buffer entry.
        for (auto& e : state.buffer)
            e.value = state.critic.forward(e.obs.data());

        // 2. Per-agent GAE.
        compute_gae(state);

        // 3. Per-agent advantage normalisation (×2 amplification as in MAPPO).
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

        // 4. Mini-batch PPO with KL early stopping.
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

    // Record this episode's per-agent fitness BEFORE clearing buffers, so the
    // evolutionary step can rank policies on their actual behavioural reward.
    record_episode_fitness();

    clear_buffer_all();
    return ts;
}

// ── Fitness tracking ────────────────────────────────────────────────────────
//
// Fitness for one episode = mean over the agent's buffered rewards in this
// episode. Rewards already encode all L-GPDP objectives (delivery credit,
// pickup credit, refusal penalty, unfinished-task penalty), so a single
// scalar captures throughput, refusal accuracy, and unfinished avoidance.
//
// We push the mean onto a rolling FIFO of length `ev_params.fitness_window`.
// Agents whose buffer is empty for the episode are skipped — they get no
// fitness sample for the window.
void MapperPolicy::record_episode_fitness() {
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

float MapperPolicy::current_fitness(int agent_id) const {
    auto it = agents_.find(agent_id);
    if (it == agents_.end() || it->second.fitness_history.empty())
        return -std::numeric_limits<float>::infinity();
    float sum = 0.f;
    for (float v : it->second.fitness_history) sum += v;
    return sum / static_cast<float>(it->second.fitness_history.size());
}

// ── Evolutionary step (the "ER" in MAPPER) ──────────────────────────────────
//
// Algorithm (per Liu et al., adapted for L-GPDP):
//
//   1. Compute fitness_i = mean(fitness_history[i]) for every tracked agent.
//      Skip agents whose history is empty.
//   2. Sort by fitness descending; let n be the # of agents with valid data.
//   3. n_elite = max(1, round(n × elite_frac))     ← top performers
//      n_worst = round(n × worst_frac)             ← bottom performers
//   4. For each worst agent w:
//        - Pick a random elite e
//        - Copy actor + critic weights from e to w
//        - Add ε ~ N(0, mutation_std²) to every weight (including biases)
//        - Clear w.fitness_history so the child gets fresh evaluation
//
// Combines RL's gradient updates with genetic-style exploration: lets a
// policy stuck in a local optimum "jump" to the basin of a stronger policy
// while preserving the elite gene pool intact. This is the explicit
// exploration mechanism that distinguishes MAPPER from vanilla decentralised
// A2C / MARL.
int MapperPolicy::evolutionary_step() {
    if (agents_.size() < 2) return 0;

    // Step 1 — collect (fitness, agent_id) for agents with usable data.
    std::vector<std::pair<float, int>> rank;
    rank.reserve(agents_.size());
    for (const auto& [aid, state] : agents_) {
        if (state.fitness_history.empty()) continue;
        rank.emplace_back(current_fitness(aid), aid);
    }
    if (rank.size() < 2) return 0;

    // Step 2 — sort descending so rank[0] is the best.
    std::sort(rank.begin(), rank.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    const int n        = static_cast<int>(rank.size());
    const int n_elite  = std::max(1,
        static_cast<int>(std::round(n * ev_params.elite_frac)));
    const int n_worst  = static_cast<int>(std::round(n * ev_params.worst_frac));
    if (n_worst <= 0 || n_elite <= 0) return 0;

    // Step 3 — mutate.
    std::uniform_int_distribution<int> elite_picker(0, n_elite - 1);
    std::normal_distribution<float>    noise(0.f, ev_params.mutation_std);
    int n_mutated = 0;

    for (int i = n - n_worst; i < n; ++i) {
        const int worst_aid = rank[i].second;
        const int elite_aid = rank[elite_picker(rng_)].second;
        if (worst_aid == elite_aid) continue;

        AgentState&       w = agents_.at(worst_aid);
        const AgentState& e = agents_.at(elite_aid);

        // Actor weights ← elite + noise.
        for (int j = 0; j < kHid * kPolicySz; ++j)
            w.actor.W1[j] = e.actor.W1[j] + noise(rng_);
        for (int j = 0; j < kHid; ++j)
            w.actor.b1[j] = e.actor.b1[j] + noise(rng_);
        for (int j = 0; j < kHid * kHid; ++j)
            w.actor.W2[j] = e.actor.W2[j] + noise(rng_);
        for (int j = 0; j < kHid; ++j)
            w.actor.b2[j] = e.actor.b2[j] + noise(rng_);
        for (int j = 0; j < kHid; ++j)
            w.actor.W3[j] = e.actor.W3[j] + noise(rng_);
        w.actor.b3 = e.actor.b3 + noise(rng_);

        // Critic weights ← elite + noise.
        for (int j = 0; j < kHid * kPolicySz; ++j)
            w.critic.W1[j] = e.critic.W1[j] + noise(rng_);
        for (int j = 0; j < kHid; ++j)
            w.critic.b1[j] = e.critic.b1[j] + noise(rng_);
        for (int j = 0; j < kHid * kHid; ++j)
            w.critic.W2[j] = e.critic.W2[j] + noise(rng_);
        for (int j = 0; j < kHid; ++j)
            w.critic.b2[j] = e.critic.b2[j] + noise(rng_);
        for (int j = 0; j < kHid; ++j)
            w.critic.W3[j] = e.critic.W3[j] + noise(rng_);
        w.critic.b3 = e.critic.b3 + noise(rng_);

        // Reset the mutated child's fitness window — its prior history
        // belongs to a different parameter regime.
        w.fitness_history.clear();
        ++n_mutated;
    }

    return n_mutated;
}

// ── Persistence ─────────────────────────────────────────────────────────────
static constexpr uint32_t kMagicMapper = 0xDEA110C1u;

void MapperPolicy::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicMapper, sizeof(kMagicMapper), 1, f);

    int32_t n = static_cast<int32_t>(agents_.size());
    std::fwrite(&n, sizeof(n), 1, f);

    for (const auto& [aid, state] : agents_) {
        int32_t id = aid;
        std::fwrite(&id, sizeof(id), 1, f);
        // Actor
        std::fwrite(state.actor.W1, sizeof(float), kHid * kPolicySz, f);
        std::fwrite(state.actor.b1, sizeof(float), kHid,             f);
        std::fwrite(state.actor.W2, sizeof(float), kHid * kHid,      f);
        std::fwrite(state.actor.b2, sizeof(float), kHid,             f);
        std::fwrite(state.actor.W3, sizeof(float), kHid,             f);
        std::fwrite(&state.actor.b3, sizeof(float), 1,               f);
        // Critic (local-features)
        std::fwrite(state.critic.W1, sizeof(float), kHid * kPolicySz, f);
        std::fwrite(state.critic.b1, sizeof(float), kHid,             f);
        std::fwrite(state.critic.W2, sizeof(float), kHid * kHid,      f);
        std::fwrite(state.critic.b2, sizeof(float), kHid,             f);
        std::fwrite(state.critic.W3, sizeof(float), kHid,             f);
        std::fwrite(&state.critic.b3, sizeof(float), 1,               f);
    }
    std::fclose(f);
}

bool MapperPolicy::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != kMagicMapper) {
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
        if (ok) agents_.emplace(id, std::move(s));
    }
    std::fclose(f);
    return ok;
}
