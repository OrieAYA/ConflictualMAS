#include "PolicyKit.hpp"
#include <algorithm>
#include <numeric>

// ── RunningMeanStd persistence ───────────────────────────────────────────────
void RunningMeanStd::save_to(std::FILE* f) const {
    if (!f) return;
    std::fwrite(&mean,  sizeof(mean),  1, f);
    std::fwrite(&var,   sizeof(var),   1, f);
    std::fwrite(&count, sizeof(count), 1, f);
}
bool RunningMeanStd::load_from(std::FILE* f) {
    if (!f) return false;
    bool ok = true;
    ok = ok && std::fread(&mean,  sizeof(mean),  1, f) == 1;
    ok = ok && std::fread(&var,   sizeof(var),   1, f) == 1;
    ok = ok && std::fread(&count, sizeof(count), 1, f) == 1;
    return ok;
}

// ── Mlp ──────────────────────────────────────────────────────────────────────
void Mlp::init(int input_dim, std::mt19937& rng, float out_gain, Init kind) {
    in_dim = input_dim;
    W1.assign(static_cast<size_t>(kHid) * in_dim, 0.f);
    b1.assign(kHid, 0.f);
    W2.assign(static_cast<size_t>(kHid) * kHid, 0.f);
    b2.assign(kHid, 0.f);
    W3.assign(kHid, 0.f);
    b3 = 0.f;

    if (kind == Init::Orthogonal) {
        policy_optim::init_orthogonal(W1.data(), kHid, in_dim, rng, std::sqrt(2.f));
        policy_optim::init_orthogonal(W2.data(), kHid, kHid,   rng, std::sqrt(2.f));
        policy_optim::init_orthogonal(W3.data(), 1,    kHid,   rng, std::sqrt(2.f));
    } else {
        policy_optim::init_he_truncated(W1.data(), in_dim, kHid, rng);
        policy_optim::init_he_truncated(W2.data(), kHid,   kHid, rng);
        policy_optim::init_he_truncated(W3.data(), kHid,   1,    rng);
    }
    for (int i = 0; i < kHid; ++i) W3[i] *= out_gain;
}

float Mlp::forward(const float* x,
                   float* h1, float* h2, float* pa1, float* pa2) const {
    float _h1[kHid], _h2[kHid], _pa1[kHid], _pa2[kHid];
    if (!h1)  h1  = _h1;
    if (!h2)  h2  = _h2;
    if (!pa1) pa1 = _pa1;
    if (!pa2) pa2 = _pa2;

    for (int i = 0; i < kHid; ++i) {
        float s = b1[i];
        const float* row = W1.data() + static_cast<size_t>(i) * in_dim;
        for (int j = 0; j < in_dim; ++j) s += row[j] * x[j];
        pa1[i] = s;
        h1[i]  = s > 0.f ? s : 0.f;
    }
    for (int i = 0; i < kHid; ++i) {
        float s = b2[i];
        const float* row = W2.data() + static_cast<size_t>(i) * kHid;
        for (int j = 0; j < kHid; ++j) s += row[j] * h1[j];
        pa2[i] = s;
        h2[i]  = s > 0.f ? s : 0.f;
    }
    float out = b3;
    for (int j = 0; j < kHid; ++j) out += W3[j] * h2[j];
    return out;
}

void Mlp::save_to(std::FILE* f) const {
    if (!f) return;
    std::fwrite(W1.data(), sizeof(float), W1.size(), f);
    std::fwrite(b1.data(), sizeof(float), b1.size(), f);
    std::fwrite(W2.data(), sizeof(float), W2.size(), f);
    std::fwrite(b2.data(), sizeof(float), b2.size(), f);
    std::fwrite(W3.data(), sizeof(float), W3.size(), f);
    std::fwrite(&b3, sizeof(float), 1, f);
}
bool Mlp::load_from(std::FILE* f) {
    if (!f || in_dim <= 0) return false;
    bool ok = true;
    ok = ok && std::fread(W1.data(), sizeof(float), W1.size(), f) == W1.size();
    ok = ok && std::fread(b1.data(), sizeof(float), b1.size(), f) == b1.size();
    ok = ok && std::fread(W2.data(), sizeof(float), W2.size(), f) == W2.size();
    ok = ok && std::fread(b2.data(), sizeof(float), b2.size(), f) == b2.size();
    ok = ok && std::fread(W3.data(), sizeof(float), W3.size(), f) == W3.size();
    ok = ok && std::fread(&b3, sizeof(float), 1, f) == 1;
    return ok;
}

// ── MlpGrads ─────────────────────────────────────────────────────────────────
void MlpGrads::resize_for(const Mlp& net) {
    dW1.assign(net.W1.size(), 0.f);
    db1.assign(net.b1.size(), 0.f);
    dW2.assign(net.W2.size(), 0.f);
    db2.assign(net.b2.size(), 0.f);
    dW3.assign(net.W3.size(), 0.f);
    db3 = 0.f;
}
void MlpGrads::zero() {
    std::fill(dW1.begin(), dW1.end(), 0.f);
    std::fill(db1.begin(), db1.end(), 0.f);
    std::fill(dW2.begin(), dW2.end(), 0.f);
    std::fill(db2.begin(), db2.end(), 0.f);
    std::fill(dW3.begin(), dW3.end(), 0.f);
    db3 = 0.f;
}
void MlpGrads::clip_global_norm(float max_norm) {
    double sq = 0.0;
    auto acc = [&sq](const std::vector<float>& v) {
        for (float x : v) sq += static_cast<double>(x) * x;
    };
    acc(dW1); acc(db1); acc(dW2); acc(db2); acc(dW3);
    sq += static_cast<double>(db3) * db3;
    const float norm = static_cast<float>(std::sqrt(sq));
    if (norm <= max_norm) return;
    const float s = max_norm / (norm + 1e-6f);
    auto scale = [s](std::vector<float>& v) { for (float& x : v) x *= s; };
    scale(dW1); scale(db1); scale(dW2); scale(db2); scale(dW3);
    db3 *= s;
}
void MlpGrads::negate() {
    auto neg = [](std::vector<float>& v) { for (float& x : v) x = -x; };
    neg(dW1); neg(db1); neg(dW2); neg(db2); neg(dW3);
    db3 = -db3;
}

// ── MlpAdam ──────────────────────────────────────────────────────────────────
void MlpAdam::resize_for(const Mlp& net) {
    mW1.assign(net.W1.size(), 0.f); vW1.assign(net.W1.size(), 0.f);
    mb1.assign(net.b1.size(), 0.f); vb1.assign(net.b1.size(), 0.f);
    mW2.assign(net.W2.size(), 0.f); vW2.assign(net.W2.size(), 0.f);
    mb2.assign(net.b2.size(), 0.f); vb2.assign(net.b2.size(), 0.f);
    mW3.assign(net.W3.size(), 0.f); vW3.assign(net.W3.size(), 0.f);
    b3 = {};
    t  = 0;
}
void MlpAdam::apply(Mlp& net, const MlpGrads& g, float lr) {
    ++t;
    policy_optim::adam_apply(net.W1.data(), g.dW1.data(), mW1.data(), vW1.data(),
                             static_cast<int>(net.W1.size()), lr, t);
    policy_optim::adam_apply(net.b1.data(), g.db1.data(), mb1.data(), vb1.data(),
                             static_cast<int>(net.b1.size()), lr, t);
    policy_optim::adam_apply(net.W2.data(), g.dW2.data(), mW2.data(), vW2.data(),
                             static_cast<int>(net.W2.size()), lr, t);
    policy_optim::adam_apply(net.b2.data(), g.db2.data(), mb2.data(), vb2.data(),
                             static_cast<int>(net.b2.size()), lr, t);
    policy_optim::adam_apply(net.W3.data(), g.dW3.data(), mW3.data(), vW3.data(),
                             static_cast<int>(net.W3.size()), lr, t);
    policy_optim::adam_apply_scalar(net.b3, g.db3, b3, t, lr);
}

// ── backprop_sample ──────────────────────────────────────────────────────────
void backprop_sample(const Mlp& net, const float* x,
                     const float* h1, const float* h2,
                     const float* pa1, const float* pa2,
                     float dz3, MlpGrads& g) {
    g.db3 += dz3;
    float dh2[kHid];
    for (int j = 0; j < kHid; ++j) {
        g.dW3[j] += dz3 * h2[j];
        dh2[j]    = dz3 * net.W3[j];
    }
    float dh1[kHid] = {};
    for (int i = 0; i < kHid; ++i) {
        const float dz2 = dh2[i] * (pa2[i] > 0.f ? 1.f : 0.f);
        g.db2[i] += dz2;
        const float* row = net.W2.data() + static_cast<size_t>(i) * kHid;
        float* grow      = g.dW2.data()  + static_cast<size_t>(i) * kHid;
        for (int j = 0; j < kHid; ++j) {
            grow[j] += dz2 * h1[j];
            dh1[j]  += dz2 * row[j];
        }
    }
    for (int i = 0; i < kHid; ++i) {
        const float dz1 = dh1[i] * (pa1[i] > 0.f ? 1.f : 0.f);
        g.db1[i] += dz1;
        float* grow = g.dW1.data() + static_cast<size_t>(i) * net.in_dim;
        for (int j = 0; j < net.in_dim; ++j) grow[j] += dz1 * x[j];
    }
}

// ── GAE ──────────────────────────────────────────────────────────────────────
void compute_gae(std::vector<Experience>& traj, float gamma, float lam) {
    float gae = 0.f, next_val = 0.f;
    for (int k = static_cast<int>(traj.size()) - 1; k >= 0; --k) {
        Experience& e = traj[k];
        const float delta = e.reward + gamma * next_val - e.value;
        gae         = delta + gamma * lam * gae;
        e.advantage = gae;
        e.ret       = gae + e.value;
        next_val    = e.value;
    }
}

// ── ppo_train ────────────────────────────────────────────────────────────────
TrainingStats ppo_train(
    Mlp& actor,  MlpAdam& actor_opt,
    Mlp* critic, MlpAdam* critic_opt,
    RunningMeanStd& vrms,
    PPOParams& hp,
    const std::vector<std::vector<Experience>*>& trajs,
    const std::function<void(const Experience&, float*)>& build_critic_input,
    std::mt19937& rng)
{
    TrainingStats ts;

    std::vector<Experience*> items;
    for (auto* traj : trajs)
        for (auto& e : *traj) items.push_back(&e);
    const int n = static_cast<int>(items.size());
    ts.n_exp = n;
    if (n == 0) return ts;

    const float eps = 1e-8f;

    // 1. Frozen V_old (denormalised through the rms the critic was last
    //    trained against) — used for GAE and value clipping.
    const float v_mu_old  = vrms.mean_f();
    const float v_std_old = vrms.std_dev();
    std::vector<float> critic_in;
    if (critic) {
        critic_in.resize(critic->in_dim);
        for (Experience* e : items) {
            build_critic_input(*e, critic_in.data());
            e->value = critic->forward(critic_in.data()) * v_std_old + v_mu_old;
        }
    } else {
        for (Experience* e : items) e->value = 0.f;
    }

    // 2. Per-trajectory GAE.
    for (auto* traj : trajs) compute_gae(*traj, hp.gamma, hp.lam_gae);

    // 3. Update the running return statistics (raw scale).
    if (critic)
        for (const Experience* e : items) vrms.update(e->ret);

    // 4. Global advantage normalisation.
    float adv_mean = 0.f;
    for (const Experience* e : items) adv_mean += e->advantage;
    adv_mean /= n;
    float adv_var = 0.f;
    for (const Experience* e : items) {
        const float d = e->advantage - adv_mean;
        adv_var += d * d;
    }
    const float adv_std = std::sqrt(adv_var / n + 1e-8f);
    ts.adv_mean = adv_mean;
    ts.adv_std  = adv_std;
    for (Experience* e : items)
        e->advantage = (e->advantage - adv_mean) / adv_std;

    // 5. Epoch loop — full-batch SGD with PPO clip / value clip / KL stop.
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    const int bsz = std::max(1, std::min(hp.batch_sz, n));

    MlpGrads a_g; a_g.resize_for(actor);
    MlpGrads c_g; if (critic) c_g.resize_for(*critic);

    const float v_mu  = vrms.mean_f();
    const float v_std = vrms.std_dev();
    const float inv_v_std = 1.f / v_std;

    float aloss_acc = 0.f, ent_acc = 0.f, closs_acc = 0.f;
    float kl_acc = 0.f, clip_acc = 0.f;
    int   n_updates = 0, epoch_run = 0;

    float h1[kHid], h2[kHid], pa1[kHid], pa2[kHid];

    for (int ep = 0; ep < hp.epochs; ++ep) {
        std::shuffle(perm.begin(), perm.end(), rng);

        float kl_epoch = 0.f;
        int   n_batch  = 0;

        for (int s = 0; s < n; s += bsz) {
            const int e_end = std::min(s + bsz, n);
            const int cur   = e_end - s;
            const float inv = 1.f / cur;

            // ── Actor mini-batch ─────────────────────────────────────────
            a_g.zero();
            float L = 0.f, H = 0.f, kl = 0.f, cf = 0.f;
            for (int pi = s; pi < e_end; ++pi) {
                const Experience& e = *items[perm[pi]];
                const float mu = Mlp::sigmoid(
                    actor.forward(e.obs.data(), h1, h2, pa1, pa2));
                const float lp_new = (e.action > 0.5f)
                    ? std::log(mu + eps) : std::log(1.f - mu + eps);
                const float r = std::exp(lp_new - e.log_prob);
                const float A = e.advantage;

                kl += e.log_prob - lp_new;
                const bool clipped = (r > 1.f + hp.clip_eps && A > 0.f)
                                  || (r < 1.f - hp.clip_eps && A < 0.f);
                if (clipped) cf += 1.f;

                const float clip_grad = clipped ? 0.f : A * r * (e.action - mu);
                const float ent_grad  = hp.ent_w
                    * std::log((1.f - mu + eps) / (mu + eps)) * mu * (1.f - mu);
                L += clipped
                    ? std::clamp(r, 1.f - hp.clip_eps, 1.f + hp.clip_eps) * A
                    : r * A;
                H += -(mu * std::log(mu + eps)
                     + (1.f - mu) * std::log(1.f - mu + eps));

                backprop_sample(actor, e.obs.data(), h1, h2, pa1, pa2,
                                (clip_grad + ent_grad) * inv, a_g);
            }
            a_g.clip_global_norm(hp.max_grad_norm);
            a_g.negate();                       // we accumulated the ASCENT direction
            actor_opt.apply(actor, a_g, hp.lr_actor);

            // ── Critic mini-batch (normalised space, value clip, Huber) ──
            float cl = 0.f;
            if (critic) {
                c_g.zero();
                for (int pi = s; pi < e_end; ++pi) {
                    const Experience& e = *items[perm[pi]];
                    build_critic_input(e, critic_in.data());
                    const float V_new = critic->forward(critic_in.data(),
                                                        h1, h2, pa1, pa2);
                    const float V_old_n = (e.value - v_mu) * inv_v_std;
                    const float ret_n   = (e.ret   - v_mu) * inv_v_std;
                    const float V_clip  = V_old_n + std::clamp(
                        V_new - V_old_n, -hp.val_clip_eps, hp.val_clip_eps);
                    const float err_new  = V_new  - ret_n;
                    const float err_clip = V_clip - ret_n;
                    cl += std::max(policy_optim::huber_value(err_new),
                                   policy_optim::huber_value(err_clip));
                    backprop_sample(*critic, critic_in.data(), h1, h2, pa1, pa2,
                                    policy_optim::huber_grad(err_new) * inv, c_g);
                }
                c_g.clip_global_norm(hp.max_grad_norm);
                critic_opt->apply(*critic, c_g, hp.lr_critic);
                cl *= inv;
            }

            aloss_acc += -(L * inv);
            ent_acc   += H * inv;
            kl_epoch  += kl * inv;
            clip_acc  += cf * inv;
            closs_acc += cl;
            ++n_batch;
        }

        ++epoch_run;
        n_updates += n_batch;
        const float mean_kl = kl_epoch / std::max(1, n_batch);
        kl_acc += mean_kl;
        if (mean_kl > hp.target_kl) break;
    }

    if (n_updates > 0) {
        ts.actor_loss  = aloss_acc / n_updates;
        ts.entropy     = ent_acc   / n_updates;
        ts.critic_loss = closs_acc / n_updates;
        ts.clip_frac   = clip_acc  / n_updates;
    }
    ts.kl_approx = (epoch_run > 0) ? kl_acc / epoch_run : 0.f;
    ts.n_epochs  = epoch_run;
    return ts;
}
