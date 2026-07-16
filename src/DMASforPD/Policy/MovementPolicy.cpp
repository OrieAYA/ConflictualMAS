#include "DMASforPD/Policy/MovementPolicy.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

// ── StarNet ──────────────────────────────────────────────────────────────────

void StarNet::init(std::mt19937& rng, float out_gain) {
    We.assign(static_cast<size_t>(kMoveEnc) * kMoveEdgeFeat, 0.f);
    be.assign(kMoveEnc, 0.f);
    policy_optim::init_orthogonal(We.data(), kMoveEnc, kMoveEdgeFeat, rng,
                                  std::sqrt(2.f));
    head.init(kMovePooled, rng, out_gain, Mlp::Init::Orthogonal);
}

float StarNet::forward(const MoveObs& o, StarCache* c) const {
    StarCache local;
    if (!c) c = &local;
    c->obs = o.x.data();
    c->n   = std::min(o.n_edges, kMoveEdgeSlots);

    float mean[kMoveEnc] = {}, mx[kMoveEnc] = {};
    for (int j = 0; j < c->n; ++j) {
        const float* e = o.x.data() + kMoveSelfFeat + j * kMoveEdgeFeat;
        for (int i = 0; i < kMoveEnc; ++i) {
            float s = be[i];
            const float* row = We.data() + static_cast<size_t>(i) * kMoveEdgeFeat;
            for (int k = 0; k < kMoveEdgeFeat; ++k) s += row[k] * e[k];
            c->pre[j][i] = s;
            const float h = s > 0.f ? s : 0.f;
            mean[i] += h;
            if (j == 0 || h > mx[i]) mx[i] = h;
        }
    }
    if (c->n > 0) {
        const float inv = 1.f / c->n;
        for (int i = 0; i < kMoveEnc; ++i) mean[i] *= inv;
    }

    for (int i = 0; i < kMoveSelfFeat; ++i) c->pooled[i] = o.x[i];
    for (int i = 0; i < kMoveEnc; ++i) {
        c->pooled[kMoveSelfFeat + i]            = mean[i];
        c->pooled[kMoveSelfFeat + kMoveEnc + i] = mx[i];
    }
    return head.forward(c->pooled, c->h1, c->h2, c->pa1, c->pa2);
}

// backprop_sample + input gradient (needed to reach the encoder).
static void head_backprop_dx(const Mlp& net, const float* x,
                             const float* h1, const float* h2,
                             const float* pa1, const float* pa2,
                             float dz3, MlpGrads& g, float* dx) {
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
    std::fill_n(dx, net.in_dim, 0.f);
    for (int i = 0; i < kHid; ++i) {
        const float dz1 = dh1[i] * (pa1[i] > 0.f ? 1.f : 0.f);
        g.db1[i] += dz1;
        const float* row = net.W1.data() + static_cast<size_t>(i) * net.in_dim;
        float* grow      = g.dW1.data()  + static_cast<size_t>(i) * net.in_dim;
        for (int j = 0; j < net.in_dim; ++j) {
            grow[j] += dz1 * x[j];
            dx[j]   += dz1 * row[j];
        }
    }
}

void StarNet::backprop(const StarCache& c, float dz3, StarGrads& g) const {
    float dpooled[kMovePooled];
    head_backprop_dx(head, c.pooled, c.h1, c.h2, c.pa1, c.pa2, dz3,
                     g.head, dpooled);
    if (c.n == 0) return;

    const float invn = 1.f / c.n;
    for (int i = 0; i < kMoveEnc; ++i) {
        int   am   = 0;
        float best = -1.f;
        for (int j = 0; j < c.n; ++j) {
            const float h = c.pre[j][i] > 0.f ? c.pre[j][i] : 0.f;
            if (h > best) { best = h; am = j; }
        }
        const float d_mean = dpooled[kMoveSelfFeat + i] * invn;
        const float d_max  = dpooled[kMoveSelfFeat + kMoveEnc + i];
        for (int j = 0; j < c.n; ++j) {
            const float dh = d_mean + (j == am ? d_max : 0.f);
            const float dz = dh * (c.pre[j][i] > 0.f ? 1.f : 0.f);
            if (dz == 0.f) continue;
            g.dbe[i] += dz;
            const float* e = c.obs + kMoveSelfFeat + j * kMoveEdgeFeat;
            float* grow = g.dWe.data() + static_cast<size_t>(i) * kMoveEdgeFeat;
            for (int k = 0; k < kMoveEdgeFeat; ++k) grow[k] += dz * e[k];
        }
    }
}

void StarNet::save_to(std::FILE* f) const {
    if (!f) return;
    std::fwrite(We.data(), sizeof(float), We.size(), f);
    std::fwrite(be.data(), sizeof(float), be.size(), f);
    head.save_to(f);
}
bool StarNet::load_from(std::FILE* f) {
    if (!f || We.empty()) return false;
    bool ok = true;
    ok = ok && std::fread(We.data(), sizeof(float), We.size(), f) == We.size();
    ok = ok && std::fread(be.data(), sizeof(float), be.size(), f) == be.size();
    return ok && head.load_from(f);
}

// ── StarGrads / StarAdam ─────────────────────────────────────────────────────

void StarGrads::resize_for(const StarNet& net) {
    dWe.assign(net.We.size(), 0.f);
    dbe.assign(net.be.size(), 0.f);
    head.resize_for(net.head);
}
void StarGrads::zero() {
    std::fill(dWe.begin(), dWe.end(), 0.f);
    std::fill(dbe.begin(), dbe.end(), 0.f);
    head.zero();
}
void StarGrads::clip_global_norm(float max_norm) {
    double sq = 0.0;
    auto acc = [&sq](const std::vector<float>& v) {
        for (float x : v) sq += static_cast<double>(x) * x;
    };
    acc(dWe); acc(dbe);
    acc(head.dW1); acc(head.db1); acc(head.dW2); acc(head.db2); acc(head.dW3);
    sq += static_cast<double>(head.db3) * head.db3;
    const float norm = static_cast<float>(std::sqrt(sq));
    if (norm <= max_norm) return;
    const float s = max_norm / (norm + 1e-6f);
    auto scale = [s](std::vector<float>& v) { for (float& x : v) x *= s; };
    scale(dWe); scale(dbe);
    scale(head.dW1); scale(head.db1); scale(head.dW2); scale(head.db2);
    scale(head.dW3);
    head.db3 *= s;
}
void StarGrads::negate() {
    auto neg = [](std::vector<float>& v) { for (float& x : v) x = -x; };
    neg(dWe); neg(dbe);
    head.negate();
}

void StarAdam::resize_for(const StarNet& net) {
    mWe.assign(net.We.size(), 0.f); vWe.assign(net.We.size(), 0.f);
    mbe.assign(net.be.size(), 0.f); vbe.assign(net.be.size(), 0.f);
    head.resize_for(net.head);
    t = 0;
}
void StarAdam::apply(StarNet& net, const StarGrads& g, float lr) {
    ++t;
    policy_optim::adam_apply(net.We.data(), g.dWe.data(), mWe.data(), vWe.data(),
                             static_cast<int>(net.We.size()), lr, t);
    policy_optim::adam_apply(net.be.data(), g.dbe.data(), mbe.data(), vbe.data(),
                             static_cast<int>(net.be.size()), lr, t);
    head.apply(net.head, g.head, lr);
}

// ── MovementPolicy ───────────────────────────────────────────────────────────

MovementPolicy::MovementPolicy() {
    hparams.clip_eps       = 0.2f;
    hparams.epochs         = 4;
    hparams.max_grad_norm  = 0.5f;
    hparams.lr_actor       = 1e-4f;
    hparams.lr_actor_init  = 1e-4f;
    hparams.lr_actor_min   = 1e-5f;
    hparams.lr_critic      = 1e-4f;
    hparams.lr_critic_init = 1e-4f;
    hparams.lr_critic_min  = 1e-5f;
    hparams.ent_w          = 5e-3f;
    hparams.ent_w_init     = 5e-3f;
    hparams.ent_w_min      = 5e-4f;
    reinit(std::random_device{}());
}

void MovementPolicy::reinit(uint32_t seed) {
    rng_.seed(seed);
    actor_.init(rng_, 0.01f);
    critic_.init(rng_, 1.0f);
    actor_opt_.resize_for(actor_);
    critic_opt_.resize_for(critic_);
    vrms_ = {};
    clear_buffers();
}

float MovementPolicy::score(const MoveObs& o) const {
    return Mlp::sigmoid(actor_.forward(o));
}

bool MovementPolicy::decide(float mu, bool explore) {
    if (!explore) return mu >= 0.5f;
    std::bernoulli_distribution d(std::clamp(mu, 0.f, 1.f));
    return d(rng_);
}

int MovementPolicy::record(int agent_id, const MoveObs& o, float mu,
                           float action, int step, int pred_steps) {
    constexpr float eps = 1e-8f;
    mu = std::clamp(mu, eps, 1.f - eps);

    MoveExperience e;
    e.obs        = o.x;
    e.n_edges    = std::min(o.n_edges, kMoveEdgeSlots);
    e.step       = step;
    e.pred_steps = std::max(1, pred_steps);
    e.action     = action;
    e.log_prob   = (action > 0.5f) ? std::log(mu) : std::log(1.f - mu);

    auto& b = buffers_[agent_id];
    b.push_back(e);
    const int idx = static_cast<int>(b.size()) - 1;
    open_[agent_id].push_back(idx);
    return idx;
}

void MovementPolicy::add_reward(int agent_id, int idx, float delta) {
    auto it = buffers_.find(agent_id);
    if (it == buffers_.end()) return;
    if (idx < 0 || idx >= static_cast<int>(it->second.size())) return;
    it->second[idx].reward += delta;
}

void MovementPolicy::credit_leg_outcome(int agent_id, int arrival_step,
                                        float w_out) {
    auto oit = open_.find(agent_id);
    if (oit == open_.end()) return;
    auto bit = buffers_.find(agent_id);
    if (bit != buffers_.end() && w_out != 0.f) {
        for (int idx : oit->second) {
            if (idx < 0 || idx >= static_cast<int>(bit->second.size())) continue;
            MoveExperience& e = bit->second[idx];
            const float pred   = static_cast<float>(e.pred_steps);
            const float actual = static_cast<float>(arrival_step - e.step);
            e.reward += w_out * std::clamp((pred - actual) / pred, -1.f, 1.f);
        }
    }
    oit->second.clear();
}

void MovementPolicy::drop_open(int agent_id) {
    auto it = open_.find(agent_id);
    if (it != open_.end()) it->second.clear();
}

void MovementPolicy::clear_buffers() {
    buffers_.clear();
    open_.clear();
}

int MovementPolicy::total_buffer_size() const {
    int n = 0;
    for (const auto& [aid, b] : buffers_) n += static_cast<int>(b.size());
    return n;
}

// ── PPO update (same machinery as ppo_train, StarNet + MoveExperience) ──────

static void move_gae(std::vector<MoveExperience>& traj, float gamma, float lam) {
    float gae = 0.f, next_val = 0.f;
    for (int k = static_cast<int>(traj.size()) - 1; k >= 0; --k) {
        MoveExperience& e = traj[k];
        const float delta = e.reward + gamma * next_val - e.value;
        gae         = delta + gamma * lam * gae;
        e.advantage = gae;
        e.ret       = gae + e.value;
        next_val    = e.value;
    }
}

TrainingStats MovementPolicy::train_round() {
    TrainingStats ts;

    std::vector<MoveExperience*> items;
    for (auto& [aid, b] : buffers_)
        for (auto& e : b) items.push_back(&e);
    const int n = static_cast<int>(items.size());
    ts.n_exp = n;
    if (n == 0) return ts;

    const float eps = 1e-8f;

    const float v_mu_old  = vrms_.mean_f();
    const float v_std_old = vrms_.std_dev();
    auto obs_of = [](const MoveExperience& e) {
        MoveObs o;
        o.x       = e.obs;
        o.n_edges = e.n_edges;
        return o;
    };
    for (MoveExperience* e : items)
        e->value = critic_.forward(obs_of(*e)) * v_std_old + v_mu_old;

    for (auto& [aid, b] : buffers_)
        if (!b.empty()) move_gae(b, hparams.gamma, hparams.lam_gae);

    for (const MoveExperience* e : items) vrms_.update(e->ret);

    float adv_mean = 0.f;
    for (const MoveExperience* e : items) adv_mean += e->advantage;
    adv_mean /= n;
    float adv_var = 0.f;
    for (const MoveExperience* e : items) {
        const float d = e->advantage - adv_mean;
        adv_var += d * d;
    }
    const float adv_std = std::sqrt(adv_var / n + 1e-8f);
    ts.adv_mean = adv_mean;
    ts.adv_std  = adv_std;
    for (MoveExperience* e : items)
        e->advantage = (e->advantage - adv_mean) / adv_std;

    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    StarGrads a_g; a_g.resize_for(actor_);
    StarGrads c_g; c_g.resize_for(critic_);

    const float v_mu  = vrms_.mean_f();
    const float v_std = vrms_.std_dev();
    const float inv_v_std = 1.f / v_std;

    float aloss_acc = 0.f, ent_acc = 0.f, closs_acc = 0.f;
    float kl_acc = 0.f, clip_acc = 0.f;
    int   n_updates = 0, epoch_run = 0;

    StarCache cache;
    const float inv = 1.f / n;

    for (int ep = 0; ep < hparams.epochs; ++ep) {
        std::shuffle(perm.begin(), perm.end(), rng_);

        a_g.zero();
        float L = 0.f, H = 0.f, kl = 0.f, cf = 0.f;
        for (int pi = 0; pi < n; ++pi) {
            const MoveExperience& e = *items[perm[pi]];
            const MoveObs o = obs_of(e);
            const float mu = Mlp::sigmoid(actor_.forward(o, &cache));
            const float lp_new = (e.action > 0.5f)
                ? std::log(mu + eps) : std::log(1.f - mu + eps);
            const float r = std::exp(lp_new - e.log_prob);
            const float A = e.advantage;

            kl += e.log_prob - lp_new;
            const bool clipped = (r > 1.f + hparams.clip_eps && A > 0.f)
                              || (r < 1.f - hparams.clip_eps && A < 0.f);
            if (clipped) cf += 1.f;

            const float clip_grad = clipped ? 0.f : A * r * (e.action - mu);
            const float ent_grad  = hparams.ent_w
                * std::log((1.f - mu + eps) / (mu + eps)) * mu * (1.f - mu);
            L += clipped
                ? std::clamp(r, 1.f - hparams.clip_eps, 1.f + hparams.clip_eps) * A
                : r * A;
            H += -(mu * std::log(mu + eps)
                 + (1.f - mu) * std::log(1.f - mu + eps));

            actor_.backprop(cache, (clip_grad + ent_grad) * inv, a_g);
        }
        a_g.clip_global_norm(hparams.max_grad_norm);
        a_g.negate();
        actor_opt_.apply(actor_, a_g, hparams.lr_actor);

        c_g.zero();
        float cl = 0.f;
        for (int pi = 0; pi < n; ++pi) {
            const MoveExperience& e = *items[perm[pi]];
            const MoveObs o = obs_of(e);
            const float V_new = critic_.forward(o, &cache);
            const float V_old_n = (e.value - v_mu) * inv_v_std;
            const float ret_n   = (e.ret   - v_mu) * inv_v_std;
            const float V_clip  = V_old_n + std::clamp(
                V_new - V_old_n, -hparams.val_clip_eps, hparams.val_clip_eps);
            const float err_new  = V_new  - ret_n;
            const float err_clip = V_clip - ret_n;
            cl += std::max(policy_optim::huber_value(err_new),
                           policy_optim::huber_value(err_clip));
            critic_.backprop(cache, policy_optim::huber_grad(err_new) * inv, c_g);
        }
        c_g.clip_global_norm(hparams.max_grad_norm);
        critic_opt_.apply(critic_, c_g, hparams.lr_critic);

        aloss_acc += -(L * inv);
        ent_acc   += H * inv;
        clip_acc  += cf * inv;
        closs_acc += cl * inv;
        ++n_updates;
        ++epoch_run;

        const float mean_kl = kl * inv;
        kl_acc += mean_kl;
        if (mean_kl > hparams.target_kl) break;
    }

    if (n_updates > 0) {
        ts.actor_loss  = aloss_acc / n_updates;
        ts.entropy     = ent_acc   / n_updates;
        ts.critic_loss = closs_acc / n_updates;
        ts.clip_frac   = clip_acc  / n_updates;
    }
    ts.kl_approx = (epoch_run > 0) ? kl_acc / epoch_run : 0.f;
    ts.n_epochs  = epoch_run;

    clear_buffers();
    return ts;
}

// ── Persistence ──────────────────────────────────────────────────────────────

static constexpr uint32_t kMagicMovement = 0xDEA110D2u;

void MovementPolicy::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicMovement, sizeof(kMagicMovement), 1, f);
    actor_.save_to(f);
    critic_.save_to(f);
    vrms_.save_to(f);
    std::fclose(f);
}

bool MovementPolicy::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != kMagicMovement) {
        std::fclose(f);
        return false;
    }
    bool ok = actor_.load_from(f) && critic_.load_from(f) && vrms_.load_from(f);
    std::fclose(f);
    return ok;
}

MovementPolicy& movement_policy() { static MovementPolicy p; return p; }
