#include "HybridPolicy.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>

// ── Singleton ────────────────────────────────────────────────────────────────

HybridPolicy::HybridPolicy() : rng_(std::random_device{}()) {}

HybridPolicy& HybridPolicy::shared() {
    static HybridPolicy instance;
    return instance;
}

// ── Base setup ───────────────────────────────────────────────────────────────

void HybridPolicy::set_base_from(const ObjectiveDMPolicy& mappo) {
    set_base_actor(mappo.actor);
}

void HybridPolicy::set_base_actor(const ActorMLP& src) {
    base_actor_ = src;   // bitwise copy of all weights
    base_set_   = true;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

float HybridPolicy::sigmoid(float x) {
    if (x >= 0.f) {
        const float z = std::exp(-x);
        return 1.f / (1.f + z);
    }
    const float z = std::exp(x);
    return z / (1.f + z);
}

float HybridPolicy::logit(float p) {
    p = std::clamp(p, 1e-6f, 1.f - 1e-6f);
    return std::log(p / (1.f - p));
}

HybridPolicy::AgentResidual& HybridPolicy::get_or_create_residual(int agent_id) {
    auto it = residuals_.find(agent_id);
    if (it != residuals_.end()) return it->second;
    AgentResidual r;
    // Zero-init → first decisions of a new agent = pure MAPPO behaviour.
    r.w.fill(0.f);
    r.b = 0.f;
    residuals_[agent_id] = r;
    buffers_[agent_id]   = {};
    return residuals_[agent_id];
}

float HybridPolicy::delta_logit(const AgentResidual& r,
                                 const PolicyFeatures& f) const {
    std::array<float, kPolicySz> x;
    f.to_array(x.data());
    float dl = r.b;
    for (int i = 0; i < kPolicySz; ++i)
        dl += r.w[i] * x[i];
    return dl;
}

void HybridPolicy::clip_residual(AgentResidual& r) {
    for (auto& wi : r.w)
        wi = std::clamp(wi, -hparams.weight_clip, hparams.weight_clip);
    r.b = std::clamp(r.b, -hparams.bias_clip, hparams.bias_clip);
}

// ── Inference ────────────────────────────────────────────────────────────────

float HybridPolicy::score(int agent_id, const PolicyFeatures& features) {
    if (!base_set_) {
        // Print warning ONCE only — base not set means we evaluate with whatever
        // ActorMLP was zero-init'd to (mu ≈ 0.5 everywhere). The user should
        // call set_base_from() before any meaningful run.
        static bool warned = false;
        if (!warned) {
            std::cerr << "[HybridPolicy] Warning: base actor not set — "
                         "all scores will be ≈ 0.5. Call set_base_from(mappo).\n";
            warned = true;
        }
    }
    std::array<float, kPolicySz> x;
    features.to_array(x.data());

    // Forward MAPPO base → μ_base, recover its logit.
    const float mu_base = base_actor_.forward(x.data());
    const float l_base  = logit(mu_base);

    // Add per-agent linear residual.
    AgentResidual& r = get_or_create_residual(agent_id);
    const float dl   = delta_logit(r, features);

    return sigmoid(l_base + dl);
}

// ── Data collection ──────────────────────────────────────────────────────────

int HybridPolicy::record(int agent_id, const PolicyFeatures& obs,
                          float action, float reward) {
    get_or_create_residual(agent_id);
    auto& buf = buffers_[agent_id];
    Experience e;
    obs.to_array(e.obs.data());
    e.agent_id = agent_id;
    e.action   = action;
    e.reward   = reward;
    // log_prob computed under the CURRENT policy (base + residual).
    const float mu = score(agent_id, obs);
    e.log_prob = (action > 0.5f) ? std::log(std::clamp(mu, 1e-6f, 1.f))
                                  : std::log(std::clamp(1.f - mu, 1e-6f, 1.f));
    buf.push_back(e);

    const int idx = static_cast<int>(buf.size()) - 1;
    recent_records_.emplace_back(agent_id, idx);
    return idx;
}

void HybridPolicy::update_reward(int agent_id, int buf_idx, float reward) {
    auto it = buffers_.find(agent_id);
    if (it == buffers_.end()) return;
    if (buf_idx < 0 || buf_idx >= (int)it->second.size()) return;
    it->second[buf_idx].reward = reward;
}

void HybridPolicy::add_to_reward(int agent_id, int buf_idx, float delta) {
    auto it = buffers_.find(agent_id);
    if (it == buffers_.end()) return;
    if (buf_idx < 0 || buf_idx >= (int)it->second.size()) return;
    it->second[buf_idx].reward += delta;
}

int HybridPolicy::buffer_size(int agent_id) const {
    auto it = buffers_.find(agent_id);
    return it == buffers_.end() ? 0 : (int)it->second.size();
}

int HybridPolicy::total_buffer_size() const {
    int n = 0;
    for (const auto& kv : buffers_) n += (int)kv.second.size();
    return n;
}

int HybridPolicy::n_recent_records() const {
    return static_cast<int>(recent_records_.size());
}

std::pair<int,int> HybridPolicy::recent_record(int i) const {
    return recent_records_[i];
}

void HybridPolicy::clear_recent_records() {
    recent_records_.clear();
}

// ── Online residual update (REINFORCE) ───────────────────────────────────────
//
// For a binary Bernoulli action a ~ π(·|s) with μ = sigmoid(logit_base + Δ),
// the policy gradient w.r.t. the residual parameters θ_r (where Δ = θ_r · x)
// for a single experience is:
//
//   ∇_θ_r log π(a|s) = (a − μ) · x       (for the linear part)
//   ∇_b   log π(a|s) = (a − μ)
//
// The advantage estimate is just the centred reward (no critic):
//   A = r − r̄_agent   (subtract per-agent rolling mean to reduce variance)
//
// Update:  θ_r ← θ_r + lr · A · ∇_θ_r log π(a|s)

void HybridPolicy::reinforce_update(AgentResidual& r, const Experience& e) {
    // Recover μ at decision time using current residual (approximation;
    // for small lr and large buffers this is accurate enough).
    const float mu_base = base_actor_.forward(e.obs.data());
    const float l_base  = logit(mu_base);

    // Compute Δ from current residual + features.
    float dl = r.b;
    for (int i = 0; i < kPolicySz; ++i)
        dl += r.w[i] * e.obs[i];
    const float mu = sigmoid(l_base + dl);

    // Baseline = rolling mean of recent rewards.
    float baseline = 0.f;
    if (!r.recent_rewards.empty())
        baseline = std::accumulate(r.recent_rewards.begin(),
                                    r.recent_rewards.end(), 0.f)
                 / r.recent_rewards.size();
    const float adv = e.reward - baseline;

    const float grad_factor = (e.action - mu) * adv * hparams.lr_residual;
    r.b += grad_factor;
    for (int i = 0; i < kPolicySz; ++i)
        r.w[i] += grad_factor * e.obs[i];

    clip_residual(r);
}

// ── Divergence-based rollback (safety net) ───────────────────────────────────
//
// Tracks a per-agent EMA of episode reward as "baseline" while the residual
// norm is still small (agent behaves nearly like MAPPO). Once the residual
// grows beyond `baseline_lock_norm`, the baseline stops updating and serves
// as a reference: if current_reward_ema falls below baseline_reward_ema −
// K × baseline_std for `divergence_episodes` consecutive episodes → hard
// rollback. This catches degradation even when the whole fleet is doing
// poorly (e.g., during stress eval) — it's the agent's OWN past that matters.

bool HybridPolicy::update_baseline_and_check_divergence(
    AgentResidual& r, float episode_mean_reward)
{
    if (!hparams.enable_divergence_rollback) return false;

    const float a = hparams.baseline_ema_alpha;

    // Always update current_reward_ema.
    if (!r.baseline_initialised) {
        r.current_reward_ema = episode_mean_reward;
    } else {
        r.current_reward_ema =
            (1.f - a) * r.current_reward_ema + a * episode_mean_reward;
    }

    // Compute current residual norm to decide if baseline should track.
    float resid_norm = std::abs(r.b);
    for (auto wi : r.w) resid_norm += std::abs(wi);

    const bool baseline_locks_in = (resid_norm < hparams.baseline_lock_norm);
    if (baseline_locks_in || !r.baseline_initialised) {
        // Update baseline EMA while residual is near zero (= MAPPO behaviour).
        if (!r.baseline_initialised) {
            r.baseline_reward_ema = episode_mean_reward;
            r.baseline_var_ema    = 1.f;  // permissive initial std
            r.baseline_initialised = true;
        } else {
            const float prev = r.baseline_reward_ema;
            r.baseline_reward_ema = (1.f - a) * prev + a * episode_mean_reward;
            const float dev = episode_mean_reward - prev;
            r.baseline_var_ema = (1.f - a) * r.baseline_var_ema + a * dev * dev;
        }
        ++r.n_baseline_episodes;
        r.consecutive_divergent_episodes = 0;  // baseline still tracking → safe
        return false;
    }

    // Past the lock threshold — divergence check is now active.
    // Need at least a few baseline episodes for a meaningful std.
    if (r.n_baseline_episodes < 3) return false;

    const float baseline_std = std::sqrt(std::max(r.baseline_var_ema, 1e-6f));
    const float threshold    = r.baseline_reward_ema
                             - hparams.divergence_threshold * baseline_std;

    if (r.current_reward_ema < threshold) {
        ++r.consecutive_divergent_episodes;
    } else {
        r.consecutive_divergent_episodes = 0;
    }

    if (r.consecutive_divergent_episodes >= hparams.divergence_episodes) {
        // Emergency hard rollback: residual is making the agent worse than
        // its own MAPPO-base past, persistently.
        r.w.fill(0.f);
        r.b = 0.f;
        r.consecutive_divergent_episodes = 0;
        r.consecutive_soft_rollbacks     = 0;  // reset the other counter too
        ++r.n_hard_rollbacks;
        return true;
    }
    return false;
}

// ── Rollback check ───────────────────────────────────────────────────────────

bool HybridPolicy::check_and_apply_rollback(int /*agent_id*/, AgentResidual& r,
                                             float fleet_mean_fitness) {
    if (r.decisions_since_check < hparams.check_period) return false;
    r.decisions_since_check = 0;

    const float threshold = fleet_mean_fitness - hparams.rollback_threshold;
    if (r.cached_fitness >= threshold) {
        // Healthy: reset consecutive counter.
        r.consecutive_soft_rollbacks = 0;
        return false;
    }

    // Underperforming → soft rollback (shrink residual toward zero).
    for (auto& wi : r.w) wi *= hparams.rollback_shrink;
    r.b *= hparams.rollback_shrink;
    r.consecutive_soft_rollbacks++;

    if (r.consecutive_soft_rollbacks >= hparams.hard_rollback_after) {
        // Hard rollback: zero out completely.
        r.w.fill(0.f);
        r.b = 0.f;
        r.consecutive_soft_rollbacks = 0;
    }
    return true;
}

// ── End-of-episode online learning + rollback ────────────────────────────────

HybridPolicy::TrainingStats HybridPolicy::train_epoch() {
    TrainingStats stats;
    if (residuals_.empty()) return stats;

    // 1) Compute per-agent episode mean reward (for both baselines and divergence).
    std::unordered_map<int, float> episode_mean_reward;
    for (auto& kv : buffers_) {
        const int aid = kv.first;
        if (kv.second.empty()) continue;
        float sum_r = 0.f;
        for (const auto& e : kv.second) sum_r += e.reward;
        episode_mean_reward[aid] = sum_r / kv.second.size();
    }

    // 2) REINFORCE update PER AGENT using PREVIOUS episodes as baseline.
    // (Updates use r.recent_rewards as-is, BEFORE we push the current episode
    //  reward — this gives an unbiased advantage estimate.)
    int   n_exp_total = 0;
    float reward_accum = 0.f;
    for (auto& kv : buffers_) {
        const int aid = kv.first;
        auto& buf = kv.second;
        if (buf.empty()) continue;
        AgentResidual& r = residuals_[aid];

        for (const auto& e : buf) {
            reinforce_update(r, e);
            reward_accum += e.reward;
            ++n_exp_total;
        }
    }

    // 3) NOW push this episode's mean reward into each agent's rolling window
    //    (used for fleet-relative rollback and as baseline next episode).
    for (auto& kv : residuals_) {
        const int aid = kv.first;
        auto it = episode_mean_reward.find(aid);
        if (it == episode_mean_reward.end()) continue;
        AgentResidual& r = kv.second;
        r.recent_rewards.push_back(it->second);
        if ((int)r.recent_rewards.size() > hparams.fitness_window)
            r.recent_rewards.erase(r.recent_rewards.begin());
        r.cached_fitness = std::accumulate(r.recent_rewards.begin(),
                                            r.recent_rewards.end(), 0.f)
                          / r.recent_rewards.size();
        r.decisions_since_check += (int)buffers_[aid].size();
    }

    // 4) Fleet mean fitness for the (legacy) fleet-relative rollback.
    float fleet_mean = 0.f; int n_agents_with_data = 0;
    for (const auto& kv : residuals_) {
        if (!kv.second.recent_rewards.empty()) {
            fleet_mean += kv.second.cached_fitness;
            ++n_agents_with_data;
        }
    }
    if (n_agents_with_data > 0) fleet_mean /= n_agents_with_data;

    // 5) Two-tier rollback check per agent:
    //    a) Divergence safety net (per-agent self-comparison vs MAPPO baseline)
    //    b) Fleet-relative soft rollback (comparison vs fleet mean)
    int n_rollbacks = 0;
    int n_divergence_triggered = 0;
    for (auto& kv : residuals_) {
        const int aid = kv.first;
        AgentResidual& r = kv.second;
        auto it = episode_mean_reward.find(aid);
        const float ep_r = (it == episode_mean_reward.end()) ? 0.f : it->second;

        const bool diverged = update_baseline_and_check_divergence(r, ep_r);
        if (diverged) {
            ++n_divergence_triggered;
            ++n_rollbacks;
            continue;  // hard reset done, skip the soft check this episode
        }
        if (check_and_apply_rollback(aid, r, fleet_mean))
            ++n_rollbacks;
    }

    // 6) Stats reporting.
    stats.n_exp      = n_exp_total;
    stats.entropy    = 0.f;  // not tracked for REINFORCE
    stats.actor_loss = (n_exp_total > 0) ? (-reward_accum / n_exp_total) : 0.f;
    // adv_std / adv_mean co-opted for rollback counters (consumed by logger).
    stats.adv_std    = static_cast<float>(n_rollbacks);
    stats.adv_mean   = static_cast<float>(n_divergence_triggered);

    // 7) Clear buffers and recent records for the next episode.
    clear_buffer_all();
    clear_recent_records();

    return stats;
}

void HybridPolicy::clear_buffer(int agent_id) {
    auto it = buffers_.find(agent_id);
    if (it != buffers_.end()) it->second.clear();
}

void HybridPolicy::clear_buffer_all() {
    for (auto& kv : buffers_) kv.second.clear();
}

void HybridPolicy::ensure_agents(int n) {
    for (int i = 0; i < n; ++i) get_or_create_residual(i);
}

float HybridPolicy::residual_norm(int agent_id) const {
    auto it = residuals_.find(agent_id);
    if (it == residuals_.end()) return 0.f;
    const auto& r = it->second;
    float s = std::abs(r.b);
    for (auto w : r.w) s += std::abs(w);
    return s;
}

int HybridPolicy::n_in_rollback() const {
    int n = 0;
    for (const auto& kv : residuals_)
        if (kv.second.consecutive_soft_rollbacks > 0) ++n;
    return n;
}

int HybridPolicy::total_divergence_rollbacks() const {
    int n = 0;
    for (const auto& kv : residuals_)
        n += kv.second.n_hard_rollbacks;
    return n;
}

// ── Persistence ──────────────────────────────────────────────────────────────
//
// Format v2 ("HBP2"):
//   [magic 4B "HBP2"]
//   [n_agents int32]
//   [ActorMLP base : raw memcpy]
//   For each agent:
//     [agent_id int32]
//     [w (kPolicySz floats)] [b float]
//     [consecutive_soft_rollbacks int32]
//     [baseline_reward_ema float] [baseline_var_ema float]
//     [current_reward_ema float]
//     [baseline_initialised uint8]
//     [consecutive_divergent_episodes int32]
//     [n_baseline_episodes int32] [n_hard_rollbacks int32]
//
// Format v1 ("HBP1") is still loadable for backward compatibility — fields
// missing in v1 are zero-initialised.

void HybridPolicy::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "[Hybrid] save failed: " << path << "\n"; return; }
    const char magic[4] = {'H','B','P','2'};
    f.write(magic, 4);
    int32_t n = (int32_t)residuals_.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(reinterpret_cast<const char*>(&base_actor_), sizeof(ActorMLP));
    for (const auto& kv : residuals_) {
        int32_t aid = kv.first;
        const auto& r = kv.second;
        f.write(reinterpret_cast<const char*>(&aid), sizeof(aid));
        f.write(reinterpret_cast<const char*>(r.w.data()),
                kPolicySz * sizeof(float));
        f.write(reinterpret_cast<const char*>(&r.b),     sizeof(float));
        f.write(reinterpret_cast<const char*>(&r.consecutive_soft_rollbacks),
                sizeof(int32_t));
        f.write(reinterpret_cast<const char*>(&r.baseline_reward_ema), sizeof(float));
        f.write(reinterpret_cast<const char*>(&r.baseline_var_ema),    sizeof(float));
        f.write(reinterpret_cast<const char*>(&r.current_reward_ema),  sizeof(float));
        uint8_t baseline_init_byte = r.baseline_initialised ? 1u : 0u;
        f.write(reinterpret_cast<const char*>(&baseline_init_byte), sizeof(uint8_t));
        f.write(reinterpret_cast<const char*>(&r.consecutive_divergent_episodes),
                sizeof(int32_t));
        f.write(reinterpret_cast<const char*>(&r.n_baseline_episodes), sizeof(int32_t));
        f.write(reinterpret_cast<const char*>(&r.n_hard_rollbacks),    sizeof(int32_t));
    }
}

bool HybridPolicy::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    const bool is_v2 = (std::strncmp(magic, "HBP2", 4) == 0);
    const bool is_v1 = (std::strncmp(magic, "HBP1", 4) == 0);
    if (!is_v1 && !is_v2) return false;
    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    f.read(reinterpret_cast<char*>(&base_actor_), sizeof(ActorMLP));
    base_set_ = true;
    residuals_.clear();
    buffers_.clear();
    for (int i = 0; i < n; ++i) {
        int32_t aid;
        AgentResidual r;
        f.read(reinterpret_cast<char*>(&aid), sizeof(aid));
        f.read(reinterpret_cast<char*>(r.w.data()), kPolicySz * sizeof(float));
        f.read(reinterpret_cast<char*>(&r.b),       sizeof(float));
        f.read(reinterpret_cast<char*>(&r.consecutive_soft_rollbacks), sizeof(int32_t));
        if (is_v2) {
            f.read(reinterpret_cast<char*>(&r.baseline_reward_ema), sizeof(float));
            f.read(reinterpret_cast<char*>(&r.baseline_var_ema),    sizeof(float));
            f.read(reinterpret_cast<char*>(&r.current_reward_ema),  sizeof(float));
            uint8_t baseline_init_byte = 0;
            f.read(reinterpret_cast<char*>(&baseline_init_byte), sizeof(uint8_t));
            r.baseline_initialised = (baseline_init_byte != 0);
            f.read(reinterpret_cast<char*>(&r.consecutive_divergent_episodes),
                   sizeof(int32_t));
            f.read(reinterpret_cast<char*>(&r.n_baseline_episodes), sizeof(int32_t));
            f.read(reinterpret_cast<char*>(&r.n_hard_rollbacks),    sizeof(int32_t));
        }
        // v1: divergence fields stay zero-initialised, baseline_initialised=false.
        residuals_[aid] = r;
        buffers_[aid]   = {};
    }
    return static_cast<bool>(f);
}
