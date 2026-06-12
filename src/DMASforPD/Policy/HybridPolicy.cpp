#include "HybridPolicy.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

void HybridPolicy::set_base_actor(const Mlp& base) {
    base_     = base;
    base_set_ = true;
}

HybridPolicy::Residual& HybridPolicy::get_or_create(int agent_id) {
    return residuals_[agent_id];   // zero-initialised = pure MAPPO hot-start
}

void HybridPolicy::ensure_agents(int n_agents) {
    for (int i = 0; i < n_agents; ++i) (void)get_or_create(i);
}

void HybridPolicy::reinit(uint32_t seed) {
    rng_.seed(seed);
    residuals_.clear();            // base actor is kept
    n_hard_rollbacks_ = 0;
    clear_buffers();
}

float HybridPolicy::logit_of(const Residual& r, const float* x) const {
    float z = base_set_ ? base_.forward(x) : 0.f;
    for (int k = 0; k < kPolicySz; ++k) z += r.w[k] * x[k];
    return z + r.b;
}

float HybridPolicy::score(int agent_id, const PolicyFeatures& f) {
    float x[kPolicySz];
    f.to_array(x);
    return Mlp::sigmoid(logit_of(get_or_create(agent_id), x));
}

void HybridPolicy::clip_residual(Residual& r) const {
    for (float& wk : r.w)
        wk = std::clamp(wk, -hparams.weight_clip, hparams.weight_clip);
    r.b = std::clamp(r.b, -hparams.bias_clip, hparams.bias_clip);
}

float HybridPolicy::residual_norm(int agent_id) const {
    auto it = residuals_.find(agent_id);
    if (it == residuals_.end()) return 0.f;
    float sq = 0.f;
    for (float wk : it->second.w) sq += wk * wk;
    return std::sqrt(sq) + std::abs(it->second.b);
}

TrainingStats HybridPolicy::train_round() {
    TrainingStats ts;

    // ── 1. Online REINFORCE on each agent's residual ─────────────────────────
    std::vector<std::pair<int, float>> episode_fitness;   // (agent, mean reward)
    for (auto& [aid, b] : buffers_) {
        if (b.empty()) continue;
        Residual& r = get_or_create(aid);

        float mean_r = 0.f;
        for (Experience& e : b) {
            // ∂log π(a|s)/∂z = a − μ at the CURRENT residual (sequential
            // online updates within the episode batch).
            const float mu = Mlp::sigmoid(logit_of(r, e.obs.data()));
            const float g  = e.reward * (e.action - mu);
            for (int k = 0; k < kPolicySz; ++k)
                r.w[k] += hparams.lr_residual * g * e.obs[k];
            r.b += hparams.lr_residual * g;
            clip_residual(r);
            mean_r += e.reward;
        }
        mean_r /= static_cast<float>(b.size());
        episode_fitness.emplace_back(aid, mean_r);
        ts.n_exp += static_cast<int>(b.size());

        r.fitness_history.push_back(mean_r);
        while (static_cast<int>(r.fitness_history.size()) > hparams.fitness_window)
            r.fitness_history.erase(r.fitness_history.begin());
    }

    if (episode_fitness.empty()) {
        clear_buffers();
        return ts;
    }

    // ── 2. Fleet-relative rollback ───────────────────────────────────────────
    float fleet_mean = 0.f;
    int   n_fit      = 0;
    for (auto& [aid, r] : residuals_) {
        if (r.fitness_history.empty()) continue;
        float m = 0.f;
        for (float v : r.fitness_history) m += v;
        fleet_mean += m / r.fitness_history.size();
        ++n_fit;
    }
    if (n_fit > 0) fleet_mean /= n_fit;
    ts.adv_mean = fleet_mean;

    for (auto& [aid, r] : residuals_) {
        if (r.fitness_history.empty()) continue;
        float m = 0.f;
        for (float v : r.fitness_history) m += v;
        m /= r.fitness_history.size();

        if (n_fit > 1 && m < fleet_mean - hparams.rollback_threshold) {
            for (float& wk : r.w) wk *= hparams.rollback_shrink;
            r.b *= hparams.rollback_shrink;
            if (++r.consecutive_softs >= hparams.hard_after) {
                r.w.fill(0.f);
                r.b = 0.f;
                r.consecutive_softs = 0;
                ++n_hard_rollbacks_;
            }
        } else {
            r.consecutive_softs = 0;
        }
    }

    // ── 3. Divergence safety rollback ────────────────────────────────────────
    if (hparams.divergence_enabled) {
        for (auto& [aid, mean_r] : episode_fitness) {
            Residual& r = get_or_create(aid);
            const float a = hparams.baseline_ema_alpha;

            if (!r.baseline_init) {
                r.baseline_ema  = mean_r;
                r.current_ema   = mean_r;
                r.baseline_init = true;
                continue;
            }
            r.current_ema = (1.f - a) * r.current_ema + a * mean_r;

            float sq = 0.f;
            for (float wk : r.w) sq += wk * wk;
            const float norm = std::sqrt(sq) + std::abs(r.b);

            if (norm < hparams.baseline_lock_norm) {
                // Residual ≈ 0 → behaviour ≈ pure MAPPO: refine the baseline.
                const float d = mean_r - r.baseline_ema;
                r.baseline_ema += a * d;
                r.baseline_var  = (1.f - a) * r.baseline_var + a * d * d;
                ++r.n_baseline_episodes;
                r.consecutive_divergent = 0;
            } else if (r.n_baseline_episodes >= 3) {
                const float sd = std::sqrt(std::max(1e-6f, r.baseline_var));
                if (r.current_ema <
                    r.baseline_ema - hparams.divergence_sigma * sd) {
                    if (++r.consecutive_divergent >= hparams.divergence_episodes) {
                        r.w.fill(0.f);
                        r.b = 0.f;
                        r.consecutive_divergent = 0;
                        r.consecutive_softs     = 0;
                        ++n_hard_rollbacks_;
                    }
                } else {
                    r.consecutive_divergent = 0;
                }
            }
        }
    }

    clear_buffers();
    return ts;
}

// ── Persistence ──────────────────────────────────────────────────────────────
static constexpr uint32_t kMagicHybrid = 0xDEA110D3u;

void HybridPolicy::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicHybrid, sizeof(kMagicHybrid), 1, f);
    const int32_t has_base = base_set_ ? 1 : 0;
    std::fwrite(&has_base, sizeof(has_base), 1, f);
    if (base_set_) base_.save_to(f);
    const int32_t n = static_cast<int32_t>(residuals_.size());
    std::fwrite(&n, sizeof(n), 1, f);
    for (const auto& [aid, r] : residuals_) {
        const int32_t id = aid;
        std::fwrite(&id, sizeof(id), 1, f);
        std::fwrite(r.w.data(), sizeof(float), kPolicySz, f);
        std::fwrite(&r.b, sizeof(float), 1, f);
    }
    std::fclose(f);
}

bool HybridPolicy::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != kMagicHybrid) {
        std::fclose(f);
        return false;
    }
    int32_t has_base = 0;
    bool ok = std::fread(&has_base, sizeof(has_base), 1, f) == 1;
    if (ok && has_base) {
        Mlp base;
        std::mt19937 tmp(0u);
        base.init(kPolicySz, tmp, 0.01f);    // size the buffers before load
        ok = base.load_from(f);
        if (ok) set_base_actor(base);
    }
    int32_t n = 0;
    ok = ok && std::fread(&n, sizeof(n), 1, f) == 1 && n >= 0;
    residuals_.clear();
    for (int32_t i = 0; i < n && ok; ++i) {
        int32_t id = 0;
        ok = std::fread(&id, sizeof(id), 1, f) == 1;
        if (!ok) break;
        Residual r;
        ok = std::fread(r.w.data(), sizeof(float), kPolicySz, f) == kPolicySz
          && std::fread(&r.b, sizeof(float), 1, f) == 1;
        if (ok) residuals_.emplace(id, std::move(r));
    }
    std::fclose(f);
    return ok;
}
