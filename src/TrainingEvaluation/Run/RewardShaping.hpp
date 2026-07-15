#ifndef TRAINING_REWARD_SHAPING_HPP
#define TRAINING_REWARD_SHAPING_HPP

#include "TrainingEvaluation/StructuresParam/EpisodeConfig.hpp"
#include <algorithm>

// ════════════════════════════════════════════════════════════════════════════
// Reward shaping v2 — pure helpers (paper Table 2 revision).
// ════════════════════════════════════════════════════════════════════════════
// Application sites live in Runner.cpp; everything here is stateless so each
// term stays a one-line formula at its call site.

// ── §8 Static normalisation scales ───────────────────────────────────────────
// Fixed at episode start from the config BOUNDS (no running statistics, so
// ablations stay reproducible). Divisor is 1 when the flag is off.
struct RewardScales {
    float task_value = 1.f;   // λ for pickup / delivery / unfinished credits
};

inline RewardScales make_reward_scales(const EpisodeConfig& cfg) {
    RewardScales s;
    if (!cfg.rs_normalization) return s;
    const float vmul = cfg.enable_task_value_heterogeneity
        ? cfg.task_value_mul_max : 1.f;
    const float imax = cfg.enable_task_value_heterogeneity
        ? cfg.task_imp_max : 2.f;   // legacy importance range is [0.5, 2.0]
    s.task_value = std::max(1.f,
        event_tuning::kTaskRewardClampMax * vmul * imax);
    return s;
}

// ── §9 Linear weight annealing ───────────────────────────────────────────────
// w(e) = floor + (w0 − floor) · max(0, 1 − e/E_anneal), floor = floor_frac·w0.
// episode < 0 (never set — eval, tests) or flag off ⇒ w0 unchanged.
inline float annealed_w(float w0, float floor_frac,
                        int episode, int e_anneal, bool enabled) {
    if (!enabled || episode < 0 || e_anneal <= 0) return w0;
    const float floor = w0 * floor_frac;
    const float p = std::min(1.f, static_cast<float>(episode)
                                  / static_cast<float>(e_anneal));
    return floor + (w0 - floor) * (1.f - p);
}

// ── §3 Latency factor (paper eq. 6) ──────────────────────────────────────────
// φ(dt) = max(0, 1 − λ · min(1, dt/t_max)); φ = 1 when t_max ≤ 0.
inline float latency_phi(int dt, float lambda, int t_max) {
    if (t_max <= 0) return 1.f;
    const float n = std::min(1.f, static_cast<float>(std::max(0, dt))
                                  / static_cast<float>(t_max));
    return std::max(0.f, 1.f - lambda * n);
}

// ── Full-revision switch (paper T/Y runs; ablations flip terms back off) ────
inline void enable_all_reward_shaping(EpisodeConfig& cfg) {
    cfg.rs_congestion_route      = true;
    cfg.rs_loser_mu              = true;
    cfg.rs_pickup_latency        = true;
    cfg.rs_unfinished_cost_ratio = true;
    cfg.rs_idle_refine           = true;
    cfg.rs_normalization         = true;
    cfg.rs_weight_annealing      = true;   // restricted to w_naff + w_idle (floor 0)
}

#endif // TRAINING_REWARD_SHAPING_HPP
