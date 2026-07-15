#ifndef ENVIRONMENT_EVENT_STREAM_HPP
#define ENVIRONMENT_EVENT_STREAM_HPP

#include <algorithm>
#include <cmath>

// ════════════════════════════════════════════════════════════════════════════
// Temporal profiles — the SINGLE argument shared by task- and congestion-event
// generation.
// ════════════════════════════════════════════════════════════════════════════
//
// A profile fd(x) is a time-variation function f : [0,1] → [0,∞) over the
// normalised episode time. It only defines WHEN events happen (density ∝ f);
// the event COUNT is fixed upstream (event_tuning: round(base · SCE · RM)).
//
//   Task stream      : arrival steps sampled with density ∝ f(t/T).
//   Congestion stream: ghost appearance steps sampled with density ∝ f(t/T).
//
// A scenario carries one profile per category (task_profile /
// congestion_profile) — the same scenario may use different shapes for each.
enum class TemporalProfile {
    Uniform,      // 1                  — constant density ("Normal" in the paper)
    Flat,         // 0.10 (constant)    — low constant congestion baseline
    RampUpDown,   // rise / relax / rebuild / peak
    ShockBurst,   // calm, then spike on [0.6, 0.8], then tail
    BuildingUp,   // t                  — linear 0 → 100% ("Climbing")
    Wave,         // sin(πt)            — single bell
    ShockPick,    // t^1.5              — accelerating late surge (paper "Shock Pick")
};

const char* temporal_profile_label(TemporalProfile p);

// f(t01) for the given profile; t01 is clamped to [0, 1].
float temporal_profile_value(TemporalProfile p, float t01);

// ════════════════════════════════════════════════════════════════════════════
// Event-stream tuning — every hand-editable scaling knob, in ONE place.
// ════════════════════════════════════════════════════════════════════════════
//
// UQF — fixed base quantities. An event count = round(base · SCE · RM) where
// SCE = EpisodeConfig::env_scale (Small/Medium/Large = 1/2/5) and RM =
// EpisodeConfig::ratio_mult (1 in training; incremented per seed in evaluation).
// The temporal profile fd(x) only shapes WHEN those events happen.
namespace event_tuning {

inline constexpr float kTaskBaseQtt  = 100.f;    // tasks         = round(base · SCE · RM)
inline constexpr float kCongBaseQtt  = 25000.f;  // ghost events  = round(base · SCE · RM) → netBPR ~x1.75
inline constexpr float kAgentBaseQtt = 10.f;     // delivery agents = round(base · SCE · AM)

// Each ghost adds an integer load drawn uniformly in [kGhostLoadMin,
// kGhostLoadMax] when it appears, and removes that same value when it expires.
inline constexpr int kGhostLoadMin = 1;
inline constexpr int kGhostLoadMax = 5;

// φh — fraction of the road segments eligible for ghost traffic (hot-way pool).
inline constexpr float kHotWayFraction = 0.40f;

// Task base-reward clamp (EpisodeGenerator::estimate_reward, ∝ P→D distance).
// Also the r_max used by the static reward-normalisation scales.
inline constexpr float kTaskRewardClampMin = 0.1f;
inline constexpr float kTaskRewardClampMax = 5.0f;

inline int derived_task_count(float sce, float rm = 1.f) {
    return std::max(1, static_cast<int>(std::round(
        kTaskBaseQtt * std::max(0.f, sce) * std::max(0.f, rm))));
}

inline int derived_ghost_count(float sce, float rm = 1.f) {
    return std::max(1, static_cast<int>(std::round(
        kCongBaseQtt * std::max(0.f, sce) * std::max(0.f, rm))));
}

inline int derived_fleet_size(float sce, float am) {
    return std::max(1, static_cast<int>(std::round(
        kAgentBaseQtt * std::max(0.f, sce) * std::max(0.f, am))));
}

} // namespace event_tuning

#endif // ENVIRONMENT_EVENT_STREAM_HPP
