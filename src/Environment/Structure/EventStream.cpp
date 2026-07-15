#include "Environment/Structure/EventStream.hpp"

const char* temporal_profile_label(TemporalProfile p) {
    switch (p) {
        case TemporalProfile::Uniform:    return "uniform";
        case TemporalProfile::Flat:       return "flat";
        case TemporalProfile::RampUpDown: return "ramp_up_down";
        case TemporalProfile::ShockBurst: return "shock_burst";
        case TemporalProfile::BuildingUp: return "building_up";
        case TemporalProfile::Wave:       return "wave";
        case TemporalProfile::ShockPick:  return "shock_pick";
    }
    return "unknown";
}

float temporal_profile_value(TemporalProfile p, float t01) {
    const float t = std::clamp(t01, 0.f, 1.f);
    switch (p) {
        case TemporalProfile::Uniform: return 1.f;
        case TemporalProfile::Flat:    return 0.10f;
        case TemporalProfile::RampUpDown:
            if      (t < 0.25f) return (t / 0.25f) * 0.30f;
            else if (t < 0.50f) return 0.30f - ((t - 0.25f) / 0.25f) * 0.20f;
            else if (t < 0.75f) return 0.10f + ((t - 0.50f) / 0.25f) * 0.40f;
            else                return 0.50f + ((t - 0.75f) / 0.25f) * 0.50f;
        case TemporalProfile::ShockBurst:
            return (t < 0.60f) ? 0.05f : (t < 0.80f ? 1.00f : 0.30f);
        case TemporalProfile::BuildingUp: return t;
        case TemporalProfile::Wave:       return std::sin(t * 3.14159265f);
        case TemporalProfile::ShockPick:  return std::pow(t, 1.5f);
    }
    return 0.f;
}
