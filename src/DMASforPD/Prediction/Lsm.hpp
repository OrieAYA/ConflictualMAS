#ifndef LSM_MODULE_HPP
#define LSM_MODULE_HPP

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// LsmModule — Liquid State Machine (Maass) for congestion prediction/signaling.
//
// Fixed random liquid: LIF neurons on a 3D lattice, 80% excitatory / 20%
// inhibitory, distance-based connectivity p = C·exp(−d²/λ²). Only the linear
// readout learns (online NLMS, supervised by realized congestion at t+H).
// Input stream: coarse region congestion/density + global signals; output:
// predicted coarse congestion at t+H per 8×8 cell + one global level.
// ════════════════════════════════════════════════════════════════════════════

static constexpr int kLsmCoarseDim = 8;
static constexpr int kLsmCells     = kLsmCoarseDim * kLsmCoarseDim;   // 64
static constexpr int kLsmGlobals   = 8;
static constexpr int kLsmIn        = 2 * kLsmCells + kLsmGlobals;     // 136
static constexpr int kLsmOut       = kLsmCells + 1;                   // 64 + global
static constexpr int kLsmN         = 256;                             // 4×4×16 lattice

class LsmModule {
public:
    LsmModule() { reinit(0xC0FFEEu); }

    void reinit(uint32_t seed);
    void reset_state();

    // One liquid tick; updates the filtered spike trace (the liquid state).
    void step(const float* u);

    const std::vector<float>& state() const { return trace_; }

    // ŷ = W_out · x for the current state or a stored past state.
    void predict(float* y) const { predict_state(trace_, y); }
    void predict_state(const std::vector<float>& x, float* y) const;

    // NLMS readout update on (x_past, realized y); returns the mse before update.
    float learn(const std::vector<float>& x_past, const float* y, float lr);

    void save(const std::string& path) const;
    bool load(const std::string& path);

private:
    std::vector<float>   Win_;    // kLsmN × kLsmIn
    std::vector<float>   Wres_;   // kLsmN × kLsmN
    std::vector<float>   Wout_;   // kLsmOut × kLsmN
    std::vector<float>   v_;      // membrane potentials
    std::vector<float>   spike_;  // last tick spikes (0/1)
    std::vector<float>   trace_;  // liquid state: x = λ_x·x + (1−λ_x)·s
};

LsmModule& lsm_module();

#endif // LSM_MODULE_HPP
