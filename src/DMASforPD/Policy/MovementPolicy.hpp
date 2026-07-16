#ifndef MOVEMENT_POLICY_HPP
#define MOVEMENT_POLICY_HPP

#include "DMASforPD/Policy/PolicyKit.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// Movement Decision Policy — PPO gate on local TD-A* replanning.
//
// One decision per node arrival: should the agent request a local replan of
// its current leg? Observation = 4 scalars (congestion of the next planned
// edge now, same edge at planning time, min congestion of the other visible
// edges, LSM area alert) + the local star graph (incident edges, permutation-
// invariant encoder). Mono-agent: one shared actor/critic for the fleet.
// Self-contained: reuses policy_optim / Mlp / PPOParams / RunningMeanStd but
// touches none of the bid-policy structures (Experience, ppo_train).
// ════════════════════════════════════════════════════════════════════════════

static constexpr int kMoveEdgeSlots = 8;   // max incident edges kept
static constexpr int kMoveEdgeFeat  = 4;   // [cong_now, plan_cong, is_next, is_from]
static constexpr int kMoveSelfFeat  = 4;   // [f1 f2 f3 f4=lsm_alert]
static constexpr int kMoveObsSz     = kMoveSelfFeat + kMoveEdgeSlots * kMoveEdgeFeat;
static constexpr int kMoveEnc       = 16;  // per-edge encoder width
static constexpr int kMovePooled    = kMoveSelfFeat + 2 * kMoveEnc;   // head input

struct MoveObs {
    std::array<float, kMoveObsSz> x{};
    int n_edges = 0;
};

struct MoveExperience {
    std::array<float, kMoveObsSz> obs{};
    int   n_edges    = 0;
    int   step       = 0;    // decision step
    int   pred_steps = 0;    // planned remaining steps to the leg objective
    float action     = 0.f;  // 1 = replan
    float log_prob   = 0.f;
    float reward     = 0.f;
    float value      = 0.f;
    float advantage  = 0.f;
    float ret        = 0.f;
};

// Forward activations cached for backprop.
struct StarCache {
    float pre[kMoveEdgeSlots][kMoveEnc];
    float pooled[kMovePooled];
    float h1[kHid], h2[kHid], pa1[kHid], pa2[kHid];
    const float* obs = nullptr;
    int n = 0;
};

struct StarNet;

struct StarGrads {
    std::vector<float> dWe, dbe;
    MlpGrads head;

    void resize_for(const StarNet& net);
    void zero();
    void clip_global_norm(float max_norm);
    void negate();
};

struct StarAdam {
    std::vector<float> mWe, vWe, mbe, vbe;
    MlpAdam head;
    int t = 0;

    void resize_for(const StarNet& net);
    void apply(StarNet& net, const StarGrads& g, float lr);
};

// Shared per-edge linear encoder + ReLU, mean‖max pooling, Mlp head.
struct StarNet {
    std::vector<float> We, be;   // kMoveEnc×kMoveEdgeFeat, kMoveEnc
    Mlp head;                    // in_dim = kMovePooled

    void  init(std::mt19937& rng, float out_gain);
    float forward(const MoveObs& o, StarCache* c = nullptr) const;
    void  backprop(const StarCache& c, float dz3, StarGrads& g) const;

    void save_to(std::FILE* f) const;
    bool load_from(std::FILE* f);
};

class MovementPolicy {
public:
    PPOParams hparams;   // IPPO-style column

    MovementPolicy();

    float score(const MoveObs& o) const;
    bool  decide(float mu, bool explore);

    int  record(int agent_id, const MoveObs& o, float mu, float action,
                int step, int pred_steps);
    void add_reward(int agent_id, int idx, float delta);

    // Deferred leg-outcome credit: every open decision of the leg receives
    // w_out · clamp((pred − actual)/pred, −1, 1) when the objective is reached.
    void credit_leg_outcome(int agent_id, int arrival_step, float w_out);
    void drop_open(int agent_id);

    TrainingStats train_round();
    void set_progress(float p) { hparams.set_progress(p); }
    void reinit(uint32_t seed);

    void save(const std::string& path) const;
    bool load(const std::string& path);

    void clear_buffers();
    int  total_buffer_size() const;

private:
    StarNet  actor_, critic_;
    StarAdam actor_opt_, critic_opt_;
    RunningMeanStd vrms_;
    std::mt19937   rng_{0xC0FFEEu};

    std::unordered_map<int, std::vector<MoveExperience>> buffers_;
    std::unordered_map<int, std::vector<int>>            open_;
};

MovementPolicy& movement_policy();

#endif // MOVEMENT_POLICY_HPP
