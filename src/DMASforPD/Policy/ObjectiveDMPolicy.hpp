#ifndef OBJECTIVE_DM_POLICY_HPP
#define OBJECTIVE_DM_POLICY_HPP

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

// ── Normalisation constants ────────────────────────────────────────────────────
static constexpr int   kPolicySz   = 10;        // feature vector length
static constexpr int   kHid        = 64;         // hidden layer width (actor & critic)
static constexpr int   kGlobSz     = 20;         // global state vector (critic input)
static constexpr float kCostScale  = 10'000.f;   // 10 km reference
static constexpr float kQueueScale = 50'000.f;   // 50 km reference
static constexpr float kMaxLoad    = 10.f;        // max tasks per agent
static constexpr float kMaxAgents  = 20.f;        // max fleet size
static constexpr float kImpMax     = 10.f;        // max task importance

// ── Feature vector ────────────────────────────────────────────────────────────
//
// Built at try_accept_task() time from TaskOffer + GlobalMemory + local state.
// All fields are clamped to [0, 1].
struct PolicyFeatures {
    float cost_diff       = 0.f; // cheapest insertion cost / kCostScale
    float profit_rate     = 0.f; // reward / (insertion_cost * 0.5 + ε)
    float current_load    = 0.f; // tasks.size() / kMaxLoad
    float queue_duration  = 0.f; // planned route cost / kQueueScale
    float efficiency_loss = 0.f; // insertion_cost / (route_cost + ε)
    float rank_in_call    = 0.f; // 1 − prev_agents.size() / kMaxAgents
    float task_importance = 0.f; // importance / kImpMax
    float n_agents_ratio  = 0.f; // active agents / kMaxAgents
    float n_alloc_ratio   = 0.f; // allocated tasks / total tasks
    float n_avail_ratio   = 0.f; // available tasks / total tasks

    void to_array(float* dst) const;
};

// ── Actor MLP: kPolicySz → kHid → kHid → 1 (sigmoid) ────────────────────────
struct ActorMLP {
    float W1[kHid * kPolicySz]{};
    float b1[kHid]{};
    float W2[kHid * kHid]{};
    float b2[kHid]{};
    float W3[kHid]{};  // output-layer weights (single neuron)
    float b3 = 1.0f;   // positive bias → initial μ ≈ 0.73 (prefer acceptance)

    // Forward pass. Optional output pointers cache intermediate activations for backprop.
    float forward(const float* x,
                  float* h1  = nullptr, float* h2  = nullptr,
                  float* pa1 = nullptr, float* pa2 = nullptr) const;

    void init_xavier(std::mt19937& rng);
};

// ── Critic MLP: kGlobSz → kHid → kHid → 1 (linear, no sigmoid) ──────────────
//
// Takes a global state vector and returns a raw scalar value estimate V(s).
struct CriticMLP {
    float W1[kHid * kGlobSz]{};
    float b1[kHid]{};
    float W2[kHid * kHid]{};
    float b2[kHid]{};
    float W3[kHid]{};
    float b3 = 0.f;

    float forward(const float* x,
                  float* h1  = nullptr, float* h2  = nullptr,
                  float* pa1 = nullptr, float* pa2 = nullptr) const;

    void init_xavier(std::mt19937& rng);
};

// ── Episode experience record ─────────────────────────────────────────────────
struct Experience {
    std::array<float, kPolicySz> obs{};
    int   agent_id    = -1;  // which delivery agent made this decision
    float action      = 0.f; // 1 = accept, 0 = reject
    float log_prob    = 0.f; // log π_old(a|s) at decision time
    float reward      = 0.f; // set to 0 at decision; updated to task value on delivery
    float value       = 0.f; // V(s) from critic (filled during train_epoch)
    float advantage   = 0.f; // GAE estimate (filled during train_epoch)
    float ret         = 0.f; // discounted return (filled during train_epoch)
};

// ── PPO hyper-parameters ──────────────────────────────────────────────────────
struct PPOParams {
    float lr_actor  = 3e-4f; // actor SGD learning rate
    float lr_critic = 1e-3f; // critic SGD learning rate
    float clip_eps  = 0.2f;  // PPO clip range
    float gamma     = 0.99f; // discount factor
    float lam_gae   = 0.95f; // GAE lambda
    float ent_w     = 0.01f; // entropy bonus coefficient
    float val_w     = 0.5f;  // critic loss coefficient (unused in separated updates)
    int   epochs    = 4;     // PPO gradient epochs per train_epoch call
    int   batch_sz  = 32;    // mini-batch size (reserved for future mini-batching)
};

// ── MAPPO shared policy (singleton) ──────────────────────────────────────────
//
// All delivery agents share one ActorMLP (parameter sharing).
// The centralised CriticMLP takes a global state vector to compute V(s).
//
// Workflow per episode:
//   1. Each agent makes accept/reject decisions → record() is called per decision.
//   2. At episode end, the simulator calls train_epoch(global_states) once.
//   3. Updated weights are immediately shared across all agents.
//
// Before training begins (untrained weights), the actor outputs μ ≈ 0.73
// (biased toward acceptance) so the system behaves like the previous always-accept
// policy while training converges.
class ObjectiveDMPolicy {
public:
    ActorMLP  actor;
    CriticMLP critic;
    PPOParams hparams;

    ObjectiveDMPolicy();

    // Shared instance — call from every delivery agent.
    static ObjectiveDMPolicy& shared();

    // ── Inference ──────────────────────────────────────────────────────────
    // Returns μ ∈ [0,1]. TAM threshold: μ ≥ 0.5 → accept.
    float score(const PolicyFeatures& features) const;

    // ── Data collection ────────────────────────────────────────────────────
    // Record one accept/reject decision for later training.
    //   action     : 1.f if score ≥ 0.5 (accepted), 0.f otherwise.
    //   reward     : pass 0.f at decision time; use update_reward() on delivery.
    //   agent_id   : which delivery agent made this decision (for per-agent GAE).
    void record(const PolicyFeatures& obs, float action, float reward, int agent_id);

    // Update the reward for a previously recorded experience.
    // Called at task delivery to set the completion reward on the accept entry.
    // buffer_idx must be < buffer_size(); silently ignored if out of range.
    void update_reward(int buffer_idx, float reward);

    // ── Training stats (populated by train_epoch) ─────────────────────────
    struct TrainingStats {
        float actor_loss  = 0.f;  // mean −L_clip (last actor epoch; higher = policy improving)
        float critic_loss = 0.f;  // mean (V − ret)²  (should decrease)
        float entropy     = 0.f;  // mean H(π) = −μlogμ − (1−μ)log(1−μ) (exploration measure)
        float adv_mean    = 0.f;  // mean raw advantage before normalisation (should stay ≈0)
        float adv_std     = 0.f;  // std of advantages (signal strength; >0 = learning signal)
        int   n_exp       = 0;    // experiences in buffer at training time
    };

    // ── Training ───────────────────────────────────────────────────────────
    // Run one PPO update cycle over the buffered experiences.
    // global_states : one kGlobSz-length array per buffered experience (same order).
    //                 If empty, critic is skipped and advantages = raw rewards.
    // Clears the buffer after training. Returns per-epoch averaged stats.
    TrainingStats train_epoch(
        const std::vector<std::array<float, kGlobSz>>& global_states = {});

    // ── Persistence ───────────────────────────────────────────────────────
    void save(const std::string& path) const;
    bool load(const std::string& path);

    int  buffer_size() const { return static_cast<int>(buffer_.size()); }
    void clear_buffer()      { buffer_.clear(); }

private:
    std::vector<Experience> buffer_;
    std::mt19937            rng_;

    struct EpochActorStats { float actor_loss; float entropy; float adv_mean; float adv_std; };

    // Compute GAE independently per agent trajectory (keyed by Experience::agent_id).
    // Avoids cross-agent bootstrap errors when the buffer mixes decisions from
    // multiple agents within the same episode.
    void            compute_gae();
    EpochActorStats update_actor();
    float           update_critic(const std::vector<std::array<float, kGlobSz>>& gs);
};

#endif // OBJECTIVE_DM_POLICY_HPP
