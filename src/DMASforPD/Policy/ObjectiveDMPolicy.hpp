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
//
// Aligned with MAPPO state-of-the-art (Yu et al., 2022):
//   - Mini-batch SGD with gradient norm clipping
//   - Value function clipping (prevents destructive critic updates)
//   - KL-divergence early stopping (stops epoch loop when policy drifts too far)
//   - Global advantage normalisation (once per episode, before epoch loop)
struct PPOParams {
    float lr_actor      = 3e-4f;  // actor SGD learning rate
    float lr_critic     = 1e-3f;  // critic SGD learning rate
    float clip_eps      = 0.2f;   // PPO policy clip range ε
    float val_clip_eps  = 0.2f;   // value function clip range (same ε, MAPPO standard)
    float max_grad_norm = 0.5f;   // L2 gradient norm clipping
    float target_kl     = 0.01f;  // KL early-stop threshold (break epoch if exceeded)
    float gamma         = 0.99f;  // discount factor
    float lam_gae       = 0.95f;  // GAE λ
    float ent_w         = 0.01f;  // entropy bonus coefficient
    int   epochs        = 10;     // max PPO gradient epochs per train_epoch call
    int   batch_sz      = 256;    // mini-batch size (experiences per SGD step)
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
        float actor_loss  = 0.f;  // mean −L_clip  (positive = policy improving)
        float critic_loss = 0.f;  // mean max(MSE_unclip, MSE_clip)  (↓ = critic converging)
        float entropy     = 0.f;  // mean H(π) = −μlogμ − (1−μ)log(1−μ)  (↓ = less random)
        float adv_mean    = 0.f;  // mean raw advantage before normalisation  (should ≈ 0)
        float adv_std     = 0.f;  // std of advantages  (signal strength; > 0 = learning)
        float kl_approx   = 0.f;  // mean approx KL(π_old ‖ π_new) per epoch  (< target_kl)
        float clip_frac   = 0.f;  // fraction of samples where ratio r was clipped  (0.1–0.3)
        int   n_exp       = 0;    // experiences processed
        int   n_epochs    = 0;    // actual epochs run (may be < max if KL triggered early stop)
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

    // Mini-batch update stats.
    struct MBStats { float loss; float entropy; float kl; float clip_frac; };

    // Compute GAE independently per agent trajectory (keyed by Experience::agent_id).
    void compute_gae();

    // Single mini-batch SGD step for actor. `perm[start..end)` indexes buffer_.
    // Advantages must already be globally normalised before calling.
    // Returns (−L_clip, H(π), approx_KL, clip_fraction) averaged over the batch.
    MBStats update_actor_mb(const std::vector<int>& perm, int start, int end);

    // Single mini-batch SGD step for critic with value clipping.
    // `perm[start..end)` indexes buffer_ and global_states simultaneously.
    // Returns mean max(MSE_unclip, MSE_clip) over the batch.
    float   update_critic_mb(const std::vector<int>& perm, int start, int end,
                              const std::vector<std::array<float, kGlobSz>>& gs);
};

#endif // OBJECTIVE_DM_POLICY_HPP
