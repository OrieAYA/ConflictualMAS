#ifndef OBJECTIVE_DM_POLICY_HPP
#define OBJECTIVE_DM_POLICY_HPP

#include "PolicyOptim.hpp"

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

// ── Normalisation constants ────────────────────────────────────────────────────
static constexpr int   kPolicySz   = 12;        // feature vector length (actor input)
static constexpr int   kHid        = 64;         // hidden layer width (actor & critic)
static constexpr int   kGlobSz     = 20;         // system-only global state storage
// MAPPO Suggestion 2 (Yu 2022 §5.2): the centralised critic should consume an
// AGENT-SPECIFIC global state (Feature-Pruned variant). Here the input is the
// concat of the 20-d system state and the 12-d PolicyFeatures of the agent
// making the decision — built on-the-fly inside train_epoch and update_critic_mb
// from buffer_[i].obs (already stored) plus global_states[i] (system, already
// stored). Avoids changing the on-disk format of global_states_ while giving
// the critic the agent-specific signal that EP-only inputs lack.
static constexpr int   kCriticIn   = kGlobSz + kPolicySz;   // 32
static constexpr float kCostScale  = 10'000.f;   // 10 km reference
static constexpr float kQueueScale = 50'000.f;   // 50 km reference
static constexpr float kMaxLoad    = 10.f;        // max tasks per agent
static constexpr float kMaxAgents  = 20.f;        // max fleet size
static constexpr float kImpMax     = 10.f;        // max task importance

// ── Feature vector ────────────────────────────────────────────────────────────
//
// Built at try_accept_task() time from TaskOffer + GlobalMemory + local state.
// All fields are clamped to [0, 1].
//
// V2 (post-input-vector-redesign) — 12 features grouped by semantics:
//
//   GLOBAL CONTEXT (4)     : profit_rate, n_agents_ratio, n_alloc_ratio, time_remaining
//   AGENT STATE (3)        : queue_duration, load_at_insertion, efficiency_loss
//   COMPETITION+IMPACT (3) : marginal_cost_relative, fleet_pressure, divergence_ratio
//   SPATIO-TEMPORAL (2)    : congestion_delta_contribution, area_heat_pickup (placeholder)
struct PolicyFeatures {
    // ── Global context (system-wide signals, ~constant within an offer) ──────
    float profit_rate       = 0.f; // task reward / insertion_cost  ∈ [0,1]
    float n_agents_ratio    = 0.f; // active agents / kMaxAgents
    float n_alloc_ratio     = 0.f; // allocated tasks / total tasks
    float time_remaining    = 0.f; // 1 − step/total_steps ∈ [0,1]

    // ── Agent state (this agent's situation) ─────────────────────────────────
    float queue_duration    = 0.f; // planned route cost / kQueueScale
    float load_at_insertion = 0.f; // tasks onboard AT THE PICKUP STEP / kMaxLoad
                                   // (accounts for drops happening before pickup)
    float efficiency_loss   = 0.f; // insertion_cost / (route_cost + insertion + ε)

    // ── Competition + impact (per-candidate differentiators) ────────────────
    float marginal_cost_relative = 0.f; // (my_cost − cheapest_other) / cheapest_other
                                        // ∈ [0,1] : 0 = I'm cheapest, 1 = I'm ≥2× more expensive.
                                        // The signal that tells the policy "am I really
                                        // the best fit, or just a candidate ?"
    float fleet_pressure         = 0.f; // n_candidates / max_candidates ∈ [0,1]
                                        // High = many competitors → I can be picky.
                                        // Low  = I'm the only option → must accept.
    float divergence_ratio       = 0.f; // (1 − cos(angle(my_heading, pickup_dir))) / 2 ∈ [0,1]
                                        // 0 = pickup is on my way, 1 = pickup is behind me.

    // ── Spatio-temporal context ──────────────────────────────────────────────
    float congestion_delta_contribution = 0.f; // BPR I'd add by accepting (proxy)
    float area_heat_pickup              = 0.f; // task density × congestion @ pickup
                                               // (placeholder until RegionStatsGrid lands)

    void to_array(float* dst) const;
};

// ── Actor MLP: kPolicySz → kHid → kHid → 1 (sigmoid) ────────────────────────
struct ActorMLP {
    float W1[kHid * kPolicySz]{};
    float b1[kHid]{};
    float W2[kHid * kHid]{};
    float b2[kHid]{};
    float W3[kHid]{};  // output-layer weights (single neuron, shrunk ×0.01 at init)
    float b3 = 0.0f;   // zero bias + small W3 → initial μ ≈ 0.5 (neutral policy)

    // Forward pass. Optional output pointers cache intermediate activations for backprop.
    float forward(const float* x,
                  float* h1  = nullptr, float* h2  = nullptr,
                  float* pa1 = nullptr, float* pa2 = nullptr) const;

    void init_xavier(std::mt19937& rng);
};

// ── Critic MLP: kCriticIn → kHid → kHid → 1 (linear, no sigmoid) ────────────
//
// Takes a FP-state vector (system 20-d + agent-specific 12-d, MAPPO Suggestion
// 2) and returns a raw scalar value estimate V(s). The caller is expected to
// assemble the kCriticIn-sized vector by concatenating GlobalState::to_array()
// (20 floats, system) with the per-agent PolicyFeatures (12 floats, captured
// at decision time). Both consumers (train_epoch frozen V_old, update_critic_mb
// forward) build the concat inline — no on-disk format change to global_states.
struct CriticMLP {
    float W1[kHid * kCriticIn]{};
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

// ── Running mean/std tracker — used for value normalisation ─────────────────
//
// MAPPO paper Suggestion 1 (Yu et al. 2022, §5.1): stabilise value learning by
// regressing the critic against NORMALISED return targets (ret - μ) / σ and
// denormalising V̂(s) = V_net(s) · σ + μ everywhere it is consumed (GAE, advantage
// estimation, downstream value queries). The running mean/std are updated with
// Welford's online algorithm from the unnormalised returns of each training pass.
// The single update path keeps numerical stability while letting the critic's
// effective output scale auto-adapt to reward distributions that drift across
// city/phase (which is exactly our setting: throughput-based rewards differ a lot
// between Tokyo_Small eval and Tokyo_Large generalize).
struct RunningMeanStd {
    double mean    = 0.0;
    double var     = 1.0;
    long long count = 0;

    void update(float x) {
        ++count;
        const double delta = static_cast<double>(x) - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = static_cast<double>(x) - mean;
        var = (count > 1)
            ? var + (delta * delta2 - var) / static_cast<double>(count)
            : 1.0;
    }

    void update_batch(const float* xs, int n) {
        for (int i = 0; i < n; ++i) update(xs[i]);
    }

    float std_dev() const {
        return static_cast<float>(std::sqrt(std::max(1e-6, var)));
    }
    float mean_f() const { return static_cast<float>(mean); }

    // Persistence — RMS state must be saved alongside critic weights so that
    // resumed/loaded policies keep the same value scale.
    void save_to(void* file) const;   // FILE* (kept void* to avoid <cstdio> here)
    bool load_from(void* file);
};

// ── PPO hyper-parameters ──────────────────────────────────────────────────────
//
// Aligned with MAPPO state-of-the-art (Yu et al., 2022):
//   - Mini-batch SGD with gradient norm clipping
//   - Value function clipping (prevents destructive critic updates)
//   - KL-divergence early stopping (stops epoch loop when policy drifts too far)
//   - Global advantage normalisation (once per episode, before epoch loop)
struct PPOParams {
    // Effective lr at each train_epoch — annealed linearly from *_init to *_min
    // by ObjectiveDMPolicy::set_progress(progress ∈ [0,1]). MAPPO SoTA (Yu+2022)
    // and standard PPO practice use linear LR decay over training.
    // Values aligned with Yu et al. 2022 appendix (lr=5e-4 for both networks).
    float lr_actor      = 5e-4f;  // current actor lr (mutated by set_progress)
    float lr_actor_init = 5e-4f;  // value at progress=0 (Yu+2022 default)
    float lr_actor_min  = 5e-5f;  // value at progress=1 (×0.1 of init)
    float lr_critic     = 5e-4f;  // current critic lr
    float lr_critic_init = 5e-4f; // Yu+2022 default
    float lr_critic_min  = 5e-5f;
    // Clip range: Yu+2022 §5.3 reports smaller ε (0.05–0.1) yields more stable
    // and higher final performance in MARL than the single-agent default 0.2;
    // we pick 0.1 as a balanced compromise (slower than 0.2 but stable).
    float clip_eps      = 0.1f;   // PPO policy clip range ε (MARL-tuned)
    float val_clip_eps  = 0.1f;   // value function clip range (same ε)
    // Gradient norm clip — Yu+2022 Tab.7 specifies 10.0 for MAPPO; deWitt+2020
    // §4 specifies < 0.5 for IPPO. The IPPO policy overrides this value to 0.5
    // in its constructor; MAPPO/MAPPER/FaithfulMAPPER use the default below.
    float max_grad_norm = 10.0f;  // L2 gradient norm clipping (Yu+2022 Tab.7)
    float target_kl     = 0.01f;  // KL early-stop threshold (break epoch if exceeded)
    float gamma         = 0.99f;  // discount factor
    float lam_gae       = 0.95f;  // GAE λ
    float ent_w         = 0.01f;  // entropy bonus coefficient (current, annealed)
    float ent_w_init    = 0.01f;  // Yu+2022 default (sufficient for exploration)
    float ent_w_min     = 0.001f; // low late — lets policy commit to a strategy
    int   epochs        = 10;     // max PPO gradient epochs per train_epoch call
    // MAPPO paper Suggestion 3 (Yu et al. 2022, §5.3): avoid splitting data
    // into mini-batches in cooperative multi-agent settings. They report best
    // performance with 1 mini-batch on 22/23 SMAC maps. The value is sized
    // beyond any realistic buffer size so update_actor/critic_mb processes the
    // entire buffer in one pass per epoch — true full-batch SGD.
    int   batch_sz      = 1 << 20;
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

    // Running mean/std of return targets. Updated at the start of every
    // train_epoch from the unnormalised returns. Used to (a) normalise critic
    // targets during the SGD updates and (b) denormalise V(s) outputs whenever
    // they are consumed downstream (GAE bootstrap, advantage computation).
    RunningMeanStd value_rms;

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

    // Update the reward for a previously recorded experience (SETS reward).
    // Called at task delivery to set the completion reward on the accept entry.
    // buffer_idx must be < buffer_size(); silently ignored if out of range.
    void update_reward(int buffer_idx, float reward);

    // Add `delta` to the existing reward (additive semantics).
    // Use this for reward shaping where multiple events contribute to one
    // experience: e.g., +pickup_reward at pickup THEN +delivery_reward at
    // delivery, or pickup credit + unfinished penalty if delivery never happens.
    void add_to_reward(int buffer_idx, float delta);

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

    // Set linear training progress ∈ [0,1] (0 = start, 1 = end). Anneals
    // hparams.lr_actor, hparams.lr_critic and hparams.ent_w from their *_init
    // toward their *_min values. Call this BEFORE each training episode so the
    // next train_epoch picks up the annealed values.
    void set_progress(float progress);

private:
    std::vector<Experience> buffer_;
    std::mt19937            rng_;

    // ── Adam optimiser state (paper-aligned: β1=0.9, β2=0.999, ε=1e-5) ──────
    // Stored alongside the networks (not inside ActorMLP/CriticMLP) so the
    // on-disk weight format stays stable; Adam moments are NOT persisted and
    // reset to zero on load. Step counters are pre-incremented before each
    // adam_apply (mandatory: bias correction with t=0 would divide by zero).
    policy_optim::AdamBuf<kHid * kPolicySz>  a_W1_adam_;
    policy_optim::AdamBuf<kHid>              a_b1_adam_;
    policy_optim::AdamBuf<kHid * kHid>       a_W2_adam_;
    policy_optim::AdamBuf<kHid>              a_b2_adam_;
    policy_optim::AdamBuf<kHid>              a_W3_adam_;
    policy_optim::AdamScalar                 a_b3_adam_;
    int                                      adam_t_actor_  = 0;

    policy_optim::AdamBuf<kHid * kCriticIn>  c_W1_adam_;
    policy_optim::AdamBuf<kHid>              c_b1_adam_;
    policy_optim::AdamBuf<kHid * kHid>       c_W2_adam_;
    policy_optim::AdamBuf<kHid>              c_b2_adam_;
    policy_optim::AdamBuf<kHid>              c_W3_adam_;
    policy_optim::AdamScalar                 c_b3_adam_;
    int                                      adam_t_critic_ = 0;

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
