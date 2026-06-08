#ifndef IPPO_POLICY_HPP
#define IPPO_POLICY_HPP

#include "ObjectiveDMPolicy.hpp"  // ActorMLP, PPOParams, Experience, kPolicySz, kHid
#include "MapperPolicy.hpp"       // MapperCriticMLP (local-features critic)
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// IPPO — Independent PPO baseline (de Witt et al., 2020)
// ════════════════════════════════════════════════════════════════════════════
//
// Faithful reimplementation of the IPPO algorithm from
//   "Is Independent Learning All You Need in the StarCraft Multi-Agent
//    Challenge?" (de Witt et al., 2020, arXiv:2011.09533).
//
// Verbatim from paper Section 4:
//   "each agent a learns a local observation based critic V_φ(z^a_t)
//    parameterised by φ using GAE with γ = 0.99 and λ = 0.95.
//    The network parameters φ, θ are shared across critics, and actors,
//    respectively."
//
// → SHARED actor (1 set of weights θ used by every agent).
// → SHARED critic (1 set of weights φ used by every agent), with input
//   = each agent's local observation z^a_t.
// → Per-agent EXPERIENCE BUFFERS (each agent's trajectory is collected
//   separately for proper per-agent GAE).
// → Centralised state is NOT used (this is what makes it "independent").
//
//   ┌────────────────┬──────────────────────┬──────────────────────┬─────────────────────────┐
//   │                │ MAPPO (CTDE)         │ IPPO (de Witt 2020)  │ MAPPER (Decentr.+Ev)    │
//   ├────────────────┼──────────────────────┼──────────────────────┼─────────────────────────┤
//   │ Actor          │ 1 shared             │ 1 shared             │ N per-agent             │
//   │ Critic         │ 1 centralised, 20-d  │ 1 shared, 12-d local │ N per-agent, 12-d local │
//   │ Exploration    │ Entropy bonus        │ Entropy bonus        │ Entropy + Evolutionary  │
//   │ State for V    │ Global system state  │ Agent's own features │ Agent's own features    │
//   └────────────────┴──────────────────────┴──────────────────────┴─────────────────────────┘
//
// Implementation notes (matching paper):
//   • Same 12-d PolicyFeatures input as the actor (no global state needed).
//   • Same PPO hyper-parameters as MAPPO/MAPPER for a fair comparison.
//   • Per-agent buffers and per-agent GAE — each trajectory has its own
//     λ-return sweep, but uses the SHARED critic's V(s) estimates.
//   • Global advantage normalisation across all agents — both shared
//     networks see experiences from every agent and need comparable scales.
//   • Single critic update per mini-batch on the global batch (all agents'
//     entries mixed), because the critic weights are shared.

class IPPOPolicy {
public:
    // The IPPO defining feature: ONE shared actor + ONE shared critic.
    // Both are used by every agent. Critic input = agent's local obs.
    ActorMLP        actor;
    MapperCriticMLP critic;
    PPOParams       hparams;

    // Value normalisation (MAPPO Suggestion 1, also applied here for fairness
    // and stability across cities with very different reward scales).
    RunningMeanStd  value_rms;

    IPPOPolicy();

    // Singleton instance (parallels ObjectiveDMPolicy::shared() and
    // MapperPolicy::shared()).
    static IPPOPolicy& shared();

    // ── Inference ───────────────────────────────────────────────────────────
    // Shared actor — does not depend on agent_id.
    float score(const PolicyFeatures& features) const;

    // ── Data collection ────────────────────────────────────────────────────
    // Appends an Experience to buffers_[agent_id]. Returns the index inside
    // that buffer for later reward shaping. Per-agent buffer keeps the
    // trajectory structure intact so GAE can sweep one agent's history at
    // a time (the shared critic is queried by observation, not by agent id).
    int  record(int agent_id, const PolicyFeatures& obs,
                float action, float reward);

    void update_reward(int agent_id, int buf_idx, float reward);
    void add_to_reward(int agent_id, int buf_idx, float delta);

    int  buffer_size(int agent_id) const;
    int  total_buffer_size() const;

    // ── Recent-records log (same mechanism as MapperPolicy) ────────────────
    int                n_recent_records() const;
    std::pair<int,int> recent_record(int i) const;
    void               clear_recent_records();

    using TrainingStats = ObjectiveDMPolicy::TrainingStats;

    // ── Training ────────────────────────────────────────────────────────────
    // PPO update with shared actor + shared critic. Per-agent GAE then
    // global advantage normalisation, then global mini-batch updates on
    // both shared networks.
    TrainingStats train_epoch();

    void clear_buffer(int agent_id);
    void clear_buffer_all();

    void save(const std::string& path) const;
    bool load(const std::string& path);

    void set_progress(float progress);

    // Xavier-initialise both shared networks.
    void init_xavier(std::mt19937& rng);

    // Pre-allocate empty buffers for agent ids [0, n_agents).
    void ensure_agents(int n_agents);

    int  n_agents() const { return static_cast<int>(buffers_.size()); }

private:
    // Per-agent experience buffer. The SAME shared critic and shared actor
    // are used to score every entry — the buffer just keeps each agent's
    // trajectory contiguous for GAE.
    std::unordered_map<int, std::vector<Experience>> buffers_;

    std::mt19937                                     rng_;
    std::vector<std::pair<int,int>>                  recent_records_;

    // ── Adam optimiser state (β1=0.9, β2=0.999, ε=1e-5) ───────────────────
    // IPPO uses a local-observation critic (input dim = kPolicySz), so the
    // critic W1 buffer is kHid×kPolicySz (smaller than MAPPO's kHid×kCriticIn).
    policy_optim::AdamBuf<kHid * kPolicySz>  a_W1_adam_;
    policy_optim::AdamBuf<kHid>              a_b1_adam_;
    policy_optim::AdamBuf<kHid * kHid>       a_W2_adam_;
    policy_optim::AdamBuf<kHid>              a_b2_adam_;
    policy_optim::AdamBuf<kHid>              a_W3_adam_;
    policy_optim::AdamScalar                 a_b3_adam_;
    int                                      adam_t_actor_  = 0;

    policy_optim::AdamBuf<kHid * kPolicySz>  c_W1_adam_;
    policy_optim::AdamBuf<kHid>              c_b1_adam_;
    policy_optim::AdamBuf<kHid * kHid>       c_W2_adam_;
    policy_optim::AdamBuf<kHid>              c_b2_adam_;
    policy_optim::AdamBuf<kHid>              c_W3_adam_;
    policy_optim::AdamScalar                 c_b3_adam_;
    int                                      adam_t_critic_ = 0;

    std::vector<Experience>& get_or_create_buffer(int agent_id);

    // Per-trajectory GAE sweep (one agent's buffer at a time).
    void compute_gae(std::vector<Experience>& buf);

    struct MBStats { float loss; float entropy; float kl; float clip_frac; };

    // Actor update on a global mini-batch of (agent_id, buf_idx) pairs.
    MBStats update_actor_mb(const std::vector<std::pair<int,int>>& perm,
                             int start, int end);

    // Critic update on a global mini-batch (shared critic, all agents' data).
    float   update_critic_mb(const std::vector<std::pair<int,int>>& perm,
                              int start, int end);
};

#endif // IPPO_POLICY_HPP
