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
// A "middle ground" between MAPPO's CTDE and MAPPER's full decentralisation.
// It serves as the principled ablation that isolates the contribution of the
// local-vs-centralised critic from the actor-decentralisation and the
// evolutionary mechanism.
//
//   ┌────────────────┬──────────────────────┬──────────────────────┬─────────────────────────┐
//   │                │ MAPPO (CTDE)         │ IPPO                 │ MAPPER (Decentralized+Ev)│
//   ├────────────────┼──────────────────────┼──────────────────────┼─────────────────────────┤
//   │ Actor          │ 1 shared             │ 1 shared             │ N per-agent             │
//   │ Critic         │ 1 centralised, 20-d  │ N local, 13-d        │ N local, 13-d           │
//   │ Exploration    │ Entropy bonus        │ Entropy bonus        │ Entropy + Evolutionary  │
//   │ State for V    │ Global system state  │ Agent's own features │ Agent's own features    │
//   └────────────────┴──────────────────────┴──────────────────────┴─────────────────────────┘
//
// Ablation power: comparing the three baselines lets us attribute observed
// gains to specific design axes —
//   MAPPO  →  IPPO :  centralised critic vs local critic
//   IPPO   →  MAPPER:  shared actor vs per-agent actor + evolutionary mutation
//
// Implementation notes:
//   • Same 13-d PolicyFeatures input as the actor (no global state needed).
//   • Same PPO hyper-parameters as MAPPO/MAPPER for a fair comparison.
//   • Per-agent buffers and per-agent GAE — each critic learns from its own
//     agent's trajectory.
//   • Global advantage normalisation across all agents — the shared actor
//     sees experiences from every agent and needs comparable advantage scales.
//   • Critic updates done per-agent within each mini-batch (grouped by aid)
//     so each critic only sees its own data.
//   • The shared actor receives one combined PPO update from all mini-batch
//     entries regardless of which agent produced them.

class IPPOPolicy {
public:
    // The IPPO defining feature: ONE shared actor, used by every agent.
    ActorMLP   actor;
    PPOParams  hparams;

    // Per-agent local critic + experience buffer.
    struct LocalCritic {
        MapperCriticMLP         critic;
        std::vector<Experience> buffer;
    };

    IPPOPolicy();

    // Singleton instance (parallels ObjectiveDMPolicy::shared() and
    // MapperPolicy::shared()).
    static IPPOPolicy& shared();

    // ── Inference ───────────────────────────────────────────────────────────
    // Shared actor — does not depend on agent_id.
    float score(const PolicyFeatures& features) const;

    // ── Data collection ────────────────────────────────────────────────────
    // Appends an Experience to agents_[agent_id].buffer. Returns the index
    // inside that buffer for later reward shaping.
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
    // PPO update with shared actor + per-agent critics. Each agent's GAE
    // uses its own critic; advantages are globally normalised so the shared
    // actor sees consistent scales across agents.
    TrainingStats train_epoch();

    void clear_buffer(int agent_id);
    void clear_buffer_all();

    void save(const std::string& path) const;
    bool load(const std::string& path);

    void set_progress(float progress);

    // Xavier-initialise the shared actor and every existing critic.
    void init_xavier(std::mt19937& rng);

    // Pre-allocate state for agent ids [0, n_agents).
    void ensure_agents(int n_agents);

    int  n_agents() const { return static_cast<int>(agents_.size()); }

private:
    std::unordered_map<int, LocalCritic> agents_;
    std::mt19937                         rng_;
    std::vector<std::pair<int,int>>      recent_records_;

    LocalCritic& get_or_create(int agent_id);

    void compute_gae(LocalCritic& a);

    struct MBStats { float loss; float entropy; float kl; float clip_frac; };

    // Actor update on a global mini-batch of (agent_id, buf_idx) pairs.
    MBStats update_actor_mb(const std::vector<std::pair<int,int>>& perm,
                             int start, int end);

    // Critic update on an agent's own indices.
    float   update_critic_mb(LocalCritic& a,
                              const std::vector<int>& indices);
};

#endif // IPPO_POLICY_HPP
