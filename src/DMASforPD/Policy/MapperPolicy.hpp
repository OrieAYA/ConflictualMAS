#ifndef MAPPER_POLICY_HPP
#define MAPPER_POLICY_HPP

#include "ObjectiveDMPolicy.hpp"  // reuses ActorMLP, PPOParams, Experience, kPolicySz, kHid
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// MAPPER — Multi-Agent Path Planning with Evolutionary RL, adapted for L-GPDP
// ════════════════════════════════════════════════════════════════════════════
//
// Adaptation of MAPPER [Liu, Sartoretti, et al., IROS 2020] to the Lifelong
// General Pickup-and-Delivery Problem. The three defining MAPPER properties
// are preserved; components that do not apply to a feature-vector PDP setting
// are replaced with L-GPDP-specific mechanisms.
//
// ── MAPPER properties preserved ─────────────────────────────────────────────
//
//   1. **Decentralised per-agent policies**:
//      Each delivery agent owns its OWN actor MLP — no parameter sharing.
//      A new agent first encountered at training time gets a Xavier-initialised
//      network. Agent k uses actor[k] only. Same-slot policies persist across
//      cities (a population-of-policies interpretation of decentralisation,
//      since our agents have homogeneous roles unlike MAPPER's heterogeneous
//      agents in a single environment).
//
//   2. **Local-only critic**:
//      Each agent also owns its own V_k(s) which takes the 13-dim local
//      PolicyFeatures vector — no centralised global state. Advantage is
//      estimated purely from the agent's own observation.
//
//   3. **Evolutionary Reinforcement Learning** (the "Ev" in MAPPER):
//      Every `period_rounds` training rounds, agents are ranked by their
//      rolling fitness (mean episode reward over the last `fitness_window`
//      episodes). The top `elite_frac` are kept; the bottom `worst_frac`
//      have their (actor + critic) weights replaced by a copy of a random
//      elite + Gaussian noise N(0, mutation_std²). This injects an explicit
//      exploration mechanism on top of PPO's entropy bonus and lets bad
//      decentralised policies catch up to good ones.
//
// ── L-GPDP-specific adaptations ─────────────────────────────────────────────
//
//   • PPO replaces A2C — our shared infra is already PPO; A2C is just PPO
//     without the clip, so this is a strict improvement.
//   • MLP replaces LSTM — our observation is a 13-dim explicit feature vector
//     (deliverability, capacity load, congestion, etc.), no time-series window
//     is needed. The original LSTM handles raw FOV images.
//   • Feature vector replaces local FOV image — see above.
//   • Fitness = mean reward over the rolling window (captures throughput,
//     refusal accuracy, and unfinished-task avoidance in one scalar — all
//     three are encoded by our reward shaping).
//   • Cross-city policy sharing — agent slot k holds one set of weights
//     across all cities (Tokyo, Kyoto, LA), preserving decentralisation
//     between slots while letting each policy benefit from multi-city
//     experience.
//
// ── Comparison vs MAPPO (CTDE) ──────────────────────────────────────────────
//
//   ┌──────────────┬─────────────────────────────┬─────────────────────────────┐
//   │              │ MAPPO (CTDE)                │ MAPPER (decentralised + Ev) │
//   ├──────────────┼─────────────────────────────┼─────────────────────────────┤
//   │ Actor        │ 1 shared (all agents)       │ N independent (one per id)  │
//   │ Critic       │ 1 centralised, 20-d global  │ N independent, 13-d local   │
//   │ GAE          │ Per-agent traj, global norm │ Per-agent traj, per-agent   │
//   │ Buffer       │ Single shared buffer        │ One buffer per agent        │
//   │ Exploration  │ Entropy bonus               │ Entropy + Evolutionary      │
//   │              │                             │ mutation/selection          │
//   │ Eval action  │ μ ≥ 0.5 → accept            │ same                        │
//   └──────────────┴─────────────────────────────┴─────────────────────────────┘

// ── Evolutionary RL hyper-parameters ────────────────────────────────────────
struct EvolutionaryParams {
    float elite_frac      = 0.2f;   // top X% kept as parents
    float worst_frac      = 0.2f;   // bottom Y% replaced by mutated copies
    float mutation_std    = 0.02f;  // Gaussian noise scale on every weight
    int   fitness_window  = 5;      // rolling # of episodes for fitness mean
    int   period_rounds   = 5;      // do an evolution step every K rounds
};

// ── Local-features critic ────────────────────────────────────────────────────
// Same hidden geometry as the actor (kPolicySz → kHid → kHid → 1) and same
// initialisation rules. Output is the scalar value V(s) — no sigmoid.
struct MapperCriticMLP {
    float W1[kHid * kPolicySz]{};
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

class MapperPolicy {
public:
    // Per-agent slate of networks, experience buffer, and rolling fitness.
    struct AgentState {
        ActorMLP                actor;
        MapperCriticMLP         critic;
        std::vector<Experience> buffer;
        // Rolling window of recent episode fitnesses (mean reward per record).
        // Used by evolutionary_step() to rank agents. Trimmed to
        // EvolutionaryParams::fitness_window entries; FIFO eviction.
        std::vector<float>      fitness_history;
    };

    PPOParams          hparams;
    EvolutionaryParams ev_params;

    MapperPolicy();

    // Singleton instance (parallels ObjectiveDMPolicy::shared()).
    static MapperPolicy& shared();

    // ── Inference (read-only). Lazily allocates state for unseen agent_id. ──
    float score(int agent_id, const PolicyFeatures& features);

    // ── Data collection ────────────────────────────────────────────────────
    // Bernoulli-style record: appends one Experience to agents_[agent_id].buffer.
    // Returns the index inside that buffer for later reward updates.
    int record(int agent_id, const PolicyFeatures& obs, float action, float reward);

    // Reward shaping: same semantics as ObjectiveDMPolicy.
    void update_reward(int agent_id, int buf_idx, float reward);
    void add_to_reward(int agent_id, int buf_idx, float delta);

    int  buffer_size(int agent_id) const;
    int  total_buffer_size() const;

    // ── Recent-records log ──────────────────────────────────────────────────
    // Each call to record() appends an (agent_id, buf_idx) pair into a global
    // log. EpisodeRunner brackets the TAM iteration with n_recent_records()
    // values before/after, then applies the refusal penalty to losing offers
    // and shaping rewards to the winner — without having to walk a single
    // shared buffer (MAPPER has N per-agent buffers).
    int                n_recent_records() const;
    std::pair<int,int> recent_record(int i) const;  // (agent_id, buf_idx)
    void               clear_recent_records();

    using TrainingStats = ObjectiveDMPolicy::TrainingStats;

    // ── Training ────────────────────────────────────────────────────────────
    // Independent A2C/PPO update per agent over their own buffer. Returns
    // averaged stats across agents (weighted by buffer size). Empty buffers
    // are skipped. Clears all buffers at the end.
    TrainingStats train_epoch();

    void clear_buffer(int agent_id);
    void clear_buffer_all();

    // ── Persistence ─────────────────────────────────────────────────────────
    void save(const std::string& path) const;
    bool load(const std::string& path);

    // Linear anneal of lr_actor, lr_critic, ent_w (same recipe as MAPPO).
    void set_progress(float progress);

    // Force Xavier initialisation of *existing* agent states with `rng` as
    // the seed source. Newly seen agents will use their own thread-local RNG.
    void init_xavier_all(std::mt19937& rng);

    // Pre-allocate state for agent ids [0, n_agents).
    void ensure_agents(int n_agents);

    // Number of distinct agents currently tracked (for reporting / save).
    int  n_agents() const { return static_cast<int>(agents_.size()); }

    // ── Evolutionary Reinforcement Learning ─────────────────────────────────
    // Rank agents by their rolling-window fitness (mean reward per record).
    // The top `elite_frac` are preserved as parents; the bottom `worst_frac`
    // are overwritten with (parent_weights + Gaussian noise * mutation_std).
    // Their fitness_history is reset so the mutated child is not immediately
    // marked as bad. Resets recent_records as a side effect (Evolution should
    // be called BETWEEN episodes, not during a task offer).
    //
    // Returns the number of agents that were mutated (0 if the population is
    // too small or fitness data is insufficient).
    int  evolutionary_step();

    // Per-agent rolling-mean fitness — -inf if no data yet.
    float current_fitness(int agent_id) const;

    // Force update fitness history from current per-agent buffer means.
    // Called automatically inside train_epoch() before buffer clearing, so
    // training code does not usually need to invoke this directly.
    void record_episode_fitness();

private:
    std::unordered_map<int, AgentState>  agents_;
    std::mt19937                         rng_;
    std::vector<std::pair<int,int>>      recent_records_;

    // Lazy creator: returns reference to agents_[agent_id], creating it
    // with Xavier-init networks if absent.
    AgentState& get_or_create(int agent_id);

    // Per-agent GAE — single trajectory, no cross-agent bootstrap (the
    // buffer is already per-agent so this is just a reverse sweep).
    void compute_gae(AgentState& a);

    // PPO mini-batch step. Returns (−L_clip, H, approx_KL, clip_fraction).
    struct MBStats { float loss; float entropy; float kl; float clip_frac; };
    MBStats update_actor_mb(AgentState& a,
                             const std::vector<int>& perm, int start, int end);
    float   update_critic_mb(AgentState& a,
                             const std::vector<int>& perm, int start, int end);
};

#endif // MAPPER_POLICY_HPP
