#ifndef FAITHFUL_MAPPER_POLICY_HPP
#define FAITHFUL_MAPPER_POLICY_HPP

#include "ObjectiveDMPolicy.hpp"  // reuses ActorMLP, PPOParams, Experience, kPolicySz, kHid
#include "MapperPolicy.hpp"       // reuses MapperCriticMLP, EvolutionaryParams (unused)
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// FaithfulMapperPolicy — paper-faithful MAPPER (Liu et al., IROS 2020)
// ════════════════════════════════════════════════════════════════════════════
//
// Companion to MapperPolicy.hpp. This class implements the SAME multi-agent
// architecture (per-agent decentralised actors + local critics) but with the
// EVOLUTIONARY STEP exactly as described in Algorithm 1 of the paper:
//
//     For each agent i:
//         Sample m ~ Uniform[0, 1]
//         R̄_i = R_i / (R_max - R_min)          ← normalised fitness
//         p_i = 1 - exp(η · R̄_i) / exp(η · R̄_best)
//         if m < p_i:
//             Θ_i ← Θ_best                      ← EXACT COPY, NO MUTATION
//
// Key differences vs the "enhanced" MapperPolicy (which uses elite-preservation
// + Gaussian mutation N(0, σ²)):
//
//   ┌──────────────────────┬──────────────────────────┬──────────────────────────┐
//   │                      │ MapperPolicy (enhanced)  │ FaithfulMapperPolicy     │
//   ├──────────────────────┼──────────────────────────┼──────────────────────────┤
//   │ Selection            │ Top elite_frac kept      │ Probabilistic replacement│
//   │                      │ Bottom worst_frac purged │ proportional to fitness  │
//   │                      │                          │ gap to best              │
//   │ Mutation             │ Gaussian N(0, 0.02²)     │ NONE (exact copy)        │
//   │                      │ on every replaced weight │                          │
//   │ Reference parent     │ Random pick from elites  │ Single best agent only   │
//   │ Fitness window       │ Rolling FIFO (5)         │ Rolling FIFO (5)         │
//   │ Cadence              │ Every period_rounds (5)  │ Every period_rounds (5)  │
//   └──────────────────────┴──────────────────────────┴──────────────────────────┘
//
// All other components (per-agent actors/critics, PPO training, GAE per
// trajectory, advantage normalisation, save/load format) are identical to
// MapperPolicy and reuse the same ActorMLP / MapperCriticMLP / Experience
// types — only the `evolutionary_step()` body differs.
//
// Why a separate class rather than a flag on MapperPolicy?
//   The singleton pattern means each class has exactly ONE shared instance.
//   Running both versions simultaneously in one training session requires
//   two distinct singletons → two distinct classes. This lets us train
//   MAPPER-enhanced and MAPPER-faithful side by side in the same Option M
//   call, on the same scenario draws, for a fair head-to-head comparison.

// ── Faithful evolutionary RL hyper-parameters ────────────────────────────────
struct FaithfulEvolutionaryParams {
    // η in the paper's formula p_i = 1 - exp(η·R̄_i)/exp(η·R̄_best).
    // Higher η → lower-fitness agents more likely to be replaced.
    // Paper default: η = 2.
    float eta             = 2.0f;

    // Rolling window for the fitness mean (matches MapperPolicy default).
    int   fitness_window  = 5;

    // Evolution cadence. Paper uses K = 50 episodes; one round in our trainer
    // = one episode per city × 6 cities = ~6 episodes, so K = 5 rounds
    // ≈ 30 episodes, somewhat tighter than the paper but matches what the
    // enhanced MapperPolicy uses (kept identical so the only difference at
    // training time is the mutation rule itself).
    int   period_rounds   = 5;
};

class FaithfulMapperPolicy {
public:
    // Same AgentState layout as MapperPolicy (per-agent actor + critic +
    // buffer + fitness window). The actor and critic types come from
    // ObjectiveDMPolicy.hpp / MapperPolicy.hpp; we just instantiate a
    // separate population here.
    struct AgentState {
        ActorMLP                actor;
        MapperCriticMLP         critic;
        std::vector<Experience> buffer;
        std::vector<float>      fitness_history;
    };

    PPOParams                  hparams;
    FaithfulEvolutionaryParams ev_params;

    FaithfulMapperPolicy();

    // Singleton instance — distinct from MapperPolicy::shared(), so both
    // populations exist simultaneously without interfering.
    static FaithfulMapperPolicy& shared();

    // ── Inference (read-only). Lazily allocates state for unseen agent_id. ──
    float score(int agent_id, const PolicyFeatures& features);

    // ── Data collection (same semantics as MapperPolicy) ────────────────────
    int  record(int agent_id, const PolicyFeatures& obs, float action, float reward);
    void update_reward(int agent_id, int buf_idx, float reward);
    void add_to_reward(int agent_id, int buf_idx, float delta);

    int  buffer_size(int agent_id) const;
    int  total_buffer_size() const;

    // ── Recent-records log (same as MapperPolicy) ──────────────────────────
    int                n_recent_records() const;
    std::pair<int,int> recent_record(int i) const;
    void               clear_recent_records();

    using TrainingStats = ObjectiveDMPolicy::TrainingStats;

    // ── Training (identical PPO + GAE recipe as MapperPolicy) ──────────────
    TrainingStats train_epoch();

    void clear_buffer(int agent_id);
    void clear_buffer_all();

    void save(const std::string& path) const;
    bool load(const std::string& path);

    void set_progress(float progress);
    void init_xavier_all(std::mt19937& rng);
    void ensure_agents(int n_agents);
    int  n_agents() const { return static_cast<int>(agents_.size()); }

    // ── Paper-faithful Evolutionary Reinforcement Learning ─────────────────
    //
    // Implements exactly Algorithm 1 (Liu et al., IROS 2020):
    //
    //   1. Compute R_i = current_fitness(i) for every agent.
    //   2. Find best agent j with R_best = max(R_i).
    //   3. Normalise: R̄_i = R_i / max(R_max - R_min, ε).
    //   4. For each agent i ≠ j:
    //        m ~ Uniform[0, 1]
    //        p_i = 1 - exp(η · R̄_i) / exp(η · R̄_best)
    //        if m < p_i:
    //            Θ_i ← Θ_j      ← COPY actor + critic, NO MUTATION
    //            fitness_history_i ← []   (reset child evaluation)
    //
    // Returns the number of agents whose weights were overwritten.
    int evolutionary_step();

    float current_fitness(int agent_id) const;
    void  record_episode_fitness();

private:
    std::unordered_map<int, AgentState>  agents_;
    std::mt19937                         rng_;
    std::vector<std::pair<int,int>>      recent_records_;

    AgentState& get_or_create(int agent_id);
    void        compute_gae(AgentState& a);

    struct MBStats { float loss; float entropy; float kl; float clip_frac; };
    MBStats update_actor_mb(AgentState& a,
                             const std::vector<int>& perm, int start, int end);
    float   update_critic_mb(AgentState& a,
                             const std::vector<int>& perm, int start, int end);
};

#endif // FAITHFUL_MAPPER_POLICY_HPP
