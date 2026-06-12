#ifndef MAPPER_POLICY_HPP
#define MAPPER_POLICY_HPP

#include "BidPolicy.hpp"

// ════════════════════════════════════════════════════════════════════════════
// MAPPER [Liu et al. 2020] — decentralised per-agent actor-critic with
// evolutionary policy selection, adapted to L-GPDP (paper §4.3.3):
//
//   - Each agent owns its actor (12 → 64 → 64 → 1) and local critic.
//   - PPO replaces the original A2C (inheriting the MAPPO settings).
//   - Evolution every ev_period training rounds: agents are ranked by mean
//     cumulative reward R_i, range-normalised R̄_i = R_i / (R_max − R_min);
//     each non-best agent replaces its weights with an EXACT copy of the
//     best agent's with probability p_i = 1 − exp(η·R̄_i) / exp(η·R̄_best),
//     η = 2. No mutation noise (paper-faithful).
// ════════════════════════════════════════════════════════════════════════════
class MapperPolicy : public IBidPolicy {
public:
    PPOParams hparams;

    struct EvolutionParams {
        float eta            = 2.0f;
        int   period_rounds  = 5;
        int   fitness_window = 5;    // rolling episodes per agent
    };
    EvolutionParams ev_params;

    MapperPolicy() : seed_(std::random_device{}()) { rng_.seed(seed_); }

    const char* name() const override { return "MAPPER"; }

    float score(int agent_id, const PolicyFeatures& f) override;

    TrainingStats train_round() override;
    void on_round_end() override;            // evolution every period_rounds
    void set_progress(float p) override { hparams.set_progress(p); }
    void ensure_agents(int n_agents) override;
    void reinit(uint32_t seed) override;

    void save(const std::string& path) const override;
    bool load(const std::string& path) override;

    // Number of agents replaced at the last evolution step (logging).
    int last_evolution_replacements() const { return last_replacements_; }

private:
    struct AgentNets {
        Mlp     actor, critic;
        MlpAdam actor_opt, critic_opt;
        RunningMeanStd     vrms;
        std::vector<float> fitness_history;   // rolling per-episode mean reward
    };

    std::unordered_map<int, AgentNets> nets_;
    uint32_t seed_;
    int      rounds_seen_       = 0;
    int      last_replacements_ = 0;

    AgentNets& get_or_create(int agent_id);
    int  evolutionary_step();
};

#endif // MAPPER_POLICY_HPP
