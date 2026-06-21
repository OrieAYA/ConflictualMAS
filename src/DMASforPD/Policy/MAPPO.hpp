#ifndef MAPPO_POLICY_HPP
#define MAPPO_POLICY_HPP

#include "BidPolicy.hpp"

// ════════════════════════════════════════════════════════════════════════════
// MAPPO [Yu et al. 2022] — shared actor + centralised critic (CTDE).
//
// Actor : kPolicySz → 64 → 64 → 1 (sigmoid), shared by every agent.
// Critic: kCriticIn = 32 (Feature-Pruned global state: 20-d system state ++
//         the 12-d input vector of the deciding agent), linear head.
// PPO   : clip 0.1, 10 epochs, full batch, lr 5e-4 → 5e-5, grad clip 10
//         (paper Table 5, MAPPO column). Orthogonal init.
// ════════════════════════════════════════════════════════════════════════════
class MappoPolicy : public IBidPolicy {
public:
    PPOParams hparams;

    MappoPolicy() { reinit(std::random_device{}()); }

    const char* name() const override { return "MAPPO"; }

    float score(int agent_id, const PolicyFeatures& f) override;

    TrainingStats train_round() override;
    void set_progress(float p) override { hparams.set_progress(p); }
    void reinit(uint32_t seed) override;

    void save(const std::string& path) const override;
    bool load(const std::string& path) override;

    // Hybrid hot-start: read-only access to the trained actor.
    const Mlp& actor() const { return actor_; }

private:
    Mlp     actor_, critic_;
    MlpAdam actor_opt_, critic_opt_;
    RunningMeanStd vrms_;
};

#endif // MAPPO_POLICY_HPP
