#ifndef IPPO_POLICY_HPP
#define IPPO_POLICY_HPP

#include "BidPolicy.hpp"

// ════════════════════════════════════════════════════════════════════════════
// IPPO [de Witt et al. 2020] — independent PPO.
//
// Shared actor + shared critic, but the critic conditions on the agent's
// LOCAL observation only (kPolicySz input — no global state). Paper-faithful
// overrides (Table 5, IPPO column): clip 0.2, 4 epochs, lr 1e-4, entropy
// 5e-3, gradient norm clip 0.5, He (variance-scaling) initialisation.
// ════════════════════════════════════════════════════════════════════════════
class IppoPolicy : public IBidPolicy {
public:
    PPOParams hparams;

    IppoPolicy() {
        hparams.clip_eps       = 0.2f;
        hparams.epochs         = 4;
        hparams.max_grad_norm  = 0.5f;
        hparams.lr_actor       = 1e-4f;
        hparams.lr_actor_init  = 1e-4f;
        hparams.lr_actor_min   = 1e-5f;
        hparams.lr_critic      = 1e-4f;
        hparams.lr_critic_init = 1e-4f;
        hparams.lr_critic_min  = 1e-5f;
        hparams.ent_w          = 5e-3f;
        hparams.ent_w_init     = 5e-3f;
        hparams.ent_w_min      = 5e-4f;
        reinit(std::random_device{}());
    }

    const char* name() const override { return "IPPO"; }

    float score(int agent_id, const PolicyFeatures& f) override;

    TrainingStats train_round() override;
    void set_progress(float p) override { hparams.set_progress(p); }
    void reinit(uint32_t seed) override;

    void save(const std::string& path) const override;
    bool load(const std::string& path) override;

private:
    Mlp     actor_, critic_;     // critic input = kPolicySz (local obs)
    MlpAdam actor_opt_, critic_opt_;
    RunningMeanStd vrms_;
};

#endif // IPPO_POLICY_HPP
