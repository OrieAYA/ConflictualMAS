#include "DMASforPD/Policy/IPPO.hpp"
#include <algorithm>
#include <cstdint>

float IppoPolicy::score(int /*agent_id*/, const PolicyFeatures& f) {
    float x[kPolicySz];
    f.to_array(x);
    return Mlp::sigmoid(actor_.forward(x));
}

void IppoPolicy::reinit(uint32_t seed) {
    rng_.seed(seed);
    actor_.init (kPolicySz, rng_, 0.01f, Mlp::Init::HeTruncated);
    critic_.init(kPolicySz, rng_, 1.0f,  Mlp::Init::HeTruncated);
    actor_opt_.resize_for(actor_);
    critic_opt_.resize_for(critic_);
    vrms_ = {};
    clear_buffers();
}

TrainingStats IppoPolicy::train_round() {
    auto trajs = trajectories();
    const auto build = [](const Experience& e, float* dst) {
        std::copy_n(e.obs.data(), kPolicySz, dst);   // local observation only
    };
    TrainingStats ts = ppo_train(actor_, actor_opt_, &critic_, &critic_opt_,
                                 vrms_, hparams, trajs, build, rng_);
    clear_buffers();
    return ts;
}

// ── Persistence ──────────────────────────────────────────────────────────────
static constexpr uint32_t kMagicIppo = 0xDEA110D1u;

void IppoPolicy::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicIppo, sizeof(kMagicIppo), 1, f);
    actor_.save_to(f);
    critic_.save_to(f);
    vrms_.save_to(f);
    std::fclose(f);
}

bool IppoPolicy::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != kMagicIppo) {
        std::fclose(f);
        return false;
    }
    bool ok = actor_.load_from(f) && critic_.load_from(f) && vrms_.load_from(f);
    std::fclose(f);
    return ok;
}
