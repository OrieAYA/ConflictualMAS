#include "MappoPolicy.hpp"
#include <algorithm>
#include <cstdint>

float MappoPolicy::score(int /*agent_id*/, const PolicyFeatures& f) {
    float x[kPolicySz];
    f.to_array(x);
    return Mlp::sigmoid(actor_.forward(x));
}

void MappoPolicy::reinit(uint32_t seed) {
    rng_.seed(seed);
    actor_.init (kPolicySz, rng_, 0.01f, Mlp::Init::Orthogonal);
    critic_.init(kCriticIn, rng_, 1.0f,  Mlp::Init::Orthogonal);
    actor_opt_.resize_for(actor_);
    critic_opt_.resize_for(critic_);
    vrms_ = {};
    clear_buffers();
}

TrainingStats MappoPolicy::train_round() {
    auto trajs = trajectories();
    const auto build = [](const Experience& e, float* dst) {
        std::copy_n(e.gs.data(),  kGlobSz,   dst);
        std::copy_n(e.obs.data(), kPolicySz, dst + kGlobSz);
    };
    TrainingStats ts = ppo_train(actor_, actor_opt_, &critic_, &critic_opt_,
                                 vrms_, hparams, trajs, build, rng_);
    clear_buffers();
    return ts;
}

// ── Persistence ──────────────────────────────────────────────────────────────
static constexpr uint32_t kMagicMappo = 0xDEA110D0u;

void MappoPolicy::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicMappo, sizeof(kMagicMappo), 1, f);
    actor_.save_to(f);
    critic_.save_to(f);
    vrms_.save_to(f);
    std::fclose(f);
}

bool MappoPolicy::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != kMagicMappo) {
        std::fclose(f);
        return false;
    }
    bool ok = actor_.load_from(f) && critic_.load_from(f) && vrms_.load_from(f);
    std::fclose(f);
    return ok;
}
