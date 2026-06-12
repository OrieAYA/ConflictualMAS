#include "MapperPolicy.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>

MapperPolicy::AgentNets& MapperPolicy::get_or_create(int agent_id) {
    auto it = nets_.find(agent_id);
    if (it != nets_.end()) return it->second;
    AgentNets n;
    n.actor.init (kPolicySz, rng_, 0.01f, Mlp::Init::Orthogonal);
    n.critic.init(kPolicySz, rng_, 1.0f,  Mlp::Init::Orthogonal);
    n.actor_opt.resize_for(n.actor);
    n.critic_opt.resize_for(n.critic);
    auto [ins, ok] = nets_.emplace(agent_id, std::move(n));
    (void)ok;
    return ins->second;
}

void MapperPolicy::ensure_agents(int n_agents) {
    for (int i = 0; i < n_agents; ++i) (void)get_or_create(i);
}

void MapperPolicy::reinit(uint32_t seed) {
    seed_ = seed;
    rng_.seed(seed);
    nets_.clear();
    rounds_seen_       = 0;
    last_replacements_ = 0;
    clear_buffers();
}

float MapperPolicy::score(int agent_id, const PolicyFeatures& f) {
    AgentNets& n = get_or_create(agent_id);
    float x[kPolicySz];
    f.to_array(x);
    return Mlp::sigmoid(n.actor.forward(x));
}

TrainingStats MapperPolicy::train_round() {
    TrainingStats acc;
    int n_total = 0;

    const auto build = [](const Experience& e, float* dst) {
        std::copy_n(e.obs.data(), kPolicySz, dst);
    };

    for (auto& [aid, b] : buffers_) {
        if (b.empty()) continue;
        AgentNets& n = get_or_create(aid);

        // Per-agent fitness = mean reward per record this episode (rolling).
        float mean_r = 0.f;
        for (const auto& e : b) mean_r += e.reward;
        mean_r /= static_cast<float>(b.size());
        n.fitness_history.push_back(mean_r);
        while (static_cast<int>(n.fitness_history.size()) > ev_params.fitness_window)
            n.fitness_history.erase(n.fitness_history.begin());

        std::vector<std::vector<Experience>*> traj{ &b };
        TrainingStats ts = ppo_train(n.actor, n.actor_opt, &n.critic, &n.critic_opt,
                                     n.vrms, hparams, traj, build, rng_);

        const int w = ts.n_exp;
        acc.actor_loss  += ts.actor_loss  * w;
        acc.critic_loss += ts.critic_loss * w;
        acc.entropy     += ts.entropy     * w;
        acc.kl_approx   += ts.kl_approx   * w;
        acc.clip_frac   += ts.clip_frac   * w;
        acc.adv_std     += ts.adv_std     * w;
        acc.n_epochs    += ts.n_epochs;
        n_total         += w;
    }

    if (n_total > 0) {
        acc.actor_loss  /= n_total;
        acc.critic_loss /= n_total;
        acc.entropy     /= n_total;
        acc.kl_approx   /= n_total;
        acc.clip_frac   /= n_total;
        acc.adv_std     /= n_total;
    }
    acc.n_exp = n_total;

    clear_buffers();
    return acc;
}

void MapperPolicy::on_round_end() {
    ++rounds_seen_;
    if (ev_params.period_rounds > 0 &&
        rounds_seen_ % ev_params.period_rounds == 0)
        last_replacements_ = evolutionary_step();
}

// Paper-faithful evolutionary selection: rank by mean cumulative reward,
// range-normalise, replace agent i's weights with an EXACT copy of the best
// agent's with probability p_i = 1 − exp(η·R̄_i)/exp(η·R̄_best).
int MapperPolicy::evolutionary_step() {
    if (nets_.size() < 2) return 0;

    std::vector<std::pair<float, int>> stats;
    stats.reserve(nets_.size());
    for (const auto& [aid, n] : nets_) {
        if (n.fitness_history.empty()) continue;
        float m = 0.f;
        for (float r : n.fitness_history) m += r;
        stats.emplace_back(m / n.fitness_history.size(), aid);
    }
    if (stats.size() < 2) return 0;

    auto best_it = std::max_element(stats.begin(), stats.end());
    const int   best_aid = best_it->second;
    const float r_best   = best_it->first;

    float r_min =  std::numeric_limits<float>::infinity();
    float r_max = -std::numeric_limits<float>::infinity();
    for (const auto& [r, aid] : stats) {
        r_min = std::min(r_min, r);
        r_max = std::max(r_max, r);
    }
    const float denom = std::max(1e-8f, r_max - r_min);

    const AgentNets& parent = nets_.at(best_aid);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    int n_replaced = 0;

    for (const auto& [r, aid] : stats) {
        if (aid == best_aid) continue;
        const float p = 1.f - std::exp(
            ev_params.eta * (r / denom - r_best / denom));
        if (uni(rng_) < p) {
            AgentNets& child = nets_.at(aid);
            child.actor  = parent.actor;
            child.critic = parent.critic;
            child.actor_opt.resize_for(child.actor);    // fresh Adam moments
            child.critic_opt.resize_for(child.critic);
            child.vrms = parent.vrms;
            child.fitness_history.clear();
            ++n_replaced;
        }
    }
    return n_replaced;
}

// ── Persistence ──────────────────────────────────────────────────────────────
static constexpr uint32_t kMagicMapper = 0xDEA110D2u;

void MapperPolicy::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicMapper, sizeof(kMagicMapper), 1, f);
    const int32_t n = static_cast<int32_t>(nets_.size());
    std::fwrite(&n, sizeof(n), 1, f);
    for (const auto& [aid, nn] : nets_) {
        const int32_t id = aid;
        std::fwrite(&id, sizeof(id), 1, f);
        nn.actor.save_to(f);
        nn.critic.save_to(f);
        nn.vrms.save_to(f);
    }
    std::fclose(f);
}

bool MapperPolicy::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != kMagicMapper) {
        std::fclose(f);
        return false;
    }
    int32_t n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1 || n < 0) {
        std::fclose(f);
        return false;
    }
    nets_.clear();
    bool ok = true;
    for (int32_t i = 0; i < n && ok; ++i) {
        int32_t id = 0;
        ok = std::fread(&id, sizeof(id), 1, f) == 1;
        if (!ok) break;
        AgentNets& nn = get_or_create(id);
        ok = nn.actor.load_from(f) && nn.critic.load_from(f)
          && nn.vrms.load_from(f);
        nn.actor_opt.resize_for(nn.actor);
        nn.critic_opt.resize_for(nn.critic);
    }
    std::fclose(f);
    return ok;
}
