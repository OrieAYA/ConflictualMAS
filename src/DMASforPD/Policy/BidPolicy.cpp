#include "BidPolicy.hpp"
#include "DMASforPD/Policy/MAPPO.hpp"
#include "DMASforPD/Policy/IPPO.hpp"
#include "DMASforPD/Policy/MAPPERInspired.hpp"
#include "DMASforPD/Policy/Hybrid.hpp"
#include <algorithm>

// ── Shared experience plumbing ───────────────────────────────────────────────

int IBidPolicy::record(int agent_id, const PolicyFeatures& f,
                       const float* gs, float action) {
    constexpr float eps = 1e-8f;
    const float mu = std::clamp(score(agent_id, f), eps, 1.f - eps);

    Experience e;
    f.to_array(e.obs.data());
    if (gs) std::copy_n(gs, kGlobSz, e.gs.data());
    e.agent_id = agent_id;
    e.action   = action;
    e.log_prob = (action > 0.5f) ? std::log(mu) : std::log(1.f - mu);

    auto& b = buf(agent_id);
    b.push_back(e);
    const int idx = static_cast<int>(b.size()) - 1;
    recent_records_.push_back({agent_id, idx});
    return idx;
}

void IBidPolicy::add_to_reward(int agent_id, int buf_idx, float delta) {
    auto it = buffers_.find(agent_id);
    if (it == buffers_.end()) return;
    if (buf_idx < 0 || buf_idx >= static_cast<int>(it->second.size())) return;
    it->second[buf_idx].reward += delta;
}

int IBidPolicy::buffer_size(int agent_id) const {
    auto it = buffers_.find(agent_id);
    return it == buffers_.end() ? 0 : static_cast<int>(it->second.size());
}

int IBidPolicy::total_buffer_size() const {
    int n = 0;
    for (const auto& [aid, b] : buffers_) n += static_cast<int>(b.size());
    return n;
}

void IBidPolicy::clear_buffers() {
    for (auto& [aid, b] : buffers_) b.clear();
    recent_records_.clear();
}

std::vector<std::vector<Experience>*> IBidPolicy::trajectories() {
    std::vector<std::vector<Experience>*> trajs;
    trajs.reserve(buffers_.size());
    for (auto& [aid, b] : buffers_)
        if (!b.empty()) trajs.push_back(&b);
    return trajs;
}

// ── Registry ─────────────────────────────────────────────────────────────────

MappoPolicy&  mappo_policy()  { static MappoPolicy  p; return p; }
IppoPolicy&   ippo_policy()   { static IppoPolicy   p; return p; }
MapperPolicy& mapper_policy() { static MapperPolicy p; return p; }
HybridPolicy& hybrid_policy() { static HybridPolicy p; return p; }

IBidPolicy& bid_policy(BidPolicyKind kind) {
    switch (kind) {
        case BidPolicyKind::IPPO:   return ippo_policy();
        case BidPolicyKind::MAPPER: return mapper_policy();
        case BidPolicyKind::Hybrid: return hybrid_policy();
        case BidPolicyKind::MAPPO:
        default:                    return mappo_policy();
    }
}
