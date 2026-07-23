#ifndef BID_POLICY_HPP
#define BID_POLICY_HPP

#include "PolicyKit.hpp"
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// IBidPolicy — the uniform contract every objective (bid) policy implements.
//
// The delivery agent calls score()/record() during a TAM session; the episode
// runner shapes rewards through (agent_id, buf_idx) handles and triggers
// train_round() at episode end. The four learning variants of the paper
// (MAPPO, IPPO, MAPPER, Hybrid) differ only in network topology, critic input
// and update rule — everything else (buffers, recent-records bracketing,
// reward plumbing) is shared here.
// ════════════════════════════════════════════════════════════════════════════

enum class BidPolicyKind { MAPPO, IPPO, MAPPER, Hybrid };

class IBidPolicy {
public:
    virtual ~IBidPolicy() = default;

    virtual const char* name() const = 0;

    // ── Inference: action score μ ∈ [0,1] ───────────────────────────────────
    virtual float score(int agent_id, const PolicyFeatures& f) = 0;

    // ── Experience: one bid decision ────────────────────────────────────────
    // gs = system-state snapshot (kGlobSz floats; consumed by the MAPPO
    // critic, ignored by the local-critic policies). Returns the buffer
    // index inside this agent's buffer.
    int record(int agent_id, const PolicyFeatures& f,
               const float* gs, float action);

    void add_to_reward(int agent_id, int buf_idx, float delta);

    int buffer_size(int agent_id) const;
    int total_buffer_size() const;
    void clear_buffers();

    // μ recorded at decision time for one buffer entry, recovered from
    // log π(a|s): a = 1 → exp(lp), a = 0 → 1 − exp(lp). −1 = invalid handle.
    // Used by the §5 idle-penalty μ band (reward shaping v2).
    float entry_mu(int agent_id, int buf_idx) const;

    // Largest |reward| across all buffered entries (sanity-check probe for
    // the reward-shaping bound test).
    float max_abs_reward() const;

    // Discount factor of the underlying PPO learner (potential-based shaping
    // §10 must use the SAME γ as the GAE). All paper variants keep the
    // PPOParams default.
    virtual float ppo_gamma() const { return PPOParams{}.gamma; }

    // ── Recent-records log ──────────────────────────────────────────────────
    // Every record() appends (agent_id, buf_idx). The runner brackets a TAM
    // session with n_recent_records() before/after to pair the TAM's i-th
    // scored candidate with the buffer entry it produced.
    int                n_recent_records() const { return static_cast<int>(recent_records_.size()); }
    std::pair<int,int> recent_record(int i) const { return recent_records_[i]; }
    void               clear_recent_records()     { recent_records_.clear(); }

    // ── Training ────────────────────────────────────────────────────────────
    // One update over the buffered episode; clears the buffers afterwards.
    virtual TrainingStats train_round() = 0;

    // True for policies that adapt at the end of EVERY episode regardless of
    // the trainer's train/eval phase (Hybrid's online residual).
    virtual bool trains_online() const { return false; }

    // Called once per training round (MAPPER runs its evolutionary
    // replacement every ev_period rounds).
    virtual void on_round_end() {}

    virtual void set_progress(float /*progress*/) {}
    virtual void ensure_agents(int /*n_agents*/) {}

    // Reset per-episode online state at each episode start (offline policies:
    // no-op — their weights are frozen in eval / trained across episodes).
    // Hybrid zeroes its per-agent residuals so every eval episode restarts
    // from the frozen MAPPO base (état 0), independent of scenario order.
    virtual void reset_episode() {}

    // Re-create the networks from scratch (fresh seed). Used by the trainer
    // at the start of each independent training seed.
    virtual void reinit(uint32_t seed) = 0;

    virtual void save(const std::string& path) const = 0;
    virtual bool load(const std::string& path) = 0;

protected:
    std::unordered_map<int, std::vector<Experience>> buffers_;
    std::vector<std::pair<int,int>>                  recent_records_;
    std::mt19937                                     rng_{std::random_device{}()};

    std::vector<Experience>& buf(int agent_id) { return buffers_[agent_id]; }

    // Collect non-empty per-agent buffers as trajectory pointers for ppo_train.
    std::vector<std::vector<Experience>*> trajectories();
};

// ── Registry (one singleton per kind) ────────────────────────────────────────
IBidPolicy& bid_policy(BidPolicyKind kind);

// Concrete accessors for trainer-side needs (checkpoint paths, Hybrid base).
class MappoPolicy;  MappoPolicy&  mappo_policy();
class IppoPolicy;   IppoPolicy&   ippo_policy();
class MapperPolicy; MapperPolicy& mapper_policy();
class HybridPolicy; HybridPolicy& hybrid_policy();

#endif // BID_POLICY_HPP
