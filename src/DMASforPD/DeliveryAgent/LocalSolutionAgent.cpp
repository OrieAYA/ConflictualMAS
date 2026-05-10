#include "LocalSolutionAgent.hpp"
#include "OperableEnvironment.hpp"
#include <algorithm>
#include <limits>
#include <vector>

// ============================================================================
// Array-based DbVNS-PDP — no heap allocation during search
// ============================================================================

namespace {

// Flat VNS search state; copyable in O(N) without heap allocation per field.
struct State {
    std::vector<int>  seq;
    std::vector<bool> pending;   // indexed by env index; ready to visit
    std::vector<bool> forbidden; // indexed by env index; blocked in this branch
    int               pending_cnt = 0;
    float             cost = 0.f;
};

// Greedy NN extension of s until no pending node remains or stuck.
// Returns true iff all pending nodes were consumed.
static bool greedy_extend(State& s, const OperableEnvironment& env,
                           const std::vector<int>& pickup_of_idx)
{
    int n = env.size();
    while (s.pending_cnt > 0) {
        int   last   = s.seq.back();
        float best_c = std::numeric_limits<float>::max();
        int   best_j = -1;
        for (int j = 0; j < n; ++j) {
            if (!s.pending[j] || s.forbidden[j]) continue;
            float c = env.get_cost(last, j);
            if (c >= 0.f && c < 1e8f && c < best_c) { best_c = c; best_j = j; }
        }
        if (best_j < 0) return false;
        s.seq.push_back(best_j);
        s.cost += best_c;
        s.pending[best_j] = false; --s.pending_cnt;
        s.forbidden[best_j] = true;
        int pi = pickup_of_idx[best_j];
        if (pi >= 0 && !s.forbidden[pi] && !s.pending[pi]) {
            s.pending[pi] = true; ++s.pending_cnt;
        }
    }
    return true;
}

// Build initial state: anchor placed, all deliveries + anchor's pickup pending.
static State make_initial(int anchor_idx, int n,
                           const std::vector<int>&  pickup_of_idx,
                           const std::vector<bool>& is_delivery_idx)
{
    State s;
    s.seq.reserve(n);
    s.pending.assign(n, false);
    s.forbidden.assign(n, false);
    s.pending_cnt = 0;
    s.cost = 0.f;

    s.seq.push_back(anchor_idx);
    s.forbidden[anchor_idx] = true;

    int anc_pick = pickup_of_idx[anchor_idx]; // -1 if anchor has no pickup pair
    for (int i = 0; i < n; ++i) {
        if (i == anchor_idx) continue;
        if (is_delivery_idx[i] || i == anc_pick) {
            s.pending[i] = true; ++s.pending_cnt;
        }
    }
    return s;
}

// Reconstruct a prefix of length `depth` by replaying full_seq[0..depth-1]
// from scratch. O(depth × N) — depth is typically small.
static State rebuild_prefix(int anchor_idx, int n,
                              const std::vector<int>&  full_seq,
                              int                      depth,
                              const OperableEnvironment& env,
                              const std::vector<int>&  pickup_of_idx,
                              const std::vector<bool>& is_delivery_idx)
{
    State s = make_initial(anchor_idx, n, pickup_of_idx, is_delivery_idx);
    for (int step = 1; step < depth; ++step) {
        int   node = full_seq[step];
        float c    = env.get_cost(s.seq.back(), node);
        s.seq.push_back(node);
        if (c >= 0.f && c < 1e8f) s.cost += c;
        if (s.pending[node]) { s.pending[node] = false; --s.pending_cnt; }
        s.forbidden[node] = true;
        int pi = pickup_of_idx[node];
        if (pi >= 0 && !s.forbidden[pi] && !s.pending[pi]) {
            s.pending[pi] = true; ++s.pending_cnt;
        }
    }
    return s;
}

// Main DbVNS-PDP search for one anchor.
static std::vector<ObjectiveNode> db_vns_search(
    const ObjectiveNode&       starting_node,
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const DbVNSParams&         params)
{
    if (env.nodes.empty()) return {};
    int n = env.size();

    std::vector<int>  pickup_of_idx(n, -1);
    std::vector<bool> is_delivery_idx(n, false);
    for (int i = 0; i < n; ++i) {
        auto it = pickup_of.find(env.nodes[i].id);
        if (it != pickup_of.end()) {
            is_delivery_idx[i] = true;
            int pi = env.find_index(it->second);
            if (pi >= 0) pickup_of_idx[i] = pi;
        }
    }

    int anchor_idx = env.find_index(starting_node.id);
    if (anchor_idx < 0) return {};

    // Initial greedy solution.
    State cur = make_initial(anchor_idx, n, pickup_of_idx, is_delivery_idx);
    if (!greedy_extend(cur, env, pickup_of_idx)) return {};

    std::vector<int> best_seq  = cur.seq;
    float            best_cost = cur.cost;

    const int k_max       = std::max(1, params.k_max);
    const int max_decomps = std::max(1, params.max_decompositions);
    int k = 1;

    for (int iter = 0; iter < params.max_iterations; ++iter) {
        int seq_sz = (int)best_seq.size();

        // VNS shake schedule: for k=1 peel 1 node, for k=k_max peel ~N/k_max nodes.
        int peel   = std::max(1, seq_sz / k_max * k);
        int depth  = std::max(1, seq_sz - peel);

        // Rebuild prefix once; copy it for each decomposition trial.
        State prefix = rebuild_prefix(anchor_idx, n, best_seq, depth,
                                      env, pickup_of_idx, is_delivery_idx);

        bool improved = false;
        for (int d = 0; d < max_decomps; ++d) {
            State trial = prefix; // O(N) copy — no heap alloc for bool vecs, only int vec
            if (greedy_extend(trial, env, pickup_of_idx) && trial.cost < best_cost) {
                best_cost = trial.cost;
                best_seq  = trial.seq;
                improved  = true;
                break;
            }
            // Forbid the first choice of this completion so next trial diverges.
            int plen = (int)prefix.seq.size();
            if ((int)trial.seq.size() > plen)
                prefix.forbidden[trial.seq[plen]] = true;
            else
                break; // stuck; no point trying further decomps
        }

        if (improved) k = 1;
        else if (k < k_max) ++k;
    }

    // Reverse build order → forward visit order (anchor ends up last).
    std::vector<ObjectiveNode> result;
    result.reserve(best_seq.size());
    for (int i = (int)best_seq.size() - 1; i >= 0; --i)
        result.push_back(env.nodes[best_seq[i]]);
    return result;
}

} // namespace

// ============================================================================
// LocalSolutionAgent
// ============================================================================

LocalSolutionAgent::LocalSolutionAgent(const ObjectiveNode& node, const DbVNSParams& p)
    : starting_node(node), params(p) {}

std::vector<ObjectiveNode> LocalSolutionAgent::plan(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const PairingMap&          /*delivery_of*/,
    osmium::object_id_type     /*agent_current*/) const
{
    return db_vns_search(starting_node, env, pickup_of, params);
}
