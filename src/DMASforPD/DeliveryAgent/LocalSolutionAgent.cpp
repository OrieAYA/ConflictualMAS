#include "LocalSolutionAgent.hpp"
#include "OperableEnvironment.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
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

// ── Forward DbVNS-PDP for lifelong replanning (tree-based, GPDP-adapted) ─────
//
// Adapts the legacy Agent::agent_search() / Agent::decompose_with_forbidden()
// / Agent::greedy_construction() trio (Legacy/Agent/Agent.cpp) to the GPDP
// mono-agent setting, following the design discussed in the project notes:
//
//   - All tasks given to the agent are MANDATORY (no acceptance/refusal):
//     the "forbidden" set semantically means POSTPONE (skip at this branching
//     point), not eliminate. The decomposition tree therefore explores
//     different ORDERS rather than different subsets.
//
//   - The greedy construction respects two GPDP constraints at every step:
//       1) pairing : a delivery is available only after its pickup is visited
//       2) capacity: a pickup is selectable only if current carry < max_capacity
//     A node infeasible at the current state is simply filtered from the
//     candidate set (txt §2 & §3): "le reste perçu comme simple nœud".
//
//   - The objective is min(total travel cost). The upper-bound (reward
//     density) of the orienteering DbVNS becomes a LOWER-bound on completion
//     cost, used as the pruning criterion (current_cost + (rest-1)*min_edge).
//
//   - The DecomposedSolution tree is persistent within a single plan call:
//     children created during greedy descents are kept so that subsequent
//     iterations can re-enter cached subtrees (and so the forbidden chain
//     walked up by decompose_with_forbidden survives across iterations).

struct FwdDecomp {
    FwdDecomp*                       parent = nullptr;
    std::unordered_map<int, FwdDecomp*> childs;   // keyed by env-idx of the edge
    std::vector<int>                 seq;         // path from root to this node
    std::vector<bool>                avail;       // currently visitable
    std::vector<bool>                forbidden;   // accumulated postpone set
    int                              avail_cnt = 0;
    int                              load = 0;    // carried packages at this point
    float                            cost = 0.f;
};

// Per-search read-only context.
struct FwdCtx {
    int                              n             = 0;
    const OperableEnvironment*       env           = nullptr;
    const std::vector<int>*          delivery_of   = nullptr; // pickup-idx → delivery-idx (or -1)
    const std::vector<int>*          pickup_of     = nullptr; // delivery-idx → pickup-idx (or -1)
    const std::vector<float>*        start_cost    = nullptr;
    int                              max_capacity  = 3;
    float                            min_edge_cost = 0.f;
};

// Pick the cheapest feasible next node from state `s`.
// Feasibility = available && !forbidden && (load < cap if pickup).
static int fwd_best_neighbor(const FwdDecomp& s, const FwdCtx& ctx) {
    int   best_j = -1;
    float best_c = std::numeric_limits<float>::max();
    for (int j = 0; j < ctx.n; ++j) {
        if (!s.avail[j] || s.forbidden[j]) continue;
        const bool is_pickup = (*ctx.delivery_of)[j] >= 0;
        if (is_pickup && s.load >= ctx.max_capacity) continue;  // skip → postpone
        const float c = s.seq.empty()
            ? (*ctx.start_cost)[j]
            : ctx.env->get_cost(s.seq.back(), j);
        if (c >= 0.f && c < 1e8f && c < best_c) { best_c = c; best_j = j; }
    }
    return best_j;
}

// Greedy extension: recursive nearest-feasible descent. Reuses existing
// childs in the tree when present; creates new ones otherwise. Mirrors
// Agent::greedy_construction() exactly.
static FwdDecomp* fwd_greedy(FwdDecomp* current, const FwdCtx& ctx) {
    if (current->avail_cnt == 0) return current;

    const int j = fwd_best_neighbor(*current, ctx);
    if (j < 0) return current;

    auto it = current->childs.find(j);
    if (it != current->childs.end()) return fwd_greedy(it->second, ctx);

    const float c = current->seq.empty()
        ? (*ctx.start_cost)[j]
        : ctx.env->get_cost(current->seq.back(), j);

    FwdDecomp* child = new FwdDecomp;
    child->parent    = current;
    child->seq       = current->seq;       child->seq.push_back(j);
    child->avail     = current->avail;     child->avail[j] = false;
    child->forbidden = current->forbidden; child->forbidden[j] = true;
    child->avail_cnt = current->avail_cnt - 1;
    child->cost      = current->cost + c;

    // Pickup → unlock its delivery and increment carry.
    // Delivery → decrement carry.
    const int di = (*ctx.delivery_of)[j];
    if (di >= 0) {
        if (!child->forbidden[di] && !child->avail[di]) {
            child->avail[di] = true;
            ++child->avail_cnt;
        }
        child->load = current->load + 1;
    } else {
        const int pi = (*ctx.pickup_of)[j];
        child->load = current->load + (pi >= 0 ? -1 : 0);
    }
    current->childs[j] = child;
    return fwd_greedy(child, ctx);
}

// Walk up from `leaf` to the root, adding each leaf's last step to the
// parent's forbidden set. Returns the chain of parents (nearest → root).
// Mirrors Agent::decompose_with_forbidden().
static std::vector<FwdDecomp*> fwd_decompose(FwdDecomp* leaf) {
    std::vector<FwdDecomp*> chain;
    FwdDecomp* current = leaf;
    while (current->parent != nullptr) {
        const int last = current->seq.back();
        current = current->parent;
        current->forbidden[last] = true;  // postpone that choice from here
        chain.push_back(current);
    }
    return chain;
}

// Lower bound on completion cost from `s`. Counts the number of nodes that
// must still be visited (available + locked deliveries whose pickup is still
// reachable) and adds (count-1) × min_edge_cost. Weak but never overestimates.
static float fwd_lower_bound(const FwdDecomp& s, const FwdCtx& ctx) {
    int remaining = 0;
    for (int j = 0; j < ctx.n; ++j) {
        if (s.forbidden[j]) continue;
        if (s.avail[j]) { ++remaining; continue; }
        // Locked delivery whose pickup is still alive must still be served.
        const int pi = (*ctx.pickup_of)[j];
        if (pi >= 0 && !s.forbidden[pi]) ++remaining;
    }
    return s.cost + std::max(0, remaining - 1) * ctx.min_edge_cost;
}

static void fwd_delete_tree(FwdDecomp* root) {
    if (!root) return;
    for (auto& kv : root->childs) fwd_delete_tree(kv.second);
    delete root;
}

static std::vector<ObjectiveNode> forward_dbvns(
    const OperableEnvironment& env,
    const std::vector<int>&    delivery_of_idx,
    const std::vector<int>&    pickup_of_idx,
    const std::vector<bool>&   initially_avail,
    const std::vector<float>&  start_cost,
    int                        max_capacity,
    int                        initial_load,
    const DbVNSParams&         params)
{
    const int n = env.size();
    if (n == 0) return {};

    // Min positive edge cost over the matrix — used by the lower bound.
    float min_edge = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const float c = env.get_cost(i, j);
            if (c > 0.f && c < 1e8f && c < min_edge) min_edge = c;
        }
    if (!std::isfinite(min_edge) || min_edge >= 1e8f) min_edge = 0.f;

    FwdCtx ctx{ n, &env, &delivery_of_idx, &pickup_of_idx, &start_cost,
                std::max(1, max_capacity), min_edge };

    // Root of the decomposition tree.
    FwdDecomp* root = new FwdDecomp;
    root->avail = initially_avail;
    root->forbidden.assign(n, false);
    for (bool b : initially_avail) if (b) ++root->avail_cnt;
    root->load = std::max(0, initial_load);
    root->cost = 0.f;

    // Initial greedy → pbest.
    FwdDecomp* pbest = fwd_greedy(root, ctx);
    if (pbest->seq.empty()) { fwd_delete_tree(root); return {}; }
    float pbest_cost = pbest->cost;

    const int   k_max               = std::max(1, params.k_max);
    const int   max_decompositions  = std::max(1, params.max_decompositions);
    const float divergence_factor   = 1.f + static_cast<float>(params.max_divergence);
    int         k = 1;
    int         consecutive_empty = 0;
    const int   max_consecutive_empty = 3;

    for (int iter = 0; iter < params.max_iterations; ++iter) {
        // A. SHAKE — climb pbest up to `until` parents (legacy formula).
        const int sz = static_cast<int>(pbest->seq.size());
        const int until = std::min(sz / k_max * k,
                                    std::max(0, sz - 2));
        FwdDecomp* shaken = pbest;
        for (int i = 0; i < until && shaken->parent; ++i) shaken = shaken->parent;

        // B. DECOMPOSE WITH FORBIDDEN — chain of parents with postpone-sets.
        std::vector<FwdDecomp*> chain = fwd_decompose(shaken);
        if (chain.empty()) {
            ++consecutive_empty;
            if (k < k_max) ++k;
            if (consecutive_empty >= max_consecutive_empty) break;
            continue;
        }

        // C. PRUNE — keep only branches whose lower bound is still promising
        //    (LB < pbest_cost × (1 + max_divergence)).
        const float prune_threshold = pbest_cost * divergence_factor;
        std::vector<FwdDecomp*> promising;
        promising.reserve(chain.size());
        for (FwdDecomp* node : chain) {
            if (fwd_lower_bound(*node, ctx) < prune_threshold)
                promising.push_back(node);
        }
        if (promising.empty()) {
            ++consecutive_empty;
            if (k < k_max) ++k;
            if (consecutive_empty >= max_consecutive_empty) break;
            continue;
        }
        if (static_cast<int>(promising.size()) > max_decompositions) {
            std::sort(promising.begin(), promising.end(),
                [&ctx](FwdDecomp* a, FwdDecomp* b) {
                    return fwd_lower_bound(*a, ctx) < fwd_lower_bound(*b, ctx);
                });
            promising.resize(max_decompositions);
        }
        consecutive_empty = 0;

        // D. EXPLORE — one greedy descent per promising decomposition.
        FwdDecomp* best_in_iter = nullptr;
        for (FwdDecomp* node : promising) {
            FwdDecomp* leaf = fwd_greedy(node, ctx);
            if (leaf->seq.size() <= node->seq.size()) continue;  // stuck
            if (leaf->avail_cnt > 0)                  continue;  // infeasible completion
            if (!best_in_iter || leaf->cost < best_in_iter->cost)
                best_in_iter = leaf;
        }

        // E. ACCEPT — VNS rule: improvement resets k, else escalate.
        if (best_in_iter && best_in_iter->cost < pbest_cost) {
            pbest      = best_in_iter;
            pbest_cost = best_in_iter->cost;
            k = 1;
        } else if (k < k_max) {
            ++k;
        }
    }

    std::vector<ObjectiveNode> result;
    result.reserve(pbest->seq.size());
    for (int idx : pbest->seq) result.push_back(env.nodes[idx]);

    fwd_delete_tree(root);
    return result;
}

// ── ALNS-PDP for mono-agent lifelong GPDP ────────────────────────────────────
//
// Ropke & Pisinger (2006) adapted to mono-agent without time windows.
//   - 3 destroy operators (random, worst, Shaw)
//   - 2 repair operators (cheapest, regret-2 over insertion positions)
//   - Simulated-annealing acceptance with cooling
//   - Adaptive roulette-wheel selection with reaction factor (paper §3.4)
//   - Noise on insertion cost for diversification
//
// "Regret-k" in the original paper compares insertion costs across multiple
// VEHICLES. In mono-agent there is only one route, so regret here measures
// the cost gap between the best and second-best (pos_P, pos_D) for a request.

struct AlnsRequest {
    int p_idx = -1;  // env-idx of pickup, -1 for singleton (carried delivery)
    int d_idx = -1;  // env-idx of delivery
};

struct AlnsCtx {
    int                              n             = 0;
    const OperableEnvironment*       env           = nullptr;
    const std::vector<float>*        start_cost    = nullptr;
    const std::vector<int>*          delivery_of   = nullptr;
    const std::vector<int>*          pickup_of     = nullptr;
    int                              max_capacity  = 3;
    int                              initial_load  = 0;
    std::vector<AlnsRequest>         requests;
    float                            max_edge_cost = 0.f;
};

static float alns_cost(const std::vector<int>& seq, const AlnsCtx& ctx) {
    if (seq.empty()) return 0.f;
    float c = (*ctx.start_cost)[seq[0]];
    for (size_t i = 1; i < seq.size(); ++i) {
        const float e = ctx.env->get_cost(seq[i-1], seq[i]);
        if (e < 0.f || e >= 1e8f) return std::numeric_limits<float>::infinity();
        c += e;
    }
    return c;
}

static bool alns_feasible(const std::vector<int>& seq, const AlnsCtx& ctx) {
    int load = ctx.initial_load;
    std::vector<char> pickup_seen(ctx.n, 0);
    for (int idx : seq) {
        const int di = (*ctx.delivery_of)[idx];
        const int pi = (*ctx.pickup_of)[idx];
        if (di >= 0) {
            ++load;
            if (load > ctx.max_capacity) return false;
            pickup_seen[idx] = 1;
        } else if (pi >= 0) {
            if (!pickup_seen[pi]) return false;
            --load;
            if (load < 0) return false;
        } else {
            --load;
            if (load < 0) return false;
        }
    }
    return true;
}

// Best feasible insertion of request r into seq. Writes positions through
// out_pP / out_pD (out_pP = -1 for singletons). Returns +inf if infeasible.
static float alns_best_insertion(const std::vector<int>& seq,
                                  const AlnsRequest& r,
                                  const AlnsCtx& ctx,
                                  int& out_pP, int& out_pD,
                                  float* second_best = nullptr)
{
    out_pP = out_pD = -1;
    const int n = (int)seq.size();
    float c1 = std::numeric_limits<float>::infinity();
    float c2 = std::numeric_limits<float>::infinity();

    auto consider = [&](float c, int pP, int pD) {
        if (c < c1) { c2 = c1; c1 = c; out_pP = pP; out_pD = pD; }
        else if (c < c2) { c2 = c; }
    };

    std::vector<int> trial;
    trial.reserve(n + 2);

    if (r.p_idx < 0) {
        for (int pos = 0; pos <= n; ++pos) {
            trial.assign(seq.begin(), seq.end());
            trial.insert(trial.begin() + pos, r.d_idx);
            if (!alns_feasible(trial, ctx)) continue;
            const float c = alns_cost(trial, ctx);
            if (std::isfinite(c)) consider(c, -1, pos);
        }
    } else {
        for (int pP = 0; pP <= n; ++pP) {
            for (int pD = pP; pD <= n; ++pD) {
                trial.clear();
                for (int j = 0; j < pP; ++j) trial.push_back(seq[j]);
                trial.push_back(r.p_idx);
                for (int j = pP; j < pD; ++j) trial.push_back(seq[j]);
                trial.push_back(r.d_idx);
                for (int j = pD; j < n; ++j) trial.push_back(seq[j]);
                if (!alns_feasible(trial, ctx)) continue;
                const float c = alns_cost(trial, ctx);
                if (std::isfinite(c)) consider(c, pP, pD);
            }
        }
    }
    if (second_best) *second_best = c2;
    return c1;
}

static void alns_apply_insertion(std::vector<int>& seq,
                                  const AlnsRequest& r, int pP, int pD)
{
    if (r.p_idx < 0) { seq.insert(seq.begin() + pD, r.d_idx); return; }
    // Insert delivery first so the pickup index doesn't shift.
    seq.insert(seq.begin() + pD, r.d_idx);
    seq.insert(seq.begin() + pP, r.p_idx);
}

// Repair operator: greedy cheapest insertion.
static bool alns_repair_cheapest(std::vector<int>& seq,
                                  std::vector<AlnsRequest>& bank,
                                  const AlnsCtx& ctx,
                                  float noise_amplitude,
                                  std::mt19937& rng)
{
    std::uniform_real_distribution<float> noise(-noise_amplitude, noise_amplitude);
    while (!bank.empty()) {
        int   best_req = -1;
        int   best_pP = -1, best_pD = -1;
        float best_c  = std::numeric_limits<float>::infinity();
        for (int i = 0; i < (int)bank.size(); ++i) {
            int pP, pD;
            float c = alns_best_insertion(seq, bank[i], ctx, pP, pD);
            if (!std::isfinite(c)) continue;
            if (noise_amplitude > 0.f) c += noise(rng);
            if (c < best_c) { best_c = c; best_req = i; best_pP = pP; best_pD = pD; }
        }
        if (best_req < 0) return false;
        alns_apply_insertion(seq, bank[best_req], best_pP, best_pD);
        bank.erase(bank.begin() + best_req);
    }
    return true;
}

// Repair operator: regret-2 insertion (positional regret in mono-agent).
static bool alns_repair_regret2(std::vector<int>& seq,
                                 std::vector<AlnsRequest>& bank,
                                 const AlnsCtx& ctx,
                                 float noise_amplitude,
                                 std::mt19937& rng)
{
    std::uniform_real_distribution<float> noise(-noise_amplitude, noise_amplitude);
    while (!bank.empty()) {
        int    best_req = -1;
        int    best_pP = -1, best_pD = -1;
        float  best_c    = std::numeric_limits<float>::infinity();
        float  max_regret = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < (int)bank.size(); ++i) {
            int pP, pD;
            float c2 = std::numeric_limits<float>::infinity();
            float c1 = alns_best_insertion(seq, bank[i], ctx, pP, pD, &c2);
            if (!std::isfinite(c1)) continue;
            float regret = std::isfinite(c2) ? (c2 - c1) : 1e6f;
            if (noise_amplitude > 0.f) regret += noise(rng);
            if (regret > max_regret) {
                max_regret = regret;
                best_req = i; best_pP = pP; best_pD = pD; best_c = c1;
            }
        }
        if (best_req < 0 || !std::isfinite(best_c)) return false;
        alns_apply_insertion(seq, bank[best_req], best_pP, best_pD);
        bank.erase(bank.begin() + best_req);
    }
    return true;
}

// Destroy operator: random removal of q requests.
static void alns_destroy_random(std::vector<int>& seq,
                                 const std::vector<AlnsRequest>& all_reqs,
                                 int q,
                                 std::vector<AlnsRequest>& bank,
                                 std::mt19937& rng)
{
    std::unordered_set<int> in_seq(seq.begin(), seq.end());
    std::vector<int> active;
    for (int i = 0; i < (int)all_reqs.size(); ++i)
        if (in_seq.count(all_reqs[i].d_idx)) active.push_back(i);
    std::shuffle(active.begin(), active.end(), rng);
    if (q > (int)active.size()) q = (int)active.size();
    std::unordered_set<int> rem_nodes;
    for (int i = 0; i < q; ++i) {
        const AlnsRequest& r = all_reqs[active[i]];
        if (r.p_idx >= 0) rem_nodes.insert(r.p_idx);
        rem_nodes.insert(r.d_idx);
        bank.push_back(r);
    }
    std::vector<int> kept;
    kept.reserve(seq.size());
    for (int idx : seq) if (!rem_nodes.count(idx)) kept.push_back(idx);
    seq = std::move(kept);
}

// Destroy operator: worst removal, randomised by p_worst.
static void alns_destroy_worst(std::vector<int>& seq,
                                const std::vector<AlnsRequest>& all_reqs,
                                int q,
                                float p_worst,
                                std::vector<AlnsRequest>& bank,
                                const AlnsCtx& ctx,
                                std::mt19937& rng)
{
    std::uniform_real_distribution<float> uy(0.f, 1.f);
    for (int it = 0; it < q; ++it) {
        std::unordered_set<int> in_seq(seq.begin(), seq.end());
        std::vector<std::pair<int, float>> scored;
        const float base = alns_cost(seq, ctx);
        for (int ri = 0; ri < (int)all_reqs.size(); ++ri) {
            const AlnsRequest& r = all_reqs[ri];
            if (!in_seq.count(r.d_idx)) continue;
            std::vector<int> trial;
            trial.reserve(seq.size());
            for (int idx : seq) {
                if (idx == r.d_idx) continue;
                if (r.p_idx >= 0 && idx == r.p_idx) continue;
                trial.push_back(idx);
            }
            const float c = alns_cost(trial, ctx);
            scored.emplace_back(ri, std::isfinite(c) ? (base - c) : 0.f);
        }
        if (scored.empty()) return;
        std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b){ return a.second > b.second; });
        const float y = uy(rng);
        int pick = static_cast<int>(std::pow(y, p_worst) * scored.size());
        if (pick >= (int)scored.size()) pick = (int)scored.size() - 1;
        const AlnsRequest& r = all_reqs[scored[pick].first];
        bank.push_back(r);
        std::vector<int> kept;
        kept.reserve(seq.size());
        for (int idx : seq) {
            if (idx == r.d_idx) continue;
            if (r.p_idx >= 0 && idx == r.p_idx) continue;
            kept.push_back(idx);
        }
        seq = std::move(kept);
    }
}

// Destroy operator: Shaw relatedness removal.
static void alns_destroy_shaw(std::vector<int>& seq,
                               const std::vector<AlnsRequest>& all_reqs,
                               int q,
                               float p_shaw,
                               std::vector<AlnsRequest>& bank,
                               const AlnsCtx& ctx,
                               std::mt19937& rng)
{
    std::unordered_set<int> in_seq(seq.begin(), seq.end());
    std::vector<int> active;
    for (int i = 0; i < (int)all_reqs.size(); ++i)
        if (in_seq.count(all_reqs[i].d_idx)) active.push_back(i);
    if (active.empty()) return;

    std::uniform_real_distribution<float> uy(0.f, 1.f);
    std::vector<int> removed;
    {
        int idx0 = std::uniform_int_distribution<int>(0, (int)active.size() - 1)(rng);
        removed.push_back(active[idx0]);
        bank.push_back(all_reqs[active[idx0]]);
        active.erase(active.begin() + idx0);
    }

    auto relatedness = [&](const AlnsRequest& a, const AlnsRequest& b) {
        float c = ctx.env->get_cost(a.d_idx, b.d_idx);
        if (c < 0.f || c >= 1e8f) c = ctx.max_edge_cost;
        return c;
    };

    while ((int)removed.size() < q && !active.empty()) {
        int seed_pos = std::uniform_int_distribution<int>(
            0, (int)removed.size() - 1)(rng);
        const AlnsRequest& seed = all_reqs[removed[seed_pos]];
        std::sort(active.begin(), active.end(),
            [&](int a, int b){
                return relatedness(all_reqs[a], seed)
                     < relatedness(all_reqs[b], seed);
            });
        const float y = uy(rng);
        int pick = static_cast<int>(std::pow(y, p_shaw) * active.size());
        if (pick >= (int)active.size()) pick = (int)active.size() - 1;
        removed.push_back(active[pick]);
        bank.push_back(all_reqs[active[pick]]);
        active.erase(active.begin() + pick);
    }

    std::unordered_set<int> rem_nodes;
    for (int ri : removed) {
        const AlnsRequest& r = all_reqs[ri];
        if (r.p_idx >= 0) rem_nodes.insert(r.p_idx);
        rem_nodes.insert(r.d_idx);
    }
    std::vector<int> kept;
    kept.reserve(seq.size());
    for (int idx : seq) if (!rem_nodes.count(idx)) kept.push_back(idx);
    seq = std::move(kept);
}

// Main ALNS loop.
static std::vector<ObjectiveNode> forward_alns(
    const OperableEnvironment& env,
    const std::vector<int>&    delivery_of_idx,
    const std::vector<int>&    pickup_of_idx,
    const std::vector<float>&  start_cost,
    int                        max_capacity,
    int                        initial_load,
    const ALNSParams&          params)
{
    const int n = env.size();
    if (n == 0) return {};

    AlnsCtx ctx;
    ctx.n = n;
    ctx.env = &env;
    ctx.start_cost = &start_cost;
    ctx.delivery_of = &delivery_of_idx;
    ctx.pickup_of   = &pickup_of_idx;
    ctx.max_capacity = std::max(1, max_capacity);
    ctx.initial_load = std::max(0, initial_load);

    ctx.max_edge_cost = 0.f;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const float c = env.get_cost(i, j);
            if (c > ctx.max_edge_cost && c < 1e8f) ctx.max_edge_cost = c;
        }

    // Build request list: paired (pickup,delivery) + singletons.
    {
        std::vector<char> consumed(n, 0);
        for (int i = 0; i < n; ++i) {
            if (consumed[i]) continue;
            const int di = delivery_of_idx[i];
            if (di >= 0) {
                AlnsRequest r; r.p_idx = i; r.d_idx = di;
                ctx.requests.push_back(r);
                consumed[i] = consumed[di] = 1;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (consumed[i]) continue;
            AlnsRequest r; r.p_idx = -1; r.d_idx = i;
            ctx.requests.push_back(r);
        }
    }

    std::mt19937 rng(0x9e3779b9u ^ static_cast<uint32_t>(n));

    // Initial solution = pure cheapest insertion (no noise).
    std::vector<int> cur;
    {
        std::vector<AlnsRequest> bank = ctx.requests;
        if (!alns_repair_cheapest(cur, bank, ctx, 0.f, rng)) return {};
    }
    std::vector<int> best = cur;
    float best_cost = alns_cost(cur, ctx);
    float cur_cost  = best_cost;

    // SA start temperature: w% worse solution accepted w.p. 0.5 →  T = w·c0 / ln 2.
    float T = (params.sa_w_start * std::max(1.f, best_cost)) / std::log(2.f);
    if (!std::isfinite(T) || T <= 0.f) T = 1.f;

    constexpr int N_DESTROY = 3, N_REPAIR = 2;
    std::array<float, N_DESTROY> dw{1.f, 1.f, 1.f};
    std::array<float, N_REPAIR>  rw{1.f, 1.f};
    std::array<float, N_DESTROY> dscore{}; std::array<int, N_DESTROY> dused{};
    std::array<float, N_REPAIR>  rscore{}; std::array<int, N_REPAIR>  rused{};

    auto roulette = [&](const auto& weights) -> int {
        float sum = 0.f;
        for (float w : weights) sum += w;
        std::uniform_real_distribution<float> ud(0.f, std::max(1e-6f, sum));
        float y = ud(rng);
        for (int i = 0; i < (int)weights.size(); ++i) {
            y -= weights[i];
            if (y <= 0.f) return i;
        }
        return (int)weights.size() - 1;
    };

    const int n_pairs = std::max(1, (int)ctx.requests.size());
    const int q_min = std::max(1, static_cast<int>(params.removal_min * n_pairs));
    const int q_max = std::max(q_min, static_cast<int>(params.removal_max * n_pairs));
    std::uniform_int_distribution<int> uq(q_min, q_max);
    const float noise_amp = params.noise_factor * ctx.max_edge_cost;

    int seg_iter = 0;

    for (int iter = 0; iter < params.max_iterations; ++iter) {
        const int q   = uq(rng);
        const int dop = roulette(dw);
        const int rop = roulette(rw);
        ++dused[dop]; ++rused[rop];

        std::vector<int> trial = cur;
        std::vector<AlnsRequest> bank;
        switch (dop) {
            case 0: alns_destroy_random(trial, ctx.requests, q, bank, rng); break;
            case 1: alns_destroy_worst (trial, ctx.requests, q, params.p_worst, bank, ctx, rng); break;
            case 2: alns_destroy_shaw  (trial, ctx.requests, q, params.p_shaw,  bank, ctx, rng); break;
        }
        const bool ok = (rop == 0)
            ? alns_repair_cheapest(trial, bank, ctx, noise_amp, rng)
            : alns_repair_regret2 (trial, bank, ctx, noise_amp, rng);
        if (!ok) continue;

        const float trial_cost = alns_cost(trial, ctx);
        if (!std::isfinite(trial_cost)) continue;

        if (trial_cost < best_cost) {
            best = trial; best_cost = trial_cost;
            cur  = trial; cur_cost  = trial_cost;
            dscore[dop] += params.sigma1; rscore[rop] += params.sigma1;
        } else if (trial_cost < cur_cost) {
            cur = trial; cur_cost = trial_cost;
            dscore[dop] += params.sigma2; rscore[rop] += params.sigma2;
        } else {
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            const float dE = trial_cost - cur_cost;
            const float p_accept = std::exp(-dE / std::max(1e-6f, T));
            if (u01(rng) < p_accept) {
                cur = trial; cur_cost = trial_cost;
                dscore[dop] += params.sigma3; rscore[rop] += params.sigma3;
            }
        }

        T *= params.sa_cooling;
        ++seg_iter;
        if (seg_iter >= params.segment_size) {
            for (int i = 0; i < N_DESTROY; ++i) {
                const float perf = (dused[i] > 0) ? dscore[i] / dused[i] : 0.f;
                dw[i] = dw[i] * (1.f - params.reaction_factor)
                      + params.reaction_factor * perf;
                if (dw[i] < 0.01f) dw[i] = 0.01f;
                dscore[i] = 0; dused[i] = 0;
            }
            for (int i = 0; i < N_REPAIR; ++i) {
                const float perf = (rused[i] > 0) ? rscore[i] / rused[i] : 0.f;
                rw[i] = rw[i] * (1.f - params.reaction_factor)
                      + params.reaction_factor * perf;
                if (rw[i] < 0.01f) rw[i] = 0.01f;
                rscore[i] = 0; rused[i] = 0;
            }
            seg_iter = 0;
        }
    }

    std::vector<ObjectiveNode> result;
    result.reserve(best.size());
    for (int idx : best) result.push_back(env.nodes[idx]);
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

std::vector<ObjectiveNode> LocalSolutionAgent::plan_sequence(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const std::vector<float>&  start_costs,
    int                        max_capacity,
    int                        initial_load,
    const DbVNSParams&         params)
{
    int n = env.size();
    if (n == 0) return {};

    // Build reverse map: pickup_id → delivery_id (for tasks in pickup_of).
    std::unordered_map<osmium::object_id_type, osmium::object_id_type> delivery_of_map;
    delivery_of_map.reserve(pickup_of.size());
    for (const auto& [del_id, pu_id] : pickup_of)
        delivery_of_map[pu_id] = del_id;

    std::vector<int>  delivery_of_idx(n, -1);  // pickup env-idx → delivery env-idx
    std::vector<int>  pickup_of_idx  (n, -1);  // delivery env-idx → pickup env-idx
    std::vector<bool> initially_avail(n, false);

    for (int i = 0; i < n; ++i) {
        const osmium::object_id_type nid = env.nodes[i].id;

        auto it_d = delivery_of_map.find(nid);
        if (it_d != delivery_of_map.end()) {
            // Pickup of an unpicked task → available immediately, unlocks delivery.
            int di = env.find_index(it_d->second);
            if (di >= 0) delivery_of_idx[i] = di;
            initially_avail[i] = true;
        } else {
            auto it_p = pickup_of.find(nid);
            if (it_p != pickup_of.end()) {
                // Delivery of an unpicked task → locked until pickup visited.
                int pi = env.find_index(it_p->second);
                if (pi >= 0) pickup_of_idx[i] = pi;
            } else {
                // No pending pickup constraint: carried delivery or free node.
                initially_avail[i] = true;
            }
        }
    }

    return forward_dbvns(env, delivery_of_idx, pickup_of_idx,
                         initially_avail, start_costs,
                         max_capacity, initial_load, params);
}

std::vector<ObjectiveNode> LocalSolutionAgent::plan_sequence_alns(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const std::vector<float>&  start_costs,
    int                        max_capacity,
    int                        initial_load,
    const ALNSParams&          params)
{
    int n = env.size();
    if (n == 0) return {};

    std::unordered_map<osmium::object_id_type, osmium::object_id_type> delivery_of_map;
    delivery_of_map.reserve(pickup_of.size());
    for (const auto& [del_id, pu_id] : pickup_of)
        delivery_of_map[pu_id] = del_id;

    std::vector<int> delivery_of_idx(n, -1);
    std::vector<int> pickup_of_idx  (n, -1);

    for (int i = 0; i < n; ++i) {
        const osmium::object_id_type nid = env.nodes[i].id;
        auto it_d = delivery_of_map.find(nid);
        if (it_d != delivery_of_map.end()) {
            int di = env.find_index(it_d->second);
            if (di >= 0) delivery_of_idx[i] = di;
        } else {
            auto it_p = pickup_of.find(nid);
            if (it_p != pickup_of.end()) {
                int pi = env.find_index(it_p->second);
                if (pi >= 0) pickup_of_idx[i] = pi;
            }
        }
    }

    return forward_alns(env, delivery_of_idx, pickup_of_idx,
                        start_costs, max_capacity, initial_load, params);
}
