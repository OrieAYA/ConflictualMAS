#include "DMASforPD/Algorithms/DbVNS.hpp"
#include "DMASforPD/Structures/OperableEnvironment.hpp"
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
    // Congestion-aware (BPR per branch) context. When time_aware, each edge is
    // costed at depart = base_step + cumulative_cost/speed, so branches deeper
    // in the decomposition are sampled at later times.
    bool                             time_aware    = false;
    float                            speed         = 1.f;
    int                              base_step     = 0;
};

// Departure step at the END of state `s` (= base + cumulative travel time).
static inline int fwd_depart_step(float accumulated_cost, const FwdCtx& ctx) {
    return ctx.base_step + static_cast<int>(accumulated_cost / ctx.speed);
}

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
            : (ctx.time_aware
                 ? ctx.env->get_cost_at(s.seq.back(), j, fwd_depart_step(s.cost, ctx))
                 : ctx.env->get_cost(s.seq.back(), j));
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
        : (ctx.time_aware
             ? ctx.env->get_cost_at(current->seq.back(), j, fwd_depart_step(current->cost, ctx))
             : ctx.env->get_cost(current->seq.back(), j));

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
    // Congestion-aware per-branch BPR: each edge costed at its traversal time.
    ctx.time_aware = env.time_aware();
    ctx.speed      = env.plan_speed();
    ctx.base_step  = env.plan_base();

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

