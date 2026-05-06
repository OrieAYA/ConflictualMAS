#include "LocalSolutionAgent.hpp"
#include "OperableEnvironment.hpp"
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

// ============================================================================
// INTERNALS
// ============================================================================

namespace {

// Node in the DbVNS decomposition tree.
// sequence: partial sequence built in reverse (top = last endpoint committed).
// pending : node ids not yet placed in the sequence.
// forbidden: node ids excluded at this branch level (set by decompose or init).
// cost    : sum of operable_env costs between consecutive entries in sequence.
struct PDPDecomposedSolution {
    PDPDecomposedSolution* parent = nullptr;
    std::unordered_map<osmium::object_id_type, PDPDecomposedSolution*> children;
    std::vector<ObjectiveNode>                                         sequence;
    std::unordered_set<osmium::object_id_type>                        pending;
    std::unordered_set<osmium::object_id_type>                        forbidden;
    float cost = 0.0f;
};

// ============================================================================
// DELETE DECOMPOSITION TREE
// ============================================================================
void delete_tree(PDPDecomposedSolution* root) {
    if (!root) return;
    for (auto& [id, child] : root->children)
        delete_tree(child);
    delete root;
}

// ============================================================================
// ESTIMATE REMAINING
// ============================================================================
// Optimistic lower bound: current cost + min edge cost to any pending node
// multiplied by the number of pending nodes.
float estimate_remaining(const PDPDecomposedSolution& node,
                          const OperableEnvironment&   env) {
    if (node.pending.empty()) return node.cost;
    if (node.sequence.empty()) return std::numeric_limits<float>::max();

    int last_idx = env.find_index(node.sequence.back().id);
    if (last_idx < 0) return std::numeric_limits<float>::max();

    float min_c = std::numeric_limits<float>::max();
    for (osmium::object_id_type pid : node.pending) {
        int pidx = env.find_index(pid);
        if (pidx < 0) continue;
        float c = env.get_cost(last_idx, pidx);
        if (c >= 0.0f && c < min_c) min_c = c;
    }
    if (min_c == std::numeric_limits<float>::max())
        return std::numeric_limits<float>::max();
    return node.cost + min_c * static_cast<float>(node.pending.size());
}

// ============================================================================
// GREEDY CONSTRUCTION
// ============================================================================
// Repeatedly picks the cheapest reachable node from pending (excluding forbidden).
// Reuses existing children when available. When a delivery node is selected,
// its pickup is queued in pending so it is placed before it in the final sequence.
PDPDecomposedSolution* greedy_construction(
    PDPDecomposedSolution*     node,
    const OperableEnvironment& env,
    const PairingMap&          pickup_of
) {
    while (!node->pending.empty()) {
        if (node->sequence.empty()) break;

        int last_idx = env.find_index(node->sequence.back().id);
        if (last_idx < 0) break;

        osmium::object_id_type best_id   = 0;
        float                  best_cost = std::numeric_limits<float>::max();
        int                    best_cidx = -1;

        for (osmium::object_id_type cid : node->pending) {
            if (node->forbidden.count(cid)) continue;
            int cidx = env.find_index(cid);
            if (cidx < 0) continue;
            float c = env.get_cost(last_idx, cidx);
            if (c >= 0.0f && c < best_cost) {
                best_cost = c;
                best_id   = cid;
                best_cidx = cidx;
            }
        }

        if (best_id == 0) break;

        // Reuse existing child if this branch was already explored.
        auto it = node->children.find(best_id);
        if (it != node->children.end()) {
            node = it->second;
            continue;
        }

        // Build new child.
        PDPDecomposedSolution* child = new PDPDecomposedSolution();
        child->parent   = node;
        child->sequence = node->sequence;
        child->sequence.push_back(env.nodes[best_cidx]);
        child->pending  = node->pending;
        child->pending.erase(best_id);
        child->cost     = node->cost + best_cost;

        // Initialize forbidden = all nodes already in sequence (prevents revisiting).
        for (const ObjectiveNode& n : child->sequence)
            child->forbidden.insert(n.id);

        // If best_id is a delivery node, queue its pickup in pending so it is
        // placed earlier in the final (reversed) sequence, respecting P_i before D_i.
        auto pk_it = pickup_of.find(best_id);
        if (pk_it != pickup_of.end()) {
            osmium::object_id_type pid = pk_it->second;
            if (!child->forbidden.count(pid))
                child->pending.insert(pid);
        }

        node->children[best_id] = child;
        node = child;
    }
    return node;
}

// ============================================================================
// DECOMPOSE WITH FORBIDDEN
// ============================================================================
// Backtracks from solution toward the root, marking each tail node as forbidden
// in its parent. Returns all visited ancestors as alternative starting points.
std::vector<PDPDecomposedSolution*> decompose_with_forbidden(
    PDPDecomposedSolution* solution
) {
    std::vector<PDPDecomposedSolution*> decomposed;
    PDPDecomposedSolution* current = solution;

    while (current->parent != nullptr) {
        osmium::object_id_type forbidden_id = current->sequence.back().id;
        current = current->parent;
        current->forbidden.insert(forbidden_id);
        decomposed.push_back(current);
    }
    return decomposed;
}

// ============================================================================
// DB_VNS_SEARCH
// ============================================================================
// Main DbVNS-PDP search loop. Returns the best sequence found in reverse-build
// order (call std::reverse to obtain the actual visit order).
std::vector<ObjectiveNode> db_vns_search(
    const ObjectiveNode&       starting_node,
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const DbVNSParams&         params
) {
    if (env.nodes.empty()) return {};

    // ---- Initial state ---------------------------------------------------
    // Stack begins with the anchor delivery node (will be visited last).
    // Pending = all other delivery nodes + pickup of the anchor (P_0).
    // Other pickups are added dynamically when their delivery is selected.
    PDPDecomposedSolution* initial = new PDPDecomposedSolution();
    initial->sequence.push_back(starting_node);
    initial->forbidden.insert(starting_node.id);

    osmium::object_id_type p0_id = 0;
    {
        auto it = pickup_of.find(starting_node.id);
        if (it != pickup_of.end()) p0_id = it->second;
    }

    for (const ObjectiveNode& n : env.nodes) {
        if (n.id == starting_node.id) continue;
        bool is_delivery = pickup_of.count(n.id) > 0;
        if (is_delivery || n.id == p0_id)
            initial->pending.insert(n.id);
    }

    // ---- Greedy initial solution ------------------------------------------
    PDPDecomposedSolution* current = greedy_construction(initial, env, pickup_of);

    if (current->sequence.size() <= 1 || !current->pending.empty()) {
        delete_tree(initial);
        return {};
    }

    PDPDecomposedSolution pbest      = *current;
    float                 pbest_cost = pbest.cost;

    // ========================================
    // PARAMETERS
    // ========================================
    const int   k_max           = params.k_max;
    const int   max_decomps     = params.max_decompositions;
    const float threshold_mult  = 1.0f + static_cast<float>(params.max_divergence);
    const int   max_consec_empty = 3;

    int k            = 1;
    int iter         = 0;
    int consec_empty = 0;

    while (iter < params.max_iterations) {

        // ========================================
        // A. SHAKE (Decomposition)
        // ========================================
        PDPDecomposedSolution* shaken = &pbest;
        int until = std::min(
            static_cast<int>(shaken->sequence.size() / k_max) * (k - 1),
            std::max(0, static_cast<int>(shaken->sequence.size()) - 2)
        );
        for (int i = 0; i < until; ++i) {
            if (shaken->parent) shaken = shaken->parent;
            else break;
        }

        std::vector<PDPDecomposedSolution*> decomposed =
            decompose_with_forbidden(shaken);

        if (decomposed.empty()) { ++k; ++iter; continue; }

        // ========================================
        // B. FILTRAGE PROMISING ADAPTATIF
        // ========================================
        float local_threshold = pbest_cost * threshold_mult;
        std::vector<PDPDecomposedSolution*> promising;

        for (auto* d : decomposed) {
            if (estimate_remaining(*d, env) <= local_threshold)
                promising.push_back(d);
        }

        if (promising.empty()) {
            ++consec_empty; ++k; ++iter;
            if (consec_empty >= max_consec_empty) break;
            continue;
        }

        if (static_cast<int>(promising.size()) > max_decomps) {
            std::sort(promising.begin(), promising.end(),
                [&env](const PDPDecomposedSolution* a, const PDPDecomposedSolution* b) {
                    return estimate_remaining(*a, env) < estimate_remaining(*b, env);
                });
            promising.resize(max_decomps);
        }

        consec_empty = 0;

        // ========================================
        // C. EXPLORATION - 1 GREEDY PAR DÉCOMPOSITION
        // ========================================
        PDPDecomposedSolution* best_from_decomp = nullptr;
        float                  best_decomp_cost = std::numeric_limits<float>::max();

        for (auto* decomp : promising) {
            PDPDecomposedSolution* rebuilt = greedy_construction(decomp, env, pickup_of);
            if (rebuilt->sequence.empty() || !rebuilt->pending.empty()) continue;
            if (rebuilt->cost < best_decomp_cost) {
                best_decomp_cost = rebuilt->cost;
                best_from_decomp = rebuilt;
            }
        }

        if (best_from_decomp) current = best_from_decomp;

        // ========================================
        // D. ACCEPTATION
        // ========================================
        if (best_decomp_cost < pbest_cost) {
            pbest      = *current;
            pbest_cost = best_decomp_cost;
            k = 1;
        } else {
            if (k != k_max) ++k;
        }

        ++iter;
    }

    // ========================================
    // E. CLEANUP
    // ========================================
    delete_tree(initial);
    return pbest.sequence;
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
    osmium::object_id_type     /*agent_current*/
) const {
    // Run DbVNS-PDP; result is in reverse-build order (anchor = last element).
    std::vector<ObjectiveNode> seq = db_vns_search(starting_node, env, pickup_of, params);
    if (seq.empty()) return {};
    // Reverse to obtain the actual visit order: first visited -> ... -> anchor (last).
    std::reverse(seq.begin(), seq.end());
    return seq;
}
