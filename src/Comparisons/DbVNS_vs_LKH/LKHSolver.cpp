#include "LKHSolver.hpp"
#include <algorithm>
#include <limits>
#include <chrono>

// ── Internal helpers (file scope) ─────────────────────────────────────────────

// Per-node data precomputed from pickup_of.
//   pair_idx[i]   = index of paired node (P<->D), -1 if not paired
//   is_delivery[i]= true iff node i is a delivery node (key in pickup_of)
static void build_pair_tables(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    std::vector<int>&          pair_idx,
    std::vector<bool>&         is_delivery)
{
    int n = env.size();
    pair_idx.assign(n, -1);
    is_delivery.assign(n, false);
    for (int i = 0; i < n; ++i) {
        auto it = pickup_of.find(env.nodes[i].id);
        if (it == pickup_of.end()) continue;
        is_delivery[i] = true;
        int p = env.find_index(it->second);
        if (p >= 0) { pair_idx[i] = p; pair_idx[p] = i; }
    }
}

static float lkh_tour_cost(const std::vector<int>& t, const OperableEnvironment& env)
{
    float c = 0.0f;
    for (int i = 0; i + 1 < static_cast<int>(t.size()); ++i) {
        float e = env.get_cost(t[i], t[i + 1]);
        if (e < 0 || e >= 1e8f) return std::numeric_limits<float>::max();
        c += e;
    }
    return c;
}

// ── Nearest-Neighbour construction (PDP-aware) ────────────────────────────────
// Delivery nodes are blocked until their pickup has been visited.

static std::vector<int> lkh_nn_construct(
    const OperableEnvironment& env,
    const std::vector<int>&    pair_idx,
    const std::vector<bool>&   is_delivery,
    int                        start_idx)
{
    int n = env.size();
    std::vector<int>  tour;  tour.reserve(n);
    std::vector<bool> visited(n, false);
    std::vector<bool> blocked(is_delivery); // delivery nodes start blocked

    auto visit = [&](int idx) {
        visited[idx] = true;
        tour.push_back(idx);
        // Visiting a pickup unblocks its delivery.
        if (!is_delivery[idx] && pair_idx[idx] >= 0)
            blocked[pair_idx[idx]] = false;
    };

    visit(start_idx);
    int current = start_idx;

    while (static_cast<int>(tour.size()) < n) {
        float best = std::numeric_limits<float>::max();
        int   next = -1;
        for (int j = 0; j < n; ++j) {
            if (visited[j] || blocked[j]) continue;
            float c = env.get_cost(current, j);
            if (c < 0 || c >= 1e8f) continue;
            if (c < best) { best = c; next = j; }
        }
        if (next < 0) {                        // fallback: any unvisited unblocked
            for (int j = 0; j < n; ++j)
                if (!visited[j] && !blocked[j]) { next = j; break; }
        }
        if (next < 0) break;
        visit(next);
        current = next;
    }
    return tour;
}

// ── 2-opt with PDP constraint maintenance ────────────────────────────────────
// Reversing segment [i+1..j] creates a violation iff for any pair (P,D)
// both in that segment and P was before D — reversal puts D before P.
// Restart-on-first-improvement. Capped at max_passes to bound O(N^5) worst case.
// Segment length also capped at max_seg so the O(seg_len) PDP check stays cheap.

static bool lkh_two_opt(
    std::vector<int>&          t,
    const OperableEnvironment& env,
    const std::vector<int>&    pair_idx,
    const std::vector<bool>&   is_delivery,
    int                        max_passes  = 40,
    int                        max_seg_len = 60)
{
    int n = static_cast<int>(t.size());
    bool any = false;

    std::vector<int> pos(n);
    int  passes  = 0;
    bool improved = true;
    while (improved && passes < max_passes) {
        improved = false;
        ++passes;
        for (int i = 0; i < n; ++i) pos[t[i]] = i;

        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 2; j < n && (j - i) <= max_seg_len; ++j) {
                int a = t[i], b = t[i + 1], c = t[j];
                int d = (j + 1 < n) ? t[j + 1] : -1;

                float ab = env.get_cost(a, b);
                float ac = env.get_cost(a, c);
                if (ab < 0 || ab >= 1e8f || ac < 0 || ac >= 1e8f) continue;

                float gain = ab - ac;
                if (d >= 0) {
                    float cd = env.get_cost(c, d);
                    float bd = env.get_cost(b, d);
                    if (cd < 0 || cd >= 1e8f || bd < 0 || bd >= 1e8f) continue;
                    gain += cd - bd;
                }
                if (gain <= 0.0f) continue;

                // PDP check: reject if any delivery in [i+1..j] has its pickup
                // also in [i+1..j] with pickup position < delivery position.
                bool ok = true;
                for (int k = i + 1; k <= j && ok; ++k) {
                    if (!is_delivery[t[k]]) continue;
                    int pi = pair_idx[t[k]];
                    if (pi < 0) continue;
                    int pp = pos[pi], dp = pos[t[k]];
                    if (pp >= i + 1 && pp <= j && pp < dp) ok = false;
                }
                if (!ok) continue;

                std::reverse(t.begin() + i + 1, t.begin() + j + 1);
                improved = true;
                any = true;
                i = n; // exit outer loop → restart with fresh pos[]
                break;
            }
        }
    }
    return any;
}

// ── Or-opt: single-node relocation (PDP-aware) ───────────────────────────────
// For each node v:
//   Delivery v: can only be inserted AFTER its pickup in the new tour.
//   Pickup v:   can only be inserted BEFORE its delivery in the new tour.
// PDP check is O(1) per candidate insertion position.
// Strategy: one full pass finds the globally best move, applies it, restarts.
// O(N^2) per pass × O(N) passes = O(N^3) total — safe for N<200.

static bool lkh_or_opt(
    std::vector<int>&          t,
    const OperableEnvironment& env,
    const std::vector<int>&    pair_idx,
    const std::vector<bool>&   is_delivery,
    int                        max_passes = 40)
{
    int n = static_cast<int>(t.size());
    bool any = false;

    std::vector<int> pos(n);

    int  passes  = 0;
    bool improved = true;
    while (improved && passes < max_passes) {
        improved = false;
        ++passes;
        for (int i = 0; i < n; ++i) pos[t[i]] = i;

        // Find the single best relocation across all nodes.
        float global_best = 0.0f;
        int   best_vi = -1, best_k = -1;

        for (int vi = 0; vi < n; ++vi) {
            int v   = t[vi];
            int pre = (vi > 0)     ? t[vi - 1] : -1;
            int nxt = (vi < n - 1) ? t[vi + 1] : -1;

            // Cost savings from removing v (bridging pre→nxt).
            float rv = 0.0f;
            if (pre >= 0) {
                float c = env.get_cost(pre, v);
                if (c < 0 || c >= 1e8f) continue;
                rv += c;
            }
            if (nxt >= 0) {
                float c = env.get_cost(v, nxt);
                if (c < 0 || c >= 1e8f) continue;
                rv += c;
            }
            if (pre >= 0 && nxt >= 0) {
                float br = env.get_cost(pre, nxt);
                if (br < 0) br = 1e9f;
                rv -= br;
            }

            int  pv    = pair_idx[v];
            bool v_del = is_delivery[v];
            int  adj   = -1;
            if (pv >= 0) {
                int pp = pos[pv];
                adj = (pp > vi) ? pp - 1 : pp;
            }

            // Maps reduced-tour index k → node index in t (vi excluded).
            auto gr = [&](int k) -> int {
                return (k < vi) ? t[k] : t[k + 1];
            };

            float local_best = 0.0f;
            int   local_k    = -1;

            for (int k = 0; k < n; ++k) {
                if (k == vi) continue;

                if (pv >= 0) {
                    if (v_del  && k <= adj) continue;
                    if (!v_del && k >  adj) continue;
                }

                int before = (k > 0)     ? gr(k - 1) : -1;
                int after  = (k < n - 1) ? gr(k)     : -1;

                float ins = 0.0f;
                if (before >= 0) {
                    float c = env.get_cost(before, v);
                    if (c < 0 || c >= 1e8f) continue;
                    ins += c;
                }
                if (after >= 0) {
                    float c = env.get_cost(v, after);
                    if (c < 0 || c >= 1e8f) continue;
                    ins += c;
                }
                if (before >= 0 && after >= 0) {
                    float c = env.get_cost(before, after);
                    if (c < 0) c = 1e9f;
                    ins -= c;
                }

                float delta = rv - ins;
                if (delta > local_best) { local_best = delta; local_k = k; }
            }

            if (local_k >= 0 && local_best > global_best) {
                global_best = local_best;
                best_vi = vi; best_k = local_k;
            }
        }

        if (best_vi < 0) break; // no improving move in full pass

        // Apply best move of this pass.
        int v = t[best_vi];
        t.erase(t.begin() + best_vi);
        t.insert(t.begin() + best_k, v);
        improved = true;
        any      = true;
    }
    return any;
}

// ── Public interface ──────────────────────────────────────────────────────────

LKHSolver::Result LKHSolver::solve(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    int                        max_restarts)
{
    auto wall0 = std::chrono::high_resolution_clock::now();

    int n = env.size();
    std::vector<int>  pair_idx;
    std::vector<bool> is_delivery;
    build_pair_tables(env, pickup_of, pair_idx, is_delivery);

    // Use pickup nodes as start candidates (they are the "free" nodes).
    std::vector<int> starts;
    starts.reserve(n);
    for (int i = 0; i < n; ++i)
        if (!is_delivery[i]) starts.push_back(i);

    // Evenly sample up to max_restarts.
    if (static_cast<int>(starts.size()) > max_restarts) {
        std::vector<int> sampled;
        sampled.reserve(max_restarts);
        double step = static_cast<double>(starts.size()) / max_restarts;
        for (int i = 0; i < max_restarts; ++i)
            sampled.push_back(starts[static_cast<size_t>(i * step)]);
        starts = std::move(sampled);
    }

    std::vector<int> best_tour;
    float            best_cost      = std::numeric_limits<float>::max();
    int              restarts_done  = 0;

    for (int si : starts) {
        auto tour = lkh_nn_construct(env, pair_idx, is_delivery, si);
        if (static_cast<int>(tour.size()) < n) continue;

        lkh_two_opt(tour, env, pair_idx, is_delivery);
        lkh_or_opt (tour, env, pair_idx, is_delivery);

        float c = lkh_tour_cost(tour, env);
        ++restarts_done;
        if (c < best_cost) { best_cost = c; best_tour = tour; }
    }

    // Convert node indices → ObjectiveNode sequence.
    std::vector<ObjectiveNode> sequence;
    sequence.reserve(best_tour.size());
    for (int idx : best_tour) sequence.push_back(env.nodes[idx]);

    auto wall1 = std::chrono::high_resolution_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        wall1 - wall0).count();

    return {std::move(sequence), best_cost, ms, restarts_done};
}
