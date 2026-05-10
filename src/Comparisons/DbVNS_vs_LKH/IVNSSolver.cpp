#include "IVNSSolver.hpp"
#include <algorithm>
#include <limits>
#include <chrono>

namespace {

// ── Tables ────────────────────────────────────────────────────────────────────

static void build_tables(
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

static float route_cost(const std::vector<int>& rt, const OperableEnvironment& env)
{
    float c = 0.f;
    for (int i = 0; i + 1 < (int)rt.size(); ++i) {
        float e = env.get_cost(rt[i], rt[i + 1]);
        if (e < 0 || e >= 1e8f) return std::numeric_limits<float>::max();
        c += e;
    }
    return c;
}

// ── Pair insertion ────────────────────────────────────────────────────────────
// Cost delta for inserting pickup p at route position ins_p, then delivery d at
// position ins_d in the new route (after p is inserted), with ins_d > ins_p.
// O(1). Returns 1e9f if any required edge is missing.
static float pair_delta(
    const std::vector<int>& rt,
    int p, int d, int ins_p, int ins_d,
    const OperableEnvironment& env)
{
    int rn = (int)rt.size();
    float delta = 0.f;

    // p insertion between rt[ins_p-1] and rt[ins_p]
    int pre_p  = (ins_p > 0)  ? rt[ins_p - 1] : -1;
    int post_p = (ins_p < rn) ? rt[ins_p]     : -1;
    if (pre_p  >= 0) { float c = env.get_cost(pre_p, p);   if (c < 0 || c >= 1e8f) return 1e9f; delta += c; }
    if (post_p >= 0) { float c = env.get_cost(p, post_p);  if (c < 0 || c >= 1e8f) return 1e9f; delta += c; }
    if (pre_p >= 0 && post_p >= 0) { float c = env.get_cost(pre_p, post_p); delta -= (c < 0 ? 1e9f : c); }

    // d insertion at ins_d in the route-with-p already inserted
    // new_route[ins_d-1] = p if ins_d-1 == ins_p, else rt[ins_d-2]
    // new_route[ins_d]   = rt[ins_d-1]  (since ins_d > ins_p)
    int pre_d  = (ins_d - 1 == ins_p) ? p : rt[ins_d - 2];
    int post_d = (ins_d <= rn) ? rt[ins_d - 1] : -1;
    { float c = env.get_cost(pre_d, d);   if (c < 0 || c >= 1e8f) return 1e9f; delta += c; }
    if (post_d >= 0) { float c = env.get_cost(d, post_d); if (c < 0 || c >= 1e8f) return 1e9f; delta += c; }
    if (post_d >= 0) { float c = env.get_cost(pre_d, post_d); delta -= (c < 0 ? 1e9f : c); }

    return delta;
}

static void apply_pair(std::vector<int>& rt, int p, int d, int ins_p, int ins_d)
{
    rt.insert(rt.begin() + ins_p, p);
    rt.insert(rt.begin() + ins_d, d);
}

static std::pair<int,int> best_pair_pos(
    const std::vector<int>& rt, int p, int d,
    const OperableEnvironment& env, float& out_delta)
{
    int rn = (int)rt.size();
    float best = 1e9f;
    int bi = 0, bj = 1;
    for (int i = 0; i <= rn; ++i) {
        for (int j = i + 1; j <= rn + 1; ++j) {
            float delta = pair_delta(rt, p, d, i, j, env);
            if (delta < best) { best = delta; bi = i; bj = j; }
        }
    }
    out_delta = best;
    return {bi, bj};
}

// ── Construction ──────────────────────────────────────────────────────────────

static std::vector<int> build_insertion(
    const OperableEnvironment& env,
    const std::vector<int>&    pair_idx,
    const std::vector<bool>&   is_delivery)
{
    int n = env.size();
    std::vector<std::pair<int,int>> pairs;
    pairs.reserve(n / 2);
    for (int i = 0; i < n; ++i)
        if (!is_delivery[i] && pair_idx[i] >= 0)
            pairs.push_back({i, pair_idx[i]});
    if (pairs.empty()) return {};

    // Seed: pair with max pickup→delivery cost (most spread in graph)
    int seed = 0; float seed_c = -1.f;
    for (int k = 0; k < (int)pairs.size(); ++k) {
        float c = env.get_cost(pairs[k].first, pairs[k].second);
        if (c >= 0.f && c < 1e8f && c > seed_c) { seed_c = c; seed = k; }
    }
    std::vector<int> rt = {pairs[seed].first, pairs[seed].second};

    for (int k = 0; k < (int)pairs.size(); ++k) {
        if (k == seed) continue;
        float delta;
        auto [bi, bj] = best_pair_pos(rt, pairs[k].first, pairs[k].second, env, delta);
        apply_pair(rt, pairs[k].first, pairs[k].second, bi, bj);
    }
    return rt;
}

// ── Free zone identification ──────────────────────────────────────────────────
// A free zone is a maximal subsequence with no intra-zone P-D pair.
// Within such a zone, all nodes can be reordered freely → unrestricted 2-opt.
static std::vector<std::pair<int,int>> free_zones(
    const std::vector<int>& rt,
    const std::vector<int>& pair_idx,
    int env_n)
{
    int rn = (int)rt.size();
    std::vector<int>  pos(env_n, -1);
    std::vector<bool> seen(env_n, false);
    for (int i = 0; i < rn; ++i) if (rt[i] < env_n) pos[rt[i]] = i;

    // Build constraint intervals [min(pos_P, pos_D), max(pos_P, pos_D)]
    std::vector<std::pair<int,int>> ivals;
    for (int i = 0; i < rn; ++i) {
        int u = rt[i]; if (u >= env_n || seen[u]) continue;
        int v = pair_idx[u]; if (v < 0 || pos[v] < 0) continue;
        ivals.push_back({std::min(i, pos[v]), std::max(i, pos[v])});
        seen[u] = seen[v] = true;
    }

    // Sort and merge overlapping intervals
    std::sort(ivals.begin(), ivals.end());
    std::vector<std::pair<int,int>> merged;
    for (auto& [l, r] : ivals) {
        if (!merged.empty() && l <= merged.back().second)
            merged.back().second = std::max(merged.back().second, r);
        else
            merged.push_back({l, r});
    }

    // Gaps between merged intervals = free zones
    std::vector<std::pair<int,int>> zones;
    int prev = 0;
    for (auto& [l, r] : merged) {
        if (l > prev) zones.push_back({prev, l - 1});
        prev = r + 1;
    }
    if (prev < rn) zones.push_back({prev, rn - 1});
    return zones;
}

// ── 2-opt on a free zone (no PDP check needed) ───────────────────────────────

static void two_opt_zone(
    std::vector<int>& rt, int l, int r,
    const OperableEnvironment& env, int max_passes = 20)
{
    if (r - l < 2) return;
    int rn = (int)rt.size();
    bool improved = true; int passes = 0;
    while (improved && passes < max_passes) {
        improved = false; ++passes;
        for (int i = l; i < r; ++i) {
            for (int j = i + 2; j <= r; ++j) {
                int a = rt[i], b = rt[i+1], c = rt[j];
                int d2 = (j + 1 < rn) ? rt[j+1] : -1;
                float ab = env.get_cost(a, b), ac = env.get_cost(a, c);
                if (ab < 0 || ab >= 1e8f || ac < 0 || ac >= 1e8f) continue;
                float gain = ab - ac;
                if (d2 >= 0) {
                    float cd = env.get_cost(c, d2), bd = env.get_cost(b, d2);
                    if (cd < 0 || cd >= 1e8f || bd < 0 || bd >= 1e8f) continue;
                    gain += cd - bd;
                }
                if (gain <= 0.f) continue;
                std::reverse(rt.begin() + i + 1, rt.begin() + j + 1);
                improved = true; i = r; break;
            }
        }
    }
}

// ── PDP-constrained 2-opt (full route) ───────────────────────────────────────

static void pdp_two_opt(
    std::vector<int>& rt, const OperableEnvironment& env,
    const std::vector<int>& pair_idx, const std::vector<bool>& is_delivery,
    int max_passes = 20, int max_seg = 60)
{
    int n = (int)rt.size();
    std::vector<int> pos(n);
    bool improved = true; int passes = 0;
    while (improved && passes < max_passes) {
        improved = false; ++passes;
        for (int i = 0; i < n; ++i) pos[rt[i]] = i;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 2; j < n && (j - i) <= max_seg; ++j) {
                int a = rt[i], b = rt[i+1], c = rt[j];
                int d2 = (j + 1 < n) ? rt[j+1] : -1;
                float ab = env.get_cost(a, b), ac = env.get_cost(a, c);
                if (ab < 0 || ab >= 1e8f || ac < 0 || ac >= 1e8f) continue;
                float gain = ab - ac;
                if (d2 >= 0) {
                    float cd = env.get_cost(c, d2), bd = env.get_cost(b, d2);
                    if (cd < 0 || cd >= 1e8f || bd < 0 || bd >= 1e8f) continue;
                    gain += cd - bd;
                }
                if (gain <= 0.f) continue;
                bool ok = true;
                for (int kk = i + 1; kk <= j && ok; ++kk) {
                    if (!is_delivery[rt[kk]]) continue;
                    int pi = pair_idx[rt[kk]]; if (pi < 0) continue;
                    int pp = pos[pi], dp = pos[rt[kk]];
                    if (pp >= i + 1 && pp <= j && pp < dp) ok = false;
                }
                if (!ok) continue;
                std::reverse(rt.begin() + i + 1, rt.begin() + j + 1);
                improved = true; i = n; break;
            }
        }
    }
}

// ── Or-opt: single-node relocation, PDP-aware ────────────────────────────────

static void or_opt(
    std::vector<int>& rt, const OperableEnvironment& env,
    const std::vector<int>& pair_idx, const std::vector<bool>& is_delivery,
    int max_passes = 20)
{
    int n = (int)rt.size();
    std::vector<int> pos(n);
    bool improved = true; int passes = 0;
    while (improved && passes < max_passes) {
        improved = false; ++passes;
        for (int i = 0; i < n; ++i) pos[rt[i]] = i;
        float gbest = 0.f; int bvi = -1, bk = -1;
        for (int vi = 0; vi < n; ++vi) {
            int v = rt[vi];
            int pre = (vi > 0)     ? rt[vi-1] : -1;
            int nxt = (vi < n-1)   ? rt[vi+1] : -1;
            float rv = 0.f;
            if (pre >= 0) { float c = env.get_cost(pre, v); if (c < 0 || c >= 1e8f) continue; rv += c; }
            if (nxt >= 0) { float c = env.get_cost(v, nxt); if (c < 0 || c >= 1e8f) continue; rv += c; }
            if (pre >= 0 && nxt >= 0) { float br = env.get_cost(pre, nxt); rv -= (br < 0 ? 1e9f : br); }
            int pv = pair_idx[v]; bool vd = is_delivery[v]; int adj = -1;
            if (pv >= 0) { int pp = pos[pv]; adj = (pp > vi) ? pp - 1 : pp; }
            auto gr = [&](int k) -> int { return (k < vi) ? rt[k] : rt[k+1]; };
            float lbest = 0.f; int lk = -1;
            for (int k = 0; k < n; ++k) {
                if (k == vi) continue;
                if (pv >= 0) { if (vd && k <= adj) continue; if (!vd && k > adj) continue; }
                int bef = (k > 0)     ? gr(k-1) : -1;
                int aft = (k < n-1)   ? gr(k)   : -1;
                float ins = 0.f;
                if (bef >= 0) { float c = env.get_cost(bef, v); if (c < 0 || c >= 1e8f) continue; ins += c; }
                if (aft >= 0) { float c = env.get_cost(v, aft); if (c < 0 || c >= 1e8f) continue; ins += c; }
                if (bef >= 0 && aft >= 0) { float c = env.get_cost(bef, aft); ins -= (c < 0 ? 1e9f : c); }
                float delta = rv - ins;
                if (delta > lbest) { lbest = delta; lk = k; }
            }
            if (lk >= 0 && lbest > gbest) { gbest = lbest; bvi = vi; bk = lk; }
        }
        if (bvi < 0) break;
        int v = rt[bvi];
        rt.erase(rt.begin() + bvi);
        rt.insert(rt.begin() + bk, v);
        improved = true;
    }
}

// ── Full refinement pass ──────────────────────────────────────────────────────

static void refine(
    std::vector<int>& rt, const OperableEnvironment& env,
    const std::vector<int>& pair_idx, const std::vector<bool>& is_delivery)
{
    int env_n = env.size();
    for (auto& [l, r] : free_zones(rt, pair_idx, env_n))
        two_opt_zone(rt, l, r, env);
    pdp_two_opt(rt, env, pair_idx, is_delivery);
    or_opt(rt, env, pair_idx, is_delivery);
}

// ── VNS shake: remove k highest-cost pairs, reinsert ─────────────────────────

static void shake(
    std::vector<int>& rt,
    const std::vector<int>& pair_idx, const std::vector<bool>& is_delivery,
    const OperableEnvironment& env, int k)
{
    int n = env.size();
    int rn = (int)rt.size();
    std::vector<int> pos(n, -1);
    for (int i = 0; i < rn; ++i) if (rt[i] < n) pos[rt[i]] = i;

    // Score each pair by total edge cost incident to its two nodes
    std::vector<std::pair<float,int>> scores;
    scores.reserve(n / 2);
    for (int i = 0; i < n; ++i) {
        if (is_delivery[i] || pair_idx[i] < 0) continue;
        int pp = pos[i], dp = pos[pair_idx[i]];
        if (pp < 0 || dp < 0) continue;
        float c = 0.f;
        if (pp > 0)    { float e = env.get_cost(rt[pp-1], i);          if (e >= 0 && e < 1e8f) c += e; }
        if (pp < rn-1) { float e = env.get_cost(i, rt[pp+1]);          if (e >= 0 && e < 1e8f) c += e; }
        if (dp > 0)    { float e = env.get_cost(rt[dp-1], pair_idx[i]); if (e >= 0 && e < 1e8f) c += e; }
        if (dp < rn-1) { float e = env.get_cost(pair_idx[i], rt[dp+1]); if (e >= 0 && e < 1e8f) c += e; }
        scores.push_back({c, i});
    }
    std::sort(scores.begin(), scores.end(), std::greater<>());
    k = std::min(k, (int)scores.size());

    // Remove k worst pairs
    std::vector<std::pair<int,int>> removed;
    removed.reserve(k);
    for (int ki = 0; ki < k; ++ki) {
        int p = scores[ki].second, d = pair_idx[p];
        removed.push_back({p, d});
        rt.erase(std::remove(rt.begin(), rt.end(), p), rt.end());
        rt.erase(std::remove(rt.begin(), rt.end(), d), rt.end());
    }

    // Reinsert at cheapest positions
    for (auto& [p, d] : removed) {
        float delta;
        auto [bi, bj] = best_pair_pos(rt, p, d, env, delta);
        apply_pair(rt, p, d, bi, bj);
    }
}

} // namespace

// ── Public interface ──────────────────────────────────────────────────────────

IVNSSolver::Result IVNSSolver::solve(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    int max_iterations,
    int max_k)
{
    auto wall0 = std::chrono::high_resolution_clock::now();

    std::vector<int>  pair_idx;
    std::vector<bool> is_delivery;
    build_tables(env, pickup_of, pair_idx, is_delivery);

    auto rt = build_insertion(env, pair_idx, is_delivery);
    if (rt.empty()) return {};

    refine(rt, env, pair_idx, is_delivery);

    std::vector<int> best = rt;
    float best_cost = route_cost(best, env);
    int k = 1, iters_done = 0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        ++iters_done;
        auto trial = best;
        shake(trial, pair_idx, is_delivery, env, k);
        refine(trial, env, pair_idx, is_delivery);
        float tc = route_cost(trial, env);
        if (tc < best_cost) { best_cost = tc; best = trial; k = 1; }
        else if (k < max_k) ++k;
    }

    std::vector<ObjectiveNode> seq;
    seq.reserve(best.size());
    for (int idx : best) seq.push_back(env.nodes[idx]);

    auto wall1 = std::chrono::high_resolution_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(wall1 - wall0).count();
    return {std::move(seq), best_cost, ms, iters_done};
}

void IVNSSolver::insert_pair(
    Result&                    existing,
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    int                        p_idx,
    int                        d_idx)
{
    int n = env.size();

    // Empty solution: just place pickup then delivery.
    if (existing.sequence.empty()) {
        existing.sequence = { env.nodes[p_idx], env.nodes[d_idx] };
        float c = env.get_cost(p_idx, d_idx);
        existing.cost = (c >= 0.f && c < 1e8f) ? c : std::numeric_limits<float>::max();
        return;
    }

    // Convert existing ObjectiveNode sequence to env-index route.
    std::vector<int> rt;
    rt.reserve(existing.sequence.size());
    for (const auto& node : existing.sequence) {
        int idx = env.find_index(node.id);
        if (idx >= 0) rt.push_back(idx);
    }

    // Find and apply the cheapest valid (p, d) insertion.
    float delta;
    auto [bi, bj] = best_pair_pos(rt, p_idx, d_idx, env, delta);
    apply_pair(rt, p_idx, d_idx, bi, bj);

    // Rebuild constraint tables for the expanded environment (includes new pair).
    std::vector<int>  pair_idx;
    std::vector<bool> is_delivery;
    build_tables(env, pickup_of, pair_idx, is_delivery);

    // Full refinement pass: free-zone 2-opt + PDP 2-opt + or-opt.
    refine(rt, env, pair_idx, is_delivery);

    // Convert back to ObjectiveNode sequence.
    existing.sequence.clear();
    existing.sequence.reserve(rt.size());
    for (int idx : rt) {
        if (idx >= 0 && idx < n)
            existing.sequence.push_back(env.nodes[idx]);
    }
    existing.cost = route_cost(rt, env);
}
