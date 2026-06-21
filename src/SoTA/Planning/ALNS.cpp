#include "SoTA/Planning/ALNS.hpp"
#include "DMASforPD/Structures/OperableEnvironment.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
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

}  // namespace

std::vector<ObjectiveNode> plan_sequence_alns(
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
