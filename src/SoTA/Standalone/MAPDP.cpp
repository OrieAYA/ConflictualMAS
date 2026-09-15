#include "SoTA/Standalone/MAPDP.hpp"
#include "DMASforPD/Policy/PolicyKit.hpp"       // policy_optim::adam_apply
#include "Environment/GeoBox/GraphSearch.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_set>

// ════════════════════════════════════════════════════════════════════════════
// Part 1/3 — numeric core: Dense, BatchNorm1d, MHA. Hand-rolled forward +
// backward + Adam (no autodiff/tensor library in this codebase — same
// convention as PolicyKit::Mlp, just runtime-sized instead of fixed 3-layer).
// ════════════════════════════════════════════════════════════════════════════

namespace {
inline float relu(float x) { return x > 0.f ? x : 0.f; }
inline float relu_grad(float pre) { return pre > 0.f ? 1.f : 0.f; }
inline void accum_into(std::vector<float>& dst, const std::vector<float>& src) {
    for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
}
}  // namespace

// ── Dense ─────────────────────────────────────────────────────────────────

void FaithfulMAPDPSolver::Dense::init(int in_dim, int out_dim,
                                       std::mt19937& rng, float gain) {
    in = in_dim; out = out_dim;
    W.assign(static_cast<size_t>(in) * out, 0.f);
    b.assign(static_cast<size_t>(out), 0.f);
    mW.assign(W.size(), 0.f); vW.assign(W.size(), 0.f);
    mb.assign(b.size(), 0.f); vb.assign(b.size(), 0.f);
    t = 0;
    const float std = gain * std::sqrt(2.f / static_cast<float>(in + out));
    std::normal_distribution<float> nd(0.f, std);
    for (float& w : W) w = nd(rng);
}

void FaithfulMAPDPSolver::Dense::forward(const float* x, float* y) const {
    for (int o = 0; o < out; ++o) {
        float s = b[o];
        const float* row = W.data() + static_cast<size_t>(o) * in;
        for (int i = 0; i < in; ++i) s += row[i] * x[i];
        y[o] = s;
    }
}

void FaithfulMAPDPSolver::Dense::backward(const float* x, const float* dy,
                                           float* dx, float* dW,
                                           float* db) const {
    for (int o = 0; o < out; ++o) {
        const float g = dy[o];
        db[o] += g;
        float* dWrow = dW + static_cast<size_t>(o) * in;
        const float* Wrow = W.data() + static_cast<size_t>(o) * in;
        for (int i = 0; i < in; ++i) {
            dWrow[i] += g * x[i];
            if (dx) dx[i] += g * Wrow[i];
        }
    }
}

void FaithfulMAPDPSolver::Dense::adam_step(const float* dW, const float* db,
                                            float lr) {
    ++t;
    policy_optim::adam_apply(W.data(), dW, mW.data(), vW.data(),
                             static_cast<int>(W.size()), lr, t);
    policy_optim::adam_apply(b.data(), db, mb.data(), vb.data(),
                             static_cast<int>(b.size()), lr, t);
}

// ── BatchNorm1d ───────────────────────────────────────────────────────────

void FaithfulMAPDPSolver::BatchNorm1d::init(int d) {
    dim = d;
    gamma.assign(dim, 1.f); beta.assign(dim, 0.f);
    running_mean.assign(dim, 0.f); running_var.assign(dim, 1.f);
    mgamma.assign(dim, 0.f); vgamma.assign(dim, 0.f);
    mbeta.assign(dim, 0.f);  vbeta.assign(dim, 0.f);
    t = 0;
}

// x: n×dim row-major. training=true uses (and updates) batch statistics;
// false uses the running statistics (frozen — used when n==1, e.g. the
// decoder's context vector, which has no batch to normalise over).
void FaithfulMAPDPSolver::BatchNorm1d::forward(const float* x, float* y,
                                                int n, bool training) {
    if (n <= 0) return;
    std::vector<float> mean(dim, 0.f), var(dim, 0.f);
    if (training && n > 1) {
        for (int d = 0; d < dim; ++d) {
            float s = 0.f;
            for (int r = 0; r < n; ++r) s += x[r * dim + d];
            mean[d] = s / n;
        }
        for (int d = 0; d < dim; ++d) {
            float s = 0.f;
            for (int r = 0; r < n; ++r) {
                const float diff = x[r * dim + d] - mean[d];
                s += diff * diff;
            }
            var[d] = s / n;
        }
        for (int d = 0; d < dim; ++d) {
            running_mean[d] = (1.f - kMomentum) * running_mean[d] + kMomentum * mean[d];
            running_var[d]  = (1.f - kMomentum) * running_var[d]  + kMomentum * var[d];
        }
    } else {
        mean = running_mean;
        var  = running_var;
    }
    for (int r = 0; r < n; ++r)
        for (int d = 0; d < dim; ++d) {
            const float xhat = (x[r * dim + d] - mean[d]) / std::sqrt(var[d] + kEps);
            y[r * dim + d] = gamma[d] * xhat + beta[d];
        }
}

// Standard batchnorm backward (textbook form). `x` must be the SAME input
// forward() was called with in training mode (n>1) — the n==1 inference path
// (running stats) is not exercised during training so no backward is needed
// for it.
void FaithfulMAPDPSolver::BatchNorm1d::backward(const float* x, const float* dy,
                                                 float* dx, int n,
                                                 float* dgamma, float* dbeta) const {
    if (n <= 1) { if (dx) std::memset(dx, 0, sizeof(float) * dim * static_cast<size_t>(std::max(n,0))); return; }
    std::vector<float> mean(dim, 0.f), var(dim, 0.f), xhat(static_cast<size_t>(n) * dim);
    for (int d = 0; d < dim; ++d) {
        float s = 0.f;
        for (int r = 0; r < n; ++r) s += x[r * dim + d];
        mean[d] = s / n;
    }
    for (int d = 0; d < dim; ++d) {
        float s = 0.f;
        for (int r = 0; r < n; ++r) { const float diff = x[r*dim+d]-mean[d]; s += diff*diff; }
        var[d] = s / n;
    }
    for (int r = 0; r < n; ++r)
        for (int d = 0; d < dim; ++d)
            xhat[r * dim + d] = (x[r * dim + d] - mean[d]) / std::sqrt(var[d] + kEps);

    for (int d = 0; d < dim; ++d) {
        float dg = 0.f, db_ = 0.f;
        for (int r = 0; r < n; ++r) {
            dg  += dy[r * dim + d] * xhat[r * dim + d];
            db_ += dy[r * dim + d];
        }
        dgamma[d] += dg;
        dbeta[d]  += db_;
    }
    if (!dx) return;
    for (int d = 0; d < dim; ++d) {
        const float inv_std = 1.f / std::sqrt(var[d] + kEps);
        float sum_dxhat = 0.f, sum_dxhat_xhat = 0.f;
        for (int r = 0; r < n; ++r) {
            const float dxhat = dy[r * dim + d] * gamma[d];
            sum_dxhat      += dxhat;
            sum_dxhat_xhat += dxhat * xhat[r * dim + d];
        }
        for (int r = 0; r < n; ++r) {
            const float dxhat = dy[r * dim + d] * gamma[d];
            dx[r * dim + d] = inv_std / n *
                (n * dxhat - sum_dxhat - xhat[r * dim + d] * sum_dxhat_xhat);
        }
    }
}

void FaithfulMAPDPSolver::BatchNorm1d::adam_step(const float* dgamma,
                                                  const float* dbeta, float lr) {
    ++t;
    policy_optim::adam_apply(gamma.data(), dgamma, mgamma.data(), vgamma.data(), dim, lr, t);
    policy_optim::adam_apply(beta.data(),  dbeta,  mbeta.data(),  vbeta.data(),  dim, lr, t);
}

// ── MHA ───────────────────────────────────────────────────────────────────

void FaithfulMAPDPSolver::MHA::init(int d_embed_, int n_heads_,
                                     std::mt19937& rng, float gain) {
    d_embed = d_embed_; n_heads = n_heads_; d_k = d_embed / n_heads;
    Wq.init(d_embed, d_embed, rng, gain);
    Wk.init(d_embed, d_embed, rng, gain);
    Wv.init(d_embed, d_embed, rng, gain);
    Wo.init(d_embed, d_embed, rng, gain);
}

void FaithfulMAPDPSolver::MHA::forward(const float* queries, int n_q,
                                        const float* keys_values, int n_kv,
                                        float* out, Cache* cache) const {
    std::vector<float> Q(static_cast<size_t>(n_q) * d_embed);
    std::vector<float> K(static_cast<size_t>(n_kv) * d_embed);
    std::vector<float> V(static_cast<size_t>(n_kv) * d_embed);
    for (int r = 0; r < n_q;  ++r) Wq.forward(queries + static_cast<size_t>(r) * d_embed, Q.data() + static_cast<size_t>(r) * d_embed);
    for (int r = 0; r < n_kv; ++r) {
        Wk.forward(keys_values + static_cast<size_t>(r) * d_embed, K.data() + static_cast<size_t>(r) * d_embed);
        Wv.forward(keys_values + static_cast<size_t>(r) * d_embed, V.data() + static_cast<size_t>(r) * d_embed);
    }

    std::vector<float> attn(static_cast<size_t>(n_heads) * n_q * n_kv);
    std::vector<float> concat(static_cast<size_t>(n_q) * d_embed, 0.f);
    const float scale = 1.f / std::sqrt(static_cast<float>(d_k));

    std::vector<float> scores(n_kv);
    for (int h = 0; h < n_heads; ++h) {
        const int off = h * d_k;
        for (int qi = 0; qi < n_q; ++qi) {
            const float* Qrow = Q.data() + static_cast<size_t>(qi) * d_embed + off;
            float mx = -std::numeric_limits<float>::max();
            for (int ki = 0; ki < n_kv; ++ki) {
                const float* Krow = K.data() + static_cast<size_t>(ki) * d_embed + off;
                float s = 0.f;
                for (int c = 0; c < d_k; ++c) s += Qrow[c] * Krow[c];
                s *= scale;
                scores[ki] = s;
                if (s > mx) mx = s;
            }
            float denom = 0.f;
            for (int ki = 0; ki < n_kv; ++ki) {
                scores[ki] = std::exp(scores[ki] - mx);
                denom += scores[ki];
            }
            denom = std::max(denom, 1e-9f);
            float* out_row = concat.data() + static_cast<size_t>(qi) * d_embed + off;
            for (int ki = 0; ki < n_kv; ++ki) {
                const float w = scores[ki] / denom;
                attn[(static_cast<size_t>(h) * n_q + qi) * n_kv + ki] = w;
                const float* Vrow = V.data() + static_cast<size_t>(ki) * d_embed + off;
                for (int c = 0; c < d_k; ++c) out_row[c] += w * Vrow[c];
            }
        }
    }
    for (int r = 0; r < n_q; ++r)
        Wo.forward(concat.data() + static_cast<size_t>(r) * d_embed, out + static_cast<size_t>(r) * d_embed);

    if (cache) {
        cache->n_q = n_q; cache->n_kv = n_kv;
        cache->Q = std::move(Q); cache->K = std::move(K); cache->V = std::move(V);
        cache->attn = std::move(attn);
        cache->q_in.assign(queries, queries + static_cast<size_t>(n_q) * d_embed);
        cache->kv_in.assign(keys_values, keys_values + static_cast<size_t>(n_kv) * d_embed);
        // concat (pre-Wo) is recomputable from Q/K/V/attn during backward, so
        // it is not separately cached — see backward().
    }
}

void FaithfulMAPDPSolver::MHA::Grad::resize_for(const MHA& m) {
    dWq.assign(m.Wq.W.size(), 0.f); dbq.assign(m.Wq.b.size(), 0.f);
    dWk.assign(m.Wk.W.size(), 0.f); dbk.assign(m.Wk.b.size(), 0.f);
    dWv.assign(m.Wv.W.size(), 0.f); dbv.assign(m.Wv.b.size(), 0.f);
    dWo.assign(m.Wo.W.size(), 0.f); dbo.assign(m.Wo.b.size(), 0.f);
}
void FaithfulMAPDPSolver::MHA::Grad::zero() {
    std::fill(dWq.begin(), dWq.end(), 0.f); std::fill(dbq.begin(), dbq.end(), 0.f);
    std::fill(dWk.begin(), dWk.end(), 0.f); std::fill(dbk.begin(), dbk.end(), 0.f);
    std::fill(dWv.begin(), dWv.end(), 0.f); std::fill(dbv.begin(), dbv.end(), 0.f);
    std::fill(dWo.begin(), dWo.end(), 0.f); std::fill(dbo.begin(), dbo.end(), 0.f);
}
void FaithfulMAPDPSolver::MHA::Grad::accumulate(const Grad& o) {
    accum_into(dWq, o.dWq); accum_into(dbq, o.dbq);
    accum_into(dWk, o.dWk); accum_into(dbk, o.dbk);
    accum_into(dWv, o.dWv); accum_into(dbv, o.dbv);
    accum_into(dWo, o.dWo); accum_into(dbo, o.dbo);
}

// Scaled dot-product attention backward (per head), then unwind the four
// linear projections. Standard derivation:
//   O = A V,  A = softmax(S),  S = Q K^T / sqrt(dk)
//   dV = A^T dO
//   dA = dO V^T
//   dS_ij = A_ij * (dA_ij - sum_l A_il dA_il)          (softmax Jacobian)
//   dQ = dS K / sqrt(dk),  dK = dS^T Q / sqrt(dk)
void FaithfulMAPDPSolver::MHA::backward(const Cache& c, const float* d_out,
                                         float* d_q, float* d_kv, Grad& g) const {
    const int n_q = c.n_q, n_kv = c.n_kv;
    // 1. Unwind Wo: d_concat = Wo^T d_out ; accumulate dWo/dbo.
    std::vector<float> d_concat(static_cast<size_t>(n_q) * d_embed, 0.f);
    std::vector<float> concat(static_cast<size_t>(n_q) * d_embed, 0.f);
    // Recompute concat (pre-Wo attention output) from the cache — cheap
    // relative to re-deriving Q/K/V, and keeps Cache smaller.
    {
        const float scale = 1.f / std::sqrt(static_cast<float>(d_k));
        (void)scale;
        for (int h = 0; h < n_heads; ++h) {
            const int off = h * d_k;
            for (int qi = 0; qi < n_q; ++qi) {
                float* row = concat.data() + static_cast<size_t>(qi) * d_embed + off;
                for (int ki = 0; ki < n_kv; ++ki) {
                    const float w = c.attn[(static_cast<size_t>(h) * n_q + qi) * n_kv + ki];
                    const float* Vrow = c.V.data() + static_cast<size_t>(ki) * d_embed + off;
                    for (int cc = 0; cc < d_k; ++cc) row[cc] += w * Vrow[cc];
                }
            }
        }
    }
    for (int r = 0; r < n_q; ++r)
        Wo.backward(concat.data() + static_cast<size_t>(r) * d_embed,
                    d_out + static_cast<size_t>(r) * d_embed,
                    d_concat.data() + static_cast<size_t>(r) * d_embed,
                    g.dWo.data(), g.dbo.data());

    // 2. Per-head attention backward → dQ, dK, dV (packed d_embed-wide).
    std::vector<float> dQ(static_cast<size_t>(n_q)  * d_embed, 0.f);
    std::vector<float> dK(static_cast<size_t>(n_kv) * d_embed, 0.f);
    std::vector<float> dV(static_cast<size_t>(n_kv) * d_embed, 0.f);
    const float scale = 1.f / std::sqrt(static_cast<float>(d_k));
    std::vector<float> dA(n_kv), Arow(n_kv);

    for (int h = 0; h < n_heads; ++h) {
        const int off = h * d_k;
        for (int qi = 0; qi < n_q; ++qi) {
            const float* dOrow = d_concat.data() + static_cast<size_t>(qi) * d_embed + off;
            const float* Qrow  = c.Q.data() + static_cast<size_t>(qi) * d_embed + off;
            float dot_sum = 0.f;
            for (int ki = 0; ki < n_kv; ++ki) {
                const float* Vrow = c.V.data() + static_cast<size_t>(ki) * d_embed + off;
                float dv = 0.f;
                for (int cc = 0; cc < d_k; ++cc) dv += dOrow[cc] * Vrow[cc];
                dA[ki] = dv;
                Arow[ki] = c.attn[(static_cast<size_t>(h) * n_q + qi) * n_kv + ki];
                dot_sum += Arow[ki] * dA[ki];
            }
            for (int ki = 0; ki < n_kv; ++ki) {
                const float dS = Arow[ki] * (dA[ki] - dot_sum);   // softmax Jacobian
                float* dQrow = dQ.data() + static_cast<size_t>(qi) * d_embed + off;
                float* dKrow = dK.data() + static_cast<size_t>(ki) * d_embed + off;
                const float* Krow = c.K.data() + static_cast<size_t>(ki) * d_embed + off;
                for (int cc = 0; cc < d_k; ++cc) {
                    dQrow[cc] += dS * scale * Krow[cc];
                    dKrow[cc] += dS * scale * Qrow[cc];
                }
                // dV accumulation (outside the ki loop above would need Arow
                // again — cheaper to fold in here since Arow[ki] is in hand).
                float* dVrow = dV.data() + static_cast<size_t>(ki) * d_embed + off;
                for (int cc = 0; cc < d_k; ++cc) dVrow[cc] += Arow[ki] * dOrow[cc];
            }
        }
    }

    // 3. Unwind Wq/Wk/Wv into d_q / d_kv (accumulate — caller decides
    //    whether q and kv share a buffer, i.e. self-attention).
    for (int r = 0; r < n_q; ++r)
        Wq.backward(c.q_in.data() + static_cast<size_t>(r) * d_embed,
                   dQ.data() + static_cast<size_t>(r) * d_embed,
                   d_q ? d_q + static_cast<size_t>(r) * d_embed : nullptr,
                   g.dWq.data(), g.dbq.data());
    for (int r = 0; r < n_kv; ++r) {
        Wk.backward(c.kv_in.data() + static_cast<size_t>(r) * d_embed,
                   dK.data() + static_cast<size_t>(r) * d_embed,
                   d_kv ? d_kv + static_cast<size_t>(r) * d_embed : nullptr,
                   g.dWk.data(), g.dbk.data());
        Wv.backward(c.kv_in.data() + static_cast<size_t>(r) * d_embed,
                   dV.data() + static_cast<size_t>(r) * d_embed,
                   d_kv ? d_kv + static_cast<size_t>(r) * d_embed : nullptr,
                   g.dWv.data(), g.dbv.data());
    }
}

void FaithfulMAPDPSolver::MHA::adam_step(const Grad& g, float lr) {
    Wq.adam_step(g.dWq.data(), g.dbq.data(), lr);
    Wk.adam_step(g.dWk.data(), g.dbk.data(), lr);
    Wv.adam_step(g.dWv.data(), g.dbv.data(), lr);
    Wo.adam_step(g.dWo.data(), g.dbo.data(), lr);
}

// ════════════════════════════════════════════════════════════════════════════
// Part 2/3 — EncoderLayer (MHA+skip+BN, FF+skip+BN — eq.8-9), Encoder init
// (eq.7 branches), shared Decoder init (eq.14-17), joint Critic init.
// ════════════════════════════════════════════════════════════════════════════

void FaithfulMAPDPSolver::EncoderLayer::init(const HParams& hp, std::mt19937& rng) {
    mha.init(hp.d_embed, hp.n_heads, rng, 1.f);
    bn1.init(hp.d_embed);
    bn2.init(hp.d_embed);
    ff1.init(hp.d_embed, hp.d_ff,    rng, 1.4142135f);   // ReLU gain
    ff2.init(hp.d_ff,    hp.d_embed, rng, 1.f);
}

// h: n × d_embed, updated in place. Mirrors eq.8-9:
//   ĥ = BN(h + MHA(h,h,h));   h' = BN(ĥ + FF(ĥ))
void FaithfulMAPDPSolver::EncoderLayer::forward(float* h, int n, bool training,
                                                 Cache* cache) {
    const int d = mha.d_embed;
    std::vector<float> h_in_copy;
    if (cache) h_in_copy.assign(h, h + static_cast<size_t>(n) * d);

    std::vector<float> mha_out(static_cast<size_t>(n) * d);
    mha.forward(h, n, h, n, mha_out.data(), cache ? &cache->mha_c : nullptr);

    std::vector<float> bn1_in(static_cast<size_t>(n) * d);
    for (size_t i = 0; i < bn1_in.size(); ++i) bn1_in[i] = h[i] + mha_out[i];
    std::vector<float> bn1_out(static_cast<size_t>(n) * d);
    bn1.forward(bn1_in.data(), bn1_out.data(), n, training);

    std::vector<float> ff1_pre(static_cast<size_t>(n) * ff1.out);
    std::vector<float> ff1_out(static_cast<size_t>(n) * ff1.out);
    for (int r = 0; r < n; ++r) {
        ff1.forward(bn1_out.data() + static_cast<size_t>(r) * d,
                   ff1_pre.data() + static_cast<size_t>(r) * ff1.out);
        for (int c = 0; c < ff1.out; ++c)
            ff1_out[static_cast<size_t>(r) * ff1.out + c] =
                relu(ff1_pre[static_cast<size_t>(r) * ff1.out + c]);
    }
    std::vector<float> ff2_out(static_cast<size_t>(n) * d);
    for (int r = 0; r < n; ++r)
        ff2.forward(ff1_out.data() + static_cast<size_t>(r) * ff1.out,
                   ff2_out.data() + static_cast<size_t>(r) * d);

    std::vector<float> bn2_in(static_cast<size_t>(n) * d);
    for (size_t i = 0; i < bn2_in.size(); ++i) bn2_in[i] = bn1_out[i] + ff2_out[i];
    bn2.forward(bn2_in.data(), h, n, training);   // final output written into h

    if (cache) {
        cache->h_in    = std::move(h_in_copy);
        cache->mha_out = std::move(mha_out);
        cache->bn1_in  = std::move(bn1_in);
        cache->bn1_out = bn1_out;
        cache->ff1_in  = bn1_out;
        cache->ff1_pre = std::move(ff1_pre);
        cache->ff1_out = std::move(ff1_out);
        cache->ff2_out = std::move(ff2_out);
        cache->bn2_in  = std::move(bn2_in);
    }
}

void FaithfulMAPDPSolver::EncoderLayer::Grad::resize_for(const EncoderLayer& l) {
    mha_g.resize_for(l.mha);
    dgamma1.assign(l.bn1.dim, 0.f); dbeta1.assign(l.bn1.dim, 0.f);
    dgamma2.assign(l.bn2.dim, 0.f); dbeta2.assign(l.bn2.dim, 0.f);
    dW_ff1.assign(l.ff1.W.size(), 0.f); db_ff1.assign(l.ff1.b.size(), 0.f);
    dW_ff2.assign(l.ff2.W.size(), 0.f); db_ff2.assign(l.ff2.b.size(), 0.f);
}
void FaithfulMAPDPSolver::EncoderLayer::Grad::zero() {
    mha_g.zero();
    std::fill(dgamma1.begin(), dgamma1.end(), 0.f); std::fill(dbeta1.begin(), dbeta1.end(), 0.f);
    std::fill(dgamma2.begin(), dgamma2.end(), 0.f); std::fill(dbeta2.begin(), dbeta2.end(), 0.f);
    std::fill(dW_ff1.begin(), dW_ff1.end(), 0.f);   std::fill(db_ff1.begin(), db_ff1.end(), 0.f);
    std::fill(dW_ff2.begin(), dW_ff2.end(), 0.f);   std::fill(db_ff2.begin(), db_ff2.end(), 0.f);
}
void FaithfulMAPDPSolver::EncoderLayer::Grad::accumulate(const Grad& o) {
    mha_g.accumulate(o.mha_g);
    accum_into(dgamma1, o.dgamma1); accum_into(dbeta1, o.dbeta1);
    accum_into(dgamma2, o.dgamma2); accum_into(dbeta2, o.dbeta2);
    accum_into(dW_ff1, o.dW_ff1);   accum_into(db_ff1, o.db_ff1);
    accum_into(dW_ff2, o.dW_ff2);   accum_into(db_ff2, o.db_ff2);
}

// d_h (n×d_embed): gradient w.r.t. the layer's OUTPUT on entry, overwritten
// with the gradient w.r.t. the layer's INPUT on return (so encoder layers
// chain by calling backward() in reverse order on the same buffer).
void FaithfulMAPDPSolver::EncoderLayer::backward(const Cache& c, int n,
                                                  float* d_h, Grad& g) const {
    const int d = mha.d_embed;
    // d_h currently holds dL/d(bn2 output).
    std::vector<float> d_bn2_in(static_cast<size_t>(n) * d, 0.f);
    bn2.backward(c.bn2_in.data(), d_h, d_bn2_in.data(), n,
                g.dgamma2.data(), g.dbeta2.data());
    // bn2_in = bn1_out + ff2_out → gradient splits identically to both.
    std::vector<float> d_bn1_out = d_bn2_in;
    std::vector<float> d_ff2_out = d_bn2_in;

    std::vector<float> d_ff1_out(static_cast<size_t>(n) * ff1.out, 0.f);
    for (int r = 0; r < n; ++r)
        ff2.backward(c.ff1_out.data() + static_cast<size_t>(r) * ff1.out,
                    d_ff2_out.data() + static_cast<size_t>(r) * d,
                    d_ff1_out.data() + static_cast<size_t>(r) * ff1.out,
                    g.dW_ff2.data(), g.db_ff2.data());
    std::vector<float> d_ff1_pre(static_cast<size_t>(n) * ff1.out);
    for (size_t i = 0; i < d_ff1_pre.size(); ++i)
        d_ff1_pre[i] = d_ff1_out[i] * relu_grad(c.ff1_pre[i]);
    std::vector<float> d_ff1_in(static_cast<size_t>(n) * d, 0.f);
    for (int r = 0; r < n; ++r)
        ff1.backward(c.ff1_in.data() + static_cast<size_t>(r) * d,
                    d_ff1_pre.data() + static_cast<size_t>(r) * ff1.out,
                    d_ff1_in.data() + static_cast<size_t>(r) * d,
                    g.dW_ff1.data(), g.db_ff1.data());
    for (size_t i = 0; i < d_bn1_out.size(); ++i) d_bn1_out[i] += d_ff1_in[i];

    std::vector<float> d_bn1_in(static_cast<size_t>(n) * d, 0.f);
    bn1.backward(c.bn1_in.data(), d_bn1_out.data(), d_bn1_in.data(), n,
                g.dgamma1.data(), g.dbeta1.data());
    // bn1_in = h_in + mha_out → gradient splits to both.
    std::vector<float> d_h_skip = d_bn1_in;
    std::vector<float> d_mha_out = d_bn1_in;

    // Self-attention: query input AND key/value input are the SAME buffer
    // (h_in), so both contributions land in the same d_h_in accumulator.
    std::vector<float> d_h_in(static_cast<size_t>(n) * d, 0.f);
    mha.backward(c.mha_c, d_mha_out.data(), d_h_in.data(), d_h_in.data(), g.mha_g);
    for (size_t i = 0; i < d_h_in.size(); ++i) d_h_in[i] += d_h_skip[i];

    std::copy(d_h_in.begin(), d_h_in.end(), d_h);
}

void FaithfulMAPDPSolver::EncoderLayer::adam_step(const Grad& g, float lr) {
    mha.adam_step(g.mha_g, lr);
    bn1.adam_step(g.dgamma1.data(), g.dbeta1.data(), lr);
    bn2.adam_step(g.dgamma2.data(), g.dbeta2.data(), lr);
    ff1.adam_step(g.dW_ff1.data(), g.db_ff1.data(), lr);
    ff2.adam_step(g.dW_ff2.data(), g.db_ff2.data(), lr);
}

// ── Encoder (eq.7: input projection branches) ───────────────────────────────
// W0 — depot branch in the paper; repurposed here as the AGENT LIVE-POSITION
// encoder (a coordinate with demand=0 is exactly what the paper's depot is:
// a reference point with no pairing and no demand — see class comment).
// Wp — pickup branch, INPUT = [x_i,y_i,d_i , x_{i+N},y_{i+N},d_{i+N}] (paired
// with its delivery, eq.7 concatenation).
// Wd — delivery-of-an-in-flight-task branch (no pending pickup to pair with).
void FaithfulMAPDPSolver::Encoder::init(const HParams& hp, std::mt19937& rng) {
    W0.init(3,     hp.d_embed, rng, 1.f);   // [x, y, demand=0]
    Wp.init(6,     hp.d_embed, rng, 1.f);   // [x,y,d]_pickup ++ [x,y,d]_delivery
    Wd.init(3,     hp.d_embed, rng, 1.f);   // [x, y, demand]
    layers.resize(static_cast<size_t>(hp.n_layers));
    for (auto& l : layers) l.init(hp, rng);
}

// ── Decoder (shared θ — see class comment on why decoders are pooled
// across agents rather than one independent set per agent index) ──────────
//
// Comm^t (eq.13) is a fixed-width CONCATENATION of every agent's
// [h_cur; C] in the paper (fine for their K∈{2,5,10}). Our fleet size K
// varies per scenario (up to ~150 agents) — a fixed concatenation would
// force ctx_proj's input width to scale with a hard-coded agent cap (tens
// of millions of parameters at that size). Adaptation: mean-pool
// [h_cur(k); C_k] over the fleet instead (permutation-invariant, any K —
// the same style of fix this codebase already applies to MovementPolicy's
// variable-arity incident-edge set).
void FaithfulMAPDPSolver::Decoder::init(const HParams& hp, std::mt19937& rng) {
    glimpse.init(hp.d_embed, hp.n_heads, rng, 1.f);
    // context input = [h̄ ; h_cur(k) ; C_k(1) ; mean_pool(Comm)(d_embed+1)]
    const int ctx_in = hp.d_embed        // h̄
                      + hp.d_embed       // h_cur(k)
                      + 1                // C_k
                      + (hp.d_embed + 1);// mean-pooled Comm
    ctx_proj.init(ctx_in, hp.d_embed, rng, 1.f);
    Wq_ptr.init(hp.d_embed, hp.d_k, rng, 1.f);
    Wk_ptr.init(hp.d_embed, hp.d_k, rng, 1.f);
}

void FaithfulMAPDPSolver::Decoder::Grad::resize_for(const Decoder& d) {
    glimpse_g.resize_for(d.glimpse);
    dW_ctx.assign(d.ctx_proj.W.size(), 0.f); db_ctx.assign(d.ctx_proj.b.size(), 0.f);
    dW_qp.assign(d.Wq_ptr.W.size(), 0.f);    db_qp.assign(d.Wq_ptr.b.size(), 0.f);
    dW_kp.assign(d.Wk_ptr.W.size(), 0.f);    db_kp.assign(d.Wk_ptr.b.size(), 0.f);
}
void FaithfulMAPDPSolver::Decoder::Grad::zero() {
    glimpse_g.zero();
    std::fill(dW_ctx.begin(), dW_ctx.end(), 0.f); std::fill(db_ctx.begin(), db_ctx.end(), 0.f);
    std::fill(dW_qp.begin(), dW_qp.end(), 0.f);   std::fill(db_qp.begin(), db_qp.end(), 0.f);
    std::fill(dW_kp.begin(), dW_kp.end(), 0.f);   std::fill(db_kp.begin(), db_kp.end(), 0.f);
}
void FaithfulMAPDPSolver::Decoder::Grad::accumulate(const Grad& o) {
    glimpse_g.accumulate(o.glimpse_g);
    accum_into(dW_ctx, o.dW_ctx); accum_into(db_ctx, o.db_ctx);
    accum_into(dW_qp, o.dW_qp);   accum_into(db_qp, o.db_qp);
    accum_into(dW_kp, o.dW_kp);   accum_into(db_kp, o.db_kp);
}
void FaithfulMAPDPSolver::Decoder::adam_step(const Grad& g, float lr) {
    glimpse.adam_step(g.glimpse_g, lr);
    ctx_proj.adam_step(g.dW_ctx.data(), g.db_ctx.data(), lr);
    Wq_ptr.adam_step(g.dW_qp.data(), g.db_qp.data(), lr);
    Wk_ptr.adam_step(g.dW_kp.data(), g.db_kp.data(), lr);
}

// ── Critic (ω) ───────────────────────────────────────────────────────────
void FaithfulMAPDPSolver::Critic::init(const HParams& hp, std::mt19937& rng) {
    proj.init(hp.d_embed, hp.d_critic, rng, 1.f);
    fc1.init(hp.d_critic, hp.d_critic, rng, 1.4142135f);
    fc2.init(hp.d_critic, 1,           rng, 1.f);
}
void FaithfulMAPDPSolver::Critic::Grad::resize_for(const Critic& c) {
    dW_proj.assign(c.proj.W.size(), 0.f); db_proj.assign(c.proj.b.size(), 0.f);
    dW_fc1.assign(c.fc1.W.size(), 0.f);   db_fc1.assign(c.fc1.b.size(), 0.f);
    dW_fc2.assign(c.fc2.W.size(), 0.f);   db_fc2.assign(c.fc2.b.size(), 0.f);
}
void FaithfulMAPDPSolver::Critic::Grad::zero() {
    std::fill(dW_proj.begin(), dW_proj.end(), 0.f); std::fill(db_proj.begin(), db_proj.end(), 0.f);
    std::fill(dW_fc1.begin(), dW_fc1.end(), 0.f);   std::fill(db_fc1.begin(), db_fc1.end(), 0.f);
    std::fill(dW_fc2.begin(), dW_fc2.end(), 0.f);   std::fill(db_fc2.begin(), db_fc2.end(), 0.f);
}
void FaithfulMAPDPSolver::Critic::Grad::accumulate(const Grad& o) {
    accum_into(dW_proj, o.dW_proj); accum_into(db_proj, o.db_proj);
    accum_into(dW_fc1, o.dW_fc1);   accum_into(db_fc1, o.db_fc1);
    accum_into(dW_fc2, o.dW_fc2);   accum_into(db_fc2, o.db_fc2);
}
void FaithfulMAPDPSolver::Critic::adam_step(const Grad& g, float lr) {
    proj.adam_step(g.dW_proj.data(), g.db_proj.data(), lr);
    fc1.adam_step(g.dW_fc1.data(), g.db_fc1.data(), lr);
    fc2.adam_step(g.dW_fc2.data(), g.db_fc2.data(), lr);
}

// ════════════════════════════════════════════════════════════════════════════
// Part 3/3 — solver logic: local ENU projection, BPR A* (mirrors CA.cpp),
// the online decoding session (paired context embedding → encoder →
// cooperative decoder + fleet handler → committed routes), the cooperative
// A2C update (eq.18-21), edge-by-edge movement (mirrors CA.cpp/HAPC.cpp),
// and checkpoint I/O.
// ════════════════════════════════════════════════════════════════════════════

namespace {

// One accepted decode step, cached for the backward pass.
struct DecodeStep {
    int agent_idx = -1;
    int chosen = -1;                    // index into DecodeTrace::backlog
    std::vector<int>   cand;            // feasible backlog indices this round
    std::vector<float> probs;           // softmax over `cand` (same order)
    FaithfulMAPDPSolver::MHA::Cache glimpse_cache;
    std::vector<float> ctx_in;          // ctx_proj input  [h̄;h_cur;C;Comm]
    std::vector<float> ctx_out;         // ctx_proj output = glimpse query
    std::vector<float> g_kt;            // glimpse output
    std::vector<float> q_ptr;           // Wq_ptr(g_kt)
};

}  // namespace

// Forward-declared (by elaborated-type-specifier) in MAPDP.hpp; defined here
// since both run_decoding_session and train_step_from_session — its only
// users — live in this translation unit.
struct DecodeTrace {
    std::vector<FaithfulMAPDPSolver::BacklogEntry> backlog;
    std::vector<float> h0;                              // n×d_embed, pre-encoder (eq.7 output)
    std::vector<FaithfulMAPDPSolver::EncoderLayer::Cache> enc_caches;  // size = n_layers
    std::vector<float> h_final;                         // n×d_embed, post-encoder
    std::vector<float> hbar;                             // d_embed, mean over h_final
    std::vector<float> K_ptr;                            // n_task×d_k (task entries only)
    std::vector<int>   task_pos;                         // backlog idx of each K_ptr row

    std::vector<DecodeStep> steps;
    std::vector<float> critic_input;    // Σ_steps Σ_j p_j h_j   (d_embed)
    std::vector<float> critic_fc1_pre, critic_fc1_out;    // d_critic each
    std::vector<float> critic_proj_out;                   // d_critic
    float value = 0.f;

    std::unordered_map<int, float> agent_return;   // Σ -distance over accepted legs
};

// ── ENU-ish local projection (episode bbox centroid as origin) ─────────────
void FaithfulMAPDPSolver::project_xy(osmium::object_id_type node,
                                     float& x, float& y) const {
    x = y = 0.f;
    if (!ctx_ || !ctx_->geo_box) return;
    auto it = ctx_->geo_box->data.nodes.find(node);
    if (it == ctx_->geo_box->data.nodes.end()) return;
    constexpr double kMetersPerDegLat = 111320.0;
    const double lat_rad = proj_lat0_ * 3.14159265358979323846 / 180.0;
    const double meters_per_deg_lon = kMetersPerDegLat * std::cos(lat_rad);
    x = static_cast<float>((it->second.lon - proj_lon0_) * meters_per_deg_lon);
    y = static_cast<float>((it->second.lat - proj_lat0_) * kMetersPerDegLat);
}

// ── BPR-aware A* (identical structure to FaithfulCASolver::bpr_a_star) ─────
FaithfulMAPDPSolver::BPRPath
FaithfulMAPDPSolver::bpr_a_star(osmium::object_id_type from,
                                osmium::object_id_type to, int start_step) const {
    BPRPath result;
    if (!ctx_ || !ctx_->geo_box || from == 0 || to == 0) return result;
    if (from == to) { result.valid = true; result.nodes = {from}; return result; }

    const auto& nodes = ctx_->geo_box->data.nodes;
    const auto& ways  = ctx_->geo_box->data.ways;
    const float speed = std::max(0.1f, ctx_->speed_mps);
    auto end_it = nodes.find(to);
    if (end_it == nodes.end()) return result;

    auto h_to_goal = [&](osmium::object_id_type n) -> float {
        return graph_search::haversine_between(*ctx_->geo_box, n, to) / speed;
    };
    struct OpenEntry {
        float f, g; int t_arrive; osmium::object_id_type node;
        bool operator>(const OpenEntry& o) const { return f > o.f; }
    };
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;
    std::unordered_map<osmium::object_id_type, float> g_score;
    std::unordered_map<osmium::object_id_type,
        std::pair<osmium::object_id_type, osmium::object_id_type>> came_from;
    std::unordered_set<osmium::object_id_type> closed;

    g_score[from] = 0.f;
    open.push({h_to_goal(from), 0.f, start_step, from});
    while (!open.empty()) {
        OpenEntry cur = open.top(); open.pop();
        if (closed.count(cur.node)) continue;
        closed.insert(cur.node);
        if (cur.node == to) {
            std::vector<osmium::object_id_type> rn, re;
            osmium::object_id_type cn = to; rn.push_back(cn);
            while (cn != from) {
                auto it = came_from.find(cn);
                if (it == came_from.end()) break;
                re.push_back(it->second.second); cn = it->second.first; rn.push_back(cn);
            }
            std::reverse(rn.begin(), rn.end()); std::reverse(re.begin(), re.end());
            result.nodes = std::move(rn); result.edges = std::move(re);
            result.trip_time = cur.g; result.valid = true;
            return result;
        }
        auto nit = nodes.find(cur.node);
        if (nit == nodes.end()) continue;
        for (auto edge_id : nit->second.incident_ways) {
            auto wit = ways.find(edge_id);
            if (wit == ways.end()) continue;
            const auto& w = wit->second;
            const osmium::object_id_type nbr = (w.node1_id == cur.node) ? w.node2_id : w.node1_id;
            if (nbr == 0 || closed.count(nbr)) continue;
            const float length_m = w.distance_meters;
            if (length_m <= 0.f) continue;
            const float base_time = length_m / speed;
            const float adj = ctx_->congestion_map
                ? ctx_->congestion_map->adjusted_cost(edge_id, base_time, length_m, cur.t_arrive)
                : base_time;
            const float tentative_g = cur.g + adj;
            auto gs = g_score.find(nbr);
            if (gs != g_score.end() && tentative_g >= gs->second) continue;
            g_score[nbr] = tentative_g;
            came_from[nbr] = {cur.node, edge_id};
            const int new_t = cur.t_arrive + std::max(1, static_cast<int>(std::ceil(adj)));
            open.push({tentative_g + h_to_goal(nbr), tentative_g, new_t, nbr});
        }
    }
    return result;
}

// ── Lifecycle ────────────────────────────────────────────────────────────

void FaithfulMAPDPSolver::init(const SolverContext& ctx) {
    ctx_ = &ctx;
    if (encoder_.layers.empty()) {     // first use: build the networks
        encoder_.init(hparams, rng_);
        decoder_.init(hparams, rng_);
        critic_.init(hparams, rng_);
    }

    // Episode-level gradient accumulator reset (see the member comment in
    // MAPDP.hpp): each episode starts a fresh accumulation; flush_training_
    // update() (called from finalize()) applies and clears it. Also covers
    // the case where the PREVIOUS run() ended without a clean finalize().
    accum_dg_.resize_for(decoder_); accum_dg_.zero();
    accum_cg_.resize_for(critic_);  accum_cg_.zero();
    accum_lg_.resize(hparams.n_layers);
    for (int l = 0; l < hparams.n_layers; ++l) { accum_lg_[l].resize_for(encoder_.layers[l]); accum_lg_[l].zero(); }
    accum_sessions_ = 0;

    agents_.clear();
    agents_.reserve(static_cast<size_t>(ctx.n_active_agents));
    for (int i = 0; i < ctx.n_active_agents; ++i) {
        AgentState a;
        a.current_node = (i < static_cast<int>(ctx.agent_start_nodes.size()))
                        ? ctx.agent_start_nodes[i] : 0;
        a.capacity = (i < static_cast<int>(ctx.per_agent_capacity.size()) &&
                     !ctx.per_agent_capacity.empty())
                   ? ctx.per_agent_capacity[i]
                   : std::max(1, ctx.max_capacity_per_agent);
        agents_.push_back(std::move(a));
    }

    // Local projection origin = mean lat/lon of the graph's nodes (cheap,
    // stable proxy for the episode bbox centroid).
    proj_lat0_ = proj_lon0_ = 0.0;
    if (ctx.geo_box && !ctx.geo_box->data.nodes.empty()) {
        double sum_lat = 0.0, sum_lon = 0.0;
        for (const auto& [id, pt] : ctx.geo_box->data.nodes) { sum_lat += pt.lat; sum_lon += pt.lon; }
        proj_lat0_ = sum_lat / ctx.geo_box->data.nodes.size();
        proj_lon0_ = sum_lon / ctx.geo_box->data.nodes.size();
    }

    pending_task_ids_.clear();
    tasks_.clear();
    appeared_ = completed_ = refused_ = 0;
    latency_sum_ = wait_sum_ = trip_sum_ = 0;
    road_pd_sum_ = 0.0; road_pd_count_ = 0;
    active_steps_sum_ = 0; wait_count_ = 0;
    capacity_violations_ = pairing_violations_ = 0;
    next_session_step_ = 0;
    instr_.init(ctx.n_active_agents, ctx.speed_mps);
}

void FaithfulMAPDPSolver::inject_task(const ScheduledTask& task, int step) {
    TaskRecord r;
    r.task_id       = static_cast<int>(tasks_.size());
    r.pickup_node   = task.pickup_node_id;
    r.delivery_node = task.delivery_node_id;
    r.arrival_step  = step;
    {
        const auto& ways = ctx_->geo_box->data.ways;
        const auto edges = graph_search::shortest_path_edges(*ctx_->geo_box, r.pickup_node, r.delivery_node);
        for (auto eid : edges) {
            auto it = ways.find(eid);
            if (it != ways.end()) r.pd_road_dist += it->second.distance_meters;
        }
    }
    tasks_.push_back(r);
    pending_task_ids_.push_back(r.task_id);
    ++appeared_;
}

// ── Backlog construction (eq.7 input side) ─────────────────────────────────
std::vector<FaithfulMAPDPSolver::BacklogEntry> FaithfulMAPDPSolver::build_backlog() const {
    std::vector<BacklogEntry> backlog;
    int budget = hparams.max_backlog_nodes;
    // Oldest-arrived pending pickups first (FIFO fairness — adaptation, no
    // paper analogue: the paper's instance is fully known upfront).
    for (int tid : pending_task_ids_) {
        if (budget <= 0) break;
        const TaskRecord& t = tasks_[tid];
        BacklogEntry pu; pu.node_id = t.pickup_node; pu.task_id = tid;
        pu.is_pickup = true; pu.available = true;
        BacklogEntry de; de.node_id = t.delivery_node; de.task_id = tid;
        de.is_pickup = false; de.available = false;   // unlocked once its pickup is decoded
        backlog.push_back(pu); backlog.push_back(de);
        --budget;
    }
    // In-flight deliveries (pickup already done) — available immediately.
    for (const auto& a : agents_) {
        for (int tid : a.in_flight_task_ids) {
            if (budget <= 0) break;
            const TaskRecord& t = tasks_[tid];
            BacklogEntry de; de.node_id = t.delivery_node; de.task_id = tid;
            de.is_pickup = false; de.available = true;
            backlog.push_back(de);
            --budget;
        }
    }
    // Agent live-position pseudo-nodes (repurposed depot branch).
    for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
        BacklogEntry ap; ap.node_id = agents_[i].current_node;
        ap.is_agent_pos = true; ap.agent_idx = i; ap.available = true;
        backlog.push_back(ap);
    }
    return backlog;
}

// ── Decoding session ────────────────────────────────────────────────────────
void FaithfulMAPDPSolver::run_decoding_session(int step) {
    if (!ctx_) return;
    auto trace = std::make_unique<DecodeTrace>();
    trace->backlog = build_backlog();
    const int n = static_cast<int>(trace->backlog.size());
    if (n == 0 || pending_task_ids_.empty()) return;
    const int d = hparams.d_embed;
    const bool training = train_mode && hparams.lr > 0.f;

    // eq.7: paired context embedding per node.
    trace->h0.assign(static_cast<size_t>(n) * d, 0.f);
    std::unordered_map<int, int> pickup_pos_of_task;   // task_id -> backlog idx (pickups only)
    for (int i = 0; i < n; ++i) {
        const auto& e = trace->backlog[i];
        if (e.is_pickup) pickup_pos_of_task[e.task_id] = i;
    }
    for (int i = 0; i < n; ++i) {
        const auto& e = trace->backlog[i];
        float x, y; project_xy(e.node_id, x, y);
        if (e.is_agent_pos) {
            const float in3[3] = {x, y, 0.f};
            encoder_.W0.forward(in3, trace->h0.data() + static_cast<size_t>(i) * d);
        } else if (e.is_pickup) {
            // paired with its delivery entry (adjacent by construction).
            const auto& de = trace->backlog[i + 1];
            float dx, dy; project_xy(de.node_id, dx, dy);
            const float in6[6] = {x, y, 1.f, dx, dy, -1.f};
            encoder_.Wp.forward(in6, trace->h0.data() + static_cast<size_t>(i) * d);
        } else {
            const float in3[3] = {x, y, -1.f};
            encoder_.Wd.forward(in3, trace->h0.data() + static_cast<size_t>(i) * d);
        }
    }

    // eq.8-9: L self-attention layers, in place.
    trace->h_final = trace->h0;
    trace->enc_caches.resize(static_cast<size_t>(hparams.n_layers));
    for (int l = 0; l < hparams.n_layers; ++l)
        encoder_.layers[l].forward(trace->h_final.data(), n, training,
                                   training ? &trace->enc_caches[l] : nullptr);

    trace->hbar.assign(d, 0.f);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < d; ++c) trace->hbar[c] += trace->h_final[static_cast<size_t>(i) * d + c];
    for (int c = 0; c < d; ++c) trace->hbar[c] /= static_cast<float>(n);

    // Cache K_ptr once per session (eq.15's K side depends only on h_j).
    for (int i = 0; i < n; ++i)
        if (!trace->backlog[i].is_agent_pos) trace->task_pos.push_back(i);
    const int n_task = static_cast<int>(trace->task_pos.size());
    trace->K_ptr.assign(static_cast<size_t>(n_task) * hparams.d_k, 0.f);
    for (int r = 0; r < n_task; ++r)
        decoder_.Wk_ptr.forward(trace->h_final.data() + static_cast<size_t>(trace->task_pos[r]) * d,
                               trace->K_ptr.data() + static_cast<size_t>(r) * hparams.d_k);

    // ── Synchronous multi-agent decoding (eq.14-17) + fleet handler ────────
    std::vector<int>   virtual_pos(agents_.size());     // backlog idx of "current location"
    std::vector<int>   virtual_load(agents_.size());
    std::vector<bool>  agent_done(agents_.size(), false);
    for (int k = 0; k < static_cast<int>(agents_.size()); ++k) {
        virtual_load[k] = static_cast<int>(agents_[k].in_flight_task_ids.size());
        for (int i = 0; i < n; ++i)
            if (trace->backlog[i].is_agent_pos && trace->backlog[i].agent_idx == k) virtual_pos[k] = i;
    }
    std::vector<bool> visited(n, false);
    for (int i = 0; i < n; ++i) if (trace->backlog[i].is_agent_pos) visited[i] = true;
    std::vector<bool> available(n);
    for (int i = 0; i < n; ++i) available[i] = trace->backlog[i].available;

    std::vector<std::vector<std::pair<osmium::object_id_type,int>>> decoded(agents_.size());
    // decoded[k] = ordered (node_id, task_id) stops with sign encoded via a
    // parallel is_pickup vector below.
    std::vector<std::vector<bool>> decoded_is_pickup(agents_.size());

    std::bernoulli_distribution coin(0.5);
    for (int round = 0; round < n; ++round) {   // hard cap: at most n rounds
        struct Proposal { int agent; int idx; };
        std::vector<Proposal> proposals;

        for (int k = 0; k < static_cast<int>(agents_.size()); ++k) {
            if (agent_done[k]) continue;
            std::vector<int> cand;
            for (int i = 0; i < n_task; ++i) {
                const int j = trace->task_pos[i];
                if (visited[j] || !available[j]) continue;
                const auto& e = trace->backlog[j];
                if (e.is_pickup && virtual_load[k] >= agents_[k].capacity) continue;
                cand.push_back(j);
            }
            if (cand.empty()) { agent_done[k] = true; continue; }

            // Comm^t: mean-pool [h_cur(k'); C_k'] over the fleet (adaptation —
            // paper concatenates, we pool for variable-K; see class comment).
            // C_k = REMAINING capacity (paper's literal quantity, eq. after
            // "Transition": C_k^{t+1} = C_k^t − d_{I_k^t}, so it DECREASES on
            // pickup and increases on delivery) — NOT onboard load, which
            // moves the opposite way and, unlike C_k, doesn't by itself tell
            // the network how much room is left without also knowing the
            // agent's ceiling.
            std::vector<float> comm_pool(d + 1, 0.f);
            for (int kk = 0; kk < static_cast<int>(agents_.size()); ++kk) {
                const float* hc = trace->h_final.data() + static_cast<size_t>(virtual_pos[kk]) * d;
                for (int c = 0; c < d; ++c) comm_pool[c] += hc[c];
                comm_pool[d] += static_cast<float>(agents_[kk].capacity - virtual_load[kk]);
            }
            for (float& v : comm_pool) v /= static_cast<float>(agents_.size());

            DecodeStep st;
            st.agent_idx = k;
            st.ctx_in.reserve(static_cast<size_t>(d) * 2 + 1 + comm_pool.size());
            st.ctx_in.insert(st.ctx_in.end(), trace->hbar.begin(), trace->hbar.end());
            st.ctx_in.insert(st.ctx_in.end(),
                trace->h_final.begin() + static_cast<size_t>(virtual_pos[k]) * d,
                trace->h_final.begin() + static_cast<size_t>(virtual_pos[k]) * d + d);
            st.ctx_in.push_back(static_cast<float>(agents_[k].capacity - virtual_load[k]));
            st.ctx_in.insert(st.ctx_in.end(), comm_pool.begin(), comm_pool.end());

            st.ctx_out.assign(d, 0.f);
            decoder_.ctx_proj.forward(st.ctx_in.data(), st.ctx_out.data());

            st.g_kt.assign(d, 0.f);
            // eq.14: glimpse attends over ALL n_task nodes (h_1..h_2N in the
            // paper), UNMASKED — Mask^t (eq.17) applies only at the final
            // pointer softmax below. (An earlier version of this code wrongly
            // restricted the glimpse's keys/values to the round's feasible
            // `cand` set — fixed to match the paper: `st.cand` still gates
            // the pointer step, `kv` here spans every task entry.)
            std::vector<float> kv(static_cast<size_t>(n_task) * d);
            for (int i = 0; i < n_task; ++i)
                std::copy_n(trace->h_final.data() + static_cast<size_t>(trace->task_pos[i]) * d, d,
                           kv.data() + static_cast<size_t>(i) * d);
            decoder_.glimpse.forward(st.ctx_out.data(), 1, kv.data(),
                                     n_task, st.g_kt.data(),
                                     training ? &st.glimpse_cache : nullptr);

            st.q_ptr.assign(hparams.d_k, 0.f);
            decoder_.Wq_ptr.forward(st.g_kt.data(), st.q_ptr.data());

            // eq.16-17: masked pointer logits (mask = restrict to `cand`).
            const float scale = 1.f / std::sqrt(static_cast<float>(hparams.d_k));
            std::vector<float> logits(cand.size());
            float mx = -std::numeric_limits<float>::max();
            for (size_t c = 0; c < cand.size(); ++c) {
                const int row = static_cast<int>(std::find(trace->task_pos.begin(),
                    trace->task_pos.end(), cand[c]) - trace->task_pos.begin());
                float z = 0.f;
                const float* Kr = trace->K_ptr.data() + static_cast<size_t>(row) * hparams.d_k;
                for (int cc = 0; cc < hparams.d_k; ++cc) z += st.q_ptr[cc] * Kr[cc];
                const float u = hparams.clip_D * std::tanh(z * scale);
                logits[c] = u;
                mx = std::max(mx, u);
            }
            float denom = 0.f;
            std::vector<float> probs(cand.size());
            for (size_t c = 0; c < cand.size(); ++c) { probs[c] = std::exp(logits[c] - mx); denom += probs[c]; }
            for (float& p : probs) p /= std::max(denom, 1e-9f);

            int choice = 0;
            if (training) {
                std::discrete_distribution<int> dd(probs.begin(), probs.end());
                choice = dd(rng_);
            } else {
                choice = static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
            }
            st.cand = cand; st.probs = probs; st.chosen = cand[choice];
            proposals.push_back({k, st.chosen});
            if (training) trace->steps.push_back(std::move(st));
        }
        if (proposals.empty()) break;

        // Fleet handler: one random winner per contested backlog index.
        std::unordered_map<int, std::vector<int>> by_node;
        for (const auto& p : proposals) by_node[p.idx].push_back(p.agent);
        for (auto& [node_idx, contenders] : by_node) {
            int winner = contenders[0];
            if (contenders.size() > 1) {
                std::uniform_int_distribution<size_t> pick(0, contenders.size() - 1);
                winner = contenders[pick(rng_)];
            }
            visited[node_idx] = true;
            const auto& e = trace->backlog[node_idx];
            decoded[winner].push_back({e.node_id, e.task_id});
            decoded_is_pickup[winner].push_back(e.is_pickup);
            if (e.is_pickup) {
                ++virtual_load[winner];
                if (pickup_pos_of_task.count(e.task_id)) {
                    const int deliv_idx = pickup_pos_of_task[e.task_id] + 1;
                    available[deliv_idx] = true;
                }
            } else {
                --virtual_load[winner];
            }
            virtual_pos[winner] = node_idx;
        }
    }

    // ── Translate decoded sequences into committed road-network routes ────
    for (int k = 0; k < static_cast<int>(agents_.size()); ++k) {
        if (decoded[k].empty()) continue;
        AgentState& a = agents_[k];
        // Replace the agent's whole remaining sequence (re-plan on every
        // accepted session, matching the paper's per-step re-decision).
        a.task_queue.clear();
        a.active_is_pickup_leg = true;
        std::vector<osmium::object_id_type> stops;
        for (size_t s = 0; s < decoded[k].size(); ++s) {
            const int tid = decoded[k][s].second;
            const bool is_pu = decoded_is_pickup[k][s];
            if (is_pu) { tasks_[tid].assigned_agent = k;
                auto it = std::find(pending_task_ids_.begin(), pending_task_ids_.end(), tid);
                if (it != pending_task_ids_.end()) pending_task_ids_.erase(it);
            }
            a.task_queue.push_back(tid);
            stops.push_back(decoded[k][s].first);
        }
        // Reward (eq.1 analogue): -static road distance per accepted leg.
        osmium::object_id_type from = a.current_node;
        float ret = 0.f;
        for (auto to : stops) {
            const auto edges = graph_search::shortest_path_edges(*ctx_->geo_box, from, to);
            float dist = 0.f;
            const auto& ways = ctx_->geo_box->data.ways;
            for (auto eid : edges) { auto it = ways.find(eid); if (it != ways.end()) dist += it->second.distance_meters; }
            ret -= dist;
            from = to;
        }
        trace->agent_return[k] = ret;

        if (a.next_idx >= static_cast<int>(a.current_path_edges.size()))
            { a.current_path_nodes.clear(); a.current_path_edges.clear(); a.next_idx = 0; }
        recommit_route(a, step);
    }

    // ── Critic forward (Σ_steps Σ_j p_j h_j, projected → 2 dense layers) ───
    if (training && !trace->steps.empty()) {
        trace->critic_input.assign(d, 0.f);
        for (auto& st : trace->steps)
            for (size_t c = 0; c < st.cand.size(); ++c) {
                const float* hj = trace->h_final.data() + static_cast<size_t>(st.cand[c]) * d;
                for (int cc = 0; cc < d; ++cc) trace->critic_input[cc] += st.probs[c] * hj[cc];
            }
        trace->critic_proj_out.assign(hparams.d_critic, 0.f);
        critic_.proj.forward(trace->critic_input.data(), trace->critic_proj_out.data());
        trace->critic_fc1_pre.assign(hparams.d_critic, 0.f);
        critic_.fc1.forward(trace->critic_proj_out.data(), trace->critic_fc1_pre.data());
        trace->critic_fc1_out.assign(hparams.d_critic, 0.f);
        for (int i = 0; i < hparams.d_critic; ++i) trace->critic_fc1_out[i] = relu(trace->critic_fc1_pre[i]);
        float v = 0.f;
        critic_.fc2.forward(trace->critic_fc1_out.data(), &v);
        trace->value = v;

        train_step_from_session(*trace);
    }
}

// ── Cooperative A2C update (eq.18-21) ───────────────────────────────────────
// One session = one on-policy "trajectory" (see class comment: the paper is
// itself ambiguous about whether V is per-timestep or per-trajectory; we
// resolve it as per-session, i.e. Monte-Carlo return over the session's
// accepted decode steps minus a single session-level critic baseline).
// SIMPLIFICATION (documented, not a paper mechanism): teammates' state
// reaching a step's context only through the mean-pooled Comm vector is
// treated as detached (stop-gradient) — ctx_proj's WEIGHTS still receive a
// correct gradient from it, but that gradient is not propagated further back
// into other agents' encoder embeddings. The dominant path (via h̄ and the
// deciding agent's own h_cur) is not affected. See the accompanying message
// for the reasoning.
//
// NOTE: this computes gradients but does NOT apply them — it ACCUMULATES
// into accum_dg_/accum_cg_/accum_lg_. The actual Adam step happens once per
// episode, in flush_training_update() (called from finalize()), to match
// MAPPO/IPPO/MAPPER's one-PPO-update-per-episode cadence (and the MAPDP
// paper's own training granularity — eq.18-21 describe one full-trajectory
// backprop, not one per intermediate decision).
void FaithfulMAPDPSolver::train_step_from_session(const DecodeTrace& trace) {
    if (trace.steps.empty()) return;
    const int d = hparams.d_embed;

    float R = 0.f;
    for (const auto& [k, ret] : trace.agent_return) R += ret;
    const float A = R - trace.value;   // eq.18, session-level advantage

    // ── Critic gradient: dV(A^2)/dV = -2A (eq.21) ──────────────────────────
    Critic::Grad cg; cg.resize_for(critic_); cg.zero();
    float dV = -2.f * A;
    std::vector<float> d_fc1_out(hparams.d_critic, 0.f);
    critic_.fc2.backward(trace.critic_fc1_out.data(), &dV, d_fc1_out.data(),
                        cg.dW_fc2.data(), cg.db_fc2.data());
    std::vector<float> d_fc1_pre(hparams.d_critic, 0.f);
    for (int i = 0; i < hparams.d_critic; ++i)
        d_fc1_pre[i] = d_fc1_out[i] * relu_grad(trace.critic_fc1_pre[i]);
    std::vector<float> d_proj_out(hparams.d_critic, 0.f);
    critic_.fc1.backward(trace.critic_proj_out.data(), d_fc1_pre.data(), d_proj_out.data(),
                        cg.dW_fc1.data(), cg.db_fc1.data());
    std::vector<float> d_critic_input(d, 0.f);
    critic_.proj.backward(trace.critic_input.data(), d_proj_out.data(), d_critic_input.data(),
                         cg.dW_proj.data(), cg.db_proj.data());
    accum_cg_.accumulate(cg);

    // ── Per-node embedding gradient accumulator (n × d_embed) ──────────────
    const int n = static_cast<int>(trace.backlog.size());
    std::vector<float> d_h(static_cast<size_t>(n) * d, 0.f);
    std::vector<float> d_hbar(d, 0.f);

    Decoder::Grad dg; dg.resize_for(decoder_); dg.zero();

    for (const auto& st : trace.steps) {
        const int k_agent = st.agent_idx;
        (void)k_agent;
        // ── Actor term: dlogit_j = A*(p_j - δ_{j,chosen}) (eq.19) ──────────
        // ── + Critic term via dp_j = dot(d_critic_input, h_j), pushed
        //    through the softmax Jacobian (this step's local softmax over
        //    `cand`, since the critic input used p_j from THIS distribution).
        std::vector<float> dp(st.cand.size(), 0.f);
        for (size_t c = 0; c < st.cand.size(); ++c) {
            const float* hj = trace.h_final.data() + static_cast<size_t>(st.cand[c]) * d;
            float dot = 0.f;
            for (int cc = 0; cc < d; ++cc) dot += d_critic_input[cc] * hj[cc];
            dp[c] = dot;
            // Also route the critic's gradient into h_j directly (expected
            // embedding = Σ p_j h_j ⇒ d(expected)/dh_j = p_j).
            for (int cc = 0; cc < d; ++cc)
                d_h[static_cast<size_t>(st.cand[c]) * d + cc] += st.probs[c] * d_critic_input[cc];
        }
        float dp_dot_p = 0.f;
        for (size_t c = 0; c < st.cand.size(); ++c) dp_dot_p += dp[c] * st.probs[c];

        std::vector<float> dlogit(st.cand.size(), 0.f);
        for (size_t c = 0; c < st.cand.size(); ++c) {
            const float delta = (st.cand[c] == st.chosen) ? 1.f : 0.f;
            const float actor_term  = A * (st.probs[c] - delta);
            const float critic_term = st.probs[c] * (dp[c] - dp_dot_p);   // softmax Jacobian
            dlogit[c] = actor_term + critic_term;
        }

        // ── Unwind eq.16 tanh clip → dz_j → dQ_ptr, dK_ptr[j] ──────────────
        const float scale = 1.f / std::sqrt(static_cast<float>(hparams.d_k));
        std::vector<float> dq_ptr(hparams.d_k, 0.f);
        for (size_t c = 0; c < st.cand.size(); ++c) {
            const int row = static_cast<int>(std::find(trace.task_pos.begin(), trace.task_pos.end(),
                                             st.cand[c]) - trace.task_pos.begin());
            const float* Kr = trace.K_ptr.data() + static_cast<size_t>(row) * hparams.d_k;
            float z = 0.f;
            for (int cc = 0; cc < hparams.d_k; ++cc) z += st.q_ptr[cc] * Kr[cc];
            const float u = hparams.clip_D * std::tanh(z * scale);
            const float dz = dlogit[c] * (hparams.clip_D * hparams.clip_D - u * u) / hparams.clip_D * scale;
            // d(tanh(z*scale))/dz = scale*(1-tanh^2); dlogit->dz = clip_D*that.
            for (int cc = 0; cc < hparams.d_k; ++cc) {
                dq_ptr[cc] += dz * Kr[cc];
                // dK_ptr[j] accumulated straight into d_h via Wk_ptr backward
                // below (one row at a time — cheap, n_task rows total).
            }
            std::vector<float> dk_row(hparams.d_k, 0.f);
            for (int cc = 0; cc < hparams.d_k; ++cc) dk_row[cc] = dz * st.q_ptr[cc];
            decoder_.Wk_ptr.backward(
                trace.h_final.data() + static_cast<size_t>(st.cand[c]) * d, dk_row.data(),
                d_h.data() + static_cast<size_t>(st.cand[c]) * d,
                dg.dW_kp.data(), dg.db_kp.data());
        }
        std::vector<float> d_gkt(d, 0.f);
        decoder_.Wq_ptr.backward(st.g_kt.data(), dq_ptr.data(), d_gkt.data(),
                                dg.dW_qp.data(), dg.db_qp.data());

        // ── Unwind glimpse MHA (eq.14, now over the FULL n_task set — see
        //    the forward-pass fix note) → d(ctx_out) [query] + d(h_j) for
        //    every task node, scattered via trace.task_pos (not st.cand).
        std::vector<float> d_ctx_out(d, 0.f);
        const int n_task_glimpse = st.glimpse_cache.n_kv;
        std::vector<float> d_kv(static_cast<size_t>(n_task_glimpse) * d, 0.f);
        decoder_.glimpse.backward(st.glimpse_cache, d_gkt.data(), d_ctx_out.data(),
                                 d_kv.data(), dg.glimpse_g);
        for (int i = 0; i < n_task_glimpse; ++i)
            for (int cc = 0; cc < d; ++cc)
                d_h[static_cast<size_t>(trace.task_pos[i]) * d + cc] += d_kv[static_cast<size_t>(i) * d + cc];

        // ── Unwind ctx_proj → d(h̄), d(h_cur(k)), d(Comm — stop-gradient) ──
        std::vector<float> d_ctx_in(st.ctx_in.size(), 0.f);
        decoder_.ctx_proj.backward(st.ctx_in.data(), d_ctx_out.data(), d_ctx_in.data(),
                                  dg.dW_ctx.data(), dg.db_ctx.data());
        for (int cc = 0; cc < d; ++cc) d_hbar[cc] += d_ctx_in[cc];
        // ctx_in layout: [hbar(d) | h_cur(d) | C_k(1) | comm_pool(d+1)] —
        // h_cur's slice starts at offset d; Comm's stop-gradient means we
        // simply don't propagate d_ctx_in[d+1+d .. end] anywhere (see class
        // comment simplification note).
    }

    accum_dg_.accumulate(dg);

    // Distribute d_hbar (mean over ALL n nodes) uniformly, then backprop
    // through the L encoder layers in reverse (eq.20's shared-φ gradient).
    for (int i = 0; i < n; ++i)
        for (int cc = 0; cc < d; ++cc)
            d_h[static_cast<size_t>(i) * d + cc] += d_hbar[cc] / static_cast<float>(n);

    std::vector<EncoderLayer::Grad> layer_grads(hparams.n_layers);
    for (int l = 0; l < hparams.n_layers; ++l) { layer_grads[l].resize_for(encoder_.layers[l]); layer_grads[l].zero(); }
    for (int l = hparams.n_layers - 1; l >= 0; --l)
        encoder_.layers[l].backward(trace.enc_caches[l], n, d_h.data(), layer_grads[l]);
    // eq.20: 1/K normalisation of the shared-encoder gradient (K = full
    // fleet size, matching the paper's literal constant).
    const float inv_k = 1.f / std::max(1, static_cast<int>(agents_.size()));
    for (int l = 0; l < hparams.n_layers; ++l) {
        for (float& v : layer_grads[l].mha_g.dWq) v *= inv_k; for (float& v : layer_grads[l].mha_g.dbq) v *= inv_k;
        for (float& v : layer_grads[l].mha_g.dWk) v *= inv_k; for (float& v : layer_grads[l].mha_g.dbk) v *= inv_k;
        for (float& v : layer_grads[l].mha_g.dWv) v *= inv_k; for (float& v : layer_grads[l].mha_g.dbv) v *= inv_k;
        for (float& v : layer_grads[l].mha_g.dWo) v *= inv_k; for (float& v : layer_grads[l].mha_g.dbo) v *= inv_k;
        for (float& v : layer_grads[l].dgamma1) v *= inv_k; for (float& v : layer_grads[l].dbeta1) v *= inv_k;
        for (float& v : layer_grads[l].dgamma2) v *= inv_k; for (float& v : layer_grads[l].dbeta2) v *= inv_k;
        for (float& v : layer_grads[l].dW_ff1) v *= inv_k;  for (float& v : layer_grads[l].db_ff1) v *= inv_k;
        for (float& v : layer_grads[l].dW_ff2) v *= inv_k;  for (float& v : layer_grads[l].db_ff2) v *= inv_k;
        accum_lg_[l].accumulate(layer_grads[l]);
    }
    ++accum_sessions_;
    // Note: W0/Wp/Wd (eq.7 input branches) are NOT updated here — they sit
    // upstream of the (already-consumed) d_h0 gradient. A production pass
    // should also backprop d_h (post first-layer's contribution to d_h_in)
    // through these branches; left as a follow-up since W0/Wp/Wd are a small
    // fraction of the total parameter count and the L self-attention layers
    // dominate representational capacity.
}

// Applies the episode's ACCUMULATED gradient (summed over every decoding
// session, averaged by session count) in one Adam step per network — the
// "one PPO update per episode" analogue for MAPDP. No-op if no session
// trained this episode (e.g. an episode with no tasks) or train_mode=false.
void FaithfulMAPDPSolver::flush_training_update() {
    if (accum_sessions_ <= 0) return;
    const float inv_n = 1.f / static_cast<float>(accum_sessions_);
    auto scale = [inv_n](std::vector<float>& v) { for (float& x : v) x *= inv_n; };

    scale(accum_cg_.dW_proj); scale(accum_cg_.db_proj);
    scale(accum_cg_.dW_fc1);  scale(accum_cg_.db_fc1);
    scale(accum_cg_.dW_fc2);  scale(accum_cg_.db_fc2);
    critic_.adam_step(accum_cg_, hparams.lr);

    scale(accum_dg_.glimpse_g.dWq); scale(accum_dg_.glimpse_g.dbq);
    scale(accum_dg_.glimpse_g.dWk); scale(accum_dg_.glimpse_g.dbk);
    scale(accum_dg_.glimpse_g.dWv); scale(accum_dg_.glimpse_g.dbv);
    scale(accum_dg_.glimpse_g.dWo); scale(accum_dg_.glimpse_g.dbo);
    scale(accum_dg_.dW_ctx); scale(accum_dg_.db_ctx);
    scale(accum_dg_.dW_qp);  scale(accum_dg_.db_qp);
    scale(accum_dg_.dW_kp);  scale(accum_dg_.db_kp);
    decoder_.adam_step(accum_dg_, hparams.lr);

    for (int l = 0; l < hparams.n_layers; ++l) {
        auto& g = accum_lg_[l];
        scale(g.mha_g.dWq); scale(g.mha_g.dbq); scale(g.mha_g.dWk); scale(g.mha_g.dbk);
        scale(g.mha_g.dWv); scale(g.mha_g.dbv); scale(g.mha_g.dWo); scale(g.mha_g.dbo);
        scale(g.dgamma1); scale(g.dbeta1); scale(g.dgamma2); scale(g.dbeta2);
        scale(g.dW_ff1); scale(g.db_ff1); scale(g.dW_ff2); scale(g.db_ff2);
        encoder_.layers[l].adam_step(g, hparams.lr);
    }

    accum_dg_.zero(); accum_cg_.zero();
    for (auto& g : accum_lg_) g.zero();
    accum_sessions_ = 0;
}

// ── Movement (identical structure to FaithfulCASolver) ─────────────────────

int FaithfulMAPDPSolver::edge_arrival_step(osmium::object_id_type edge_id, int t_enter) {
    if (!ctx_) return t_enter + 1;
    const auto& ways = ctx_->geo_box->data.ways;
    auto it = ways.find(edge_id);
    if (it == ways.end()) return t_enter + 1;
    const float length_m = it->second.distance_meters;
    const float base_time = length_m / std::max(0.1f, ctx_->speed_mps);
    if (!ctx_->congestion_map)
        return t_enter + std::max(1, static_cast<int>(std::ceil(base_time)));
    const int self_w = std::max(1, ctx_->congestion_map->params.load_per_agent);
    const float adj = ctx_->congestion_map->adjusted_cost(edge_id, base_time, length_m, t_enter, self_w);
    instr_.record_edge_entry(base_time, adj, ctx_->congestion_map->get_load(edge_id, t_enter));
    return t_enter + std::max(1, static_cast<int>(std::ceil(adj)));
}

void FaithfulMAPDPSolver::fire_stop(AgentState& a, int step) {
    if (a.task_queue.empty()) return;
    const int tid = a.task_queue.front();
    if (tid < 0 || tid >= static_cast<int>(tasks_.size())) {
        a.task_queue.erase(a.task_queue.begin()); a.active_is_pickup_leg = true; return;
    }
    TaskRecord& t = tasks_[tid];
    if (a.active_is_pickup_leg) {
        t.picked_step = step;
        a.in_flight_task_ids.push_back(tid);
        if (static_cast<int>(a.in_flight_task_ids.size()) > a.capacity) ++capacity_violations_;
        wait_sum_ += (t.picked_step - t.arrival_step); ++wait_count_;
        a.active_is_pickup_leg = false;
        a.task_queue.erase(a.task_queue.begin());
        return;
    }
    if (t.picked_step < 0) ++pairing_violations_;
    t.delivered_step = step;
    latency_sum_ += (t.delivered_step - t.arrival_step);
    trip_sum_    += (t.delivered_step - std::max(t.picked_step, t.arrival_step));
    if (t.pd_road_dist > 0.f) { road_pd_sum_ += t.pd_road_dist; ++road_pd_count_; }
    ++completed_;
    instr_.record_delivery(static_cast<int>(&a - agents_.data()));
    auto it = std::find(a.in_flight_task_ids.begin(), a.in_flight_task_ids.end(), tid);
    if (it != a.in_flight_task_ids.end()) a.in_flight_task_ids.erase(it);
    a.task_queue.erase(a.task_queue.begin());
    a.active_is_pickup_leg = true;
}

void FaithfulMAPDPSolver::recommit_route(AgentState& a, int step) {
    if (!ctx_ || !ctx_->congestion_map || !ctx_->geo_box) return;
    std::vector<osmium::object_id_type> stops;
    stops.reserve(a.task_queue.size());
    bool leg_is_pickup = a.active_is_pickup_leg;
    for (int tid : a.task_queue) {
        if (tid < 0 || tid >= static_cast<int>(tasks_.size())) continue;
        const TaskRecord& t = tasks_[tid];
        stops.push_back(leg_is_pickup ? t.pickup_node : t.delivery_node);
        leg_is_pickup = true;   // only the FRONT task may be mid-delivery-leg
    }
    bool first_leg = true;
    const int tail = commit_agent_route(
        *ctx_->congestion_map, *ctx_->geo_box, ctx_->speed_mps,
        a.current_node, stops, step, a.committed_occ,
        [&](osmium::object_id_type f, osmium::object_id_type to, int t) {
            if (first_leg) {
                first_leg = false;
                if (!a.current_path_edges.empty() &&
                    a.next_idx < static_cast<int>(a.current_path_edges.size()) &&
                    f == a.current_node && to == a.current_path_nodes.back())
                    return std::vector<osmium::object_id_type>(
                        a.current_path_edges.begin() + a.next_idx, a.current_path_edges.end());
            }
            return bpr_a_star(f, to, t).edges;
        });
    (void)tail;
    if (a.current_path_edges.empty() && !stops.empty()) {
        BPRPath p = bpr_a_star(a.current_node, stops.front(), step);
        if (p.valid) { a.current_path_nodes = p.nodes; a.current_path_edges = p.edges; a.next_idx = 0;
                       a.current_edge_t_enter = step; a.arrival_step_next_node = -1; }
    }
}

void FaithfulMAPDPSolver::advance_agent(AgentState& a, int step) {
    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) return;
    if (a.arrival_step_next_node < 0)
        a.arrival_step_next_node = edge_arrival_step(a.current_path_edges[a.next_idx], step);
    if (ctx_ && ctx_->congestion_map)
        instr_.sample_route_exposure(ctx_->congestion_map->get_load(a.current_path_edges[a.next_idx], step));
    if (step < a.arrival_step_next_node) { ++active_steps_sum_; return; }
    if (ctx_) {
        const auto& ways = ctx_->geo_box->data.ways;
        auto wit = ways.find(a.current_path_edges[a.next_idx]);
        if (wit != ways.end()) instr_.record_edge_traversal(wit->second.distance_meters);
    }
    ++a.next_idx;
    a.current_node = a.current_path_nodes[a.next_idx];
    if (a.next_idx >= static_cast<int>(a.current_path_edges.size())) {
        fire_stop(a, step);
        if (!a.task_queue.empty()) recommit_route(a, step);
        else { a.current_path_nodes.clear(); a.current_path_edges.clear(); a.next_idx = 0; }
        return;
    }
    a.current_edge_t_enter = step;
    a.arrival_step_next_node = edge_arrival_step(a.current_path_edges[a.next_idx], step);
    ++active_steps_sum_;
}

void FaithfulMAPDPSolver::step(int timestep) {
    if (ctx_) {
        instr_.sample_congestion(ctx_->congestion_map);
        if (ctx_->ghost) instr_.sample_ghost(ctx_->ghost->n_active_now());
    }
    bool any_idle = false;
    for (const auto& a : agents_) if (a.task_queue.empty()) { any_idle = true; break; }
    if (timestep >= next_session_step_ && (!pending_task_ids_.empty() || any_idle)) {
        instr_.time_allocation([&]{ run_decoding_session(timestep); });
        next_session_step_ = timestep + std::max(1, hparams.replan_every_steps);
    }
    for (auto& a : agents_) advance_agent(a, timestep);
}

SolverMetrics FaithfulMAPDPSolver::finalize() {
    // One optimizer step for the whole episode, from every session's
    // accumulated gradient — see flush_training_update()'s comment. No-op
    // when train_mode was false for this run (accum_* stayed empty since
    // train_step_from_session is only reached when training).
    if (train_mode) flush_training_update();

    SolverMetrics m;
    m.tasks_appeared  = appeared_;
    m.tasks_completed = completed_;
    m.tasks_refused   = refused_;
    m.throughput_rate = (appeared_ > 0) ? std::min(1.f, static_cast<float>(completed_) / appeared_) : 0.f;
    m.accept_rate     = (appeared_ > 0) ? static_cast<float>(appeared_ - refused_) / appeared_ : 0.f;
    if (completed_ > 0) {
        m.latency_mean    = static_cast<double>(latency_sum_) / completed_;
        m.mean_trip_steps = static_cast<double>(trip_sum_)    / completed_;
    }
    if (wait_count_ > 0) m.mean_wait_steps = static_cast<double>(wait_sum_) / wait_count_;
    if (road_pd_count_ > 0) m.mean_road_pd_m = road_pd_sum_ / road_pd_count_;
    if (ctx_ && ctx_->n_active_agents > 0 && ctx_->total_steps > 0)
        m.agent_utilisation = static_cast<float>(active_steps_sum_) / (static_cast<float>(ctx_->n_active_agents) * ctx_->total_steps);
    m.capacity_violations = capacity_violations_;
    m.pairing_violations  = pairing_violations_;
    if (ctx_ && ctx_->total_steps > 0) {
        const double mean_active = static_cast<double>(active_steps_sum_) / ctx_->total_steps;
        m.latency_per_agent = m.latency_mean / std::max(1.0, mean_active);
    }
    instr_.finalize_into(m);
    return m;
}

// ── Checkpoint I/O (φ + θ + ω; Adam moments reset on load) ─────────────────
void FaithfulMAPDPSolver::reinit(uint32_t seed) {
    rng_.seed(seed);
    encoder_ = Encoder{};
    decoder_ = Decoder{};
    critic_  = Critic{};
    encoder_.init(hparams, rng_);
    decoder_.init(hparams, rng_);
    critic_.init(hparams, rng_);
}

namespace {
void write_dense(std::FILE* f, const FaithfulMAPDPSolver::Dense& d) {
    int32_t in = d.in, out = d.out;
    std::fwrite(&in, sizeof(in), 1, f); std::fwrite(&out, sizeof(out), 1, f);
    std::fwrite(d.W.data(), sizeof(float), d.W.size(), f);
    std::fwrite(d.b.data(), sizeof(float), d.b.size(), f);
}
bool read_dense(std::FILE* f, FaithfulMAPDPSolver::Dense& d) {
    int32_t in = 0, out = 0;
    if (std::fread(&in, sizeof(in), 1, f) != 1 || std::fread(&out, sizeof(out), 1, f) != 1) return false;
    if (in != d.in || out != d.out) return false;   // architecture mismatch
    return std::fread(d.W.data(), sizeof(float), d.W.size(), f) == d.W.size()
        && std::fread(d.b.data(), sizeof(float), d.b.size(), f) == d.b.size();
}
void write_bn(std::FILE* f, const FaithfulMAPDPSolver::BatchNorm1d& b) {
    std::fwrite(b.gamma.data(), sizeof(float), b.gamma.size(), f);
    std::fwrite(b.beta.data(), sizeof(float), b.beta.size(), f);
    std::fwrite(b.running_mean.data(), sizeof(float), b.running_mean.size(), f);
    std::fwrite(b.running_var.data(), sizeof(float), b.running_var.size(), f);
}
bool read_bn(std::FILE* f, FaithfulMAPDPSolver::BatchNorm1d& b) {
    return std::fread(b.gamma.data(), sizeof(float), b.gamma.size(), f) == b.gamma.size()
        && std::fread(b.beta.data(), sizeof(float), b.beta.size(), f) == b.beta.size()
        && std::fread(b.running_mean.data(), sizeof(float), b.running_mean.size(), f) == b.running_mean.size()
        && std::fread(b.running_var.data(), sizeof(float), b.running_var.size(), f) == b.running_var.size();
}
void write_mha(std::FILE* f, const FaithfulMAPDPSolver::MHA& m) {
    write_dense(f, m.Wq); write_dense(f, m.Wk); write_dense(f, m.Wv); write_dense(f, m.Wo);
}
bool read_mha(std::FILE* f, FaithfulMAPDPSolver::MHA& m) {
    return read_dense(f, m.Wq) && read_dense(f, m.Wk) && read_dense(f, m.Wv) && read_dense(f, m.Wo);
}
}  // namespace

static constexpr uint32_t kMagicMapdp = 0xDEA110D5u;

void FaithfulMAPDPSolver::save(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&kMagicMapdp, sizeof(kMagicMapdp), 1, f);
    write_dense(f, encoder_.W0); write_dense(f, encoder_.Wp); write_dense(f, encoder_.Wd);
    for (const auto& l : encoder_.layers) {
        write_mha(f, l.mha); write_bn(f, l.bn1); write_bn(f, l.bn2);
        write_dense(f, l.ff1); write_dense(f, l.ff2);
    }
    write_mha(f, decoder_.glimpse);
    write_dense(f, decoder_.ctx_proj); write_dense(f, decoder_.Wq_ptr); write_dense(f, decoder_.Wk_ptr);
    write_dense(f, critic_.proj); write_dense(f, critic_.fc1); write_dense(f, critic_.fc2);
    std::fclose(f);
}

bool FaithfulMAPDPSolver::load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != kMagicMapdp) { std::fclose(f); return false; }
    if (encoder_.layers.empty()) { encoder_.init(hparams, rng_); decoder_.init(hparams, rng_); critic_.init(hparams, rng_); }
    bool ok = read_dense(f, encoder_.W0) && read_dense(f, encoder_.Wp) && read_dense(f, encoder_.Wd);
    for (auto& l : encoder_.layers) {
        ok = ok && read_mha(f, l.mha) && read_bn(f, l.bn1) && read_bn(f, l.bn2)
                && read_dense(f, l.ff1) && read_dense(f, l.ff2);
    }
    ok = ok && read_mha(f, decoder_.glimpse)
            && read_dense(f, decoder_.ctx_proj) && read_dense(f, decoder_.Wq_ptr) && read_dense(f, decoder_.Wk_ptr);
    ok = ok && read_dense(f, critic_.proj) && read_dense(f, critic_.fc1) && read_dense(f, critic_.fc2);
    std::fclose(f);
    return ok;
}
