#ifndef SOTA_MAPDP_HPP
#define SOTA_MAPDP_HPP

#include "SoTA/SolverFramework.hpp"
#include <array>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
// Zong, Zheng, Li & Jin 2022 (AAAI) — MAPDP: cooperative multi-agent RL for
// the Pickup and Delivery Problem. Paired context embedding + a Transformer
// encoder shared across the fleet, per-agent cooperative MHA decoders with a
// communication embedding, trained via a joint-critic A2C.
//
// ── Adaptation summary (paper → this environment) ──────────────────────────
// Same class of adaptation CALGPDP already applies to HAPC/CA ("keep the
// mechanism, swap the environment-specific piece"):
//   - Node features: paired [x, y] via a local ENU projection of the graph's
//     lat/lon around the episode bbox centroid (replaces the paper's
//     synthetic 5×5 square) + demand d_i = +1 pickup / -1 delivery (our
//     tasks are single-unit; PDPTask carries no payload weight, so the
//     paper's [1,10] variable demand has no GPDP analogue here).
//   - Distances e_ij: road-network shortest path (GraphSearch A*), replacing
//     Euclidean — identical adaptation to HAPC/CA.
//   - Construction: the paper solves ONE fully-known 2N-node instance from a
//     shared depot, start to finish. Our GPDP is an online lifelong stream,
//     no depot. MAPDP is therefore re-run as a periodic joint re-planner:
//     each decoding SESSION starts from every agent's LIVE position and
//     capacity (this is exactly the paper's own state definition — remaining
//     capacity + current trajectory) and solves the currently-known backlog
//     (in-flight + unassigned tasks, capped at hparams.max_backlog_nodes) via
//     the same synchronous multi-agent decoding loop; the resulting sequence
//     is translated into committed road-network routes exactly like CA/HAPC
//     (commit_agent_route). Sessions trigger on new task arrival and whenever
//     an agent completes its current leg (idle) — the same replan cadence
//     the other standalone solvers use.
//   - Fleet handler ("several agents make the same decision to the same
//     node"): unchanged — one random winner per contested node per
//     synchronous decode step, losers stay at their current location for
//     that step (mirrors the paper exactly).
//   - Per-agent decoder parameters θ_k → SHARED across agents (θ_k = θ ∀k).
//     This is literally the paper's own MAPDP-SP ablation ("all agent
//     decoders share the same parameters... slightly outperformed [by full
//     MAPDP] and still superior to many other baselines", −0.5 to −2.4% in
//     their Table 2) — and it is REQUIRED here, not optional: our fleet size
//     K varies per scenario (agents_mult ∈ {0.7,1.0,2.5}) and per episode,
//     whereas the paper trains one fixed-K set of independent decoders per
//     experiment. A shared decoder generalises across any K; per-agent-index
//     decoders structurally cannot.
//   - Everything else — paired context embedding (eq.7), Transformer encoder
//     (eq.8-12), communication embedding (eq.13), cooperative decoder
//     (eq.14-17, clip D=10), joint critic + cooperative A2C (eq.18-21) — is
//     implemented as specified.
// ════════════════════════════════════════════════════════════════════════════
class FaithfulMAPDPSolver : public ISolver {
public:
    FaithfulMAPDPSolver() = default;
    ~FaithfulMAPDPSolver() override = default;

    // Paper hyperparameters (§ Experiments): L=3 attention layers, H=8 heads,
    // d_k=128 per head (⇒ d_embed = d_k·H = 1024, eq. following (12)), dc=128
    // critic projection, clip D=10 (eq.16), lr=1e-3, Adam. Untuned for this
    // environment beyond the two adaptation knobs at the bottom.
    struct HParams {
        int   n_layers   = 3;     // L
        int   n_heads    = 8;     // H
        int   d_k        = 128;   // per-head query/key/value width
        int   d_embed    = 1024;  // d_h = d_k · H — keep in sync with d_k/n_heads
        int   d_ff       = 512;   // feed-forward hidden width inside each encoder layer
        int   d_critic   = 128;   // dc (critic projection width)
        float clip_D     = 10.f;  // eq.(16) tanh clip — exploration bound on logits
        float lr         = 1e-3f; // Adam, β1=0.9 β2=0.999 (policy_optim defaults)

        // ── Adaptation knobs (no paper analogue, see class comment) ────────
        int max_backlog_nodes = 64;  // cap on |L| per decoding session (perf)
        int replan_every_steps = 1;  // 1 = replan-on-event only (see step())
    };
    HParams hparams;

    void          init(const SolverContext& ctx) override;
    void          inject_task(const ScheduledTask& task, int step) override;
    void          step(int timestep) override;
    SolverMetrics finalize() override;
    const char*   name() const override { return "MAPDP"; }

    // Gates the A2C update inside run_decoding_session — mirrors
    // EpisodeRunner::train_mode (a plain public bool the caller sets before
    // each run()). true = sample + learn; false = greedy argmax, no backward.
    bool train_mode = true;

    // Re-create every network from scratch with a fresh seed (mirrors
    // MappoPolicy::reinit) — one fresh MAPDP per independent training seed,
    // exactly like train_grid does for MAPPO/IPPO/MAPPER.
    void reinit(uint32_t seed);

    // Weights only (φ: encoder + paired-context; θ: shared decoder; ω: critic).
    // Adam moments are NOT persisted (reset to zero on load), matching the
    // BidPolicy checkpoint convention already used by MAPPO/IPPO/MAPPER.
    bool load(const std::string& path);
    void save(const std::string& path) const;

private:
    // ── Generic dense layer: runtime-sized, own Adam state ─────────────────
    // (PolicyKit::Mlp is a fixed 3-layer 12→64→64→1 net — not reusable for
    // the variable-width projections a Transformer needs, so MAPDP carries
    // its own small autodiff-free linear-algebra kit, built the same way:
    // hand-rolled forward + backward + Adam, no external tensor library.)
    struct Dense {
        int in = 0, out = 0;
        std::vector<float> W, b;              // W: out×in row-major
        std::vector<float> mW, vW, mb, vb;     // Adam moments
        int t = 0;

        void init(int in_dim, int out_dim, std::mt19937& rng, float gain);
        void forward(const float* x, float* y) const;   // y[out] = Wx + b
        // dL/dx accumulated into dx (may be null); dL/dW,dL/db accumulated
        // into dW,db (caller-owned scratch, zeroed by the caller per batch).
        void backward(const float* x, const float* dy,
                      float* dx, float* dW, float* db) const;
        void adam_step(const float* dW, const float* db, float lr);
    };

    // ── Running-stats BatchNorm1d (Ioffe & Szegedy 2015, eq.8-9's BN^ℓ) ────
    struct BatchNorm1d {
        int dim = 0;
        std::vector<float> gamma, beta;         // learned scale/shift
        std::vector<float> running_mean, running_var;
        std::vector<float> mgamma, vgamma, mbeta, vbeta;
        int t = 0;
        static constexpr float kMomentum = 0.1f;
        static constexpr float kEps      = 1e-5f;

        void init(int d);
        // Batch forward over `n` rows of `dim` floats (updates running stats
        // in training mode; uses running stats in inference mode).
        void forward(const float* x, float* y, int n, bool training);
        void backward(const float* x, const float* dy, float* dx, int n,
                      float* dgamma, float* dbeta) const;
        void adam_step(const float* dgamma, const float* dbeta, float lr);
    };

    // ── Multi-head attention block (eq.10-12) ──────────────────────────────
    // Self-attention (encoder: query=key=value=the node set) or cross
    // attention (decoder glimpse: 1 query row against the node set).
    //
    // Forward caches every intermediate the backward pass needs (Q, K, V and
    // the post-softmax attention weights, per head) — cheap relative to the
    // O(n²·d_embed) attention cost itself, and it is what makes MHA::backward
    // a plain replay rather than a second forward pass.
    struct MHA {
        int d_embed = 0, n_heads = 0, d_k = 0;
        Dense Wq, Wk, Wv, Wo;   // each d_embed × d_embed (Wo projects concat back)

        struct Cache {
            int n_q = 0, n_kv = 0;
            std::vector<float> Q, K, V;      // n_q×d_embed, n_kv×d_embed, n_kv×d_embed
            std::vector<float> attn;         // n_heads × n_q × n_kv (post-softmax)
            std::vector<float> q_in, kv_in;  // raw inputs (needed by Dense::backward)
        };
        struct Grad {
            std::vector<float> dWq, dbq, dWk, dbk, dWv, dbv, dWo, dbo;
            void resize_for(const MHA& m);
            void zero();
            void accumulate(const Grad& other);   // += , for episode-level accumulation
        };

        void init(int d_embed_, int n_heads_, std::mt19937& rng, float gain);
        // queries: n_q × d_embed, keys/values: n_kv × d_embed (h_1..h_2N).
        // out: n_q × d_embed. cache non-null ⇒ store activations for backward.
        void forward(const float* queries, int n_q,
                    const float* keys_values, int n_kv,
                    float* out, Cache* cache = nullptr) const;
        // d_out: n_q × d_embed. Accumulates into d_q (n_q×d_embed, may be
        // null if the query input has no upstream, e.g. it's a fresh
        // concat) and d_kv (n_kv×d_embed, accumulated — self-attention
        // callers pass the SAME buffer for d_q and d_kv and both
        // contributions land in it). Accumulates weight grads into `g`.
        void backward(const Cache& c, const float* d_out,
                     float* d_q, float* d_kv, Grad& g) const;
        void adam_step(const Grad& g, float lr);
    };

    // One encoder layer: MHA + skip + BN, then 2-layer ReLU FF + skip + BN.
    struct EncoderLayer {
        MHA          mha;
        BatchNorm1d  bn1, bn2;
        Dense        ff1, ff2;   // d_embed → d_ff → d_embed

        struct Cache {
            MHA::Cache mha_c;
            std::vector<float> h_in, mha_out, bn1_in, bn1_out;   // n×d_embed each
            std::vector<float> ff1_in, ff1_pre, ff1_out;         // n×d_embed, n×d_ff, n×d_ff
            std::vector<float> ff2_out, bn2_in;                  // n×d_embed
        };
        struct Grad {
            MHA::Grad mha_g;
            std::vector<float> dgamma1, dbeta1, dgamma2, dbeta2;
            std::vector<float> dW_ff1, db_ff1, dW_ff2, db_ff2;
            void resize_for(const EncoderLayer& l);
            void zero();
            void accumulate(const Grad& other);
        };

        void init(const HParams& hp, std::mt19937& rng);
        // in-place over `n` node embeddings (n × d_embed).
        void forward(float* h, int n, bool training, Cache* cache = nullptr);
        // d_h: n×d_embed, gradient w.r.t. this layer's OUTPUT; overwritten
        // in place with the gradient w.r.t. this layer's INPUT on return.
        void backward(const Cache& c, int n, float* d_h, Grad& g) const;
        void adam_step(const Grad& g, float lr);
    };

    // ── Encoder / paired-context embedding (eq.7-9) ────────────────────────
    struct Encoder {
        Dense W0, Wp, Wd;                 // depot / pickup(paired) / delivery branches
        std::vector<EncoderLayer> layers; // size = hparams.n_layers

        void init(const HParams& hp, std::mt19937& rng);
    };

    // ── Shared decoder (θ; per-agent-index params of the paper collapse to
    // ONE shared set — see class comment) — glimpse MHA (eq.14) + single-head
    // pointer/compatibility layer with tanh clip (eq.15-17).
    struct Decoder {
        MHA   glimpse;              // eq.14, H heads, over [h̄; h_cur; C; Comm]
        Dense ctx_proj;             // builds the query vector h_{k,(c)}^t
        Dense Wq_ptr, Wk_ptr;       // eq.15, single-head pointer projections

        struct Grad {
            MHA::Grad glimpse_g;
            std::vector<float> dW_ctx, db_ctx, dW_qp, db_qp, dW_kp, db_kp;
            void resize_for(const Decoder& d);
            void zero();
            void accumulate(const Grad& other);
        };

        void init(const HParams& hp, std::mt19937& rng);
        void adam_step(const Grad& g, float lr);
    };

    // ── Joint critic (eq.18-21 text): weighted sum of π(v)·h_i per agent,
    // concatenated/summed across the fleet, projected to dc then 2 dense
    // layers → scalar V(s).
    struct Critic {
        Dense proj;    // n_active_agents-weighted sum (d_embed) → d_critic
        Dense fc1;     // d_critic → d_critic
        Dense fc2;     // d_critic → 1

        struct Grad {
            std::vector<float> dW_proj, db_proj, dW_fc1, db_fc1, dW_fc2, db_fc2;
            void resize_for(const Critic& c);
            void zero();
            void accumulate(const Grad& other);
        };

        void init(const HParams& hp, std::mt19937& rng);
        void adam_step(const Grad& g, float lr);
    };

    Encoder encoder_;   // φ — shared by the whole fleet
    Decoder decoder_;   // θ — shared across agents (adaptation, see above)
    Critic  critic_;    // ω
    std::mt19937 rng_{0xC0FFEEu};

    // ── Episode-level gradient accumulation ─────────────────────────────────
    // MAPPO/IPPO/MAPPER each take exactly ONE PPO update per episode
    // (buffer every TAM decision during the episode, train_round() once at
    // the end). MAPDP's own paper likewise backpropagates once per full
    // problem instance solved (eq.18-21 describe ONE episode's trajectory).
    // Our lifelong adaptation runs many decoding SESSIONS per episode
    // (replanning on every arrival / idle event), so — to match that "one
    // update per episode" cadence rather than one update per session —
    // every session's gradients are ACCUMULATED here and only applied
    // (Adam step) once, in finalize(). See flush_training_update().
    Decoder::Grad accum_dg_;
    Critic::Grad  accum_cg_;
    std::vector<EncoderLayer::Grad> accum_lg_;
    int  accum_sessions_ = 0;
    void flush_training_update();

    // ── Environment-facing state (mirrors CA.hpp's AgentState/TaskRecord) ──
    struct AgentState {
        osmium::object_id_type current_node = 0;
        int capacity = 1;
        std::vector<int> in_flight_task_ids;   // picked, not yet delivered

        // Decoded sequence awaiting execution (node ids, road graph),
        // refreshed by run_decoding_session(); consumed edge-by-edge exactly
        // like CA/HAPC.
        std::vector<osmium::object_id_type> current_path_nodes;
        std::vector<osmium::object_id_type> current_path_edges;
        std::vector<int>                    task_queue;       // task ids, in visit order
        bool active_is_pickup_leg = true;
        int  next_idx               = 0;
        int  arrival_step_next_node = -1;

        CommittedOcc committed_occ;   // congestion footprint (commit_agent_route)
    };

    struct TaskRecord {
        int  task_id = -1;
        osmium::object_id_type pickup_node = 0, delivery_node = 0;
        int  arrival_step = 0, picked_step = -1, delivered_step = -1;
        int  assigned_agent = -1;
        float pd_road_dist = 0.f;
    };

    const SolverContext* ctx_ = nullptr;
    std::vector<AgentState> agents_;
    std::vector<TaskRecord> tasks_;
    std::vector<int> pending_task_ids_;   // arrived, not yet in any agent's queue

    // Local ENU projection origin (episode bbox centroid) — paper's x_i.
    double proj_lat0_ = 0.0, proj_lon0_ = 0.0;
    void   project_xy(osmium::object_id_type node, float& x, float& y) const;

    // Not cached: cost depends on start step (same convention as CA/HAPC).
    struct BPRPath {
        std::vector<osmium::object_id_type> nodes, edges;
        float trip_time = 0.f;
        bool  valid = false;
    };
    BPRPath bpr_a_star(osmium::object_id_type from, osmium::object_id_type to,
                       int start_step) const;

    // One entry of the per-session node set fed to the encoder: either a
    // real task node (pickup / delivery, is_agent_pos=false) or an agent's
    // live-position pseudo-node (is_agent_pos=true, repurposes the paper's
    // depot branch — see class comment).
    struct BacklogEntry {
        osmium::object_id_type node_id = 0;
        int  task_id = -1;      // -1 for agent pseudo-nodes
        bool is_pickup = false; // meaningful only if task_id >= 0
        bool is_agent_pos = false;
        int  agent_idx = -1;    // meaningful only if is_agent_pos
        bool available = false; // precedence gate (deliveries start locked)
    };
    // Builds the backlog (capped at hparams.max_backlog_nodes task entries,
    // oldest-arrived first — FIFO fairness; adaptation, no paper analogue)
    // plus one pseudo-node per active agent.
    std::vector<BacklogEntry> build_backlog() const;

    void          run_decoding_session(int step);
    void          train_step_from_session(const struct DecodeTrace& trace);

    int  next_session_step_ = 0;   // gates hparams.replan_every_steps cadence

    int edge_arrival_step(osmium::object_id_type edge_id, int t_enter);
    void advance_agent(AgentState& a, int step);
    void fire_stop(AgentState& a, int step);
    // Republishes an agent's full remaining sequence on the shared
    // CongestionMap (commit_agent_route) — mirrors CA/HAPC's recommit_route.
    void recommit_route(AgentState& a, int step);

    //  Bookkeeping for metrics (identical fields to CA/HAPC)
    int  appeared_ = 0, completed_ = 0, refused_ = 0;
    long latency_sum_ = 0, wait_sum_ = 0, trip_sum_ = 0;
    double road_pd_sum_ = 0.0;
    int    road_pd_count_ = 0;
    long   active_steps_sum_ = 0;
    int    wait_count_ = 0;
    int    capacity_violations_ = 0, pairing_violations_ = 0;
    SolverInstrumentation instr_;

    int edge_arrival_step(osmium::object_id_type edge_id, int t_enter);
    void advance_agent(AgentState& a, int step);
    void fire_stop(AgentState& a, int step);
};

#endif // SOTA_MAPDP_HPP
