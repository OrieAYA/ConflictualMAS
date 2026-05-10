#include "ComparisonTest.hpp"
#include "LKHSolver.hpp"
#include "IVNSSolver.hpp"
#include "DMASforPD/DeliveryAgent/LocalSolutionAgent.hpp"
#include "DMASforPD/DeliveryAgent/OperableEnvironment.hpp"
#include "DMASforPD/Utility/PDPTask.hpp"
#include "DMASforPD/Utility/ObjectiveNode.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <limits>
#define _USE_MATH_DEFINES
#include <cmath>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

// ============================================================================
// DATA TYPES
// ============================================================================

using StopCoords = std::unordered_map<std::string, std::pair<double, double>>;
using AllCoords  = std::unordered_map<std::string, StopCoords>;

struct SolverResult {
    std::vector<ObjectiveNode> sequence;
    float     cost        = std::numeric_limits<float>::max();
    float     dist_km     = -1.f;
    float     t_ratio     = -1.f;
    float     d_ratio     = -1.f;
    float     local_eff   = -1.f; // avg(min_edge / chosen_edge) ∈ (0,1]
    long long exec_ms     = 0;
    bool      valid       = false;
    int       pdp_violations = 0;
    int       unreachable    = 0;
};

struct CompResult {
    std::string route_id;
    int         num_stops  = 0;
    int         num_tasks  = 0;
    float       actual_cost    = 0.f;
    float       actual_dist_km = -1.f;
    std::string depot_code;
    SolverResult dbvns;
    SolverResult lkh;
    SolverResult ivns;
    // DbVNS-specific agent metrics
    int         anchors_tried = 0;
    int         anchors_valid = 0;
    std::string best_anchor;
    int         lkh_restarts  = 0;
};

// ============================================================================
// JSON HELPERS (self-contained)
// ============================================================================

static json cmp_load_json(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    json j; f >> j; return j;
}

static std::unordered_map<std::string, std::unordered_map<std::string, float>>
cmp_parse_travel_times(const json& jroute)
{
    std::unordered_map<std::string, std::unordered_map<std::string, float>> tt;
    for (auto& [from, tos] : jroute.items())
        for (auto& [to, val] : tos.items())
            if (val.is_number()) tt[from][to] = val.get<float>();
    return tt;
}

static std::unordered_map<std::string, int>
cmp_parse_actual_seq(const json& jseq)
{
    std::unordered_map<std::string, int> seq;
    for (auto& [stop, pos] : jseq.items())
        if (pos.is_number()) seq[stop] = static_cast<int>(pos.get<double>());
    return seq;
}

// ============================================================================
// COST HELPERS
// ============================================================================

using TT = std::unordered_map<std::string, std::unordered_map<std::string, float>>;

static float cmp_actual_cost(const std::unordered_map<std::string, int>& aseq, const TT& tt)
{
    std::vector<std::pair<int, std::string>> ord;
    ord.reserve(aseq.size());
    for (auto& [s, p] : aseq) ord.push_back({p, s});
    std::sort(ord.begin(), ord.end());
    float c = 0.f;
    for (size_t i = 0; i + 1 < ord.size(); ++i) {
        auto ia = tt.find(ord[i].second);   if (ia == tt.end()) continue;
        auto ib = ia->second.find(ord[i+1].second); if (ib == ia->second.end()) continue;
        c += ib->second;
    }
    return c;
}

static float cmp_seq_full_cost(
    const std::vector<ObjectiveNode>&                                  seq,
    const std::unordered_map<osmium::object_id_type, std::string>&    id_to_code,
    const std::string&                                                 depot_code,
    const TT&                                                          tt)
{
    if (seq.empty()) return std::numeric_limits<float>::max();
    auto travel = [&](const std::string& a, const std::string& b) -> float {
        auto ia = tt.find(a); if (ia == tt.end()) return 1e9f;
        auto ib = ia->second.find(b); return (ib == ia->second.end()) ? 1e9f : ib->second;
    };
    float cost = 0.f;
    auto it0 = id_to_code.find(seq[0].id);
    if (it0 == id_to_code.end()) return std::numeric_limits<float>::max();
    cost += travel(depot_code, it0->second);
    for (size_t i = 0; i + 1 < seq.size(); ++i) {
        auto ia = id_to_code.find(seq[i].id);
        auto ib = id_to_code.find(seq[i+1].id);
        if (ia == id_to_code.end() || ib == id_to_code.end())
            return std::numeric_limits<float>::max();
        cost += travel(ia->second, ib->second);
    }
    return cost;
}

// ============================================================================
// DISTANCE HELPERS
// ============================================================================

static double cmp_haversine(double lat1, double lon1, double lat2, double lon2)
{
    constexpr double PI = 3.14159265358979323846;
    const double R = 6371.0;
    double dlat = (lat2 - lat1) * PI / 180.0;
    double dlon = (lon2 - lon1) * PI / 180.0;
    double a = std::sin(dlat/2)*std::sin(dlat/2)
             + std::cos(lat1*PI/180.0)*std::cos(lat2*PI/180.0)
             * std::sin(dlon/2)*std::sin(dlon/2);
    return 2.0 * R * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
}

static float cmp_seq_dist(
    const std::vector<ObjectiveNode>&                               seq,
    const std::unordered_map<osmium::object_id_type, std::string>& id_to_code,
    const StopCoords&                                               coords)
{
    double total = 0.0;
    for (size_t i = 0; i + 1 < seq.size(); ++i) {
        auto ia = id_to_code.find(seq[i].id);
        auto ib = id_to_code.find(seq[i+1].id);
        if (ia == id_to_code.end() || ib == id_to_code.end()) continue;
        auto ca = coords.find(ia->second);
        auto cb = coords.find(ib->second);
        if (ca == coords.end() || cb == coords.end()) continue;
        total += cmp_haversine(ca->second.first, ca->second.second,
                               cb->second.first, cb->second.second);
    }
    return static_cast<float>(total);
}

static float cmp_actual_dist(
    const std::unordered_map<std::string, int>& aseq, const StopCoords& coords)
{
    std::vector<std::pair<int, std::string>> ord;
    ord.reserve(aseq.size());
    for (auto& [s, p] : aseq) ord.push_back({p, s});
    std::sort(ord.begin(), ord.end());
    double total = 0.0;
    for (size_t i = 0; i + 1 < ord.size(); ++i) {
        auto ca = coords.find(ord[i].second);
        auto cb = coords.find(ord[i+1].second);
        if (ca == coords.end() || cb == coords.end()) continue;
        total += cmp_haversine(ca->second.first, ca->second.second,
                               cb->second.first, cb->second.second);
    }
    return static_cast<float>(total);
}

// ── Local efficiency metric ──────────────────────────────────────────────────
// For each route edge (u→v): ratio = min_finite_cost_from_u / chosen_cost.
// Average over all edges ∈ (0,1].  1.0 = every step was the cheapest possible.
static float cmp_local_efficiency(
    const std::vector<ObjectiveNode>& seq,
    const OperableEnvironment&        env)
{
    if (seq.size() < 2) return 0.f;
    int n = env.size();
    double sum = 0.0;
    int count = 0;
    for (size_t i = 0; i + 1 < seq.size(); ++i) {
        int from = env.find_index(seq[i].id);
        int to   = env.find_index(seq[i + 1].id);
        if (from < 0 || to < 0) continue;
        float chosen = env.get_cost(from, to);
        if (chosen <= 0.f || chosen >= 1e8f) continue;
        float min_c = std::numeric_limits<float>::max();
        for (int j = 0; j < n; ++j) {
            if (j == from) continue;
            float c = env.get_cost(from, j);
            if (c > 0.f && c < 1e8f && c < min_c) min_c = c;
        }
        if (min_c == std::numeric_limits<float>::max()) continue;
        sum += static_cast<double>(min_c) / static_cast<double>(chosen);
        ++count;
    }
    return count > 0 ? static_cast<float>(sum / count) : 0.f;
}

// ============================================================================
// COORDINATE LOADING (NaN -> null preprocessing)
// ============================================================================

static AllCoords cmp_load_coords(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) return {};
    std::string raw((std::istreambuf_iterator<char>(f)), {});
    std::string cleaned; cleaned.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ) {
        if (i + 3 <= raw.size() && raw[i]=='N' && raw[i+1]=='a' && raw[i+2]=='N')
            { cleaned += "null"; i += 3; }
        else cleaned += raw[i++];
    }
    AllCoords result;
    try {
        json j = json::parse(cleaned);
        for (auto& [rid, rdata] : j.items()) {
            if (!rdata.contains("stops")) continue;
            StopCoords sc;
            for (auto& [code, sdata] : rdata["stops"].items())
                if (sdata.contains("lat") && sdata.contains("lng")
                    && sdata["lat"].is_number() && sdata["lng"].is_number())
                    sc[code] = { sdata["lat"].get<double>(), sdata["lng"].get<double>() };
            result[rid] = std::move(sc);
        }
    } catch (...) {}
    return result;
}

// ============================================================================
// ARCHITECTURE AUDIT
// ============================================================================

struct SeqAudit { int pdp = 0; int unreachable = 0; };

static SeqAudit cmp_audit(
    const std::vector<ObjectiveNode>&                                  seq,
    const PairingMap&                                                  pickup_of,
    const std::unordered_map<osmium::object_id_type, std::string>&    id_to_code,
    const TT&                                                          tt)
{
    SeqAudit a;
    if (seq.empty()) return a;
    std::unordered_map<osmium::object_id_type, int> pos;
    pos.reserve(seq.size());
    for (int i = 0; i < static_cast<int>(seq.size()); ++i) pos[seq[i].id] = i;
    for (const auto& node : seq) {
        auto it = pickup_of.find(node.id);
        if (it == pickup_of.end()) continue;
        auto pit = pos.find(it->second);
        if (pit == pos.end() || pit->second >= pos.at(node.id)) ++a.pdp;
    }
    for (size_t i = 0; i + 1 < seq.size(); ++i) {
        auto ia = id_to_code.find(seq[i].id);
        auto ib = id_to_code.find(seq[i+1].id);
        if (ia == id_to_code.end() || ib == id_to_code.end()) { ++a.unreachable; continue; }
        auto fa = tt.find(ia->second);
        if (fa == tt.end()) { ++a.unreachable; continue; }
        auto fb = fa->second.find(ib->second);
        if (fb == fa->second.end() || fb->second >= 1e8f) ++a.unreachable;
    }
    return a;
}

// ============================================================================
// ENVIRONMENT CONSTRUCTION
// ============================================================================

static std::vector<PDPTask> cmp_build_env(
    const std::vector<std::string>&                                stops,
    const TT&                                                      tt,
    OperableEnvironment&                                           env,
    PairingMap&                                                    pickup_of,
    PairingMap&                                                    delivery_of,
    std::unordered_map<std::string, osmium::object_id_type>&       code_to_id,
    std::unordered_map<osmium::object_id_type, std::string>&       id_to_code)
{
    for (size_t i = 0; i < stops.size(); ++i) {
        osmium::object_id_type id = static_cast<osmium::object_id_type>(i + 1);
        code_to_id[stops[i]] = id;
        id_to_code[id]        = stops[i];
    }
    int n_pairs = static_cast<int>(stops.size()) / 2;
    std::vector<PDPTask> tasks(n_pairs);
    for (int p = 0; p < n_pairs; ++p) {
        osmium::object_id_type pick_id = code_to_id[stops[p * 2]];
        osmium::object_id_type del_id  = code_to_id[stops[p * 2 + 1]];
        tasks[p].task_id  = p;
        tasks[p].pickup   = ObjectiveNode{pick_id,  0};
        tasks[p].delivery = ObjectiveNode{del_id,   0};
        env.add_task(tasks[p]);
        pickup_of [del_id]  = pick_id;
        delivery_of[pick_id] = del_id;
    }
    int n = env.size();
    for (int i = 0; i < n; ++i) {
        const std::string& si = id_to_code[env.nodes[i].id];
        for (int j = 0; j < n; ++j) {
            if (i == j) { env.set_cost(i, j, 0.0f); continue; }
            const std::string& sj = id_to_code[env.nodes[j].id];
            auto it = tt.find(si);
            if (it != tt.end()) {
                auto jt = it->second.find(sj);
                if (jt != it->second.end()) { env.set_cost(i, j, jt->second); continue; }
            }
            env.set_cost(i, j, 1e9f);
        }
    }
    return tasks;
}

// ============================================================================
// DBVNS RUNNER
// ============================================================================

struct DbVNSAgentStats {
    int    anchors_tried = 0;
    int    anchors_valid = 0;
    osmium::object_id_type best_anchor_id = 0;
};

static std::vector<ObjectiveNode> cmp_run_dbvns(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const PairingMap&          delivery_of,
    osmium::object_id_type     depot_id,
    const DbVNSParams&         params,
    DbVNSAgentStats&           stats)
{
    stats = {};
    // One LocalSolutionAgent per delivery node — no cap.
    std::vector<const ObjectiveNode*> anchors;
    anchors.reserve(env.nodes.size());
    for (const auto& node : env.nodes)
        if (pickup_of.find(node.id) != pickup_of.end())
            anchors.push_back(&node);
    stats.anchors_tried = static_cast<int>(anchors.size());

    std::vector<ObjectiveNode> best;
    float best_cost = std::numeric_limits<float>::max();

    for (const ObjectiveNode* np : anchors) {
        LocalSolutionAgent lsa(*np, params);
        auto cand = lsa.plan(env, pickup_of, delivery_of, depot_id);
        if (cand.empty()) continue;
        float cost = 0.f;
        for (size_t i = 0; i + 1 < cand.size(); ++i) {
            int ai = env.find_index(cand[i].id), bi = env.find_index(cand[i+1].id);
            if (ai < 0 || bi < 0) { cost = std::numeric_limits<float>::max(); break; }
            float c = env.get_cost(ai, bi);
            if (c < 0 || c >= 1e8f) { cost = std::numeric_limits<float>::max(); break; }
            cost += c;
        }
        ++stats.anchors_valid;
        if (cost < best_cost) { best_cost = cost; best = std::move(cand); stats.best_anchor_id = np->id; }
    }
    return best;
}

// ============================================================================
// SVG RENDERER
// ============================================================================

static void cmp_render_svg(
    const std::string&                                               short_id,
    const std::string&                                               solver_label,
    const std::vector<ObjectiveNode>&                               seq,
    const std::unordered_map<osmium::object_id_type, std::string>& id_to_code,
    const StopCoords&                                               coords,
    float t_ratio, long long exec_ms,
    const std::string&                                               out_path)
{
    if (seq.empty() || coords.empty()) return;
    struct Pt { double lat, lng; int pos; bool is_last; };
    std::vector<Pt> pts; pts.reserve(seq.size());
    int n = static_cast<int>(seq.size());
    for (int i = 0; i < n; ++i) {
        auto ic = id_to_code.find(seq[i].id); if (ic == id_to_code.end()) continue;
        auto co = coords.find(ic->second);    if (co == coords.end()) continue;
        pts.push_back({co->second.first, co->second.second, i, i == n - 1});
    }
    if (pts.empty()) return;

    double mn_lat = pts[0].lat, mx_lat = pts[0].lat;
    double mn_lng = pts[0].lng, mx_lng = pts[0].lng;
    for (auto& p : pts) {
        mn_lat = std::min(mn_lat, p.lat); mx_lat = std::max(mx_lat, p.lat);
        mn_lng = std::min(mn_lng, p.lng); mx_lng = std::max(mx_lng, p.lng);
    }
    double pad_lat = (mx_lat - mn_lat) * 0.08 + 1e-6;
    double pad_lng = (mx_lng - mn_lng) * 0.08 + 1e-6;
    mn_lat -= pad_lat; mx_lat += pad_lat;
    mn_lng -= pad_lng; mx_lng += pad_lng;

    const int W = 680, H = 680, LEG = 90;
    auto px = [&](double lng) { return (lng - mn_lng) / (mx_lng - mn_lng) * W; };
    auto py = [&](double lat) { return H - (lat - mn_lat) / (mx_lat - mn_lat) * H; };

    std::ofstream svg(out_path);
    svg << "<svg width=\"" << W << "\" height=\"" << (H + LEG)
        << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    svg << "<rect width=\"" << W << "\" height=\"" << H << "\" fill=\"#f5f7fa\" rx=\"4\"/>\n";

    svg << "<polyline fill=\"none\" stroke=\"#7bafd4\" stroke-width=\"1\" opacity=\"0.5\" points=\"";
    for (auto& p : pts)
        svg << std::fixed << std::setprecision(2) << px(p.lng) << "," << py(p.lat) << " ";
    svg << "\"/>\n";

    for (auto& p : pts) {
        float t = (n > 1) ? static_cast<float>(p.pos) / (n - 1) : 0.5f;
        int r = static_cast<int>(40  + t * 215);
        int g = static_cast<int>(120 - t * 50);
        int b = static_cast<int>(200 - t * 160);
        if (p.is_last) {
            svg << "<circle cx=\"" << px(p.lng) << "\" cy=\"" << py(p.lat)
                << "\" r=\"7\" fill=\"#ff6600\" stroke=\"white\" stroke-width=\"2\"/>\n";
        } else {
            svg << "<circle cx=\"" << px(p.lng) << "\" cy=\"" << py(p.lat)
                << "\" r=\"3\" fill=\"rgb(" << r << "," << g << "," << b << ")\"/>\n";
        }
    }
    if (!pts.empty())
        svg << "<circle cx=\"" << px(pts[0].lng) << "\" cy=\"" << py(pts[0].lat)
            << "\" r=\"5\" fill=\"#22bb55\" stroke=\"white\" stroke-width=\"1.5\"/>\n";

    // Legend
    svg << "<rect y=\"" << H << "\" width=\"" << W << "\" height=\"" << LEG
        << "\" fill=\"#e8ecf0\"/>\n";
    svg << "<text x=\"8\" y=\"" << (H + 16)
        << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#222\">"
        << short_id << " [" << solver_label << "]"
        << "  |  " << n << " stops"
        << "  |  t_ratio=" << std::fixed << std::setprecision(3) << t_ratio
        << "  |  " << exec_ms << " ms</text>\n";

    int bx = 8, by = H + 26, bw = 140, bh = 10;
    for (int s = 0; s < 6; ++s) {
        float t = s / 5.0f;
        int r = static_cast<int>(40+t*215), g = static_cast<int>(120-t*50), b = static_cast<int>(200-t*160);
        svg << "<rect x=\"" << (bx + s*bw/6) << "\" y=\"" << by
            << "\" width=\"" << (bw/6+1) << "\" height=\"" << bh
            << "\" fill=\"rgb(" << r << "," << g << "," << b << ")\"/>\n";
    }
    svg << "<text x=\"" << (bx+bw+4) << "\" y=\"" << (by+9)
        << "\" font-family=\"monospace\" font-size=\"11\" fill=\"#444\">"
        << "visit order: blue=early  purple=mid  red=late</text>\n";
    svg << "<circle cx=\"" << (bx+2) << "\" cy=\"" << (H+52)
        << "\" r=\"5\" fill=\"#22bb55\" stroke=\"white\" stroke-width=\"1\"/>\n";
    svg << "<text x=\"" << (bx+10) << "\" y=\"" << (H+56)
        << "\" font-family=\"monospace\" font-size=\"11\" fill=\"#444\">first stop</text>\n";
    svg << "<circle cx=\"" << (bx+72) << "\" cy=\"" << (H+52)
        << "\" r=\"7\" fill=\"#ff6600\" stroke=\"white\" stroke-width=\"1.5\"/>\n";
    svg << "<text x=\"" << (bx+82) << "\" y=\"" << (H+56)
        << "\" font-family=\"monospace\" font-size=\"11\" fill=\"#444\">last stop</text>\n";
    svg << "<text x=\"" << (bx+160) << "\" y=\"" << (H+56)
        << "\" font-family=\"monospace\" font-size=\"11\" fill=\"#888\">"
        << solver_label << "</text>\n";

    svg << "</svg>\n";
}

// ============================================================================
// PRINT HELPERS
// ============================================================================

static std::string cmp_fmt_time(float s) {
    if (s >= 1e8f) return "    N/A";
    int m = static_cast<int>(s) / 60, sec = static_cast<int>(s) % 60;
    std::ostringstream o;
    o << std::setw(3) << m << "m" << std::setw(2) << std::setfill('0') << sec << "s";
    return o.str();
}

static std::string cmp_seq_preview(
    const std::string&                                               depot,
    const std::vector<ObjectiveNode>&                               seq,
    const std::unordered_map<osmium::object_id_type, std::string>& id_to_code,
    int head = 3, int tail = 2)
{
    if (seq.empty()) return "(empty)";
    int n = static_cast<int>(seq.size());
    auto code = [&](osmium::object_id_type id) {
        auto it = id_to_code.find(id); return it != id_to_code.end() ? it->second : std::string("?");
    };
    std::ostringstream oss;
    oss << depot;
    int sh = std::min(head, n);
    for (int i = 0; i < sh; ++i) oss << " > " << code(seq[i].id);
    int mid = n - sh - tail;
    if (mid > 0) {
        oss << " > ...+" << mid << "...";
        for (int i = n - tail; i < n; ++i) oss << " > " << code(seq[i].id);
    } else {
        for (int i = sh; i < n; ++i) oss << " > " << code(seq[i].id);
    }
    return oss.str();
}

static void cmp_sep(int w = 100) { std::cout << std::string(w, '-') << "\n"; }

// ============================================================================
// MAIN TEST
// ============================================================================

void test_comparison(const std::string& data_dir)
{
    const std::string sep = data_dir.find('/') != std::string::npos ? "/" : "\\";

    const std::string tt_path     = data_dir + sep + "model_apply_inputs" + sep + "new_travel_times.json";
    const std::string actual_path = data_dir + sep + "model_score_inputs" + sep + "new_actual_sequences.json";
    const std::string coords_path = data_dir + sep + "model_apply_inputs" + sep + "new_route_data.json";

    // Results dir: src/Comparisons/DbVNS_vs_LKH/results/
    // Source tree root is 5 levels above data_dir:
    //   data_dir = .../AmazonDataset/~/.rc-cli/data  (3 dirs up = AmazonDataset/)
    //   then 2 more up = src/
    // Build path relative to the Comparisons folder next to AmazonDataset.
    std::string results_dir;
    {
        std::string d = data_dir;
        // data_dir = .../src/AmazonDataset/~/.rc-cli/data  (4 pops → src/)
        for (int i = 0; i < 4; ++i) {
            size_t p = d.find_last_of("/\\");
            if (p != std::string::npos) d = d.substr(0, p);
        }
        results_dir = d + sep + "Comparisons" + sep + "DbVNS_vs_LKH" + sep + "results";
    }
    fs::create_directories(results_dir);

    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "  DbVNS vs LKH - Amazon Last Mile Benchmark\n";
    std::cout << std::string(100, '=') << "\n\n";

    // DbVNS parameters (same as standalone test for fair comparison).
    DbVNSParams params;
    params.max_iterations     = 30;
    params.k_max              = 4;
    params.max_decompositions = 3;
    params.max_divergence     = 0.5;
    const int lkh_restarts = 10;

    std::cout << "DbVNS : max_iter=" << params.max_iterations
              << "  k_max=" << params.k_max
              << "  max_decomps=" << params.max_decompositions
              << "  (1 agent/delivery)\n";
    std::cout << "LKH   : max_restarts=" << lkh_restarts
              << "  (NN + 2-opt + or-opt  per restart)\n";
    const int ivns_iters = 50;
    const int ivns_k     = 4;
    std::cout << "IVNS  : max_iter=" << ivns_iters
              << "  max_k=" << ivns_k
              << "  (cheapest-insertion + free-zone 2-opt + PDP 2-opt)\n\n";

    // Load data
    json jtt, jactual;
    try {
        std::cout << "Loading travel times... "; std::cout.flush();
        jtt = cmp_load_json(tt_path);
        std::cout << "ok (" << jtt.size() << " routes)\n";
        std::cout << "Loading actual seqs...  "; std::cout.flush();
        jactual = cmp_load_json(actual_path);
        std::cout << "ok (" << jactual.size() << " routes)\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n"; return;
    }

    std::cout << "Loading stop coords...  "; std::cout.flush();
    AllCoords all_coords = cmp_load_coords(coords_path);
    std::cout << (all_coords.empty() ? "unavailable\n" : "ok\n") << std::flush;
    std::cout << "Results dir: " << results_dir << "\n\n" << std::flush;

    // Route selection (same as standalone: seed=42, up to 10).
    std::vector<std::string> all_ids;
    for (const auto& item : jtt.items())
        if (jactual.contains(item.key())) all_ids.push_back(item.key());
    { std::mt19937 rng(42); std::shuffle(all_ids.begin(), all_ids.end(), rng); }
    const int max_routes = std::min(10, static_cast<int>(all_ids.size()));
    all_ids.resize(max_routes);

    std::cout << "Testing " << max_routes << " routes:\n\n" << std::flush;

    std::vector<CompResult> results;

    for (const std::string& rid : all_ids) {
        std::string short_id = rid.substr(rid.rfind('_') + 1, 8);
        auto tt   = cmp_parse_travel_times(jtt.at(rid));
        auto jrsq = jactual.at(rid);
        auto aseq = cmp_parse_actual_seq(jrsq.contains("actual") ? jrsq.at("actual") : jrsq);

        std::string depot_code;
        for (auto& [s, p] : aseq) if (p == 0) { depot_code = s; break; }
        if (depot_code.empty()) {
            std::cout << "[" << short_id << "] No depot - skip\n" << std::flush; continue;
        }

        std::vector<std::string> stops;
        stops.reserve(aseq.size() - 1);
        for (auto& [s, p] : aseq) if (p != 0) stops.push_back(s);
        if (stops.size() % 2 != 0) stops.pop_back();
        int n_stops = static_cast<int>(stops.size());
        int n_tasks = n_stops / 2;
        if (n_tasks == 0) {
            std::cout << "[" << short_id << "] Too few stops - skip\n" << std::flush; continue;
        }

        float actual_cost = cmp_actual_cost(aseq, tt);

        // Build shared environment (both solvers use the same one).
        OperableEnvironment env;
        PairingMap pickup_of, delivery_of;
        std::unordered_map<std::string, osmium::object_id_type> code_to_id;
        std::unordered_map<osmium::object_id_type, std::string> id_to_code;
        auto tasks = cmp_build_env(stops, tt, env, pickup_of, delivery_of, code_to_id, id_to_code);
        osmium::object_id_type depot_id =
            static_cast<osmium::object_id_type>(stops.size() + 1);

        // Coordinates for this route.
        const StopCoords* rc = nullptr;
        if (!all_coords.empty()) {
            auto it = all_coords.find(rid);
            if (it != all_coords.end()) rc = &it->second;
        }

        float actual_km = rc ? cmp_actual_dist(aseq, *rc) : -1.f;

        std::cout << "[" << short_id << "] "
                  << std::setw(3) << n_stops << " stops, "
                  << std::setw(3) << n_tasks << " tasks"
                  << "  |  actual: " << cmp_fmt_time(actual_cost) << "\n" << std::flush;

        // ── Run DbVNS ────────────────────────────────────────────────────────
        std::cout << "  DbVNS running...  " << std::flush;
        DbVNSAgentStats vns_stats;
        auto vt0 = std::chrono::high_resolution_clock::now();
        auto vseq = cmp_run_dbvns(env, pickup_of, delivery_of, depot_id,
                                   params, vns_stats);
        auto vt1 = std::chrono::high_resolution_clock::now();
        long long v_ms = std::chrono::duration_cast<std::chrono::milliseconds>(vt1 - vt0).count();
        std::cout << "done (" << v_ms << " ms)\n" << std::flush;

        SolverResult dbvns_r;
        dbvns_r.valid    = !vseq.empty() && static_cast<int>(vseq.size()) == n_stops;
        dbvns_r.exec_ms  = v_ms;
        dbvns_r.sequence = std::move(vseq);
        if (dbvns_r.valid) {
            dbvns_r.cost    = cmp_seq_full_cost(dbvns_r.sequence, id_to_code, depot_code, tt);
            dbvns_r.t_ratio = (actual_cost > 0.f) ? dbvns_r.cost / actual_cost : -1.f;
            if (rc) {
                dbvns_r.dist_km = cmp_seq_dist(dbvns_r.sequence, id_to_code, *rc);
                if (dbvns_r.dist_km > 0.f && actual_km > 0.f)
                    dbvns_r.d_ratio = dbvns_r.dist_km / actual_km;
            }
            auto au = cmp_audit(dbvns_r.sequence, pickup_of, id_to_code, tt);
            dbvns_r.pdp_violations = au.pdp;
            dbvns_r.unreachable    = au.unreachable;
            dbvns_r.local_eff = cmp_local_efficiency(dbvns_r.sequence, env);
        }

        // ── Run IVNS ─────────────────────────────────────────────────────────
        std::cout << "  IVNS  running...  " << std::flush;
        auto ivns_res = IVNSSolver::solve(env, pickup_of, ivns_iters, ivns_k);
        std::cout << "done (" << ivns_res.exec_ms << " ms)\n" << std::flush;

        SolverResult ivns_r;
        ivns_r.valid    = !ivns_res.sequence.empty()
                        && static_cast<int>(ivns_res.sequence.size()) == n_stops;
        ivns_r.exec_ms  = ivns_res.exec_ms;
        ivns_r.sequence = std::move(ivns_res.sequence);
        if (ivns_r.valid) {
            ivns_r.cost    = cmp_seq_full_cost(ivns_r.sequence, id_to_code, depot_code, tt);
            ivns_r.t_ratio = (actual_cost > 0.f) ? ivns_r.cost / actual_cost : -1.f;
            if (rc) {
                ivns_r.dist_km = cmp_seq_dist(ivns_r.sequence, id_to_code, *rc);
                if (ivns_r.dist_km > 0.f && actual_km > 0.f)
                    ivns_r.d_ratio = ivns_r.dist_km / actual_km;
            }
            auto au = cmp_audit(ivns_r.sequence, pickup_of, id_to_code, tt);
            ivns_r.pdp_violations = au.pdp;
            ivns_r.unreachable    = au.unreachable;
            ivns_r.local_eff = cmp_local_efficiency(ivns_r.sequence, env);
        }

        // ── Run LKH ─────────────────────────────────────────────────────────
        auto lkh_res = LKHSolver::solve(env, pickup_of, lkh_restarts);

        SolverResult lkh_r;
        lkh_r.valid    = !lkh_res.sequence.empty()
                       && static_cast<int>(lkh_res.sequence.size()) == n_stops;
        lkh_r.exec_ms  = lkh_res.exec_ms;
        lkh_r.sequence = std::move(lkh_res.sequence);
        if (lkh_r.valid) {
            lkh_r.cost    = cmp_seq_full_cost(lkh_r.sequence, id_to_code, depot_code, tt);
            lkh_r.t_ratio = (actual_cost > 0.f) ? lkh_r.cost / actual_cost : -1.f;
            if (rc) {
                lkh_r.dist_km = cmp_seq_dist(lkh_r.sequence, id_to_code, *rc);
                if (lkh_r.dist_km > 0.f && actual_km > 0.f)
                    lkh_r.d_ratio = lkh_r.dist_km / actual_km;
            }
            auto au = cmp_audit(lkh_r.sequence, pickup_of, id_to_code, tt);
            lkh_r.pdp_violations = au.pdp;
            lkh_r.unreachable    = au.unreachable;
            lkh_r.local_eff = cmp_local_efficiency(lkh_r.sequence, env);
        }

        // ── Per-route print ──────────────────────────────────────────────────
        auto print_solver = [&](const char* label, const SolverResult& sr,
                                const std::string& extra) {
            std::cout << "  " << label << ": ";
            if (sr.valid) {
                std::cout << cmp_fmt_time(sr.cost)
                          << "  t_ratio=" << std::fixed << std::setprecision(3) << sr.t_ratio;
                if (sr.dist_km >= 0.f)
                    std::cout << "  dist=" << std::setprecision(1) << sr.dist_km
                              << "km  d_ratio=" << std::setprecision(3) << sr.d_ratio;
                if (sr.local_eff >= 0.f)
                    std::cout << "  loc_eff=" << std::setprecision(3) << sr.local_eff;
                std::cout << "  " << extra
                          << "  | " << sr.exec_ms << " ms";
                if (sr.pdp_violations > 0)
                    std::cout << "  [!PDP:" << sr.pdp_violations << "]";
            } else {
                std::cout << "INCOMPLETE  " << extra << "  | " << sr.exec_ms << " ms";
            }
            std::cout << "\n";
        };

        std::string dbvns_extra = "anchors=" + std::to_string(vns_stats.anchors_valid)
                                + "/" + std::to_string(vns_stats.anchors_tried);
        std::string lkh_extra  = "restarts=" + std::to_string(lkh_res.restarts_done);
        std::string ivns_extra = "iters=" + std::to_string(ivns_res.iters_done);

        print_solver("DbVNS", dbvns_r, dbvns_extra);
        print_solver("LKH  ", lkh_r,   lkh_extra);
        print_solver("IVNS ", ivns_r,   ivns_extra);

        // Sequence previews
        if (dbvns_r.valid)
            std::cout << "  DbVNS seq: "
                      << cmp_seq_preview(depot_code, dbvns_r.sequence, id_to_code) << "\n";
        if (lkh_r.valid)
            std::cout << "  LKH   seq: "
                      << cmp_seq_preview(depot_code, lkh_r.sequence, id_to_code) << "\n";
        if (ivns_r.valid)
            std::cout << "  IVNS  seq: "
                      << cmp_seq_preview(depot_code, ivns_r.sequence, id_to_code) << "\n";

        // SVG renders
        if (rc) {
            std::string svg_d = results_dir + sep + short_id + "_dbvns.svg";
            std::string svg_l = results_dir + sep + short_id + "_lkh.svg";
            std::string svg_i = results_dir + sep + short_id + "_ivns.svg";
            if (dbvns_r.valid)
                cmp_render_svg(short_id, "DbVNS", dbvns_r.sequence,
                               id_to_code, *rc, dbvns_r.t_ratio, v_ms, svg_d);
            if (lkh_r.valid)
                cmp_render_svg(short_id, "LKH", lkh_r.sequence,
                               id_to_code, *rc, lkh_r.t_ratio, lkh_res.exec_ms, svg_l);
            if (ivns_r.valid)
                cmp_render_svg(short_id, "IVNS", ivns_r.sequence,
                               id_to_code, *rc, ivns_r.t_ratio, ivns_res.exec_ms, svg_i);
            std::cout << "  Renders: " << short_id << "_dbvns.svg"
                      << "  |  " << short_id << "_lkh.svg"
                      << "  |  " << short_id << "_ivns.svg\n";
        }
        std::cout << "\n";
        std::cout.flush();

        // Store result
        std::string best_anchor_code = "-";
        {
            auto it = id_to_code.find(vns_stats.best_anchor_id);
            if (it != id_to_code.end()) best_anchor_code = it->second;
        }
        CompResult cr;
        cr.route_id        = rid;
        cr.num_stops       = n_stops;
        cr.num_tasks       = n_tasks;
        cr.actual_cost     = actual_cost;
        cr.actual_dist_km  = actual_km;
        cr.depot_code      = depot_code;
        cr.dbvns           = dbvns_r;
        cr.lkh             = lkh_r;
        cr.ivns            = ivns_r;
        cr.anchors_tried   = vns_stats.anchors_tried;
        cr.anchors_valid   = vns_stats.anchors_valid;
        cr.best_anchor     = best_anchor_code;
        cr.lkh_restarts    = lkh_res.restarts_done;
        results.push_back(cr);
    }

    if (results.empty()) { std::cout << "\nNo results.\n"; return; }

    // ── Summary table ────────────────────────────────────────────────────────
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "  Summary\n";
    std::cout << std::string(100, '=') << "\n\n";

    // Header
    std::cout << std::left  << std::setw(12) << "Route"
              << std::right << std::setw(6)  << "Stops"
              << std::setw(9)  << "DbVNS_t"
              << std::setw(9)  << "LKH_t"
              << std::setw(9)  << "IVNS_t"
              << std::setw(9)  << "Actual_t"
              << std::setw(8)  << "D_ratio"
              << std::setw(8)  << "L_ratio"
              << std::setw(8)  << "I_ratio"
              << std::setw(8)  << "ms_D"
              << std::setw(8)  << "ms_L"
              << std::setw(8)  << "ms_I"
              << "\n";
    cmp_sep(110);

    int   valid_both = 0;
    float sum_d_tr = 0, sum_l_tr = 0, sum_i_tr = 0;
    float sum_d_km = 0, sum_l_km = 0, sum_a_km = 0;
    float sum_d_eff = 0, sum_l_eff = 0, sum_i_eff = 0;
    long long sum_d_ms = 0, sum_l_ms = 0, sum_i_ms = 0;

    for (const auto& r : results) {
        std::string sid = r.route_id.substr(r.route_id.rfind('_') + 1, 8);
        std::cout << std::left  << std::setw(12) << sid
                  << std::right << std::setw(6)  << r.num_stops;

        if (r.dbvns.valid)
            std::cout << std::setw(9) << cmp_fmt_time(r.dbvns.cost);
        else
            std::cout << std::setw(9) << "N/A";

        if (r.lkh.valid)
            std::cout << std::setw(9) << cmp_fmt_time(r.lkh.cost);
        else
            std::cout << std::setw(9) << "N/A";

        if (r.ivns.valid)
            std::cout << std::setw(9) << cmp_fmt_time(r.ivns.cost);
        else
            std::cout << std::setw(9) << "N/A";

        std::cout << std::setw(9) << cmp_fmt_time(r.actual_cost);

        if (r.dbvns.valid) std::cout << std::setw(8) << std::fixed << std::setprecision(3) << r.dbvns.t_ratio;
        else               std::cout << std::setw(8) << "-";
        if (r.lkh.valid)  std::cout << std::setw(8) << r.lkh.t_ratio;
        else               std::cout << std::setw(8) << "-";
        if (r.ivns.valid) std::cout << std::setw(8) << r.ivns.t_ratio;
        else               std::cout << std::setw(8) << "-";

        std::cout << std::setw(8) << r.dbvns.exec_ms
                  << std::setw(8) << r.lkh.exec_ms
                  << std::setw(8) << r.ivns.exec_ms
                  << "\n";

        if (r.dbvns.valid && r.lkh.valid && r.ivns.valid) {
            ++valid_both;
            sum_d_tr += r.dbvns.t_ratio; sum_l_tr += r.lkh.t_ratio; sum_i_tr += r.ivns.t_ratio;
            sum_d_ms += r.dbvns.exec_ms; sum_l_ms += r.lkh.exec_ms; sum_i_ms += r.ivns.exec_ms;
            if (r.dbvns.dist_km  >= 0) sum_d_km += r.dbvns.dist_km;
            if (r.lkh.dist_km   >= 0) sum_l_km += r.lkh.dist_km;
            if (r.actual_dist_km >= 0) sum_a_km += r.actual_dist_km;
            if (r.dbvns.local_eff >= 0) sum_d_eff += r.dbvns.local_eff;
            if (r.lkh.local_eff  >= 0) sum_l_eff += r.lkh.local_eff;
            if (r.ivns.local_eff >= 0) sum_i_eff += r.ivns.local_eff;
        }
    }
    cmp_sep(110);

    if (valid_both == 0) { std::cout << "\nNo valid paired results.\n"; return; }
    float nf = static_cast<float>(valid_both);

    // ── Comparison summary ───────────────────────────────────────────────────
    std::cout << "\n  Routes with both valid: " << valid_both << "\n\n";

    std::cout << "  ── Time ratio (t_ratio = solver_time / actual_time) " << std::string(48, '-') << "\n";
    std::cout << "     DbVNS  avg t_ratio : " << std::fixed << std::setprecision(3) << (sum_d_tr/nf)
              << "  (1.000 = matches driver,  <1.000 = beats driver)\n";
    std::cout << "     LKH    avg t_ratio : " << (sum_l_tr/nf) << "\n";
    std::cout << "     IVNS   avg t_ratio : " << (sum_i_tr/nf) << "\n";
    float improvement = (sum_d_tr - sum_l_tr) / nf;
    std::cout << "     LKH vs DbVNS       : " << std::showpos << std::setprecision(3)
              << (-improvement) << std::noshowpos
              << "  (" << std::setprecision(1) << (improvement / (sum_d_tr/nf) * 100.f) << "% reduction)\n";
    float ivns_vs_lkh = (sum_i_tr - sum_l_tr) / nf;
    std::cout << "     IVNS vs LKH        : " << std::showpos << std::setprecision(3)
              << (-ivns_vs_lkh) << std::noshowpos << "\n";

    std::cout << "\n  ── Distance " << std::string(87, '-') << "\n";
    if (sum_d_km > 0 && sum_l_km > 0) {
        std::cout << "     DbVNS  avg dist   : " << std::setprecision(1) << (sum_d_km/nf) << " km/route\n";
        std::cout << "     LKH    avg dist   : " << (sum_l_km/nf) << " km/route\n";
        std::cout << "     Actual avg dist   : " << (sum_a_km/nf) << " km/route\n";
    }

    std::cout << "\n  ── Execution time " << std::string(81, '-') << "\n";
    float speedup_lkh  = (sum_l_ms > 0) ? static_cast<float>(sum_d_ms) / static_cast<float>(sum_l_ms) : 0.f;
    float speedup_ivns = (sum_i_ms > 0) ? static_cast<float>(sum_d_ms) / static_cast<float>(sum_i_ms) : 0.f;
    std::cout << "     DbVNS  total ms   : " << sum_d_ms << " ms"
              << "   avg=" << static_cast<long long>(sum_d_ms / valid_both) << " ms/route\n";
    std::cout << "     LKH    total ms   : " << sum_l_ms << " ms"
              << "   avg=" << static_cast<long long>(sum_l_ms / valid_both) << " ms/route\n";
    std::cout << "     IVNS   total ms   : " << sum_i_ms << " ms"
              << "   avg=" << static_cast<long long>(sum_i_ms / valid_both) << " ms/route\n";
    std::cout << "     LKH  speedup vs DbVNS : " << std::setprecision(1) << speedup_lkh  << "x\n";
    std::cout << "     IVNS speedup vs DbVNS : " << speedup_ivns << "x\n";

    // ── Architecture validation ──────────────────────────────────────────────
    int d_pdp = 0, l_pdp = 0, i_pdp = 0;
    for (const auto& r : results) {
        if (r.dbvns.valid) d_pdp += r.dbvns.pdp_violations;
        if (r.lkh.valid)  l_pdp += r.lkh.pdp_violations;
        if (r.ivns.valid) i_pdp += r.ivns.pdp_violations;
    }
    std::cout << "\n  ── PDP constraint validation " << std::string(71, '-') << "\n";
    std::cout << "     DbVNS violations : " << d_pdp << (d_pdp == 0 ? "  (all routes pass)" : "  [FAIL]") << "\n";
    std::cout << "     LKH  violations  : " << l_pdp << (l_pdp == 0 ? "  (all routes pass)" : "  [FAIL]") << "\n";
    std::cout << "     IVNS violations  : " << i_pdp << (i_pdp == 0 ? "  (all routes pass)" : "  [FAIL]") << "\n";

    std::cout << "\n  ── Local efficiency  avg(min_edge / chosen_edge) ∈ (0,1] " << std::string(40, '-') << "\n";
    std::cout << "     DbVNS  avg loc_eff : " << std::fixed << std::setprecision(3) << (sum_d_eff/nf)
              << "  (1.000 = always chose cheapest neighbor)\n";
    std::cout << "     LKH    avg loc_eff : " << (sum_l_eff/nf) << "\n";
    std::cout << "     IVNS   avg loc_eff : " << (sum_i_eff/nf) << "\n";

    std::cout << "\n" << std::string(100, '=') << "\n\n";
}
