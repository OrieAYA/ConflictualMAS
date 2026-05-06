#include "AmazonTest.hpp"
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
// DATA STRUCTURES
// ============================================================================

// Per-route result collected after running DbVNS.
struct RouteResult {
    std::string route_id;
    int         num_stops;            // delivery stops (excluding depot)
    int         num_tasks;            // PDP pairs created
    float       dbvns_cost;           // travel time in seconds  (DbVNS sequence)
    float       actual_cost;          // travel time in seconds  (ground truth)
    float       t_ratio;              // dbvns_cost / actual_cost  — time ratio
    float       dbvns_dist_km = -1.f; // route length in km       (DbVNS sequence)
    float       actual_dist_km= -1.f; // route length in km       (ground truth)
    float       d_ratio       = -1.f; // dbvns_dist / actual_dist — distance ratio
    float       tasks_per_km  = -1.f; // num_tasks / dbvns_dist_km
    long long   exec_ms;
    bool        valid;
    int         sequence_len;
    // ── architectural validation ──
    int         pdp_violations = 0;
    int         unreachable    = 0;
    // ── agent details ──
    std::string depot_code;
    int         anchors_tried    = 0;
    int         anchors_valid    = 0;
    std::string best_anchor_code;
};

// Audit results from verify_sequence().
struct SequenceAudit {
    int pdp_violations = 0;
    int unreachable    = 0;
};

// Stats collected by run_dbvns() — agent-level metrics.
struct DbVNSStats {
    int    anchors_tried = 0;
    int    anchors_valid = 0;       // anchors that produced a complete sequence
    osmium::object_id_type best_anchor_id = 0;
};

// ============================================================================
// JSON LOADING HELPERS
// ============================================================================

static json load_json(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    json j;
    f >> j;
    return j;
}

// ============================================================================
// ROUTE PARSING
// ============================================================================

// Returns stop → {to_stop → seconds}
static std::unordered_map<std::string,
                          std::unordered_map<std::string, float>>
parse_travel_times_for_route(const json& jroute)
{
    std::unordered_map<std::string,
                       std::unordered_map<std::string, float>> tt;
    for (auto& [from, tos] : jroute.items()) {
        for (auto& [to, val] : tos.items()) {
            if (val.is_number())
                tt[from][to] = val.get<float>();
        }
    }
    return tt;
}

// Returns stop → sequence_position
static std::unordered_map<std::string, int>
parse_actual_sequence(const json& jseq)
{
    std::unordered_map<std::string, int> seq;
    for (auto& [stop, pos] : jseq.items())
        if (pos.is_number())
            seq[stop] = static_cast<int>(pos.get<double>());
    return seq;
}

// ============================================================================
// GROUND-TRUTH COST
// ============================================================================

// Compute total travel time from depot through the actual sequence.
// actual_seq: stop → position (0 = depot).
static float compute_actual_cost(
    const std::unordered_map<std::string, int>&
        actual_seq,
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, float>>& tt)
{
    // Sort all stops by their actual position.
    std::vector<std::pair<int, std::string>> ordered;
    ordered.reserve(actual_seq.size());
    for (auto& [s, p] : actual_seq) ordered.push_back({p, s});
    std::sort(ordered.begin(), ordered.end());

    float cost = 0.0f;
    for (size_t i = 0; i + 1 < ordered.size(); ++i) {
        const std::string& a = ordered[i].second;
        const std::string& b = ordered[i + 1].second;
        auto ia = tt.find(a);
        if (ia == tt.end()) continue;
        auto ib = ia->second.find(b);
        if (ib == ia->second.end()) continue;
        cost += ib->second;
    }
    return cost;
}

// ============================================================================
// ARCHITECTURE VALIDATION - SEQUENCE AUDIT
// ============================================================================

// Checks two architectural invariants of DbVNS-PDP on the best sequence:
//   1. PDP constraint  : for every (pickup_i, delivery_i) pair, pickup_i must
//      appear strictly before delivery_i in the visit order.
//   2. Reachability    : no consecutive edge in the solution carries a sentinel
//      cost (≥1e8 s), which would indicate a missing travel-time entry.
static SequenceAudit audit_sequence(
    const std::vector<ObjectiveNode>&                                          seq,
    const PairingMap&                                                          pickup_of,
    const std::unordered_map<osmium::object_id_type, std::string>&             id_to_code,
    const std::unordered_map<std::string,
                              std::unordered_map<std::string, float>>&         tt)
{
    SequenceAudit a;
    if (seq.empty()) return a;

    // O(1) position lookup
    std::unordered_map<osmium::object_id_type, int> pos;
    pos.reserve(seq.size());
    for (int i = 0; i < static_cast<int>(seq.size()); ++i)
        pos[seq[i].id] = i;

    // 1. PDP constraint: delivery node's pickup must appear at a lower index.
    for (const auto& node : seq) {
        auto it = pickup_of.find(node.id);
        if (it == pickup_of.end()) continue;           // not a delivery node
        auto pit = pos.find(it->second);
        if (pit == pos.end() || pit->second >= pos.at(node.id))
            ++a.pdp_violations;
    }

    // 2. Reachability: scan consecutive edges.
    for (size_t i = 0; i + 1 < seq.size(); ++i) {
        auto ita = id_to_code.find(seq[i].id);
        auto itb = id_to_code.find(seq[i + 1].id);
        if (ita == id_to_code.end() || itb == id_to_code.end()) { ++a.unreachable; continue; }
        auto fa = tt.find(ita->second);
        if (fa == tt.end()) { ++a.unreachable; continue; }
        auto fb = fa->second.find(itb->second);
        if (fb == fa->second.end() || fb->second >= 1e8f) ++a.unreachable;
    }

    return a;
}

// ============================================================================
// ENVIRONMENT CONSTRUCTION
// ============================================================================

// Build OperableEnvironment and PairingMaps from delivery stop list + TT matrix.
// Stops are paired consecutively: (stops[0], stops[1]), (stops[2], stops[3]), …
// Returns the tasks vector (kept alive for the env to reference its nodes).
static std::vector<PDPTask> build_environment(
    const std::vector<std::string>&                                    stops,
    const std::unordered_map<std::string,
                              std::unordered_map<std::string, float>>& tt,
    OperableEnvironment&  env,
    PairingMap&           pickup_of,
    PairingMap&           delivery_of,
    std::unordered_map<std::string, osmium::object_id_type>& code_to_id,
    std::unordered_map<osmium::object_id_type, std::string>& id_to_code)
{
    // Assign sequential IDs: stop index + 1 (0 is reserved for depot).
    for (size_t i = 0; i < stops.size(); ++i) {
        osmium::object_id_type id = static_cast<osmium::object_id_type>(i + 1);
        code_to_id[stops[i]] = id;
        id_to_code[id]        = stops[i];
    }

    // Pair consecutive stops as (pickup, delivery).
    int n_pairs = static_cast<int>(stops.size()) / 2;
    std::vector<PDPTask> tasks(n_pairs);
    for (int p = 0; p < n_pairs; ++p) {
        osmium::object_id_type pick_id = code_to_id[stops[p * 2]];
        osmium::object_id_type del_id  = code_to_id[stops[p * 2 + 1]];

        tasks[p].task_id  = p;
        tasks[p].pickup   = ObjectiveNode{pick_id, 0};
        tasks[p].delivery = ObjectiveNode{del_id,  0};

        env.add_task(tasks[p]);
        pickup_of [del_id]  = pick_id;
        delivery_of[pick_id] = del_id;
    }

    // Fill cost matrix from travel times.
    int n = env.size();
    for (int i = 0; i < n; ++i) {
        const std::string& si = id_to_code[env.nodes[i].id];
        for (int j = 0; j < n; ++j) {
            if (i == j) { env.set_cost(i, j, 0.0f); continue; }
            const std::string& sj = id_to_code[env.nodes[j].id];
            auto it = tt.find(si);
            if (it != tt.end()) {
                auto jt = it->second.find(sj);
                if (jt != it->second.end()) {
                    env.set_cost(i, j, jt->second);
                    continue;
                }
            }
            env.set_cost(i, j, 1e9f); // unreachable sentinel
        }
    }

    return tasks;
}

// ============================================================================
// DBVNS - RUN ALL LSAs AND PICK BEST
// ============================================================================

static std::vector<ObjectiveNode> run_dbvns(
    const OperableEnvironment& env,
    const PairingMap&          pickup_of,
    const PairingMap&          delivery_of,
    osmium::object_id_type     depot_id,
    const DbVNSParams&         params,
    int                        max_anchors,
    DbVNSStats&                stats)
{
    stats = {};

    // Collect delivery nodes (anchors) and subsample if too many.
    std::vector<const ObjectiveNode*> anchors;
    anchors.reserve(env.nodes.size());
    for (const auto& node : env.nodes)
        if (pickup_of.find(node.id) != pickup_of.end())
            anchors.push_back(&node);

    // Evenly stride through the anchor list so we cover the range uniformly.
    if (static_cast<int>(anchors.size()) > max_anchors) {
        std::vector<const ObjectiveNode*> sampled;
        sampled.reserve(max_anchors);
        double step = static_cast<double>(anchors.size()) / max_anchors;
        for (int i = 0; i < max_anchors; ++i)
            sampled.push_back(anchors[static_cast<size_t>(i * step)]);
        anchors = std::move(sampled);
    }

    stats.anchors_tried = static_cast<int>(anchors.size());

    std::vector<ObjectiveNode> best_seq;
    float best_cost = std::numeric_limits<float>::max();

    for (const ObjectiveNode* node_ptr : anchors) {
        const ObjectiveNode& node = *node_ptr;

        LocalSolutionAgent lsa(node, params);
        std::vector<ObjectiveNode> candidate =
            lsa.plan(env, pickup_of, delivery_of, depot_id);
        if (candidate.empty()) continue;

        // Evaluate candidate cost from the env cost matrix.
        float cost = 0.0f;
        for (size_t i = 0; i + 1 < candidate.size(); ++i) {
            int ai = env.find_index(candidate[i].id);
            int bi = env.find_index(candidate[i + 1].id);
            if (ai < 0 || bi < 0) { cost = std::numeric_limits<float>::max(); break; }
            float c = env.get_cost(ai, bi);
            if (c < 0.0f || c >= 1e8f) { cost = std::numeric_limits<float>::max(); break; }
            cost += c;
        }

        ++stats.anchors_valid;
        if (cost < best_cost) {
            best_cost            = cost;
            best_seq             = std::move(candidate);
            stats.best_anchor_id = node.id;
        }
    }
    return best_seq;
}

// ============================================================================
// SEQUENCE COST WITH DEPOT
// ============================================================================

static float sequence_full_cost(
    const std::vector<ObjectiveNode>&                                   seq,
    const std::unordered_map<osmium::object_id_type, std::string>&      id_to_code,
    const std::string&                                                   depot_code,
    const std::unordered_map<std::string,
                              std::unordered_map<std::string, float>>&  tt)
{
    if (seq.empty()) return std::numeric_limits<float>::max();

    auto travel = [&](const std::string& a, const std::string& b) -> float {
        auto ia = tt.find(a);
        if (ia == tt.end()) return 1e9f;
        auto ib = ia->second.find(b);
        return (ib == ia->second.end()) ? 1e9f : ib->second;
    };

    float cost = 0.0f;

    // Depot → first stop
    auto it0 = id_to_code.find(seq[0].id);
    if (it0 == id_to_code.end()) return std::numeric_limits<float>::max();
    cost += travel(depot_code, it0->second);

    // Sequence legs
    for (size_t i = 0; i + 1 < seq.size(); ++i) {
        auto ita = id_to_code.find(seq[i].id);
        auto itb = id_to_code.find(seq[i + 1].id);
        if (ita == id_to_code.end() || itb == id_to_code.end())
            return std::numeric_limits<float>::max();
        cost += travel(ita->second, itb->second);
    }
    return cost;
}

// ============================================================================
// REPORT
// ============================================================================

// stop code -> {lat, lng}  (forward-declared here so distance helpers can use it)
using StopCoords = std::unordered_map<std::string, std::pair<double, double>>;
// route id  -> StopCoords
using AllCoords  = std::unordered_map<std::string, StopCoords>;

// ============================================================================
// DISTANCE HELPERS
// ============================================================================

static double haversine_km(double lat1, double lon1, double lat2, double lon2)
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

// Total haversine distance of the DbVNS sequence (km).
static float seq_dist_km(
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
        total += haversine_km(ca->second.first,  ca->second.second,
                              cb->second.first,  cb->second.second);
    }
    return static_cast<float>(total);
}

// Total haversine distance of the ground-truth sequence (km).
static float actual_dist_km(
    const std::unordered_map<std::string, int>& aseq,
    const StopCoords&                            coords)
{
    std::vector<std::pair<int, std::string>> ordered;
    ordered.reserve(aseq.size());
    for (auto& [s, p] : aseq) ordered.push_back({p, s});
    std::sort(ordered.begin(), ordered.end());

    double total = 0.0;
    for (size_t i = 0; i + 1 < ordered.size(); ++i) {
        auto ca = coords.find(ordered[i].second);
        auto cb = coords.find(ordered[i+1].second);
        if (ca == coords.end() || cb == coords.end()) continue;
        total += haversine_km(ca->second.first,  ca->second.second,
                              cb->second.first,  cb->second.second);
    }
    return static_cast<float>(total);
}

// ============================================================================
// COORDINATE LOADING  (new_route_data.json — NaN preprocessing required)
// ============================================================================

static AllCoords load_all_coords(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) return {};

    // new_route_data.json has bare NaN values (invalid JSON) in zone_id fields.
    // Replace every standalone NaN token with null before parsing.
    std::string raw((std::istreambuf_iterator<char>(f)), {});
    std::string cleaned;
    cleaned.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ) {
        if (i + 3 <= raw.size() && raw[i]=='N' && raw[i+1]=='a' && raw[i+2]=='N') {
            cleaned += "null"; i += 3;
        } else {
            cleaned += raw[i++];
        }
    }

    AllCoords result;
    try {
        json j = json::parse(cleaned);
        for (auto& [rid, rdata] : j.items()) {
            if (!rdata.contains("stops")) continue;
            StopCoords sc;
            for (auto& [code, sdata] : rdata["stops"].items()) {
                if (sdata.contains("lat") && sdata.contains("lng")
                    && sdata["lat"].is_number() && sdata["lng"].is_number())
                    sc[code] = { sdata["lat"].get<double>(), sdata["lng"].get<double>() };
            }
            result[rid] = std::move(sc);
        }
    } catch (...) {}
    return result;
}

// ============================================================================
// SVG RENDERER
// ============================================================================

static void render_route_svg(
    const std::string&                                               short_id,
    const std::vector<ObjectiveNode>&                               seq,
    const std::unordered_map<osmium::object_id_type, std::string>& id_to_code,
    const StopCoords&                                               coords,
    float ratio, long long exec_ms,
    const std::string&                                               out_path)
{
    if (seq.empty() || coords.empty()) return;

    // Build ordered list of points with sequence position.
    struct Pt { double lat, lng; int pos; bool is_anchor; };
    std::vector<Pt> pts;
    pts.reserve(seq.size());
    int n = static_cast<int>(seq.size());
    for (int i = 0; i < n; ++i) {
        auto ic = id_to_code.find(seq[i].id);
        if (ic == id_to_code.end()) continue;
        auto co = coords.find(ic->second);
        if (co == coords.end()) continue;
        pts.push_back({co->second.first, co->second.second, i, i == n - 1});
    }
    if (pts.empty()) return;

    // Bounding box with 10 % padding.
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

    const int W = 680, H = 680, LEG = 80;   // 80px legend panel at bottom
    auto px = [&](double lng) { return (lng - mn_lng) / (mx_lng - mn_lng) * W; };
    auto py = [&](double lat) { return H - (lat - mn_lat) / (mx_lat - mn_lat) * H; };

    std::ofstream svg(out_path);
    svg << "<svg width=\"" << W << "\" height=\"" << (H + LEG)
        << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";

    // Map background
    svg << "<rect width=\"" << W << "\" height=\"" << H
        << "\" fill=\"#f5f7fa\" rx=\"4\"/>\n";

    // Sequence polyline
    svg << "<polyline fill=\"none\" stroke=\"#7bafd4\" stroke-width=\"1\""
           " opacity=\"0.5\" points=\"";
    for (auto& p : pts)
        svg << std::fixed << std::setprecision(2) << px(p.lng) << "," << py(p.lat) << " ";
    svg << "\"/>\n";

    // Stops — gradient: blue(early) -> purple(mid) -> orange-red(late)
    for (auto& p : pts) {
        float t = (n > 1) ? static_cast<float>(p.pos) / (n - 1) : 0.5f;
        int r = static_cast<int>(40  + t * 215);
        int g = static_cast<int>(120 - t * 50);
        int b = static_cast<int>(200 - t * 160);
        if (p.is_anchor) {
            svg << "<circle cx=\"" << px(p.lng) << "\" cy=\"" << py(p.lat)
                << "\" r=\"7\" fill=\"#ff6600\" stroke=\"white\" stroke-width=\"2\"/>\n";
        } else {
            svg << "<circle cx=\"" << px(p.lng) << "\" cy=\"" << py(p.lat)
                << "\" r=\"3\" fill=\"rgb(" << r << "," << g << "," << b << ")\"/>\n";
        }
    }
    // First stop override: green
    if (!pts.empty())
        svg << "<circle cx=\"" << px(pts[0].lng) << "\" cy=\"" << py(pts[0].lat)
            << "\" r=\"5\" fill=\"#22bb55\" stroke=\"white\" stroke-width=\"1.5\"/>\n";

    // ── Legend panel ──────────────────────────────────────────────────────
    svg << "<rect y=\"" << H << "\" width=\"" << W << "\" height=\"" << LEG
        << "\" fill=\"#e8ecf0\"/>\n";

    // Row 1 — route info
    svg << "<text x=\"8\" y=\"" << (H + 18)
        << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#222\">"
        << short_id << "  |  " << n << " stops"
        << "  |  t_ratio=" << std::fixed << std::setprecision(3) << ratio
        << " (DbVNS time / actual time)"
        << "  |  " << exec_ms << " ms"
        << "</text>\n";

    // Row 2 — color key
    // Gradient bar: 6 steps from blue to orange-red
    int bar_x = 8, bar_y = H + 28, bar_w = 140, bar_h = 10;
    for (int s = 0; s < 6; ++s) {
        float t = s / 5.0f;
        int r2 = static_cast<int>(40  + t * 215);
        int g2 = static_cast<int>(120 - t * 50);
        int b2 = static_cast<int>(200 - t * 160);
        svg << "<rect x=\"" << (bar_x + s * bar_w/6) << "\" y=\"" << bar_y
            << "\" width=\"" << (bar_w/6 + 1) << "\" height=\"" << bar_h
            << "\" fill=\"rgb(" << r2 << "," << g2 << "," << b2 << ")\"/>\n";
    }
    svg << "<text x=\"" << (bar_x + bar_w + 4) << "\" y=\"" << (bar_y + 9)
        << "\" font-family=\"monospace\" font-size=\"11\" fill=\"#444\">"
        << "visit order: blue=early  purple=mid  red=late</text>\n";

    // Row 3 — special markers key
    svg << "<circle cx=\"" << (bar_x + 2) << "\" cy=\"" << (H + 54)
        << "\" r=\"5\" fill=\"#22bb55\" stroke=\"white\" stroke-width=\"1\"/>\n";
    svg << "<text x=\"" << (bar_x + 10) << "\" y=\"" << (H + 58)
        << "\" font-family=\"monospace\" font-size=\"11\" fill=\"#444\">first stop</text>\n";

    svg << "<circle cx=\"" << (bar_x + 72) << "\" cy=\"" << (H + 54)
        << "\" r=\"7\" fill=\"#ff6600\" stroke=\"white\" stroke-width=\"1.5\"/>\n";
    svg << "<text x=\"" << (bar_x + 82) << "\" y=\"" << (H + 58)
        << "\" font-family=\"monospace\" font-size=\"11\" fill=\"#444\">"
        << "last stop (best LSA anchor)</text>\n";

    svg << "</svg>\n";
}

// ============================================================================
// PRINT HELPERS
// ============================================================================

static void print_separator(int width = 80) {
    std::cout << std::string(width, '-') << "\n";
}

// Returns a compact one-line preview of the sequence:
//   depot > A > B > C > ... +N ... > Y > Z
static std::string seq_preview(
    const std::string&                                               depot,
    const std::vector<ObjectiveNode>&                               seq,
    const std::unordered_map<osmium::object_id_type, std::string>& id_to_code,
    int head = 3, int tail = 2)
{
    if (seq.empty()) return "(empty)";
    int n = static_cast<int>(seq.size());

    auto code = [&](osmium::object_id_type id) -> std::string {
        auto it = id_to_code.find(id);
        return it != id_to_code.end() ? it->second : "?";
    };

    std::ostringstream oss;
    oss << depot;
    int show_head = std::min(head, n);
    for (int i = 0; i < show_head; ++i) oss << " > " << code(seq[i].id);

    int middle = n - show_head - tail;
    if (middle > 0) {
        oss << " > ...+" << middle << "...";
        for (int i = n - tail; i < n; ++i) oss << " > " << code(seq[i].id);
    } else {
        for (int i = show_head; i < n; ++i) oss << " > " << code(seq[i].id);
    }
    return oss.str();
}

static std::string format_time(float seconds) {
    if (seconds >= 1e8f) return "    N/A";
    int m = static_cast<int>(seconds) / 60;
    int s = static_cast<int>(seconds) % 60;
    std::ostringstream oss;
    oss << std::setw(3) << m << "m" << std::setw(2) << std::setfill('0') << s << "s";
    return oss.str();
}

// ============================================================================
// MAIN TEST FUNCTION
// ============================================================================

void test_amazon_routes(const std::string& data_dir)
{
    const std::string sep = data_dir.find('/') != std::string::npos ? "/" : "\\";

    const std::string tt_path     = data_dir + sep + "model_apply_inputs" + sep + "new_travel_times.json";
    const std::string actual_path = data_dir + sep + "model_score_inputs" + sep + "new_actual_sequences.json";
    const std::string coords_path = data_dir + sep + "model_apply_inputs" + sep + "new_route_data.json";

    // Results folder — created next to the AmazonDataset directory.
    // data_dir is  .../AmazonDataset/~/.rc-cli/data  → go up 3 levels.
    std::string results_dir;
    {
        std::string d = data_dir;
        for (int i = 0; i < 3; ++i) {
            size_t p = d.find_last_of("/\\");
            if (p != std::string::npos) d = d.substr(0, p);
        }
        results_dir = d + sep + "results";
    }
    fs::create_directories(results_dir);

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  Amazon Last Mile Challenge - DbVNS Experiment\n";
    std::cout << std::string(80, '=') << "\n\n";

    // ---- DbVNS parameters (reduced for experiment speed) ------------------
    DbVNSParams params;
    params.max_iterations     = 15;
    params.k_max              = 3;
    params.max_decompositions = 2;
    params.max_divergence     = 0.5;

    // Max LSA anchors per route: trying all N/2 delivery nodes as anchors is
    // O(N³) in practice; capping at 15 keeps the experiment tractable.
    const int max_anchors = 15;

    std::cout << "DbVNS params: max_iter=" << params.max_iterations
              << "  k_max=" << params.k_max
              << "  max_decomps=" << params.max_decompositions
              << "  max_div=" << params.max_divergence
              << "  max_anchors=" << max_anchors << "\n\n";

    // ---- Load JSON files --------------------------------------------------
    json jtt, jactual;
    try {
        std::cout << "Loading travel times... "; std::cout.flush();
        jtt = load_json(tt_path);
        std::cout << "ok (" << jtt.size() << " routes)\n";

        std::cout << "Loading actual seqs...  "; std::cout.flush();
        jactual = load_json(actual_path);
        std::cout << "ok (" << jactual.size() << " routes)\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return;
    }

    // Load stop coordinates for SVG rendering (NaN pre-processed internally).
    std::cout << "Loading stop coords...  "; std::cout.flush();
    AllCoords all_coords = load_all_coords(coords_path);
    std::cout << (all_coords.empty() ? "unavailable (SVG skipped)\n" : "ok\n");
    std::cout << "Results dir:  " << results_dir << "\n\n";

    // ---- Collect route IDs present in both files --------------------------
    std::vector<std::string> all_ids;
    for (const auto& item : jtt.items())
        if (jactual.contains(item.key()))
            all_ids.push_back(item.key());

    if (all_ids.empty()) {
        std::cerr << "No routes found with data in all three files.\n";
        return;
    }

    // Shuffle and take up to 10.
    {
        std::mt19937 rng(42);
        std::shuffle(all_ids.begin(), all_ids.end(), rng);
    }
    const int max_routes = std::min(10, static_cast<int>(all_ids.size()));
    all_ids.resize(max_routes);

    std::cout << "Testing " << max_routes << " routes:\n\n";

    // ---- Per-route loop --------------------------------------------------
    std::vector<RouteResult> results;

    for (const std::string& rid : all_ids) {
        // Shorten route ID for display.
        std::string short_id = rid.substr(rid.rfind('_') + 1, 8);

        // Parse travel times + actual sequence for this route.
        auto tt = parse_travel_times_for_route(jtt.at(rid));

        const json& jroute_seq = jactual.at(rid);
        const json& jstops_seq = jroute_seq.contains("actual")
                                 ? jroute_seq.at("actual")
                                 : jroute_seq;
        auto aseq = parse_actual_sequence(jstops_seq);

        // Identify depot (pos 0).
        std::string depot_code;
        for (auto& [s, p] : aseq)
            if (p == 0) { depot_code = s; break; }

        if (depot_code.empty()) {
            std::cout << "[" << short_id << "] No depot found - skip\n";
            continue;
        }

        // Delivery stops = everything except depot.
        std::vector<std::string> delivery_stops;
        delivery_stops.reserve(aseq.size() - 1);
        for (auto& [s, p] : aseq)
            if (p != 0) delivery_stops.push_back(s);

        // Ensure even count for clean pairing.
        if (delivery_stops.size() % 2 != 0)
            delivery_stops.pop_back();

        int n_stops = static_cast<int>(delivery_stops.size());
        int n_tasks = n_stops / 2;

        if (n_tasks == 0) {
            std::cout << "[" << short_id << "] Too few stops - skip\n";
            continue;
        }

        // Ground-truth cost.
        float actual_cost = compute_actual_cost(aseq, tt);

        // Build OperableEnvironment.
        OperableEnvironment env;
        PairingMap pickup_of, delivery_of;
        std::unordered_map<std::string, osmium::object_id_type> code_to_id;
        std::unordered_map<osmium::object_id_type, std::string> id_to_code;

        auto tasks = build_environment(delivery_stops, tt, env,
                                        pickup_of, delivery_of,
                                        code_to_id, id_to_code);

        // Depot fake ID (not in env - used only for cost evaluation).
        osmium::object_id_type depot_id =
            static_cast<osmium::object_id_type>(delivery_stops.size() + 1);

        // Run DbVNS.
        std::cout << "[" << short_id << "] "
                  << std::setw(3) << n_stops << " stops, "
                  << std::setw(3) << n_tasks << " tasks  ...  ";
        std::cout.flush();

        DbVNSStats vns_stats;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<ObjectiveNode> best_seq =
            run_dbvns(env, pickup_of, delivery_of, depot_id, params, max_anchors, vns_stats);
        auto t1 = std::chrono::high_resolution_clock::now();
        long long exec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        bool valid = !best_seq.empty() &&
                     best_seq.size() == static_cast<size_t>(n_stops);

        float dbvns_cost = valid
            ? sequence_full_cost(best_seq, id_to_code, depot_code, tt)
            : std::numeric_limits<float>::max();

        float t_ratio = (valid && actual_cost > 0.0f)
            ? dbvns_cost / actual_cost : -1.0f;

        // Architecture validation audit.
        SequenceAudit audit;
        if (valid) audit = audit_sequence(best_seq, pickup_of, id_to_code, tt);

        // Distance (haversine km) — only if coords available for this route.
        float dbvns_km = -1.f, actual_km = -1.f, d_ratio = -1.f, tpkm = -1.f;
        const StopCoords* route_coords = nullptr;
        if (!all_coords.empty()) {
            auto cit = all_coords.find(rid);
            if (cit != all_coords.end()) {
                route_coords = &cit->second;
                if (valid)
                    dbvns_km = seq_dist_km(best_seq, id_to_code, *route_coords);
                actual_km = actual_dist_km(aseq, *route_coords);
                if (dbvns_km > 0.f && actual_km > 0.f)
                    d_ratio = dbvns_km / actual_km;
                if (dbvns_km > 0.f)
                    tpkm = static_cast<float>(n_tasks) / dbvns_km;
            }
        }

        // Resolve best anchor stop code.
        std::string best_anchor_code = "-";
        {
            auto it = id_to_code.find(vns_stats.best_anchor_id);
            if (it != id_to_code.end()) best_anchor_code = it->second;
        }

        // ---- Print one-line progress + detail block -----------------------
        if (valid) {
            std::cout << format_time(dbvns_cost)
                      << " vs " << format_time(actual_cost)
                      << "  t_ratio=" << std::fixed << std::setprecision(3) << t_ratio
                      << "  (" << exec_ms << " ms)";
            if (audit.pdp_violations > 0) std::cout << "  [!PDP:" << audit.pdp_violations << "]";
            if (audit.unreachable     > 0) std::cout << "  [!REACH:" << audit.unreachable << "]";
            std::cout << "\n";
            // Agent stats
            std::cout << "  agent   : anchors " << vns_stats.anchors_valid
                      << "/" << vns_stats.anchors_tried << " valid"
                      << "  |  best anchor: " << best_anchor_code
                      << "  |  " << std::fixed << std::setprecision(1)
                      << (static_cast<float>(exec_ms) / n_stops) << " ms/stop\n";
            // Distance metrics
            if (dbvns_km >= 0.f) {
                std::cout << "  metrics : dist DbVNS=" << std::fixed << std::setprecision(1)
                          << dbvns_km << "km"
                          << "  Actual=" << actual_km << "km"
                          << "  d_ratio=" << std::setprecision(3) << d_ratio
                          << "  |  " << std::setprecision(2) << tpkm << " tasks/km\n";
            }
            // Solution preview
            std::cout << "  solution: "
                      << seq_preview(depot_code, best_seq, id_to_code) << "\n";
            // SVG render
            if (route_coords) {
                std::string svg_path = results_dir + sep + short_id + ".svg";
                render_route_svg(short_id, best_seq, id_to_code,
                                 *route_coords, t_ratio, exec_ms, svg_path);
                std::cout << "  render  : " << svg_path << "\n";
            }
        } else {
            std::cout << "incomplete sequence (" << exec_ms << " ms)\n";
            std::cout << "  agent   : anchors " << vns_stats.anchors_valid
                      << "/" << vns_stats.anchors_tried << " valid\n";
        }

        RouteResult r;
        r.route_id          = rid;
        r.num_stops         = n_stops;
        r.num_tasks         = n_tasks;
        r.dbvns_cost        = dbvns_cost;
        r.actual_cost       = actual_cost;
        r.t_ratio           = t_ratio;
        r.dbvns_dist_km     = dbvns_km;
        r.actual_dist_km    = actual_km;
        r.d_ratio           = d_ratio;
        r.tasks_per_km      = tpkm;
        r.exec_ms           = exec_ms;
        r.valid             = valid;
        r.sequence_len      = static_cast<int>(best_seq.size());
        r.pdp_violations    = audit.pdp_violations;
        r.unreachable       = audit.unreachable;
        r.depot_code        = depot_code;
        r.anchors_tried     = vns_stats.anchors_tried;
        r.anchors_valid     = vns_stats.anchors_valid;
        r.best_anchor_code  = best_anchor_code;
        results.push_back(r);
    }

    // ---- Summary table ---------------------------------------------------
    if (results.empty()) {
        std::cout << "\nNo results to summarize.\n";
        return;
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  Summary\n";
    std::cout << std::string(80, '=') << "\n\n";

    // Header
    std::cout << std::left  << std::setw(12) << "Route"
              << std::right << std::setw(7)  << "Stops"
              << std::setw(7)  << "Tasks"
              << std::setw(10) << "DbVNS_t"
              << std::setw(10) << "Act_t"
              << std::setw(9)  << "t_ratio"
              << std::setw(9)  << "DbVNS_km"
              << std::setw(9)  << "Act_km"
              << std::setw(9)  << "d_ratio"
              << std::setw(9)  << "T/km"
              << std::setw(9)  << "ms"
              << "\n";
    print_separator(100);

    int    valid_count  = 0;
    float  sum_tratio   = 0.0f, best_tratio = 1e9f, worst_tratio = 0.0f;
    float  sum_dratio   = 0.0f, best_dratio = 1e9f, worst_dratio = 0.0f;
    int    dratio_count = 0;
    float  sum_tpkm     = 0.0f;
    long long total_ms  = 0;
    float  sum_dbvns    = 0.0f, sum_actual  = 0.0f;
    float  sum_dbvns_km = 0.0f, sum_act_km  = 0.0f;

    for (const auto& r : results) {
        std::string short_id = r.route_id.substr(r.route_id.rfind('_') + 1, 8);
        std::cout << std::left  << std::setw(12) << short_id
                  << std::right << std::setw(7)  << r.num_stops
                  << std::setw(7)  << r.num_tasks;

        if (r.valid) {
            std::cout << std::setw(10) << format_time(r.dbvns_cost)
                      << std::setw(10) << format_time(r.actual_cost)
                      << std::setw(9)  << std::fixed << std::setprecision(3) << r.t_ratio;
            if (r.dbvns_dist_km >= 0.f) {
                std::cout << std::setw(9)  << std::setprecision(1) << r.dbvns_dist_km
                          << std::setw(9)  << r.actual_dist_km
                          << std::setw(9)  << std::setprecision(3) << r.d_ratio
                          << std::setw(9)  << std::setprecision(2) << r.tasks_per_km;
            } else {
                std::cout << std::setw(9) << "-" << std::setw(9) << "-"
                          << std::setw(9) << "-" << std::setw(9) << "-";
            }
            std::cout << std::setw(9) << r.exec_ms;
            ++valid_count;
            sum_tratio += r.t_ratio;  sum_dbvns += r.dbvns_cost; sum_actual += r.actual_cost;
            if (r.t_ratio < best_tratio)  best_tratio  = r.t_ratio;
            if (r.t_ratio > worst_tratio) worst_tratio = r.t_ratio;
            if (r.d_ratio >= 0.f) {
                sum_dratio += r.d_ratio; sum_tpkm += r.tasks_per_km;
                sum_dbvns_km += r.dbvns_dist_km; sum_act_km += r.actual_dist_km;
                if (r.d_ratio < best_dratio)  best_dratio  = r.d_ratio;
                if (r.d_ratio > worst_dratio) worst_dratio = r.d_ratio;
                ++dratio_count;
            }
        } else {
            std::cout << std::setw(10) << "N/A" << std::setw(10) << format_time(r.actual_cost)
                      << std::setw(9) << "N/A" << std::setw(9) << "-" << std::setw(9) << "-"
                      << std::setw(9) << "-"   << std::setw(9) << "-"
                      << std::setw(9) << r.exec_ms;
        }
        total_ms += r.exec_ms;
        std::cout << "\n";
    }

    print_separator(100);

    std::cout << "\n  Routes tested : " << results.size()
              << "  |  Valid : " << valid_count
              << "  |  Total time : " << total_ms << " ms"
              << "  (" << std::fixed << std::setprecision(2) << (total_ms/1000.0) << " s)\n";

    if (valid_count > 0) {
        float n_f       = static_cast<float>(valid_count);
        float avg_tr    = sum_tratio / n_f;

        // ── Ratio explanation ───────────────────────────────────────────────
        std::cout << "\n  Ratio definitions:\n";
        std::cout << "    t_ratio = DbVNS travel time  / actual travel time  (seconds, from travel_times.json)\n";
        std::cout << "    d_ratio = DbVNS route length / actual route length (km, haversine from lat/lng)\n";
        std::cout << "    T/km    = tasks solved per km of DbVNS route (density metric)\n";
        std::cout << "    < 1.0 -> DbVNS beats ground truth   |   > 1.0 -> ground truth is better\n";

        // ── Time stats ─────────────────────────────────────────────────────
        float sum_sq = 0.0f;
        for (const auto& r : results)
            if (r.valid) { float d = r.t_ratio - avg_tr; sum_sq += d*d; }
        float stddev_tr = (valid_count > 1)
            ? std::sqrt(sum_sq / (n_f - 1)) : 0.0f;

        std::cout << "\n  ── Time (" << valid_count << " routes) " << std::string(60, '-') << "\n";
        std::cout << "    DbVNS  avg : " << format_time(sum_dbvns / n_f) << "/route"
                  << "   total : " << std::fixed << std::setprecision(0) << sum_dbvns << " s\n";
        std::cout << "    Actual avg : " << format_time(sum_actual / n_f) << "/route"
                  << "   total : " << sum_actual << " s\n";
        std::cout << "    t_ratio    : avg=" << std::setprecision(3) << avg_tr
                  << "  std=" << stddev_tr
                  << "  best=" << best_tratio
                  << "  worst=" << worst_tratio << "\n";
        if (avg_tr < 1.05f)
            std::cout << "    => DbVNS competitive with Amazon drivers.\n";
        else if (avg_tr < 1.30f)
            std::cout << "    => DbVNS within 30% of ground truth.\n";
        else
            std::cout << "    => DbVNS ~" << std::setprecision(0) << (avg_tr - 1.f)*100.f
                      << "% slower than ground truth (arbitrary PDP pairing is the main bottleneck).\n";

        // ── Distance stats ──────────────────────────────────────────────────
        if (dratio_count > 0) {
            float nd = static_cast<float>(dratio_count);
            float sum_sq_d = 0.0f;
            for (const auto& r : results)
                if (r.d_ratio >= 0.f) { float d = r.d_ratio - sum_dratio/nd; sum_sq_d += d*d; }
            float stddev_dr = (dratio_count > 1) ? std::sqrt(sum_sq_d/(nd-1)) : 0.0f;

            std::cout << "\n  ── Distance (" << dratio_count << " routes) " << std::string(57, '-') << "\n";
            std::cout << "    DbVNS  avg : " << std::fixed << std::setprecision(1)
                      << (sum_dbvns_km/nd) << " km/route\n";
            std::cout << "    Actual avg : " << (sum_act_km/nd) << " km/route\n";
            std::cout << "    d_ratio    : avg=" << std::setprecision(3) << (sum_dratio/nd)
                      << "  std=" << stddev_dr
                      << "  best=" << best_dratio
                      << "  worst=" << worst_dratio << "\n";
            std::cout << "    T/km       : avg=" << std::setprecision(2) << (sum_tpkm/nd)
                      << " tasks/km  (DbVNS route)\n";
        }

        // ── Architecture validation ─────────────────────────────────────────
        int total_pdp  = 0, total_reach = 0, routes_with_violation = 0;
        for (const auto& r : results) {
            if (!r.valid) continue;
            total_pdp   += r.pdp_violations;
            total_reach += r.unreachable;
            if (r.pdp_violations > 0 || r.unreachable > 0) ++routes_with_violation;
        }

        std::cout << "\n  ── Architecture Validation " << std::string(52, '-') << "\n";
        std::cout << "  PDP constraint  (P_i before D_i) : ";
        if (total_pdp == 0)
            std::cout << valid_count << "/" << valid_count << " routes pass  - 0 violations\n";
        else
            std::cout << (valid_count - routes_with_violation) << "/" << valid_count
                      << " routes pass  - " << total_pdp << " violation(s)  [FAIL]\n";

        std::cout << "  Unreachable edges in solutions   : ";
        if (total_reach == 0)
            std::cout << "none\n";
        else
            std::cout << total_reach << " edge(s) with sentinel cost  [WARN]\n";

        std::cout << "  Sequence completeness            : "
                  << valid_count << "/" << results.size() << " routes - all stops visited\n";

        // ── Execution scaling ───────────────────────────────────────────────
        long long min_ms = std::numeric_limits<long long>::max(), max_ms = 0;
        long long sum_ms = 0;
        float min_mps = std::numeric_limits<float>::max(), max_mps = 0.0f, sum_mps = 0.0f;
        for (const auto& r : results) {
            if (!r.valid) continue;
            if (r.exec_ms < min_ms) min_ms = r.exec_ms;
            if (r.exec_ms > max_ms) max_ms = r.exec_ms;
            sum_ms += r.exec_ms;
            float mps = static_cast<float>(r.exec_ms) / static_cast<float>(r.num_stops);
            if (mps < min_mps) min_mps = mps;
            if (mps > max_mps) max_mps = mps;
            sum_mps += mps;
        }

        // Pearson correlation between num_stops and exec_ms to check scaling trend.
        float n_f2 = static_cast<float>(valid_count);
        float mean_stops = 0.0f, mean_ms_f = static_cast<float>(sum_ms) / n_f2;
        for (const auto& r : results) if (r.valid) mean_stops += r.num_stops;
        mean_stops /= n_f2;
        float cov = 0.0f, var_s = 0.0f, var_m = 0.0f;
        for (const auto& r : results) {
            if (!r.valid) continue;
            float ds = r.num_stops - mean_stops;
            float dm = static_cast<float>(r.exec_ms) - mean_ms_f;
            cov   += ds * dm;
            var_s += ds * ds;
            var_m += dm * dm;
        }
        float pearson = (var_s > 0.0f && var_m > 0.0f)
            ? cov / std::sqrt(var_s * var_m) : 0.0f;

        std::cout << "\n  ── Execution Scaling " << std::string(58, '-') << "\n";
        std::cout << "  ms/stop :  min=" << std::fixed << std::setprecision(1) << min_mps
                  << "   avg=" << (sum_mps / n_f)
                  << "   max=" << max_mps << "\n";
        std::cout << "  exec ms :  min=" << min_ms << "   avg="
                  << static_cast<long long>(sum_ms / valid_count)
                  << "   max=" << max_ms << "\n";
        std::cout << "  Pearson correlation (stops vs ms) : "
                  << std::fixed << std::setprecision(3) << pearson;
        if (pearson > 0.8f)
            std::cout << "  - strong positive (time scales with problem size, as expected)\n";
        else if (pearson > 0.4f)
            std::cout << "  - moderate positive\n";
        else
            std::cout << "  - weak (other factors dominate timing)\n";
    }

    std::cout << "\n" << std::string(80, '=') << "\n\n";
}
