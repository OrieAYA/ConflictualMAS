#include "PlanningComparisonTest.hpp"

#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include "Training/EpisodeConfig.hpp"
#include "Training/EpisodeRunner.hpp"
#include "Training/CityConfig.hpp"
#include "DMASforPD/GlobalMemory/GlobalMemory.hpp"
#include "DMASforPD/DeliveryAgent/DeliveryAgent.hpp"
#include "DMASforPD/Utility/AgentSolution.hpp"
#include "DMASforPD/Utility/PDPTask.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

// ────────────────────────────────────────────────────────────────────────────
// Data structures
// ────────────────────────────────────────────────────────────────────────────

namespace {

struct TaskTrace {
    int   task_id;
    int   arrival_step;
    int   picked_step;
    int   delivered_step;
    int   agent_id;
    osmium::object_id_type pickup_node;
    osmium::object_id_type delivery_node;
    double pickup_lat, pickup_lon;
    double delivery_lat, delivery_lon;
};

struct AgentTrace {
    int agent_id;
    struct Leg {
        osmium::object_id_type from, to;
        int dep_step, arr_step;
        double from_lat, from_lon, to_lat, to_lon;
        double length_m;
    };
    std::vector<Leg> legs;
};

struct ComparisonResult {
    // Identity
    int         seed;
    std::string city;
    float       density_mult;
    float       agents_mult;
    std::string scenario_label;
    std::string mode;

    // Performance
    int    n_tasks_arrived  = 0;
    int    n_accepted       = 0;
    int    n_delivered      = 0;
    int    n_unfinished     = 0;
    double total_distance_m = 0.0;
    int    pairing_violations  = 0;
    int    capacity_violations = 0;
    int    n_agents_used    = 0;
    long long wallclock_ms  = 0;

    // Spatial complexity
    double bbox_area_km2          = 0.0;  // bbox of accepted pickup+delivery points
    double mean_pd_distance_m     = 0.0;  // mean direct P→D distance over accepted tasks
    double mean_inter_pickup_m    = 0.0;  // mean nearest-neighbour pickup distance
    double convex_hull_area_km2   = 0.0;  // convex hull of all served pickup+delivery
    double agent_route_overlap    = 0.0;  // sum of pairwise route overlap area proxy

    // ── Task-completion quality (lifted directly from EpisodeRunner metrics) ──
    float throughput_rate             = 0.f;  // delivered / arrived
    float accept_rate                 = 0.f;  // accepted / offered
    float latency_mean                = 0.f;  // mean steps appearance → completion
    float mean_wait_steps             = 0.f;  // mean steps appearance → pickup arrival
    float mean_trip_steps             = 0.f;  // mean steps pickup → delivery
    float mean_road_pd_m              = 0.f;  // mean A* road distance P→D
    float delivery_route_efficiency   = 0.f;  // ideal / actual ∈ (0,1]; 1=no detour

    // Traces (only populated when needed)
    std::vector<TaskTrace>  tasks;
    std::vector<AgentTrace> agents;
};

struct CityHandle {
    const CityConfig* config = nullptr;
    GeoBox            geobox;
    std::unique_ptr<Pathfinder> pathfinder;
};

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

constexpr double kPi = 3.14159265358979323846;

static double haversine_m(double lat1, double lon1, double lat2, double lon2) {
    constexpr double R = 6371000.0;
    const double dlat = (lat2 - lat1) * kPi / 180.0;
    const double dlon = (lon2 - lon1) * kPi / 180.0;
    const double la1  = lat1 * kPi / 180.0;
    const double la2  = lat2 * kPi / 180.0;
    const double a = std::sin(dlat/2)*std::sin(dlat/2)
                   + std::cos(la1)*std::cos(la2)*std::sin(dlon/2)*std::sin(dlon/2);
    return R * 2.0 * std::atan2(std::sqrt(a), std::sqrt(1-a));
}

static EpisodeConfig make_test_episode_config(const CityConfig& cc, int max_tasks) {
    EpisodeConfig ep;
    ep.city = &cc;
    ep.speed_mps = 5.0f;
    ep.min_task_dist_m = 150.f;
    ep.max_task_dist_m = 1500.f;
    ep.hot_zone_radius = 300.f;
    // Mono-agent comparison: a single agent absorbs ALL tasks across phases,
    // so the planning algorithm is the only variable between modes.
    // Capacity is set far above any regime's max_tasks (high=30, peak ≈75 with
    // density_mult up to 2.5) so capacity-aware insertion in receive_task() can
    // never refuse a task — every offered task IS assigned to the lone agent.
    // This isolates the planner: MCA/DH/DbVNS/ALNS each plan over the SAME set
    // of accepted tasks, with no allocation skew from capacity overflow.
    ep.max_tasks_per_agent = 200;
    const float per_phase = static_cast<float>(max_tasks) / 3.f;
    ep.phases = {
        { 1200, per_phase, 1, 1, 0.0f, 2 },
        { 1200, per_phase, 1, 1, 0.5f, 3 },
        { 1200, per_phase, 1, 1, 1.0f, 3 },
    };
    return ep;
}

// Per-seed saturation regime for mono-agent planning comparison.
//   low    : few tasks, slow arrival rate → agent should deliver all
//   medium : enough tasks to keep the agent busy → some unfinished expected
//   high   : oversaturated → many unfinished, planner quality decisive
struct RegimeConfig {
    std::string label;
    int   max_tasks;       // total tasks generated across the episode
    float density_min;
    float density_max;
};

static const RegimeConfig kRegimes[3] = {
    { "low_saturation",    4,  0.4f, 0.7f },   // ~2–3 tasks total
    { "medium_saturation", 12, 0.8f, 1.2f },   // ~10–14 tasks
    { "high_saturation",   30, 1.5f, 2.5f },   // ~45–75 tasks — overload
};

// Decorrelate regime from city: rotating regime every `n_cities` seeds so each
// (city × regime) combination is exercised across the seed batch.
static const RegimeConfig& regime_for(int seed_idx, int n_cities) {
    int r = (seed_idx / std::max(1, n_cities)) % 3;
    return kRegimes[r];
}

static EpisodeScenario sample_scenario(std::mt19937& rng,
                                        const RegimeConfig& reg) {
    std::uniform_real_distribution<float> ud(reg.density_min, reg.density_max);
    EpisodeScenario s;
    s.density_mult = ud(rng);
    s.agents_mult  = 1.0f;   // mono-agent: never scale fleet size
    s.label        = reg.label.c_str();
    return s;
}

// Convex hull (Andrew's monotone chain) for spatial complexity.
struct Pt2 { double x, y; };
static double cross(const Pt2& O, const Pt2& A, const Pt2& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}
static std::vector<Pt2> convex_hull(std::vector<Pt2> pts) {
    const int n = static_cast<int>(pts.size());
    if (n < 3) return pts;
    std::sort(pts.begin(), pts.end(),
              [](const Pt2& a, const Pt2& b){
                  return (a.x != b.x) ? a.x < b.x : a.y < b.y;
              });
    std::vector<Pt2> h(2*n);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(h[k-2], h[k-1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    for (int i = n-2, t = k+1; i >= 0; --i) {
        while (k >= t && cross(h[k-2], h[k-1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    h.resize(k - 1);
    return h;
}
static double polygon_area(const std::vector<Pt2>& poly) {
    double s = 0;
    const int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        const Pt2& a = poly[i];
        const Pt2& b = poly[(i+1) % n];
        s += a.x * b.y - b.x * a.y;
    }
    return std::abs(s) * 0.5;
}

// ────────────────────────────────────────────────────────────────────────────
// City loader (cached per-process)
// ────────────────────────────────────────────────────────────────────────────

static std::unique_ptr<CityHandle> load_city(const CityConfig* cc,
                                              const std::string& cache_root) {
    auto h = std::make_unique<CityHandle>();
    h->config = cc;
    const std::string cache_path = cache_root + "/" + cc->name + ".json";
    if (GeoBoxManager::cache_exists(cache_path)) {
        std::cout << "[Load] " << cc->name << " <- cache\n";
        h->geobox = GeoBoxManager::load_geobox(cache_path);
    } else {
        std::cout << "[Load] " << cc->name << " <- OSM " << cc->osm_path << "\n";
        h->geobox = create_geo_box(cc->osm_path,
                                    cc->bbox.min_lon, cc->bbox.min_lat,
                                    cc->bbox.max_lon, cc->bbox.max_lat);
        if (h->geobox.is_valid) {
            fs::create_directories(cache_root);
            GeoBoxManager::save_geobox(h->geobox, cache_path);
        }
    }
    if (!h->geobox.is_valid) return nullptr;
    h->pathfinder = std::make_unique<Pathfinder>(h->geobox);
    return h;
}

// ────────────────────────────────────────────────────────────────────────────
// Validity & spatial metrics
// ────────────────────────────────────────────────────────────────────────────

// Strong validation of solution feasibility.
//   - Pairing constraint  : pickup_step < delivery_step for every served task
//   - Capacity constraint : peak simultaneous carry ≤ max_carry per agent
//   - Temporal coherence  : pickup_step ≥ arrival_step, delivered_step > picked_step
//   - Spatial coherence   : agent moves only along the OSM road network — guaranteed
//                            by EdgeCursor + schedule_next_edge in the runtime, but
//                            we still cross-check that consecutive served-events of
//                            an agent (sorted by step) are reachable: their step
//                            delta ≥ ceil(haversine(p1, p2) / speed). A negative step
//                            delta or a leap shorter than physical minimum signals
//                            a teleportation bug.
//
// Any violation increments the respective counter and emits a stderr warning
// so the test fails loudly (visible in the report) rather than silently passing.
static void validate_traces(ComparisonResult& r, float speed_mps,
                             int max_carry_capacity) {
    // 1) Pairing + per-task temporal coherence
    for (const auto& t : r.tasks) {
        if (t.agent_id < 0) continue;  // refused tasks are not the planner's job
        // Accepted but never picked while delivery happened: impossible by code,
        // but defend against future regressions.
        if (t.delivered_step >= 0 && t.picked_step < 0) {
            ++r.pairing_violations;
            std::cerr << "[VALIDATE] task " << t.task_id
                      << " delivered without pickup\n";
        }
        // Pickup must come strictly before delivery in time.
        if (t.delivered_step >= 0 && t.picked_step >= 0 &&
            t.picked_step >= t.delivered_step) {
            ++r.pairing_violations;
            std::cerr << "[VALIDATE] task " << t.task_id
                      << " picked@" << t.picked_step
                      << " ≥ delivered@" << t.delivered_step << "\n";
        }
        // Pickup must be after the task was created (arrived in system).
        if (t.picked_step >= 0 && t.arrival_step >= 0 &&
            t.picked_step < t.arrival_step) {
            ++r.pairing_violations;
            std::cerr << "[VALIDATE] task " << t.task_id
                      << " picked@" << t.picked_step
                      << " before arrival@" << t.arrival_step << "\n";
        }
    }

    // 2) Per-agent capacity profile by replaying chronological events.
    struct Ev { int t; int delta; int task_id; };
    for (const auto& a : r.agents) {
        std::vector<Ev> evs;
        for (const auto& t : r.tasks) {
            if (t.agent_id != a.agent_id) continue;
            if (t.picked_step    >= 0) evs.push_back({t.picked_step,    +1, t.task_id});
            if (t.delivered_step >= 0) evs.push_back({t.delivered_step, -1, t.task_id});
        }
        std::sort(evs.begin(), evs.end(),
                  [](const Ev& x, const Ev& y){ return x.t < y.t; });
        int load = 0, peak = 0;
        for (const auto& e : evs) {
            load += e.delta;
            if (load > peak) peak = load;
            if (load < 0) {
                ++r.pairing_violations;
                std::cerr << "[VALIDATE] agent " << a.agent_id
                          << " negative load after task " << e.task_id
                          << " @step " << e.t << "\n";
            }
        }
        if (peak > max_carry_capacity) {
            r.capacity_violations += peak - max_carry_capacity;
            std::cerr << "[VALIDATE] agent " << a.agent_id
                      << " peak load " << peak
                      << " > max_carry " << max_carry_capacity << "\n";
        }
    }

    // 3) Spatial-temporal coherence per agent: consecutive served-events must
    //    be physically reachable. A node-to-node displacement at maximum speed
    //    needs at least haversine/speed steps. A shorter time delta = bug.
    for (const auto& a : r.agents) {
        std::vector<const TaskTrace*> evs;
        for (const auto& t : r.tasks) if (t.agent_id == a.agent_id) evs.push_back(&t);
        // Build (step, lat, lon) pairs for pickup/delivery events, sort by step.
        struct Pt { int step; double lat; double lon; int task_id; bool is_pu; };
        std::vector<Pt> pts;
        for (auto* t : evs) {
            if (t->picked_step >= 0)
                pts.push_back({t->picked_step,    t->pickup_lat,   t->pickup_lon,   t->task_id, true});
            if (t->delivered_step >= 0)
                pts.push_back({t->delivered_step, t->delivery_lat, t->delivery_lon, t->task_id, false});
        }
        std::sort(pts.begin(), pts.end(),
                  [](const Pt& x, const Pt& y){ return x.step < y.step; });
        for (size_t i = 1; i < pts.size(); ++i) {
            const double d = haversine_m(pts[i-1].lat, pts[i-1].lon,
                                          pts[i].lat,   pts[i].lon);
            const int dt = pts[i].step - pts[i-1].step;
            // Minimum physical time to cross d metres at speed_mps.
            const int min_steps = static_cast<int>(std::ceil(d / std::max(speed_mps, 0.1f)));
            // 20% tolerance for OSM path being longer than haversine.
            const int min_with_tol = static_cast<int>(min_steps * 0.8);
            if (dt < min_with_tol && d > 100.0) {
                ++r.pairing_violations;  // co-opted counter (spatial+temporal)
                std::cerr << "[VALIDATE] agent " << a.agent_id
                          << " teleport-suspect: " << pts[i-1].task_id
                          << " (" << (pts[i-1].is_pu ? "P" : "D") << ") "
                          << "→ " << pts[i].task_id
                          << " (" << (pts[i].is_pu ? "P" : "D") << ") "
                          << ": dist=" << static_cast<int>(d) << "m "
                          << "dt=" << dt << " steps (min " << min_steps << ")\n";
            }
        }
    }
}

static void compute_spatial_metrics(ComparisonResult& r) {
    // Filter to accepted tasks (those with an assigned agent).
    std::vector<const TaskTrace*> acc;
    for (const auto& t : r.tasks) if (t.agent_id >= 0) acc.push_back(&t);
    if (acc.empty()) return;

    // BBox area (km²) over pickup+delivery coords.
    double mn_lat=+1e9, mx_lat=-1e9, mn_lon=+1e9, mx_lon=-1e9;
    for (auto* t : acc) {
        mn_lat = std::min({mn_lat, t->pickup_lat,  t->delivery_lat});
        mx_lat = std::max({mx_lat, t->pickup_lat,  t->delivery_lat});
        mn_lon = std::min({mn_lon, t->pickup_lon,  t->delivery_lon});
        mx_lon = std::max({mx_lon, t->pickup_lon,  t->delivery_lon});
    }
    {
        const double w = haversine_m(mn_lat, mn_lon, mn_lat, mx_lon);
        const double h = haversine_m(mn_lat, mn_lon, mx_lat, mn_lon);
        r.bbox_area_km2 = (w * h) / 1e6;
    }

    // Mean direct P→D distance.
    double sum_pd = 0.0;
    for (auto* t : acc)
        sum_pd += haversine_m(t->pickup_lat, t->pickup_lon,
                               t->delivery_lat, t->delivery_lon);
    r.mean_pd_distance_m = sum_pd / acc.size();

    // Mean nearest-neighbour pickup distance (spatial spread of demand).
    double sum_nn = 0.0; int n_nn = 0;
    for (size_t i = 0; i < acc.size(); ++i) {
        double best = 1e18;
        for (size_t j = 0; j < acc.size(); ++j) {
            if (i == j) continue;
            const double d = haversine_m(acc[i]->pickup_lat, acc[i]->pickup_lon,
                                         acc[j]->pickup_lat, acc[j]->pickup_lon);
            if (d < best) best = d;
        }
        if (best < 1e17) { sum_nn += best; ++n_nn; }
    }
    r.mean_inter_pickup_m = (n_nn > 0) ? (sum_nn / n_nn) : 0.0;

    // Convex hull area (km²) over all served points — local-frame projection.
    std::vector<Pt2> pts;
    pts.reserve(acc.size() * 2);
    const double cos_lat = std::cos((mn_lat + mx_lat) * 0.5 * kPi / 180.0);
    for (auto* t : acc) {
        pts.push_back({t->pickup_lon   * 111320.0 * cos_lat,
                       t->pickup_lat   * 111320.0});
        pts.push_back({t->delivery_lon * 111320.0 * cos_lat,
                       t->delivery_lat * 111320.0});
    }
    auto hull = convex_hull(pts);
    r.convex_hull_area_km2 = polygon_area(hull) / 1e6;

    // Route overlap proxy: how much agents' bbox overlap with each other.
    // For each agent build the bbox of its leg endpoints; pairwise IoU sum.
    struct AB { double mn_x, mx_x, mn_y, mx_y; bool valid=false; };
    std::vector<AB> abs(r.agents.size());
    for (size_t i = 0; i < r.agents.size(); ++i) {
        AB& b = abs[i];
        b.mn_x = +1e18; b.mx_x = -1e18; b.mn_y = +1e18; b.mx_y = -1e18;
        for (const auto& l : r.agents[i].legs) {
            const double fx = l.from_lon * 111320.0 * cos_lat;
            const double fy = l.from_lat * 111320.0;
            const double tx = l.to_lon   * 111320.0 * cos_lat;
            const double ty = l.to_lat   * 111320.0;
            b.mn_x = std::min({b.mn_x, fx, tx});
            b.mx_x = std::max({b.mx_x, fx, tx});
            b.mn_y = std::min({b.mn_y, fy, ty});
            b.mx_y = std::max({b.mx_y, fy, ty});
            b.valid = true;
        }
    }
    double overlap = 0.0;
    for (size_t i = 0; i < abs.size(); ++i) {
        if (!abs[i].valid) continue;
        for (size_t j = i + 1; j < abs.size(); ++j) {
            if (!abs[j].valid) continue;
            const double ox = std::max(0.0,
                std::min(abs[i].mx_x, abs[j].mx_x) - std::max(abs[i].mn_x, abs[j].mn_x));
            const double oy = std::max(0.0,
                std::min(abs[i].mx_y, abs[j].mx_y) - std::max(abs[i].mn_y, abs[j].mn_y));
            overlap += (ox * oy) / 1e6;
        }
    }
    r.agent_route_overlap = overlap;
}

// ────────────────────────────────────────────────────────────────────────────
// SVG renderer
// ────────────────────────────────────────────────────────────────────────────

static std::string time_color(float t01) {
    t01 = std::clamp(t01, 0.f, 1.f);
    int r = static_cast<int>(40  + t01 * 215);
    int g = static_cast<int>(120 - t01 * 50);
    int b = static_cast<int>(200 - t01 * 160);
    std::ostringstream os;
    os << "rgb(" << r << "," << g << "," << b << ")";
    return os.str();
}

static void render_comparison_svg(const ComparisonResult& r,
                                   const GeoBox& gb,
                                   const std::string& out_path,
                                   int W = 1100, int H = 1100) {
    double mn_lat = +1e9, mx_lat = -1e9, mn_lon = +1e9, mx_lon = -1e9;
    for (const auto& t : r.tasks) {
        if (t.pickup_lat == 0 && t.pickup_lon == 0) continue;
        mn_lat = std::min({mn_lat, t.pickup_lat,  t.delivery_lat});
        mx_lat = std::max({mx_lat, t.pickup_lat,  t.delivery_lat});
        mn_lon = std::min({mn_lon, t.pickup_lon,  t.delivery_lon});
        mx_lon = std::max({mx_lon, t.pickup_lon,  t.delivery_lon});
    }
    if (mn_lat >= mx_lat || mn_lon >= mx_lon) {
        std::cerr << "[render] degenerate bbox, skip " << out_path << "\n";
        return;
    }
    const double pad_lat = (mx_lat - mn_lat) * 0.06;
    const double pad_lon = (mx_lon - mn_lon) * 0.06;
    mn_lat -= pad_lat; mx_lat += pad_lat;
    mn_lon -= pad_lon; mx_lon += pad_lon;
    constexpr int LEG = 140;
    auto px = [&](double lon){ return (lon - mn_lon)/(mx_lon - mn_lon) * W; };
    auto py = [&](double lat){ return H - (lat - mn_lat)/(mx_lat - mn_lat) * H; };

    std::ofstream svg(out_path);
    if (!svg) { std::cerr << "[render] cannot open " << out_path << "\n"; return; }
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W
        << "\" height=\"" << (H + LEG) << "\">\n";
    svg << "<rect width=\"" << W << "\" height=\"" << H << "\" fill=\"#fafbfc\"/>\n";

    int sampled = 0;
    for (const auto& [wid, way] : gb.data.ways) {
        if (sampled++ > 8000) break;
        if (way.points.size() < 2) continue;
        svg << "<polyline fill=\"none\" stroke=\"#dde3ea\" stroke-width=\"0.6\" points=\"";
        for (const auto& p : way.points)
            svg << std::fixed << std::setprecision(2)
                << px(p.lon) << "," << py(p.lat) << " ";
        svg << "\"/>\n";
    }

    const float ep_steps = 3600.f;
    for (const auto& a : r.agents) {
        for (const auto& l : a.legs) {
            const float t01 = std::clamp(l.arr_step / ep_steps, 0.f, 1.f);
            svg << "<line x1=\"" << px(l.from_lon) << "\" y1=\"" << py(l.from_lat)
                << "\" x2=\""    << px(l.to_lon)   << "\" y2=\"" << py(l.to_lat)
                << "\" stroke=\"" << time_color(t01) << "\""
                << " stroke-width=\"1.4\" opacity=\"0.85\"/>\n";
        }
    }
    for (const auto& t : r.tasks) {
        if (t.pickup_lat == 0 && t.pickup_lon == 0) continue;
        const float t01 = std::clamp(t.arrival_step / ep_steps, 0.f, 1.f);
        const std::string c = time_color(t01);
        svg << "<circle cx=\"" << px(t.pickup_lon) << "\" cy=\"" << py(t.pickup_lat)
            << "\" r=\"5\" fill=\"" << c << "\" stroke=\"white\" stroke-width=\"1.2\"/>\n";
        svg << "<rect x=\""   << (px(t.delivery_lon)-5) << "\" y=\"" << (py(t.delivery_lat)-5)
            << "\" width=\"10\" height=\"10\" fill=\"white\" stroke=\"" << c
            << "\" stroke-width=\"2.2\"/>\n";
        svg << "<line x1=\"" << px(t.pickup_lon) << "\" y1=\"" << py(t.pickup_lat)
            << "\" x2=\""    << px(t.delivery_lon) << "\" y2=\"" << py(t.delivery_lat)
            << "\" stroke=\"" << c << "\" stroke-width=\"0.8\" stroke-dasharray=\"3,3\" opacity=\"0.45\"/>\n";
    }

    svg << "<rect y=\"" << H << "\" width=\"" << W << "\" height=\"" << LEG
        << "\" fill=\"#edf0f3\"/>\n";
    svg << "<text x=\"12\" y=\"" << (H + 20)
        << "\" font-family=\"monospace\" font-size=\"13\" fill=\"#222\">"
        << "seed=" << r.seed << "  city=" << r.city
        << "  mode=" << r.mode
        << "  density=" << std::fixed << std::setprecision(2) << r.density_mult
        << "  agents=" << r.agents_mult
        << "  (" << r.scenario_label << ")"
        << "</text>\n";
    svg << "<text x=\"12\" y=\"" << (H + 40)
        << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#222\">"
        << "arrived=" << r.n_tasks_arrived
        << " accepted=" << r.n_accepted
        << " delivered=" << r.n_delivered
        << " unfinished=" << r.n_unfinished
        << " dist=" << std::setprecision(0) << r.total_distance_m << "m"
        << " agents=" << r.n_agents_used
        << "</text>\n";
    svg << "<text x=\"12\" y=\"" << (H + 58)
        << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#444\">"
        << "spatial: bbox=" << std::fixed << std::setprecision(2) << r.bbox_area_km2 << "km²"
        << " hull=" << r.convex_hull_area_km2 << "km²"
        << " meanPD=" << std::setprecision(0) << r.mean_pd_distance_m << "m"
        << " meanNN=" << r.mean_inter_pickup_m << "m"
        << "</text>\n";
    svg << "<text x=\"12\" y=\"" << (H + 76)
        << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#555\">"
        << "validity: pairing_viol=" << r.pairing_violations
        << " capacity_viol=" << r.capacity_violations
        << " wallclock=" << r.wallclock_ms << "ms"
        << "</text>\n";

    const int bx = 12, by = H + 95, bw = 220, bh = 12;
    for (int s = 0; s < 8; ++s) {
        float t01 = s / 7.f;
        svg << "<rect x=\"" << (bx + s * bw / 8) << "\" y=\"" << by
            << "\" width=\"" << (bw / 8 + 1) << "\" height=\"" << bh
            << "\" fill=\"" << time_color(t01) << "\"/>\n";
    }
    svg << "<text x=\"" << (bx + bw + 6) << "\" y=\"" << (by + 10)
        << "\" font-family=\"monospace\" font-size=\"11\" fill=\"#444\">"
        << "time gradient: blue=t=0  red=t=" << static_cast<int>(ep_steps) << "</text>\n";
    svg << "</svg>\n";
}

// ────────────────────────────────────────────────────────────────────────────
// Trace extraction from a finished EpisodeRunner
// ────────────────────────────────────────────────────────────────────────────

static ComparisonResult run_one(EpisodeRunner& runner,
                                 const GeoBox& gb,
                                 PolicyMode mode,
                                 const std::string& mode_name,
                                 const EpisodeScenario& sc) {
    runner.train_mode  = false;
    runner.policy_mode = mode;

    auto t0 = std::chrono::steady_clock::now();
    RunResult res = runner.run(/*city_index*/0, /*num_cities*/1, sc);
    auto t1 = std::chrono::steady_clock::now();

    ComparisonResult cr;
    cr.mode = mode_name;
    cr.wallclock_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cr.n_tasks_arrived = res.metrics.tasks_appeared;
    cr.n_delivered     = res.metrics.tasks_completed;
    cr.n_accepted      = static_cast<int>(std::round(
                            res.metrics.accept_rate * res.metrics.tasks_appeared));
    cr.n_unfinished    = cr.n_accepted - cr.n_delivered;

    // Task-completion quality (already finalised by EpisodeRunner).
    cr.throughput_rate           = res.metrics.throughput_rate;
    cr.accept_rate               = res.metrics.accept_rate;
    cr.latency_mean              = res.metrics.latency_mean;
    cr.mean_wait_steps           = res.metrics.mean_wait_steps;
    cr.mean_trip_steps           = res.metrics.mean_trip_steps;
    cr.mean_road_pd_m            = res.metrics.mean_road_pd_m;
    cr.delivery_route_efficiency = res.metrics.delivery_route_efficiency;

    auto node_coords = [&](osmium::object_id_type nid,
                           double& lat, double& lon) {
        auto it = gb.data.nodes.find(nid);
        if (it == gb.data.nodes.end()) { lat = 0; lon = 0; return false; }
        lat = it->second.lat; lon = it->second.lon; return true;
    };
    auto collect_task = [&](const PDPTask* t) {
        if (!t) return;
        TaskTrace tt;
        tt.task_id        = t->task_id;
        tt.arrival_step   = t->timeline.created_step;
        tt.picked_step    = t->timeline.picked_step;
        tt.delivered_step = t->timeline.delivered_step;
        tt.agent_id       = t->agent_id;
        tt.pickup_node    = t->pickup.id;
        tt.delivery_node  = t->delivery.id;
        node_coords(t->pickup.id,   tt.pickup_lat,   tt.pickup_lon);
        node_coords(t->delivery.id, tt.delivery_lat, tt.delivery_lon);
        cr.tasks.push_back(tt);
    };
    for (const PDPTask* t : runner.memory().finished_tasks)  collect_task(t);
    for (const PDPTask* t : runner.memory().allocated_tasks) collect_task(t);
    for (const PDPTask* t : runner.memory().available_tasks) collect_task(t);

    // ── Build per-agent HISTORICAL trajectory ─────────────────────────────
    //
    // Previously this used `solution.sequence` which only contains undelivered
    // remainder at episode end — giving a misleading "teleport" view in the
    // SVG. The correct approach is to replay served events in chronological
    // order from the cr.tasks list. Each (step, node) is a real waypoint the
    // agent visited. The starting position completes the timeline.
    //
    // Includes BOTH delivered and only-picked tasks, plus the tail of the
    // remaining plan (if any) for completeness.
    struct Ev { int step; osmium::object_id_type node; double lat, lon; };
    std::unordered_map<int, std::vector<Ev>> agent_events;
    for (const auto& t : cr.tasks) {
        if (t.agent_id < 0) continue;
        if (t.picked_step >= 0)
            agent_events[t.agent_id].push_back(
                {t.picked_step, t.pickup_node, t.pickup_lat, t.pickup_lon});
        if (t.delivered_step >= 0)
            agent_events[t.agent_id].push_back(
                {t.delivered_step, t.delivery_node, t.delivery_lat, t.delivery_lon});
    }
    int n_used = 0;
    for (const auto& ap : runner.agents()) {
        if (!ap) continue;
        AgentTrace at;
        at.agent_id = ap->agent_id;
        // Start from the agent's spawn (current_node holds final pos; use it
        // as approximation of last-known position).
        osmium::object_id_type prev_node = ap->current_node;
        int prev_step = 0;
        double prev_lat=0, prev_lon=0;
        node_coords(prev_node, prev_lat, prev_lon);

        auto& evs = agent_events[ap->agent_id];
        std::sort(evs.begin(), evs.end(),
                  [](const Ev& a, const Ev& b){ return a.step < b.step; });

        for (const auto& e : evs) {
            AgentTrace::Leg leg;
            leg.from = prev_node;       leg.to       = e.node;
            leg.dep_step = prev_step;   leg.arr_step = e.step;
            leg.from_lat = prev_lat;    leg.from_lon = prev_lon;
            leg.to_lat   = e.lat;       leg.to_lon   = e.lon;
            leg.length_m = haversine_m(prev_lat, prev_lon, e.lat, e.lon);
            at.legs.push_back(leg);
            prev_node = e.node;
            prev_step = e.step;
            prev_lat  = e.lat;
            prev_lon  = e.lon;
        }

        // Append the still-planned tail (unfinished commitments) to make the
        // visualisation complete — these legs reflect agent.solution.sequence.
        for (const auto& step : ap->solution.sequence) {
            double to_lat=0, to_lon=0;
            if (!node_coords(step.node.id, to_lat, to_lon)) continue;
            if (step.estimated_arrival <= prev_step) continue;  // already covered
            AgentTrace::Leg leg;
            leg.from = prev_node;     leg.to       = step.node.id;
            leg.dep_step = prev_step; leg.arr_step = step.estimated_arrival;
            leg.from_lat = prev_lat;  leg.from_lon = prev_lon;
            leg.to_lat   = to_lat;    leg.to_lon   = to_lon;
            leg.length_m = haversine_m(prev_lat, prev_lon, to_lat, to_lon);
            at.legs.push_back(leg);
            prev_node = step.node.id;
            prev_step = step.estimated_arrival;
            prev_lat  = to_lat;
            prev_lon  = to_lon;
        }
        if (!at.legs.empty()) ++n_used;
        cr.agents.push_back(at);
    }
    cr.n_agents_used = n_used;
    cr.total_distance_m = 0.0;
    for (const auto& a : cr.agents)
        for (const auto& l : a.legs)
            cr.total_distance_m += l.length_m;
    return cr;
}

// ────────────────────────────────────────────────────────────────────────────
// CSV writers
// ────────────────────────────────────────────────────────────────────────────

static void write_tasks_csv(const ComparisonResult& cr, const std::string& path) {
    std::ofstream f(path);
    if (!f) return;
    f << "task_id,arrival_step,picked_step,delivered_step,agent_id,"
      << "pickup_node,pickup_lat,pickup_lon,"
      << "delivery_node,delivery_lat,delivery_lon\n";
    for (const auto& t : cr.tasks) {
        f << t.task_id << "," << t.arrival_step << ","
          << t.picked_step << "," << t.delivered_step << "," << t.agent_id << ","
          << t.pickup_node << ","
          << std::fixed << std::setprecision(7)
          << t.pickup_lat << "," << t.pickup_lon << ","
          << t.delivery_node << ","
          << t.delivery_lat << "," << t.delivery_lon << "\n";
    }
}

static void write_agents_csv(const ComparisonResult& cr, const std::string& path) {
    std::ofstream f(path);
    if (!f) return;
    f << "agent_id,leg_idx,from_node,to_node,dep_step,arr_step,"
      << "from_lat,from_lon,to_lat,to_lon,length_m\n";
    for (const auto& a : cr.agents) {
        for (size_t i = 0; i < a.legs.size(); ++i) {
            const auto& l = a.legs[i];
            f << a.agent_id << "," << i << ","
              << l.from << "," << l.to << ","
              << l.dep_step << "," << l.arr_step << ","
              << std::fixed << std::setprecision(7)
              << l.from_lat << "," << l.from_lon << ","
              << l.to_lat   << "," << l.to_lon   << ","
              << std::setprecision(1) << l.length_m << "\n";
        }
    }
}

static void write_summary_header(std::ofstream& f) {
    f << "seed,city,density_mult,agents_mult,scenario,mode,"
      // Performance counters
      << "arrived,accepted,delivered,unfinished,total_dist_m,"
      << "pairing_viol,capacity_viol,n_agents_used,"
      // Task-completion quality
      << "throughput_rate,accept_rate,latency_mean,mean_wait_steps,"
      << "mean_trip_steps,mean_road_pd_m,delivery_route_efficiency,"
      // Time complexity
      << "wallclock_ms,compute_time_per_task_ms,compute_time_per_decision_us,"
      // Spatial complexity
      << "bbox_area_km2,convex_hull_km2,mean_pd_m,mean_nn_pickup_m,"
      << "agent_route_overlap_km2\n";
}
static void append_summary_row(std::ofstream& f, const ComparisonResult& cr) {
    const double per_task =
        cr.n_tasks_arrived > 0 ? static_cast<double>(cr.wallclock_ms) / cr.n_tasks_arrived
                                : 0.0;
    // Each task arrival triggers one TAM-driven decision per active agent;
    // a coarse proxy for "per-decision compute time" (in µs) divides total
    // wallclock by (n_arrived × n_agents_used) — useful for scaling analysis.
    const double per_decision_us =
        (cr.n_tasks_arrived > 0 && cr.n_agents_used > 0)
            ? (static_cast<double>(cr.wallclock_ms) * 1000.0)
                / (cr.n_tasks_arrived * cr.n_agents_used)
            : 0.0;
    f << cr.seed << "," << cr.city << ","
      << std::fixed << std::setprecision(3) << cr.density_mult << ","
      << cr.agents_mult << "," << cr.scenario_label << "," << cr.mode << ","
      << cr.n_tasks_arrived << "," << cr.n_accepted << ","
      << cr.n_delivered << "," << cr.n_unfinished << ","
      << std::setprecision(1) << cr.total_distance_m << ","
      << cr.pairing_violations << "," << cr.capacity_violations << ","
      << cr.n_agents_used << ","
      // Task-completion quality
      << std::setprecision(4)
      << cr.throughput_rate           << ","
      << cr.accept_rate               << ","
      << std::setprecision(2)
      << cr.latency_mean              << ","
      << cr.mean_wait_steps           << ","
      << cr.mean_trip_steps           << ","
      << std::setprecision(1)
      << cr.mean_road_pd_m            << ","
      << std::setprecision(4)
      << cr.delivery_route_efficiency << ","
      // Time complexity
      << cr.wallclock_ms << ","
      << std::setprecision(2) << per_task << ","
      << per_decision_us << ","
      // Spatial complexity
      << std::setprecision(3) << cr.bbox_area_km2 << ","
      << cr.convex_hull_area_km2 << ","
      << std::setprecision(1) << cr.mean_pd_distance_m << ","
      << cr.mean_inter_pickup_m << ","
      << std::setprecision(3) << cr.agent_route_overlap << "\n";
}

}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Public entry point — BATCH
// ────────────────────────────────────────────────────────────────────────────

int run_planning_comparison_test(uint32_t base_seed,
                                  int max_tasks,
                                  const std::string& osm_root,
                                  const std::string& cache_root,
                                  const std::string& output_dir,
                                  int n_seeds,
                                  int detail_every,
                                  const std::vector<std::string>& cities_in,
                                  float density_min,
                                  float density_max)
{
    if (max_tasks <= 0 || max_tasks > 100) max_tasks = 25;
    if (n_seeds   <= 0) n_seeds = 100;
    if (detail_every <= 0) detail_every = 10;
    if (base_seed == 0) {
        std::random_device rd;
        base_seed = rd();
    }

    std::vector<std::string> cities = cities_in;
    if (cities.empty())
        cities = { "Tokyo_Small", "Kyoto_Small", "LosAngeles_Small" };

    std::cout << "[PlanningBatch] base_seed=" << base_seed
              << "  n_seeds=" << n_seeds
              << "  detail_every=" << detail_every
              << "  max_tasks=" << max_tasks
              << "  cities=" << cities.size()
              << "  density∈[" << density_min << "," << density_max << "]\n";

    fs::create_directories(output_dir);
    fs::create_directories(output_dir + "/details");

    // Load all cities once (geobox cached on disk after first run).
    CityRegistry::set_osm_root(osm_root);
    const auto train = CityRegistry::train_cities();
    std::vector<std::unique_ptr<CityHandle>> handles;
    for (const std::string& name : cities) {
        const CityConfig* cc = nullptr;
        for (auto* c : train) if (c->name == name) { cc = c; break; }
        if (!cc) {
            std::cerr << "[PlanningBatch] city '" << name
                      << "' not in CityRegistry — skipping.\n";
            continue;
        }
        auto h = load_city(cc, cache_root);
        if (!h) {
            std::cerr << "[PlanningBatch] failed to load " << name << " — skipping.\n";
            continue;
        }
        handles.push_back(std::move(h));
    }
    if (handles.empty()) {
        std::cerr << "[PlanningBatch] no city loaded — abort.\n";
        return 2;
    }

    // Summary CSV (one row per seed × mode).
    const std::string summary_path = output_dir + "/planning_summary.csv";
    std::ofstream summary(summary_path);
    if (!summary) {
        std::cerr << "[PlanningBatch] cannot open " << summary_path << "\n";
        return 3;
    }
    write_summary_header(summary);

    const std::vector<std::pair<PolicyMode, std::string>> modes = {
        // RMCA(r) = cheapest insertion (eq. 12) + anytime LNS Algorithm 3
        // (destroy-3-random + greedy cheapest repair, 30 iters), local to each
        // agent's queue. Paper-faithful for single-agent regret since regret
        // collapses to cheapest-first in mono-route. See DeliveryAgent.cpp
        // for the LNS body and EpisodeRunner.cpp MCA branch comment.
        { PolicyMode::MCA,           "RMCA_Chen2021_LNS" },
        { PolicyMode::DoubleHorizon, "DoubleHorizon_MitrovicMinic" },
        { PolicyMode::DbVNS,         "DbVNS_lifelong_replan" },
        { PolicyMode::ALNS,          "ALNS_RopkePisinger" },
    };

    auto t_global0 = std::chrono::steady_clock::now();

    for (int s = 0; s < n_seeds; ++s) {
        const uint32_t seed = base_seed + static_cast<uint32_t>(s);
        const int      n_cities = static_cast<int>(handles.size());
        const int      city_idx = s % n_cities;
        CityHandle&    H = *handles[city_idx];

        const RegimeConfig& reg = regime_for(s, n_cities);
        std::mt19937 sc_rng(seed ^ 0x9e3779b9u);
        const EpisodeScenario sc = sample_scenario(sc_rng, reg);

        EpisodeConfig ep = make_test_episode_config(*H.config, reg.max_tasks);

        const bool save_details = (s % detail_every) == 0;

        std::cout << "\n[" << (s + 1) << "/" << n_seeds
                  << "] seed=" << seed
                  << "  city=" << H.config->name
                  << "  regime=" << reg.label
                  << "  max_tasks=" << reg.max_tasks
                  << "  density=" << std::fixed << std::setprecision(2) << sc.density_mult
                  << "\n";

        for (const auto& [mode, name] : modes) {
            EpisodeRunner runner(ep, H.geobox, *H.pathfinder, seed);
            ComparisonResult cr = run_one(runner, H.geobox, mode, name, sc);
            cr.seed            = seed;
            cr.city            = H.config->name;
            cr.density_mult    = sc.density_mult;
            cr.agents_mult     = sc.agents_mult;
            cr.scenario_label  = sc.label;
            validate_traces(cr, ep.speed_mps, ep.max_tasks_per_agent);
            compute_spatial_metrics(cr);

            std::cout << "    " << name
                      << "  arr=" << cr.n_tasks_arrived
                      << " del=" << cr.n_delivered
                      << " unf=" << cr.n_unfinished
                      << " hull=" << std::fixed << std::setprecision(2)
                      << cr.convex_hull_area_km2 << "km²"
                      << " " << cr.wallclock_ms << "ms\n";

            append_summary_row(summary, cr);
            summary.flush();

            if (save_details) {
                std::ostringstream stem;
                stem << "seed_" << std::setw(4) << std::setfill('0') << s
                     << "_" << H.config->name << "_" << name;
                const std::string base = output_dir + "/details/" + stem.str();
                write_tasks_csv(cr,  base + "_tasks.csv");
                write_agents_csv(cr, base + "_agents.csv");
                render_comparison_svg(cr, H.geobox, base + ".svg");
            }
        }
    }

    auto t_global1 = std::chrono::steady_clock::now();
    const long long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  t_global1 - t_global0).count();
    summary.close();
    std::cout << "\n[PlanningBatch] done. "
              << "total wallclock=" << (total_ms / 1000) << " s\n"
              << "summary CSV: " << summary_path << "\n";
    return 0;
}
