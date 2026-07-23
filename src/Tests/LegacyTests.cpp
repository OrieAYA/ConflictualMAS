#include "LegacyTests.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Legacy/Common/Pathfinding.hpp"
#include "Legacy/Common/Memory.hpp"
#include "Legacy/Agent/Agent.hpp"
#include "Legacy/Agent/MetaAgent.hpp"
#include "Legacy/Agent/GlobalSolutionConstructor.hpp"
#include "Legacy/MHProcs/VNS.hpp"
#include "Legacy/MHProcs/PSO.hpp"
#include "utility.hpp"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <limits>
#include <string>

// ============================================================================
// LOCATION HELPER
// ============================================================================

static void select_location(std::string& location_name,
                             double& min_lat, double& min_lon,
                             double& max_lat, double& max_lon,
                             std::string& bbox)
{
    std::cout << "\n=== Location ===\n";
    std::cout << "1. Asakusa\n2. Shibuya\n3. Shinjuku\nChoice [1]: ";
    int choice = 1;
    std::cin >> choice;

    switch (choice) {
        case 2:
            location_name = "shibuya";
            min_lat = 35.658; min_lon = 139.699;
            max_lat = 35.661; max_lon = 139.704;
            bbox = "139.699,35.658,139.704,35.661";
            break;
        case 3:
            location_name = "shinjuku";
            min_lat = 35.689; min_lon = 139.698;
            max_lat = 35.702; max_lon = 139.710;
            bbox = "139.698,35.689,139.710,35.702";
            break;
        default:
            location_name = "asakusa";
            min_lat = 35.705; min_lon = 139.785;
            max_lat = 35.718; max_lon = 139.800;
            bbox = "139.785,35.705,139.800,35.718";
            break;
    }
    std::cout << "Location: " << location_name << "\n";
}

// ============================================================================
// G — GLOBAL SOLUTION CONSTRUCTOR
// ============================================================================

void legacy_run_global(const std::string& cache_dir)
{
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST GLOBAL SOLUTION CONSTRUCTOR\n";
    std::cout << std::string(80, '=') << "\n\n";

    const std::string input_cache_path  = cache_dir + "\\asakusa_test_agent_raw.json";
    const std::string output_cache_path = cache_dir + "\\global_solution_result.json";
    const std::string output_map_name   = "global_solution_map";

    GeoBox geo_box = GeoBoxManager::load_geobox(input_cache_path);
    if (!geo_box.is_valid) {
        std::cerr << "Failed to load: " << input_cache_path << "\n";
        return;
    }

    Pathfinder pathfinder(geo_box);
    GlobalMemory global_memory(geo_box, pathfinder);
    global_memory.initialize_group_stats();

    global_memory.length_constraint  = 1000.0f;
    global_memory.search_coefficient = 1.4f;

    GlobalSolutionConstructor constructor(geo_box, pathfinder, global_memory);

    MetaAgentConfig base("Agent_Config");
    base.params.max_divergence_from_gbest  = 0.3;
    base.params.max_iterations_per_agent   = 100;
    base.params.tabu_list_size             = 20;

    auto make_config = [&](const char* name, double t, double r, double s) {
        MetaAgentConfig c = base;
        c.name = name;
        c.characteristics[1] = t;
        c.characteristics[2] = r;
        c.characteristics[3] = s;
        constructor.add_meta_agent_config(c);
    };
    make_config("Agent_Temples",     100, 0,   0);
    make_config("Agent_Restaurants", 0,   100, 0);
    make_config("Agent_Shops",       0,   0,   100);
    make_config("Agent_All",         100, 100, 100);

    auto t0 = std::chrono::high_resolution_clock::now();
    constructor.initialize_reward_densities();
    constructor.run_all_meta_agents();
    auto t1 = std::chrono::high_resolution_clock::now();

    GlobalSolution gbest = constructor.tabu_search(50, 10);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };

    std::cout << "\nTotal: " << ms(t0, t2) << " ms"
              << "  (meta-agents: " << ms(t0, t1)
              << " ms, tabu: " << ms(t1, t2) << " ms)\n";
    constructor.print_summary();

    // Mark routes — best meta-agent solutions
    GeoBox render_box = GeoBoxManager::load_geobox(input_cache_path);
    Pathfinder render_pf(render_box);
    int grp = 1;
    for (const auto& [meta, sol] : gbest.solution_to_meta_agent) {
        for (const auto& path : sol.paths)
            for (auto wid : path.path_edges)
                render_pf.update_way_group(wid, grp);
        ++grp;
    }

    GeoBoxManager::save_geobox(geo_box, output_cache_path);
    GeoBoxManager::render_geobox(render_box, output_map_name, 2000, 2000);
    std::cout << "Map: " << output_map_name << ".png\n";
}

// ============================================================================
// V — VNS ORIENTEERING
// ============================================================================

void legacy_run_vns(const std::string& cache_dir)
{
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST VNS — 4 AGENTS\n";
    std::cout << std::string(80, '=') << "\n\n";

    const std::string cache_path = cache_dir + "\\asakusa_test_agent_raw.json";

    GeoBox geo_box = GeoBoxManager::load_geobox(cache_path);
    if (!geo_box.is_valid) { std::cerr << "Failed to load GeoBox\n"; return; }

    Pathfinder pathfinder(geo_box);
    GlobalMemory global_memory(geo_box, pathfinder);
    global_memory.length_constraint  = 1000.0;
    global_memory.search_coefficient = 1.5f;

    VNSParams vns_params;
    vns_params.max_iterations   = 100;
    vns_params.k_max             = 5;
    vns_params.max_iterations_vnd = 30;
    vns_params.use_two_opt      = true;
    vns_params.use_insertion    = true;
    vns_params.use_swap         = true;
    vns_params.use_add_remove   = true;

    struct Cfg { std::string name; std::vector<int> chars; };
    std::vector<int> base(100, 0);
    std::vector<Cfg> cfgs = {
        {"Agent_Temples",     [&]{ auto c=base; c[1]=100; return c; }()},
        {"Agent_Restaurants", [&]{ auto c=base; c[2]=100; return c; }()},
        {"Agent_Shops",       [&]{ auto c=base; c[3]=100; return c; }()},
        {"Agent_All",         [&]{ auto c=base; c[1]=c[2]=c[3]=100; return c; }()},
    };

    std::vector<VNSResult> results;
    for (const auto& cfg : cfgs) {
        std::cout << "Running " << cfg.name << "...\n";
        VNSSolver solver(geo_box, pathfinder, global_memory);
        results.push_back(solver.solve(cfg.chars, 1000.0, vns_params));
    }

    std::cout << "\n" << std::left << std::setw(20) << "Agent"
              << std::right << std::setw(12) << "Fitness"
              << std::setw(8)  << "POIs"
              << std::setw(12) << "Dist(m)"
              << std::setw(10) << "Time(ms)"
              << std::setw(8)  << "Valid" << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (size_t i = 0; i < cfgs.size(); ++i) {
        const auto& r = results[i];
        std::cout << std::left  << std::setw(20) << cfgs[i].name
                  << std::right << std::setw(12) << (int)r.fitness
                  << std::setw(8)  << r.num_pois
                  << std::setw(12) << (int)r.distance
                  << std::setw(10) << r.execution_time_ms
                  << std::setw(8)  << (r.is_valid ? "ok" : "x") << "\n";
    }

    // Render each agent
    for (size_t i = 0; i < cfgs.size(); ++i) {
        const auto& r = cfgs[i];
        const auto& res = results[i];
        if (!res.is_valid || res.solution.empty()) continue;

        GeoBox box = GeoBoxManager::load_geobox(cache_path);
        Pathfinder pf(box);
        int grp = 1;
        for (size_t j = 0; j + 1 < res.solution.size(); ++j) {
            auto paths = global_memory.check_path(res.solution[j], res.solution[j + 1]);
            if (!paths.empty() && paths[0].cost != std::numeric_limits<float>::max())
                for (auto wid : paths[0].path_edges)
                    pf.update_way_group(wid, grp);
        }
        GeoBoxManager::render_geobox(box, "VNS_" + r.name, 2000, 2000);
        std::cout << "Map: VNS_" << r.name << ".png\n";
    }
}

// ============================================================================
// P — PSO MTTDS
// ============================================================================

void legacy_run_pso(const std::string& cache_dir)
{
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST PSO MTTDS — 4 AGENTS\n";
    std::cout << std::string(80, '=') << "\n\n";

    const std::string cache_path = cache_dir + "\\asakusa_test_agent_raw.json";

    GeoBox geo_box = GeoBoxManager::load_geobox(cache_path);
    if (!geo_box.is_valid) { std::cerr << "Failed to load GeoBox\n"; return; }

    Pathfinder pathfinder(geo_box);
    GlobalMemory global_memory(geo_box, pathfinder);
    global_memory.length_constraint  = 1000.0;
    global_memory.search_coefficient = 1.5f;

    MTTDSPSOParams pso_params;
    pso_params.num_particles  = 30;
    pso_params.max_iterations = 100;
    pso_params.w              = 0.9;
    pso_params.c1             = 2.0;
    pso_params.c2             = 2.0;
    pso_params.mutation_rate  = 0.25;
    pso_params.use_local_search = true;

    struct Cfg { std::string name; std::vector<int> chars; };
    std::vector<int> base(100, 0);
    std::vector<Cfg> cfgs = {
        {"Agent_Temples",     [&]{ auto c=base; c[1]=100; return c; }()},
        {"Agent_Restaurants", [&]{ auto c=base; c[2]=100; return c; }()},
        {"Agent_Shops",       [&]{ auto c=base; c[3]=100; return c; }()},
        {"Agent_All",         [&]{ auto c=base; c[1]=c[2]=c[3]=100; return c; }()},
    };

    std::vector<MTTDSPSOResult> results;
    for (const auto& cfg : cfgs) {
        std::cout << "Running " << cfg.name << "...\n";
        MTTDS_PSOSolver solver(geo_box, pathfinder, global_memory);
        results.push_back(solver.solve(cfg.chars, 1000.0, pso_params));
    }

    std::cout << "\n" << std::left << std::setw(20) << "Agent"
              << std::right << std::setw(12) << "Fitness"
              << std::setw(8)  << "POIs"
              << std::setw(12) << "Dist(m)"
              << std::setw(10) << "Time(ms)"
              << std::setw(8)  << "Valid" << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (size_t i = 0; i < cfgs.size(); ++i) {
        const auto& res = results[i];
        std::cout << std::left  << std::setw(20) << cfgs[i].name
                  << std::right << std::setw(12) << (int)res.fitness
                  << std::setw(8)  << res.num_pois
                  << std::setw(12) << (int)res.distance
                  << std::setw(10) << res.execution_time_ms
                  << std::setw(8)  << (res.is_valid ? "ok" : "x") << "\n";
    }

    for (size_t i = 0; i < cfgs.size(); ++i) {
        const auto& res = results[i];
        if (!res.is_valid || res.solution.empty()) continue;

        GeoBox box = GeoBoxManager::load_geobox(cache_path);
        Pathfinder pf(box);
        int grp = 1;
        for (size_t j = 0; j + 1 < res.solution.size(); ++j) {
            auto paths = global_memory.check_path(res.solution[j], res.solution[j + 1]);
            if (!paths.empty() && paths[0].cost != std::numeric_limits<float>::max())
                for (auto wid : paths[0].path_edges)
                    pf.update_way_group(wid, grp);
        }
        std::string out = cache_dir + "\\pso_" + cfgs[i].name + "_result.json";
        GeoBoxManager::save_geobox(box, out);
        GeoBoxManager::render_geobox(box, "PSO_" + cfgs[i].name, 2000, 2000);
        std::cout << "Map: PSO_" << cfgs[i].name << ".png\n";
    }
}

// ============================================================================
// E — CREATE GEOBOX
// ============================================================================

void legacy_create_geobox(const std::string& osm_file, const std::string& cache_dir)
{
    std::string location_name, bbox;
    double min_lat, min_lon, max_lat, max_lon;
    select_location(location_name, min_lat, min_lon, max_lat, max_lon, bbox);

    FlickrConfig config;
    config.api_key              = flickr_api_key_from_env();
    config.bbox                 = bbox;
    config.poi_assignment_radius = 15.0;
    config.min_date             = "2020-01-01";
    config.max_date             = "2024-12-31";

    complete_workflow(osm_file, min_lon, min_lat, max_lon, max_lat,
                      cache_dir, location_name, location_name + "_raw",
                      2000, 2000, config, false);
}

// ============================================================================
// I — INITIALIZE POI
// ============================================================================

void legacy_init_poi(const std::string& cache_dir)
{
    std::string location_name, bbox;
    double min_lat, min_lon, max_lat, max_lon;
    select_location(location_name, min_lat, min_lon, max_lat, max_lon, bbox);

    FlickrConfig config;
    config.api_key              = flickr_api_key_from_env();
    config.bbox                 = bbox;
    config.poi_assignment_radius = 15.0;
    config.min_date             = "2020-01-01";
    config.max_date             = "2024-12-31";

    std::string group_str, objective, in_name, out_name;
    std::cout << "Group number: ";  std::cin >> group_str;
    std::cout << "Objective (e.g. temple): "; std::cin >> objective;
    std::cout << "Cache to load: "; std::cin >> in_name;
    std::cout << "Cache to save: "; std::cin >> out_name;

    std::string in_path  = cache_dir + "\\" + in_name  + ".json";
    std::string out_path = cache_dir + "\\" + out_name + ".json";

    GeoBox geo_box = GeoBoxManager::load_geobox(in_path);
    if (!geo_box.is_valid) { std::cerr << "Failed to load: " << in_path << "\n"; return; }

    int group_nb = std::stoi(group_str);
    config.search_word = objective;

    std::string flickr_cache = cache_dir + "\\flickr_" + location_name + "_" + objective + "_cache.json";
    geo_box = apply_objectives(geo_box, config, flickr_cache, true, group_nb);

    auto git = geo_box.data.objective_groups.find(group_nb);
    if (git != geo_box.data.objective_groups.end())
        std::cout << "Group " << group_nb << ": " << git->second.node_ids.size() << " POIs\n";
    else
        std::cout << "No POIs found for '" << objective << "'\n";

    GeoBoxManager::save_geobox(geo_box, out_path);
    std::cout << "Saved: " << out_path << "\n";
}

// ============================================================================
// R — RENDER
// ============================================================================

void legacy_render(const std::string& cache_dir)
{
    std::string in_name, out_name;
    std::cout << "Cache to load: "; std::cin >> in_name;
    std::cout << "Output map name: "; std::cin >> out_name;

    std::string in_path = cache_dir + "\\" + in_name + ".json";
    GeoBox geo_box = GeoBoxManager::load_geobox(in_path);
    if (!geo_box.is_valid) { std::cerr << "Failed to load: " << in_path << "\n"; return; }

    bool ok = GeoBoxManager::render_geobox(geo_box, out_name, 2000, 2000);
    std::cout << (ok ? "Map: " : "Render failed: ") << out_name << ".png\n";
}
