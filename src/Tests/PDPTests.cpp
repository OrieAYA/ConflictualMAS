#include "PDPTests.hpp"
#include "Environment/GeoBox/Box.hpp"
#include "Environment/GeoBox/GeoBoxManager.hpp"
#include "Environment/GeoBox/GraphSearch.hpp"
#include "DMASforPD/Agents/Manager.hpp"
#include "DMASforPD/Structures/ObjectiveCache.hpp"
#include "DMASforPD/Agents/DeliveryAgent.hpp"
#include "DMASforPD/Structures/ObjectiveNode.hpp"
#include "utility.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

// ============================================================================
// test_pdp_system
// ============================================================================

void test_pdp_system(const std::string& cache_dir)
{
    std::cout << "\n=== PDP System Test ===\n\n";

    // ---- Config (interactive) ---------------------------------------------
    std::string cache_name, output_name;
    int   num_agents = 2, num_tasks = 2, max_steps = 300;
    float speed_mps  = 10.0f;

    std::cout << "Cache to load (no .json): "; std::cin >> cache_name;
    std::cout << "Output map name:          "; std::cin >> output_name;
    std::cout << "Agents [2]:               "; std::cin >> num_agents;
    std::cout << "Tasks  [2]:               "; std::cin >> num_tasks;
    std::cout << "Speed m/step [10]:        "; std::cin >> speed_mps;
    std::cout << "Max sim steps [300]:      "; std::cin >> max_steps;

    // ---- Load GeoBox ------------------------------------------------------
    std::string cache_path = cache_dir + "\\" + cache_name + ".json";
    GeoBox geo_box = GeoBoxManager::load_geobox(cache_path);
    if (!geo_box.is_valid) {
        std::cerr << "Failed to load: " << cache_path << "\n";
        return;
    }
    std::cout << "\nNodes: " << geo_box.data.nodes.size()
              << "  Ways: " << geo_box.data.ways.size()
              << "  Groups: " << geo_box.data.objective_groups.size() << "\n";

    // ---- Init system ------------------------------------------------------
    PDPGlobalMemory memory(geo_box);

    // ---- Collect objective nodes ------------------------------------------
    const auto& groups = geo_box.data.objective_groups;
    if (groups.empty()) {
        std::cerr << "No objective groups in GeoBox. Run C first to add POIs.\n";
        return;
    }

    // Gather all nodes ordered by group, then by position in node_ids.
    std::vector<ObjectiveNode> all_nodes;
    for (const auto& [gid, grp] : groups)
        for (auto nid : grp.node_ids)
            all_nodes.push_back(ObjectiveNode{nid, gid});

    if ((int)all_nodes.size() < num_tasks * 2) {
        std::cout << "Only " << all_nodes.size() << " objective nodes available — "
                  << "adjusting tasks to " << (all_nodes.size() / 2) << "\n";
        num_tasks = static_cast<int>(all_nodes.size() / 2);
    }
    if (num_tasks == 0) {
        std::cerr << "Not enough nodes to form any task pair.\n";
        return;
    }

    // ---- Create delivery agents -------------------------------------------
    // Agents start at the pickup node of their first task (valid road node).
    std::vector<std::unique_ptr<DeliveryAgent>> agents;
    for (int i = 0; i < num_agents; ++i) {
        osmium::object_id_type start = all_nodes[0].id;
        auto agent = std::make_unique<DeliveryAgent>(i, start);
        agent->connect(memory);
        memory.task_agent.add_delivery_agent(*agent);
        agents.push_back(std::move(agent));
    }
    std::cout << "Agents created: " << agents.size() << "\n";

    // ---- Create tasks (consecutive pair = pickup + delivery) --------------
    int tasks_created = 0;
    for (int i = 0; i + 1 < (int)all_nodes.size() && tasks_created < num_tasks; i += 2) {
        int task_id = memory.add_task(all_nodes[i], all_nodes[i + 1]);
        memory.task_agent.on_new_task(task_id, memory);
        ++tasks_created;
    }
    std::cout << "Tasks created:  " << tasks_created << "\n";

    // ---- Simulation loop --------------------------------------------------
    std::cout << "Running allocation loop (" << max_steps << " steps)...\n";
    for (int step = 0; step < max_steps; ++step) {
        memory.advance_time(step);
        memory.task_agent.step(memory);
        if (memory.count_allocated() + memory.count_finished() >= tasks_created)
            break;
    }
    std::cout << "Allocated: " << memory.count_allocated()
              << "  Finished: " << memory.count_finished()
              << "  Available: " << memory.count_available() << "\n";

    // ---- Plan all active agents -------------------------------------------
    for (auto& agent : agents) {
        if (agent->status == AgentStatus::Active)
            agent->plan(memory, speed_mps);
    }

    // ---- Mark routes on the geobox ----------------------------------------
    int color = 1;
    for (const auto& agent : agents) {
        if (agent->solution.empty()) { ++color; continue; }

        osmium::object_id_type from = *agent->solution.current_position;
        for (const auto& step : agent->solution.sequence) {
            const ObjectivePath* p = memory.get_or_compute_path(
                from, step.node.id, step.node.group_id);
            if (p && p->valid())
                for (auto way_id : p->edges)
                    graph_search::set_way_group(geo_box, way_id, color);
            from = step.node.id;
        }
        ++color;
    }

    // ---- Save + render ----------------------------------------------------
    std::string out_cache = cache_dir + "\\" + output_name + "_result.json";
    GeoBoxManager::save_geobox(geo_box, out_cache);
    bool ok = GeoBoxManager::render_geobox(geo_box, output_name, 2000, 2000);
    std::cout << (ok ? "Map: " : "Render failed: ") << output_name << ".png\n";
    std::cout << "Cache: " << out_cache << "\n";
}

// ============================================================================
// pdp_create_geobox
// ============================================================================

void pdp_create_geobox(const std::string& osm_file, const std::string& cache_dir)
{
    std::cout << "\n=== Create PDP GeoBox ===\n\n";

    // Location selection
    std::cout << "1. Asakusa  2. Shibuya  3. Shinjuku\nChoice [1]: ";
    int choice = 1; std::cin >> choice;

    std::string location_name, bbox;
    double min_lat, min_lon, max_lat, max_lon;
    switch (choice) {
        case 2:
            location_name = "shibuya"; bbox = "139.699,35.658,139.704,35.661";
            min_lat=35.658; min_lon=139.699; max_lat=35.661; max_lon=139.704; break;
        case 3:
            location_name = "shinjuku"; bbox = "139.698,35.689,139.710,35.702";
            min_lat=35.689; min_lon=139.698; max_lat=35.702; max_lon=139.710; break;
        default:
            location_name = "asakusa"; bbox = "139.785,35.705,139.800,35.718";
            min_lat=35.705; min_lon=139.785; max_lat=35.718; max_lon=139.800; break;
    }

    // Objectives for group 1 (pickup) and group 2 (delivery)
    std::string pickup_word, delivery_word, out_name;
    std::cout << "Pickup objective (e.g. temple):   "; std::cin >> pickup_word;
    std::cout << "Delivery objective (e.g. restaurant): "; std::cin >> delivery_word;
    std::cout << "Output cache name (no .json):     "; std::cin >> out_name;

    FlickrConfig config;
    config.api_key               = flickr_api_key_from_env();
    config.bbox                  = bbox;
    config.poi_assignment_radius = 15.0;
    config.min_date              = "2020-01-01";
    config.max_date              = "2024-12-31";

    // Step 1: build raw GeoBox
    std::string raw_path = cache_dir + "\\pdp_" + location_name + "_raw.json";
    complete_workflow(osm_file, min_lon, min_lat, max_lon, max_lat,
                      cache_dir, "pdp_" + location_name, "pdp_" + location_name + "_raw",
                      2000, 2000, config, false);

    GeoBox geo_box = GeoBoxManager::load_geobox(raw_path);
    if (!geo_box.is_valid) {
        std::cerr << "Failed to load raw GeoBox: " << raw_path << "\n";
        return;
    }

    // Step 2: apply pickup objectives (group 1)
    config.search_word = pickup_word;
    std::string flickr1 = cache_dir + "\\flickr_pdp_" + pickup_word + "_cache.json";
    geo_box = apply_objectives(geo_box, config, flickr1, true, 1);

    // Step 3: apply delivery objectives (group 2)
    config.search_word = delivery_word;
    std::string flickr2 = cache_dir + "\\flickr_pdp_" + delivery_word + "_cache.json";
    geo_box = apply_objectives(geo_box, config, flickr2, true, 2);

    auto g1 = geo_box.data.objective_groups.find(1);
    auto g2 = geo_box.data.objective_groups.find(2);
    std::cout << "Group 1 (pickup):   " << (g1 != geo_box.data.objective_groups.end() ? (int)g1->second.node_ids.size() : 0) << " POIs\n";
    std::cout << "Group 2 (delivery): " << (g2 != geo_box.data.objective_groups.end() ? (int)g2->second.node_ids.size() : 0) << " POIs\n";

    std::string out_path = cache_dir + "\\" + out_name + ".json";
    GeoBoxManager::save_geobox(geo_box, out_path);
    GeoBoxManager::render_geobox(geo_box, out_name + "_preview", 2000, 2000);
    std::cout << "Saved: " << out_path << "\n";
    std::cout << "Preview: " << out_name << "_preview.png\n";
}
