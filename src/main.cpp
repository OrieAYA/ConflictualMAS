#include <iostream>
#include <iomanip>
#include "Box.hpp"
#include "MapRenderer.hpp"
#include "GeoBoxManager.hpp"
#include "utility.hpp"

int main() {

    const std::string osm_file = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    const std::string cache_dir = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\geobox_cache_folder";
    
    return test();
    return 0;
}

/* Paramètres Shibuya
create_save_render(
    osm_file,
    139.699, 35.658, 139.704, 35.661,  // Shibuya coordinates
    cache_dir + "\\shibuya_example.json",
    "shibuya_from_scratch",
    2000, 2000
);
*/

/* Paramètres Shibuya
load_and_render(
    cache_dir + "\\shibuya_example.json",
    "shibuya_small_render",
    "shibuya_large_render",
    1000, 1000,
    4000, 4000
);
*/

/*
// Configuration Flickr pour Asakusa
FlickrConfig config;
config.api_key = "API_KEY";
config.search_word = "temple";
config.bbox = "139.785,35.705,139.800,35.718";  // Asakusa
config.poi_assignment_radius = 15.0;
config.min_date = "2020-01-01";
config.max_date = "2024-12-31";

with_flickr_objectives(
    osm_file,
    139.785, 35.705, 139.800, 35.718,  // Asakusa coordinates
    cache_dir + "\\asakusa_temples.json",
    config,
    "asakusa_temples_map",
    2000, 2000
);
*/

/*
complete_workflow(
    osm_file,
    139.699, 35.658, 139.704, 35.661,  // Shibuya coordinates
    cache_dir,
    "shibuya_workflow",
    "shibuya_workflow_result",
    2000, 2000
);
*/