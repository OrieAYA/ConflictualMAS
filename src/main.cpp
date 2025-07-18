#include <iostream>
#include <iomanip>
#include "Box.hpp"
#include "MapRenderer.hpp"
#include "GeoBoxManager.hpp"
#include "utility.hpp"

int main() {
    
    const std::string osm_file = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    const std::string cache_dir = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\geobox_cache_folder";

    const std::string cache_path = cache_dir + "\\asakusa_139.785000_35.705000_139.800000_35.718000.json";
    
    FlickrConfig config;
    config.api_key = "API_KEY";
    config.search_word = "temple";
    config.bbox = "139.785,35.705,139.800,35.718";  // Asakusa
    config.poi_assignment_radius = 15.0;
    config.min_date = "2020-01-01";
    config.max_date = "2024-12-31";

    complete_workflow(
        osm_file,
        139.785, 35.705, 139.800, 35.718,  // Asakusa coordinates
        cache_dir,
        "asakusa",
        "asakusa",
        2000, 2000,
        config,
        true  // Utiliser les objectifs Flickr
    );
    
    /*
    GeoBox loaded_geo_box = GeoBoxManager::load_geobox(cache_path);
    
    if (!loaded_geo_box.is_valid) {
        std::cout << "Erreur lors du rechargement" << std::endl;
        return 0;
    }
    else {
        std::cout << "Succes du chargement" << std::endl;
    }
    */

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
GeoBox geo_box = GeoBoxManager::load_geobox(cache_dir + "\\shibuya_example.json");
*/

/*
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
    139.785, 35.705, 139.800, 35.718,  // Asakusa coordinates
    cache_dir,
    "asakusa",
    "asakusa",
    2000, 2000,
    config,
    true  // Utiliser les objectifs Flickr
);
*/