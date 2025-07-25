#include <iostream>
#include <iomanip>
#include "Box.hpp"
#include "MapRenderer.hpp"
#include "GeoBoxManager.hpp"
#include "Pathfinding.hpp"
#include "utility.hpp"

int main() {
    
    const std::string osm_file = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    const std::string cache_dir = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\geobox_cache_folder";

    const std::string cache_path = cache_dir + "\\asakusa.json";

    std::string rep;
    std::cout << "New Geobox and cache (G/g) or System Creation and Pathfinding (P/p): ";
    std::cin >> rep;
    
    if (rep == "G" || rep == "g"){
    FlickrConfig config;
    config.api_key = "9568c6342a890ef1ba342f54c4c1160f";
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
    
    } else if (rep == "P" || rep == "p") {

    GeoBox geo_box = GeoBoxManager::load_geobox(cache_path);
    
    if (!geo_box.is_valid) {
        std::cout << "Erreur lors du rechargement" << std::endl;
        return 0;
    }
    else {
        std::cout << "Succes du chargement" << std::endl;
    }
    
    Pathfinder PfSystem(geo_box);

    int group_nb = 1;

    auto group_it = geo_box.data.objective_groups.find(group_nb);
    if (group_it == geo_box.data.objective_groups.end()) {
        std::cout << "Groupe " << group_nb << " n'existe pas!" << std::endl;
        return 0;
    }
    else {
        std::cout << "Groupe trouvé: " << group_it->second.name << std::endl;
    }

    auto& node_list = geo_box.data.objective_groups[group_nb].node_ids;

    if (node_list.size()<2){
        std::cout << "Longeur de liste empechant la creation de routes" << std::endl;
        return 0;
    }
    else {
        std::cout << "Lancement du calcul des routes" << std::endl;
    }

    bool success = PfSystem.Subgraph_construction(PfSystem, node_list, group_nb);
    
    if (success) {
        std::cout << "Construction du sous-graphe réussie!" << std::endl;
    } else {
        std::cout << "Échec de la construction du sous-graphe" << std::endl;
    }

    GeoBoxManager::render_geobox(geo_box, "asakusa", 2000, 2000);

    }

    std::cout << "Fin de l'application" << std::endl;

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