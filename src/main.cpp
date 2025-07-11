#include <iostream>
#include <iomanip>
#include "Box.hpp"
#include "MapRenderer.hpp"

int main() {
    std::cout << "Starting ConflictualMAS application." << std::endl;

    const std::string osm_filepath = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    
    // Configuration utilisateur
    double min_lon, min_lat, max_lon, max_lat;
    std::string map_name;
    
    std::cout << "\n=== Configuration de la Bounding Box ===" << std::endl;
    
    char use_default;
    std::cout << "\nUtiliser les valeurs par défaut (Shibuya : y / Asakusa : a) ? (y/a/n): ";
    std::cin >> use_default;
    
    if (use_default == 'y' || use_default == 'Y') {
        min_lon = 139.699;
        min_lat = 35.658;
        max_lon = 139.704;
        max_lat = 35.661;
        map_name = "shibuya_map";
    } else if (use_default == 'a' || use_default == 'A') {
        min_lon = 139.785;
        min_lat = 35.705;
        max_lon = 139.800;
        max_lat = 35.718;
        map_name = "asakusa_map";
    } else {
        std::cout << "Min Longitude: ";
        std::cin >> min_lon;
        std::cout << "Min Latitude: ";
        std::cin >> min_lat;
        std::cout << "Max Longitude: ";
        std::cin >> max_lon;
        std::cout << "Max Latitude: ";
        std::cin >> max_lat;
        
        std::cout << "Nom de la carte (sans extension): ";
        std::cin >> map_name;
    }
    
    std::cout << "\n=== Coordonnées sélectionnées ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Bounding Box: (" << min_lon << ", " << min_lat << ") à (" << max_lon << ", " << max_lat << ")" << std::endl;
    std::cout << "Nom de la carte: " << map_name << std::endl;
    
    try {
        //Création indépendante de la box
        std::cout << "\n=== Création de la GeoBox ===" << std::endl;
        GeoBox geo_box = create_geo_box(osm_filepath, min_lon, min_lat, max_lon, max_lat);
        
        if (!geo_box.is_valid) {
            std::cerr << "Erreur lors de la création de la GeoBox" << std::endl;
            return 1;
        }
        
        std::cout << "GeoBox créée avec succès!" << std::endl;
        std::cout << "Nodes: " << geo_box.data.nodes.size() << std::endl;
        std::cout << "Ways: " << geo_box.data.ways.size() << std::endl;

        // Configuration
        FlickrConfig config;
        config.api_key = "9568c6342a890ef1ba342f54c4c1160f";
        config.search_word = "temple";
        config.bbox = std::to_string(min_lon) + "," + std::to_string(min_lat) + "," + 
              std::to_string(max_lon) + "," + std::to_string(max_lat);  
        config.poi_assignment_radius = 150.0;
        config.min_date = "2020-01-01";
        config.max_date = "2024-12-31";

        //Application des objectifs
        geo_box = apply_objectives(geo_box, config, "cache.json", true);

        //Verif Objectifs
        std::cout << "\n=== Résultats finaux ===" << std::endl;
        std::cout << "Total nodes: " << geo_box.data.nodes.size() << std::endl;
        std::cout << "Total ways: " << geo_box.data.ways.size() << std::endl;
        
        //Rendu indépendant de la map
        if (geo_box.data.nodes.empty()) {
            std::cout << "Aucune donnée dans la bounding box spécifiée." << std::endl;
            return 1;
        }
        
        std::cout << "\n=== Rendu de la carte ===" << std::endl;
        bool render_success = render_map(geo_box, map_name, 2000, 2000);
        
        if (render_success) {
            std::cout << "Carte rendue avec succès!" << std::endl;
        } else {
            std::cerr << "Erreur lors du rendu de la carte" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Erreur: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nApplication terminée avec succès." << std::endl;
    return 0;
}