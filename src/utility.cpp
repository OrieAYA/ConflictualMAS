#include <iostream>
#include <iomanip>
#include "Box.hpp"
#include "MapRenderer.hpp"
#include "GeoBoxManager.hpp"
#include "utility.hpp"

int test() {
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
        std::cout << "\n=== Création de la GeoBox ===" << std::endl;
        GeoBox geo_box = create_geo_box(osm_filepath, min_lon, min_lat, max_lon, max_lat);
        
        if (!geo_box.is_valid) {
            std::cerr << "Erreur lors de la création de la GeoBox" << std::endl;
            return 1;
        }
        
        std::cout << "GeoBox créée avec succès!" << std::endl;
        std::cout << "Nodes: " << geo_box.data.nodes.size() << std::endl;
        std::cout << "Ways: " << geo_box.data.ways.size() << std::endl;

        // Configuration Flickr
        FlickrConfig config;
        config.api_key = "9568c6342a890ef1ba342f54c4c1160f";
        config.search_word = "temple";
        config.bbox = std::to_string(min_lon) + "," + std::to_string(min_lat) + "," + 
              std::to_string(max_lon) + "," + std::to_string(max_lat);  
        config.poi_assignment_radius = 10.0;
        config.min_date = "2020-01-01";
        config.max_date = "2024-12-31";

        // ✅ DÉLÉGUER à apply_objectives (Box.cpp)
        geo_box = apply_objectives(geo_box, config, "cache.json", true);

        // Verification des objectifs
        std::cout << "\n=== Résultats finaux ===" << std::endl;
        std::cout << "Total nodes: " << geo_box.data.nodes.size() << std::endl;
        std::cout << "Total ways: " << geo_box.data.ways.size() << std::endl;
        
        // ✅ DÉLÉGUER à render_map (MapRenderer.cpp)
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

// Créer, sauvegarder et rendre une GeoBox
void create_save_render(const std::string& osm_file,
                       double min_lon, double min_lat, double max_lon, double max_lat,
                       const std::string& cache_path,
                       const std::string& output_name,
                       int width, int height) {
    
    std::cout << "=== Exemple : Créer, sauvegarder et rendre ===" << std::endl;
    std::cout << "OSM File: " << osm_file << std::endl;
    std::cout << "BBox: (" << min_lon << ", " << min_lat << ") à (" << max_lon << ", " << max_lat << ")" << std::endl;
    std::cout << "Cache: " << cache_path << std::endl;
    std::cout << "Output: " << output_name << " (" << width << "x" << height << ")" << std::endl;

    GeoBox geo_box = create_geo_box(osm_file, min_lon, min_lat, max_lon, max_lat);
    
    if (geo_box.is_valid) {
        // Sauvegarder et rendre via GeoBoxManager
        GeoBoxManager::save_geobox(geo_box, cache_path);
        GeoBoxManager::render_geobox(geo_box, output_name, width, height);
    } else {
        std::cerr << "Erreur lors de la création de la GeoBox" << std::endl;
    }
}

// GeoBox avec objectifs Flickr
void with_flickr_objectives(const std::string& osm_file,
                           double min_lon, double min_lat, double max_lon, double max_lat,
                           const std::string& cache_path,
                           const FlickrConfig& flickr_config,
                           const std::string& output_name,
                           int width, int height) {
    
    std::cout << "=== Exemple : GeoBox avec objectifs Flickr ===" << std::endl;
    std::cout << "OSM File: " << osm_file << std::endl;
    std::cout << "BBox: (" << min_lon << ", " << min_lat << ") à (" << max_lon << ", " << max_lat << ")" << std::endl;
    std::cout << "Cache: " << cache_path << std::endl;
    std::cout << "Flickr search: " << flickr_config.search_word << std::endl;
    std::cout << "Output: " << output_name << " (" << width << "x" << height << ")" << std::endl;
    
    GeoBox geo_box = create_geo_box(osm_file, min_lon, min_lat, max_lon, max_lat);
    
    if (geo_box.is_valid) {
        // Ajouter les objectifs Flickr
        geo_box = apply_objectives(geo_box, flickr_config, "flickr_cache.json", true);
        
        // Afficher info, sauvegarder et rendre
        GeoBoxManager::display_geobox_info(geo_box);
        GeoBoxManager::save_geobox(geo_box, cache_path);
        GeoBoxManager::render_geobox(geo_box, output_name, width, height);
    } else {
        std::cerr << "Erreur lors de la création de la GeoBox avec objectifs" << std::endl;
    }
}

// Workflow complet
void complete_workflow(const std::string& osm_file,
                      double min_lon, double min_lat, double max_lon, double max_lat,
                      const std::string& cache_dir,
                      const std::string& cache_prefix,
                      const std::string& output_name,
                      int width, int height,
                      const FlickrConfig& flickr_config,
                      bool use_flickr_objectives) {
    
    // Générer le nom complet du cache
    std::string cache_name = GeoBoxManager::generate_cache_name(cache_prefix);
    std::string cache_path = cache_dir + "\\" + cache_name;
    
    std::cout << "Étape 1: Création de la GeoBox..." << std::endl;
    GeoBox original_geo_box = create_geo_box(osm_file, min_lon, min_lat, max_lon, max_lat);
    
    if (!original_geo_box.is_valid) {
        std::cout << "Erreur lors de la création de la GeoBox" << std::endl;
        return;
    }
    
    if (use_flickr_objectives) {
        std::cout << "Application des objectifs Flickr..." << std::endl;
        FlickrConfig adapted_config = flickr_config;
        adapted_config.bbox = std::to_string(min_lon) + "," + std::to_string(min_lat) + "," + 
                             std::to_string(max_lon) + "," + std::to_string(max_lat);
        
        original_geo_box = apply_objectives(original_geo_box, adapted_config, "flickr_cache.json", true);
    }
    
    std::cout << "Étape 2: Sauvegarde..." << std::endl;
    if (!GeoBoxManager::save_geobox(original_geo_box, cache_path)) {
        std::cout << "Erreur lors de la sauvegarde" << std::endl;
        return;
    }

    std::cout << "Étape 3: Rechargement depuis le cache..." << std::endl;
    GeoBox loaded_geo_box = GeoBoxManager::load_geobox(cache_path);
    
    if (!loaded_geo_box.is_valid) {
        std::cout << "Erreur lors du rechargement" << std::endl;
        return;
    }

    std::cout << "Étape 4: Rendu de la carte..." << std::endl;
    if (!GeoBoxManager::render_geobox(loaded_geo_box, output_name, width, height)) {
        std::cout << "Erreur lors du rendu" << std::endl;
        return;
    }
}