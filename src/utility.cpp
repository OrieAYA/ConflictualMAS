#include <iostream>
#include <iomanip>
#include "Box.hpp"
#include "MapRenderer.hpp"
#include "GeoBoxManager.hpp"  // SEUL include nécessaire maintenant
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
        config.poi_assignment_radius = 10.0;
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
    
    // Créer GeoBox
    GeoBox geo_box = GeoBoxManager::create_geobox(osm_file, min_lon, min_lat, max_lon, max_lat);
    
    if (geo_box.is_valid) {
        // Sauvegarder
        GeoBoxManager::save_geobox(geo_box, cache_path);
        
        // Rendre
        GeoBoxManager::render_geobox(geo_box, output_name, width, height);
    } else {
        std::cerr << "Erreur lors de la création de la GeoBox" << std::endl;
    }
}

// Charger depuis le cache et rendre
void load_and_render(const std::string& cache_path,
                    const std::string& output_name_small,
                    const std::string& output_name_large,
                    int width_small, int height_small,
                    int width_large, int height_large) {
    
    std::cout << "=== Exemple : Charger depuis le cache et rendre ===" << std::endl;
    std::cout << "Cache: " << cache_path << std::endl;
    std::cout << "Output small: " << output_name_small << " (" << width_small << "x" << height_small << ")" << std::endl;
    std::cout << "Output large: " << output_name_large << " (" << width_large << "x" << height_large << ")" << std::endl;
    
    // Vérifier si le cache existe
    if (GeoBoxManager::cache_exists(cache_path)) {
        // Charger depuis le cache
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_path);
        
        if (geo_box.is_valid) {
            // Rendre avec différentes tailles
            GeoBoxManager::render_geobox(geo_box, output_name_small, width_small, height_small);
            GeoBoxManager::render_geobox(geo_box, output_name_large, width_large, height_large);
        } else {
            std::cerr << "Erreur lors du chargement de la GeoBox" << std::endl;
        }
    } else {
        std::cout << "Cache non trouvé: " << cache_path << std::endl;
        std::cout << "Exécutez d'abord create_save_render() avec ce chemin" << std::endl;
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
    
    // Créer GeoBox avec objectifs
    GeoBox geo_box = GeoBoxManager::create_geobox_with_objectives(
        osm_file, min_lon, min_lat, max_lon, max_lat, flickr_config
    );
    
    if (geo_box.is_valid) {
        // Afficher info
        GeoBoxManager::display_geobox_info(geo_box);
        
        // Sauvegarder
        GeoBoxManager::save_geobox(geo_box, cache_path);
        
        // Rendre
        GeoBoxManager::render_geobox(geo_box, output_name, width, height);
    } else {
        std::cerr << "Erreur lors de la création de la GeoBox avec objectifs" << std::endl;
    }
}

// Workflow complet (créer, sauvegarder, recharger, rendre)
void complete_workflow(const std::string& osm_file,
                      double min_lon, double min_lat, double max_lon, double max_lat,
                      const std::string& cache_dir,
                      const std::string& cache_prefix,
                      const std::string& output_name,
                      int width, int height) {
    
    std::cout << "=== Exemple : Workflow complet ===" << std::endl;
    std::cout << "OSM File: " << osm_file << std::endl;
    std::cout << "BBox: (" << min_lon << ", " << min_lat << ") à (" << max_lon << ", " << max_lat << ")" << std::endl;
    std::cout << "Cache dir: " << cache_dir << std::endl;
    std::cout << "Cache prefix: " << cache_prefix << std::endl;
    std::cout << "Output: " << output_name << " (" << width << "x" << height << ")" << std::endl;
    
    // Générer le nom complet du cache
    std::string cache_name = GeoBoxManager::generate_cache_name(min_lon, min_lat, max_lon, max_lat, cache_prefix);
    std::string cache_path = cache_dir + "\\" + cache_name;
    
    std::cout << "Cache path: " << cache_path << std::endl;
    
    std::cout << "Étape 1: Création..." << std::endl;
    GeoBox original_geo_box = GeoBoxManager::create_geobox(osm_file, min_lon, min_lat, max_lon, max_lat);
    
    if (!original_geo_box.is_valid) {
        std::cout << "Erreur lors de la création" << std::endl;
        return;
    }
    
    std::cout << "Étape 2: Sauvegarde..." << std::endl;
    if (!GeoBoxManager::save_geobox(original_geo_box, cache_path)) {
        std::cout << "Erreur lors de la sauvegarde" << std::endl;
        return;
    }

    std::cout << "Étape 3: Rechargement..." << std::endl;
    GeoBox loaded_geo_box = GeoBoxManager::load_geobox(cache_path);
    
    if (!loaded_geo_box.is_valid) {
        std::cout << "Erreur lors du rechargement" << std::endl;
        return;
    }

    std::cout << "Étape 4: Rendu..." << std::endl;
    GeoBoxManager::render_geobox(loaded_geo_box, output_name, width, height);
    
    std::cout << "Workflow terminé avec succès!" << std::endl;
}