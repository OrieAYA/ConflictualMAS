#include <iostream>
#include <iomanip>
#include "Box.hpp"
#include "MapRenderer.hpp"
#include "GeoBoxManager.hpp"
#include "Pathfinding.hpp"
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
        
        original_geo_box = apply_objectives(original_geo_box, adapted_config, "flickr_cache.json", true, 1);
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

void validate_data_integrity(const GeoBox& geo_box) {
    std::cout << "\n=== VALIDATION DE L'INTÉGRITÉ DES DONNÉES ===" << std::endl;
    
    int problematic_nodes = 0;
    int problematic_ways = 0;
    int total_nodes = geo_box.data.nodes.size();
    int total_ways = geo_box.data.ways.size();
    
    std::cout << "Total nodes: " << total_nodes << std::endl;
    std::cout << "Total ways: " << total_ways << std::endl;
    std::cout << "\n--- VÉRIFICATION DES NODES ---" << std::endl;
    
    // VÉRIFICATION 1: Tous les nodes ont au moins une way incidente
    std::vector<osmium::object_id_type> orphan_nodes;
    
    for (const auto& [node_id, node] : geo_box.data.nodes) {
        if (node.incident_ways.empty()) {
            orphan_nodes.push_back(node_id);
            problematic_nodes++;
        }
    }
    
    if (!orphan_nodes.empty()) {
        std::cout << "❌ NODES ORPHELINS (" << orphan_nodes.size() << " trouvés):" << std::endl;
        for (size_t i = 0; i < std::min(orphan_nodes.size(), size_t(10)); ++i) {
            std::cout << "  Node " << orphan_nodes[i] << " (pas de ways incidents)" << std::endl;
        }
        if (orphan_nodes.size() > 10) {
            std::cout << "  ... et " << (orphan_nodes.size() - 10) << " autres" << std::endl;
        }
    } else {
        std::cout << "✓ Tous les nodes ont au moins une way incidente" << std::endl;
    }
    
    // VÉRIFICATION 2: Toutes les ways ont exactement deux nodes valides
    std::cout << "\n--- VÉRIFICATION DES WAYS ---" << std::endl;
    
    std::vector<osmium::object_id_type> invalid_ways;
    
    for (const auto& [way_id, way] : geo_box.data.ways) {
        bool node1_exists = geo_box.data.nodes.count(way.node1_id);
        bool node2_exists = geo_box.data.nodes.count(way.node2_id);
        bool nodes_different = (way.node1_id != way.node2_id);
        
        if (!node1_exists || !node2_exists || !nodes_different) {
            invalid_ways.push_back(way_id);
            problematic_ways++;
            
            if (invalid_ways.size() <= 10) {
                std::cout << "❌ Way " << way_id << ": ";
                if (!node1_exists) std::cout << "node1(" << way.node1_id << ") manquant ";
                if (!node2_exists) std::cout << "node2(" << way.node2_id << ") manquant ";
                if (!nodes_different) std::cout << "nodes identiques ";
                std::cout << std::endl;
            }
        }
    }
    
    if (!invalid_ways.empty()) {
        std::cout << "❌ WAYS INVALIDES (" << invalid_ways.size() << " trouvées)" << std::endl;
        if (invalid_ways.size() > 10) {
            std::cout << "  ... (" << (invalid_ways.size() - 10) << " autres ways invalides)" << std::endl;
        }
    } else {
        std::cout << "✓ Toutes les ways ont exactement deux nodes valides" << std::endl;
    }
    
    // VÉRIFICATION 3: Cohérence bidirectionnelle
    std::cout << "\n--- VÉRIFICATION DE LA COHÉRENCE BIDIRECTIONNELLE ---" << std::endl;
    
    int missing_references = 0;
    
    for (const auto& [node_id, node] : geo_box.data.nodes) {
        for (const auto& way_id : node.incident_ways) {
            auto way_it = geo_box.data.ways.find(way_id);
            
            if (way_it == geo_box.data.ways.end()) {
                if (missing_references < 10) {
                    std::cout << "❌ Node " << node_id << " référence way inexistant " << way_id << std::endl;
                }
                missing_references++;
            } else {
                bool way_references_node = (way_it->second.node1_id == node_id || 
                                           way_it->second.node2_id == node_id);
                if (!way_references_node) {
                    if (missing_references < 10) {
                        std::cout << "❌ Node " << node_id << " dans way " << way_id 
                                  << " mais way ne référence pas le node" << std::endl;
                    }
                    missing_references++;
                }
            }
        }
    }
    
    if (missing_references == 0) {
        std::cout << "✓ Cohérence bidirectionnelle parfaite" << std::endl;
    } else {
        std::cout << "❌ " << missing_references << " références incohérentes trouvées" << std::endl;
    }
    
    // RÉSUMÉ FINAL
    std::cout << "\n=== RÉSUMÉ DE LA VALIDATION ===" << std::endl;
    std::cout << "Nodes problématiques: " << problematic_nodes << "/" << total_nodes 
              << " (" << (100.0 * problematic_nodes / total_nodes) << "%)" << std::endl;
    std::cout << "Ways problématiques: " << problematic_ways << "/" << total_ways 
              << " (" << (100.0 * problematic_ways / total_ways) << "%)" << std::endl;
    std::cout << "Références incohérentes: " << missing_references << std::endl;
    
    if (problematic_nodes == 0 && problematic_ways == 0 && missing_references == 0) {
        std::cout << "🎉 DONNÉES PARFAITEMENT INTÈGRES !" << std::endl;
    } else {
        std::cout << "⚠️  PROBLÈMES D'INTÉGRITÉ DÉTECTÉS" << std::endl;
    }
}

bool verif_pathfinding(Pathfinder& PfSystem,
    const std::vector<osmium::object_id_type>& objective_nodes,
    int path_group){

        if(objective_nodes.empty()){
            return false;
        }

        std::vector<osmium::object_id_type> file = {objective_nodes[0]};
        std::unordered_set<osmium::object_id_type> visited;
        std::unordered_set<osmium::object_id_type> objective_set(objective_nodes.begin(), objective_nodes.end());

        int counter = 0;

        while (!file.empty()){

            osmium::object_id_type act_node = file[0];
            file.erase(file.begin());

            if(visited.count(act_node)) continue;
            visited.insert(act_node);
            
            if(objective_set.count(act_node)){
                counter++;
                if(counter==objective_nodes.size()){
                    return true;
                }  
            }

            for(const auto& in_way : PfSystem.geo_box.data.nodes[act_node].incident_ways){
                auto& act_way = PfSystem.geo_box.data.ways[in_way];
                if(act_way.groupe == path_group){
                    if(act_node == act_way.node1_id){
                        file.push_back(act_way.node2_id);
                    } else {
                        file.push_back(act_way.node1_id);
                    }
                }
            }
        }

        std::vector<osmium::object_id_type> next_check = {};
        
        for(const auto& visited_node : objective_nodes){
            if(!visited.count(visited_node)){
                next_check.push_back(visited_node);
            }
        }

        std::cout << "----------------------" << std::endl;
        std::cout << "Une des composantes : " << std::endl;
        std::cout << "----------------------" << std::endl;

        for(const auto& visited_node : objective_nodes){
            if(visited.count(visited_node)){
                std::cout << "Deja visite : " << visited_node << std::endl;
            }
        }

        if(!next_check.empty()){
            verif_pathfinding(PfSystem, next_check, path_group);
        }

        return false;
    }