#include <iostream>
#include <iomanip>
#include "Box.hpp"
#include "MapRenderer.hpp"
#include "GeoBoxManager.hpp"
#include "Pathfinding.hpp"
#include "utility.hpp"
#include <string>

int main() {
    
    const std::string osm_file = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    const std::string cache_dir = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\geobox_cache_folder";

    // ========== SELECTION DE LA LOCALISATION ==========
    std::cout << "\n=== Sélection de la localisation ===" << std::endl;
    std::cout << "1. Asakusa (Temples traditionnels)" << std::endl;
    std::cout << "2. Shibuya (Centre urbain)" << std::endl;
    std::cout << "3. Shinjuku (Quartier d'affaires)" << std::endl;
    std::cout << "Choisissez une localisation (1-3) : ";
    
    int location_choice;
    std::cin >> location_choice;
    
    // Variables de configuration selon la localisation
    std::string location_name;
    double min_lat, min_lon, max_lat, max_lon;
    std::string bbox;
    
    switch(location_choice) {
        case 1: // Asakusa
            location_name = "asakusa";
            min_lat = 35.705; min_lon = 139.785;
            max_lat = 35.718; max_lon = 139.800;
            bbox = "139.785,35.705,139.800,35.718";
            std::cout << "Localisation sélectionnée : Asakusa" << std::endl;
            break;
            
        case 2: // Shibuya
            location_name = "shibuya";
            min_lat = 35.658; min_lon = 139.699;
            max_lat = 35.661; max_lon = 139.704;
            bbox = "139.699,35.658,139.704,35.661";
            std::cout << "Localisation sélectionnée : Shibuya" << std::endl;
            break;
            
        case 3: // Shinjuku
            location_name = "shinjuku";
            min_lat = 35.689; min_lon = 139.698;
            max_lat = 35.702; max_lon = 139.710;
            bbox = "139.698,35.689,139.710,35.702";
            std::cout << "Localisation sélectionnée : Shinjuku" << std::endl;
            break;
            
        default:
            std::cout << "Choix invalide, utilisation d'Asakusa par défaut" << std::endl;
            location_name = "asakusa";
            min_lat = 35.705; min_lon = 139.785;
            max_lat = 35.718; max_lon = 139.800;
            bbox = "139.785,35.705,139.800,35.718";
            break;
    }
    
    const std::string cache_path = cache_dir + "\\" + location_name + ".json";
    std::cout << "=== Fin sélection localisation ===\n" << std::endl;

    std::string rep;
    std::cout << "New Geobox and cache (G/g), Initialize POI (I/i), System Creation and Pathfinding (P/p), Verify data (V/v), Verify Pf (F/f): ";
    std::cin >> rep;

    FlickrConfig config;
    config.api_key = "9568c6342a890ef1ba342f54c4c1160f";
    config.bbox = bbox;  // Utilise la bbox de la localisation sélectionnée
    config.poi_assignment_radius = 15.0;
    config.min_date = "2020-01-01";
    config.max_date = "2024-12-31";
    
    if (rep == "G" || rep == "g"){

        complete_workflow(
            osm_file,
            min_lon, min_lat, max_lon, max_lat,  // Coordonnées de la localisation sélectionnée
            cache_dir,
            location_name,
            location_name + "_raw",
            2000, 2000,
            config,
            false  // Utiliser les objectifs Flickr
        );
        
    } else if (rep == "I" || rep == "i") {

        // ========== INITIALISATION DES POI FLICKR ==========
        std::cout << "\n=== Initialisation des Points d'Intérêt Flickr ===" << std::endl;
        
        std::string group_input;
        std::cout << "Numéro du groupe à créer : ";
        std::cin >> group_input;

        int group_nb = std::stoi(group_input);

        std::string objective_input;
        std::cout << "Objectifs Flickr (ex: temple, restaurant, shop, park) : ";
        std::cin >> objective_input;

        GeoBox geo_box = GeoBoxManager::load_geobox(cache_path);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur lors du rechargement du cache raw" << std::endl;
            return 0;
        }
        else {
            std::cout << "Succès du chargement du cache raw" << std::endl;
        }

        // Configuration Flickr avec les objectifs saisis
        config.search_word = objective_input;

        try {
            // Appliquer les objectifs Flickr à la GeoBox existante
            std::string flickr_cache = cache_dir + "\\flickr_" + objective_input + "_cache.json";
            geo_box = apply_objectives(geo_box, config, flickr_cache, true, group_nb);
            std::cout << "Objectifs Flickr appliqués avec succès pour: " << objective_input << std::endl;
            
            // Sauvegarder la GeoBox avec les nouveaux objectifs
            std::string objectives_cache = cache_dir + "\\" + location_name + "_group" + std::to_string(group_nb) + "_objectives.json";
            GeoBoxManager::save_geobox(geo_box, objectives_cache);
            std::cout << "GeoBox avec POI sauvegardée: " << objectives_cache << std::endl;
            
            // Vérifier que le groupe a été créé
            auto group_it = geo_box.data.objective_groups.find(group_nb);
            if (group_it != geo_box.data.objective_groups.end()) {
                std::cout << "Groupe " << group_nb << " créé avec " << group_it->second.node_ids.size() << " POI" << std::endl;
            } else {
                std::cout << "Attention: Aucun POI trouvé pour les objectifs '" << objective_input << "'" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Erreur lors de l'application des objectifs Flickr: " << e.what() << std::endl;
        }
        
        std::cout << "=== Fin initialisation POI ===\n" << std::endl;
        
    } else if (rep == "P" || rep == "p") {

        std::string group_input;
        std::cout << "Numéro du groupe pour pathfinding : ";
        std::cin >> group_input;

        int group_nb = std::stoi(group_input);

        // Charger la GeoBox avec les objectifs déjà initialisés
        std::string objectives_cache = cache_dir + "\\" + location_name + "_group" + std::to_string(group_nb) + "_objectives.json";
        GeoBox geo_box = GeoBoxManager::load_geobox(objectives_cache);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur: Cache d'objectifs introuvable. Utilisez d'abord l'option 'I' pour initialiser les POI." << std::endl;
            std::cout << "Fichier recherché: " << objectives_cache << std::endl;
            return 0;
        }
        else {
            std::cout << "Succès du chargement du cache d'objectifs" << std::endl;
        }
        
        Pathfinder PfSystem(geo_box);

        auto group_it = geo_box.data.objective_groups.find(group_nb);
        if (group_it == geo_box.data.objective_groups.end()) {
            std::cout << "Groupe " << group_nb << " n'existe pas! Utilisez d'abord l'option 'I'." << std::endl;
            return 0;
        }
        else {
            std::cout << "Groupe trouvé: " << group_it->second.name << std::endl;
        }

        auto& node_list = geo_box.data.objective_groups[group_nb].node_ids;

        if (node_list.size() < 2){
            std::cout << "Longueur de liste empêchant la création de routes (minimum 2 POI requis)" << std::endl;
            std::cout << "POI disponibles: " << node_list.size() << std::endl;
            return 0;
        }
        else {
            std::cout << "Lancement du calcul des routes pour " << node_list.size() << " POI" << std::endl;
        }

        bool success = PfSystem.Subgraph_construction(PfSystem, node_list, group_nb);
        
        if (success) {
            std::cout << "Construction du sous-graphe réussie!" << std::endl;
            std::string pathfinding_cache = cache_dir + "\\" + location_name + "_pf_" + std::to_string(group_nb) + ".json";
            GeoBoxManager::save_geobox(geo_box, pathfinding_cache);
            std::cout << "GeoBox avec pathfinding sauvegardée: " << pathfinding_cache << std::endl;
        } else {
            std::cout << "Échec de la construction du sous-graphe" << std::endl;
        }

        const std::string rendering_name = location_name + "_group" + std::to_string(group_nb);

        GeoBoxManager::render_geobox(geo_box, rendering_name, 2000, 2000);

    } else if (rep == "V" || rep == "v") {
        
        // NOUVELLE OPTION - Validation de l'intégrité des données
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_path);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur lors du rechargement de la GeoBox" << std::endl;
            return 0;
        }
        
        // Appel de la fonction de validation depuis utility
        validate_data_integrity(geo_box);
        
    } else if (rep == "F" || rep == "f") {

        std::string group_input;
        std::cout << "Which group to check (nb) : ";
        std::cin >> group_input;

        int group_nb = std::stoi(group_input);

        // CORRECTION: Utiliser std::to_string au lieu de group_input directement
        const std::string cache_path_pf = cache_dir + "\\" + location_name + "_pf_" + std::to_string(group_nb) + ".json";

        GeoBox geo_box = GeoBoxManager::load_geobox(cache_path_pf);

        if (!geo_box.is_valid) {
            std::cout << "Erreur lors du rechargement de la GeoBox" << std::endl;
            return 0;
        }

        Pathfinder PfSystem(geo_box);

        auto group_it = geo_box.data.objective_groups.find(group_nb);
        if (group_it == geo_box.data.objective_groups.end()) {
            std::cout << "Groupe " << group_nb << " n'existe pas!" << std::endl;
            return 0;
        }
        else {
            std::cout << "Groupe trouvé: " << group_it->second.name << std::endl;
        }

        auto& node_list = geo_box.data.objective_groups[group_nb].node_ids;

        if (node_list.size() < 2){
            std::cout << "Longeur de liste empechant la creation de routes" << std::endl;
            return 0;
        }
        else {
            std::cout << "Lancement du calcul des routes" << std::endl;
        }

        std::cout << "Nombre de points objectifs : " << node_list.size() << std::endl;

        bool success = verif_pathfinding(PfSystem, node_list, group_nb);

        if (success) {
            std::cout << "Le sous graph est complet" << std::endl;
        } else {
            std::cout << "Le sous graph a un probleme" << std::endl;
        }

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