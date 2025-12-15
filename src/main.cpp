#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include "Box.hpp"
#include "MapRenderer.hpp"
#include "GeoBoxManager.hpp"
#include "Pathfinding.hpp"
#include "utility.hpp"
#include <string>
#include "MHProcs/ACO.hpp"
#include "MHProcs/GRASP.hpp"
#include "MHProcs/VNS.hpp"
#include "MHProcs/PSO.hpp"
#include "OverpassAPI/OverpassAPI.hpp"

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
    std::string cache_name;
    
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
    std::cout << "New Geobox and cache (G/g), Initialize POI (I/i), System Creation and Pathfinding (P/p), Mh procedure (A/a), Verify data (V/v), Verify Pf (F/f), Render only (R/r), Complete Graph (C/c): ";
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

        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
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
            std::string flickr_cache = cache_dir + "\\flickr_" + location_name + "_" + objective_input + "_cache.json";
            geo_box = apply_objectives(geo_box, config, flickr_cache, true, group_nb);
            std::cout << "Objectifs Flickr appliqués avec succès pour: " << objective_input << std::endl;
            
            std::cout << "Cache Name to save : ";
            std::cin >> cache_name;
            cache_name = cache_dir + "//" + cache_name + ".json";
            GeoBoxManager::save_geobox(geo_box, cache_name);
            std::cout << "GeoBox avec POI sauvegardée: " << cache_name << std::endl;
            
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
        
        std::cout << "\n=== Pathfinding pour tous les groupes ===" << std::endl;
        
        // Charger la GeoBox avec les objectifs déjà initialisés
        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur: Cache d'objectifs introuvable. Utilisez d'abord l'option 'I' pour initialiser les POI." << std::endl;
            std::cout << "Fichier recherché: " << cache_name << std::endl;
            return 0;
        } else {
            std::cout << "Succès du chargement du cache d'objectifs" << std::endl;
            std::cout << "Nombre total de groupes: " << geo_box.data.objective_groups.size() << std::endl;
        }

        Pathfinder PfSystem(geo_box);

        auto debut = std::chrono::high_resolution_clock::now();
        
        bool global_success = false;
        int groups_processed = 0;
        int groups_successful = 0;
        
        // Parcourir tous les groupes existants
        for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
            
            std::cout << "\n--- TRAITEMENT GROUPE " << group_id << " ---" << std::endl;
            std::cout << "Nom: " << group_info.name << std::endl;
            std::cout << "POI disponibles: " << group_info.node_ids.size() << std::endl;
            
            groups_processed++;
            
            // Vérifier si le groupe a assez de POI
            if (group_info.node_ids.size() < 2) {
                std::cout << "GROUPE IGNORÉ: Minimum 2 POI requis pour le pathfinding" << std::endl;
                continue; // Passer au groupe suivant
            }
            
            std::cout << "Lancement du calcul des routes..." << std::endl;
            
            // Exécuter le pathfinding pour ce groupe
            bool group_success = PfSystem.Subgraph_construction(PfSystem, group_info.node_ids, group_id);
            
            if (group_success) {
                std::cout << "GROUPE " << group_id << ": SUCCÈS" << std::endl;
                groups_successful++;
                global_success = true; // Au moins un groupe a réussi
            } else {
                std::cout << "GROUPE " << group_id << ": ÉCHEC" << std::endl;
            }
        }

        auto fin = std::chrono::high_resolution_clock::now();
        
        // Résultats finaux
        std::cout << "\n=== RÉSULTATS FINAUX ===" << std::endl;
        std::cout << "Groupes traités: " << groups_processed << std::endl;
        std::cout << "Groupes réussis: " << groups_successful << std::endl;
        std::cout << "Taux de succès: " << (groups_processed > 0 ? (100.0 * groups_successful / groups_processed) : 0) << "%" << std::endl;
        
        if (global_success) {
            std::cout << "Construction du sous-graphe réussie globalement!" << std::endl;
            std::cout << "Cache Name to save : ";
            std::cin >> cache_name;
            cache_name = cache_dir + "//" + cache_name + ".json";
            GeoBoxManager::save_geobox(geo_box, cache_name);
            std::cout << "GeoBox avec pathfinding sauvegardée: " << cache_name << std::endl;
        } else {
            std::cout << "Aucun pathfinding réussi" << std::endl;
        }
        
        std::string output_name;
        std::cout << "Nom de sortie pour la carte : ";
        std::cin >> output_name;

        // Rendu de la carte
        bool render_success = GeoBoxManager::render_geobox(geo_box, output_name, 2000, 2000);
        
        if (render_success) {
            std::cout << "Carte rendue avec succès: " << output_name << std::endl;
        } else {
            std::cout << "Erreur lors du rendu de la carte" << std::endl;
        }

        auto duree = std::chrono::duration_cast<std::chrono::milliseconds>(fin - debut);
        
        std::cout << "Temps d'exécution: " << duree.count() << " ms" << std::endl;

    } else if (rep == "C" || rep == "c"){

        // Charger la GeoBox avec les objectifs déjà initialisés
        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur: Cache d'objectifs introuvable. Utilisez d'abord l'option 'I' pour initialiser les POI." << std::endl;
            std::cout << "Fichier recherché: " << cache_name << std::endl;
            return 0;
        }
        else {
            std::cout << "Succès du chargement du cache d'objectifs" << std::endl;
        }
        
        Pathfinder PfSystem(geo_box);

        int group_nb = 1;
        bool success = false;

        while(geo_box.data.objective_groups.find(group_nb) != geo_box.data.objective_groups.end()){

            auto& node_list = geo_box.data.objective_groups[group_nb].node_ids;

            if (node_list.size() < 2){
                std::cout << "Longueur de liste empêchant la création de routes (minimum 2 POI requis)" << std::endl;
                std::cout << "POI disponibles: " << node_list.size() << std::endl;
                group_nb++;
                continue;
            }
            else {
                std::cout << "Lancement du calcul des routes pour " << node_list.size() << " POI" << std::endl;
            }

            success = success | PfSystem.Subgraph_construction(PfSystem, node_list, group_nb);

            group_nb++;
        }

        if (success) {
            std::cout << "Construction du sous-graphe réussie!" << std::endl;
            std::cout << "Cache Name to save : ";
            std::cin >> cache_name;
            cache_name = cache_dir + "//" + cache_name + ".json";
            GeoBoxManager::save_geobox(geo_box, cache_name);
            std::cout << "GeoBox avec pathfinding sauvegardée: " << cache_name << std::endl;
        } else {
            std::cout << "Échec de la construction du sous-graphe" << std::endl;
        }

        std::string output_name;
        std::cout << "Nom de sortie pour la carte : ";
        std::cin >> output_name;
    
        // Rendu de la carte
        bool render_success = GeoBoxManager::render_geobox(geo_box, output_name, 2000, 2000);

        if (render_success) {
            std::cout << "Carte rendue avec succès: " << output_name << std::endl;
        } else {
            std::cout << "Erreur lors du rendu de la carte" << std::endl;
        }
    
    } else if (rep == "V" || rep == "v") {
        
        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur lors du rechargement de la GeoBox" << std::endl;
            return 0;
        }
        
        // Appel de la fonction de validation depuis utility
        validate_data_integrity(geo_box);
        
    } else if (rep == "O" || rep == "o") {
    
        std::cout << "\n=== Récupération de données Overpass API ===" << std::endl;
        
        try {
            OverpassAPI api;
            api.setTimeout(300); // 5 minutes
            
            std::cout << "\nLocalisation sélectionnée: " << location_name << std::endl;
            std::cout << "BBox: " << bbox << std::endl;
            
            // Construire la requête
            std::string query = api.getBoundingBoxQuery(min_lat, min_lon, max_lat, max_lon);
            
            // Nom du fichier de sortie
            std::string output_file = cache_dir + "\\overpass_" + location_name + "_raw.json";
            
            std::cout << "\nRécupération des données..." << std::endl;
            std::cout << "Cela peut prendre quelques minutes..." << std::endl;
            
            // Récupérer et sauvegarder
            bool success = api.queryToFile(query, output_file);
            
            if (success) {
                std::cout << "\n✓ Données récupérées avec succès!" << std::endl;
                std::cout << "Fichier: " << output_file << std::endl;
            } else {
                std::cout << "\n✗ Échec de la récupération des données" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Erreur: " << e.what() << std::endl;
        }
        
        std::cout << "=== Fin Overpass API ===\n" << std::endl;

    }else if (rep == "F" || rep == "f") {

        std::string group_input;
        std::cout << "Which group to check (nb) : ";
        std::cin >> group_input;

        int group_nb = std::stoi(group_input);

        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);

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

    } else if (rep == "R" || rep == "r") {

        // ========== RENDER ONLY ==========
        std::cout << "\n=== Rendu de carte uniquement ===" << std::endl;
        
        std::cout << "Cache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur lors du chargement de la GeoBox" << std::endl;
            return 0;
        }
        else {
            std::cout << "Succès du chargement de la GeoBox" << std::endl;
        }
        
        std::string output_name;
        std::cout << "Nom de sortie pour la carte : ";
        std::cin >> output_name;
        
        // Rendu de la carte
        bool render_success = GeoBoxManager::render_geobox(geo_box, output_name, 2000, 2000);
        
        if (render_success) {
            std::cout << "Carte rendue avec succès: " << output_name << std::endl;
        } else {
            std::cout << "Erreur lors du rendu de la carte" << std::endl;
        }

    } else if (rep == "A" || rep == "a") {
        
        // ========== SELECTION DE LA METAHEURISTIQUE ==========
        std::cout << "\n=== Sélection de la méthode métaheuristique ===" << std::endl;
        std::cout << "1. ACO (Ant Colony Optimization)" << std::endl;
        std::cout << "2. GRASP (Greedy Randomized Adaptive Search)" << std::endl;
        std::cout << "3. VNS (Variable Neighborhood Search)" << std::endl;
        std::cout << "4. PSO (Particle Swarm Optimization)" << std::endl;
        std::cout << "Choisissez une méthode (1-4) : ";
        
        int metaheuristic_choice;
        std::cin >> metaheuristic_choice;

        // Charger la GeoBox avec les objectifs déjà initialisés
        std::cout << "\nCache Name to load : ";
        std::cin >> cache_name;
        cache_name = cache_dir + "//" + cache_name + ".json";
        GeoBox geo_box = GeoBoxManager::load_geobox(cache_name);
        
        if (!geo_box.is_valid) {
            std::cout << "Erreur: Cache d'objectifs introuvable. Utilisez d'abord l'option 'I' pour initialiser les POI." << std::endl;
            std::cout << "Fichier recherché: " << cache_name << std::endl;
            return 0;
        } else {
            std::cout << "Succès du chargement du cache d'objectifs" << std::endl;
        }

        auto debut = std::chrono::high_resolution_clock::now();
        bool overall_success = true;
        int processed_groups = 0;

        switch(metaheuristic_choice) {
            case 1: { // ACO
                std::cout << "\n=== PATHFINDING ACO POUR TOUS LES GROUPES ===" << std::endl;
                
                // Configuration des paramètres ACO
                ACOParams aco_params;
                
                std::cout << "\n=== Configuration ACO ===" << std::endl;
                std::cout << "Utiliser les paramètres par défaut? (y/n): ";
                char use_default;
                std::cin >> use_default;
                
                if (use_default != 'y' && use_default != 'Y') {
                    std::cout << "Nombre de fourmis [" << aco_params.num_ants << "]: ";
                    std::string input;
                    std::cin >> input;
                    if (!input.empty()) aco_params.num_ants = std::stoi(input);
                    
                    std::cout << "Nombre d'itérations [" << aco_params.max_iterations << "]: ";
                    std::cin >> input;
                    if (!input.empty()) aco_params.max_iterations = std::stoi(input);
                    
                    std::cout << "Alpha (importance phéromones) [" << aco_params.alpha << "]: ";
                    std::cin >> input;
                    if (!input.empty()) aco_params.alpha = std::stod(input);
                    
                    std::cout << "Beta (importance distance) [" << aco_params.beta << "]: ";
                    std::cin >> input;
                    if (!input.empty()) aco_params.beta = std::stod(input);
                    
                    std::cout << "Rho (évaporation) [" << aco_params.rho << "]: ";
                    std::cin >> input;
                    if (!input.empty()) aco_params.rho = std::stod(input);
                }

                std::cout << "\nParamètres ACO utilisés:" << std::endl;
                std::cout << "  Fourmis: " << aco_params.num_ants << std::endl;
                std::cout << "  Itérations: " << aco_params.max_iterations << std::endl;
                std::cout << "  Alpha: " << aco_params.alpha << std::endl;
                std::cout << "  Beta: " << aco_params.beta << std::endl;
                std::cout << "  Rho: " << aco_params.rho << std::endl;

                // Exécution ACO
                ACOSolver aco_solver(geo_box);
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() < 2) {
                        std::cout << "Groupe " << group_id << " ignoré (moins de 2 POI)" << std::endl;
                        continue;
                    }

                    std::cout << "\n--- TRAITEMENT ACO GROUPE " << group_id << " ---" << std::endl;
                    std::cout << "Nom: " << group_info.name << std::endl;
                    std::cout << "POI: " << group_info.node_ids.size() << std::endl;

                    bool group_success = aco_solver.solve_single_group(
                        group_info.node_ids, 
                        group_id, 
                        aco_params
                    );

                    if (group_success) {
                        std::cout << "Groupe " << group_id << ": SUCCÈS" << std::endl;
                        processed_groups++;
                    } else {
                        std::cout << "Groupe " << group_id << ": ÉCHEC" << std::endl;
                        overall_success = false;
                    }
                }
                break;
            }
            
            case 2: { // GRASP
                std::cout << "\n=== PATHFINDING GRASP POUR TOUS LES GROUPES ===" << std::endl;
                
                // Configuration des paramètres GRASP
                GRASPParams grasp_params;
                
                std::cout << "\n=== Configuration GRASP ===" << std::endl;
                std::cout << "Utiliser les paramètres par défaut? (y/n): ";
                char use_default;
                std::cin >> use_default;
                
                if (use_default != 'y' && use_default != 'Y') {
                    std::cout << "Nombre d'itérations [" << grasp_params.max_iterations << "]: ";
                    std::string input;
                    std::cin >> input;
                    if (!input.empty()) grasp_params.max_iterations = std::stoi(input);
                    
                    std::cout << "Alpha (randomisation 0.0-1.0) [" << grasp_params.alpha << "]: ";
                    std::cin >> input;
                    if (!input.empty()) grasp_params.alpha = std::stod(input);
                    
                    std::cout << "Itérations recherche locale [" << grasp_params.local_search_iterations << "]: ";
                    std::cin >> input;
                    if (!input.empty()) grasp_params.local_search_iterations = std::stoi(input);
                    
                    std::cout << "Utiliser 2-opt? (y/n) [" << (grasp_params.use_2opt ? "y" : "n") << "]: ";
                    std::cin >> input;
                    if (!input.empty()) grasp_params.use_2opt = (input[0] == 'y' || input[0] == 'Y');
                }

                std::cout << "\nParamètres GRASP utilisés:" << std::endl;
                std::cout << "  Itérations: " << grasp_params.max_iterations << std::endl;
                std::cout << "  Alpha: " << grasp_params.alpha << std::endl;
                std::cout << "  Recherche locale: " << grasp_params.local_search_iterations << std::endl;
                std::cout << "  2-opt: " << (grasp_params.use_2opt ? "Oui" : "Non") << std::endl;

                // Exécution GRASP
                GRASPSolver grasp_solver(geo_box);
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() < 2) {
                        std::cout << "Groupe " << group_id << " ignoré (moins de 2 POI)" << std::endl;
                        continue;
                    }

                    std::cout << "\n--- TRAITEMENT GRASP GROUPE " << group_id << " ---" << std::endl;
                    std::cout << "Nom: " << group_info.name << std::endl;
                    std::cout << "POI: " << group_info.node_ids.size() << std::endl;

                    bool group_success = grasp_solver.solve_single_group(
                        group_info.node_ids, 
                        group_id, 
                        grasp_params
                    );

                    if (group_success) {
                        std::cout << "Groupe " << group_id << ": SUCCÈS" << std::endl;
                        processed_groups++;
                    } else {
                        std::cout << "Groupe " << group_id << ": ÉCHEC" << std::endl;
                        overall_success = false;
                    }
                }
                break;
            }
            
            case 3: { // VNS
                std::cout << "\n=== PATHFINDING VNS POUR TOUS LES GROUPES ===" << std::endl;
                
                // Configuration des paramètres VNS
                VNSParams vns_params;
                
                std::cout << "\n=== Configuration VNS ===" << std::endl;
                std::cout << "Utiliser les paramètres par défaut? (y/n): ";
                char use_default;
                std::cin >> use_default;
                
                if (use_default != 'y' && use_default != 'Y') {
                    std::cout << "Nombre d'itérations [" << vns_params.max_iterations << "]: ";
                    std::string input;
                    std::cin >> input;
                    if (!input.empty()) vns_params.max_iterations = std::stoi(input);
                    
                    std::cout << "Nombre de voisinages [" << vns_params.max_neighborhoods << "]: ";
                    std::cin >> input;
                    if (!input.empty()) vns_params.max_neighborhoods = std::stoi(input);
                    
                    std::cout << "Intensité perturbation (0.1-0.5) [" << vns_params.shaking_intensity << "]: ";
                    std::cin >> input;
                    if (!input.empty()) vns_params.shaking_intensity = std::stod(input);
                    
                    std::cout << "First improvement? (y/n) [" << (vns_params.use_first_improvement ? "y" : "n") << "]: ";
                    std::cin >> input;
                    if (!input.empty()) vns_params.use_first_improvement = (input[0] == 'y' || input[0] == 'Y');
                }

                std::cout << "\nParamètres VNS utilisés:" << std::endl;
                std::cout << "  Itérations: " << vns_params.max_iterations << std::endl;
                std::cout << "  Voisinages: " << vns_params.max_neighborhoods << std::endl;
                std::cout << "  Intensité: " << vns_params.shaking_intensity << std::endl;
                std::cout << "  First improvement: " << (vns_params.use_first_improvement ? "Oui" : "Non") << std::endl;

                // Exécution VNS
                VNSSolver vns_solver(geo_box);
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() < 2) {
                        std::cout << "Groupe " << group_id << " ignoré (moins de 2 POI)" << std::endl;
                        continue;
                    }

                    std::cout << "\n--- TRAITEMENT VNS GROUPE " << group_id << " ---" << std::endl;
                    std::cout << "Nom: " << group_info.name << std::endl;
                    std::cout << "POI: " << group_info.node_ids.size() << std::endl;

                    bool group_success = vns_solver.solve_single_group(
                        group_info.node_ids, 
                        group_id, 
                        vns_params
                    );

                    if (group_success) {
                        std::cout << "Groupe " << group_id << ": SUCCÈS" << std::endl;
                        processed_groups++;
                    } else {
                        std::cout << "Groupe " << group_id << ": ÉCHEC" << std::endl;
                        overall_success = false;
                    }
                }
                break;
            }
            
            case 4: { // PSO
                std::cout << "\n=== PATHFINDING PSO POUR TOUS LES GROUPES ===" << std::endl;
                
                // Configuration des paramètres PSO
                PSOParams pso_params;
                
                std::cout << "\n=== Configuration PSO ===" << std::endl;
                std::cout << "Utiliser les paramètres par défaut? (y/n): ";
                char use_default;
                std::cin >> use_default;
                
                if (use_default != 'y' && use_default != 'Y') {
                    std::cout << "Nombre de particules [" << pso_params.num_particles << "]: ";
                    std::string input;
                    std::cin >> input;
                    if (!input.empty()) pso_params.num_particles = std::stoi(input);
                    
                    std::cout << "Nombre d'itérations [" << pso_params.max_iterations << "]: ";
                    std::cin >> input;
                    if (!input.empty()) pso_params.max_iterations = std::stoi(input);
                    
                    std::cout << "Inertie (w) [" << pso_params.w << "]: ";
                    std::cin >> input;
                    if (!input.empty()) pso_params.w = std::stod(input);
                    
                    std::cout << "Coefficient cognitif (c1) [" << pso_params.c1 << "]: ";
                    std::cin >> input;
                    if (!input.empty()) pso_params.c1 = std::stod(input);
                    
                    std::cout << "Coefficient social (c2) [" << pso_params.c2 << "]: ";
                    std::cin >> input;
                    if (!input.empty()) pso_params.c2 = std::stod(input);
                }

                std::cout << "\nParamètres PSO utilisés:" << std::endl;
                std::cout << "  Particules: " << pso_params.num_particles << std::endl;
                std::cout << "  Itérations: " << pso_params.max_iterations << std::endl;
                std::cout << "  Inertie (w): " << pso_params.w << std::endl;
                std::cout << "  Cognitif (c1): " << pso_params.c1 << std::endl;
                std::cout << "  Social (c2): " << pso_params.c2 << std::endl;

                // Exécution PSO
                PSOSolver pso_solver(geo_box);
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() < 2) {
                        std::cout << "Groupe " << group_id << " ignoré (moins de 2 POI)" << std::endl;
                        continue;
                    }

                    std::cout << "\n--- TRAITEMENT PSO GROUPE " << group_id << " ---" << std::endl;
                    std::cout << "Nom: " << group_info.name << std::endl;
                    std::cout << "POI: " << group_info.node_ids.size() << std::endl;

                    bool group_success = pso_solver.solve_single_group(
                        group_info.node_ids, 
                        group_id, 
                        pso_params
                    );

                    if (group_success) {
                        std::cout << "Groupe " << group_id << ": SUCCÈS" << std::endl;
                        processed_groups++;
                    } else {
                        std::cout << "Groupe " << group_id << ": ÉCHEC" << std::endl;
                        overall_success = false;
                    }
                }
                break;
            }
            
            default:
                std::cout << "Choix invalide, utilisation d'ACO par défaut" << std::endl;
                ACOSolver default_aco_solver(geo_box);
                ACOParams default_params;
                
                for (auto& [group_id, group_info] : geo_box.data.objective_groups) {
                    if (group_info.node_ids.size() >= 2) {
                        if (default_aco_solver.solve_single_group(group_info.node_ids, group_id, default_params)) {
                            processed_groups++;
                        } else {
                            overall_success = false;
                        }
                    }
                }
                break;
        }

        auto fin = std::chrono::high_resolution_clock::now();

        // Résultats finaux
        std::cout << "\n=== RÉSULTATS FINAUX ===" << std::endl;
        std::cout << "Groupes traités avec succès: " << processed_groups << std::endl;
        std::cout << "Statut global: " << (overall_success ? "SUCCÈS" : "PARTIEL") << std::endl;

        if (processed_groups > 0) {
            std::cout << "\nCache Name to save : ";
            std::cin >> cache_name;
            cache_name = cache_dir + "//" + cache_name + ".json";
            GeoBoxManager::save_geobox(geo_box, cache_name);
            std::cout << "GeoBox avec pathfinding métaheuristique sauvegardée: " << cache_name << std::endl;
        }

        std::string output_name;
        std::cout << "Nom de sortie pour la carte : ";
        std::cin >> output_name;

        // Rendu de la carte
        bool render_success = GeoBoxManager::render_geobox(geo_box, output_name, 2000, 2000);

        if (render_success) {
            std::cout << "Carte rendue avec succès: " << output_name << std::endl;
        } else {
            std::cout << "Erreur lors du rendu de la carte" << std::endl;
        }

        auto duree = std::chrono::duration_cast<std::chrono::milliseconds>(fin - debut);
        std::cout << "Temps d'exécution: " << duree.count() << " ms" << std::endl;
    } // <-- CORRECTION: Accolade manquante ajoutée ici

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