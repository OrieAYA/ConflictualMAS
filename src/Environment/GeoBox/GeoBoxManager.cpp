#include "GeoBoxManager.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

// Sauvegarder une GeoBox
bool GeoBoxManager::save_geobox(const GeoBox& geo_box, const std::string& filepath) {
    std::cout << "=== Sauvegarde de GeoBox ===" << std::endl;
    std::cout << "Fichier: " << filepath << std::endl;
    
    if (!geo_box.is_valid) {
        std::cerr << "Erreur: GeoBox invalide, impossible de sauvegarder" << std::endl;
        return false;
    }
    
    try {
        json j;
        
        // Métadonnées
        j["version"] = "1.0";
        j["timestamp"] = std::time(nullptr);
        j["source_file"] = geo_box.source_file;
        j["is_valid"] = geo_box.is_valid;
        
        // Données géographiques
        j["data"] = serialize_data(geo_box.data);
        j["bbox"] = serialize_bbox(geo_box.bbox);
        
        // Statistiques pour validation
        json stats_json;
        stats_json["nodes_count"] = geo_box.data.nodes.size();
        stats_json["ways_count"] = geo_box.data.ways.size();
        stats_json["objective_groups_count"] = geo_box.data.objective_groups.size();
        j["stats"] = stats_json;
        
        // Écrire dans le fichier
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Cannot open file for writing: " << filepath << std::endl;
            return false;
        }
        
        file << j.dump(2); // Indentation de 2 pour la lisibilité
        file.close();
        
        std::cout << "GeoBox sauvegardée avec succès!" << std::endl;
        std::cout << "  Nodes: " << geo_box.data.nodes.size() << std::endl;
        std::cout << "  Ways: " << geo_box.data.ways.size() << std::endl;
        std::cout << "  Objective groups: " << geo_box.data.objective_groups.size() << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de la sauvegarde: " << e.what() << std::endl;
        return false;
    }
}

// Récupérer une GeoBox sauvegardée
GeoBox GeoBoxManager::load_geobox(const std::string& filepath) {
    std::cout << "=== Chargement de GeoBox ===" << std::endl;
    std::cout << "Fichier: " << filepath << std::endl;
    
    if (!cache_exists(filepath)) {
        std::cerr << "Erreur: Fichier de cache introuvable" << std::endl;
        return GeoBox(); // GeoBox invalide
    }
    
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Cannot open file for reading: " << filepath << std::endl;
            return GeoBox(); // GeoBox invalide
        }
        
        json j;
        file >> j;
        file.close();
        
        // Vérifier la version
        if (j.contains("version")) {
            std::cout << "Cache version: " << j["version"] << std::endl;
        }
        
        // Reconstruire la GeoBox
        GeoBox geo_box;
        geo_box.source_file = j.value("source_file", "unknown");
        geo_box.is_valid = j.value("is_valid", false);
        geo_box.data = deserialize_data(j["data"]);
        geo_box.bbox = deserialize_bbox(j["bbox"]);
        
        // Vérifier les statistiques
        if (j.contains("stats")) {
            auto stats = j["stats"];
            std::cout << "GeoBox chargée avec succès!" << std::endl;
            std::cout << "  Nodes: " << stats.value("nodes_count", 0) << std::endl;
            std::cout << "  Ways: " << stats.value("ways_count", 0) << std::endl;
            std::cout << "  Objective groups: " << stats.value("objective_groups_count", 0) << std::endl;
        }
        
        return geo_box;
        
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors du chargement: " << e.what() << std::endl;
        return GeoBox(); // GeoBox invalide
    }
}

// Rendre une GeoBox en carte
bool GeoBoxManager::render_geobox(const GeoBox& geo_box, 
                                const std::string& output_name,
                                int width, int height) {
    
    std::cout << "=== Rendu de GeoBox ===" << std::endl;
    std::cout << "Nom de sortie: " << output_name << std::endl;
    std::cout << "Dimensions: " << width << "x" << height << std::endl;
    
    if (!geo_box.is_valid) {
        std::cerr << "Erreur: GeoBox invalide, impossible de rendre" << std::endl;
        return false;
    }
    
    if (geo_box.data.nodes.empty()) {
        std::cerr << "Erreur: Aucune donnée à rendre" << std::endl;
        return false;
    }
    
    bool success = render_map(geo_box, output_name, width, height);
    
    if (success) {
        std::cout << "Rendu réussi!" << std::endl;
    } else {
        std::cerr << "Erreur lors du rendu" << std::endl;
    }
    
    return success;
}

// === FONCTIONS UTILITAIRES ===

// Vérifier si un fichier de cache existe
bool GeoBoxManager::cache_exists(const std::string& filepath) {
    return std::filesystem::exists(filepath);
}

// Générer un nom de fichier de cache
std::string GeoBoxManager::generate_cache_name(const std::string& prefix) {
    std::ostringstream oss;
    oss << prefix << ".json";
    return oss.str();
}

// Afficher les informations d'une GeoBox
void GeoBoxManager::display_geobox_info(const GeoBox& geo_box) {
    std::cout << "\n=== Informations GeoBox ===" << std::endl;
    std::cout << "Valide: " << (geo_box.is_valid ? "Oui" : "Non") << std::endl;
    
    if (!geo_box.is_valid) {
        std::cout << "GeoBox invalide" << std::endl;
        return;
    }
    
    std::cout << "Fichier source: " << geo_box.source_file << std::endl;
    std::cout << "Nodes: " << geo_box.data.nodes.size() << std::endl;
    std::cout << "Ways: " << geo_box.data.ways.size() << std::endl;
    std::cout << "Groupes d'objectifs: " << geo_box.data.objective_groups.size() << std::endl;
    
    // Afficher la bounding box
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "BBox: (" 
              << geo_box.bbox.bottom_left().lon() << ", " 
              << geo_box.bbox.bottom_left().lat() << ") à (" 
              << geo_box.bbox.top_right().lon() << ", " 
              << geo_box.bbox.top_right().lat() << ")" << std::endl;
    
    // Afficher les groupes d'objectifs
    if (!geo_box.data.objective_groups.empty()) {
        geo_box.data.print_objective_groups();
    }
}

// === FONCTIONS INTERNES DE SÉRIALISATION ===

// Convertir MyData en JSON
json GeoBoxManager::serialize_data(const MyData& data) {
    json j;
    
    // Sérialiser les nodes
    json nodes_json = json::object();
    for (const auto& [node_id, point] : data.nodes) {
        json point_json;
        point_json["lat"] = point.lat;
        point_json["lon"] = point.lon;
        point_json["id"] = point.id;
        point_json["incident_ways"] = json::array();
        for (const auto& way_id : point.incident_ways) {
            point_json["incident_ways"].push_back(way_id);
        }
        point_json["groupes"] = json::array();
        for (int group : point.groupes) {
            point_json["groupes"].push_back(group);
        }
        point_json["objective_id"] = point.objective_id;
        
        nodes_json[std::to_string(node_id)] = point_json;
    }
    j["nodes"] = nodes_json;
    
    // Sérialiser les ways
    json ways_json = json::object();
    for (const auto& [way_id, way] : data.ways) {
        json way_json;
        way_json["id"] = way.id;
        way_json["node1_id"] = way.node1_id;
        way_json["node2_id"] = way.node2_id;
        way_json["groupes"] = json::array();
        for (int group : way.groupes) {
            way_json["groupes"].push_back(group);
        }
        way_json["distance_meters"] = way.distance_meters;
        way_json["points"] = json::array();
        
        for (const auto& point : way.points) {
            json point_json;
            point_json["id"] = point.id;
            way_json["points"].push_back(point_json);
        }
        
        ways_json[std::to_string(way_id)] = way_json;
    }
    j["ways"] = ways_json;
    
    // Sérialiser les groupes d'objectifs
    json groups_json = json::object();
    for (const auto& [group_id, group] : data.objective_groups) {
        json group_json;
        group_json["id"] = group.id;
        group_json["name"] = group.name;
        group_json["description"] = group.description;
        group_json["point_count"] = group.point_count;
        group_json["node_ids"] = group.node_ids;
        
        groups_json[std::to_string(group_id)] = group_json;
    }
    j["objective_groups"] = groups_json;
    
    return j;
}

// Convertir JSON en MyData
MyData GeoBoxManager::deserialize_data(const json& j) {
    MyData data;
    
    // Désérialiser les nodes
    if (j.contains("nodes")) {
        for (const auto& [node_id_str, point_json] : j["nodes"].items()) {
            osmium::object_id_type node_id = std::stoull(node_id_str);
            
            MyData::Point point;
            point.lat = point_json["lat"];
            point.lon = point_json["lon"];
            point.id = point_json["id"];
            
            // Désérialiser incident_ways
            point.incident_ways.clear();
            if (point_json.contains("incident_ways")) {
                for (const auto& way_id : point_json["incident_ways"]) {
                    point.incident_ways.push_back(way_id.get<osmium::object_id_type>());
                }
            }
            
            // MODIFIÉ: Désérialiser les groupes multiples
            point.groupes.clear();
            if (point_json.contains("groupes")) {
                // Nouveau format avec groupes multiples
                for (const auto& group : point_json["groupes"]) {
                    int group_val = group.get<int>();
                    if (group_val != 0) {
                        point.groupes.insert(group_val);
                    }
                }
            } else if (point_json.contains("groupe")) {
                // Compatibilité avec l'ancien format
                int old_group = point_json["groupe"];
                if (old_group != 0) {
                    point.groupes.insert(old_group);
                }
            }
            
            point.objective_id = point_json.value("objective_id", "");
            
            data.nodes[node_id] = point;
        }
    }
    
    // Désérialiser les ways
    if (j.contains("ways")) {
        for (const auto& [way_id_str, way_json] : j["ways"].items()) {
            osmium::object_id_type way_id = std::stoull(way_id_str);
            
            MyData::Way way;
            way.id = way_json["id"];
            way.node1_id = way_json["node1_id"];
            way.node2_id = way_json["node2_id"];
            way.distance_meters = way_json["distance_meters"];
            
            // MODIFIÉ: Désérialiser les groupes multiples
            way.groupes.clear();
            if (way_json.contains("groupes")) {
                // Nouveau format avec groupes multiples
                for (const auto& group : way_json["groupes"]) {
                    int group_val = group.get<int>();
                    if (group_val != 0) {
                        way.groupes.insert(group_val);
                    }
                }
            } else if (way_json.contains("groupe")) {
                // Compatibilité avec l'ancien format
                int old_group = way_json.value("groupe", 0);
                if (old_group != 0) {
                    way.groupes.insert(old_group);
                }
            }
            
            data.ways[way_id] = way;
        }
    }
    
    // Désérialiser les groupes d'objectifs (inchangé)
    if (j.contains("objective_groups")) {
        for (const auto& [group_id_str, group_json] : j["objective_groups"].items()) {
            int group_id = std::stoi(group_id_str);
            
            ObjectiveGroup group;
            group.id = group_json["id"];
            group.name = group_json["name"];
            group.description = group_json["description"];
            group.point_count = group_json["point_count"];
            group.node_ids = group_json.value("node_ids", std::vector<osmium::object_id_type>{});
            
            data.objective_groups[group_id] = group;
        }
    }
    
    return data;
}

// Convertir osmium::Box en JSON
json GeoBoxManager::serialize_bbox(const osmium::Box& bbox) {
    json j;
    
    json bottom_left;
    bottom_left["lon"] = bbox.bottom_left().lon();
    bottom_left["lat"] = bbox.bottom_left().lat();
    
    json top_right;
    top_right["lon"] = bbox.top_right().lon();
    top_right["lat"] = bbox.top_right().lat();
    
    j["bottom_left"] = bottom_left;
    j["top_right"] = top_right;
    
    return j;
}

// Convertir JSON en osmium::Box
osmium::Box GeoBoxManager::deserialize_bbox(const json& j) {
    osmium::Box bbox;
    
    double min_lon = j["bottom_left"]["lon"];
    double min_lat = j["bottom_left"]["lat"];
    double max_lon = j["top_right"]["lon"];
    double max_lat = j["top_right"]["lat"];
    
    bbox.extend(osmium::Location(min_lon, min_lat));
    bbox.extend(osmium::Location(max_lon, max_lat));
    
    return bbox;
}