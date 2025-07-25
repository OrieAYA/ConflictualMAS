#ifndef GEOBOX_MANAGER_HPP
#define GEOBOX_MANAGER_HPP

#include "Box.hpp"
#include "MapRenderer.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <ctime>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class GeoBoxManager {
public:
    
    // Sauvegarder une GeoBox
    static bool save_geobox(const GeoBox& geo_box, const std::string& filepath);
    
    // Récupérer une GeoBox sauvegardée
    static GeoBox load_geobox(const std::string& filepath);
    
    // Rendre une GeoBox en carte
    static bool render_geobox(const GeoBox& geo_box, 
                            const std::string& output_name,
                            int width = 2000, int height = 2000);
    
    // === FONCTIONS UTILITAIRES ===
    
    // Vérifier si un fichier de cache existe
    static bool cache_exists(const std::string& filepath);
    
    // Générer un nom de fichier de cache
    static std::string generate_cache_name(const std::string& prefix = "geobox");
    
    // Afficher les informations d'une GeoBox
    static void display_geobox_info(const GeoBox& geo_box);

private:
    // === FONCTIONS INTERNES DE SÉRIALISATION ===
    
    // Convertir MyData en JSON
    static json serialize_data(const MyData& data);
    
    // Convertir JSON en MyData
    static MyData deserialize_data(const json& j);
    
    // Convertir osmium::Box en JSON
    static json serialize_bbox(const osmium::Box& bbox);
    
    // Convertir JSON en osmium::Box
    static osmium::Box deserialize_bbox(const json& j);
};

#endif // GEOBOX_MANAGER_HPP