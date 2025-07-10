#include "Box.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <osmium/visitor.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/io/error.hpp>

// Implémentation de la fonction haversine dans MyHandler
double MyHandler::calculate_haversine_distance(double lat1, double lon1, double lat2, double lon2) const {
    const double R = 6371000.0; // Rayon de la Terre en mètres
    const double M_PI = 3.14159265358979323846;
    const double dLat = (lat2 - lat1) * M_PI / 180.0;
    const double dLon = (lon2 - lon1) * M_PI / 180.0;
    const double a = sin(dLat/2) * sin(dLat/2) +
                    cos(lat1 * M_PI / 180.0) * cos(lat2 * M_PI / 180.0) *
                    sin(dLon/2) * sin(dLon/2);
    const double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return R * c;
}

// Fonction indépendante pour créer une GeoBox
GeoBox create_geo_box(const std::string& osm_filename, 
                      double min_lon, double min_lat, 
                      double max_lon, double max_lat) {
    
    std::cout << "Creating GeoBox from: " << osm_filename << std::endl;
    std::cout << "BBox: (" << min_lon << ", " << min_lat << ") to (" << max_lon << ", " << max_lat << ")" << std::endl;
    
    try {
        // Créer et configurer le handler
        MyHandler handler;
        handler.set_bounding_box(min_lon, min_lat, max_lon, max_lat);
        handler.enable_bounding_box_filter(true);
        
        // Traiter le fichier OSM
        osmium::io::Reader reader(osm_filename, osmium::io::read_meta::no);
        osmium::apply(reader, handler);
        reader.close();
        
        // Ajouter les ways incidents aux points
        for (const auto& [way_id, way] : handler.data_collector.ways) {
            for (const auto& point : way.points) {
                auto it = handler.data_collector.nodes.find(point.id);
                if (it != handler.data_collector.nodes.end()) {
                    it->second.incident_ways.push_back(way_id);
                }
            }
        }
        
        std::cout << "Processing complete:" << std::endl;
        std::cout << "  Nodes found: " << handler.data_collector.nodes.size() << std::endl;
        std::cout << "  Ways found: " << handler.data_collector.ways.size() << std::endl;
        
        // Retourner une GeoBox valide
        return GeoBox(handler.data_collector, handler.Map_bbox, osm_filename);
        
    } catch (const osmium::io_error& e) {
        std::cerr << "OSM I/O Error: " << e.what() << std::endl;
        return GeoBox(); // GeoBox invalide
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return GeoBox(); // GeoBox invalide
    }
}