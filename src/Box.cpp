#include "Box.hpp"
#include <iostream>
#include <algorithm>
#include <osmium/visitor.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/io/error.hpp>

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
        
        std::cout << "Processing complete:" << std::endl;
        std::cout << "  Nodes found: " << handler.data_collector.nodes.size() << std::endl;
        std::cout << "  Ways found: " << handler.data_collector.ways.size() << std::endl;
        
        // Afficher quelques exemples de nodes
        if (!handler.data_collector.nodes.empty()) {
            std::cout << "\nFirst 3 nodes:" << std::endl;
            int count = 0;
            for (const auto& [id, point] : handler.data_collector.nodes) {
                if (count >= 3) break;
                std::cout << "  Node " << point.id << ": (" << point.lat << ", " << point.lon << ")" << std::endl;
                count++;
            }
        }
        
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