#include "Box.hpp"
#include <iostream>
#include <algorithm>
#include <osmium/visitor.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/io/error.hpp>
#include <osmium/geom/haversine.hpp>

MyHandler ProcessOSMData(const std::string& osm_filename, double min_lon, double min_lat, double max_lon, double max_lat) {
    MyHandler handler;
    
    // Définir la bounding box personnalisée
    handler.set_bounding_box(min_lon, min_lat, max_lon, max_lat);
    
    std::cout << "Reading OSM file: " << osm_filename << std::endl;
    std::cout << "Filtering for Custom BBox: "
              << handler.Map_bbox.bottom_left().lon() << ", "
              << handler.Map_bbox.bottom_left().lat() << " to "
              << handler.Map_bbox.top_right().lon() << ", "
              << handler.Map_bbox.top_right().lat() << std::endl;

    try {
        // Créer le reader sans bounding box
        osmium::io::Reader reader(osm_filename, osmium::io::read_meta::no);
        
        // Activer le filtrage par bounding box dans le handler
        handler.enable_bounding_box_filter(true);
        std::cout << "Bounding box filtering enabled in handler." << std::endl;
        
        osmium::apply(reader, handler);
        reader.close();

        std::cout << "Finished reading file." << std::endl;
        std::cout << "Nodes found in BBox: " << handler.data_collector.nodes.size() << std::endl;
        std::cout << "Ways found in BBox: " << handler.data_collector.ways.size() << std::endl;

        if (!handler.data_collector.nodes.empty()) {
            std::cout << "\nFirst 5 Nodes in BBox:\n";
            int count = 0;
            for (const auto& [id, point] : handler.data_collector.nodes) {
                if (count >= 5) break;
                std::cout << "  Node ID: " << point.id 
                         << ", Lat: " << point.lat 
                         << ", Lon: " << point.lon << std::endl;
                count++;
            }
        }

    } catch (const osmium::io_error& e) {
        std::cerr << "OSM I/O Error: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }

    return handler;
}

int Test(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <osm_pbf_file>\n";
        return 1;
    }

    std::string osm_filename = argv[1];
    
    try {
        // Utiliser la bounding box par défaut de Shibuya
        MyHandler handler = ProcessOSMData(osm_filename, 139.700, 35.655, 139.715, 35.665);
        
        std::cout << "Test completed successfully." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}