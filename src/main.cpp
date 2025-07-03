#include <iostream>
#include <Box.hpp>
#include <MapRenderer.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/visitor.hpp>
#include <osmium/io/error.hpp>
#include <osmium/geom/coordinates.hpp>

int main() {
    std::cout << "Starting ConflictualMAS application." << std::endl;

    const std::string osm_filepath = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    
    try {
        MyHandler handler;

        std::cout << "Reading OSM file: " << osm_filepath << std::endl;
    
        osmium::io::Reader reader(osm_filepath, osmium::io::read_meta::no);
    
        handler.enable_bounding_box_filter(true);
        std::cout << "Bounding box filtering enabled in handler." << std::endl;
    
        osmium::apply(reader, handler);
        reader.close();

        std::cout << "Data collection complete." << std::endl;
        std::cout << "Nodes collected: " << handler.data_collector.nodes.size() << std::endl;
        std::cout << "Ways collected: " << handler.data_collector.ways.size() << std::endl;

        if (!handler.data_collector.nodes.empty()) {
            MapRenderer renderer;
            osmium::Box render_bbox(139.7, 35.655, 139.715, 35.665);
            
            renderer.render_shibuya_map(
                handler.data_collector, 
                render_bbox, 
                "shibuya_map.png", 
                2000, 
                2000
            );
        } else {
            std::cout << "No data found in the specified bounding box." << std::endl;
        }

    } catch (const osmium::io_error& e) {
        std::cerr << "Osmium IO Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Application finished." << std::endl;
    return 0;
}
