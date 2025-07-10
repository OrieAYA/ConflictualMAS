#include <iostream>
#include <iomanip>
#include <Box.hpp>
#include <MapRenderer.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/visitor.hpp>
#include <osmium/io/error.hpp>
#include <osmium/geom/coordinates.hpp>

int main() {
    std::cout << "Starting ConflictualMAS application." << std::endl;

    const std::string osm_filepath = "C:\\Users\\screp\\OneDrive\\Bureau\\Algorithms\\ConflictualMAS\\src\\maps\\kanto-latest.osm.pbf";
    
    // Demander à l'utilisateur de saisir les coordonnées de la bounding box
    double min_lon, min_lat, max_lon, max_lat;
    std::string map_name;
    
    std::cout << "\n=== Configuration de la Bounding Box ===" << std::endl;
    std::cout << "Entrez les coordonnées de la zone à extraire:" << std::endl;
    
    // Proposer des valeurs par défaut
    std::cout << "\nValeurs par défaut (Shibuya):" << std::endl;
    std::cout << "  Min Longitude: 139.700" << std::endl;
    std::cout << "  Min Latitude:  35.655" << std::endl;
    std::cout << "  Max Longitude: 139.715" << std::endl;
    std::cout << "  Max Latitude:  35.665" << std::endl;
    
    char use_default;
    std::cout << "\nUtiliser les valeurs par défaut (Shibuya) ? (y/n): ";
    std::cin >> use_default;
    
    if (use_default == 'y' || use_default == 'Y') {
        min_lon = 139.700;
        min_lat = 35.655;
        max_lon = 139.715;
        max_lat = 35.665;
        map_name = "shibuya_map";
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
    
    // Afficher les coordonnées sélectionnées
    std::cout << "\n=== Coordonnées sélectionnées ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Bounding Box: (" << min_lon << ", " << min_lat << ") à (" << max_lon << ", " << max_lat << ")" << std::endl;
    std::cout << "Nom de la carte: " << map_name << std::endl;
    
    try {
        // Utiliser la fonction ProcessOSMData pour traiter les données
        std::cout << "\n=== Traitement des données OSM ===" << std::endl;
        MyHandler handler = ProcessOSMData(osm_filepath, min_lon, min_lat, max_lon, max_lat);

        std::cout << "\nData collection complete." << std::endl;
        std::cout << "Nodes collected: " << handler.data_collector.nodes.size() << std::endl;
        std::cout << "Ways collected: " << handler.data_collector.ways.size() << std::endl;

        if (!handler.data_collector.nodes.empty()) {
            std::cout << "\n=== Rendu de la carte ===" << std::endl;
            MapRenderer renderer;
            
            // Utiliser la même bounding box que celle utilisée pour le filtrage
            renderer.render_shibuya_map(
                handler.data_collector, 
                handler.Map_bbox,  // Utiliser la bounding box du handler
                map_name, 
                2000, 
                2000
            );
        } else {
            std::cout << "No data found in the specified bounding box." << std::endl;
            std::cout << "Vérifiez que les coordonnées sont correctes et que la zone contient des données OSM." << std::endl;
        }

    } catch (const osmium::io_error& e) {
        std::cerr << "Osmium IO Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nApplication finished." << std::endl;
    return 0;
}