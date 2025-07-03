#ifndef BOX_HPP
#define BOX_HPP

#include <vector>
#include <unordered_map>
#include <string>
#include <osmium/osm/location.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/osm/box.hpp>  // Added missing include
#include <osmium/handler.hpp>
#include <osmium/geom/relations.hpp>

// Unified MyData struct
struct MyData {
    struct Point {
        double lat;
        double lon;
        osmium::object_id_type id = 0;
        
        // Constructor for compatibility
        Point() = default;
        Point(double lat, double lon, osmium::object_id_type id = 0) 
            : lat(lat), lon(lon), id(id) {}
    };
    
    struct Way {
        osmium::object_id_type id;
        std::vector<Point> points;
        
        Way() : id(0) {}
        Way(osmium::object_id_type id) : id(id) {}
    };

    // Use maps for efficient lookup by ID
    std::unordered_map<osmium::object_id_type, Point> nodes;
    std::unordered_map<osmium::object_id_type, Way> ways;
    
    // Helper methods for compatibility with vector-based code
    std::vector<Point> get_nodes_vector() const {
        std::vector<Point> result;
        result.reserve(nodes.size());
        for (const auto& [id, point] : nodes) {
            result.push_back(point);
        }
        return result;
    }
    
    std::vector<Way> get_ways_vector() const {
        std::vector<Way> result;
        result.reserve(ways.size());
        for (const auto& [id, way] : ways) {
            result.push_back(way);
        }
        return result;
    }
};

class MyHandler : public osmium::handler::Handler {
private:
    bool m_use_bbox_filter = false;
    std::unordered_map<osmium::object_id_type, osmium::Location> m_node_locations;

public:
    osmium::Box shibuya_bbox;
    MyData data_collector;

    MyHandler() {
        shibuya_bbox.extend(osmium::Location(139.700, 35.655)); // SW corner
        shibuya_bbox.extend(osmium::Location(139.715, 35.665)); // NE corner
    }
    
    void enable_bounding_box_filter(bool enable = true) {
        m_use_bbox_filter = enable;
    }
    
    void node(const osmium::Node& node) {
        const osmium::Location node_loc = node.location();
        
        if (!node_loc.valid()) return;
        
        // Stocker la localisation du node si filtrage activé
        if (m_use_bbox_filter) {
            m_node_locations[node.id()] = node_loc;
            
            // Filtrer le node
            if (!shibuya_bbox.contains(node_loc)) {
                return;
            }
        }
        
        // Ajouter le node aux données collectées
        data_collector.nodes[node.id()] = MyData::Point(
            node_loc.lat(), 
            node_loc.lon(), 
            node.id()
        );
    }
    
    void way(const osmium::Way& way) {
        if (way.nodes().empty()) return;
        
        MyData::Way current_way(way.id());
        bool has_nodes_in_bbox = false;
        
        // Si le filtrage est activé, vérifier si le way intersecte la bounding box
        if (m_use_bbox_filter) {
            bool intersects_bbox = false;
            
            for (const auto& node_ref : way.nodes()) {
                auto it = m_node_locations.find(node_ref.ref());
                if (it != m_node_locations.end()) {
                    if (shibuya_bbox.contains(it->second)) {
                        intersects_bbox = true;
                        break;
                    }
                }
            }
            
            if (!intersects_bbox) {
                return;
            }
        }
        
        // Construire le way avec les nodes disponibles
        for (const auto& node_ref : way.nodes()) {
            auto it = data_collector.nodes.find(node_ref.ref());
            if (it != data_collector.nodes.end()) {
                current_way.points.push_back(it->second);
                has_nodes_in_bbox = true;
            }
        }
        
        if (has_nodes_in_bbox && current_way.points.size() > 1) {
            data_collector.ways[way.id()] = current_way;
        }
    }
};

int Test(int argc, char* argv[]);

#endif // BOX_HPP