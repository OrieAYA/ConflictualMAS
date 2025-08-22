#include "Box.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <limits>
#include <queue>
#include <tuple>
#include <osmium/visitor.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/io/error.hpp>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ====================================================================
// FONCTIONS UTILITAIRES
// ====================================================================

double calculate_haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    const double R = 6371000.0;
    const double M_PI = 3.14159265358979323846;
    const double dLat = (lat2 - lat1) * M_PI / 180.0;
    const double dLon = (lon2 - lon1) * M_PI / 180.0;
    const double a = sin(dLat/2) * sin(dLat/2) +
                    cos(lat1 * M_PI / 180.0) * cos(lat2 * M_PI / 180.0) *
                    sin(dLon/2) * sin(dLon/2);
    const double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return R * c;
}

double MyHandler::calculate_haversine_distance(double lat1, double lon1, double lat2, double lon2) const {
    return ::calculate_haversine_distance(lat1, lon1, lat2, lon2);
}

osmium::object_id_type find_nearest_point(const MyData& data, double target_lat, double target_lon) {
    double min_distance = std::numeric_limits<double>::max();
    osmium::object_id_type nearest_id = 0;
    
    for (const auto& [node_id, point] : data.nodes) {
        double distance = calculate_haversine_distance(point.lat, point.lon, target_lat, target_lon);
        if (distance < min_distance) {
            min_distance = distance;
            nearest_id = node_id;
        }
    }
    
    return nearest_id;
}

bool is_valid_way_type(const osmium::Way& way) {
    bool has_highway = false;
    
    for (const auto& tag : way.tags()) {
        std::string key = tag.key();
        
        if (key == "highway") {
            has_highway = true;
        }
        
        if (key == "building" || key == "landuse" || key == "amenity" || 
            key == "leisure" || key == "natural" || key == "area" ||
            key == "barrier" || key == "waterway") {
            return false;
        }
    }
    
    return has_highway;
}

// ====================================================================
// MYHANDLER IMPLEMENTATION - AVEC CONNEXION IMMÉDIATE
// ====================================================================

void MyHandler::set_bounding_box(double min_lon, double min_lat, double max_lon, double max_lat) {
    Map_bbox = osmium::Box();
    Map_bbox.extend(osmium::Location(min_lon, min_lat));
    Map_bbox.extend(osmium::Location(max_lon, max_lat));
}

void MyHandler::node(const osmium::Node& node) {
    const osmium::Location node_loc = node.location();
    if (!node_loc.valid()) return;
    
    if (m_use_bbox_filter) {
        m_node_locations[node.id()] = node_loc;
        if (!Map_bbox.contains(node_loc)) {
            return;
        }
    }
    
    data_collector.nodes[node.id()] = MyData::Point(
        node_loc.lat(), 
        node_loc.lon(), 
        node.id()
    );
}

void MyHandler::connect_way_to_nodes(osmium::object_id_type way_id, 
                                    osmium::object_id_type node1_id, 
                                    osmium::object_id_type node2_id) {
    auto& node1 = data_collector.nodes[node1_id];
    auto& node2 = data_collector.nodes[node2_id];
    
    node1.incident_ways.push_back(way_id);
    node2.incident_ways.push_back(way_id);
}

void MyHandler::way(const osmium::Way& way) {
    if (way.nodes().empty()) return;
    if (!is_valid_way_type(way)) return;
    
    if (m_use_bbox_filter) {
        bool intersects_bbox = false;
        for (const auto& node_ref : way.nodes()) {
            auto it = m_node_locations.find(node_ref.ref());
            if (it != m_node_locations.end() && Map_bbox.contains(it->second)) {
                intersects_bbox = true;
                break;
            }
        }
        if (!intersects_bbox) return;
    }

    // Collecter les IDs des nodes valides
    std::vector<osmium::object_id_type> valid_node_ids;
    for (const auto& node_ref : way.nodes()) {
        auto it = data_collector.nodes.find(node_ref.ref());
        if (it != data_collector.nodes.end()) {
            valid_node_ids.push_back(node_ref.ref());
        }
    }
    
    if (valid_node_ids.size() < 2) return;

    if (valid_node_ids.size() == 2) {
        // Way simple avec 2 nodes
        MyData::Way current_way(way.id());
        current_way.node1_id = valid_node_ids[0];
        current_way.node2_id = valid_node_ids[1];
        
        // Récupérer les points pour le calcul de distance
        const auto& node1 = data_collector.nodes[valid_node_ids[0]];
        const auto& node2 = data_collector.nodes[valid_node_ids[1]];
        current_way.points = {node1, node2};
        current_way.distance_meters = static_cast<float>(calculate_haversine_distance(
            node1.lat, node1.lon, node2.lat, node2.lon
        ));
        
        // Ajouter le way
        data_collector.ways[way.id()] = current_way;
        
        // CONNEXION IMMÉDIATE
        connect_way_to_nodes(way.id(), current_way.node1_id, current_way.node2_id);
        
    } else {
        // Segmentation pour ways avec 3+ nodes
        osmium::object_id_type base_way_id = way.id();
        
        for (size_t i = 0; i < valid_node_ids.size() - 1; ++i) {
            MyData::Way segment_way;
            segment_way.id = (i == 0) ? base_way_id : generate_new_way_id(base_way_id, i);
            segment_way.node1_id = valid_node_ids[i];
            segment_way.node2_id = valid_node_ids[i + 1];
            
            // Récupérer les points pour le calcul
            const auto& node1 = data_collector.nodes[valid_node_ids[i]];
            const auto& node2 = data_collector.nodes[valid_node_ids[i + 1]];
            segment_way.points = {node1, node2};
            segment_way.distance_meters = static_cast<float>(calculate_haversine_distance(
                node1.lat, node1.lon, node2.lat, node2.lon
            ));
            
            // Ajouter le segment
            data_collector.ways[segment_way.id] = segment_way;
            
            // CONNEXION IMMÉDIATE
            connect_way_to_nodes(segment_way.id, segment_way.node1_id, segment_way.node2_id);
        }
    }
}

// ====================================================================
// CRÉATION DE GEOBOX SIMPLIFIÉE
// ====================================================================

GeoBox create_geo_box(const std::string& osm_filename, 
                      double min_lon, double min_lat, 
                      double max_lon, double max_lat) {
    
    std::cout << "Creating GeoBox from: " << osm_filename << std::endl;
    std::cout << "BBox: (" << min_lon << ", " << min_lat << ") to (" << max_lon << ", " << max_lat << ")" << std::endl;
    
    try {
        MyHandler handler;
        handler.set_bounding_box(min_lon, min_lat, max_lon, max_lat);
        handler.enable_bounding_box_filter(true);
        
        osmium::io::Reader reader(osm_filename, osmium::io::read_meta::no);
        osmium::apply(reader, handler);
        reader.close();
        
        // NOTE: Plus besoin de boucle de connexion - fait automatiquement dans MyHandler::way()
        
        // Nettoyage des nodes orphelins uniquement
        auto nodes_it = handler.data_collector.nodes.begin();
        while (nodes_it != handler.data_collector.nodes.end()) {
            if (nodes_it->second.incident_ways.empty()) {
                nodes_it = handler.data_collector.nodes.erase(nodes_it);
            } else {
                ++nodes_it;
            }
        }
        
        std::cout << "Processing complete:" << std::endl;
        std::cout << "  Nodes found: " << handler.data_collector.nodes.size() << std::endl;
        std::cout << "  Ways found: " << handler.data_collector.ways.size() << std::endl;

        GeoBox temp_box(handler.data_collector, handler.Map_bbox, osm_filename);
        return connect_isolated_components(temp_box);
        
    } catch (const osmium::io_error& e) {
        std::cerr << "OSM I/O Error: " << e.what() << std::endl;
        return GeoBox();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return GeoBox();
    }
}

// ====================================================================
// APPLICATION DES OBJECTIFS
// ====================================================================

GeoBox apply_objectives(GeoBox geo_box, const FlickrConfig& flickr_config, 
                       const std::string& cache_filename, bool use_cache, int group_id) {
    
    if (!geo_box.is_valid) {
        std::cerr << "Cannot apply objectives: GeoBox is invalid" << std::endl;
        return geo_box;
    }
    
    std::cout << "\n=== Application des objectifs Flickr ===" << std::endl;
    
    FlickrAPIClient client(flickr_config);
    std::vector<FlickrPOI> pois;
    
    if (use_cache && std::filesystem::exists(cache_filename)) {
        pois = client.load_pois_from_file(cache_filename);
    }
    
    if (pois.empty()) {
        std::cout << "Fetching fresh data from Flickr API..." << std::endl;
        pois = client.fetch_pois();
        
        if (!pois.empty()) {
            client.save_pois_to_file(pois, cache_filename);
        }
    }
    
    if (pois.empty()) {
        std::cout << "No POIs found, returning original GeoBox" << std::endl;
        return geo_box;
    }
    
    ObjectiveGroup group(group_id, flickr_config.search_word, 
                        "Points d'intérêt basés sur les photos Flickr");
    geo_box.data.objective_groups[group_id] = group;
    
    int assigned_count = 0;
    std::unordered_set<osmium::object_id_type> assigned_nodes;
    
    for (const auto& poi : pois) {
        osmium::object_id_type nearest_id = find_nearest_point(geo_box.data, poi.latitude, poi.longitude);
        
        if (nearest_id != 0) {
            auto it = geo_box.data.nodes.find(nearest_id);
            if (it != geo_box.data.nodes.end()) {
                double distance = calculate_haversine_distance(
                    it->second.lat, it->second.lon, poi.latitude, poi.longitude
                );
                
                if (distance <= flickr_config.poi_assignment_radius) {
                    if (assigned_nodes.count(nearest_id) == 0) {
                        assigned_nodes.insert(nearest_id);
                        it->second.groupe = group_id;
                        geo_box.data.objective_groups[group_id].node_ids.push_back(nearest_id);
                        assigned_count++;
                    }
                    it->second.objective_id = poi.id;
                }
            }
        }
    }
    
    geo_box.data.objective_groups[group_id].point_count = assigned_count;
    
    std::cout << "POIs Flickr récupérés: " << pois.size() << std::endl;
    std::cout << "Points assignés: " << assigned_count << std::endl;
    
    return geo_box;
}

// ====================================================================
// API FLICKR IMPLEMENTATION
// ====================================================================

size_t FlickrAPIClient::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

std::string FlickrAPIClient::build_flickr_url(const std::string& method, const std::map<std::string, std::string>& params) const {
    std::string url = "https://api.flickr.com/services/rest/?method=" + method;
    url += "&api_key=" + config.api_key;
    url += "&format=json&nojsoncallback=1";
    
    for (const auto& [key, value] : params) {
        url += "&" + key + "=" + value;
    }
    
    return url;
}

std::string FlickrAPIClient::make_http_request(const std::string& url) const {
    CURL* curl;
    CURLcode res;
    std::string response;
    
    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        
        if (res != CURLE_OK) {
            std::cerr << "CURL Error: " << curl_easy_strerror(res) << std::endl;
            return "";
        }
    }
    
    return response;
}

std::vector<FlickrPOI> FlickrAPIClient::fetch_pois() const {
    std::vector<FlickrPOI> pois;
    
    std::map<std::string, std::string> params = {
        {"media", "photos"},
        {"has_geo", "1"},
        {"text", config.search_word},
        {"bbox", config.bbox},
        {"extras", "description,geo,tags"},
        {"per_page", "500"}
    };
    
    if (!config.min_date.empty()) params["min_upload_date"] = config.min_date;
    if (!config.max_date.empty()) params["max_upload_date"] = config.max_date;
    
    std::string url = build_flickr_url("flickr.photos.search", params);
    std::string response = make_http_request(url);
    
    if (response.empty()) {
        std::cerr << "Failed to fetch data from Flickr API" << std::endl;
        return pois;
    }
    
    try {
        json j = json::parse(response);
        
        if (j.contains("stat") && j["stat"] == "ok" && j.contains("photos")) {
            auto photos = j["photos"]["photo"];
            
            for (const auto& photo : photos) {
                if (photo.contains("latitude") && photo.contains("longitude")) {
                    FlickrPOI poi;
                    poi.id = photo["id"];
                    poi.latitude = std::stod(photo["latitude"].get<std::string>());
                    poi.longitude = std::stod(photo["longitude"].get<std::string>());
                    poi.title = photo.value("title", "");
                    
                    if (photo.contains("description") && photo["description"].contains("_content")) {
                        poi.description = photo["description"]["_content"];
                    }
                    
                    if (photo.contains("tags")) {
                        std::string tags_str = photo["tags"];
                        std::istringstream iss(tags_str);
                        std::string tag;
                        while (iss >> tag) {
                            poi.tags.push_back(tag);
                        }
                    }
                    
                    pois.push_back(poi);
                }
            }
        }
    } catch (const json::exception& e) {
        std::cerr << "JSON Parse Error: " << e.what() << std::endl;
    }
    
    std::cout << "Fetched " << pois.size() << " POIs from Flickr" << std::endl;
    return pois;
}

void FlickrAPIClient::save_pois_to_file(const std::vector<FlickrPOI>& pois, const std::string& filename) const {
    json j;
    
    for (const auto& poi : pois) {
        json poi_json;
        poi_json["id"] = poi.id;
        poi_json["latitude"] = poi.latitude;
        poi_json["longitude"] = poi.longitude;
        poi_json["title"] = poi.title;
        poi_json["description"] = poi.description;
        poi_json["tags"] = poi.tags;
        
        j[poi.id] = poi_json;
    }
    
    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(4);
        file.close();
        std::cout << "POIs saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to save POIs to " << filename << std::endl;
    }
}

std::vector<FlickrPOI> FlickrAPIClient::load_pois_from_file(const std::string& filename) const {
    std::vector<FlickrPOI> pois;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Cache file not found: " << filename << std::endl;
        return pois;
    }
    
    try {
        json j;
        file >> j;
        
        for (const auto& [id, poi_json] : j.items()) {
            FlickrPOI poi;
            poi.id = poi_json["id"];
            poi.latitude = poi_json["latitude"];
            poi.longitude = poi_json["longitude"];
            poi.title = poi_json.value("title", "");
            poi.description = poi_json.value("description", "");
            
            if (poi_json.contains("tags")) {
                poi.tags = poi_json["tags"];
            }
            
            pois.push_back(poi);
        }
        
        std::cout << "Loaded " << pois.size() << " POIs from cache: " << filename << std::endl;
    } catch (const json::exception& e) {
        std::cerr << "Error reading POI cache: " << e.what() << std::endl;
    }
    
    return pois;
}

// ====================================================================
// FONCTIONS DE CONNEXION DES COMPOSANTES
// ====================================================================

GeoBox connect_isolated_components(GeoBox geo_box) {
    if (!geo_box.is_valid) return geo_box;
    
    std::cout << "\n=== Connexion des composantes isolées ===" << std::endl;
    
    auto components = find_components_simple(geo_box.data);
    
    if (components.size() <= 1) {
        std::cout << "Une seule composante - pas de connexion nécessaire" << std::endl;
        return geo_box;
    }
    
    std::cout << "Composantes trouvées: " << components.size() << std::endl;
    
    // Trouver la plus grande composante
    size_t main_component_idx = 0;
    for (size_t i = 1; i < components.size(); ++i) {
        if (components[i].size() > components[main_component_idx].size()) {
            main_component_idx = i;
        }
    }
    
    std::cout << "Composante principale: " << components[main_component_idx].size() << " nodes" << std::endl;
    
    // Connecter chaque petite composante à la principale
    osmium::object_id_type next_way_id = get_max_way_id(geo_box.data) + 1;
    
    for (size_t i = 0; i < components.size(); ++i) {
        if (i == main_component_idx) continue;
        
        auto [main_node, isolated_node, distance] = find_closest_nodes(
            geo_box.data, components[main_component_idx], components[i]);
            
        std::cout << "Connexion composante " << i << " (distance: " << static_cast<int>(distance) << "m)" << std::endl;
        
        create_connecting_way(geo_box.data, next_way_id++, main_node, isolated_node, distance);
    }
    
    return geo_box;
}

std::vector<std::vector<osmium::object_id_type>> find_components_simple(const MyData& data) {
    std::unordered_set<osmium::object_id_type> visited;
    std::vector<std::vector<osmium::object_id_type>> components;
    
    for (const auto& [node_id, node] : data.nodes) {
        if (!visited.count(node_id)) {
            std::vector<osmium::object_id_type> component;
            bfs_explore(data, node_id, visited, component);
            if (!component.empty()) {
                components.push_back(component);
            }
        }
    }
    
    return components;
}

void bfs_explore(const MyData& data, osmium::object_id_type start,
                std::unordered_set<osmium::object_id_type>& visited,
                std::vector<osmium::object_id_type>& component) {
    
    std::queue<osmium::object_id_type> queue;
    queue.push(start);
    visited.insert(start);
    
    while (!queue.empty()) {
        osmium::object_id_type current = queue.front();
        queue.pop();
        component.push_back(current);
        
        auto node_it = data.nodes.find(current);
        if (node_it == data.nodes.end()) continue;
        
        // Explorer les voisins via les ways incidents
        for (const auto& way_id : node_it->second.incident_ways) {
            auto way_it = data.ways.find(way_id);
            if (way_it == data.ways.end()) continue;
            
            std::vector<osmium::object_id_type> neighbors = {way_it->second.node1_id, way_it->second.node2_id};
            
            for (const auto& neighbor_id : neighbors) {
                if (neighbor_id != current && !visited.count(neighbor_id)) {
                    visited.insert(neighbor_id);
                    queue.push(neighbor_id);
                }
            }
        }
    }
}

osmium::object_id_type get_max_way_id(const MyData& data) {
    osmium::object_id_type max_id = 0;
    for (const auto& [way_id, way] : data.ways) {
        if (way_id > max_id) {
            max_id = way_id;
        }
    }
    return max_id;
}

std::tuple<osmium::object_id_type, osmium::object_id_type, double> 
find_closest_nodes(const MyData& data, 
                  const std::vector<osmium::object_id_type>& comp1,
                  const std::vector<osmium::object_id_type>& comp2) {
    
    double min_distance = std::numeric_limits<double>::max();
    osmium::object_id_type best_node1 = 0;
    osmium::object_id_type best_node2 = 0;
    
    for (const auto& node1_id : comp1) {
        auto node1_it = data.nodes.find(node1_id);
        if (node1_it == data.nodes.end()) continue;
        
        for (const auto& node2_id : comp2) {
            auto node2_it = data.nodes.find(node2_id);
            if (node2_it == data.nodes.end()) continue;
            
            double distance = calculate_haversine_distance(
                node1_it->second.lat, node1_it->second.lon,
                node2_it->second.lat, node2_it->second.lon
            );
            
            if (distance < min_distance) {
                min_distance = distance;
                best_node1 = node1_id;
                best_node2 = node2_id;
            }
        }
    }
    
    return std::make_tuple(best_node1, best_node2, min_distance);
}

void create_connecting_way(MyData& data, osmium::object_id_type way_id, 
                          osmium::object_id_type node1_id, osmium::object_id_type node2_id, 
                          double distance) {
    
    auto node1_it = data.nodes.find(node1_id);
    auto node2_it = data.nodes.find(node2_id);
    
    if (node1_it == data.nodes.end() || node2_it == data.nodes.end()) {
        std::cerr << "Erreur: nodes introuvables pour connexion" << std::endl;
        return;
    }
    
    // Créer le nouveau way
    MyData::Way new_way(way_id);
    new_way.node1_id = node1_id;
    new_way.node2_id = node2_id;
    new_way.points = {node1_it->second, node2_it->second};
    new_way.distance_meters = static_cast<float>(distance);
    new_way.groupe = 0;
    
    // Ajouter le way aux données
    data.ways[way_id] = new_way;
    
    // CONNEXION IMMÉDIATE (cohérent avec MyHandler::way())
    node1_it->second.incident_ways.push_back(way_id);
    node2_it->second.incident_ways.push_back(way_id);
    
    std::cout << "Way de connexion créé: " << way_id 
              << " (" << node1_id << " -> " << node2_id << ")" << std::endl;
}