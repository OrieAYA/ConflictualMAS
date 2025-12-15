// ===== OverpassAPI.hpp =====
#ifndef OVERPASS_API_HPP
#define OVERPASS_API_HPP

#include <string>
#include <curl/curl.h>

class OverpassAPI {
public:
    OverpassAPI(const std::string& endpoint = "https://overpass-api.de/api/interpreter");
    ~OverpassAPI();
    
    // Récupérer les données brutes (JSON)
    std::string query(const std::string& overpass_query);
    
    // Sauvegarder directement dans un fichier
    bool queryToFile(const std::string& overpass_query, const std::string& output_file);
    
    // Requêtes pré-construites
    std::string getBoundingBoxQuery(double south, double west, double north, double east);
    
    void setTimeout(int timeout_seconds);
    
private:
    std::string endpoint_url;
    CURL* curl;
    int timeout;
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
    std::string buildQuery(const std::string& query_body);
};

#endif // OVERPASS_API_HPP

// ===== OverpassAPI.cpp =====
#include "OverpassAPI.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

OverpassAPI::OverpassAPI(const std::string& endpoint) 
    : endpoint_url(endpoint), timeout(180) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }
}

OverpassAPI::~OverpassAPI() {
    if (curl) {
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}

size_t OverpassAPI::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

void OverpassAPI::setTimeout(int timeout_seconds) {
    timeout = timeout_seconds;
}

std::string OverpassAPI::buildQuery(const std::string& query_body) {
    std::string full_query = "[out:json][timeout:" + std::to_string(timeout) + "];\n";
    full_query += query_body;
    full_query += "\nout body;\n>;out skel qt;";
    return full_query;
}

std::string OverpassAPI::query(const std::string& overpass_query) {
    if (!curl) {
        throw std::runtime_error("CURL not initialized");
    }
    
    std::string response_string;
    std::string full_query = buildQuery(overpass_query);
    
    // URL encode
    char* encoded_query = curl_easy_escape(curl, full_query.c_str(), full_query.length());
    if (!encoded_query) {
        throw std::runtime_error("Failed to encode query");
    }
    
    std::string post_data = "data=" + std::string(encoded_query);
    curl_free(encoded_query);
    
    // Configuration CURL
    curl_easy_setopt(curl, CURLOPT_URL, endpoint_url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout + 10L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    // Requête
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
    }
    
    // Vérifier le code HTTP
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    
    if (http_code != 200) {
        throw std::runtime_error("HTTP request failed with code: " + std::to_string(http_code));
    }
    
    return response_string;
}

bool OverpassAPI::queryToFile(const std::string& overpass_query, const std::string& output_file) {
    try {
        std::cout << "Fetching data from Overpass API..." << std::endl;
        std::string result = query(overpass_query);
        
        std::ofstream out_file(output_file);
        if (!out_file.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return false;
        }
        
        out_file << result;
        out_file.close();
        
        std::cout << "Data saved to: " << output_file << std::endl;
        std::cout << "Size: " << result.length() << " bytes" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

std::string OverpassAPI::getBoundingBoxQuery(double south, double west, double north, double east) {
    std::string bbox = std::to_string(south) + "," + std::to_string(west) + "," +
                       std::to_string(north) + "," + std::to_string(east);
    
    std::string query = "(\n";
    query += "  node(" + bbox + ");\n";
    query += "  way(" + bbox + ");\n";
    query += "  relation(" + bbox + ");\n";
    query += ");";
    
    return query;
}