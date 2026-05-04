#ifndef OVERPASS_API_HPP
#define OVERPASS_API_HPP

#include <string>
#include <curl/curl.h>

class OverpassAPI {
public:
    OverpassAPI(const std::string& endpoint = "https://overpass-api.de/api/interpreter");
    ~OverpassAPI();
    
    std::string query(const std::string& overpass_query);
    
    bool queryToFile(const std::string& overpass_query, const std::string& output_file);
    
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