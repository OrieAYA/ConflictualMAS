#ifndef HASHES_HPP
#define HASHES_HPP

#include "Box.hpp"

struct PairHash {
    size_t operator()(const std::pair<osmium::object_id_type, osmium::object_id_type>& p) const {
        return std::hash<osmium::object_id_type>{}(p.first) ^ std::hash<osmium::object_id_type>{}(p.second);
    }
};

#endif // HASHES_HPP