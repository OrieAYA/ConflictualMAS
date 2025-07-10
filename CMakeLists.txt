cmake_minimum_required(VERSION 3.10)
project(ConflictualMAS CXX)

add_compile_definitions(NOMINMAX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "C:/Users/screp/OneDrive/Bureau/Algorithms/ConflictualMAS/cmake")

set(LIBOSMIUM_DIR "C:/libs/libosmium")
if(NOT EXISTS "${LIBOSMIUM_DIR}/include/osmium")
    message(FATAL_ERROR "Libosmium non trouvée à ${LIBOSMIUM_DIR}. Veuillez ajuster LIBOSMIUM_DIR ou la cloner.")
endif()
include_directories("${LIBOSMIUM_DIR}/include")

#set(MAPNIK_DIR "C:/libs/mapnik")
#if(NOT EXISTS "${MAPNIK_DIR}/include/osmium")
#    message(FATAL_ERROR "Mapnik non trouvée à ${MAPNIK_DIR}. Veuillez ajuster MAPNIK_DIR ou la cloner.")
#endif()
#include_directories("${MAPNIK_DIR}/include")

find_package(Boost 1.55 COMPONENTS system iostreams REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)
find_package(EXPAT REQUIRED)
find_package(Protozero REQUIRED)

include_directories(SYSTEM ${PROTOZERO_INCLUDE_DIR})
include_directories("C:/Users/screp/OneDrive/Bureau/Algorithms/ConflictualMAS/src")

#find_path(OSMIUM_INCLUDE_DIRS "osmium/version.hpp")
#target_include_directories(main PRIVATE ${OSMIUM_INCLUDE_DIRS})

find_package(Mapnik CONFIG REQUIRED)

set(SOURCES
    src/main.cpp
    src/Box.cpp
    src/MapRenderer.cpp
)
add_executable(main ${SOURCES})

target_link_libraries(main PRIVATE
    Boost::system
    Boost::iostreams
    #Libosmium::osmium
    mapnik::mapnik
    mapnik::json
    mapnik::wkt
    ${ZLIB_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${EXPAT_LIBRARIES}
)