cmake_minimum_required(VERSION 3.10)
project(ConflictualMAS LANGUAGES CXX)

add_compile_definitions(NOMINMAX)

set(PROJECT_PATH "C:/Users/S52504ZrN/Desktop/Algorithm/")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH "${PROJECT_PATH}ConflictualMAS/cmake")

find_package(Boost CONFIG REQUIRED COMPONENTS system iostreams)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)
find_package(EXPAT REQUIRED)
find_package(CURL REQUIRED)
find_package(libosmium CONFIG REQUIRED)
find_package(Mapnik CONFIG REQUIRED)

include_directories(SYSTEM ${PROTOZERO_INCLUDE_DIR})
include_directories("${PROJECT_PATH}ConflictualMAS/src")

set(SOURCES
    src/main.cpp
    src/Box.cpp
    src/MapRenderer.cpp
    src/Pathfinding.cpp
    src/GeoBoxManager.cpp
    src/utility.cpp
    src/MHProcs/ACO.cpp
    src/MHProcs/GRASP.cpp
    src/MHProcs/VNS.cpp
    src/MHProcs/PSO.cpp
    src/OverpassAPI/OverpassAPI.cpp
)

add_executable(main ${SOURCES})

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(main PRIVATE
    Boost::system
    Boost::iostreams
    libosmium::libosmium
    mapnik::mapnik
    mapnik::json
    mapnik::wkt
    CURL::libcurl
    ZLIB::ZLIB
    BZip2::BZip2
    EXPAT::EXPAT
)

#Make : cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/libs/vcpkg/scripts/buildsystems/vcpkg.cmake
#Build / Compile : cmake --build . 