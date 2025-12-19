cmake_minimum_required(VERSION 3.10)
project(ConflictualMAS LANGUAGES CXX)

add_compile_definitions(NOMINMAX)

set(PROJECT_PATH "C:/Users/screp/OneDrive/Bureau/Algorithms/")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH "${PROJECT_PATH}ConflictualMAS/cmake")

list(APPEND CMAKE_MODULE_PATH "C:/Users/screp/OneDrive/Bureau/Algorithms/ConflictualMAS/cmake")

set(LIBOSMIUM_DIR "C:/libs/libosmium")
if(NOT EXISTS "${LIBOSMIUM_DIR}/include/osmium")
    message(FATAL_ERROR "Libosmium non trouvée à ${LIBOSMIUM_DIR}. Veuillez ajuster LIBOSMIUM_DIR ou la cloner.")
endif()
include_directories("${LIBOSMIUM_DIR}/include")

find_package(Boost 1.55 COMPONENTS system iostreams REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)
find_package(EXPAT REQUIRED)
find_package(CURL REQUIRED)
find_package(Protozero REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
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
    src/Agent/Agent.cpp
    src/Agent/MetaAgent.cpp
)
add_executable(main ${SOURCES})

target_link_libraries(main PRIVATE
    Boost::system
    Boost::iostreams
    mapnik::mapnik
    mapnik::json
    mapnik::wkt
    CURL::libcurl
    ${ZLIB_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${EXPAT_LIBRARIES}
)

#Be sure to create a new build file and perform commands in it
#Make : cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/libs/vcpkg/scripts/buildsystems/vcpkg.cmake
#Build / Compile : cmake --build . 

#[[
In Dev Command

Installation of vcpkg : 

cd C:/
mkdir libs
cd libs
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

Installation of Boost :

cd C:/libs/vcpkg
vcpkg install boost
(for real long so take time)

Or just
vcpkg install boost-system
vcpkg install boost-iostreams

Installation of CURL :

vcpkg install curl

Installation of Libosmium :

vcpkg install libosmium (for protozero install without problems)
cd C:/libs
git clone https://github.com/osmcode/libosmium

Integrate vcpkg to CMake Globally :

vcpkg integrate install

]]