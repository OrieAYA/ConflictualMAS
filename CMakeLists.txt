cmake_minimum_required(VERSION 3.10)
project(ConflictualMAS LANGUAGES CXX)

add_compile_definitions(NOMINMAX)

set(PROJECT_PATH "C:/")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH "${PROJECT_PATH}ConflictualMAS/cmake")

#set(LIBOSMIUM_DIR "C:/libs/libosmium")
#if(NOT EXISTS "${LIBOSMIUM_DIR}/include/osmium")
#    message(FATAL_ERROR "Libosmium non trouvée à ${LIBOSMIUM_DIR}. Veuillez ajuster LIBOSMIUM_DIR ou la cloner.")
#endif()
#include_directories("${LIBOSMIUM_DIR}/include")

find_package(Boost CONFIG REQUIRED COMPONENTS system iostreams)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)
find_package(EXPAT REQUIRED)
find_package(CURL REQUIRED)
#find_package(libosmium CONFIG REQUIRED)
find_path(OSMIUM_INCLUDE_DIRS "osmium/version.hpp")
find_package(Mapnik CONFIG REQUIRED)

include_directories(SYSTEM ${PROTOZERO_INCLUDE_DIR})

set(SOURCES
    src/main.cpp
    src/utility.cpp
    src/Comparisons/DbVNS_vs_LKH/LKHSolver.cpp
    src/Comparisons/DbVNS_vs_LKH/IVNSSolver.cpp
    src/Comparisons/DbVNS_vs_LKH/ComparisonTest.cpp
    src/DMASforPD/Policy/ObjectiveDMPolicy.cpp
    src/DMASforPD/Policy/IPPOPolicy.cpp
    src/DMASforPD/Policy/MapperPolicy.cpp
    src/DMASforPD/Policy/HybridPolicy.cpp
    src/Training/CityConfig.cpp
    src/Training/EpisodeConfig.cpp
    src/Training/EpisodeGenerator.cpp
    src/Training/EpisodeRunner.cpp
    src/Training/TrainingLogger.cpp
    src/Training/MultiCityTrainer.cpp
    src/Training/TrainingSmokeTest.cpp
    src/Legacy/Tests/LegacyTests.cpp
    src/DMASforPD/Tests/PDPTests.cpp
    src/AmazonDataset/AmazonTest.cpp
    src/Legacy/Agent/Agent.cpp
    src/Legacy/Agent/MetaAgent.cpp
    src/Legacy/Agent/GlobalSolutionConstructor.cpp
    src/Environment/GeoBox/Box.cpp
    src/Environment/GeoBox/GeoBoxManager.cpp
    src/Environment/Render/MapRenderer.cpp
    src/Legacy/Common/Pathfinding.cpp
    src/Legacy/Common/Memory.cpp
    src/Legacy/MHProcs/ACO.cpp
    src/Legacy/MHProcs/GRASP.cpp
    src/Legacy/MHProcs/VNS.cpp
    src/Legacy/MHProcs/PSO.cpp
    src/Environment/OverpassAPI/OverpassAPI.cpp
    src/Environment/Congestion/CongestionMap.cpp
    src/DMASforPD/GlobalMemory/ObjectiveCache.cpp
    src/DMASforPD/GlobalMemory/GlobalMemory.cpp
    src/DMASforPD/Utility/AgentSolution.cpp
    src/DMASforPD/DeliveryAgent/DeliveryAgent.cpp
    src/DMASforPD/DeliveryAgent/OperableEnvironment.cpp
    src/DMASforPD/DeliveryAgent/LocalSolutionAgent.cpp
    src/DMASforPD/TaskAgent/TaskAgent.cpp
    src/DMASforPD/Utility/TimedPath.cpp
    src/DMASforPD/TaskAgent/TaskAllocationModule.cpp
)

add_executable(main ${SOURCES})

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/Training
)

target_link_libraries(main PRIVATE
    Boost::system
    Boost::iostreams
    ${OSMIUM_INCLUDE_DIRS}
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