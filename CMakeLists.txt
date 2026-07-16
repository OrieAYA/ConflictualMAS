cmake_minimum_required(VERSION 3.10)
project(ConflictualMAS LANGUAGES CXX)

add_compile_definitions(NOMINMAX)

set(PROJECT_PATH "C:/")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH "${PROJECT_PATH}ConflictualMAS/cmake")

#set(LIBOSMIUM_DIR "C:/libs/libosmium")
#if(NOT EXISTS "${LIBOSMIUM_DIR}/include/osmium")
#    message(FATAL_ERROR "Libosmium non trouvﾃｩe ﾃ ${LIBOSMIUM_DIR}. Veuillez ajuster LIBOSMIUM_DIR ou la cloner.")
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
    src/DMASforPD/Agents/DeliveryAgent.cpp
    src/DMASforPD/Agents/Manager.cpp
    src/DMASforPD/Agents/TaskAgent.cpp
    src/DMASforPD/Algorithms/DbVNS.cpp
    src/DMASforPD/Algorithms/TAM.cpp
    src/DMASforPD/Algorithms/TDAStar.cpp
    src/DMASforPD/Policy/BidPolicy.cpp
    src/DMASforPD/Policy/Hybrid.cpp
    src/DMASforPD/Policy/IPPO.cpp
    src/DMASforPD/Policy/MAPPERInspired.cpp
    src/DMASforPD/Policy/MAPPO.cpp
    src/DMASforPD/Policy/MovementPolicy.cpp
    src/DMASforPD/Policy/PolicyKit.cpp
    src/DMASforPD/Prediction/Lsm.cpp
    src/DMASforPD/Structures/AgentSolution.cpp
    src/DMASforPD/Structures/ObjectiveCache.cpp
    src/DMASforPD/Structures/OperableEnvironment.cpp
    src/DMASforPD/Structures/TimePaths.cpp
    src/Environment/GeoBox/Box.cpp
    src/Environment/GeoBox/GeoBoxManager.cpp
    src/Environment/GeoBox/GraphSearch.cpp
    src/Environment/Render/MapRenderer.cpp
    src/Environment/Simulation/CongestionMap.cpp
    src/Environment/Simulation/GhostTrafficController.cpp
    src/Environment/Simulation/MovingGhosts.cpp
    src/Environment/Simulation/TemporalChainList.cpp
    src/Environment/Simulation/TemporalGraph.cpp
    src/Environment/Structure/Episode.cpp
    src/Environment/Structure/EpisodeManager.cpp
    src/Environment/Structure/EventStream.cpp
    src/Legacy/Agent/Agent.cpp
    src/Legacy/Agent/GlobalSolutionConstructor.cpp
    src/Legacy/Agent/MetaAgent.cpp
    src/Legacy/Common/Memory.cpp
    src/Legacy/Common/Pathfinding.cpp
    src/Legacy/MHProcs/ACO.cpp
    src/Legacy/MHProcs/GRASP.cpp
    src/Legacy/MHProcs/PSO.cpp
    src/Legacy/MHProcs/VNS.cpp
    src/SoTA/Planning/ALNS.cpp
    src/SoTA/ScoringLike/RMCA.cpp
    src/SoTA/SolverFramework.cpp
    src/SoTA/Standalone/CA.cpp
    src/SoTA/Standalone/HAPC.cpp
    src/SoTA/TaskAgentLike/TP.cpp
    src/Tests/CongestionOnlyTest.cpp
    src/Tests/EndToEndTests.cpp
    src/Tests/GeoBoxConnectivityTest.cpp
    src/Tests/LegacyTests.cpp
    src/Tests/MechanicsTests.cpp
    src/Tests/ModuleTests.cpp
    src/Tests/PDPTests.cpp
    src/Tests/PlanningComparisonTest.cpp
    src/Tests/RegressionTests.cpp
    src/Tests/StructureTests.cpp
    src/Tests/TamMcTest.cpp
    src/Tests/TemporalChainListTest.cpp
    src/Tests/TrainingSmokeTest.cpp
    src/TrainingEvaluation/Run/Metrics.cpp
    src/TrainingEvaluation/Run/Runner.cpp
    src/TrainingEvaluation/Run/RunnerAllocation.cpp
    src/TrainingEvaluation/Run/RunnerMovement.cpp
    src/TrainingEvaluation/Run/RunnerPrediction.cpp
    src/TrainingEvaluation/Run/Trainer.cpp
    src/TrainingEvaluation/StructuresParam/CityConfig.cpp
    src/TrainingEvaluation/StructuresParam/ScenarioConfig.cpp
    src/main.cpp
    src/utility.cpp
)

add_executable(main ${SOURCES})

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/src
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