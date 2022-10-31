cmake_minimum_required(VERSION 3.12)
project(ORB_SLAM2)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17) # check c++11
add_definitions(-DCOMPILEDWITHC11)
if(MSVC)
   add_compile_options(/bigobj /MP /utf-8)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glog REQUIRED)
find_package(tomlplusplus CONFIG REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(${PROJECT_NAME} STATIC
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/Thirdparty
        ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        )

target_link_libraries(${PROJECT_NAME} PUBLIC
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/DBoW2.lib
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/g2o.lib
        glog::glog
        tomlplusplus::tomlplusplus
        fmt::fmt
        )

###################
# Build examples  #
###################

add_executable(mono_shenzhen Examples/Monocular/mono_shenzhen.cc)
target_link_libraries(mono_shenzhen ${PROJECT_NAME})

