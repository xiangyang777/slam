cmake_minimum_required(VERSION 2.8)
project(ch5)

set(CMAKE_CXX_FLAGS "-std=c++11")


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
)


add_executable(work optical_flow.cpp)

target_link_libraries(work ${OpenCV_LIBS} ${EIGEN3_LIBS} ${Pangolin_LIBRARIES} ${Sophus_INCLUDE_DIRS})

