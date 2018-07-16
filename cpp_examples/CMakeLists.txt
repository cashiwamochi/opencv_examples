cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MACOSX_RPATH 1) # This is needed in macOS

project(OPENCV_EXAMPLES)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(OpenCV_DIR ~/.local/OpenCV/share/OpenCV)
find_package(OpenCV REQUIRED)
IF (OpenCV_FOUND)
  message([message] OpenCV_VER:${OpenCV_VERSION})
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIBRARY_DIRS})
  add_definitions(${OpenCV_DEFINITIONS})
ENDIF()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

add_subdirectory (
  Calibration
)

add_subdirectory (
  Triangulation
)

add_subdirectory (
  Version
)
