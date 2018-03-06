cmake_minimum_required(VERSION 3.5)
project(PointUtil)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ~/lib/cmake)
message(STATUS ${CMAKE_MODULE_PATH})
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(BOOST_PYTHON REQUIRED)
message("BOOST_PYTHON LIBRARIES ARE " ${BOOST_PYTHON_INCLUDE_DIR})
include_directories(${BOOST_PYTHON_INCLUDE_DIR})
link_directories(${BOOST_PYTHON_LIBRARY_DIR})

find_package(FLANN REQUIRED)
message("FLANN LIBRARIES ARE " ${FLANN_LIBRARY_DIR})
message("FLANN INCLUDE ARE " ${FLANN_INCLUDE_DIR})
include_directories(${FLANN_INCLUDE_DIR})
link_directories(${FLANN_LIBRARY_DIR})

find_package(PythonLibs REQUIRED)
message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
message("PYTHON INCLUDE ARE " ${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

add_definitions(-O2)

find_package(CUDA REQUIRED)

set(SOURCE_FILES Wrapper.cpp Downsample.cpp Downsample.cu SampleBlock.cpp SampleBlock.cu ComputeCovar.cu VoxelPooling.cpp)
cuda_add_library(PointUtil SHARED ${SOURCE_FILES})
#add_library(PointUtil SHARED ${SOURCE_FILES})

#CUDA_ADD_CUBLAS_TO_TARGET(PointUtil)

#add_executable(HashTesting HashTest.cpp)

target_link_libraries(PointUtil
        boost_numpy
        boost_python
        -lflann
        -lflann_cuda
        ${PYTHON_LIBRARIES}
        )