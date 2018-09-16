cmake_minimum_required(VERSION 3.5)
project(PointUtil)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ~/lib/cmake)
message(STATUS ${CMAKE_MODULE_PATH})
#set(CMAKE_VERBOSE_MAKEFILE ON)


set(BOOST_PYTHON_INCLUDE_DIR /home/liuyuan/software/boost/include)      # path contains boost/python/object_core
set(BOOST_PYTHON_LIBRARY_DIR /home/liuyuan/software/boost/lib)          # path contains boost/python/libboost_numpy*

set(FLANN_INCLUDE_DIR /usr/local/include)   # path contains flann/flann.h
set(FLANN_LIBRARY_DIR /usr/local/lib)       # path contains libflann.so*

# !!! note the python here should be consistent with the python used for building boost-python
set(PYTHON_INCLUDE_DIRS /home/liuyuan/anaconda3/include/python3.6m)   # path contains Python.h
set(PYTHON_LIBRARIES /home/liuyuan/anaconda3/lib/libpython3.6m.so)



message("BOOST_PYTHON LIBRARIES ARE " ${BOOST_PYTHON_INCLUDE_DIR})
message("BOOST_PYTHON INCLUDE ARE " ${BOOST_PYTHON_LIBRARY_DIR})
include_directories(${BOOST_PYTHON_INCLUDE_DIR})
link_directories(${BOOST_PYTHON_LIBRARY_DIR})

#find_package(FLANN REQUIRED)
message("FLANN LIBRARIES ARE " ${FLANN_LIBRARY_DIR})
message("FLANN INCLUDE ARE " ${FLANN_INCLUDE_DIR})
include_directories(${FLANN_INCLUDE_DIR})
link_directories(${FLANN_LIBRARY_DIR})

#find_package(PythonLibs REQUIRED)
message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
message("PYTHON INCLUDE ARE " ${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

add_definitions(-O2)

find_package(CUDA REQUIRED)

set(SOURCE_FILES Wrapper.cpp Downsample.cpp Downsample.cu SampleBlock.cpp SampleBlock.cu
        ComputeCovar.cu VoxelPooling.cpp Interpolate.cu Interpolate.cpp)
cuda_add_library(PointUtil SHARED ${SOURCE_FILES})

# !!! change the 36 to your version
target_link_libraries(PointUtil
        -lboost_numpy36
        -lboost_python36
        -lflann
        -lflann_cuda
        ${PYTHON_LIBRARIES}
        )