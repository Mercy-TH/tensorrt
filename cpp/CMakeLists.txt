cmake_minimum_required(VERSION 3.0)
project(tensorrt)
add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)


if(UNIX)
    message("current platform: Linux ")
    # cuda directory
    include_directories(/usr/local/cuda/include/)
    link_directories(/usr/local/cuda/lib64/)

    # tensorrt
    include_directories(/usr/local/TensorRT-8.2.3.0/include/)
    link_directories(/usr/local/TensorRT-8.2.3.0/lib/)
elseif(WIN32)
    message("current platform: Windows ")
    set(OpenCV_DIR "D:\\opencv\\mingw_build\\install")
    # cuda directory
    include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\include)
    link_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\lib)

    # tensorrt
    include_directories(C:\\Program Files\\TensorRT_origin\\TensorRT-8.2.5.1\\include)
    link_directories(C:\\Program Files\\TensorRT_origin\\TensorRT-8.2.5.1\\lib)
else()
    message("current platform: unkonw ")
endif()


# opencv-4.6.0
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV库: ${OpenCV_LIBS}")
message(STATUS "OpenCV版本: ${OpenCV_VERSION}")
message(STATUS "OpenCV路径： ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_DIR:　${OpenCV_DIR}")




add_executable(main main.cpp)
target_link_libraries(main
        ${OpenCV_LIBS})

add_definitions(-O2 -pthread)
