cmake_minimum_required(VERSION 3.18)
project(green_context LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture to sm_100a (Blackwell)
set(CMAKE_CUDA_ARCHITECTURES "100a")

# Set build type to RelWithDebInfo
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Find CUDA Toolkit for Driver API
find_package(CUDAToolkit REQUIRED)

# Create cuda_cache directory in build directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/cuda_cache)

# Add CUDA flags for verbose output and keeping intermediate files
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -v --keep --keep-dir=${CMAKE_BINARY_DIR}/cuda_cache")

# Automatically discover and build all .cu files in current directory
file(GLOB CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# Create an executable for each .cu file
foreach(CU_FILE ${CU_FILES})
    # Get the filename without extension
    get_filename_component(FILE_NAME ${CU_FILE} NAME_WE)
    
    # Create executable
    add_executable(${FILE_NAME} ${CU_FILE})

    # Link CUDA Driver API
    target_link_libraries(${FILE_NAME} PRIVATE CUDA::cuda_driver)
    
    message(STATUS "Building executable: ${FILE_NAME}")
endforeach()

# Print summary
list(LENGTH CU_FILES NUM_FILES)
message(STATUS "Total .cu files found: ${NUM_FILES}")
