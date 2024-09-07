cmake_minimum_required(VERSION 3.18)
project(rasterizer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Set Python paths manually based on your conda environment
set(ENV_PATH /home/jizong/Workspace/dConstruct/gaussian-splatting/submodules/diff-gaussian-rasterization-w-pose-depth/venv)
set(CMAKE_PREFIX_PATH ${ENV_PATH})
set(PYTHON_INCLUDE_DIR "${ENV_PATH}/include/python3.10")
set(PYTHON_LIBRARY "${ENV_PATH}/lib/libpython3.10.so")
set(Torch_DIR "${ENV_PATH}/lib/python3.10/site-packages/torch/share/cmake/Torch")
set(TORCH_INCLUDE_DIRS
        "${ENV_PATH}/include/python3.10"
        "${ENV_PATH}/lib/python3.10/site-packages/torch/include"
        "${ENV_PATH}/lib/python3.10/site-packages/torch/include/torch/csrc/api/include"
)


set(CUDA_SOURCES
        cuda_rasterizer/backward.cu
        cuda_rasterizer/forward.cu
        cuda_rasterizer/rasterizer_impl.cu
        rasterize_points.cu
)

include_directories("/usr/local/cuda/include")
#include_directories(${PYTHON_INCLUDE_DIR})

add_executable(rasterizer main.cpp ${CUDA_SOURCES})
add_library(CudaRasterizer
        cuda_rasterizer/backward.h
        cuda_rasterizer/backward.cu
        cuda_rasterizer/forward.h
        cuda_rasterizer/forward.cu
        cuda_rasterizer/auxiliary.h
        cuda_rasterizer/rasterizer_impl.cu
        cuda_rasterizer/rasterizer_impl.h
        cuda_rasterizer/rasterizer.h
        rasterize_points.cu
        rasterize_points.h
        ext.cpp
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "75;86;89")


target_include_directories(CudaRasterizer PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rasterizer PUBLIC ${TORCH_INCLUDE_DIRS})

set_target_properties(rasterizer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(rasterizer PRIVATE TORCH_USE_CUDA_DSA)

target_include_directories(rasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_compile_options(rasterizer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)

# Find and link necessary libraries
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Link libraries, including Python and PyTorch
target_link_libraries(rasterizer "${TORCH_LIBRARIES}" ${PYTHON_LIBRARY} ${CUDA_LIBRARIES})
target_link_libraries(CudaRasterizer "${TORCH_LIBRARIES}" ${PYTHON_LIBRARY} ${CUDA_LIBRARIES})
