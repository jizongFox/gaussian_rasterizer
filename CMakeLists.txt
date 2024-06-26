cmake_minimum_required(VERSION 3.18)
project(rasterizer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)


set(CUDA_SOURCES
    cuda_rasterizer/backward.cu
    cuda_rasterizer/forward.cu
    cuda_rasterizer/rasterizer_impl.cu
    rasterize_points.cu
)
include_directories("/usr/local/cuda/include")

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

target_include_directories(CudaRasterizer PUBLIC "/opt/miniconda3/envs/gaussian_splatting/include/python3.10")
target_include_directories(CudaRasterizer PUBLIC "/opt/miniconda3/envs/gaussian_splatting/lib/python3.10/site-packages/torch/include")
target_include_directories(CudaRasterizer PUBLIC "/opt/miniconda3/envs/gaussian_splatting/lib/python3.10/site-packages/torch/include/torch/csrc/api/include")

target_include_directories(rasterizer PUBLIC "/opt/miniconda3/envs/gaussian_splatting/include/python3.10")
target_include_directories(rasterizer PUBLIC "/opt/miniconda3/envs/gaussian_splatting/lib/python3.10/site-packages/torch/include")
target_include_directories(rasterizer PUBLIC "/opt/miniconda3/envs/gaussian_splatting/lib/python3.10/site-packages/torch/include/torch/csrc/api/include")

set_target_properties(rasterizer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(rasterizer PRIVATE TORCH_USE_CUDA_DSA)

target_include_directories(rasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_compile_options(rasterizer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
