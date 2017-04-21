cmake_minimum_required(VERSION 3.7)
set(CUDA_HOST_COMPILER /usr/bin/g++-5)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -Xcompiler -fPIC;
    -std=c++11
    -Xcompiler -Wunused-function
    -gencode arch=compute_53,code=sm_53
    -gencode=arch=compute_53,code=compute_53
    -gencode arch=compute_52,code=sm_52
    -gencode=arch=compute_52,code=compute_52
    -gencode arch=compute_50,code=sm_50
    -gencode=arch=compute_50,code=compute_50
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_30,code=sm_30
    )

FILE(GLOB_RECURSE TEST_SRC tests/*.cu tests/*.cuh tests/*.hpp)

include_directories(include)
include_directories(tests)

cuda_add_executable(
    run_tests
    ${TEST_SRC}
)
