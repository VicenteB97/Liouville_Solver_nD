cmake_minimum_required(VERSION 3.22)

set(CMAKE_HOST_SYSTEM Windows)

# IMPORTANT: HERE WE DEFINE THE VARIABLES TO BE USED THROUGHOUT THE CMAKE FILE
set(PROJCT_NAME     LIOUVILLE_EQ_SOLVER)
set(EXECUTABLE_NAME Simulation)
set(LIB_NAME        Solver_Lib)

project(${PROJCT_NAME} VERSION 4.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        ON)
set(CMAKE_CUDA_ARCHITECTURES    60 75 80)

find_package(OpenMP)

option(CMAKE_BUILD_TYPE    "RELEASE")
option(COMPILE_EXECUTABLE  "Include executable compiling" ON)

add_library(${LIB_NAME} STATIC IVP_SOLVER.cu)

target_include_directories(${LIB_NAME} PUBLIC
    "./"
    "${CMAKE_BINARY_DIR}/configured_files/include")
   
# COMMENT THE FOLLOWING LINE WHEN COMPILING IN LINUX
if(CMAKE_HOST_SYSTEM MATCHES Windows)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}") # absolutely necessary for windows (because it's retarded')!
endif()

add_executable(${EXECUTABLE_NAME} main.cu)

if(CMAKE_HOST_SYSTEM MATCHES Windows)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -openmp -lgomp -gencode arch=compute_80,code=sm_80>)
elseif(CMAKE_HOST_SYSTEM MATCHES Linux)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp -lgomp -gencode arch=compute_80,code=sm_80>)
endif()

target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${LIB_NAME} OpenMP::OpenMP_CXX)
