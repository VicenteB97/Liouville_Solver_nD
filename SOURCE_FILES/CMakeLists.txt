# IF NO CXX COMPILER IS AVAILABLE, TYPE export CXX=/usr/bin/g++

## IN ORDER TO EXECUTE FROM THE TERMINAL (UBUNTU), ALL YOU HAVE TO DO IS ACCESS THE BUILD FOLDER:
# /mnt/c/Users/Vicentin/source/repos/CMAKE_LIOUVILLE/build

cmake_minimum_required(VERSION 3.22)

# IMPORTANT: HERE WE DEFINE THE VARIABLES TO BE USED THROUGHOUT THE CMAKE FILE
set(PROJCT_NAME     LIOUVILLE_EQ_SOLVER)
set(EXECUTABLE_NAME Simulation)
set(LIB_NAME        Solver_Lib)

project(${PROJCT_NAME} VERSION 2.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        ON)

find_package(OpenMP)

option(CMAKE_BUILD_TYPE    RELEASE)
option(COMPILE_EXECUTABLE "Include executable compiling" ON)

add_subdirectory(Configured)

add_library(${LIB_NAME} STATIC 
FULL_SIMULATION.cu
Adapt_Mesh.cuh
Case_definition.cuh
Classes.cuh
Dynamics.cuh
Impulse_Transformations.cuh
Mat_Ops.cuh
Parameter_distributions.cuh
Simulation_parameters.cuh)

target_include_directories(${LIB_NAME} PUBLIC
    "./"
    "${CMAKE_BINARY_DIR}/configured_files/include")


add_executable(${EXECUTABLE_NAME} 00_MAIN.cu)

target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp -lgomp>)
target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${LIB_NAME} OpenMP::OpenMP_CXX)

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_ARCHITECTURES 75)