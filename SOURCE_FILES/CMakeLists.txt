cmake_minimum_required(VERSION 3.22)

# IMPORTANT: HERE WE DEFINE THE VARIABLES TO BE USED THROUGHOUT THE CMAKE FILE
set(PROJCT_NAME     LIOUVILLE_EQ_SOLVER)
set(EXECUTABLE_NAME Simulation)
set(LIB_NAME        Solver_Lib)

project(${PROJCT_NAME} VERSION 2.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        ON)

find_package(OpenMP)

option(CMAKE_BUILD_TYPE    "RELEASE")
option(COMPILE_EXECUTABLE  "Include executable compiling" ON)

add_subdirectory(Configured)

add_library(${LIB_NAME} STATIC FULL_SIMULATION.cu)

target_include_directories(${LIB_NAME} PUBLIC
    "./"
    "${CMAKE_BINARY_DIR}/configured_files/include")
   
# COMMENT THE FOLLOWING LINE WHEN COMPILING IN LINUX
# include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}") # absolutely necessary for windows (because it's retarded')!

add_executable(${EXECUTABLE_NAME} 00_MAIN.cu)

target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp -lgomp>)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${LIB_NAME} OpenMP::OpenMP_CXX)

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_ARCHITECTURES 75)