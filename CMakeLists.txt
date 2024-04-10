cmake_minimum_required(VERSION 3.22)

set(CMAKE_HOST_SYSTEM Windows)

# IMPORTANT: HERE WE DEFINE THE VARIABLES TO BE USED THROUGHOUT THE CMAKE FILE
set(PROJCT_NAME     LIOUVILLE_EQ_SOLVER)
set(EXECUTABLE_NAME Simulation)
set(LIB_NAME        Solver_Lib)

project(${PROJCT_NAME} VERSION 3.3.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        ON)
set(CMAKE_CUDA_ARCHITECTURES    60 75 80)

find_package(OpenMP)

option(CMAKE_BUILD_TYPE    "RELEASE")
option(COMPILE_EXECUTABLE  "Include executable compiling" ON)
   
# COMMENT THE FOLLOWING LINE WHEN COMPILING IN LINUX
if(CMAKE_HOST_SYSTEM MATCHES Windows)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}") # absolutely necessary for windows (because it's retarded')!
endif()

option(COMPILE_EXECUTABLE "Turn on executable" OFF)

add_subdirectory(configured)
add_subdirectory(src)
if(COMPILE_EXECUTABLE)
    add_subdirectory(app)
endif()
