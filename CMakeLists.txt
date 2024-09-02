cmake_minimum_required(VERSION 3.22)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# IMPORTANT: HERE WE DEFINE THE VARIABLES TO BE USED THROUGHOUT THE CMAKE FILE
set(PROJECT_NAME    LIOUVILLE_EQ_SOLVER)
set(EXECUTABLE_NAME Simulation)
set(LIB_NAME        Solver_Lib)
set(MY_SOURCE_DIR  ${CMAKE_SOURCE_DIR})

project(${PROJECT_NAME} VERSION 4.0.0 LANGUAGES CXX)

option(BUILD_CUDA "Build the CUDA version" ON)
option(BUILD_SYCL "Build the SYCL version" OFF)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        ON)

if(BUILD_CUDA)
    enable_language(CUDA)
    add_compile_options(-D USECUDA) # this creates a preprocessor macro USECUDA. Just check if it is defined in the code
    set(CMAKE_CUDA_ARCHITECTURES    60 70 80)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    message(STATUS "Building CUDA project")
endif(BUILD_CUDA)

find_package(OpenMP REQUIRED)

message(STATUS "Building in ${CMAKE_HOST_SYSTEM_NAME}")

add_subdirectory(configured)
add_subdirectory(src)
add_subdirectory(app)
