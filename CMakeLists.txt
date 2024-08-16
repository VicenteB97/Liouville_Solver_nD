cmake_minimum_required(VERSION 3.22)

# set(CMAKE_HOST_SYSTEM Windows)

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

project(${PROJECT_NAME} VERSION 4.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        ON)
set(CMAKE_CUDA_ARCHITECTURES    60 70 80)

find_package(OpenMP)

if(CMAKE_HOST_SYSTEM MATCHES Windows)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

message(STATUS "Building in ${CMAKE_HOST_SYSTEM_NAME}")

# NEEDED FOR CUDA PROJECTS
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(configured)
add_subdirectory(src)
add_subdirectory(app)
