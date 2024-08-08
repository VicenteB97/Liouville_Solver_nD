set(LIBRARY_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}")

set(LIBRARY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Particle.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Domain.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/waveletTransform/waveletTransform.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/ivpSolver.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/probabilityDistributions/probability.cu"
)

set(LIBRARY_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Domain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Particle.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/waveletTransform/waveletTransform.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/ivpSolver.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/probabilityDistributions/probability.cuh"
)

add_library(${LIB_NAME} STATIC ${LIBRARY_HEADERS} ${LIBRARY_SOURCES})

target_include_directories(
    ${LIB_NAME} PUBLIC 
    ${LIBRARY_INCLUDES}
    "${CMAKE_BINARY_DIR}/configured_files/include"
    "${LIOUVILLE_EQ_SOLVER_SOURCE_DIR}/../boost"
    "${LIOUVILLE_EQ_SOLVER_SOURCE_DIR}/../boost_dir"
    "${LIOUVILLE_EQ_SOLVER_SOURCE_DIR}/ext/boost"
)

if(CMAKE_HOST_SYSTEM MATCHES Windows)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -openmp -lgomp -gencode arch=compute_80,code=sm_80>)
elseif(CMAKE_HOST_SYSTEM MATCHES Linux)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp -lgomp -gencode arch=compute_80,code=sm_80>)
endif()