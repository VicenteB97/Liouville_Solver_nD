set(LIBRARY_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(LIBRARY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/Adapt_Mesh.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/Domain.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/Integrator.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/Interpolation.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/PointSearch.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/Probability.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/Simulation_parameters.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/Sim_data.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/Impulse_transformations.cu"
)

set(LIBRARY_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Adapt_Mesh.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Domain.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Integrator.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Interpolation.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ivpSolver.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/PointSearch.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Probability.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Simulation_parameters.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Sim_data.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Impulse_transformations.cuh"
)

add_library(${LIB_NAME} STATIC ${LIBRARY_HEADERS} ${LIBRARY_SOURCES})

target_include_directories(
    ${LIB_NAME} PUBLIC 
    ${LIBRARY_INCLUDES}
    "${CMAKE_BINARY_DIR}/configured_files/include"
    "${LIOUVILLE_EQ_SOLVER_SOURCE_DIR}/../boost"
    "${LIOUVILLE_EQ_SOLVER_SOURCE_DIR}/../boost_dir"
)

if(CMAKE_HOST_SYSTEM MATCHES Windows)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -openmp -lgomp -gencode arch=compute_80,code=sm_80>)
elseif(CMAKE_HOST_SYSTEM MATCHES Linux)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp -lgomp -gencode arch=compute_80,code=sm_80>)
endif()