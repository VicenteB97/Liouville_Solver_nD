set(LIBRARY_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}")

if(BUILD_CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(LIBRARY_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Particle.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Domain.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/waveletTransform/waveletTransform.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/ivpSolver.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/probabilityDistributions/Probability.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/adaptiveMeshRefinement/Adapt_Mesh.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/Sim_data.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/Simulation_parameters.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/interpolation/Interpolation.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/integrators/Integrator.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/integrators/Impulse_transformations.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/PointSearch.cu"
    )
else()
    set(LIBRARY_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Particle.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Domain.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/waveletTransform/waveletTransform.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/ivpSolver.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/probabilityDistributions/Probability.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/adaptiveMeshRefinement/Adapt_Mesh.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/Sim_data.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/Simulation_parameters.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/interpolation/Interpolation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/integrators/Integrator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/integrators/Impulse_transformations.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/PointSearch.cpp"
    )
endif()

set(LIBRARY_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Domain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mesh/Particle.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/waveletTransform/waveletTransform.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/ivpSolver.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/probabilityDistributions/Probability.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/adaptiveMeshRefinement/Adapt_Mesh.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/Sim_data.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ivpSolver/Simulation_parameters.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interpolation/Interpolation.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integrators/Integrator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integrators/Impulse_transformations.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/PointSearch.hpp"
)

add_library(${LIB_NAME} STATIC ${LIBRARY_HEADERS} ${LIBRARY_SOURCES})

# target_precompile_headers(${LIB_NAME} PUBLIC headers.hpp)
# set(CMAKE_PCH_INSTANTIATE_TEMPLATES ON) ONLY WHEN USING CLANG COMPILER

target_include_directories(
    ${LIB_NAME} PUBLIC 
    ${LIBRARY_INCLUDES}
    "${CMAKE_BINARY_DIR}/configured_files/include"
    "${LIOUVILLE_EQ_SOLVER_SOURCE_DIR}/../boost"
    "${LIOUVILLE_EQ_SOLVER_SOURCE_DIR}/../boost_dir"
    "${LIOUVILLE_EQ_SOLVER_SOURCE_DIR}/ext/boost"
)

# Here, we must also check if we're compiling with CUDA or just C++
if(CMAKE_HOST_SYSTEM MATCHES Windows)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -openmp -lgomp -gencode arch=compute_80,code=sm_80>)
elseif(CMAKE_HOST_SYSTEM MATCHES Linux)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp -lgomp -gencode arch=compute_80,code=sm_80>)
endif()
