add_executable(${EXECUTABLE_NAME} main.cu)

if(CMAKE_HOST_SYSTEM MATCHES Windows)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -openmp -lgomp -gencode arch=compute_80,code=sm_80>)
elseif(CMAKE_HOST_SYSTEM MATCHES Linux)
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp -lgomp -gencode arch=compute_80,code=sm_80>)
endif()

target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${LIB_NAME} OpenMP::OpenMP_CXX)