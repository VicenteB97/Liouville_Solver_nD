set(LIBRARY_SOURCES
    "ivpSolver.cu"
)

add_library(${LIB_NAME} STATIC ${LIBRARY_SOURCES})

target_include_directories(
    ${LIB_NAME} PUBLIC 
    "./include"
    "${CMAKE_BINARY_DIR}/configured_files/include"
)