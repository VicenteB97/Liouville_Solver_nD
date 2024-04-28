
add_executable(${EXECUTABLE_NAME} main.cu)

target_include_directories(${EXECUTABLE_NAME} PRIVATE ${LIBRARY_INCLUDES})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${LIB_NAME} OpenMP::OpenMP_CXX)
