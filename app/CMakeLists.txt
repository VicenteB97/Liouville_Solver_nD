if(BUILD_CUDA)
	add_executable(${EXECUTABLE_NAME} main.cu)
else()
	add_executable(${EXECUTABLE_NAME} main.cpp)
endif()

target_include_directories(${EXECUTABLE_NAME} PRIVATE ${LIBRARY_INCLUDES})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${LIB_NAME} OpenMP::OpenMP_CXX)
