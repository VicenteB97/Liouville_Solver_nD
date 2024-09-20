# Still have to include the library that will take care of the tests (Google test for example, or catch2 at https://github.com/catchorg/Catch2)
set(LIBRARY_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/../src/")

# FOR WAVELET TRANSFORM 
if(BUILD_CUDA)
add_executable(WVLT_test test_waveletTransform.cu)
else()
add_executable(WVLT_test test_waveletTransform.cpp)
endif()

target_link_libraries(WVLT_test PRIVATE ${LIB_NAME}) 

# FOR ADAPTIVE MESH REFINEMENT
if(BUILD_CUDA)
add_executable(AMR_test test_adaptiveMeshRefinement.cu)
else()
add_executable(AMR_test test_adaptiveMeshRefinement.cpp)
endif()

target_link_libraries(AMR_test PRIVATE ${LIB_NAME}) 