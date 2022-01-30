project(geodesic2)

set(CMAKE_C_FLAGS "-DBINROOT=\"\\\"${CMAKE_BINARY_DIR}\\\"\"")

add_custom_target(copy_kernel cp ${CMAKE_SOURCE_DIR}/src/geodesic.cl ${CMAKE_BINARY_DIR}/geodesic.cl)
add_executable(geodesic2 src/geodesic.c src/calc.c src/dispatcher.c src/opencl.c)
target_include_directories(geodesic2 PUBLIC src)
add_dependencies(geodesic2 copy_kernel)
target_link_libraries(geodesic2 OpenCL m pthread)
