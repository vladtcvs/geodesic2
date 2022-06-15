project(geodesic2)

set(CMAKE_C_FLAGS "-DBINROOT=\"\\\"${CMAKE_BINARY_DIR}\\\"\"")

file(COPY ${CMAKE_SOURCE_DIR}/src/geodesic.cl DESTINATION ${CMAKE_BINARY_DIR}/)

add_executable(geodesic2 src/geodesic.c src/calc.c src/dispatcher.c src/opencl.c)
target_include_directories(geodesic2 PUBLIC src)
target_link_libraries(geodesic2 OpenCL m pthread)
