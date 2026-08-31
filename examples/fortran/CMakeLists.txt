cmake_minimum_required(VERSION 3.18)
project(coverage_kernel_example C Fortran)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../fortran/src)

find_package(HDF5 REQUIRED COMPONENTS Fortran)

add_library(coverage_kernel STATIC
  ${KERNEL_SRC}/MOD_CoverageKernel_Types.f90
  ${KERNEL_SRC}/MOD_CoverageKernel_BSpline.f90
  ${KERNEL_SRC}/MOD_CoverageKernel_Interp1D.f90
  ${KERNEL_SRC}/MOD_CoverageKernel_Classifier.f90
  ${KERNEL_SRC}/MOD_CoverageKernel_Grid.f90
  ${KERNEL_SRC}/MOD_CoverageKernel_IO.f90
  ${KERNEL_SRC}/MOD_CoverageKernel.f90
)
target_include_directories(coverage_kernel PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY} ${HDF5_Fortran_INCLUDE_DIRS})
target_link_libraries(coverage_kernel PUBLIC ${HDF5_Fortran_LIBRARIES})
target_compile_options(coverage_kernel PRIVATE -Wall -std=f2008)

add_executable(monte_carlo_walk monte_carlo_walk.f90)
target_link_libraries(monte_carlo_walk PRIVATE coverage_kernel)
target_include_directories(monte_carlo_walk PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})
