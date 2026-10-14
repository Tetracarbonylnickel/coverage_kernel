cmake_minimum_required(VERSION 3.18)
# C is enabled alongside Fortran only because conda-forge's HDF5 CMake config
# pulls in FindThreads, which requires C or CXX to be enabled in the project
# even though this library itself is pure Fortran.
project(coverage_kernel C Fortran)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

find_package(HDF5 REQUIRED COMPONENTS Fortran)

# Module order matters: each depends only on the ones listed before it.
add_library(coverage_kernel STATIC
  src/MOD_CoverageKernel_Types.f90
  src/MOD_CoverageKernel_BSpline.f90
  src/MOD_CoverageKernel_Interp1D.f90
  src/MOD_CoverageKernel_Classifier.f90
  src/MOD_CoverageKernel_MLP.f90
  src/MOD_CoverageKernel_Grid.f90
  src/MOD_CoverageKernel_IO.f90
  src/MOD_CoverageKernel.f90
)
target_include_directories(coverage_kernel PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY} ${HDF5_Fortran_INCLUDE_DIRS})
target_link_libraries(coverage_kernel PUBLIC ${HDF5_Fortran_LIBRARIES})
target_compile_options(coverage_kernel PRIVATE -Wall -std=f2008)

enable_testing()
add_executable(test_coverage_kernel test/test_coverage_kernel.f90)
target_link_libraries(test_coverage_kernel PRIVATE coverage_kernel)
target_include_directories(test_coverage_kernel PRIVATE ${CMAKE_Fortran_MODULE_DIRECTORY})
add_test(NAME coverage_kernel_cross_validation COMMAND test_coverage_kernel
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
