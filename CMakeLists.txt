cmake_minimum_required(VERSION 3.12)

find_package(OpenCL REQUIRED)

add_library(BitonicSort INTERFACE BitonicSort.hpp)
target_link_libraries(BitonicSort INTERFACE OpenCL::OpenCL)
target_compile_features(BitonicSort INTERFACE cxx_std_20)

add_executable(sorter sorter.cpp)
target_link_libraries(sorter BitonicSort)

add_executable(BenchBitonicSort BenchBitonicSort.cpp)
target_link_libraries(BenchBitonicSort BitonicSort tbb)
target_compile_definitions(BenchBitonicSort PRIVATE MEASURE)

find_package(GTest)
if (TARGET GTest::gtest_main)
    add_executable(TestBitonicSort TestBitonicSort.cpp)
    target_link_libraries(TestBitonicSort BitonicSort GTest::gtest_main)
endif()
