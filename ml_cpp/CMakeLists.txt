cmake_minimum_required(VERSION 3.14)
project(echosurf_ml VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

    # Check for AVX2 support
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        message(STATUS "AVX2 support enabled")
    else()
        check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE4)
        if(COMPILER_SUPPORTS_SSE4)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
            message(STATUS "SSE4.1 support enabled")
        endif()
    endif()
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /arch:AVX2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif()

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(ECHOSURF_ML_SOURCES
    src/tensor.cpp
    src/layers.cpp
    src/model.cpp
    src/model_loader.cpp
)

# Static library
add_library(echosurf_ml_static STATIC ${ECHOSURF_ML_SOURCES})
target_include_directories(echosurf_ml_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Shared library
add_library(echosurf_ml SHARED ${ECHOSURF_ML_SOURCES})
target_include_directories(echosurf_ml PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 QUIET)

    if(NOT pybind11_FOUND)
        # Download pybind11 if not found
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(echosurf_ml_cpp python/echosurf_ml.cpp)
    target_link_libraries(echosurf_ml_cpp PRIVATE echosurf_ml_static)

    # Install Python module to site-packages
    install(TARGETS echosurf_ml_cpp
        LIBRARY DESTINATION ${Python3_SITELIB}
    )
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_executable(benchmark_inference benchmarks/benchmark_inference.cpp)
    target_link_libraries(benchmark_inference PRIVATE echosurf_ml_static)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE echosurf_ml_static)
    add_test(NAME TensorTests COMMAND test_tensor)

    add_executable(test_layers tests/test_layers.cpp)
    target_link_libraries(test_layers PRIVATE echosurf_ml_static)
    add_test(NAME LayerTests COMMAND test_layers)

    add_executable(test_model tests/test_model.cpp)
    target_link_libraries(test_model PRIVATE echosurf_ml_static)
    add_test(NAME ModelTests COMMAND test_model)
endif()

# Installation
install(TARGETS echosurf_ml echosurf_ml_static
    EXPORT echosurf_ml_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT echosurf_ml_targets
    FILE echosurf_ml_targets.cmake
    NAMESPACE echosurf::
    DESTINATION lib/cmake/echosurf_ml
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    echosurf_ml_config_version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/echosurf_ml_config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/echosurf_ml_config.cmake
    INSTALL_DESTINATION lib/cmake/echosurf_ml
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/echosurf_ml_config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/echosurf_ml_config_version.cmake
    DESTINATION lib/cmake/echosurf_ml
)
