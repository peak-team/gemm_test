cmake_minimum_required(VERSION 3.24)

project(gemm_test LANGUAGES C CXX)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the build type." FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

cmake_host_system_information(RESULT N_CPU QUERY NUMBER_OF_PHYSICAL_CORES)
message(STATUS "N_CPU = ${N_CPU}")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(CMAKE_BUILD_TYPE MATCHES "Release")
    add_compile_options("SHELL:-Wall -O3 -ffunction-sections -fdata-sections")
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        message(STATUS "Linux detected")
        if(NOT CMAKE_C_COMPILER_ID MATCHES "NVHPC")
            add_link_options("SHELL:-static-libgcc -Wl,-z,noexecstack,--gc-sections")
        else()
            add_link_options("SHELL:-Wl,-z,noexecstack,--gc-sections")
            message(STATUS "Skipping -static-libgcc for NVHPC compiler")
        endif()
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(STATUS "macOS detected")
        add_link_options("SHELL:-Wl,-dead_strip")
    endif()
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_options("SHELL:-Wall -O0 -g")
endif()

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(CUDAToolkit 11.4 REQUIRED)
enable_language(CUDA)

message(STATUS "MPI launcher: ${MPIEXEC_EXECUTABLE}")
message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")

# Select architectures supported by the active toolkit. Users may override this,
# for example with -DCMAKE_CUDA_ARCHITECTURES=100 on Blackwell systems.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        set(_default_cuda_architectures 80 86 89 90 100)
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
        set(_default_cuda_architectures 80 86 89 90)
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
        set(_default_cuda_architectures 80 86 89)
    else()
        set(_default_cuda_architectures 80 86)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_default_cuda_architectures}" CACHE STRING
        "CUDA architectures to build" FORCE)
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

function(configure_cuda_rpath target)
    # HPC SDK can install cuBLAS outside the core CUDA directory. Embed the
    # imported target directories so the binaries find libcublas at runtime.
    set(_cuda_runtime_rpath)
    foreach(_dependency CUDA::cublas CUDA::cublasLt CUDA::cudart)
        foreach(_property IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE
                          IMPORTED_LOCATION_RELWITHDEBINFO IMPORTED_LOCATION_DEBUG)
            get_target_property(_location ${_dependency} ${_property})
            if(_location AND NOT _location MATCHES "-NOTFOUND$")
                get_filename_component(_directory "${_location}" DIRECTORY)
                list(APPEND _cuda_runtime_rpath "${_directory}")
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _cuda_runtime_rpath)

    if(_cuda_runtime_rpath)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${_cuda_runtime_rpath}"
            INSTALL_RPATH_USE_LINK_PATH ON)
    endif()
endfunction()

add_library(gemm_test_common STATIC
    src/gemm_test_common.cu
    src/gemm_test_operations.cu)

target_include_directories(gemm_test_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(gemm_test_common
    PUBLIC
        HAVE_CUDA
        HAVE_MPI)

target_link_libraries(gemm_test_common
    PUBLIC
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cudart
        CUDA::cuda_driver
        MPI::MPI_CXX
        Threads::Threads)

foreach(test gemm_test_mpi gemm_test_mpi_sm)
    add_executable(${test}
        src/${test}.cu)
    target_link_libraries(${test}
        PRIVATE
            gemm_test_common)
    configure_cuda_rpath(${test})
endforeach()
