cmake_minimum_required(VERSION 3.24)

# Cargo owns both the native PyO3 build and cuda-oxide device compilation.
# Configuring documentation builds does not discover a C/C++ or CUDA compiler.
project(rapids_singlecell_cuda LANGUAGES NONE)
option(RSC_BUILD_EXTENSIONS "Build the Rust CUDA extension" ON)

if (RSC_BUILD_EXTENSIONS)
  if (DEFINED RSC_BUILD_RUST AND NOT RSC_BUILD_RUST)
    message(FATAL_ERROR "Rust is the only GPU backend. Use RSC_BUILD_EXTENSIONS=OFF for documentation builds.")
  endif()
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter)
  include(cmake/RustBackend.cmake)
  add_rust_cuda_module()
else()
  message(STATUS "RSC_BUILD_EXTENSIONS=OFF: skipping the native GPU backend")
endif()
