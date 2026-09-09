# Every native module shares one PyO3 extension and one cuda-oxide PTX image.
function(add_rust_cuda_module)
  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The native Rust backend supports Linux only")
  endif()
  # Cargo only needs a toolkit root. FindCUDAToolkit creates C/C++ link
  # targets and requires FindThreads; it would reintroduce compiler discovery.
  find_path(RSC_CUDA_TOOLKIT_ROOT NAMES include/cuda.h
      HINTS "${CUDAToolkit_ROOT}" "$ENV{CUDA_TOOLKIT_PATH}" "$ENV{CUDA_HOME}" "$ENV{CUDA_PATH}"
      PATHS /usr/local/cuda)
  if (NOT RSC_CUDA_TOOLKIT_ROOT)
    find_program(RSC_NVCC_EXECUTABLE nvcc)
    if (RSC_NVCC_EXECUTABLE)
      get_filename_component(rsc_nvcc_real "${RSC_NVCC_EXECUTABLE}" REALPATH)
      get_filename_component(rsc_cuda_bin "${rsc_nvcc_real}" DIRECTORY)
      get_filename_component(RSC_CUDA_TOOLKIT_ROOT "${rsc_cuda_bin}" DIRECTORY)
    endif()
  endif()
  if (NOT EXISTS "${RSC_CUDA_TOOLKIT_ROOT}/include/cuda.h")
    message(FATAL_ERROR "CUDA Toolkit 13.0+ headers are required. Set CUDAToolkit_ROOT or CUDA_TOOLKIT_PATH.")
  endif()
  file(STRINGS "${RSC_CUDA_TOOLKIT_ROOT}/include/cuda.h" rsc_cuda_version_line
      REGEX "^#define CUDA_VERSION [0-9]+")
  string(REGEX MATCH "[0-9]+" rsc_cuda_version "${rsc_cuda_version_line}")
  if (NOT rsc_cuda_version OR rsc_cuda_version LESS 13000)
    message(FATAL_ERROR "The native Rust backend requires CUDA Toolkit 13.0 or newer")
  endif()

  find_program(RSC_CARGO_EXECUTABLE cargo REQUIRED)
  find_program(RSC_CARGO_OXIDE_EXECUTABLE cargo-oxide REQUIRED)
  set(RSC_RUST_CUDA_ARCH "sm_75" CACHE STRING "PTX architecture for the Rust CUDA kernels")
  if (NOT RSC_RUST_CUDA_ARCH MATCHES "^sm_[0-9]+[af]?$")
    message(FATAL_ERROR "RSC_RUST_CUDA_ARCH must be a single CUDA architecture such as sm_75")
  endif()

  set(rust_source_dir "${PROJECT_SOURCE_DIR}/rust")
  set(rust_build_dir "${CMAKE_CURRENT_BINARY_DIR}/rust")
  set(ptx_file "${rust_build_dir}/ptx/rsc_kernels.ptx")
  set(host_library "${rust_build_dir}/python-target/release/lib_rust_cuda.so")
  set(module_file "${rust_build_dir}/_rust_cuda.abi3.so")
  file(GLOB stub_files CONFIGURE_DEPENDS "${rust_source_dir}/python/*.pyi")
  set(editable_dir "${PROJECT_SOURCE_DIR}/src/rapids_singlecell/_cuda")
  set(cuda_toolkit_dir "${RSC_CUDA_TOOLKIT_ROOT}")

  # Keep compiler caches separate: cuda-oxide uses a custom rustc backend for
  # device code, while the PyO3 extension is an ordinary native Rust cdylib.
  file(GLOB_RECURSE kernel_sources CONFIGURE_DEPENDS "${rust_source_dir}/kernels/src/*.rs")
  file(GLOB_RECURSE host_sources CONFIGURE_DEPENDS "${rust_source_dir}/python/src/*.rs")
  set(common_dependencies
      "${rust_source_dir}/Cargo.toml"
      "${rust_source_dir}/Cargo.lock"
      "${rust_source_dir}/rust-toolchain.toml")
  # Let Cargo share GNU Make's parallelism limit when CMake supports it.
  set(cargo_job_server_options)
  if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    set(cargo_job_server_options JOB_SERVER_AWARE TRUE)
  endif()

  add_custom_command(
      OUTPUT "${ptx_file}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${rust_build_dir}/ptx"
      # Cargo does not track the externally emitted PTX. Force this small owner
      # crate to compile when CMake needs PTX, including after a CMake clean or
      # an accidentally deleted PTX file, while retaining dependency caches.
      COMMAND "${RSC_CARGO_EXECUTABLE}" clean
          --manifest-path "${rust_source_dir}/kernels/Cargo.toml"
          --target-dir "${rust_build_dir}/kernels-target"
          --package rsc-kernels --release --locked
      COMMAND ${CMAKE_COMMAND} -E env
          "CUDA_TOOLKIT_PATH=${cuda_toolkit_dir}"
          "CUDA_OXIDE_PTX_DIR=${rust_build_dir}/ptx"
          "${RSC_CARGO_EXECUTABLE}" oxide build
          --arch "${RSC_RUST_CUDA_ARCH}"
          --cargo-target-dir "${rust_build_dir}/kernels-target"
          --device-codegen-crate rsc_kernels
          -- --manifest-path "${rust_source_dir}/kernels/Cargo.toml" --lib --release --locked
      DEPENDS ${kernel_sources} ${common_dependencies}
          "${rust_source_dir}/kernels/Cargo.toml"
      WORKING_DIRECTORY "${rust_source_dir}"
      COMMENT "Compiling Rust CUDA kernels to ${RSC_RUST_CUDA_ARCH} PTX with cuda-oxide"
      VERBATIM USES_TERMINAL ${cargo_job_server_options})

  add_custom_command(
      OUTPUT "${module_file}"
      BYPRODUCTS "${host_library}"
      COMMAND ${CMAKE_COMMAND} -E env
          "CUDA_TOOLKIT_PATH=${cuda_toolkit_dir}"
          "RSC_CUDA_OXIDE_PTX=${ptx_file}"
          "PYO3_PYTHON=${Python_EXECUTABLE}"
          "PYO3_BUILD_EXTENSION_MODULE=1"
          "${RSC_CARGO_EXECUTABLE}" build
          --manifest-path "${rust_source_dir}/python/Cargo.toml"
          --target-dir "${rust_build_dir}/python-target" --release --locked
      COMMAND ${CMAKE_COMMAND} -E copy "${host_library}" "${module_file}.tmp"
      COMMAND ${CMAKE_COMMAND} -E rename "${module_file}.tmp" "${module_file}"
      DEPENDS "${ptx_file}" ${host_sources} ${common_dependencies}
          "${rust_source_dir}/python/Cargo.toml" "${rust_source_dir}/python/build.rs"
          "${rust_source_dir}/kernels/exports.txt"
      WORKING_DIRECTORY "${rust_source_dir}"
      COMMENT "Building the shared PyO3 CUDA extension with embedded PTX"
      VERBATIM USES_TERMINAL ${cargo_job_server_options})

  # Always refresh the editable copy, including after switching backend in the
  # same build directory. Stage and rename shared libraries to preserve any
  # mappings held by running Python processes. The Python router preserves the
  # existing import paths for submodules exported by the shared Rust extension.
  add_custom_target(_rust_cuda ALL
      COMMAND ${CMAKE_COMMAND} -E copy "${module_file}" "${editable_dir}/_rust_cuda.abi3.so.tmp"
      COMMAND ${CMAKE_COMMAND} -E rename "${editable_dir}/_rust_cuda.abi3.so.tmp" "${editable_dir}/_rust_cuda.abi3.so"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${stub_files} "${editable_dir}"
      COMMAND ${CMAKE_COMMAND} -E touch "${editable_dir}/py.typed"
      DEPENDS "${module_file}" ${stub_files}
      VERBATIM)
  install(FILES "${module_file}" ${stub_files} DESTINATION rapids_singlecell/_cuda)
  install(FILES "${editable_dir}/py.typed" DESTINATION rapids_singlecell/_cuda)
  message(STATUS "GPU backend: Rust/cuda-oxide/PyO3 (${RSC_RUST_CUDA_ARCH})")
endfunction()
