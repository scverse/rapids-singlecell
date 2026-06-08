cmake_minimum_required(VERSION 3.24)

project(rapids_singlecell_cuda LANGUAGES CXX)

# Option to disable building compiled extensions (for docs/RTD)
option(RSC_BUILD_EXTENSIONS "Build CUDA/C++ extensions" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (RSC_BUILD_EXTENSIONS)
  enable_language(CUDA)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
  find_package(nanobind CONFIG REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  set(RSC_RMM_HINTS)
  set(RSC_RAPIDS_CMAKE_PREFIXES)
  set(RSC_CCCL_HINTS)
  set(RSC_RAPIDS_LOGGER_HINTS)
  set(RSC_NVTX3_HINTS)
  macro(_rsc_collect_rapids_python_prefix _rsc_prefix)
    if (NOT "${_rsc_prefix}" STREQUAL "")
      file(GLOB _rsc_rmm_dirs "${_rsc_prefix}/lib/python*/site-packages/librmm/lib64/cmake/rmm")
      file(GLOB _rsc_rapids_prefixes
        "${_rsc_prefix}/lib/python*/site-packages/librmm/lib64"
        "${_rsc_prefix}/lib/python*/site-packages/librmm/lib64/rapids"
        "${_rsc_prefix}/lib/python*/site-packages/rapids_logger/lib64"
        "${_rsc_prefix}/lib/python*/site-packages/nvidia/cu*/lib"
      )
      file(GLOB _rsc_cccl_dirs
        "${_rsc_prefix}/lib/python*/site-packages/librmm/lib64/rapids/cmake/cccl"
        "${_rsc_prefix}/lib/python*/site-packages/nvidia/cu*/lib/cmake/cccl"
      )
      file(GLOB _rsc_rapids_logger_dirs "${_rsc_prefix}/lib/python*/site-packages/rapids_logger/lib64/cmake/rapids_logger")
      file(GLOB _rsc_nvtx3_dirs "${_rsc_prefix}/lib/python*/site-packages/librmm/lib64/cmake/nvtx3")
      list(APPEND RSC_RMM_HINTS ${_rsc_rmm_dirs})
      list(APPEND RSC_RAPIDS_CMAKE_PREFIXES ${_rsc_rapids_prefixes})
      list(APPEND RSC_CCCL_HINTS ${_rsc_cccl_dirs})
      list(APPEND RSC_RAPIDS_LOGGER_HINTS ${_rsc_rapids_logger_dirs})
      list(APPEND RSC_NVTX3_HINTS ${_rsc_nvtx3_dirs})
    endif()
  endmacro()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, pathlib; spec = importlib.util.find_spec('librmm'); print(pathlib.Path(spec.origin).parent / 'lib64' / 'cmake' / 'rmm' if spec else '')"
    OUTPUT_VARIABLE RSC_PYTHON_RMM_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if (RSC_PYTHON_RMM_DIR AND EXISTS "${RSC_PYTHON_RMM_DIR}/rmm-config.cmake")
    list(APPEND RSC_RMM_HINTS "${RSC_PYTHON_RMM_DIR}")
  endif()
  # Wheel builds install librmm/rapids_logger into the isolated build env and
  # write build/.librmm_dir from CIBW_BEFORE_BUILD.  publish.yml also symlinks
  # those shared libraries into /usr/local/lib so auditwheel can see and exclude
  # them instead of bundling RAPIDS runtime libraries into the wheel.
  if(DEFINED ENV{RSC_LIBRMM_DIR} AND EXISTS "$ENV{RSC_LIBRMM_DIR}/lib64/cmake/rmm/rmm-config.cmake")
    set(_rsc_librmm_marker "$ENV{RSC_LIBRMM_DIR}")
  elseif(EXISTS "${CMAKE_SOURCE_DIR}/build/.librmm_dir")
    file(READ "${CMAKE_SOURCE_DIR}/build/.librmm_dir" _rsc_librmm_marker)
    string(STRIP "${_rsc_librmm_marker}" _rsc_librmm_marker)
  else()
    set(_rsc_librmm_marker "")
  endif()
  if(NOT "${_rsc_librmm_marker}" STREQUAL "" AND EXISTS "${_rsc_librmm_marker}/lib64/cmake/rmm/rmm-config.cmake")
    file(GLOB _rsc_marker_rmm_dirs "${_rsc_librmm_marker}/lib64/cmake/rmm")
    file(GLOB _rsc_marker_rapids_prefixes
      "${_rsc_librmm_marker}/lib64"
      "${_rsc_librmm_marker}/lib64/rapids"
      "${_rsc_librmm_marker}/../rapids_logger/lib64"
    )
    file(GLOB _rsc_marker_cccl_dirs
      "${_rsc_librmm_marker}/lib64/rapids/cmake/cccl"
    )
    file(GLOB _rsc_marker_rapids_logger_dirs "${_rsc_librmm_marker}/../rapids_logger/lib64/cmake/rapids_logger")
    file(GLOB _rsc_marker_nvtx3_dirs "${_rsc_librmm_marker}/lib64/cmake/nvtx3")
    list(APPEND RSC_RMM_HINTS ${_rsc_marker_rmm_dirs})
    list(APPEND RSC_RAPIDS_CMAKE_PREFIXES ${_rsc_marker_rapids_prefixes})
    list(APPEND RSC_CCCL_HINTS ${_rsc_marker_cccl_dirs})
    list(APPEND RSC_RAPIDS_LOGGER_HINTS ${_rsc_marker_rapids_logger_dirs})
    list(APPEND RSC_NVTX3_HINTS ${_rsc_marker_nvtx3_dirs})
  endif()
  foreach(_rsc_python_prefix IN ITEMS "${Python_ROOT_DIR}" "${Python3_ROOT_DIR}")
    _rsc_collect_rapids_python_prefix("${_rsc_python_prefix}")
  endforeach()
  foreach(_rsc_env_prefix IN ITEMS "$ENV{CONDA_PREFIX}" "$ENV{VIRTUAL_ENV}")
    _rsc_collect_rapids_python_prefix("${_rsc_env_prefix}")
  endforeach()
  string(REPLACE ":" ";" _rsc_path_entries "$ENV{PATH}")
  foreach(_rsc_path_entry IN LISTS _rsc_path_entries)
    get_filename_component(_rsc_path_prefix "${_rsc_path_entry}/.." ABSOLUTE)
    _rsc_collect_rapids_python_prefix("${_rsc_path_prefix}")
  endforeach()
  if (RSC_RAPIDS_CMAKE_PREFIXES)
    list(APPEND CMAKE_PREFIX_PATH ${RSC_RAPIDS_CMAKE_PREFIXES})
    if (RSC_CCCL_HINTS)
      list(GET RSC_CCCL_HINTS 0 _rsc_cccl_dir)
      set(CCCL_DIR "${_rsc_cccl_dir}" CACHE PATH "Path to CCCL package config" FORCE)
    endif()
    if (RSC_RAPIDS_LOGGER_HINTS)
      list(GET RSC_RAPIDS_LOGGER_HINTS 0 _rsc_rapids_logger_dir)
      set(rapids_logger_DIR "${_rsc_rapids_logger_dir}" CACHE PATH "Path to rapids_logger package config" FORCE)
    endif()
    if (RSC_NVTX3_HINTS)
      list(GET RSC_NVTX3_HINTS 0 _rsc_nvtx3_dir)
      set(nvtx3_DIR "${_rsc_nvtx3_dir}" CACHE PATH "Path to nvtx3 package config" FORCE)
    endif()
  endif()
  if (RSC_RMM_HINTS)
    find_package(rmm CONFIG REQUIRED HINTS ${RSC_RMM_HINTS})
  else()
    find_package(rmm CONFIG REQUIRED)
  endif()

  # CCCL 3.3.0 (shipped by RAPIDS 26.04) declares the
  # cudaDevAttrHostNumaMemoryPoolsSupported device-attribute specialization under
  # `_CCCL_CTK_AT_LEAST(12, 6)`, but the CUDA runtime only added that enum in 12.9.
  # So compiling the RMM/CCCL-including TUs (the Wilcoxon scratch allocator) against
  # a CUDA 12.6-12.8 toolkit fails with a cryptic
  # `error: the global scope has no "cudaDevAttrHostNumaMemoryPoolsSupported"`.
  # CCCL fixed the guard to `_CCCL_CTK_AT_LEAST(12, 9)` after 3.3.0 (cccl PR #7838),
  # so RAPIDS >= 26.06 (CCCL > 3.3.0) closes the gap -- only flag the buggy CCCL.
  # Fail fast with an actionable message. Prebuilt wheels are unaffected: they are
  # built on CUDA 12.2 (below the guard), so the enum is never referenced.
  set(_rsc_cccl_buggy_numa_guard TRUE)
  if (DEFINED CCCL_VERSION AND CCCL_VERSION VERSION_GREATER 3.3.0)
    set(_rsc_cccl_buggy_numa_guard FALSE)
  endif()
  if (NOT RSC_SKIP_CUDA_VERSION_CHECK
      AND _rsc_cccl_buggy_numa_guard
      AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.6
      AND CUDAToolkit_VERSION VERSION_LESS 12.9)
    message(FATAL_ERROR
      "Cannot build rapids_singlecell from source with CUDA ${CUDAToolkit_VERSION} against "
      "CCCL ${CCCL_VERSION} (RAPIDS 26.04): it references cudaDevAttrHostNumaMemoryPoolsSupported, "
      "which the CUDA 12.6-12.8 toolkit does not define (NVIDIA added it in 12.9). "
      "Use CUDA >= 12.9 (or <= 12.5), upgrade to RAPIDS >= 26.06 (CCCL > 3.3.0 fixes the guard), "
      "or install the prebuilt wheel (pip install rapids-singlecell-cu12). "
      "If your toolkit does define this enum, override with -DRSC_SKIP_CUDA_VERSION_CHECK=ON.")
  endif()

  message(STATUS "Using RMM for CUDA extension scratch allocations")
  message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "RSC_BUILD_EXTENSIONS=OFF -> skipping compiled extensions for docs")
endif()

# Helper to declare a nanobind CUDA module uniformly
function(add_nb_cuda_module target src)
  if (RSC_BUILD_EXTENSIONS)
    nanobind_add_module(${target} STABLE_ABI LTO
        ${src}
    )
    target_link_libraries(${target} PRIVATE CUDA::cudart)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    install(TARGETS ${target} LIBRARY DESTINATION rapids_singlecell/_cuda)
    # Generate type stubs at install time (for wheel installs)
    nanobind_add_stub(${target}_stub
        MODULE ${target}
        OUTPUT rapids_singlecell/_cuda/${target}.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:${target}>
        DEPENDS ${target}
        INSTALL_TIME
        MARKER_FILE rapids_singlecell/_cuda/py.typed
    )
    # Generate type stubs at build time (for editable installs)
    nanobind_add_stub(${target}_stub_dev
        MODULE ${target}
        OUTPUT ${target}.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:${target}>
        DEPENDS ${target}
    )
    # Copy built module + stub into source tree for editable installs
    add_custom_command(TARGET ${target}_stub_dev POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/${target}.pyi
            ${PROJECT_SOURCE_DIR}/src/rapids_singlecell/_cuda/${target}.pyi
        COMMAND ${CMAKE_COMMAND} -E touch
            ${PROJECT_SOURCE_DIR}/src/rapids_singlecell/_cuda/py.typed
    )
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:${target}>
            ${PROJECT_SOURCE_DIR}/src/rapids_singlecell/_cuda/$<TARGET_FILE_NAME:${target}>
    )
  endif()
endfunction()

if (RSC_BUILD_EXTENSIONS)
  # CUDA modules
  add_nb_cuda_module(_mean_var_cuda     src/rapids_singlecell/_cuda/mean_var/mean_var.cu)
  add_nb_cuda_module(_sparse2dense_cuda src/rapids_singlecell/_cuda/sparse2dense/sparse2dense.cu)
  add_nb_cuda_module(_scale_cuda        src/rapids_singlecell/_cuda/scale/scale.cu)
  add_nb_cuda_module(_qc_cuda           src/rapids_singlecell/_cuda/qc/qc.cu)
  add_nb_cuda_module(_qc_dask_cuda      src/rapids_singlecell/_cuda/qc_dask/qc_kernels_dask.cu)
  add_nb_cuda_module(_bbknn_cuda        src/rapids_singlecell/_cuda/bbknn/bbknn.cu)
  add_nb_cuda_module(_norm_cuda         src/rapids_singlecell/_cuda/norm/norm.cu)
  add_nb_cuda_module(_gmm_cuda          src/rapids_singlecell/_cuda/gmm/gmm.cu)
  target_link_libraries(_gmm_cuda PRIVATE CUDA::cublas)
  target_link_libraries(_gmm_cuda PRIVATE CUDA::cusolver)
  add_nb_cuda_module(_mixscale_cuda     src/rapids_singlecell/_cuda/mixscale/mixscale.cu)
  add_nb_cuda_module(_pr_cuda           src/rapids_singlecell/_cuda/pr/pr.cu)
  add_nb_cuda_module(_nn_descent_cuda   src/rapids_singlecell/_cuda/nn_descent/nn_descent.cu)
  add_nb_cuda_module(_aucell_cuda       src/rapids_singlecell/_cuda/aucell/aucell.cu)
  add_nb_cuda_module(_nanmean_cuda      src/rapids_singlecell/_cuda/nanmean/nanmean.cu)
  add_nb_cuda_module(_autocorr_cuda     src/rapids_singlecell/_cuda/autocorr/autocorr.cu)
  add_nb_cuda_module(_cooc_cuda         src/rapids_singlecell/_cuda/cooc/cooc.cu)
  add_nb_cuda_module(_aggr_cuda         src/rapids_singlecell/_cuda/aggr/aggr.cu)
  add_nb_cuda_module(_spca_cuda         src/rapids_singlecell/_cuda/spca/spca.cu)
  add_nb_cuda_module(_ligrec_cuda       src/rapids_singlecell/_cuda/ligrec/ligrec.cu)
  add_nb_cuda_module(_pv_cuda           src/rapids_singlecell/_cuda/pv/pv.cu)
  add_nb_cuda_module(_edistance_cuda    src/rapids_singlecell/_cuda/edistance/edistance.cu)
  add_nb_cuda_module(_sinkhorn_cuda     src/rapids_singlecell/_cuda/sinkhorn/sinkhorn.cu)
  add_nb_cuda_module(_guide_assignment_cuda src/rapids_singlecell/_cuda/guide_assignment/guide_assignment.cu)
  add_nb_cuda_module(_pseudobulk_cuda   src/rapids_singlecell/_cuda/pseudobulk/pseudobulk.cu)
  add_nb_cuda_module(_hvg_cuda          src/rapids_singlecell/_cuda/hvg/hvg.cu)
  add_nb_cuda_module(_kde_cuda          src/rapids_singlecell/_cuda/kde/kde.cu)
  add_nb_cuda_module(_wilcoxon_cuda     src/rapids_singlecell/_cuda/wilcoxon/wilcoxon.cu)
  target_sources(_wilcoxon_cuda PRIVATE src/rapids_singlecell/_cuda/wilcoxon/wilcoxon_rmm.cu)
  target_link_libraries(_wilcoxon_cuda PRIVATE rmm::rmm)
  add_nb_cuda_module(_wilcoxon_sparse_cuda src/rapids_singlecell/_cuda/wilcoxon/wilcoxon_sparse.cu)
  target_sources(_wilcoxon_sparse_cuda PRIVATE src/rapids_singlecell/_cuda/wilcoxon/wilcoxon_rmm.cu)
  target_link_libraries(_wilcoxon_sparse_cuda PRIVATE rmm::rmm)
  add_nb_cuda_module(_rank_stats_cuda   src/rapids_singlecell/_cuda/rank_genes/rank_stats.cu)
  # Harmony CUDA modules
  add_nb_cuda_module(_harmony_scatter_cuda   src/rapids_singlecell/_cuda/harmony/scatter/scatter.cu)
  add_nb_cuda_module(_harmony_outer_cuda     src/rapids_singlecell/_cuda/harmony/outer/outer.cu)
  add_nb_cuda_module(_harmony_colsum_cuda    src/rapids_singlecell/_cuda/harmony/colsum/colsum.cu)
  add_nb_cuda_module(_harmony_kmeans_cuda    src/rapids_singlecell/_cuda/harmony/kmeans/kmeans.cu)
  add_nb_cuda_module(_harmony_normalize_cuda src/rapids_singlecell/_cuda/harmony/normalize/normalize.cu)
  add_nb_cuda_module(_harmony_pen_cuda       src/rapids_singlecell/_cuda/harmony/pen/pen.cu)
  add_nb_cuda_module(_harmony_clustering_cuda src/rapids_singlecell/_cuda/harmony/clustering/clustering.cu)
  target_link_libraries(_harmony_clustering_cuda PRIVATE CUDA::cublas)
  add_nb_cuda_module(_harmony_correction_cuda src/rapids_singlecell/_cuda/harmony/correction/correction_fast.cu)
  target_link_libraries(_harmony_correction_cuda PRIVATE CUDA::cublas)
  add_nb_cuda_module(_harmony_correction_batched_cuda src/rapids_singlecell/_cuda/harmony/correction/correction_batched.cu)
  target_link_libraries(_harmony_correction_batched_cuda PRIVATE CUDA::cublas)
  # Wilcoxon binned histogram CUDA module
  add_nb_cuda_module(_wilcoxon_binned_cuda   src/rapids_singlecell/_cuda/wilcoxon_binned/wilcoxon_binned.cu)
endif()
