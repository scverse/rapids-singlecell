FROM quay.io/pypa/manylinux_2_28_x86_64

# Install GCC 12 (CUDA 12.2 does not support GCC > 12)
RUN yum -y install gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ && \
    yum clean all

# Add NVIDIA CUDA repo (RHEL8/Alma8 base in manylinux_2_28)
RUN yum -y install dnf-plugins-core && \
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    yum -y clean all && yum -y makecache && \
    # Install what you link against, plus cusolver's runtime nvJitLink dependency
    # (cusolver's ELF NEEDs libnvJitLink but its RPM does not declare it, so it
    # must be installed explicitly or the nanobind stub-generation import fails).
    yum -y install \
      cuda-nvcc-12-2 \
      cuda-cudart-12-2 \
      cuda-cudart-devel-12-2 \
      libcublas-12-2 \
      libcublas-devel-12-2 \
      libcusparse-12-2 \
      libcusparse-devel-12-2 \
      libcusolver-12-2 \
      libcusolver-devel-12-2 \
      libnvjitlink-12-2 && \
    yum clean all

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/usr/local/cuda/bin:${PATH}
