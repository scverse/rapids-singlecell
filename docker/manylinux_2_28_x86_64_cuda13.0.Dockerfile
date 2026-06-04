FROM quay.io/pypa/manylinux_2_28_x86_64

# Add NVIDIA CUDA repo (RHEL8/Alma8 base in manylinux_2_28)
RUN yum -y install dnf-plugins-core && \
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    yum -y clean all && yum -y makecache && \
    # Install only what you actually link against
    yum -y install \
      cuda-nvcc-13-0 \
      cuda-cudart-13-0 \
      cuda-cudart-devel-13-0 \
      libcublas-13-0 \
      libcublas-devel-13-0 \
      libcusparse-13-0 \
      libcusparse-devel-13-0 \
      libcusolver-13-0 \
      libcusolver-devel-13-0 && \
    yum clean all

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
