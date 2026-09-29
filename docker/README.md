# Container deprecation notice

The project-provided CUDA 12 and CUDA 13 container images and their dependency
images will be deprecated in a future release. Container builds and publication
continue for now; no deprecation date has been set.

Please plan to migrate to the Conda environments or prebuilt wheels with CUDA-X
Data Science dependencies described in the [installation guide](../docs/installation.md).

The `rapids-singlecell-cu12` and `rapids-singlecell-cu13` Python packages remain
supported. The manylinux images used by CI to build those wheels are also unaffected.
