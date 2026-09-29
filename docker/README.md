# Deprecated container recipes

The project-provided CUDA 12 and CUDA 13 container images and their dependency
images are deprecated. Automated builds and publication have stopped. Existing
images are retained, but will no longer receive updates, including security fixes;
their `latest` tags will no longer track new rapids-singlecell releases.

The Dockerfiles and local build script in this directory are retained as
unmaintained reference material. For maintained installations, use the Conda
environments or prebuilt wheels with CUDA-X Data Science dependencies described in
the [installation guide](../docs/installation.md).

The `rapids-singlecell-cu12` and `rapids-singlecell-cu13` Python packages remain
supported. The manylinux images used by CI to build those wheels are also unaffected.
