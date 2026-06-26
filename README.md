[![Stars](https://img.shields.io/github/stars/scverse/rapids_singlecell?style=flat&logo=GitHub&color=blue)](https://github.com/scverse/rapids_singlecell/stargazers)
[![PyPI](https://img.shields.io/pypi/v/rapids-singlecell?logo=PyPI)](https://pypi.org/project/rapids-singlecell)
[![Downloads](https://static.pepy.tech/badge/rapids-singlecell)](https://pepy.tech/project/rapids-singlecell)
[![Documentation Status](https://readthedocs.org/projects/rapids-singlecell/badge/?version=latest)](https://rapids-singlecell.readthedocs.io/en/latest/?badge=latest)
[![CI-Pass](https://github.com/scverse/rapids_singlecell/actions/workflows/test-gpu.yml/badge.svg)](https://github.com/scverse/rapids_singlecell/actions/workflows/test-gpu.yml)
[![codecov](https://codecov.io/gh/scverse/rapids-singlecell/graph/badge.svg?token=PFHJEQD94X)](https://codecov.io/gh/scverse/rapids-singlecell)
[![Chat](https://img.shields.io/badge/zulip-join_chat-%2367b08f.svg)](https://scverse.zulipchat.com)


# rapids-singlecell: GPU-Accelerated Single-Cell Analysis within scverse®

rapids-singlecell provides GPU-accelerated single-cell analysis with an AnnData-first API.
It is largely compatible with Scanpy and includes selected functionality from Squidpy, decoupler, and pertpy.
Computations use CuPy and NVIDIA RAPIDS for performance on large datasets.

- **GPU acceleration**: Common single-cell workflows on `AnnData` run on the GPU.
- **Ecosystem compatibility**: Works with Scanpy APIs; includes pieces from Squidpy, decoupler, and pertpy.
- **Simple installation**: Available via Conda and PyPI.

## Documentation

For more information please have a look through the [documentation](https://rapids-singlecell.readthedocs.io/en/latest/)

## Citation

If you use this tool, please cite: [![arXiv](https://img.shields.io/badge/arXiv-2603.02402-b31b1b.svg)](https://doi.org/10.48550/arXiv.2603.02402)

Please cite the relevant tools if used: [decoupler](https://doi.org/10.1093/bioadv/vbac016) for decoupler functions, [squidpy](https://doi.org/10.1038/s41592-021-01358-2) for spatial analysis, and [pertpy](https://doi.org/10.1038/s41592-024-02233-6) for perturbation analysis.

[//]: # (numfocus-fiscal-sponsor-attribution)

rapids-singlecell is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>
