```{include} ../README.md
:start-line: 0
:end-line: 17
```

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

```{div}
:style: height: 0.5rem
```

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Installation {octicon}`plug;1em;`
:link: installation
:link-type: doc

New to *rapids-singlecell*? Check out the installation guide.
:::

:::{grid-item-card} Tutorials {octicon}`play;1em;`
:link: tutorials
:link-type: doc

The tutorials walk you through real-world applications of rapids-singlecell.
:::

:::{grid-item-card} API reference {octicon}`book;1em;`
:link: api/index
:link-type: doc

The API reference contains a detailed description of
the rapids-singlecell API.
:::

:::{grid-item-card} GitHub {octicon}`mark-github;1em;`
:link: https://github.com/scverse/rapids_singlecell

Find a bug? Interested in improving rapids-singlecell? Checkout our GitHub for the latest developments.
:::
::::


## Citation

If you use rapids-singlecell, please cite:

> Dicks, S., et al. (2026). GPU-accelerated single-cell analysis at scale with rapids-singlecell. *arXiv*. [https://doi.org/10.48550/arXiv.2603.02402](https://doi.org/10.48550/arXiv.2603.02402)

## News

* 30.04.26 [**v0.15.0 released!**](https://scverse.org/blog/2026-rsc-goes-nanobind/) This release ships precompiled CUDA kernels via [nanobind](https://github.com/wjakob/nanobind) — no CUDA toolkit needed at install time. Prebuilt wheels for x86_64 and aarch64 support CUDA 12 and 13, covering Turing through Blackwell GPUs. Install with `pip install rapids-singlecell-cu13` (or `-cu12`). See the [installation guide](installation.md) for details.
* 04.03.26 **rapids-singlecell is now on arXiv!** Check out our preprint: [GPU-accelerated single-cell analysis at scale with rapids-singlecell](https://doi.org/10.48550/arXiv.2603.02402)
* 01.07.25 *rapids-singlecell* is now an [scverse® core package](https://scverse.org/blog/2025-core-expansion/)
* 12.06.25 *rapids-singlecell* was highlighted in an other NVIDIA [technical blog post](https://developer.nvidia.com/blog/driving-toward-billion-cell-analysis-and-biological-breakthroughs-with-rapids-singlecell/)
* 07.08.23 *rapids-singlecell* is now part of scverse® ecosystem.
* 04.08.23 Thanks to the great team at [scverse](https://www.scverse.org)® *rapids-singlecell* is now automatically tested with CI
* 27.06.23 I'm very honored to announce that I was invited to co-author a [technical blog post](https://developer.nvidia.com/blog/gpu-accelerated-single-cell-rna-analysis-with-rapids-singlecell/) that demonstrates the capabilities and performance of *rapids-singlecell* for NVIDIA.
