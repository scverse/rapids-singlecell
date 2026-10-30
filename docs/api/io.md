# Reading data: `io`

Read AnnData zarr stores lazily for out-of-core analysis, with chunks read straight into GPU memory
and decompressed there if zarr-kvikio, KvikIO and nvCOMP are installed (see {doc}`/out_of_core`).
The codec pipeline that does so comes from zarr-kvikio (`"zarr_kvikio.KvikioCodecPipeline"`).

```{eval-rst}
.. module:: rapids_singlecell.io
.. currentmodule:: rapids_singlecell

.. autosummary::
    :toctree: generated

    io.read_lazy
    io.convert_zarr
    io.check
    io.enable
    io.release_memory
```
