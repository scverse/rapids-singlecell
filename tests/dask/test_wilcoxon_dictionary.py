from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as sparse
import numpy as np
from scipy.sparse import csr_matrix

from rapids_singlecell.tools._rank_genes_groups._wilcoxon_dictionary import (
    assemble_records,
    pack_dictionary,
)


def test_lossless_mixed_records(tmp_path):
    blocks, expected, groups = [], [], []
    rows, cols = 4, 512
    indices = np.tile(np.arange(cols, dtype=np.int32), rows)
    ptr = np.arange(rows + 1, dtype=np.int32) * cols
    for i, distinct in enumerate((7, rows * cols)):
        values = np.arange(distinct, dtype=np.float32)
        values[:4] = [-0.0, 0.0, np.nextafter(np.float32(1), np.float32(2)), 1.0]
        if i:
            indices[127:130] = [129, 127, 127]  # Noncanonical, crossing a warp boundary.
            values[0] = 0  # CuPy canonicalization sums duplicates starting from +0.
        host = csr_matrix((np.resize(values, rows * cols), indices.copy(), ptr.copy()), shape=(rows, cols))
        labels = np.arange(201 * i, 201 * i + rows, dtype=np.uint16)
        csr = sparse.csr_matrix((cp.asarray(host.data), cp.asarray(indices), cp.asarray(ptr)), shape=host.shape)
        pieces = pack_dictionary(csr, labels, [0, 30, 290, cols])
        offsets, words, aux, bits = pieces[1]
        assert bool(bits) == (i == 0)
        if i == 0:  # Stage one dictionary and its records from the worker's spill store.
            np.save(tmp_path / "table.npy", aux.get())
            aux = np.load(tmp_path / "table.npy", mmap_mode="r")
            words = words.get()
        blocks.append((offsets, words, aux, bits))
        np.testing.assert_array_equal(csr.indices.get(), indices)
        host.sum_duplicates()
        col = host[:, 171:270].tocsc()
        expected.append(col.data.reshape(-1, rows))
        groups.append(labels[col.indices].reshape(-1, rows))
    offsets, values, labels = assemble_records(blocks, 141, 240)
    np.testing.assert_array_equal(offsets, np.arange(100) * 2 * rows)
    np.testing.assert_array_equal(values.get().view(np.uint32), np.column_stack(expected).ravel().view(np.uint32))
    np.testing.assert_array_equal(labels.get(), np.column_stack(groups).ravel())
