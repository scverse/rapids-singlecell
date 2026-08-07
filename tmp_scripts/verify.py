#!/usr/bin/env python3
from __future__ import annotations

import cupy as cp
import numpy as np
import scipy.sparse as sp

import rapids_singlecell as rsc
from rapids_singlecell.squidpy_gpu._sepal import _compute_idxs


def build_rect_grid(n=3):
    n_cells = n * n
    coords = (
        np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="xy"), -1)
        .reshape(-1, 2)
        .astype(np.float32)
    )
    rows, cols, data = [], [], []

    def idx(r, c):
        return r * n + c

    for r in range(n):
        for c in range(n):
            i = idx(r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    j = idx(rr, cc)
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0)
    g = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32)
    return g, coords


def verify_invariants_and_remaps(max_neighs=4):
    g_np, coords_np = build_rect_grid(3)
    g = rsc.get.X_to_GPU(g_np)
    spatial = rsc.get.X_to_GPU(coords_np.astype(np.float32))
    degrees = cp.diff(g.indptr)

    sat, sat_idx, unsat, nearest_sat = _compute_idxs(g, degrees, spatial, max_neighs)

    n_cells = g.shape[0]
    n_sat = len(sat)
    n_unsat = len(unsat)

    reorder_indices = cp.concatenate([sat, unsat])
    old_to_new = cp.empty(n_cells, dtype=cp.int32)
    old_to_new[reorder_indices] = cp.arange(n_cells, dtype=cp.int32)

    # Invariants
    assert cp.all(old_to_new[sat] == cp.arange(n_sat)), (
        "sat positions not contiguous at front"
    )
    assert cp.all(old_to_new[unsat] == cp.arange(n_sat, n_cells)), (
        "unsat positions not after sat"
    )

    # Remaps for kernel inputs
    sat_idx_mapped = old_to_new[sat_idx]
    nearest_sat_mapped = old_to_new[nearest_sat]

    assert sat_idx_mapped.shape == sat_idx.shape
    assert cp.all(nearest_sat_mapped >= 0) and cp.all(nearest_sat_mapped < n_sat), (
        "nearest_sat must map into [0, n_sat)"
    )

    # Neighbor-sum preservation
    rng = cp.random.RandomState(0)
    conc_orig = rng.rand(n_cells)
    conc_reordered = conc_orig[reorder_indices]
    for k in range(n_sat):
        s1 = conc_orig[sat_idx[k]].sum()
        s2 = conc_reordered[sat_idx_mapped[k]].sum()
        cp.testing.assert_allclose(
            s1, s2, rtol=0, atol=0, err_msg=f"neighbor sum mismatch at sat row {k}"
        )

    return {
        "n_cells": int(n_cells),
        "n_sat": int(n_sat),
        "n_unsat": int(n_unsat),
        "ok": True,
    }


def verify_single_step_update(max_neighs=4, dt=1e-2):
    g_np, coords_np = build_rect_grid(3)
    g = rsc.get.X_to_GPU(g_np)
    spatial = rsc.get.X_to_GPU(coords_np.astype(np.float32))
    degrees = cp.diff(g.indptr)

    sat, sat_idx, unsat, nearest_sat = _compute_idxs(g, degrees, spatial, max_neighs)

    n_cells = g.shape[0]
    n_sat = len(sat)

    reorder_indices = cp.concatenate([sat, unsat])
    old_to_new = cp.empty(n_cells, dtype=cp.int32)
    old_to_new[reorder_indices] = cp.arange(n_cells, dtype=cp.int32)

    sat_idx_mapped = old_to_new[sat_idx]
    nearest_sat_mapped = old_to_new[nearest_sat]

    rng = cp.random.RandomState(1)
    conc_orig = rng.rand(n_cells)
    conc_reo = conc_orig[reorder_indices].copy()

    # Rectangular Laplacian
    neighbor_sum_orig = cp.array([conc_orig[sat_idx[i]].sum() for i in range(n_sat)])
    neighbor_sum_reo = cp.array(
        [conc_reo[sat_idx_mapped[i]].sum() for i in range(n_sat)]
    )
    centers_orig = conc_orig[sat]
    centers_reo = conc_reo[:n_sat]
    d2_sat_orig = neighbor_sum_orig - 4.0 * centers_orig
    d2_sat_reo = neighbor_sum_reo - 4.0 * centers_reo

    conc_orig2 = conc_orig.copy()
    conc_reo2 = conc_reo.copy()

    conc_orig2[sat] = cp.maximum(0.0, conc_orig2[sat] + d2_sat_orig * dt)
    conc_reo2[:n_sat] = cp.maximum(0.0, conc_reo2[:n_sat] + d2_sat_reo * dt)

    # Unsat update using nearest sat derivative (per-unsat order)
    for i in range(len(unsat)):
        u_global = unsat[i]
        ns = nearest_sat[i]  # nearest sat node id

        # Find position of ns in sat array
        ns_pos_orig = cp.where(sat == ns)[0][0]
        conc_orig2[u_global] = cp.maximum(
            0.0, conc_orig2[u_global] + d2_sat_orig[ns_pos_orig] * dt
        )

        # In reordered space, nearest_sat_mapped[i] gives us the position in reordered array
        # But we need the position in the saturated block (0..n_sat-1)
        ns_pos_reo = nearest_sat_mapped[i]  # This should be < n_sat
        u_reo = n_sat + i
        conc_reo2[u_reo] = cp.maximum(
            0.0, conc_reo2[u_reo] + d2_sat_reo[ns_pos_reo] * dt
        )

    # Map back reordered → original
    inv = cp.empty_like(old_to_new)
    inv[old_to_new] = cp.arange(n_cells, dtype=cp.int32)
    conc_reconstructed = conc_reo2[inv]

    cp.testing.assert_allclose(
        conc_orig2,
        conc_reconstructed,
        rtol=0,
        atol=1e-12,
        err_msg="single-step update mismatch",
    )

    return {"ok": True}


def main():
    res1 = verify_invariants_and_remaps()
    print(
        "Invariant+remap checks:",
        {k: int(v) if isinstance(v, (np.integer,)) else v for k, v in res1.items()},
    )
    res2 = verify_single_step_update()
    print("Single-step update equivalence:", res2)
    print("All mapping tests passed.")


if __name__ == "__main__":
    main()
