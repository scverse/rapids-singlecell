"""cuTile kernels for Harmony with bfloat16 assignments (R); everything else float32.

Built ahead of time by ``export_harmony_tile.py`` and embedded in the extension
modules; nothing here runs at import time. Float32 operands are split into a
bfloat16 high and low part so the products run on tensor cores with float32
accumulation (about 16 mantissa bits for the float32 side, exact for R).
"""

from __future__ import annotations

import cuda.tile as ct

KP = 128  # clusters, padded (Harmony's fused path supports at most 128)


def _split(x):
    hi = ct.astype(x, ct.bfloat16)
    return hi, ct.astype(x - ct.astype(hi, ct.float32), ct.bfloat16)


def make_rtz(dp: int, row_tile: int, occupancy: int):
    """Partial sums of R^T Z: block g adds row tiles g*per .. g*per+per-1 into P[g]."""

    @ct.kernel(occupancy=occupancy)
    def rtz(R, Z, P, per: int, n_tiles: int):
        g = ct.bid(0)
        acc = ct.zeros((KP, dp), dtype=ct.float32)
        for t in range(per):
            i = g * per + t
            if i < n_tiles:
                r = ct.transpose(
                    ct.load(R, (i, 0), (row_tile, KP), padding_mode=ct.PaddingMode.ZERO)
                )
                z_hi, z_lo = _split(
                    ct.load(Z, (i, 0), (row_tile, dp), padding_mode=ct.PaddingMode.ZERO)
                )
                acc = ct.mma(r, z_hi, acc)
                acc = ct.mma(r, z_lo, acc)
        ct.store(P, (g, 0, 0), ct.reshape(acc, (1, KP, dp)))

    return rtz


def make_neg_rw(dp: int, row_tile: int, occupancy: int):
    """Out = -(R @ W) for one batch: R (rows x K) bfloat16, W (K x D) float32."""

    @ct.kernel(occupancy=occupancy)
    def neg_rw(R, W, Out):
        i = ct.bid(0)
        r = ct.load(R, (i, 0), (row_tile, KP), padding_mode=ct.PaddingMode.ZERO)
        w_hi, w_lo = _split(
            ct.load(W, (0, 0), (KP, dp), padding_mode=ct.PaddingMode.ZERO)
        )
        acc = ct.zeros((row_tile, dp), dtype=ct.float32)
        acc = ct.mma(r, w_hi, acc)
        acc = ct.mma(r, w_lo, acc)
        ct.store(Out, (i, 0), -acc)

    return neg_rw
