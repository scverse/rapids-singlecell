from __future__ import annotations

import _warnings
import warnings
from types import SimpleNamespace

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy import sparse as cpx_sparse
from natsort import natsorted
from scanpy.datasets import pbmc3k_processed, pbmc68k_reduced
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import rapids_singlecell as rsc
from rapids_singlecell._utils._random import _LegacyRng
from rapids_singlecell.tools import _leiden

HAS_NATIVE = _leiden._ld is not None and hasattr(_leiden._ld, "leiden")
needs_native = pytest.mark.skipif(
    not HAS_NATIVE, reason="native Leiden extension not built"
)
FLAVORS = ["cugraph", pytest.param("rapids", marks=needs_native)]


@pytest.fixture
def adata_neighbors():
    return pbmc68k_reduced()


@pytest.fixture
def unpatched_warn(monkeypatch):
    """The interpreter's own `warnings.warn`, for tests of a warning's location.

    `rapids_dask_dependency.utils.patch_warning_stacklevel` restores the function
    without try/finally, so after some dask tests a wrapper stays installed that
    moves every warning up the stack.
    """
    monkeypatch.setattr(warnings, "warn", _warnings.warn)


# ---------- helpers ----------


def _graph(name):
    """CSR test graphs (fp32, canonical, symmetric)."""
    if name == "pbmc68k":
        a = pbmc68k_reduced().obsp["connectivities"]
    elif name == "pbmc3k":
        a = pbmc3k_processed().obsp["connectivities"]
    elif name == "sbm":
        rng = np.random.default_rng(1)
        blocks = np.repeat(np.arange(30), 20)
        iu = np.triu_indices(blocks.size, 1)
        same = blocks[iu[0]] == blocks[iu[1]]
        keep = rng.random(iu[0].size) < np.where(same, 0.4, 0.004)
        a = sparse.coo_matrix(
            (np.ones(keep.sum()), (iu[0][keep], iu[1][keep])), shape=(600, 600)
        )
        a = a + a.T
    elif name == "rings":  # 30 cliques of 5, each linked to the next
        rows, cols = [], []
        for q in range(30):
            iu = np.triu_indices(5, 1)
            rows += list(5 * q + iu[0]) + [5 * q + 4]
            cols += list(5 * q + iu[1]) + [5 * ((q + 1) % 30)]
        a = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(150, 150))
        a = a + a.T
    elif name == "pairs":  # weakly linked pairs: refinement re-enters TOP
        rng = np.random.default_rng(0)
        rows, cols, w = [], [], []
        for i in range(30):
            rows += [2 * i] + ([2 * i + 1] if i < 29 else [])
            cols += [2 * i + 1] + ([2 * i + 2] if i < 29 else [])
            w += list(rng.uniform(0.8, 1.2, 2 if i < 29 else 1))
        a = sparse.coo_matrix((w, (rows, cols)), shape=(60, 60))
        a = a + a.T
    elif name == "star":  # a hub of degree 3000 and a path through its leaves
        n = 3001
        rows = [0] * 3000 + list(range(1, n - 1))
        cols = list(range(1, n)) + list(range(2, n))
        a = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        a = a + a.T
    elif name == "components":  # blocks plus isolated vertices
        rng = np.random.default_rng(2)
        mats = [
            sparse.random(40 + 9 * i, 40 + 9 * i, density=0.15, random_state=rng)
            for i in range(5)
        ]
        a = sparse.block_diag([*mats, sparse.csr_matrix((20, 20))])
        a = a + a.T
    else:
        raise ValueError(name)
    a = sparse.csr_matrix(a, dtype=np.float32)
    a.setdiag(0)
    a.eliminate_zeros()
    a.sort_indices()
    return a


def native(adjacency, resolutions=(0.5, 1.0), seed=3, n_iterations=2, **kwargs):
    """``flavor='rapids'`` with the native seed ``seed`` for every resolution."""
    return _leiden._leiden_rapids(
        adjacency,
        [float(r) for r in np.atleast_1d(resolutions)],
        rng=_LegacyRng(seed),
        n_iterations=n_iterations,
        use_weights=kwargs.pop("use_weights", True),
        **kwargs,
    )


def assert_same(a, b):
    """Bitwise equal labels and modularity."""
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.membership.tobytes() == y.membership.tobytes()
        assert float.hex(x.modularity) == float.hex(y.modularity)
        assert x.n_clusters == y.n_clusters


def _fp64_modularity(adjacency, labels, resolution, *, use_weights=True):
    """Q with resolution, float64 on the host; diagonal and non-positive entries ignored."""
    m = sparse.coo_matrix(adjacency)
    w = m.data.astype(np.float64)
    counted = (m.row != m.col) & ((w > 0) if use_weights else (w != 0))
    row, col = m.row[counted], m.col[counted]
    w = w[counted] if use_weights else np.ones(counted.sum())
    labels = np.asarray(labels)
    two_m = w.sum()
    inner = w[labels[row] == labels[col]].sum()
    volumes = np.bincount(labels, weights=np.bincount(row, w, m.shape[0]))
    return inner / two_m - resolution * ((volumes / two_m) ** 2).sum()


def _disconnected(adjacency, labels):
    """Number of communities whose induced subgraph is not connected."""
    m = sparse.coo_matrix(adjacency)
    keep = (labels[m.row] == labels[m.col]) & (m.row != m.col)
    keep &= m.data > 0
    inner = sparse.csr_matrix(
        (np.ones(keep.sum()), (m.row[keep], m.col[keep])), adjacency.shape
    )
    _, comp = connected_components(inner, directed=False)
    pieces = {(int(c), int(lab)) for c, lab in zip(comp, labels, strict=True)}
    return len(pieces) - len(set(labels.tolist()))


def _assert_size_ordered(labels: pd.Series):
    assert labels.cat.categories.tolist() == [
        str(i) for i in range(len(labels.cat.categories))
    ]
    codes = labels.cat.codes.to_numpy()
    sizes = np.bincount(codes)
    assert (sizes > 0).all()
    assert (np.diff(sizes) <= 0).all()
    # equal sizes: ordered by their first cell
    first = np.array([np.flatnonzero(codes == c)[0] for c in range(sizes.size)])
    for a in range(sizes.size - 1):
        if sizes[a] == sizes[a + 1]:
            assert first[a] < first[a + 1]


def _with_index_dtypes(a, indptr_dtype, indices_dtype):
    b = sparse.csr_matrix(a, copy=True)
    # set after construction: scipy's constructor would downcast to int32
    b.indptr = b.indptr.astype(indptr_dtype)
    b.indices = b.indices.astype(indices_dtype)
    return b


# ---------- API: ports of scanpy's and rsc's leiden tests ----------


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("resolution", [1, 2])
@pytest.mark.parametrize("n_iterations", [-1, 3])
def test_leiden_basic(adata_neighbors, flavor, resolution, n_iterations):
    rsc.tl.leiden(
        adata_neighbors,
        flavor=flavor,
        resolution=resolution,
        n_iterations=n_iterations,
        key_added="leiden_custom",
    )
    params = adata_neighbors.uns["leiden_custom"]["params"]
    assert params["resolution"] == resolution
    assert params["n_iterations"] == n_iterations
    assert params["random_state"] == 0


@needs_native  # cuGraph is not reproducible for a fixed seed
@pytest.mark.parametrize("rng_arg", ["rng", "random_state"])
def test_leiden_random_state(rng_arg):
    adata = pbmc3k_processed()
    adata_1, adata_1_again, adata_2 = (
        rsc.tl.leiden(adata, copy=True, **{rng_arg: seed}) for seed in (1, 1, 42)
    )
    pd.testing.assert_series_equal(adata_1.obs["leiden"], adata_1_again.obs["leiden"])
    assert float.hex(adata_1.uns["leiden"]["modularity"]) == float.hex(
        adata_1_again.uns["leiden"]["modularity"]
    )
    assert not adata_2.obs["leiden"].equals(adata_1_again.obs["leiden"])
    assert ("random_state" in adata_1.uns["leiden"]["params"]) == (
        rng_arg == "random_state"
    )


@pytest.mark.parametrize("flavor", FLAVORS)
def test_leiden_objective_function(adata_neighbors, flavor):
    rsc.tl.leiden(adata_neighbors, flavor=flavor, objective_function="modularity")
    with pytest.raises(NotImplementedError, match="objective_function='modularity'"):
        rsc.tl.leiden(adata_neighbors, flavor=flavor, objective_function="CPM")


@pytest.mark.parametrize("flavor", FLAVORS)
def test_clustering_custom_key(adata_neighbors, flavor):
    default_res, custom_resolutions = 0.8, [0.9, 1.1]
    custom_keys = [f"leiden_{res}" for res in custom_resolutions]
    rsc.tl.leiden(adata_neighbors, flavor=flavor, resolution=default_res)
    for key, res in zip(custom_keys, custom_resolutions, strict=True):
        rsc.tl.leiden(adata_neighbors, flavor=flavor, resolution=res, key_added=key)
    # all clustering parameters are added to user provided keys, not overwritten
    assert adata_neighbors.uns["leiden"]["params"]["resolution"] == default_res
    for key, res in zip(custom_keys, custom_resolutions, strict=True):
        assert adata_neighbors.uns[key]["params"]["resolution"] == res


@pytest.mark.parametrize("flavor", FLAVORS)
def test_restrict_to_key(adata_neighbors, flavor):
    adata = adata_neighbors
    rsc.tl.leiden(adata, flavor=flavor)
    restrict_to = ("leiden", ["0", "1"])
    rsc.tl.leiden(adata, flavor=flavor, restrict_to=restrict_to)
    # scanpy: the default key gets the `_R` suffix in obs *and* uns
    assert "leiden_R" in adata.obs
    assert adata.uns["leiden_R"]["params"]["resolution"] == 1.0
    assert adata.uns["leiden"]["params"]["resolution"] == 1.0
    in_0_1 = adata.obs["leiden"].isin(["0", "1"]).to_numpy()
    assert adata.obs["leiden_R"][in_0_1].str.startswith("0-1,").all()
    np.testing.assert_array_equal(
        adata.obs["leiden_R"][~in_0_1].astype(str),
        adata.obs["leiden"][~in_0_1].astype(str),
    )
    categories = adata.obs["leiden_R"].cat.categories.tolist()
    assert categories == natsorted(categories)
    rsc.tl.leiden(adata, flavor=flavor, restrict_to=restrict_to, resolution=[0.5, 1])
    assert {"leiden_R_0.5", "leiden_R_1"} <= set(adata.obs.columns)
    assert adata.uns["leiden_R"]["params"]["resolution"] == [0.5, 1]
    rsc.tl.leiden(adata, flavor=flavor, restrict_to=restrict_to, key_added="sub")
    assert "sub" in adata.obs
    assert "sub" in adata.uns
    assert "sub_R" not in adata.obs


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("resolution", [0.5, 1.0, [0.25, 2.0]])
def test_labels_size_ordered(adata_neighbors, flavor, resolution):
    rsc.tl.leiden(adata_neighbors, flavor=flavor, resolution=resolution)
    keys = (
        ["leiden"]
        if isinstance(resolution, float)
        else [f"leiden_{r}" for r in resolution]
    )
    if len(keys) > 1:
        modularity = adata_neighbors.uns["leiden"]["modularity"]
        assert [type(q) for q in modularity] == [float] * len(keys)
    for key in keys:
        labels = adata_neighbors.obs[key]
        _assert_size_ordered(labels)
        # same categorical as scanpy's natsorted string labels
        expected = pd.Categorical(
            labels.astype(str), categories=natsorted(labels.astype(str).unique())
        )
        assert labels.cat.categories.tolist() == expected.categories.tolist()
        np.testing.assert_array_equal(labels.cat.codes, expected.codes)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("resolution", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("use_weights", [True, False])
def test_modularity_is_exact(adata_neighbors, flavor, resolution, use_weights):
    rsc.tl.leiden(
        adata_neighbors, flavor=flavor, resolution=resolution, use_weights=use_weights
    )
    q = adata_neighbors.uns["leiden"]["modularity"]
    assert type(q) is float
    a = adata_neighbors.obsp["connectivities"]
    codes = adata_neighbors.obs["leiden"].cat.codes.to_numpy()
    expected = _fp64_modularity(a, codes, resolution, use_weights=use_weights)
    assert q == pytest.approx(expected, rel=0, abs=1e-9)
    if flavor == "cugraph":  # bitwise the exact Q of its labels
        graph, info, weighted = _leiden._ingest(
            _leiden._as_csr(a), use_weights=use_weights, host_input=False
        )
        exact = _leiden._exact_modularity(graph, info, use_weights=weighted)
        assert float.hex(q) == float.hex(exact(codes, int(codes.max()) + 1, resolution))


def test_modularity_matches_igraph(adata_neighbors):
    igraph = pytest.importorskip("igraph")
    adjacency = adata_neighbors.obsp["connectivities"]
    rsc.tl.leiden(adata_neighbors, resolution=[0.5, 1.5])
    m = sparse.triu(adjacency, k=1).tocoo()
    g = igraph.Graph(n=adjacency.shape[0], edges=list(zip(m.row, m.col)))
    for res, q in zip([0.5, 1.5], adata_neighbors.uns["leiden"]["modularity"]):
        membership = adata_neighbors.obs[f"leiden_{res}"].cat.codes.tolist()
        expected = g.modularity(membership, weights=m.data, resolution=res)
        assert q == pytest.approx(expected, rel=0, abs=1e-9)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_isolated_vertices(adata_neighbors, flavor):
    # rsc #474: isolated cells are kept, each as its own cluster
    adjacency = adata_neighbors.obsp["connectivities"].tolil()
    isolated = [3, 50, 699]
    adjacency[isolated, :] = 0
    adjacency[:, isolated] = 0
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    rsc.tl.leiden(adata_neighbors, flavor=flavor, adjacency=adjacency)
    labels = adata_neighbors.obs["leiden"]
    assert len(labels) == adata_neighbors.n_obs
    counts = labels.value_counts()
    assert (counts[labels.iloc[isolated]] == 1).all()
    _assert_size_ordered(labels)


def test_copy(adata_neighbors):
    result = rsc.tl.leiden(adata_neighbors, copy=True)
    assert "leiden" not in adata_neighbors.obs
    assert "leiden" in result.obs
    assert rsc.tl.leiden(adata_neighbors) is None


# ---------- argument handling ----------


_FLAVOR_MSG = "flavor must be either 'rapids' or 'cugraph', but 'foo'"
_N_ITER_MSG = "n_iterations must be a positive integer or -1"
_RES_MSG = r"resolution must be finite and in \[0, 2\*\*20\]"
_UNKNOWN_MSG = "unexpected keyword arguments"
BOTH = ("rapids", "cugraph")


@pytest.mark.parametrize(
    ("flavors", "kwargs", "error", "match"),
    [
        (BOTH, {"directed": True}, ValueError, r"Cannot use flavor=.*directed"),
        (BOTH, {"partition_type": object}, ValueError, "Do not pass in partition_type"),
        (("foo",), {}, ValueError, _FLAVOR_MSG),
        (("foo",), {"use_dask": True}, ValueError, _FLAVOR_MSG),
        *(
            (BOTH, {"n_iterations": n}, ValueError, _N_ITER_MSG)
            for n in (0, -2, 1.5, True, "2")
        ),
        *(
            (BOTH, {"resolution": r}, ValueError, _RES_MSG)
            for r in (-0.5, np.nan, np.inf, -np.inf, 2.0**21, [1.0, np.nan], ["1"])
        ),
        (("rapids",), {"resolution": []}, ValueError, "non-empty"),
        (("rapids",), {"max_iter": 5}, TypeError, _UNKNOWN_MSG),
        (("rapids",), {"foo": 1}, TypeError, _UNKNOWN_MSG),
        (("cugraph",), {"beta": 0.01}, TypeError, _UNKNOWN_MSG),
        (("cugraph",), {"initial_membership": [0, 1]}, TypeError, _UNKNOWN_MSG),
    ],
)
def test_invalid_arguments(adata_neighbors, flavors, kwargs, error, match):
    for flavor in flavors:
        with pytest.raises(error, match=match):
            rsc.tl.leiden(adata_neighbors, flavor=flavor, **kwargs)
        assert "leiden" not in adata_neighbors.obs


@needs_native
def test_n_iterations_above_limit(adata_neighbors, unpatched_warn):
    with pytest.warns(
        UserWarning, match=r"values above 20 run until stable \(as -1\)"
    ) as record:
        rsc.tl.leiden(adata_neighbors, n_iterations=100)
    assert record[0].filename == __file__
    assert adata_neighbors.uns["leiden"]["params"]["n_iterations"] == 100
    rsc.tl.leiden(adata_neighbors, n_iterations=-1, key_added="stable")
    pd.testing.assert_series_equal(
        adata_neighbors.obs["leiden"], adata_neighbors.obs["stable"], check_names=False
    )


@needs_native
def test_rapids_parameter_checks(adata_neighbors):
    n = adata_neighbors.n_obs
    for beta in (-0.1, np.nan):
        with pytest.raises(ValueError, match="beta must be finite and >= 0"):
            rsc.tl.leiden(adata_neighbors, beta=beta)
    with pytest.raises(NotImplementedError, match="beta > 0"):
        rsc.tl.leiden(adata_neighbors, beta=0.01)
    for bad in ([0] * (n - 1), np.zeros(n, float)):
        with pytest.raises(ValueError, match="initial_membership"):
            rsc.tl.leiden(adata_neighbors, initial_membership=bad)
    with pytest.raises(ValueError, match="negative"):
        rsc.tl.leiden(adata_neighbors, initial_membership=np.full(n, -1))
    with pytest.raises(ValueError, match="square"):
        rsc.tl.leiden(
            adata_neighbors, adjacency=adata_neighbors.obsp["connectivities"][:, :10]
        )
    assert "leiden" not in adata_neighbors.obs


@pytest.mark.parametrize("native_module", [None, object()], ids=["absent", "stub"])
def test_rapids_unavailable(adata_neighbors, monkeypatch, native_module):
    monkeypatch.setattr(_leiden, "_ld", native_module)
    with pytest.raises(NotImplementedError, match="flavor='rapids' is not available"):
        rsc.tl.leiden(adata_neighbors, flavor="rapids")
    assert "leiden" not in adata_neighbors.obs
    # flavor='cugraph' still runs; its modularity is summed in float64
    rsc.tl.leiden(adata_neighbors, flavor="cugraph", resolution=0.5)
    codes = adata_neighbors.obs["leiden"].cat.codes.to_numpy()
    expected = _fp64_modularity(adata_neighbors.obsp["connectivities"], codes, 0.5)
    q = adata_neighbors.uns["leiden"]["modularity"]
    assert q == pytest.approx(expected, rel=0, abs=1e-12)


def _record_flavor_calls(monkeypatch) -> list[str]:
    """Replace both flavor back ends by stubs that return one cluster per call."""
    calls = []

    def stub(flavor):
        def run(adjacency, resolutions, **kwargs):
            calls.append(flavor)
            n = adjacency.shape[0]
            return [
                SimpleNamespace(
                    membership=np.zeros(n, np.int32), modularity=0.0, n_clusters=1
                )
                for _ in resolutions
            ]

        return run

    monkeypatch.setattr(_leiden, "_leiden_rapids", stub("rapids"))
    monkeypatch.setattr(_leiden, "_leiden_cugraph", stub("cugraph"))
    monkeypatch.setattr(_leiden, "_ld", SimpleNamespace(leiden=None))
    return calls


def test_default_flavor(adata_neighbors, monkeypatch):
    import inspect

    default = inspect.signature(rsc.tl.leiden).parameters["flavor"].default
    assert default == _leiden._DEFAULT_FLAVOR == "rapids"
    calls = _record_flavor_calls(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        rsc.tl.leiden(adata_neighbors)
        rsc.tl.leiden(adata_neighbors, flavor="rapids", resolution=[0.5, 1])
        # the deprecations are tied to the default: with cugraph as the
        # default (as before the flip), cugraph and use_dask calls do not warn
        monkeypatch.setattr(_leiden, "_DEFAULT_FLAVOR", "cugraph")
        rsc.tl.leiden(adata_neighbors, flavor="cugraph", theta=0.5)
        rsc.tl.leiden(adata_neighbors, flavor="cugraph", use_dask=True)
    assert calls == ["rapids"] * 2 + ["cugraph"] * 2


def test_deprecations(adata_neighbors, monkeypatch, unpatched_warn):
    calls = _record_flavor_calls(monkeypatch)
    with pytest.warns(FutureWarning, match=r"flavor='cugraph' is deprecated") as rec:
        rsc.tl.leiden(adata_neighbors, flavor="cugraph")
    assert rec[0].filename == __file__
    for flavor in ("rapids", "cugraph"):
        with pytest.warns(FutureWarning) as rec:
            rsc.tl.leiden(adata_neighbors, flavor=flavor, use_dask=True)
        # one warning: the use_dask one already names the cuGraph implementation
        assert [str(w.message)[:23] for w in rec] == ["use_dask is deprecated "]
    with pytest.warns(FutureWarning, match=r"theta is a cuGraph parameter"):
        rsc.tl.leiden(adata_neighbors, flavor="rapids", theta=1.0)
    assert calls == ["cugraph", "cugraph", "cugraph", "rapids"]


def test_theta_warns_before_unavailable_rapids(adata_neighbors, monkeypatch):
    monkeypatch.setattr(_leiden, "_ld", None)
    with (
        pytest.warns(FutureWarning, match="theta"),
        pytest.raises(NotImplementedError),
    ):
        rsc.tl.leiden(adata_neighbors, flavor="rapids", theta=1.0)


# ---------- seeds ----------


@needs_native
def test_seeds(adata_neighbors):
    """A legacy seed is used for every resolution; a Generator (or rng=int)
    gives one independent draw per resolution."""
    a = adata_neighbors.obsp["connectivities"]
    expected = {s: native(a, [0.5, 2.0], seed=s) for s in (0, 7)}
    rsc.tl.leiden(adata_neighbors, resolution=[0.5, 2.0], key_added="d")
    rsc.tl.leiden(adata_neighbors, resolution=[0.5, 2.0], random_state=7, key_added="s")
    for key, seed in (("d", 0), ("s", 7)):
        for i, res in enumerate((0.5, 2.0)):
            codes = adata_neighbors.obs[f"{key}_{res}"].cat.codes.to_numpy()
            np.testing.assert_array_equal(codes, expected[seed][i].membership)
    rng = np.random.default_rng(7)
    draws = [int(rng.integers(0, 2**32)) for _ in range(2)]
    for arg in (7, np.random.default_rng(7)):
        rsc.tl.leiden(adata_neighbors, resolution=[0.5, 2.0], rng=arg, key_added="g")
        assert "random_state" not in adata_neighbors.uns["g"]["params"]
        for res, draw in zip((0.5, 2.0), draws, strict=True):
            codes = adata_neighbors.obs[f"g_{res}"].cat.codes.to_numpy()
            np.testing.assert_array_equal(
                codes, native(a, res, seed=draw)[0].membership
            )


# ---------- input handling ----------


@needs_native
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_index_dtypes_and_placement(monkeypatch, dtype):
    """int32 / int64 indptr x indices, host (scipy: read in place where the GPU
    reads pageable memory, else copied in chunks) and device (cupyx) input are
    bitwise equal (also across a re-quantisation: gamma = 20)."""
    a = _graph("pbmc68k").astype(dtype)
    resolutions = [0.5, 1.0, 20.0]
    expected = native(cpx_sparse.csr_matrix(a), resolutions)
    monkeypatch.setattr(_leiden, "_H2D_CHUNK_BYTES", 1 << 12)  # several chunks
    for zero_copy in (True, False):
        monkeypatch.setattr(_leiden, "_HOST_ZERO_COPY", zero_copy)
        for ip in (np.int32, np.int64):
            for ix in (np.int32, np.int64):
                b = _with_index_dtypes(a, ip, ix)
                assert_same(native(b, resolutions), expected)


@needs_native
def test_quantisation_invariance():
    """fp64 data holding fp32 values quantises exactly like the fp32 data, and
    weights scaled by a power of two quantise to the same integers."""
    a = _graph("pbmc3k")
    expected = native(a)
    assert_same(native(a.astype(np.float64)), expected)
    b = a.copy()
    b.data *= np.float32(2.0**-20)
    for x, y in zip(native(b), expected, strict=True):
        np.testing.assert_array_equal(x.membership, y.membership)


@needs_native
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("zero_copy", [True, False])
def test_out_of_range_index(monkeypatch, index_dtype, zero_copy):
    """A column index outside [0, n) is rejected on every input path (an int64
    one is not wrapped into range by narrowing)."""
    a = _with_index_dtypes(_graph("pbmc68k"), np.int32, index_dtype)
    a.indices[a.indptr[1] - 1] = a.shape[0] + (2**32 if index_dtype == np.int64 else 5)
    monkeypatch.setattr(_leiden, "_HOST_ZERO_COPY", zero_copy)
    with pytest.raises(ValueError, match="column index out of range"):
        native(a)


@needs_native
@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_weights(adata_neighbors, bad):
    adjacency = adata_neighbors.obsp["connectivities"].copy()
    adjacency.data[5] = bad
    with pytest.raises(ValueError, match="finite and non-negative"):
        rsc.tl.leiden(adata_neighbors, adjacency=adjacency)
    assert "leiden" not in adata_neighbors.obs
    # without weights only the pattern counts
    rsc.tl.leiden(adata_neighbors, adjacency=adjacency, use_weights=False)


@needs_native
@pytest.mark.parametrize("device", [False, True])
def test_non_canonical_input(device):
    """Unsorted rows and duplicate entries (as stored) give the result of the
    canonical float64 matrix (duplicates summed)."""
    a = _graph("pbmc68k")
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    split = rng.random(a.nnz) < 0.2  # stored twice, the weight halved
    data = np.where(split, a.data / 2, a.data).astype(np.float32)
    rows = np.concatenate([rows, rows[split]])
    cols = np.concatenate([a.indices, a.indices[split]])
    data = np.concatenate([data, data[split]])
    order = np.lexsort((rng.random(rows.size), rows))  # rows kept, entries shuffled
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=a.shape[0]))])
    messy = sparse.csr_matrix((data[order], cols[order], indptr), shape=a.shape)
    assert not messy.has_canonical_format
    clean = messy.astype(np.float64)
    clean.sum_duplicates()
    expected = native(clean)
    if device:
        messy = cpx_sparse.csr_matrix(messy)
    assert_same(native(messy), expected)


@needs_native
def test_asymmetric_input_warns(unpatched_warn):
    a = _graph("pbmc68k")
    directed = sparse.triu(a, k=1, format="csr")
    adata = AnnData(obs=pd.DataFrame(index=[str(i) for i in range(a.shape[0])]))
    with pytest.warns(UserWarning, match=r"not symmetric; clustering A \+ A.T") as rec:
        rsc.tl.leiden(adata, adjacency=directed)
    assert rec[0].filename == __file__
    with pytest.warns(UserWarning, match="not symmetric"):
        parts = native(directed)
    assert_same(parts, native((directed + directed.T).tocsr()))


@needs_native
def test_explicit_zeros_self_loops_and_unweighted():
    """Stored zeros and the diagonal change nothing; use_weights=False equals
    weight 1 on every stored nonzero (explicit zeros stay uncounted)."""
    a = _graph("pbmc3k")
    expected = native(a)
    rng = np.random.default_rng(0)
    loops = a.tolil()
    for v in rng.choice(a.shape[0], 50, replace=False):
        loops[v, v] = rng.uniform(0.1, 2.0)
    assert_same(native(loops.tocsr()), expected)
    zeros = a.copy()
    upper = sparse.triu(a, k=1).tocoo()
    pick = rng.choice(upper.nnz, 300, replace=False)
    for v, u in zip(upper.row[pick], upper.col[pick], strict=True):
        zeros[v, u] = zeros[u, v] = 0  # explicit zeros, kept
    assert zeros.nnz == a.nnz
    dropped = zeros.copy()
    dropped.eliminate_zeros()
    assert_same(native(zeros), native(dropped))
    ones = dropped.copy()
    ones.data[:] = 1
    assert_same(native(zeros, use_weights=False), native(ones))


@needs_native
@pytest.mark.parametrize(
    "adjacency",
    [
        sparse.csr_matrix((5, 5), dtype=np.float32),
        sparse.identity(5, dtype=np.float32, format="csr"),  # diagonal only
        sparse.csr_matrix(
            (np.zeros(2, np.float32), ([0, 1], [1, 0])), shape=(5, 5)
        ),  # explicit zeros
    ],
    ids=["empty", "diagonal", "zeros"],
)
def test_no_counted_edges(adjacency):
    adata = AnnData(obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]))
    rsc.tl.leiden(adata, adjacency=adjacency, resolution=[0.5, 1])
    for key in ("leiden_0.5", "leiden_1"):
        assert adata.obs[key].tolist() == ["0", "1", "2", "3", "4"]
    assert adata.uns["leiden"]["modularity"] == [0.0, 0.0]


@needs_native
def test_initial_membership(adata_neighbors):
    """A warm start does not lower Q; arbitrary non-negative codes are ranked
    to 0..C-1 with their order kept (equal to the compact codes)."""
    a = adata_neighbors.obsp["connectivities"]
    start = native(a, 1.0, seed=1)[0]

    def warm(init):
        return native(a, 1.0, seed=2, n_iterations=1, initial_membership=init)

    codes = start.membership.astype(np.int64) * 10**12 + 17
    assert warm(codes)[0].modularity >= start.modularity
    assert_same(warm(codes), warm(start.membership))
    assert_same(warm(codes), warm(cp.asarray(start.membership)))


@needs_native
def test_gamma_edge_cases():
    """gamma = 0 joins every connected component; a tiny gamma, a pendant
    vertex of one weight quantum and the largest gamma run deterministically."""
    a = _graph("components")
    n_components = connected_components(a, directed=False)[0]
    part = native(a, 0.0)[0]
    assert part.n_clusters == n_components
    n = a.shape[0]
    tail = sparse.csr_matrix(([1e-20, 1e-20], ([0, n], [n, 0])), shape=(n + 1, n + 1))
    b = (sparse.block_diag([a, sparse.csr_matrix((1, 1))]) + tail).tocsr()
    b = b.astype(np.float32)
    b.sort_indices()
    resolutions = [2.0**-10, 2.0**-30, 2.0**20]
    assert_same(native(b, resolutions), native(b, resolutions))


@needs_native
@pytest.mark.parametrize(
    ("dtype", "factor"),
    [(np.float64, 1e-300), (np.float64, 1e300), (np.float32, 1e-40)],
)
def test_extreme_weights(dtype, factor):
    a = _graph("pbmc68k").astype(np.float64)
    a.data *= factor
    a = a.astype(dtype)
    assert (a.data > 0).all()
    for part in native(a):
        assert _disconnected(a, part.membership) == 0


# ---------- guarantees ----------


@needs_native
@pytest.mark.parametrize("name", ["pbmc68k", "pbmc3k", "sbm", "pairs", "star"])
def test_connected_communities(name):
    a = _graph(name)
    for seed in range(3):
        for n_iterations in (1, 2, -1):
            for part in native(a, [0.1, 0.5, 1.0, 2.0, 5.0], seed, n_iterations):
                assert _disconnected(a, part.membership) == 0


@needs_native
@pytest.mark.parametrize("name", ["pbmc3k", "sbm", "components", "star"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_modularity_matches_fp64(name, dtype):
    """The reported (exact) Q equals the float64 recomputation, and the
    cuGraph flavor's exact Q of the same labels."""
    a = _graph(name).astype(dtype)
    graph, info, weighted = _leiden._ingest(a, use_weights=True, host_input=False)
    exact = _leiden._exact_modularity(graph, info, use_weights=weighted)
    for res, part in zip((0.5, 1.0, 2.0), native(a, [0.5, 1.0, 2.0]), strict=True):
        expected = _fp64_modularity(a, part.membership, res)
        assert part.modularity == pytest.approx(expected, rel=0, abs=1e-9)
        q = exact(part.membership, part.n_clusters, res)
        assert q == pytest.approx(part.modularity, rel=0, abs=1e-12)


# ---------- determinism ----------


@needs_native
def test_resolution_list_equals_single_calls():
    """A list call equals separate calls with the same seed, also across a
    re-quantisation (gamma > 16 changes the scale)."""
    a = _graph("pbmc68k")
    gammas = [0.5, 20.0, 1.0, 1.0]
    for g, part in zip(gammas, native(a, gammas), strict=True):
        assert_same([part], native(a, g))


@needs_native
def test_cross_call_workspace(monkeypatch):
    """Results never depend on workspace contents: a fresh workspace filled
    with garbage, and the workspace of another graph's call, give the bits of
    a fresh call; so does a workspace too small for the levels (rerun)."""
    a, b = _graph("pbmc3k"), _graph("sbm")
    expected = native(a, [0.5, 1.0, 2.0])
    native(b, [0.5, 1.0])
    assert_same(native(a, [0.5, 1.0, 2.0]), expected)
    empty = cp.empty

    def dirty(*args, **kwargs):
        out = empty(*args, **kwargs)
        out.view(cp.uint8)[...] = 0xA5
        return out

    _leiden._WORKSPACE_CACHE.clear()
    monkeypatch.setattr(cp, "empty", dirty)
    assert_same(native(a, [0.5, 1.0, 2.0]), expected)
    monkeypatch.setattr(cp, "empty", empty)
    monkeypatch.setattr(_leiden, "_ARENA_FACTOR", 0.05)
    assert_same(native(a, [0.5, 1.0, 2.0]), expected)


@needs_native
def test_workspace_cache(monkeypatch):
    """Under rsc's default allocator (no pool) a call leaves its buffer, up to
    the cap, to the next call, which reuses it if large enough. A rerun frees
    the buffer before it allocates a larger one."""
    assert _leiden._workspace_cache_applies()
    with cp.cuda.using_allocator(cp.get_default_memory_pool().malloc):
        assert not _leiden._workspace_cache_applies()
    live, allocs, large = {}, [], 1 << 20

    class Block:  # one cudaMalloc, tracked until it is freed
        def __init__(self, size):
            self.mem = cp.cuda.Memory(size)
            live[self.mem.ptr] = size

        def __del__(self):
            live.pop(self.mem.ptr)

    def malloc(size):
        block = Block(size)
        allocs.append((size, sum(s for s in live.values() if s >= large)))
        memory = cp.cuda.UnownedMemory(block.mem.ptr, size, block)
        return cp.cuda.MemoryPointer(memory, 0)

    monkeypatch.setattr(_leiden, "_workspace_cache_applies", lambda: True)
    cache, device, a = _leiden._WORKSPACE_CACHE, cp.cuda.Device().id, _graph("pbmc3k")
    cache.clear()
    with cp.cuda.using_allocator(malloc):
        expected = native(a)
        cached = cache[device].nbytes
        allocs.clear()
        assert_same(native(a), expected)
        native(_graph("sbm"))
        assert not [s for s, _ in allocs if s >= large]  # no new workspace
        assert cache[device].nbytes == cached
        monkeypatch.setattr(_leiden, "_ARENA_FACTOR", 0.05)
        allocs.clear()
        assert_same(native(a), expected)  # reruns
        reruns = [(s, total) for s, total in allocs if s >= large]
        assert len(reruns) >= 2
        assert all(s == total for s, total in reruns)  # one workspace at a time
        monkeypatch.setattr(_leiden, "_WORKSPACE_CACHE_BYTES", 1 << 16)
        native(a)
        assert device not in cache
        assert not [s for s in live.values() if s >= large]


@needs_native
def test_streams_and_allocators():
    """The legacy NULL stream, a non-blocking stream and CuPy's memory pool
    give the same bits."""
    a = _graph("pbmc68k")
    expected = native(a, [0.5, 1.0])
    with cp.cuda.Stream(non_blocking=True) as stream:
        assert stream.ptr != 0
        assert_same(native(a, [0.5, 1.0]), expected)
    allocator = cp.cuda.get_allocator()
    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)
    try:
        assert_same(native(a, [0.5, 1.0]), expected)
    finally:
        cp.cuda.set_allocator(allocator)


# ---------- quality ----------


@needs_native
def test_quality_pbmc3k():
    """pbmc3k seeds 0-4: mean Q at least NetworKit's ParallelLeiden mean at
    gamma = 0.5 / 1 / 2, and 7-9 clusters at gamma = 1."""
    a = _graph("pbmc3k")
    q = {0.5: [], 1.0: [], 2.0: []}
    for seed in range(5):
        for g, part in zip(q, native(a, list(q), seed), strict=True):
            q[g].append(part.modularity)
            if g == 1.0:
                assert 7 <= part.n_clusters <= 9
    assert np.mean(q[0.5]) >= 0.8027
    assert np.mean(q[1.0]) >= 0.6583
    assert np.mean(q[2.0]) >= 0.5228


@needs_native
def test_recovers_planted_communities():
    """Every clique of a ring of cliques, and every SBM block, is one cluster
    (at gamma = 2: gamma = 1 merges neighbouring cliques, the resolution limit)."""
    for name, size in (("rings", 5), ("sbm", 20)):
        labels = native(_graph(name), 2.0)[0].membership
        blocks = labels.reshape(-1, size)
        assert (blocks == blocks[:, :1]).all()
        assert len(np.unique(blocks[:, 0])) == blocks.shape[0]


# ---------- the legacy cuGraph flavor ----------


def test_cugraph_arguments(adata_neighbors, monkeypatch):
    import cugraph

    seen = []
    original = cugraph.leiden

    def spy(g, **kwargs):
        seen.append(kwargs)
        return original(g, **kwargs)

    monkeypatch.setattr(cugraph, "leiden", spy)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        rsc.tl.leiden(adata_neighbors, flavor="cugraph", n_iterations=50)
    # ignored silently (the old rsc default 100 is often passed explicitly)
    assert not [w for w in record if "n_iterations" in str(w.message)]
    assert adata_neighbors.uns["leiden"]["params"]["n_iterations"] == 50
    rsc.tl.leiden(adata_neighbors, flavor="cugraph", theta=0.5, max_iter=7)
    rsc.tl.leiden(adata_neighbors, flavor="cugraph", resolution=[0.5, 1.0], rng=3)
    # n_iterations is not forwarded: cuGraph's max_iter counts levels and
    # returns the refined partition when hit
    assert seen[0]["max_iter"] == 100
    assert seen[0]["theta"] == 1.0
    assert seen[0]["random_state"] == 0
    assert seen[1]["max_iter"] == 7
    assert seen[1]["theta"] == 0.5
    rng = np.random.default_rng(3)
    assert [s["random_state"] for s in seen[2:]] == [
        int(rng.integers(0, 2**32)) for _ in range(2)
    ]


def test_cugraph_helpers():
    groups = np.array([5, 5, 3, 3, 9, 7, 7, 7])
    codes, n_clusters = _leiden._size_ordered_codes(groups)
    assert n_clusters == 4
    np.testing.assert_array_equal(codes, [1, 1, 2, 2, 3, 0, 0, 0])
    assert codes.dtype == np.int32
    # missing vertices get their own groups
    frame = pd.DataFrame({"vertex": [4, 0, 2], "partition": [1, 1, 0]})
    groups = _leiden._vertex_groups(frame, 6)
    assert groups[[0, 2, 4]].tolist() == [1, 0, 1]
    assert len({groups[1], groups[3], groups[5], 0, 1}) == 5
