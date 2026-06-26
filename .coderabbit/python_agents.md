# AI Code Review Guidelines - rapids_singlecell Python

**Role**: Act as a principal engineer with 10+ years experience in GPU computing and single-cell genomics. Focus ONLY on CRITICAL and HIGH issues.

**Target**: Sub-3% false positive rate. Be direct, concise, minimal.

**Context**: rapids_singlecell provides GPU-accelerated single-cell analysis methods, compatible with the scverse ecosystem (scanpy, squidpy, pertpy, decoupler). It uses CuPy, cuML, and custom CUDA kernels via nanobind.

## IGNORE These Issues

- Style/formatting (pre-commit hooks handle this)
- Minor naming preferences (unless truly misleading)
- Personal taste on implementation (unless impacts maintainability)
- Nits that don't affect functionality
- Already-covered issues (one comment per root cause)
- Type annotations in docstrings (types belong in function signatures only)

## CRITICAL Issues (Always Comment)

### Algorithm Correctness
- Logic errors in single-cell analysis implementations
- Incorrect statistical computations (mean, variance, rank tests, p-values)
- Numerical instability causing wrong results (overflow, underflow, precision loss)
- Breaking changes to algorithm behavior vs scanpy/squidpy/pertpy reference implementations
- **dtype mismatches**: float32 vs float64 silent promotion or truncation, especially between CuPy arrays and Python scalars
- **Data layout bugs**: incorrect sparse format assumptions (CSR vs CSC vs COO)

### GPU Memory Safety
- Device memory leaks (CuPy arrays not freed, circular references)
- Out-of-memory on large datasets (2M+ cells) due to unnecessary copies or intermediates
- Silent data corruption from dtype mismatches between Python and CUDA kernels
- **Passing float64 arrays to kernels templated on float32** (silent misinterpretation, not crashes)

### API Compatibility
- Breaking changes to public function signatures
- Behavior diverging from scanpy/squidpy/pertpy without justification
- Removing or renaming public functions without deprecation

### Sparse Matrix Handling
- Incorrect sparse format conversions (CSR↔CSC↔COO) causing wrong results
- Unnecessary format conversions that waste GPU memory and time
- Not handling sparse input when dense is expected (or vice versa)

## HIGH Issues (Comment if Substantial)

### Scalability
**This library must handle datasets with millions of cells and thousands of groups.** Every review should consider: "What happens when this runs on 2M cells with 10,000 categories?" Code that works on 5,000 cells can silently produce wrong results, run out of memory, or take hours at real scale.

- Unnecessary GPU↔CPU transfers (`.get()` calls in hot paths)
- Computing over all groups then filtering, instead of subsetting first
- Per-element `int(cupy_array[i])` in Python loops (triggers GPU→CPU sync per iteration)
- Unnecessary sparse format conversions (CSC→CSR when kernel supports CSC directly)
- Missing `cp.cuda.Stream` usage for concurrent operations
- Allocating large temporary arrays when in-place operations are possible
- **O(k^2) pair computation** where k is number of groups — with k=10,000 that's 100M pairs
- **Pairwise distance matrices**: never compute full n×n or k×k distance matrices. With n=2M cells a pairwise distance matrix is 16TB in float32. Even at the group level, k=10,000 groups means 100M pairs. Always use blocked/streaming approaches, kernel fusions, or approximate methods instead.
- **Dense matrices where sparse would suffice** — a 2M x 30,000 dense float32 matrix is ~224GB

### Numerical Stability at Scale
At millions of cells, numerical edge cases that "never happen" on small data become routine:

- **Extreme z-scores**: Wilcoxon/t-test z-scores of |z| > 50 are common with 2M cells. Standard normal CDF (`ndtr`, `norm.sf`) underflows to 0.0 around z=37. Use `erfc(|z| / sqrt(2))` which is accurate to ~1e-300.
- **Variance underflow/overflow**: with millions of observations, intermediate products in variance calculations can overflow float32. Use float64 for statistical accumulators.
- **Tie correction at scale**: rank-based tests with millions of tied values produce extreme tie correction factors. Ensure these don't overflow or lose precision.
- **Near-singular matrices**: PCA/SVD on datasets with near-zero variance genes. Check for degenerate eigenvalues.
- Division by zero/near-zero without epsilon checks
- Accumulation errors in iterative algorithms
- Unsafe casting between numeric types
- **CuPy NEP 50 type promotion**: mixing Python float (float64) scalars with float32 arrays

**The general principle**: if a function computes statistics, distances, or p-values, assume it will see extreme values at scale. Choose numerically robust formulations (erfc over ndtr, log-space over linear, float64 over float32 for accumulators) even if the simpler version "works" on test data.

### Input Validation
- Missing checks for required `.obs`/`.var` columns in AnnData
- Not handling edge cases (zero cells in a group, single gene, empty selections)
- Using truthiness checks (`if layer_key:`) instead of `is None` / `is not None` for optional parameters

### Test Quality
- Missing validation of numerical correctness (only checking "runs without error")
- Missing comparison with scanpy/squidpy/pertpy reference implementations
- Missing edge case coverage (empty groups, single cell, all-zero genes)
- **Using external datasets** (tests must use synthetic data or bundled fixtures)
- Missing tests for both sparse and dense inputs
- Missing tests for Dask-backed AnnData where applicable
- **Never change test tolerances without justification** (state current, proposed, actual error, and reason)

### Documentation
- Missing or incorrect docstrings for public functions
- Parameters not documented
- Missing notes about GPU-specific behavior differences

### Security
- Unsafe deserialization of data files
- Missing bounds checking allowing resource exhaustion

### Magic Numbers
- Hard-coded numeric literals (128, 256, 512, 1024, etc.) in kernel configurations, thresholds, or tile sizes without named constants
- Use descriptive constants: `BLOCK_SIZE = 256`, `SHARED_MEM_THRESHOLD = 48 * 1024`
- Tile sizes, block dimensions, and heuristic thresholds must all be named

### Missing Kernel Error Checking
- After calling nanobind CUDA kernel wrappers from Python, the next CuPy operation may silently consume a pending CUDA error
- After RawKernel launches, call `cp.cuda.runtime.getLastError()` to surface launch failures immediately (e.g., shared memory overflow, invalid grid dimensions)
- This is especially important in development and testing — a kernel that silently fails produces garbage results that look like algorithm bugs

## MEDIUM Issues (Comment Selectively)

- Edge cases not handled (empty AnnData, single observation)
- Deprecated API usage
- Minor inefficiencies in non-critical code paths

## Review Protocol

1. **Algorithm correctness**: Does the implementation match the reference (scanpy/squidpy/pertpy)? Numerical stability?
2. **GPU memory**: Any leaks? Unnecessary copies? dtype mismatches?
3. **Sparse handling**: Correct format used? Unnecessary conversions?
4. **Performance**: Unnecessary CPU↔GPU transfers? Computing over all groups instead of subsetting?
5. **API compatibility**: Breaking changes? Consistent with scverse conventions?
6. **Input validation**: AnnData fields checked? Edge cases handled?
7. **Ask, don't tell**: "Have you considered X?" not "You should do X"

## Review the Spirit, Not Just the Letter

Before commenting, understand **what the PR is trying to achieve** and evaluate whether the approach serves that goal. This is a high-performance GPU library — many patterns that look "wrong" by general Python standards exist for good reasons:

- A verbose loop might exist to **avoid a costly sparse format conversion**. Don't suggest "simplifying" it with a scipy call that triggers CSC→CSR.
- A RawKernel might duplicate logic that CuPy already provides. It probably exists because the **CuPy version requires a format conversion or extra pass** that the kernel avoids. Don't suggest removing it.
- Code that looks "over-optimized" (manual tiling, fused kernels, pre-computed indices) is often the **result of profiling on real data**. Don't suggest "cleaner" alternatives without understanding the performance implications.
- A function might deviate from scanpy's implementation. If the deviation is **intentional** (numerical stability, GPU efficiency, precision), that's not a bug.

**Ask yourself**: Does this comment help the PR achieve its goal, or does it push the code toward a different goal (cleanliness, convention, simplicity) at the expense of the actual intent?

## Quality Threshold

Before commenting, ask:
1. Is this actually wrong/risky, or just different?
2. Would this cause a real problem (wrong results, crash, OOM)?
3. Does this comment add unique value?
4. Does this comment respect what the PR is trying to accomplish?

**If no to any: Skip the comment.**

## Output Format

- Use severity labels: CRITICAL, HIGH, MEDIUM
- Be concise: One-line issue summary + one-line impact
- Provide code suggestions when you have concrete fixes
- No preamble or sign-off

## Examples to Follow

**CRITICAL** (dtype mismatch):
```text
CRITICAL: float64 array passed to float32 kernel

Issue: `X.astype(cp.float32)` missing before kernel call; X may be float64
Why: Nanobind kernel templated on float32 silently misinterprets float64 data
Impact: Garbage results, not crashes

Suggested fix:
X = X.astype(cp.float32, copy=False)
```

**CRITICAL** (group subsetting):
```text
CRITICAL: Computing over all groups then filtering results

Issue: Pairwise distances computed for all k=10000 groups, then filtered to requested 3
Why: Wastes GPU memory and compute time (3333x more work than needed)

Suggested fix:
needed = set(groups_list) | set(reference_groups)
mask = adata.obs[groupby].isin(needed)
adata_sub = adata[mask]
```

**HIGH** (GPU sync in loop):
```text
HIGH: Per-element GPU→CPU sync in Python loop

Issue: `int(cupy_array[i])` inside loop triggers sync per iteration
Why: With k=10000, that's ~5 seconds of sync overhead

Suggested fix:
values = cupy_array.get()  # Single transfer
for i in range(len(values)):
    v = int(values[i])
```

**HIGH** (None check):
```text
HIGH: Truthiness check on optional parameter

Issue: `if layer_key:` instead of `if layer_key is not None:`
Why: Empty string or 0 would incorrectly be treated as None

Suggested fix:
if layer_key is not None:
```

**HIGH** (p-value precision loss):
```text
HIGH: Using normal CDF/SF instead of erfc for p-values

Issue: `p = 2 * ndtr(-abs(z))` or `p = 2 * norm.sf(abs(z))` loses precision for extreme z-scores
Why: ndtr/sf returns 0.0 for z > ~37, while erfc is accurate to ~1e-300
Impact: Highly significant genes in large datasets get p=0.0 instead of correct tiny p-values

Bad:
from cupyx.scipy.special import ndtr
p_values = 2.0 * ndtr(-cp.abs(z))  # Underflows to 0.0 for large |z|

Good:
from cupyx.scipy import special as cupyx_special
p_values = cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))  # Accurate in tails
```

**HIGH** (float32 z-scores):
```text
HIGH: Statistical test computed in float32

Issue: z-scores and p-values computed in float32 instead of float64
Why: float32 has ~7 decimal digits of precision; z-scores beyond ~10 lose meaning,
     and erfc/ndtr on float32 z-scores gives useless p-values for significant genes
Impact: Rankings of highly significant genes become arbitrary

Suggested fix:
# Ensure float64 for statistical computations
variance = variance.astype(cp.float64)
z = diff / cp.sqrt(variance)
```

**HIGH** (tolerance change without justification):
```text
HIGH: Test tolerance loosened without explanation

Issue: atol changed from 1e-7 to 1e-3 without documenting why
Why: May mask real numerical regression
Consider: Document the actual error and why the increase is acceptable
```

## Examples to Avoid

**Boilerplate** (avoid):
- "Single-cell Analysis: Highly variable genes are important for..."
- "GPU Computing: CuPy provides NumPy-compatible GPU arrays..."

**Subjective style** (ignore):
- "Consider using a list comprehension here"
- "This function could be split into smaller functions"
- "Prefer f-strings over .format()"

---

## Project-Specific Considerations

**CuPy RawKernels** (`_kernels/` directories):
- Kernel code is defined as Python strings compiled at runtime
- Review launch configurations (grid, block dimensions)
- Check shared memory usage vs device limits
- Verify dtype consistency between Python arrays and kernel parameters
- `@cp.fuse` with Python float scalars: CuPy 14 (NumPy 2.0 rules) keeps float32, CuPy 13 promotes to float64

**Nanobind CUDA Extensions** (`_cuda/` directory):
- C++ kernels compiled at build time via CMake/scikit-build-core
- Changes to `.cu`/`.cuh` files require rebuild (`uv pip install -e .`)
- Template parameter `T` must match the array dtype passed from Python
- Type stubs (`.pyi`) are auto-generated

**AnnData Integration**:
- Functions receive `AnnData` objects and operate on `.X`, `.layers`, `.obsm`, `.obsp`
- Respect layer/obsm key parameters
- Store results in appropriate AnnData slots (`.var`, `.uns`, `.obsm`)
- Support both in-memory and backed (Dask) AnnData

**Sparse Matrix Support**:
- Most functions should handle both dense (`cp.ndarray`) and sparse (`cupyx.scipy.sparse`) input
- Prefer CSR for row-wise operations, CSC for column-wise
- Avoid unnecessary format conversions (CSC→CSR) — write kernels that support the input format
- RawKernels exist specifically to avoid costly format conversions

**Multi-GPU Support**:
- Follow the 4-phase pattern: Split → Transfer → Launch → Gather
- Never interleave chunks across devices
- Single-GPU is a special case (early return when `n_devices == 1`)

**Dask Integration**:
- Some functions support Dask-backed AnnData for out-of-core processing
- Test both eager (CuPy) and lazy (Dask) paths

---

## Common Bug Patterns

### 1. dtype Promotion Issues
**Pattern**: Mixing Python float scalars with float32 CuPy arrays

**Red flags**:
- `array * 2.0` where array is float32 (may promote to float64)
- `cp.fuse` kernels with Python float arguments
- Not casting scalars to `array.dtype.type(value)`

### 2. Group Computation Waste
**Pattern**: Computing over all groups then filtering results

**Red flags**:
- Full pairwise computation followed by DataFrame `.loc[groups]`
- Building all-pairs index before checking which groups are requested
- Missing early subsetting of AnnData to requested groups

### 3. Sparse Format Mismatch
**Pattern**: Incorrect assumptions about sparse matrix format

**Red flags**:
- Calling `.tocsr()` without checking if already CSR
- Kernel assuming CSR but receiving CSC
- Format conversion in hot paths (inside loops)

### 4. GPU Sync Overhead
**Pattern**: Individual element access on GPU arrays from Python

**Red flags**:
- `cupy_array[i]` in a Python loop
- `int(cupy_scalar)` repeated many times
- `.get()` calls inside loops instead of bulk transfer

### 5. Numerical Approaches That Break at Scale
**Pattern**: Using formulations that work on small data but fail with millions of cells

**Red flags**:
- `ndtr(-abs(z))` or `norm.sf(abs(z))` instead of `erfc(abs(z) / sqrt(2))` — underflows at z~37
- `2 * (1 - ndtr(abs(z)))` — catastrophic cancellation when ndtr ≈ 1
- z-scores, p-values, or variance accumulators in float32 instead of float64
- Dense distance/similarity matrices for k=10,000 groups (100M entries)
- Allocating `n_cells × n_genes` dense intermediates (2M × 30K = 224GB in float32)
- Sorting or unique operations on arrays of size O(n_cells × n_groups)

**The principle**: At 2M cells with 10K categories, every "rare" edge case becomes routine — extreme z-scores (|z| > 50), near-zero variances, massive tie counts, near-singular matrices. Choose formulations that are robust across the full range, not just correct on textbook examples. If a simpler formulation breaks at scale, the robust one is the correct one.

### 6. None Check Errors
**Pattern**: Using truthiness instead of identity checks for optional parameters

**Red flags**:
- `if param:` where param could be 0, empty string, or empty array
- `if not param:` for optional parameters
- Missing distinction between "not provided" and "provided as falsy value"

---

## Code Review Checklists

### When Reviewing Preprocessing Functions
- [ ] Does the function handle both sparse and dense input?
- [ ] Are results stored in the correct AnnData slot?
- [ ] Is the computation in-place when `copy=False`?
- [ ] Are dtype conversions explicit (no silent float64 promotion)?
- [ ] Does behavior match scanpy for the same parameters?

### When Reviewing Kernel Wrappers
- [ ] Is the array dtype checked before passing to the kernel?
- [ ] Are launch configurations (grid, block) computed correctly?
- [ ] Is shared memory usage within device limits?
- [ ] Are results transferred back correctly?

### When Reviewing Statistical Tests
- [ ] Are p-values computed correctly (two-sided vs one-sided)?
- [ ] Are ties handled properly in rank tests?
- [ ] Is multiple testing correction applied where expected?
- [ ] Do results match scanpy/scipy for the same inputs?

### When Reviewing Multi-GPU Code
- [ ] Is work split without interleaving across devices?
- [ ] Are streams properly managed per device?
- [ ] Does single-GPU case work (early return)?
- [ ] Are results gathered and combined correctly?

### When Reviewing Tests
- [ ] Are numerical results compared with reference implementations?
- [ ] Are edge cases tested (empty, single cell, all-zero)?
- [ ] Are both sparse and dense inputs tested?
- [ ] Are datasets synthetic (no external dependencies)?
- [ ] Are tolerance changes justified with actual error values?

---

**Remember**: Focus on correctness and performance. Catch real bugs (wrong results, OOM, dtype mismatches), ignore style preferences. For rapids_singlecell: numerical correctness vs reference implementations and GPU memory efficiency are paramount.
