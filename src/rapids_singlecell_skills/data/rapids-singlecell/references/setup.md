# Runtime setup and API recovery

Read this file for a new or broken runtime, installation/version checks, a
missing helper, or failed preflight. Keep routine setup outside the notebook.

## Match the package and skill

- Treat the installed RSC package and skill as one versioned artifact. For a
  filesystem copy, run `rapids-singlecell-install-skills --check` with the matching
  `--agent` (`codex`, `claude`, `claude-science`, or `agents`) or exact `--dest`.
  Use `python -m rapids_singlecell_skills.install` if the script is unavailable.
- For an uploaded skill, record the active RSC version and source revision; do not
  claim a package match that cannot be checked.
- For installation help, use the official
  [installation guide](https://rapids-singlecell.readthedocs.io/en/latest/installation.html)
  or the repository's `main` Conda environments for
  [CUDA 12](https://github.com/scverse/rapids_singlecell/blob/main/conda/rsc_rapids_26.04_cuda12.yml)
  and
  [CUDA 13](https://github.com/scverse/rapids_singlecell/blob/main/conda/rsc_rapids_26.04_cuda13.yml).

## Run disposable preflight

Run `rapids-singlecell-check-kernel` before Jupyter; if unavailable, run
`python -m rapids_singlecell_skills.kernel`. Add `--mode managed` only for
intentional oversubscription. Stop on failure. Preflight does not configure the
notebook process, so initialize RMM again there before CuPy or RSC imports.

## Recover from sandbox-blocked kernel transport

- Use kernel-less execution only after preflight passes and startup logs identify
  denied Jupyter/ZMQ socket creation. `Kernel died before replying to kernel_info`
  alone is not diagnostic; investigate import, ABI, GPU, and OOM failures first.
- Use an available, tested in-process notebook executor in one fresh disposable child
  interpreter—not the agent process. Execute cells in order, stop on first error, and
  write counts, streams, rich displays, figures, and tracebacks to a notebook copy.
  Unsupported magics, widgets/comms, or async behavior are blockers. Preserve a
  scheduler-set `CUDA_VISIBLE_DEVICES`; otherwise set it in the child environment
  before any CUDA import. If no tested executor is available, report the blocker;
  claim execution only after persisted outputs are inspected.

## Discover the live API

Use a disposable process so failed imports or GPU allocations do not contaminate
the notebook:

```bash
python -m rapids_singlecell_skills.api search "<intent>"
python -m rapids_singlecell_skills.api describe <symbol> --parameter <name>
```

Request one parameter or section before `--full`. If the helper CLI and module are
unavailable, inspect the installed public callable directly:

```python
import inspect
import rapids_singlecell as rsc

call = rsc.pp.highly_variable_genes  # replace with the candidate public callable
print(inspect.signature(call))
print(inspect.getdoc(call))
help(call)
```

Consult the current official documentation next. Inspect active RSC implementation
source only when public introspection is insufficient and license compatibility has
been verified. A search miss is not proof that a capability is absent.
