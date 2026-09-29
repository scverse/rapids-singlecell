# Setup and recovery

## Install

- The skill and the installed rsc package are one versioned artifact; `rapids-singlecell-install-skills --check --agent <codex|claude|claude-science|agents>` (or `--dest`) compares a copied skill with the package.
- For a new environment, use the Conda files in the repository's [`conda/`](https://github.com/scverse/rapids_singlecell/tree/main/conda) directory that match the CUDA major version, or the [installation guide](https://rapids-singlecell.readthedocs.io/en/latest/installation.html).
- Without the console scripts, run `python -m rapids_singlecell_skills.kernel` and `python -m rapids_singlecell_skills.install`.

## Preflight failures

- `rapids-singlecell-check-kernel` runs in a disposable process; `--mode managed` checks the oversubscription route.
- Fix import, ABI, driver and GPU visibility failures before starting Jupyter; the notebook must still configure RMM itself.

## Kernel-less execution

- Use it only after preflight passes and the Jupyter startup log shows denied ZMQ socket creation; `Kernel died before replying to kernel_info` alone points to import, ABI, GPU or OOM failures.
- Execute the cells in order in one fresh child interpreter, not the agent process, stop at the first error, and persist outputs, figures and tracebacks into a notebook copy.
- Keep a scheduler-set `CUDA_VISIBLE_DEVICES`; magics, widgets and async cells are blockers, and execution counts only once persisted outputs have been read.

## API discovery without the helper

```python
import inspect

import rapids_singlecell as rsc

call = rsc.pp.highly_variable_genes
print(inspect.signature(call))
print(inspect.getdoc(call))
```

- Consult the official documentation next, and the installed source only when introspection is insufficient.
- A search miss is not proof that rsc lacks a capability.
