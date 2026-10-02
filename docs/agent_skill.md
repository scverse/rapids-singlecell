(agent-skill)=
# Agent skill

Every release ships a version-matched, model-neutral analysis skill for coding agents.
It teaches the agent the rapids-singlecell API and current best practices.

```{warning}
The agent skill is new and experimental.
Its content, installer, and commands may change substantially between releases.
```

## 1. Install the skill

Copy the bundled skill into your agent's personal skill directory:

`````{tab-set}
````{tab-item} Claude Code
```bash
rapids-singlecell-install-skills --agent claude
```
Installs to `$CLAUDE_CONFIG_DIR/skills/` (default `~/.claude/skills/`).
````
````{tab-item} Codex
```bash
rapids-singlecell-install-skills --agent codex
```
Installs to `$CODEX_HOME/skills/` (default `~/.codex/skills/`).
````
````{tab-item} Claude Science
```bash
rapids-singlecell-install-skills --agent claude-science
```
Installs into the active organization from `~/.claude-science/active-org.json`.
````
````{tab-item} Other agents
```bash
rapids-singlecell-install-skills --agent agents
# or any custom location
rapids-singlecell-install-skills --dest /other/skills/rapids-singlecell
```
`--agent agents` installs to `~/.agents/skills/`.
````
`````

| Option | Effect |
|---|---|
| `--check` | Compare the installed copy with the bundle in the active package |
| `--force` | Replace a differing copy, e.g. after upgrading rapids-singlecell |
| `--print-path` | Print the location of the bundled skill |

## 2. Check the GPU environment

Before starting Jupyter, verify RMM, GPU/CUDA execution, the RSC import, native extensions, and a representative RSC kernel:

```bash
rapids-singlecell-check-kernel
rapids-singlecell-check-kernel --mode managed  # planned oversubscription
```

The preflight is disposable, so configure RMM again at the start of the notebook.

## Bootstrap from a standalone skill

````{tip}
If the agent only has the skill but RSC is not installed yet, create a fresh environment
from the matching definition on `main`
([CUDA 13](https://github.com/scverse/rapids_singlecell/blob/main/conda/rsc_rapids_26.08_cuda13.yml),
[CUDA 12](https://github.com/scverse/rapids_singlecell/blob/main/conda/rsc_rapids_26.08_cuda12.yml)):

```bash
CONDA_CHANNEL_PRIORITY=flexible mamba env create --name rsc \
  --file /path/to/rsc_rapids_26.08_cuda13.yml
```
````
