# Agentic AI: NEP-Driven Deposition Optimization

Closed-loop workflow that generates structures, runs VASP AIMD, trains a GPUMD NEP
(neuroevolution potential), evaluates it against energy/force RMSE, thermal
conductivity (kappa) and thermal boundary conductance (TBC) criteria, loops back
for more training data when the potential is insufficient, then uses the trained
NEP to simulate and optimize deposition conditions on Purdue Anvil or TAMU ACES.

Migrated from the `nep-deposition-workflow` Claude skill. Full stage-by-stage
docs are in [`SKILL.md`](SKILL.md); cluster setup notes are in
[`references/hpc_notes.md`](references/hpc_notes.md).

## Quick start

```bash
pip install -r requirements.txt
# Fill in every FILL_ME_IN in config/clusters.yaml and config/experiment_correlations.yaml,
# and review thresholds in config/criteria.yaml.
python scripts/agentic_orchestrator.py --cluster anvil --max-rounds 5
```

## Layout

| Path | Contents |
| --- | --- |
| `scripts/` | One script per stage, plus `agentic_orchestrator.py` and shared helpers in `_common.py` |
| `config/` | Cluster, evaluation-criteria and experiment configs (placeholders to fill in) |
| `templates/` | SLURM `.sbatch` templates for VASP, NEP training and deposition runs |
| `references/` | HPC notes for Anvil vs ACES |

Outputs go to `runs/` and `deposition/`, which are git-ignored.
