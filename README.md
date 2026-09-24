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

## Example: AlN on Si(111)

[NEP-AlN-Si](https://github.com/cook-the-library/NEP-AlN-Si) is a worked example of
this workflow applied to one real system. It shows how to adapt the generic stages
to a specific material:
- stage 1 builds AlN/Si seed structures, including a 5:4 coincidence AlN(0001)/Si(111) interface
- stage 7 deposits Al and N as separate streams onto Si(111), with the V/III ratio as a sweep variable
- the configs carry AlN/Si criteria

It also holds the round 0, 1 and 1.5 training scripts that were run on Anvil.
Start from it when setting up a new material system.
