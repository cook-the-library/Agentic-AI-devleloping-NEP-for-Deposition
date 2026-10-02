# Agentic AI for NEP Development in Deposition

Neuroevolution potentials (NEP) trained in rounds that grow their own dataset until
every check passes, on Purdue Anvil or TAMU ACES.

1. **Input**: the film, the substrate and an additional gas.
2. **Step 0** (once): resolve the gas species (fragments, inert gas) and the solid ones
   (elements, interfacial compounds).
3. Then, for rounds r = 0 … N (at most N = 4):
   - **Step 1**: generate 1000 unique structures. Round 0 builds them from crystal
     templates; later rounds run MD with the previous NEP.
   - **Step 2**: label them with DFT, add them to the unique dataset, and fit the NEP
     on every round's data.
   - **Step 3**: run Check A (held-out test loss) and Check B (AIMD vs NEP energy trend
     while gas hits the substrate, the film, and the film on substrate). If both pass,
     this is the **final NEP**, ready for deposition MD. If either fails, repeat the
     round.

Full step-by-step docs are in [`SKILL.md`](SKILL.md); cluster setup notes are in
[`references/hpc_notes.md`](references/hpc_notes.md).

## Quick start

```bash
pip install -r requirements.txt
# Fill in every FILL_ME_IN in config/system.yaml and config/clusters.yaml,
# review config/dft.yaml, config/sampling.yaml and config/criteria.yaml,
# and export VASP_PP_PATH.
python scripts/agentic_orchestrator.py --cluster anvil
```

## Layout

| Path | Contents |
| --- | --- |
| `scripts/` | One script per step, plus `agentic_orchestrator.py` and shared helpers in `_common.py` |
| `config/` | `system.yaml` (input), `sampling.yaml` (Step 1), `dft.yaml` (Step 2), `criteria.yaml` (Step 3), `clusters.yaml` |
| `templates/` | SLURM `.sbatch` templates for VASP, NEP training and Step 1 MD sampling |
| `references/` | HPC notes for Anvil vs ACES |

Outputs go to `runs/` (git-ignored); the final potential is `runs/final_nep/nep.txt`.
