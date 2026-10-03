---
name: nep-deposition-workflow
description: Run the agentic AI workflow that develops a GPUMD NEP (neuroevolution potential) for deposition — resolving gas and solid species from the film, substrate and additional gas, generating 1000 unique structures per round (crystal-template families in round 0, amorphous and 0–50 eV collision MD with the previous NEP afterwards), labelling them with VASP DFT, training the NEP on the whole unique dataset, and checking held-out test loss (Check A) and AIMD vs NEP energy trends for gas hitting a substrate, a film and a film on substrate (Check B), repeating up to round N = 4 until every check passes, on Purdue Anvil or TAMU ACES. Use when the user mentions this workflow, NEP training for deposition, VASP+NEP active learning, or Anvil/ACES job submission for this pipeline.
---

# Agentic AI for NEP Development in Deposition

Neuroevolution potentials (NEP) trained in rounds that grow their own dataset until
every check passes. Each step is a script under `scripts/`. When this skill is invoked,
Claude acts as the orchestrator: work out which step the user is at, run (or help run)
it, read its output, and follow the decision rules below.

## The workflow

```
INPUT            STEP 0 · ONCE          ROUND r = 0 … N  (max N = 4)                                   ALL PASS
Film        ──▶  Resolve species   ──▶  STEP 1 Generate ──▶ STEP 2 Label & train ──▶ STEP 3 Evaluate ──▶ Final NEP
Substrate        adds gas and solid      1000 unique         DFT labels, then          test loss and      ready for
Additional gas   compounds               structures          NEP fit                   AIMD collisions    deposition MD
                 gas: fragments,              ▲                    │                         │
                      inert gas               │      Unique dataset R0 R1 R2 … RN            │
                 solid: elements,             │      exact-duplicate check,                  │
                        interfacial           │      append · train on all                   │
                        compounds             └──────────── any check fails → repeat ────────┘
```

| Step | Script | What it does |
|---|---|---|
| INPUT | `config/system.yaml` | Film, substrate, additional gas, inert gas, interfacial compounds |
| Step 0 · once | `resolve_species.py` | Adds gas species (molecules, every fragment, inert gas) and solids (film, substrate, elemental crystals, interfacial compounds) → `runs/species.json` |
| Step 1 | `generate_structures.py` | 1000 unique structures per round (see below) |
| Step 2 | `submit_vasp.py`, `vasp_to_nep_dataset.py`, `submit_nep_training.py` | DFT labels; append to the unique dataset; NEP fit on all rounds |
| Step 3 | `evaluate_potential.py`, `decide_next_step.py` | Check A and Check B; final NEP or repeat |
| Check B reference | `setup_check_b.py` | AIMD of gas hitting a substrate, a film, a film on substrate (run once, with round 0) |
| All of it | `agentic_orchestrator.py` | Runs the loop, waiting on SLURM between steps |

Each step has a reference page with its inputs, outputs, settings and common stops:
`references/step0_resolve_species.md`, `references/step1_generate.md`,
`references/step2_label_train.md`, `references/step3_evaluate.md`; and
`references/reading_results.md` for reading a round and recovering from a failed or
blocked one. Read the page for a step before running or debugging it.

### Step 0 · resolve species (once)

From the INPUT, `resolve_species.py` adds every gas fragment of the gas molecules
(NH3 → NH2, NH, N, H), the inert gas, an elemental crystal for each non-gas element
(diatomic/atomic elements stay gas-only), the listed interfacial compounds and every
solid's melting point. It cannot predict which interfacial compounds form: it lists the
element pairs that have none, and Claude asks the user about them rather than adding any.

### Step 1 · the 1000 structures of each round

**Round 0** — built from crystal templates, no NEP yet: **bulk, slab, interface, dimer,
trimer, adsorbate**. Also sampled: strained and rattled cells, point defects,
gas molecules and fragments, interfacial compounds, short-range repulsion.

- **Pristine** (round 0 only, a few): every crystal and surface at several supercell
  sizes (`round0.pristine.supercell_sizes`), with no defects, strain or rattling.
- **Non-pristine** (everything else): solid structures (bulk, compound, slab,
  interface, the slab under an adsorbate, and the MD starting cells in later rounds)
  get 0–15 % vacancies and 0–10 % interstitials, drawn per structure
  (`round0.point_defects`). Every non-pristine structure gets small thermal rattling:
  Maxwell-Boltzmann displacements at 10–50 % of the melting point of the crystal it
  is built from (`round0.thermal_rattle`).

**Rounds 1 … N** — MD with the previous NEP: **amorphous** (melt-quench) and
**collision** (gas molecules, fragments and inert atoms hitting the substrate, the film
and the film on substrate; impacts span 0–50 eV). The round-0 families stay in,
weighted toward the worst-loss buckets of the previous round's Check A. This MD needs
compute, so the orchestrator runs Step 1 of rounds 1 … N as a SLURM job.

Every structure has at most 120 atoms (hard limit) and passes an exact-duplicate check
against the whole dataset R0..RN, so each round adds 1000 new unique structures. Family shares, sizes and MD settings are in
`config/sampling.yaml`. The usual Step 1 stop is an interface that won't fit within
`interface_max_strain` under the atom limit: ask the user whether to loosen the strain or
change the surfaces.

### Step 2 · label & train

- **DFT labels** (`submit_vasp.py`): VASP single points, settings per family from
  `config/dft.yaml` (Gamma-only and spin-polarised for vacuum boxes and radicals), packed 25
  structures per SLURM job. Each run ends `VASP_DONE` or `VASP_FAILED`.
- **Unique dataset** (`vasp_to_nep_dataset.py`): finished runs from rounds 0 … r together,
  each tagged with its family as the Check A bucket, stress kept only for fully periodic
  families, and a train/test split fixed per structure so test data stays held out.
- **NEP fit** (`submit_nep_training.py`): GPUMD `nep` with standard NEP4 settings, not
  tuned for the system.

### Step 3 · checks after every training round (`config/criteria.yaml`)

- **Check A · held-out test loss** — energy ≤ 10 meV/atom, force ≤ 250 meV/Å,
  stress ≤ 250 (virial, meV/atom). Reported overall and per bucket (family).
- **Check B · AIMD vs NEP energy trend, gas hits** a substrate, a film, a film on
  substrate. The AIMD reference runs once, with round 0 (`setup_check_b.py`): the first
  gas molecule falls straight down at 10 eV onto each pristine target, NVE for 300 × 0.5 fs.
  Each round the NEP is evaluated on those AIMD frames; both E(t) curves, shifted to start
  at zero, must agree within 10 meV/atom (`energy_trend_rmse_meV_per_atom_max`).

Both within criterion → **final NEP** (`runs/final_nep/nep.txt`). Otherwise the round is
repeated with more structures where the loss is worst, up to round N = 4
(`rounds.max_round`). `blocked` means a human must step in: training or the Check B AIMD
hasn't finished, or round N still fails.

After every round, report the Check A RMSEs against their thresholds, the three worst
buckets, each Check B target's RMSE and the decision; when a round fails or blocks, use
`references/reading_results.md` to say whether more data will help or something else
(sampling weights, `nep.in`, VASP settings) needs the user's decision.

## Before running anything

Fill in every `FILL_ME_IN` and review the defaults:

1. **`config/system.yaml`** — film and substrate formulas, crystal template files and
   surfaces; gas molecules; inert gas; templates for interfacial compounds and for any
   element ASE can't build (Step 0 tells you which); melting points of the film,
   substrate, every interfacial compound and every elemental crystal (Step 0 lists
   missing ones). Claude may look these up (web search) — say which source each value
   came from, and use the decomposition temperature for compounds that decompose
   before melting.
2. **`config/clusters.yaml`** — SLURM account, partitions, modules, executables and the
   conda env for Step 1's MD job. See `references/hpc_notes.md`.
3. **`config/dft.yaml`** — INCAR defaults, KSPACING, spin, structures per SLURM job.
   Check ENCUT/KSPACING convergence for your system.
4. **`config/sampling.yaml`** and **`config/criteria.yaml`** — family shares, MD
   settings, check thresholds and Check B's impact energy.
5. `VASP_PP_PATH` pointing at your POTCAR directory.

Nothing here fabricates materials-science inputs: unfilled config stops the step with a
message rather than being guessed.

## Running

Step by step (recommended the first time, to inspect each output):

```bash
python scripts/resolve_species.py
python scripts/generate_structures.py --round 0
python scripts/submit_vasp.py --round 0 --cluster anvil
python scripts/setup_check_b.py --cluster anvil
python scripts/vasp_to_nep_dataset.py --round 0         # once the VASP jobs finish
python scripts/submit_nep_training.py --round 0 --cluster anvil
python scripts/evaluate_potential.py --round 0           # once training and Check B AIMD finish
python scripts/decide_next_step.py --round 0
```

`submit_vasp.py`, `setup_check_b.py` and `submit_nep_training.py` take `--dry-run` to
render inputs without submitting. Or run the whole loop:

```bash
python scripts/agentic_orchestrator.py --cluster anvil
```

The orchestrator stops when the decision is `sufficient` (final NEP written), when it is
`blocked`, or when a stage or SLURM job fails, reporting why, so the human stays in the
loop for anything cluster-side (queue limits, modules, VASP convergence, licences).

## Layout of `runs/`

```
runs/
  species.json                  # Step 0
  check_b/<target>/             # Check B AIMD reference (substrate, film, film_on_substrate)
  round_000/
    structures/ structures_manifest.json   # Step 1 (family, hash per structure)
    vasp/                                  # Step 2 DFT labels (+ _jobs/ batch scripts, job_ids.json)
    nep_dataset/ train.xyz test.xyz        # Step 2, all rounds 0..r
    nep_model/ nep.txt                     # Step 2 NEP fit
    evaluation.json check_b_series.json    # Step 3 checks
    decision.json                          # Step 3 decision
  round_001/ …                             # only if round 0 failed a check
  final_nep/nep.txt                        # ALL PASS
```

## What Claude should do vs. what the human must do

Claude should: resolve species, generate structures, render and submit jobs, parse VASP
and NEP output, run the checks, make the repeat/final call from the configured
thresholds, and summarise results — including the worst-loss buckets and the Check B
E(t) curves.

Claude should NOT invent SLURM accounts, partition names, POTCAR choices, crystal
templates, or interfacial compounds. If a config value a step needs is still
`FILL_ME_IN`, stop and ask — a wrong SLURM account fails loudly, but a wrong template or
INCAR choice silently wastes allocation.
