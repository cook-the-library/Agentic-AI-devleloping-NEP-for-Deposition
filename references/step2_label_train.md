# Step 2 · Label & train

Three scripts, run in order once the round's structures exist.

## 2a · DFT labels — `scripts/submit_vasp.py --round r --cluster {anvil,aces}`

**Reads:** `structures_manifest.json`, `config/dft.yaml`, `config/clusters.yaml`, `$VASP_PP_PATH`
· **Writes:** `runs/round_XXX/vasp/<id>/` (INCAR, POSCAR, POTCAR), `vasp/_jobs/batch_*.sbatch`,
`vasp/job_ids.json`

- VASP single points (`IBRION = -1`, `NSW = 0`) with the `incar` defaults (ENCUT 520,
  EDIFF 1e-6, ISYM 0, k-points by `KSPACING = 0.25`, no KPOINTS file).
- Per family: `isolated_families` (dimer, trimer, gas, repulsion) use Gamma only;
  `spin_polarized_families` (those plus adsorbate, collision) set `ISPIN = 2`.
- POSCARs are species-sorted, so POTCAR is the POTCARs in POSCAR order from
  `$VASP_PP_PATH/potpaw_PBE/<El>/POTCAR`.
- Structures are packed `structures_per_job` (25) per SLURM job: 40 jobs per round, not 1000.
  Each directory ends with `VASP_DONE` or `VASP_FAILED` in `vasp.out`; one failure doesn't
  stop the batch, and a resubmitted batch skips finished directories.
- `--dry-run` writes inputs and scripts without submitting.

## 2b · Unique dataset — `scripts/vasp_to_nep_dataset.py --round r`

**Writes:** `runs/round_XXX/nep_dataset/train.xyz`, `test.xyz`, `dataset_summary.json`

- Appends: rounds 0..r together ("train on all").
- Only directories with `VASP_DONE` and a parseable `vasprun.xml` with energy and forces.
- Each frame carries `config_type = <family>`, Check A's bucket.
- Stress, written as virial, only for `virial_families` (bulk, compound, defect,
  amorphous). Slabs and vacuum boxes keep energy and forces only.
- Train/test is fixed per structure by its hash (`test_fraction` 0.1), so a test structure
  stays held out in every later round.

## 2c · NEP fit — `scripts/submit_nep_training.py --round r --cluster ...`

**Writes:** `runs/round_XXX/nep_model/` (nep.in, train/test.xyz, submit.sbatch, job_id.json);
GPUMD writes `nep.txt` there.

`nep.in` starts from standard NEP4 values (cutoff 6/4 Å, n_max 4 4, basis 8 8,
l_max 4 2 1, 30 neurons, λ1 = λ2 = 0.05, batch 1000, population 50, 100 000 generations)
with `type` set to every element in the training set. These are not tuned for any system:
if `loss.out` diverges or plateaus high, revisit them with the user.

## Claude's job here

- Before the first real submission, confirm ENCUT/KSPACING convergence, spin and the
  POTCAR set with the user; never pick POTCARs or accounts yourself.
- After the VASP jobs: count `VASP_FAILED` per family (`grep -l VASP_FAILED`). A family with
  many failures means its INCAR needs attention (e.g. ALGO, NELM, smearing). Report it and
  fix the settings before training on a skewed dataset.

## Common stops

| Message | Fix |
|---|---|
| `VASP_PP_PATH is not set` / `POTCAR not found for [...]` | Point it at the licensed POTCAR directory. |
| `Cannot proceed ... placeholders ... FILL_ME_IN` | Fill the named fields in `config/clusters.yaml`. |
| `No finished VASP results (or an empty split)` | The VASP jobs haven't finished, or all failed; check `vasp.out` / `slurm-*.err`. |
| `sbatch not found` | Run on the cluster's login node, or use `--dry-run`. |
