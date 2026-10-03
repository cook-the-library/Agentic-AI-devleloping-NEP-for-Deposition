# Step 3 · Evaluate and decide

## Check B reference — `scripts/setup_check_b.py --cluster ...` (once, with round 0)

**Writes:** `runs/check_b/{substrate,film,film_on_substrate}/`, `check_b/job_ids.json`

For each target, the slab (or interface, at the middle of `interface_gap_A`) is built as
in Step 1 and kept pristine. The projectile (`check_b.projectile`, default the first gas
molecule, else the first inert atom) starts `start_height_A` (4 Å) above the top atom over
the cell centre and moves straight down at `impact_energy_eV` (10 eV); the bottom
`fixed_bottom_A` (2 Å) is fixed. VASP runs NVE AIMD (`IBRION = 0`, `SMASS = -3`,
`NSW = 300`, `POTIM = 0.5` fs, `ISPIN = 2`), with the initial velocities in POSCAR (Å/fs).
Systems over 120 atoms are refused.

## Checks — `scripts/evaluate_potential.py --round r`

Runs the round's `nep.txt` on CPU (calorine) and writes `evaluation.json` and
`check_b_series.json`.

- **Check A · held-out test loss** on `test.xyz`: RMSE of energy (meV/atom), force
  components (meV/Å) and virial (meV/atom, only frames with stress), overall and per bucket.
  Pass: energy ≤ 10, force ≤ 250, stress ≤ 250 (`check_a`). Each bucket's **score** is its
  worst RMSE/threshold ratio; Step 1 of the next round samples more where it is high.
- **Check B · AIMD vs NEP energy trend**: the NEP's energy on every AIMD frame. Both curves
  are shifted to start at 0 and compared per atom. Pass: RMSE ≤
  `energy_trend_rmse_meV_per_atom_max` (10 meV/atom) for every target. A target whose AIMD
  hasn't finished is `pending`.

## Decision — `scripts/decide_next_step.py --round r`

| Outcome | When | Next |
|---|---|---|
| `sufficient` | Check A and Check B both pass | `nep.txt` is copied to `runs/final_nep/` with `final_nep.json`. Done. |
| `insufficient` | A check fails and r < `max_round` (4) | Step 1 for round r + 1. |
| `blocked` | NEP training not finished; Check B AIMD not finished; or round N still fails | A human decides. The reason is in `decision.json`. |

See `references/reading_results.md` for how to read a failure and what to change.
