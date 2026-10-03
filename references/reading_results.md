# Reading results and recovering

After every round, summarise for the user from `evaluation.json` and `decision.json`:
the overall Check A RMSEs against their thresholds, the three worst buckets with their
scores, each Check B target's energy-trend RMSE, and the decision. Never report a pass
the files don't show.

## A round fails (`insufficient`)

The loop's default response, more structures where the loss is worst, is automatic. Look
further when the pattern says more data won't help:

| Pattern | Likely cause | What to do |
|---|---|---|
| One or two buckets far above the rest (score ≫ 1), others ≤ 1 | Under-sampled configurations | Let the next round reweight; optionally raise that family's weight in `config/sampling.yaml`. |
| Repulsion or dimer buckets dominate with huge errors | Very short distances the NEP cutoff/basis can't fit | Check `repulsion_distance_scale`; consider the NEP's ZBL option; discuss with the user. |
| Every bucket fails by a similar factor | Model capacity or training length | Revisit `nep.in` (neurons, generations); check `loss.out` converged. |
| Check A passes, Check B fails on one target | Collision configurations under-represented near that surface | Rounds 1…N add collision MD; check `check_b_series.json` for where the curves part (impact vs relaxation). |
| Stress RMSE high only | Few frames carry virial | Confirm `virial_families`; check the bulk/compound buckets. |

## A round is `blocked`

| Reason in `decision.json` | Action |
|---|---|
| `NEP training not complete yet` | Wait for the job, or check `nep_run.log` / `slurm-*.err` if it failed. |
| `Check B AIMD reference not finished for ...` | Wait for `runs/check_b/_jobs`; a failed AIMD needs its INCAR/POSCAR checked (e.g. too-close start height). |
| `Round N is max_round=N and still fails` | Report the trend across rounds and ask the user: raise `rounds.max_round`, change sampling or `nep.in`, or accept the best round manually. |

## A stage or SLURM job fails

The orchestrator stops and names it. Read the stage's message or the job's
`slurm-*.err` first. Queue limits, module names, licences and allocations are the
user's to fix; report them instead of retrying.
