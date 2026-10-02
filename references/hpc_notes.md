# HPC notes: Purdue Anvil vs TAMU ACES

Both clusters use SLURM, so the stage scripts in this skill are cluster-agnostic —
everything cluster-specific lives in `config/clusters.yaml`. This file is a checklist
for filling that config in correctly. It intentionally does not hardcode account
numbers, partition names, or module versions: those change over time and per
allocation, and getting them wrong from stale documentation wastes queue time or
(worse) silently runs on the wrong allocation.

## Before your first submission on either cluster

1. **Confirm your account/allocation string.**
   - Anvil: run `myaccount` (or check the ACCESS allocations dashboard at
     https://allocations.access-ci.org) for your project code.
   - ACES: check the TAMU HPRC portal (https://hprc.tamu.edu) for your allocation
     name, or run `myproject` if available on the login node.
2. **Confirm partition/queue names with `sinfo`.** Names and node counts change as
   clusters are reconfigured — don't trust a value from an old script or a doc you
   read months ago. Run `sinfo -o "%P %a %l %D %c"` on the login node and pick the
   partition that matches the job size (CPU: VASP and Step 1 MD sampling; GPU: NEP training).
3. **Confirm module names with `module avail vasp`, `module spider gpumd` (or
   equivalent).** VASP is licensed software — you need to be part of the license
   group on that cluster before any module load will succeed. GPUMD (which provides
   the `nep` training binary) is frequently not a system module on either
   cluster; if so, build it yourself (it's a small CUDA/C++ codebase, see
   https://gpumd.org) and point `config/clusters.yaml`'s `executables:` section at
   your own binary paths instead of relying on `module load`.
4. **Confirm walltime and node/core limits for your queue** — the defaults in
   `config/clusters.yaml` (4h per VASP job, 24h NEP training, 12h Step 1 MD
   sampling) are starting points, not cluster policy. Each VASP job runs
   `structures_per_job` structures from `config/dft.yaml` back to back (25 by
   default, so 40 jobs per 1000-structure round), so size `walltime_vasp` for the
   whole batch.
5. **Create the Python environment** named in `common.conda_env` with
   `pip install -r requirements.txt`. Step 1 of rounds 1..N runs as a SLURM job
   in it and needs `calorine` to run the previous round's NEP on CPU.
6. **Set `scratch_dir`** to your actual scratch path on each cluster (Anvil:
   typically under `/anvil/scratch/<x-username>`; ACES: typically under
   `/scratch/user/<username>`) — home directories on both clusters are usually
   quota-limited and not meant for job I/O at this scale.

## Why the skill doesn't guess these values

A wrong SLURM account or partition name fails immediately and loudly at `sbatch`
time — annoying but harmless. A wrong assumption baked silently into a script
(e.g., guessing your core count, or guessing which VASP module version you're
licensed for) can either waste a chunk of your compute allocation or produce
results under the wrong settings without an obvious error. This skill's scripts
check `config/clusters.yaml` for `FILL_ME_IN` placeholders and stop with an
explicit message rather than substituting a guessed default for anything
cluster- or license-specific.

## Useful SLURM commands (same on both clusters)

- `squeue -u $USER` — see your running/pending jobs
- `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed` — check a finished job's
  outcome
- `scancel <jobid>` — cancel a job
- `sinfo` — partitions, node states, limits

`agentic_orchestrator.py` polls with `squeue`/`sacct` at the interval set by
`common.poll_interval_seconds` in `config/clusters.yaml`.
