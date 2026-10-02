#!/usr/bin/env python3
"""Drive the whole workflow end to end:

  INPUT (config/system.yaml: film, substrate, additional gas)
  Step 0, once   resolve_species.py         gas fragments, inert gas, elements, interfacial compounds
  round r = 0 .. N (N = rounds.max_round = 4):
    Step 1       generate_structures.py     1000 unique structures (round 0 from templates,
                                            rounds 1..N by MD with the previous NEP, run as a SLURM job)
    Step 2       submit_vasp.py             DFT labels
                 vasp_to_nep_dataset.py     append to the unique dataset, train/test over all rounds
                 submit_nep_training.py     NEP fit
    Step 3       evaluate_potential.py      Check A (test loss) and Check B (AIMD vs NEP, gas hits)
                 decide_next_step.py        all pass -> final NEP; any fail -> next round
  Check B's AIMD reference (setup_check_b.py) is submitted once, with round 0's DFT labels.

Each stage runs as its own subprocess so it stays independently runnable and
inspectable. Stops, rather than looping or papering over problems, when:
  - the decision is "sufficient"  -> final NEP in runs/final_nep/nep.txt
  - the decision is "blocked"     -> prints why (incl. round N reached without passing)
  - a stage exits non-zero, or a SLURM job fails or times out
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CHECK_B_DIR,
    FINAL_NEP_DIR,
    SPECIES_PATH,
    cluster_config,
    eprint,
    job_state,
    load_clusters_config,
    load_criteria_config,
    read_json,
    render_template,
    round_dir,
    sbatch_context,
    submit_job,
)

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_stage(script: str, args: list[str], ok_codes=(0,)) -> int:
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
    print(f"\n[orchestrator] === {' '.join(cmd[1:])} ===", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc not in ok_codes:
        eprint(f"[orchestrator] {script} exited {rc}. Stopping.")
        sys.exit(rc)
    return rc


def wait_jobs(job_ids, label: str, poll: int, max_hours: float):
    pending = set(job_ids)
    if not pending:
        return
    print(f"[orchestrator] Waiting on {len(pending)} job(s) for {label}...", flush=True)
    deadline = time.time() + max_hours * 3600
    failed = set()
    while pending and time.time() < deadline:
        for jid in list(pending):
            state = job_state(jid)
            if state == "COMPLETED":
                pending.discard(jid)
            elif state == "FAILED":
                pending.discard(jid)
                failed.add(jid)
        if pending:
            time.sleep(poll)
    if pending:
        eprint(f"[orchestrator] Timed out waiting on {label} jobs {sorted(pending)}. Stopping.")
        sys.exit(1)
    if failed:
        eprint(f"[orchestrator] {label} job(s) {sorted(failed)} FAILED -- check their slurm-*.err. Stopping.")
        sys.exit(1)


def submit_md_generation(round_num: int, cluster: str) -> str:
    """Rounds 1..N: Step 1 runs MD with the previous NEP, so it runs on a compute node."""
    cfg = cluster_config(cluster)
    log_dir = round_dir(round_num) / "generation_job"
    log_dir.mkdir(parents=True, exist_ok=True)
    ctx = sbatch_context(cfg, job_name=f"gen_r{round_num:03d}", workdir=log_dir, kind="python")
    ctx["command"] = f"scripts/generate_structures.py --round {round_num}"
    sbatch_path = log_dir / "submit.sbatch"
    sbatch_path.write_text(render_template("python.sbatch.template", ctx))
    return submit_job(sbatch_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", choices=["anvil", "aces"], required=True)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--local-md", action="store_true",
                        help="Run Step 1's MD (rounds 1..N) in this process instead of as a SLURM job.")
    args = parser.parse_args()

    common = load_clusters_config().get("common", {})
    poll = common.get("poll_interval_seconds", 60)
    max_hours = common.get("max_poll_hours", 48)
    max_round = load_criteria_config()["rounds"]["max_round"]

    if not SPECIES_PATH.exists():
        run_stage("resolve_species.py", [])

    for r in range(args.start_round, max_round + 1):
        rr = ["--round", str(r)]
        r_dir = round_dir(r)

        # Step 1: generate
        if r == 0 or args.local_md:
            run_stage("generate_structures.py", rr)
        else:
            wait_jobs([submit_md_generation(r, args.cluster)], f"round {r} MD sampling", poll, max_hours)
            if not (r_dir / "structures_manifest.json").exists():
                eprint(f"[orchestrator] Round {r} generation job finished without a manifest. Stopping.")
                sys.exit(1)

        # Step 2: label & train (Check B's AIMD reference goes in with round 0)
        run_stage("submit_vasp.py", [*rr, "--cluster", args.cluster])
        if not (CHECK_B_DIR / "job_ids.json").exists():
            run_stage("setup_check_b.py", ["--cluster", args.cluster])
        wait_jobs(set(read_json(r_dir / "vasp" / "job_ids.json")["jobs"].values()), f"round {r} DFT", poll, max_hours)
        run_stage("vasp_to_nep_dataset.py", rr)
        run_stage("submit_nep_training.py", [*rr, "--cluster", args.cluster])
        wait_jobs([read_json(r_dir / "nep_model" / "job_id.json")["job_id"]], f"round {r} NEP training",
                  poll, max_hours)

        # Step 3: evaluate (Check B needs its AIMD reference finished)
        wait_jobs(set(read_json(CHECK_B_DIR / "job_ids.json")["jobs"].values()), "Check B AIMD", poll, max_hours)
        run_stage("evaluate_potential.py", rr, ok_codes=(0, 1))
        run_stage("decide_next_step.py", rr)
        decision = read_json(r_dir / "decision.json")
        print(f"[orchestrator] Round {r}: {decision['outcome']} -- {decision['reason']}", flush=True)

        if decision["outcome"] == "sufficient":
            print(f"[orchestrator] ALL PASS. Final NEP, ready for deposition MD: {FINAL_NEP_DIR / 'nep.txt'}")
            return
        if decision["outcome"] == "blocked":
            sys.exit(1)

    eprint(f"[orchestrator] Reached max_round={max_round} without passing. Stopping.")
    sys.exit(1)


if __name__ == "__main__":
    main()
