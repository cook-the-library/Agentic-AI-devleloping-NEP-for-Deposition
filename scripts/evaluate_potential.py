#!/usr/bin/env python3
"""Stage 4: Evaluate the trained NEP against the AIMD energy comparison in
config/criteria.yaml: energy/force/virial RMSE vs the AIMD (VASP) test set.

Energy/force/virial RMSE come directly from GPUMD's own test-set output files
(energy_test.out, force_test.out, virial_test.out), which NEP writes after
training completes -- no separate calculation needed, just parsing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decide_next_step import check_accuracy
from _common import (
    ConfigError,
    cluster_config,
    eprint,
    job_state,
    load_criteria_config,
    read_json,
    require_filled,
    round_dir,
    write_json,
)

import math


def _rmse(pred, target):
    n = len(pred)
    if n == 0:
        return None
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, target)) / n)


def parse_gpumd_out_columns(path: Path):
    """GPUMD NEP *_test.out files are whitespace-columns of [predicted..., target...]
    (equal split). Returns (predicted_flat, target_flat) as flat lists of floats,
    one entry per scalar component (so force files contribute 3 entries/atom/frame)."""
    if not path.exists():
        return None, None
    predicted, target = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        vals = [float(x) for x in parts]
        half = len(vals) // 2
        predicted.extend(vals[:half])
        target.extend(vals[half:])
    return predicted, target


def compute_accuracy_metrics(model_dir: Path) -> dict:
    metrics = {}
    for kind, fname in [("energy", "energy_test.out"), ("force", "force_test.out"), ("virial", "virial_test.out")]:
        pred, tgt = parse_gpumd_out_columns(model_dir / fname)
        rmse = _rmse(pred, tgt) if pred else None
        metrics[f"{kind}_rmse"] = rmse
        metrics[f"{kind}_n_points"] = len(pred) if pred else 0
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--cluster", choices=["anvil", "aces"], required=True)
    args = parser.parse_args()

    criteria = load_criteria_config()
    cluster_cfg = cluster_config(args.cluster)
    require_filled(cluster_cfg, context=f"evaluate_potential.py --cluster {args.cluster}")

    r_dir = round_dir(args.round)
    model_dir = r_dir / "nep_model"

    job_info_path = model_dir / "job_id.json"
    if job_info_path.exists():
        info = read_json(job_info_path)
        state = job_state(info["job_id"])
        if state != "COMPLETED":
            eprint(f"[evaluate_potential] NEP training job {info['job_id']} is {state}, not COMPLETED yet. "
                   f"Nothing to evaluate.")
            write_json(r_dir / "evaluation.json", {"round": args.round, "status": "training_not_complete", "job_state": state})
            sys.exit(1)

    accuracy = compute_accuracy_metrics(model_dir)
    accuracy_failures = check_accuracy({"accuracy": accuracy}, criteria)

    evaluation = {
        "round": args.round,
        "cluster": args.cluster,
        "accuracy": accuracy,
        "accuracy_failures": accuracy_failures,
        # decide_next_step.py reads this back; per_structure_errors is left [] here
        # since GPUMD's *_test.out files don't carry structure ids by default --
        # wire this up (e.g. via NEP's per-structure dump options) if you want
        # generate_structures.py's error-biased resampling to be non-stubbed.
        "per_structure_errors": [],
    }
    write_json(r_dir / "evaluation.json", evaluation)

    print(f"[evaluate_potential] Round {args.round}: energy_rmse={accuracy.get('energy_rmse')}, "
          f"force_rmse={accuracy.get('force_rmse')}, virial_rmse={accuracy.get('virial_rmse')}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[evaluate_potential] CONFIG ERROR: {e}")
        sys.exit(2)
