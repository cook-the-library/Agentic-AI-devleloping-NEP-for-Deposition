#!/usr/bin/env python3
"""Step 3: Checks after every training round (thresholds in config/criteria.yaml).

Check A -- held-out test loss: the round's NEP on test.xyz, RMSE of
  energy (meV/atom), force (meV/A) and stress (virial, meV/atom), overall and
  per bucket (config_type = Step 1 family). Each bucket's score is its worst
  RMSE/threshold ratio; Step 1 of the next round samples more where the
  score is highest.

Check B -- AIMD vs NEP energy trend while gas hits a substrate, a film and a
  film on substrate: the NEP is evaluated on the frames of the AIMD reference
  (setup_check_b.py) and the RMSE between the two E(t) curves, each shifted
  to start at zero, is compared per atom against the threshold.

Writes runs/round_XXX/evaluation.json (and check_b_series.json with the E(t)
curves). decide_next_step.py turns it into the round's decision.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common
from _common import (
    CHECK_B_DIR,
    ConfigError,
    eprint,
    job_state,
    load_criteria_config,
    read_json,
    require_ase,
    round_dir,
    write_json,
)


def _rmse(values):
    return math.sqrt(float(np.mean(np.square(values)))) if len(values) else None


def check_a(test_xyz: Path, calc, cfg: dict) -> dict:
    from ase.io import read

    thresholds = {
        "energy_rmse_meV_per_atom": cfg["energy_rmse_meV_per_atom_max"],
        "force_rmse_meV_per_A": cfg["force_rmse_meV_per_A_max"],
        "stress_rmse_meV_per_atom": cfg["stress_rmse_meV_per_atom_max"],
    }
    errs = defaultdict(lambda: {"e": [], "f": [], "s": []})
    for frame in read(test_xyz, index=":"):
        bucket = frame.info.get("config_type", "unknown")
        n = len(frame)
        e_ref = frame.info["energy"] if "energy" in frame.info else frame.get_potential_energy()
        f_ref = frame.arrays["forces"] if "forces" in frame.arrays else frame.get_forces()
        v_ref = frame.info.get("virial")
        atoms = frame.copy()
        atoms.calc = calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        for key in ("all", bucket):
            errs[key]["e"].append(1000 * (e - e_ref) / n)
            errs[key]["f"].extend((1000 * (f - f_ref)).ravel())
        if v_ref is not None:
            v = -atoms.get_stress(voigt=False) * atoms.get_volume()
            dv = 1000 * (v.ravel() - np.asarray(v_ref, dtype=float).ravel()) / n
            for key in ("all", bucket):
                errs[key]["s"].extend(dv)

    def summarise(d):
        out = {
            "n_structures": len(d["e"]),
            "energy_rmse_meV_per_atom": _rmse(d["e"]),
            "force_rmse_meV_per_A": _rmse(d["f"]),
            "stress_rmse_meV_per_atom": _rmse(d["s"]),
        }
        ratios = [out[k] / t for k, t in thresholds.items() if out[k] is not None and t]
        out["score"] = max(ratios) if ratios else None
        return out

    overall = summarise(errs.pop("all"))
    failures = [f"{k}={overall[k]:.2f} exceeds {t}" for k, t in thresholds.items()
                if overall[k] is not None and overall[k] > t]
    return {"overall": overall, "buckets": {b: summarise(d) for b, d in sorted(errs.items())},
            "thresholds": thresholds, "passed": not failures, "failures": failures}


def check_b(calc, cfg: dict, series_out: dict) -> dict:
    from ase.io import read

    limit = cfg["energy_trend_rmse_meV_per_atom_max"]
    jobs = read_json(CHECK_B_DIR / "job_ids.json").get("jobs", {}) if (CHECK_B_DIR / "job_ids.json").exists() else {}
    targets, failures, pending = {}, [], []
    for target in cfg["targets"]:
        d = CHECK_B_DIR / target
        vasp_out = d / "vasp.out"
        if not vasp_out.exists() or "VASP_DONE" not in vasp_out.read_text()[-200:]:
            state = job_state(jobs[target]) if target in jobs else "NOT_SUBMITTED"
            targets[target] = {"status": "aimd_not_finished", "job_state": state}
            pending.append(target)
            continue
        frames = read(d / "vasprun.xml", index=":", format="vasp-xml")
        n = len(frames[0])
        e_aimd = np.array([f.get_potential_energy() for f in frames])
        e_nep = []
        for f in frames:
            a = f.copy()
            a.calc = calc
            e_nep.append(a.get_potential_energy())
        e_nep = np.array(e_nep)
        trend_aimd = 1000 * (e_aimd - e_aimd[0]) / n
        trend_nep = 1000 * (e_nep - e_nep[0]) / n
        rmse = _rmse(trend_nep - trend_aimd)
        ok = rmse <= limit
        targets[target] = {"status": "pass" if ok else "fail", "n_frames": len(frames),
                           "energy_trend_rmse_meV_per_atom": rmse}
        series_out[target] = {"step": list(range(len(frames))),
                              "aimd_meV_per_atom": trend_aimd.tolist(), "nep_meV_per_atom": trend_nep.tolist()}
        if not ok:
            failures.append(f"{target}: energy trend RMSE {rmse:.2f} meV/atom exceeds {limit}")
    return {"targets": targets, "threshold_meV_per_atom": limit, "pending": pending,
            "passed": not failures and not pending, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    require_ase()
    criteria = load_criteria_config()
    r_dir = round_dir(args.round)
    model_dir = r_dir / "nep_model"

    job_info_path = model_dir / "job_id.json"
    if job_info_path.exists():
        state = job_state(read_json(job_info_path)["job_id"])
        if state not in ("COMPLETED", "UNKNOWN"):
            eprint(f"[evaluate_potential] NEP training is {state}, not COMPLETED. Nothing to evaluate.")
            write_json(r_dir / "evaluation.json", {"round": args.round, "status": "training_not_complete"})
            sys.exit(1)

    calc = _common.nep_calculator(model_dir / "nep.txt")
    a = check_a(r_dir / "nep_dataset" / "test.xyz", calc, criteria["check_a"])
    series = {}
    b = check_b(calc, criteria["check_b"], series)
    write_json(r_dir / "check_b_series.json", series)
    write_json(r_dir / "evaluation.json", {"round": args.round, "check_a": a, "check_b": b})

    o = a["overall"]
    fmt = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
    print(f"[evaluate_potential] Round {args.round} Check A: E={fmt(o['energy_rmse_meV_per_atom'])} meV/atom, "
          f"F={fmt(o['force_rmse_meV_per_A'])} meV/A, S={fmt(o['stress_rmse_meV_per_atom'])} meV/atom "
          f"-> {'PASS' if a['passed'] else 'FAIL'}")
    worst = sorted(((v["score"], k) for k, v in a["buckets"].items() if v["score"] is not None), reverse=True)[:3]
    print(f"[evaluate_potential] worst-loss buckets: {', '.join(f'{k} ({s:.2f})' for s, k in worst)}")
    print(f"[evaluate_potential] Round {args.round} Check B: "
          + ", ".join(f"{t}={v['status']}" for t, v in b["targets"].items())
          + f" -> {'PASS' if b['passed'] else ('PENDING' if b['pending'] else 'FAIL')}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[evaluate_potential] CONFIG ERROR: {e}")
        sys.exit(2)
