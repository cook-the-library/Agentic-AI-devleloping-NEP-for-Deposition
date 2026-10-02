#!/usr/bin/env python3
"""Step 3 decision: read runs/round_XXX/evaluation.json and decide.

  sufficient    Check A and Check B both pass -> copy the round's nep.txt to
                runs/final_nep/ (the final NEP, ready for deposition MD)
  insufficient  any check fails -> repeat: Step 1 generates round r+1,
                weighted toward the worst-loss buckets
  blocked       a human must step in: training or the Check B AIMD hasn't
                finished, or the failing round is already max_round (N = 4)

Writes runs/round_XXX/decision.json and always exits 0 (the decision is the
output) -- callers read its "outcome" field.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FINAL_NEP_DIR, ConfigError, eprint, load_criteria_config, read_json, round_dir, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    criteria = load_criteria_config()
    r_dir = round_dir(args.round)
    eval_path = r_dir / "evaluation.json"
    if not eval_path.exists():
        raise ConfigError(f"{eval_path} not found -- run evaluate_potential.py for round {args.round} first.")
    ev = read_json(eval_path)

    failures = []
    if ev.get("status") == "training_not_complete":
        outcome, reason = "blocked", "NEP training not complete yet."
    elif ev["check_b"]["pending"]:
        outcome = "blocked"
        reason = f"Check B AIMD reference not finished for: {', '.join(ev['check_b']['pending'])}."
    else:
        failures = [f"Check A: {f}" for f in ev["check_a"]["failures"]] + \
                   [f"Check B: {f}" for f in ev["check_b"]["failures"]]
        if failures:
            outcome, reason = "insufficient", "; ".join(failures)
        else:
            outcome, reason = "sufficient", "Check A and Check B pass."

    max_round = criteria["rounds"]["max_round"]
    if outcome == "insufficient" and args.round >= max_round:
        outcome = "blocked"
        reason = (f"Round {args.round} is max_round={max_round} (N) and still fails, so no round "
                  f"{args.round + 1}. Raise rounds.max_round in config/criteria.yaml if more rounds are "
                  f"warranted. Failures: {reason}")

    if outcome == "sufficient":
        FINAL_NEP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r_dir / "nep_model" / "nep.txt", FINAL_NEP_DIR / "nep.txt")
        write_json(FINAL_NEP_DIR / "final_nep.json", {"round": args.round, "evaluation": ev})

    decision = {
        "round": args.round,
        "outcome": outcome,   # "sufficient" | "insufficient" | "blocked"
        "reason": reason,
        "failures": failures,
        "next_round": args.round + 1 if outcome == "insufficient" else None,
    }
    write_json(r_dir / "decision.json", decision)
    print(f"[decide_next_step] Round {args.round}: {outcome.upper()} -- {reason}")
    if outcome == "sufficient":
        print(f"[decide_next_step] Final NEP: {FINAL_NEP_DIR / 'nep.txt'}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[decide_next_step] CONFIG ERROR: {e}")
        sys.exit(2)
