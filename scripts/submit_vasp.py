#!/usr/bin/env python3
"""Step 2a: DFT labels -- VASP single points for this round's structures, on
Anvil or ACES.

For each structure in runs/round_XXX/structures_manifest.json this writes
INCAR (settings per family from config/dft.yaml, k-points via KSPACING),
POSCAR and POTCAR into runs/round_XXX/vasp/<id>/, then packs the directories
into SLURM jobs of config/dft.yaml's structures_per_job and submits them.
Job ids go to runs/round_XXX/vasp/job_ids.json ({structure id: job id}).

POTCAR requires licensed VASP pseudopotentials that this workflow cannot ship
or fetch. Set VASP_PP_PATH to your POTCAR directory
($VASP_PP_PATH/potpaw_PBE/<Element>/POTCAR) -- see references/hpc_notes.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CONFIG_DIR,
    ConfigError,
    cluster_config,
    eprint,
    load_yaml,
    read_json,
    render_template,
    require_filled,
    round_dir,
    sbatch_context,
    submit_job,
    write_json,
)


def load_dft_config() -> dict:
    return load_yaml(CONFIG_DIR / "dft.yaml")


def incar_for_family(family: str, dft: dict, extra: dict | None = None) -> dict:
    tags = dict(dft["incar"])
    if family in dft.get("isolated_families", []):
        tags["KSPACING"] = dft.get("isolated_kspacing", 10.0)
    if family in dft.get("spin_polarized_families", []):
        tags["ISPIN"] = 2
    tags.setdefault("IBRION", -1)  # single point
    tags.setdefault("NSW", 0)
    if extra:
        tags.update(extra)
    return tags


def write_incar(path: Path, tags: dict):
    lines = []
    for k, v in tags.items():
        if isinstance(v, bool):
            v = ".TRUE." if v else ".FALSE."
        lines.append(f"{k} = {v}")
    path.write_text("\n".join(lines) + "\n")


def poscar_symbols(poscar: Path) -> list[str]:
    """Species in POSCAR order (line 6 of a VASP 5 POSCAR)."""
    return poscar.read_text().splitlines()[5].split()


def write_potcar(structure_dir: Path, symbols: list[str]):
    pp_path = os.environ.get("VASP_PP_PATH")
    if not pp_path:
        raise ConfigError(
            "VASP_PP_PATH is not set. POTCAR files are licensed VASP "
            "pseudopotentials this workflow cannot generate or fetch -- set "
            "VASP_PP_PATH to your local pseudopotential directory (see "
            "references/hpc_notes.md) and re-run this stage."
        )
    potcar_dir = Path(pp_path) / "potpaw_PBE"
    missing = [el for el in symbols if not (potcar_dir / el / "POTCAR").exists()]
    if missing:
        raise ConfigError(f"POTCAR not found under {potcar_dir} for element(s): {missing}.")
    with open(structure_dir / "POTCAR", "w") as out:
        for el in symbols:
            out.write((potcar_dir / el / "POTCAR").read_text())


def submit_batches(dirs: list[Path], cluster_cfg: dict, *, job_prefix: str, log_dir: Path,
                   per_job: int, dry_run: bool) -> dict:
    """Pack directories into SLURM jobs; returns {directory name: job id}."""
    log_dir.mkdir(parents=True, exist_ok=True)
    job_ids = {}
    for b, start in enumerate(range(0, len(dirs), per_job)):
        chunk = dirs[start:start + per_job]
        ctx = sbatch_context(cluster_cfg, job_name=f"{job_prefix}_{b:03d}", workdir=log_dir, kind="vasp")
        ctx["dirs"] = [str(d) for d in chunk]
        sbatch_path = log_dir / f"batch_{b:03d}.sbatch"
        sbatch_path.write_text(render_template("vasp.sbatch.template", ctx))
        if dry_run:
            eprint(f"[submit_vasp] (dry-run) rendered {sbatch_path} ({len(chunk)} structures), not submitting")
            continue
        job_id = submit_job(sbatch_path)
        for d in chunk:
            job_ids[d.name] = job_id
        print(f"[submit_vasp] {sbatch_path.name}: {len(chunk)} structures as job {job_id}")
    return job_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--cluster", choices=["anvil", "aces"], required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Write inputs and render sbatch scripts but don't call sbatch")
    args = parser.parse_args()

    cfg = cluster_config(args.cluster)
    require_filled(cfg, context=f"submit_vasp.py --cluster {args.cluster}")
    dft = load_dft_config()

    r_dir = round_dir(args.round)
    manifest = read_json(r_dir / "structures_manifest.json")
    vasp_dir = r_dir / "vasp"

    dirs = []
    for entry in manifest["structures"]:
        sdir = vasp_dir / entry["id"]
        sdir.mkdir(parents=True, exist_ok=True)
        poscar = sdir / "POSCAR"
        poscar.write_text(Path(entry["path"]).read_text())
        write_incar(sdir / "INCAR", incar_for_family(entry["family"], dft))
        write_potcar(sdir, poscar_symbols(poscar))
        dirs.append(sdir)

    job_ids = submit_batches(dirs, cfg, job_prefix=f"vasp_r{args.round:03d}", log_dir=vasp_dir / "_jobs",
                             per_job=dft["structures_per_job"], dry_run=args.dry_run)
    write_json(vasp_dir / "job_ids.json", {"round": args.round, "cluster": args.cluster, "jobs": job_ids})
    print(f"[submit_vasp] Round {args.round}: {len(dirs)} structures in {len(set(job_ids.values()))} jobs on {args.cluster}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[submit_vasp] CONFIG ERROR: {e}")
        sys.exit(2)
