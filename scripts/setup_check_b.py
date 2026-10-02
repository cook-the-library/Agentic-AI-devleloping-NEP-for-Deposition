#!/usr/bin/env python3
"""Check B reference (run once, alongside round 0's DFT labels): AIMD of a gas
projectile hitting a substrate, a film, and a film on substrate.

For each target in config/criteria.yaml's check_b.targets this builds the
slab (or interface) the same way Step 1 does, places the projectile
start_height_A above the top atom moving straight down with impact_energy_eV,
holds the bottom fixed_bottom_A of the slab fixed, and writes an NVE VASP
AIMD run (initial velocities in POSCAR) to runs/check_b/<target>/.

Every round, evaluate_potential.py evaluates that round's NEP on the AIMD
frames and compares the two energy-vs-time curves.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CHECK_B_DIR,
    SPECIES_PATH,
    ConfigError,
    cluster_config,
    dict_to_atoms,
    eprint,
    load_criteria_config,
    load_sampling_config,
    read_json,
    require_ase,
    require_filled,
    write_json,
)
from generate_structures import Builder
from submit_vasp import incar_for_family, load_dft_config, poscar_symbols, submit_batches, write_incar, write_potcar


def projectile(species: dict, name: str | None):
    from ase import Atoms

    gas = species["gas"]
    pool = {**gas["molecules"], **gas["fragments"]}
    if name:
        if name in pool:
            return dict_to_atoms(pool[name])
        if name in gas["inert"]:
            return Atoms(name, positions=[[0, 0, 0]])
        raise ConfigError(f"check_b.projectile '{name}' is not a gas species from Step 0.")
    if gas["molecules"]:
        return dict_to_atoms(next(iter(gas["molecules"].values())))
    if gas["inert"]:
        return Atoms(gas["inert"][0], positions=[[0, 0, 0]])
    raise ConfigError("Check B needs a gas species, but config/system.yaml has no gas or inert gas.")


def build_system(builder: Builder, target: str, proj, cb: dict):
    from ase.constraints import FixAtoms

    slab = builder.interface() if target == "film_on_substrate" else builder.add_slab_vacuum(builder.slab(target))
    top = slab.positions[:, 2].max()
    cell = slab.get_cell().copy()
    cell[2][2] = max(cell[2][2], top + cb["start_height_A"] + 8.0)
    proj = proj.copy()
    proj.positions -= proj.positions.mean(axis=0)
    centre = 0.5 * (cell[0] + cell[1])
    proj.positions += [centre[0], centre[1], top + cb["start_height_A"]]
    speed = math.sqrt(2 * cb["impact_energy_eV"] / proj.get_masses().sum())  # Angstrom/(ASE time unit)
    a = slab + proj
    a.set_cell(cell)
    a.set_pbc(True)
    v = np.zeros((len(a), 3))
    v[len(slab):, 2] = -speed
    a.set_velocities(v)
    order = sorted(range(len(a)), key=lambda i: a[i].symbol)  # group species so POTCAR and velocities line up
    a = a[order]
    bottom = a.positions[[i for i in range(len(a)) if order[i] < len(slab)], 2].min()
    a.set_constraint(FixAtoms(indices=[i for i in range(len(a))
                                       if order[i] < len(slab) and a.positions[i, 2] < bottom + cb["fixed_bottom_A"]]))
    return a


def write_poscar_with_velocities(atoms, path: Path):
    from ase import units

    v = atoms.get_velocities() * units.fs  # -> Angstrom/fs
    plain = atoms.copy()
    plain.set_array("momenta", None)  # ASE would otherwise write its own velocity block
    plain.write(path, format="vasp", direct=True, sort=False)
    with open(path, "a") as f:
        f.write("Cartesian\n")
        for row in v:
            f.write(f"  {row[0]: .10f} {row[1]: .10f} {row[2]: .10f}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", choices=["anvil", "aces"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    require_ase()
    if not SPECIES_PATH.exists():
        raise ConfigError(f"{SPECIES_PATH} not found -- run Step 0 (resolve_species.py) first.")
    cfg = cluster_config(args.cluster)
    require_filled(cfg, context=f"setup_check_b.py --cluster {args.cluster}")
    cb = load_criteria_config()["check_b"]
    dft = load_dft_config()
    species = read_json(SPECIES_PATH)
    builder = Builder(species, load_sampling_config(), np.random.default_rng(0))
    proj = projectile(species, cb.get("projectile"))

    dirs = []
    for target in cb["targets"]:
        d = CHECK_B_DIR / target
        d.mkdir(parents=True, exist_ok=True)
        atoms = build_system(builder, target, proj, cb)
        write_poscar_with_velocities(atoms, d / "POSCAR")
        write_incar(d / "INCAR", incar_for_family("collision", dft, {
            "IBRION": 0, "NSW": cb["steps"], "POTIM": cb["timestep_fs"], "SMASS": -3,  # NVE
        }))
        write_potcar(d, poscar_symbols(d / "POSCAR"))
        dirs.append(d)
        print(f"[setup_check_b] {target}: {len(atoms)} atoms, {proj.get_chemical_formula()} at "
              f"{cb['impact_energy_eV']} eV -> {d}")

    job_ids = submit_batches(dirs, cfg, job_prefix="check_b_aimd", log_dir=CHECK_B_DIR / "_jobs",
                             per_job=1, dry_run=args.dry_run)
    write_json(CHECK_B_DIR / "job_ids.json", {"cluster": args.cluster, "jobs": job_ids})


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[setup_check_b] CONFIG ERROR: {e}")
        sys.exit(2)
