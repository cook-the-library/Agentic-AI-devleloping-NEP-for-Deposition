#!/usr/bin/env python3
"""Step 2b: Append this round's DFT labels to the unique dataset and write
train.xyz / test.xyz over ALL rounds R0..RN ("append, train on all").

Each structure is assigned to train or test once, from its hash, so a test
structure stays a test structure in every later round and Check A always
measures held-out loss. Every frame carries config_type=<family>, which
Check A uses as its loss bucket. Stress (as virial) is kept only for the
families in config/dft.yaml's virial_families.

Only structures whose VASP run finished (VASP_DONE in vasp.out) and parsed
with energy and forces are included -- a failed calculation would poison
the training set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ConfigError, eprint, read_json, require_ase, round_dir, write_json
from submit_vasp import load_dft_config


def load_vasp_result(structure_dir: Path):
    """ase.Atoms with energy/forces/stress attached, or None if the
    calculation didn't finish or can't be parsed."""
    from ase.io import read

    vasp_out = structure_dir / "vasp.out"
    if not vasp_out.exists() or "VASP_DONE" not in vasp_out.read_text()[-200:]:
        return None
    try:
        atoms = read(structure_dir / "vasprun.xml", format="vasp-xml")
        atoms.get_potential_energy()
        atoms.get_forces()
    except Exception as e:
        eprint(f"[vasp_to_nep_dataset] {structure_dir.name}: unusable output ({e}), skipping.")
        return None
    return atoms


def to_nep_frame(atoms, family: str, keep_virial: bool):
    """Plain Atoms with energy, forces and (optionally) virial for GPUMD's extxyz."""
    import numpy as np
    from ase import Atoms

    out = Atoms(atoms.get_chemical_symbols(), positions=atoms.get_positions(),
                cell=atoms.get_cell(), pbc=True)
    out.info["energy"] = float(atoms.get_potential_energy())
    out.info["config_type"] = family
    out.arrays["forces"] = np.array(atoms.get_forces())
    if keep_virial:
        try:
            stress = atoms.get_stress(voigt=False)  # eV/A^3
            out.info["virial"] = (-stress * atoms.get_volume()).reshape(9)
        except Exception:
            pass
    return out


def is_test(structure_hash: str, test_fraction: float) -> bool:
    return int(structure_hash[:8], 16) % 10000 < test_fraction * 10000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    require_ase()
    from ase.io import write

    dft = load_dft_config()
    virial_families = set(dft.get("virial_families", []))
    train, test, counts = [], [], {}
    for r in range(args.round + 1):
        r_dir = round_dir(r, create=False)
        manifest_path = r_dir / "structures_manifest.json"
        if not manifest_path.exists():
            raise ConfigError(f"{manifest_path} missing -- round {r} was never generated.")
        done = 0
        for entry in read_json(manifest_path)["structures"]:
            atoms = load_vasp_result(r_dir / "vasp" / entry["id"])
            if atoms is None:
                continue
            frame = to_nep_frame(atoms, entry["family"], entry["family"] in virial_families)
            (test if is_test(entry["hash"], dft["test_fraction"]) else train).append(frame)
            done += 1
        counts[r] = {"labelled": done, "generated": len(read_json(manifest_path)["structures"])}

    if not train or not test:
        eprint("[vasp_to_nep_dataset] No finished VASP results (or an empty split) -- nothing to write. "
               "Check that submit_vasp.py's jobs have finished.")
        sys.exit(1)

    out_dir = round_dir(args.round) / "nep_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / "train.xyz", train, format="extxyz")
    write(out_dir / "test.xyz", test, format="extxyz")
    write_json(out_dir / "dataset_summary.json",
               {"round": args.round, "per_round": counts, "n_train": len(train), "n_test": len(test)})
    print(f"[vasp_to_nep_dataset] Round {args.round}: {len(train)} train / {len(test)} test "
          f"structures from rounds 0..{args.round} -> {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[vasp_to_nep_dataset] CONFIG ERROR: {e}")
        sys.exit(2)
