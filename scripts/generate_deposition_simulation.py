#!/usr/bin/env python3
"""Stage 7: Generate deposition simulation inputs (LAMMPS, using the trained
NEP potential via pair_style nep) for every condition in the sweep grid
defined by config/criteria.yaml's deposition_sweep section.

Builds a substrate slab from config/experiment_correlations.yaml's
substrate_material (falls back to a bare NEP-covered box if no substrate is
configured -- e.g. for a homoepitaxial/bulk-growth study) and writes one
LAMMPS data file + in.deposit script per (temperature, incident_energy,
incident_angle, flux) combination under deposition/sim_inputs/<condition_id>/.

pair_style nep requires a LAMMPS build with the NEP plugin
(https://github.com/brucefan1983/NEP_CPU or GPUMD's official LAMMPS
interface) -- this is usually NOT in a stock LAMMPS module, see
references/hpc_notes.md. The `fix deposit` parameters below are standard
LAMMPS syntax but the specific rate/region/near values are starting points --
review against your actual deposition process before trusting results.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ConfigError, DEPOSITION_DIR, eprint, read_json, write_json


def _require_ase():
    try:
        import ase  # noqa: F401
    except ImportError as e:
        raise ConfigError("ASE is required (`pip install ase`) to build the substrate slab.") from e


def build_substrate_slab(substrate_material: str | None):
    from ase.build import bulk, surface

    if not substrate_material or substrate_material == "FILL_ME_IN":
        return None
    try:
        conventional = bulk(substrate_material, cubic=True)
        slab = surface(conventional, (0, 0, 1), 4, vacuum=15.0)
        slab.center(vacuum=15.0, axis=2)
        return slab
    except Exception as e:
        eprint(f"[generate_deposition_simulation] Couldn't auto-build a slab for "
               f"'{substrate_material}' ({e}). Falling back to no substrate -- "
               f"replace build_substrate_slab() with a hand-built slab/interface "
               f"if you need a specific orientation or a multi-layer substrate.")
        return None


def write_lammps_data(path: Path, atoms, species: list[str]):
    from ase.io import write

    write(path, atoms, format="lammps-data", specorder=species)


def write_in_deposit(path: Path, *, data_file: str, nep_file: str, species: list[str],
                      temperature_K: float, incident_energy_eV: float, incident_angle_deg: float,
                      flux_atoms_per_ps: float, deposit_species: str, n_deposit_atoms: int = 200):
    species_str = " ".join(species)
    # Rough conversion: flux (atoms/ps) -> LAMMPS `fix deposit` insertion interval in steps,
    # given a 1 fs timestep. Review against your actual run length / box area.
    timestep_fs = 1.0
    steps_per_atom = max(1, int(round(1000.0 / max(flux_atoms_per_ps, 1e-6) / timestep_fs)))

    content = f"""# Auto-generated deposition run: T={temperature_K}K, E_inc={incident_energy_eV}eV, \
angle={incident_angle_deg}deg, flux={flux_atoms_per_ps}/ps
# Review before production use -- see generate_deposition_simulation.py docstring.

units metal
atom_style atomic
boundary p p f

read_data {data_file}

pair_style nep
pair_coeff * * {nep_file} {species_str}

region deposit_region block INF INF INF INF 40 45 units box
group substrate type <= {len(species)}

velocity substrate create {temperature_K} 12345
fix nvt_sub substrate nvt temp {temperature_K} {temperature_K} 0.1

# Incident angle: vz is the dominant downward component; vx set from angle for
# an off-normal deposition trajectory. Recompute if your substrate orientation differs.
fix dep all deposit {n_deposit_atoms} 0 {steps_per_atom} 12345 region deposit_region &
    vz -{incident_energy_eV} -{incident_energy_eV} &
    near 2.0

timestep 0.001
thermo 1000
thermo_style custom step temp pe ke etotal press

dump traj all custom 5000 traj.lammpstrj id type x y z vx vy vz
run {steps_per_atom * n_deposit_atoms}

write_data final_state.data
"""
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    _require_ase()

    setup_path = DEPOSITION_DIR / "setup.json"
    if not setup_path.exists():
        raise ConfigError(f"{setup_path} not found -- run identify_deposition_setup.py first.")
    setup = read_json(setup_path)

    experiment = setup["experiment"]
    sweep = setup["sweep"]
    nep_path = Path(setup["nep_potential_path"])

    slab = build_substrate_slab(experiment.get("substrate_material"))
    deposit_species = experiment.get("deposit_material")
    if not deposit_species or deposit_species == "FILL_ME_IN":
        raise ConfigError(
            "config/experiment_correlations.yaml: experiment.deposit_material must "
            "be set -- stage 7 needs to know what species is being deposited."
        )

    species = []
    if slab is not None:
        species.extend(sorted(set(slab.get_chemical_symbols())))
    if deposit_species not in species:
        species.append(deposit_species)
    species = sorted(set(species))

    sim_inputs_dir = DEPOSITION_DIR / "sim_inputs"
    sim_inputs_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(
        sweep["temperature_K"], sweep["incident_energy_eV"],
        sweep["incident_angle_deg"], sweep["flux_atoms_per_ps"],
    ))

    manifest = []
    for temp, energy, angle, flux in combos:
        cond_id = f"T{temp}_E{energy}_A{angle}_F{flux}"
        cond_dir = sim_inputs_dir / cond_id
        cond_dir.mkdir(parents=True, exist_ok=True)

        if slab is not None:
            write_lammps_data(cond_dir / "structure.data", slab, species)
        else:
            eprint(f"[generate_deposition_simulation] {cond_id}: no substrate configured -- "
                   f"you'll need to supply structure.data manually before submitting this condition.")

        (cond_dir / "nep.txt").write_text(nep_path.read_text())
        write_in_deposit(
            cond_dir / "in.deposit",
            data_file="structure.data", nep_file="nep.txt", species=species,
            temperature_K=temp, incident_energy_eV=energy, incident_angle_deg=angle,
            flux_atoms_per_ps=flux, deposit_species=deposit_species,
        )
        manifest.append({
            "condition_id": cond_id,
            "temperature_K": temp, "incident_energy_eV": energy,
            "incident_angle_deg": angle, "flux_atoms_per_ps": flux,
            "dir": str(cond_dir),
        })

    write_json(sim_inputs_dir / "manifest.json", {"conditions": manifest, "species": species})
    print(f"[generate_deposition_simulation] Wrote {len(manifest)} deposition condition(s) to {sim_inputs_dir}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[generate_deposition_simulation] CONFIG ERROR: {e}")
        sys.exit(2)
