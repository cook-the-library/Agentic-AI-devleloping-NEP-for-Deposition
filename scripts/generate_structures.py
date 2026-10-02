#!/usr/bin/env python3
"""Step 1: Generate one round of unique structures (default 1000).

Round 0 -- built from the crystal templates resolved in Step 0, no NEP yet:
  bulk, slab, interface, dimer, trimer, adsorbate
  plus strained and rattled cells, point defects (defect), gas molecules and
  fragments (gas), interfacial compounds (compound), short-range repulsion
  (repulsion).

Rounds 1..N -- MD with the previous round's NEP:
  amorphous  melt-quench of bulk cells
  collision  gas species (molecules, fragments, inert atoms) shot at the
             substrate, the film and the film on substrate, 0-50 eV
  The round-0 families stay in, weighted toward the buckets (families) with
  the worst test loss in the previous round's Check A.

Every solid structure (bulk, compound, slab, interface, the slab under an
adsorbate, and the starting cells of the MD) can carry vacancies and
interstitials, per config/sampling.yaml's point_defects.

No structure has more than HARD_MAX_ATOMS (120) atoms, whatever
config/sampling.yaml's max_atoms says.

Every structure passes an exact-duplicate check against this round and all
previous rounds, so the dataset R0..RN stays unique; generation continues
until the round has its full count.

Writes runs/round_XXX/structures/<id>.vasp and structures_manifest.json.
"""
from __future__ import annotations

import argparse
import functools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common
from _common import (
    SPECIES_PATH,
    ConfigError,
    dict_to_atoms,
    eprint,
    load_sampling_config,
    read_json,
    require_ase,
    round_dir,
    structure_hash,
    write_json,
)

HARD_MAX_ATOMS = 120
ROUND0_FAMILIES = ["bulk", "slab", "interface", "dimer", "trimer", "adsorbate",
                   "defect", "gas", "compound", "repulsion"]
MD_FAMILIES = ["amorphous", "collision"]


class Builder:
    def __init__(self, species: dict, cfg: dict, rng: np.random.Generator):
        self.sp = species
        self.cfg = cfg
        self.r0 = cfg["round0"]
        self.rng = rng
        self.max_atoms = min(cfg["max_atoms"], HARD_MAX_ATOMS)

    # ---- species --------------------------------------------------------
    @functools.cached_property
    def film(self):
        return dict_to_atoms(self.sp["film"]["structure"])

    @functools.cached_property
    def substrate(self):
        return dict_to_atoms(self.sp["substrate"]["structure"])

    @functools.cached_property
    def elements(self):
        return {k: dict_to_atoms(v) for k, v in self.sp["solid"]["elements"].items()}

    @functools.cached_property
    def compounds(self):
        return [dict_to_atoms(c["structure"]) for c in self.sp["solid"]["interfacial_compounds"]]

    @functools.cached_property
    def gas_species(self):
        """Molecules, fragments and inert atoms -- adsorbates and projectiles."""
        from ase import Atoms

        out = [dict_to_atoms(d) for d in self.sp["gas"]["molecules"].values()]
        out += [dict_to_atoms(d) for d in self.sp["gas"]["fragments"].values()]
        out += [Atoms(s, positions=[[0, 0, 0]]) for s in self.sp["gas"]["inert"]]
        for a in out:
            a.set_cell([0, 0, 0])
            a.set_pbc(False)
        return out

    @property
    def all_symbols(self):
        return self.sp["elements"]

    def solid_symbols(self):
        syms = set(self.film.get_chemical_symbols()) | set(self.substrate.get_chemical_symbols())
        for c in self.compounds:
            syms |= set(c.get_chemical_symbols())
        return sorted(syms)

    def available(self, family: str) -> bool:
        if family == "compound":
            return bool(self.compounds)
        if family in ("gas", "adsorbate", "collision"):
            return bool(self.gas_species)
        return True

    # ---- helpers --------------------------------------------------------
    def u(self, lo_hi):
        return float(self.rng.uniform(*lo_hi))

    def pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def rattle(self, atoms, lo_hi=None):
        stdev = self.u(lo_hi or self.r0["rattle_A"])
        atoms.positions += self.rng.normal(0.0, stdev, size=(len(atoms), 3))
        return atoms

    def strain(self, atoms, inplane_only=False):
        m = self.r0["strain_max"]
        eps = self.rng.uniform(-m, m, size=(3, 3))
        eps = 0.5 * (eps + eps.T)
        if inplane_only:
            eps[2, :] = 0
            eps[:, 2] = 0
        atoms.set_cell(atoms.get_cell() @ (np.eye(3) + eps), scale_atoms=True)
        return atoms

    def supercell(self, atoms, min_atoms=1):
        """Random repeat (i, j, k) of a periodic cell within max_atoms."""
        n = len(atoms)
        options = [(i, j, k) for i in range(1, 5) for j in range(1, 5) for k in range(1, 5)
                   if min_atoms <= n * i * j * k <= self.max_atoms]
        if not options:
            options = [(1, 1, 1)] if n <= self.max_atoms else []
        if not options:
            return None
        return atoms.repeat(self.pick(options))

    def isolate(self, atoms):
        atoms.set_pbc(False)
        atoms.set_cell([0, 0, 0])
        atoms.center(vacuum=self.cfg["isolated_vacuum_A"])
        atoms.set_pbc(True)
        return atoms

    def random_rotate(self, atoms):
        atoms.euler_rotate(*self.rng.uniform(0, 360, size=3), center="COM")
        return atoms

    def rcov(self, sym):
        from ase.data import atomic_numbers, covalent_radii

        return covalent_radii[atomic_numbers[sym]]

    # ---- slabs / interface ---------------------------------------------
    def slab(self, which: str, layers=None):
        """Slab of the film or substrate on its configured surface, a, b in
        the xy plane, c along z, no vacuum yet."""
        from ase.build import surface
        from ase.geometry import Cell

        bulk = self.film if which == "film" else self.substrate
        miller = tuple(self.sp[which]["surface"])
        layers = layers or self.r0["slab_layers"]
        while layers >= 2:
            # periodic=True misplaces atoms for conventional cells (e.g. diamond 111)
            s = surface(bulk, miller, layers, vacuum=None, periodic=False)
            if len(s) <= self.max_atoms:
                break
            layers -= 1
        if len(s) > self.max_atoms:
            raise ConfigError(f"A 2-layer {which} slab on {miller} already exceeds max_atoms={self.max_atoms}.")
        cell = s.get_cell().copy()
        if abs(cell[0][2]) > 1e-6 or abs(cell[1][2]) > 1e-6:
            raise ConfigError(f"{which} slab on {miller}: surface vectors are not in the xy plane.")
        cell[2] = [0, 0, 1.0]  # placeholder height; add_slab_vacuum sets the real one
        s.set_cell(cell)
        s.set_cell(Cell.fromcellpar(s.cell.cellpar()), scale_atoms=True)  # a along x, b in xy
        return s

    def add_slab_vacuum(self, atoms):
        z = atoms.positions[:, 2]
        atoms.positions[:, 2] -= z.min()
        cell = atoms.get_cell().copy()
        cell[2] = [0, 0, z.max() - z.min() + self.cfg["slab_vacuum_A"]]
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        return atoms

    @functools.cached_property
    def interface_match(self):
        """Smallest-strain lateral match within max_atoms: substrate repeated
        i x j, film on any small integer in-plane supercell M (so 60/120-degree
        and rotated cells can still match)."""
        import itertools

        sub, film = self.slab("substrate"), self.slab("film")
        a_s, a_f = sub.cell[:2, :2], film.cell[:2, :2]
        # room for interstitials and a gas projectile on top of the interface
        limit = self.max_atoms - self.r0["point_defects"]["max_per_structure"] \
            - max((len(g) for g in self.gas_species), default=0)
        mats = [np.array(m).reshape(2, 2) for m in itertools.product(range(-3, 4), repeat=4)]
        mats = [m for m in mats if round(np.linalg.det(m)) > 0]
        best = None
        for i1, j1 in itertools.product(range(1, 5), repeat=2):
            n_sub = len(sub) * i1 * j1
            target = np.diag([i1, j1]) @ a_s
            for m in mats:
                n = n_sub + len(film) * round(np.linalg.det(m))
                if n > limit:
                    continue
                F = np.linalg.solve(m @ a_f, target)
                if np.linalg.det(F) <= 0:
                    continue
                strain = float(np.max(np.abs(np.linalg.svd(F, compute_uv=False) - 1)))
                if best is None or (strain, n) < (best[0], best[1]):
                    best = (strain, n, (i1, j1), m)
        if best is None or best[0] > self.r0["interface_max_strain"]:
            got = f"none fits {limit} atoms" if best is None else f"best lateral strain {best[0]:.1%}"
            raise ConfigError(
                f"Cannot fit the film slab onto the substrate slab ({got}; limit "
                f"interface_max_strain={self.r0['interface_max_strain']}, {limit} atoms = max_atoms "
                f"{self.max_atoms} minus room for point defects and the largest gas species). "
                f"Raise those in config/sampling.yaml or change the surfaces in config/system.yaml.")
        eprint(f"[generate_structures] interface: substrate {best[2]} x film {best[3].tolist()}, "
               f"lateral strain {best[0]:.1%}, {best[1]} atoms")
        return best

    def interface(self):
        from ase.build import make_supercell

        _, _, (i1, j1), m = self.interface_match
        sub = self.slab("substrate").repeat((i1, j1, 1))
        P = np.eye(3, dtype=int)
        P[:2, :2] = m
        film = make_supercell(self.slab("film"), P, wrap=False)
        frac = film.get_scaled_positions(wrap=False)[:, :2] % 1.0  # wrap in-plane only
        z = film.positions[:, 2] - film.positions[:, 2].min()
        top = sub.positions[:, 2].max() + self.u(self.r0["interface_gap_A"])
        film.positions = frac[:, :1] * sub.cell[0] + frac[:, 1:2] * sub.cell[1]
        film.positions[:, 2] = z + top
        both = sub + film
        both.set_cell(sub.get_cell())
        return self.add_slab_vacuum(both)

    # ---- point defects (vacancies and interstitials) --------------------
    def insert_interstitial(self, atoms, zlim=None):
        """One extra atom at a random site at least 0.75 x (sum of covalent
        radii) from every atom; inside the slab (zlim) for surfaces."""
        if len(atoms) + 1 > self.max_atoms:
            return None
        sym = self.pick(self.all_symbols)
        r_min = 0.75 * (self.rcov(sym) + min(self.rcov(x) for x in set(atoms.get_chemical_symbols())))
        for _ in range(200):
            if zlim is None:
                p = self.rng.uniform(0, 1, size=3) @ atoms.cell
            else:
                f = self.rng.uniform(0, 1, size=2)
                p = f[0] * atoms.cell[0] + f[1] * atoms.cell[1]
                p[2] = self.u(zlim)
            trial = atoms.copy()
            trial.append(sym)
            trial.positions[-1] = p
            if trial.get_distances(len(trial) - 1, range(len(atoms)), mic=True).min() > r_min:
                return trial
        return None

    def add_point_defects(self, atoms, surface=False):
        """With probability point_defects.fraction, add 1..max_per_structure
        vacancies/interstitials. Interstitials go inside the slab for
        surfaces and fall back to a vacancy when there is no room."""
        pd = self.r0["point_defects"]
        if atoms is None or self.rng.random() >= pd["fraction"]:
            return atoms
        for _ in range(int(self.rng.integers(1, pd["max_per_structure"] + 1))):
            if self.pick(["vacancy", "interstitial"]) == "interstitial":
                z = atoms.positions[:, 2]
                new = self.insert_interstitial(atoms, (z.min(), z.max()) if surface else None)
                if new is not None:
                    atoms = new
                    continue
            if len(atoms) > 2:
                del atoms[int(self.rng.integers(len(atoms)))]
        return atoms

    # ---- round-0 families ----------------------------------------------
    def build(self, family: str):
        return getattr(self, f"make_{family}")()

    def make_bulk(self):
        base = self.pick([self.film, self.substrate, *self.elements.values()])
        a = self.add_point_defects(self.supercell(base))
        return a and self.rattle(self.strain(a))

    def make_compound(self):
        a = self.add_point_defects(self.supercell(self.pick(self.compounds)))
        return a and self.rattle(self.strain(a))

    def make_slab(self):
        s = self.slab(self.pick(["film", "substrate"]))
        reps = [(i, j) for i in range(1, 4) for j in range(1, 4) if len(s) * i * j <= self.max_atoms]
        s = self.add_point_defects(self.add_slab_vacuum(s.repeat((*self.pick(reps), 1))), surface=True)
        return self.rattle(self.strain(s, inplane_only=True))

    def make_interface(self):
        return self.rattle(self.strain(self.add_point_defects(self.interface(), surface=True), inplane_only=True))

    def _n_body(self, n, scale):
        from ase import Atoms

        syms = [self.pick(self.all_symbols) for _ in range(n)]
        pos = [np.zeros(3)]
        for k in range(1, n):
            ref = int(self.rng.integers(k))
            d = self.u(scale) * (self.rcov(syms[ref]) + self.rcov(syms[k]))
            v = self.rng.normal(size=3)
            pos.append(pos[ref] + min(d, 6.0) * v / np.linalg.norm(v))
        return self.isolate(Atoms(syms, positions=pos))

    def make_dimer(self):
        return self._n_body(2, self.r0["dimer_distance_scale"])

    def make_trimer(self):
        return self._n_body(3, self.r0["dimer_distance_scale"])

    def make_repulsion(self):
        return self._n_body(int(self.pick([2, 3])), self.r0["repulsion_distance_scale"])

    def make_gas(self):
        return self.rattle(self.isolate(self.random_rotate(self.pick(self.gas_species).copy())), (0.0, 0.1))

    def make_adsorbate(self):
        s = self.add_point_defects(self.add_slab_vacuum(self.slab(self.pick(["film", "substrate"]))), surface=True)
        mol = self.random_rotate(self.pick(self.gas_species).copy())
        if len(s) + len(mol) > self.max_atoms:
            return None
        frac = self.rng.uniform(0, 1, size=2)
        site = frac[0] * s.cell[0] + frac[1] * s.cell[1]
        mol.positions -= mol.positions.mean(axis=0)
        mol.positions[:, 2] -= mol.positions[:, 2].min()
        mol.positions += [site[0], site[1], s.positions[:, 2].max() + self.u(self.r0["adsorbate_height_A"])]
        both = s + mol
        both.set_cell(s.get_cell())
        return self.rattle(self.add_slab_vacuum(both), (0.0, 0.08))

    def make_defect(self):
        a = self.supercell(self.pick([self.film, self.substrate, *self.compounds]), min_atoms=8)
        if a is None:
            return None
        kind = self.pick(["vacancy", "substitution", "interstitial"])
        i = int(self.rng.integers(len(a)))
        if kind == "vacancy":
            del a[i]
        elif kind == "substitution":
            others = [s for s in self.solid_symbols() if s != a[i].symbol] or self.all_symbols
            a[i].symbol = self.pick(others)
        else:
            a = self.insert_interstitial(a)
            if a is None:
                return None
        return self.rattle(self.strain(a, inplane_only=False), (0.0, 0.08))

    # ---- rounds 1..N: MD with the previous NEP -------------------------
    def md_snapshots(self, family: str, calc):
        return getattr(self, f"md_{family}")(calc)

    def _pick_snapshots(self, traj):
        k = min(self.cfg["rounds_1_to_N"]["snapshots_per_run"], len(traj))
        idx = sorted(self.rng.choice(len(traj), size=k, replace=False))
        return [traj[i] for i in idx]

    def md_amorphous(self, calc):
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        c = self.cfg["rounds_1_to_N"]["amorphous"]
        a = self.add_point_defects(self.supercell(self.pick([self.film, self.substrate, *self.compounds]), min_atoms=16))
        if a is None:
            return []
        a.calc = calc
        t_melt = self.u(c["melt_temperature_K"])
        MaxwellBoltzmannDistribution(a, temperature_K=t_melt, rng=self.rng)
        dyn = Langevin(a, c["timestep_fs"] * units.fs, temperature_K=t_melt, friction=0.02, rng=self.rng)
        traj = []
        half = c["steps"] // 2
        every = max(1, c["steps"] // 50)

        def record():
            traj.append(_clean(a))

        dyn.attach(record, interval=every)
        dyn.run(half)
        for k in range(10):  # linear quench in 10 steps of temperature
            dyn.set_temperature(temperature_K=t_melt + (c["quench_to_K"] - t_melt) * (k + 1) / 10)
            dyn.run(half // 10)
        return self._pick_snapshots(traj)

    def md_collision(self, calc):
        from ase import units
        from ase.constraints import FixAtoms
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet

        c = self.cfg["rounds_1_to_N"]["collision"]
        target = self.pick(["substrate", "film", "film_on_substrate"])
        slab = self.interface() if target == "film_on_substrate" else self.add_slab_vacuum(self.slab(target))
        slab = self.add_point_defects(slab, surface=True)
        proj = self.random_rotate(self.pick(self.gas_species).copy())
        if len(slab) + len(proj) > self.max_atoms:
            return []
        cell = slab.get_cell().copy()
        top = slab.positions[:, 2].max()
        cell[2][2] = max(cell[2][2], top + c["start_height_A"] + 8.0)
        slab.set_cell(cell)
        MaxwellBoltzmannDistribution(slab, temperature_K=c["surface_temperature_K"], rng=self.rng)

        frac = self.rng.uniform(0, 1, size=2)
        site = frac[0] * cell[0] + frac[1] * cell[1]
        proj.positions -= proj.positions.mean(axis=0)
        proj.positions += [site[0], site[1], top + c["start_height_A"]]
        energy = self.u(c["energy_eV"])
        theta = math.radians(self.u((0, c["max_polar_angle_deg"])))
        phi = self.u((0, 2 * math.pi))
        direction = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), -math.cos(theta)])
        speed = math.sqrt(2 * energy / proj.get_masses().sum())  # ASE units: eV, amu, Angstrom
        proj.set_velocities(np.tile(speed * direction, (len(proj), 1)))

        a = slab + proj
        a.set_cell(cell)
        a.set_pbc(True)
        bottom = a.positions[: len(slab), 2].min()
        a.set_constraint(FixAtoms(indices=[i for i in range(len(slab))
                                           if a.positions[i, 2] < bottom + c["fixed_bottom_A"]]))
        a.calc = calc
        dyn = VelocityVerlet(a, c["timestep_fs"] * units.fs)
        traj = []
        dyn.attach(lambda: traj.append(_clean(a)), interval=max(1, c["steps"] // 50))
        dyn.run(c["steps"])
        return self._pick_snapshots(traj)


def _clean(atoms):
    a = atoms.copy()
    a.set_constraint()
    a.calc = None
    if a.has("momenta"):
        a.set_array("momenta", None)
    return a


def bucket_scores(prev_eval_path: Path) -> dict:
    """Previous round's Check A score per bucket (family): the worst ratio of
    test RMSE to its threshold. Higher = worse = sampled more."""
    if not prev_eval_path.exists():
        return {}
    buckets = read_json(prev_eval_path).get("check_a", {}).get("buckets", {})
    return {fam: b["score"] for fam, b in buckets.items() if b.get("score") is not None}


def allocate(total: int, weights: dict) -> dict:
    s = sum(weights.values())
    if total <= 0 or s <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def previous_hashes(round_num: int) -> set:
    seen = set()
    for r in range(round_num):
        m = round_dir(r, create=False) / "structures_manifest.json"
        if m.exists():
            seen.update(e["hash"] for e in read_json(m)["structures"])
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--n-structures", type=int, default=None,
                        help="Overrides structures_per_round in config/sampling.yaml")
    args = parser.parse_args()

    require_ase()
    if not SPECIES_PATH.exists():
        raise ConfigError(f"{SPECIES_PATH} not found -- run Step 0 (resolve_species.py) first.")
    cfg = load_sampling_config()
    total = args.n_structures or cfg["structures_per_round"]
    rng = np.random.default_rng([cfg.get("seed", 0), args.round])
    b = Builder(read_json(SPECIES_PATH), cfg, rng)

    r0_weights = {f: cfg["round0"]["weights"].get(f, 0) for f in ROUND0_FAMILIES if b.available(f)}
    calc = None
    if args.round == 0:
        plan = allocate(total, r0_weights)
    else:
        prev = round_dir(args.round - 1, create=False)
        calc = _common.nep_calculator(prev / "nep_model" / "nep.txt")
        scores = bucket_scores(prev / "evaluation.json")
        if not scores:
            eprint("[generate_structures] No per-bucket loss from the previous round; using base weights.")
        md_cfg = cfg["rounds_1_to_N"]
        n_md = round(total * md_cfg["md_fraction"])
        md_w = {f: md_cfg["md_weights"].get(f, 0) * scores.get(f, 1.0) for f in MD_FAMILIES if b.available(f)}
        r0_w = {f: w * scores.get(f, 1.0) for f, w in r0_weights.items()}
        plan = {**allocate(n_md, md_w), **allocate(total - n_md, r0_w)}
    eprint(f"[generate_structures] Round {args.round} plan: {plan}")

    seen = previous_hashes(args.round)
    out_dir = round_dir(args.round) / "structures"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, duplicates = [], 0
    for family, count in plan.items():
        made, attempts = 0, 0
        while made < count:
            attempts += 1
            if attempts > 20 * count + 100:
                raise RuntimeError(f"Could not generate {count} unique '{family}' structures "
                                   f"(made {made} after {attempts} attempts).")
            batch = b.md_snapshots(family, calc) if family in MD_FAMILIES else [b.build(family)]
            for atoms in batch:
                if atoms is None or made >= count or len(atoms) > b.max_atoms:
                    continue
                h = structure_hash(atoms)
                if h in seen:
                    duplicates += 1
                    continue
                seen.add(h)
                sid = f"r{args.round:03d}_{family}_{made:04d}"
                path = out_dir / f"{sid}.vasp"
                atoms.write(path, format="vasp", direct=True, sort=True)  # grouped species, so POTCAR order matches
                manifest.append({"id": sid, "family": family, "path": str(path), "hash": h,
                                 "natoms": len(atoms)})
                made += 1

    write_json(round_dir(args.round) / "structures_manifest.json",
               {"round": args.round, "plan": plan, "duplicates_rejected": duplicates, "structures": manifest})
    print(f"[generate_structures] Round {args.round}: wrote {len(manifest)} unique structures to {out_dir} "
          f"({duplicates} exact duplicates rejected)")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[generate_structures] CONFIG ERROR: {e}")
        sys.exit(2)
