#!/usr/bin/env python3
"""Step 0 (run once): resolve the species the training data must cover.

Reads the INPUT in config/system.yaml (film, substrate, additional gas) and
adds the gas and solid species around it:
  gas:   the gas molecules, every connected fragment of them (e.g. NH3 ->
         NH2, NH, N, H), and the inert gas
  solid: the film and substrate crystals, every element as its elemental
         crystal (diatomic/atomic elements such as N, O, H stay gas only), and
         the interfacial compounds listed in config/system.yaml

Writes runs/species.json, which Step 1 (generate_structures.py) builds every
round's structures from.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    SPECIES_PATH,
    ConfigError,
    atoms_to_dict,
    eprint,
    load_system_config,
    read_template,
    require_ase,
    write_json,
)

SOLID_SYMMETRIES = {"fcc", "bcc", "hcp", "diamond", "sc"}
GAS_SYMMETRIES = {"diatom", "atom"}


def formula_elements(formula: str) -> set[str]:
    from ase.formula import Formula

    return set(Formula(formula).count())


def load_gas_molecule(entry):
    from ase.build import molecule

    if isinstance(entry, str):
        if "FILL_ME_IN" in entry:
            raise ConfigError("config/system.yaml: gas.molecules still has FILL_ME_IN (use [] for no gas).")
        try:
            return entry, molecule(entry)
        except KeyError as e:
            raise ConfigError(
                f"ase.build.molecule doesn't know gas '{entry}'. Give it as "
                f"{{name: {entry}, structure_file: ...}} in config/system.yaml."
            ) from e
    return entry["name"], read_template(entry.get("structure_file"), f"gas '{entry.get('name')}'")


def connected(atoms) -> bool:
    from ase.neighborlist import NeighborList, natural_cutoffs
    from scipy.sparse.csgraph import connected_components

    if len(atoms) == 1:
        return True
    nl = NeighborList(natural_cutoffs(atoms, mult=1.2), self_interaction=False, bothways=True)
    nl.update(atoms)
    n_comp, _ = connected_components(nl.get_connectivity_matrix(sparse=True))
    return n_comp == 1


def fragments(mol) -> dict:
    """All connected fragments reachable by removing atoms one at a time,
    keyed by chemical formula (first geometry found is kept)."""
    found = {}
    frontier = [mol]
    while frontier:
        nxt = []
        for parent in frontier:
            if len(parent) == 1:
                continue
            for i in range(len(parent)):
                pieces = [parent[[j for j in range(len(parent)) if j != i]], parent[[i]]]
                for piece in pieces:
                    if not connected(piece):
                        continue
                    key = piece.get_chemical_formula()
                    if key in found or key == mol.get_chemical_formula():
                        continue
                    found[key] = piece
                    nxt.append(piece)
        frontier = nxt
    return found


def is_gas_element(symbol: str) -> bool:
    from ase.data import atomic_numbers, reference_states

    state = reference_states[atomic_numbers[symbol]]
    return bool(state) and state.get("symmetry") in GAS_SYMMETRIES


def element_crystal(symbol: str, templates: dict):
    """Elemental crystal: a template from config/system.yaml's
    element_structures, else ASE's bulk() for simple lattices."""
    from ase.build import bulk
    from ase.data import atomic_numbers, reference_states

    if symbol in templates:
        return read_template(templates[symbol], f"element_structures.{symbol}")
    state = reference_states[atomic_numbers[symbol]]
    if state and state.get("symmetry") in SOLID_SYMMETRIES:
        try:
            return bulk(symbol)
        except Exception:
            pass
    raise ConfigError(
        f"ASE has no simple crystal for element {symbol}. Add a template for it "
        f"under element_structures in config/system.yaml (e.g. {symbol}: structures/{symbol}.cif).")


def melting_point(value, context: str) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"config/system.yaml: {context} melting point is not set (got {value!r}).") from None
    if t <= 0:
        raise ConfigError(f"config/system.yaml: {context} melting point must be positive (got {t}).")
    return t


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_ase()
    cfg = load_system_config()

    film = read_template(cfg["film"]["structure_file"], "film")
    substrate = read_template(cfg["substrate"]["structure_file"], "substrate")
    for name, atoms in (("film", film), ("substrate", substrate)):
        expected = formula_elements(cfg[name]["formula"])
        if set(atoms.get_chemical_symbols()) != expected:
            raise ConfigError(
                f"config/system.yaml: {name}.structure_file contains "
                f"{sorted(set(atoms.get_chemical_symbols()))}, but {name}.formula "
                f"'{cfg[name]['formula']}' has {sorted(expected)}.")

    molecules = {}
    gas_fragments = {}
    for entry in cfg["gas"].get("molecules") or []:
        name, mol = load_gas_molecule(entry)
        molecules[name] = mol
        for key, frag in fragments(mol).items():
            gas_fragments.setdefault(key, frag)
    for name in molecules:
        gas_fragments.pop(name, None)
    inert = list(cfg["gas"].get("inert") or [])

    compounds = []
    for entry in cfg.get("interfacial_compounds") or []:
        atoms = read_template(entry.get("structure_file"), f"interfacial compound '{entry.get('formula')}'")
        compounds.append({"formula": entry["formula"], "structure": atoms_to_dict(atoms),
                          "melting_point_K": melting_point(entry.get("melting_point_K"),
                                                           f"interfacial compound {entry['formula']}")})

    film_el = set(film.get_chemical_symbols())
    sub_el = set(substrate.get_chemical_symbols())
    gas_el = set().union(*(set(m.get_chemical_symbols()) for m in molecules.values())) if molecules else set()
    all_el = sorted(film_el | sub_el | gas_el | {s for c in compounds for s in c["structure"]["symbols"]})
    gas_only_el = [s for s in all_el if s in inert or is_gas_element(s)]
    templates = cfg.get("element_structures") or {}
    solid_el = {s: atoms_to_dict(element_crystal(s, templates)) for s in all_el if s not in gas_only_el}
    el_tm = cfg.get("element_melting_points_K") or {}
    missing = [s for s in solid_el if s not in el_tm]
    if missing:
        raise ConfigError(f"config/system.yaml: element_melting_points_K is missing {missing} "
                          f"(Step 0 adds these elemental crystals).")
    el_tm = {s: melting_point(el_tm[s], f"element {s}") for s in solid_el}

    # Interfacial compounds: report every film/gas element x substrate element
    # pair, and which ones have a template.
    pairs = sorted({tuple(sorted(p)) for p in itertools.product(film_el | gas_el, sub_el) if p[0] != p[1]})
    covered = []
    uncovered = []
    for a, b in pairs:
        hit = [c["formula"] for c in compounds if {a, b} <= set(c["structure"]["symbols"])]
        (covered if hit else uncovered).append({"pair": [a, b], "compounds": hit})
    for u in uncovered:
        eprint(f"[resolve_species] NOTE: no interfacial compound listed for {u['pair'][0]}-{u['pair'][1]}. "
               f"Add one under interfacial_compounds in config/system.yaml if it can form.")

    species = {
        "film": {"formula": cfg["film"]["formula"], "surface": cfg["film"]["surface"],
                 "structure": atoms_to_dict(film),
                 "melting_point_K": melting_point(cfg["film"].get("melting_point_K"), "film")},
        "substrate": {"formula": cfg["substrate"]["formula"], "surface": cfg["substrate"]["surface"],
                      "structure": atoms_to_dict(substrate),
                      "melting_point_K": melting_point(cfg["substrate"].get("melting_point_K"), "substrate")},
        "gas": {
            "molecules": {k: atoms_to_dict(v) for k, v in molecules.items()},
            "fragments": {k: atoms_to_dict(v) for k, v in gas_fragments.items()},
            "inert": inert,
        },
        "solid": {
            "elements": solid_el,
            "element_melting_points_K": el_tm,
            "interfacial_compounds": compounds,
        },
        "elements": sorted(set(all_el) | set(inert)),
        "gas_only_elements": gas_only_el,
        "interface_pairs": covered + uncovered,
    }
    write_json(SPECIES_PATH, species)
    print(f"[resolve_species] gas: molecules={list(molecules)} fragments={list(gas_fragments)} inert={inert}")
    print(f"[resolve_species] solid: film={cfg['film']['formula']} substrate={cfg['substrate']['formula']} "
          f"elements={list(solid_el)} compounds={[c['formula'] for c in compounds]}")
    print(f"[resolve_species] wrote {SPECIES_PATH}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[resolve_species] CONFIG ERROR: {e}")
        sys.exit(2)
