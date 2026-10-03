# Step 1 · Generate 1000 unique structures

**Script:** `scripts/generate_structures.py --round r` · **Reads:** `runs/species.json`,
`config/sampling.yaml`, and for r ≥ 1 the previous round's `nep_model/nep.txt` and
`evaluation.json` · **Writes:** `runs/round_XXX/structures/*.vasp`, `structures_manifest.json`

## Round 0 (no NEP yet)

Families and their shares (`round0.weights`): bulk 3, slab 2, interface 2, adsorbate 2,
dimer 1, trimer 1, defect 1, gas 1, compound 1, repulsion 1. A family with nothing to
build from (no compounds, no gas) is dropped and the rest renormalised.

| Family | Built from |
|---|---|
| bulk / compound | Film, substrate, elemental crystals / interfacial compounds, random supercells |
| slab | Film or substrate on its `surface`, `slab_layers` layers, 12 Å vacuum |
| interface | Film on substrate: smallest-strain lateral match within `interface_max_strain` and the atom limit |
| adsorbate | A gas molecule, fragment or inert atom 1–3 Å above a slab |
| dimer, trimer / repulsion | Random species at 0.8–2.5× / 0.35–0.8× the sum of covalent radii |
| gas | An isolated molecule, fragment or atom |
| defect | A single vacancy, substitution (the only source of substitutions) or interstitial in bulk |

**Pristine vs. non-pristine**

- **Pristine** (round 0 only, a few): every crystal and surface at `pristine.supercell_sizes`
  sizes, no defects, strain or rattling. Flagged `"pristine": true` in the manifest.
- **Non-pristine solids** (bulk, compound, slab, interface, the slab under an adsorbate, MD
  starting cells): vacancies 0–15 % and interstitials 0–10 % of the atoms, drawn per
  structure and rounded down. Interstitials (any element of the system) go in first, at
  ≥ 0.75 × the covalent-radius sum from every atom, inside the slab for surfaces.
- **Every non-pristine structure** is strained (solids, ±5 %) and thermally rattled:
  Gaussian displacements with variance k_B·T/k (k = 5 eV/Å²), so lengths follow the
  Maxwell distribution; T = 10–50 % of the melting point of the crystal it is built from
  (the lower of film/substrate for interfaces, the film's for gas, dimers, trimers).

## Rounds 1 … N (MD with the previous NEP, via calorine on CPU)

Half the round (`md_fraction`) comes from MD, 5 snapshots per run:

- **amorphous** — melt at 2000–4000 K, then quench to 300 K (Langevin).
- **collision** — a gas species hits the substrate, film or film on substrate at
  0–50 eV, ≤ 60° from normal, surface at 300 K, bottom 2 Å fixed (NVE).

The other half re-samples the round-0 families, each weighted by its **bucket score** from
the previous Check A (worst RMSE/threshold ratio): worse buckets get more structures.
The orchestrator runs this step as a SLURM job (`templates/python.sbatch.template`,
`walltime_python`) because the MD needs compute; `--local-md` runs it in-process.

## Limits that always hold

- **≤ 120 atoms** per structure (`HARD_MAX_ATOMS`; `max_atoms` can only lower it). Hosts
  leave room for 10 % interstitials, and interfaces also for the largest gas species.
- **Exact-duplicate check:** a hash of species, cell and positions (4 decimals, order-
  independent) against this round and every earlier one.

## Common stops

| Message | Fix |
|---|---|
| `Cannot fit the film slab onto the substrate slab (best lateral strain X %)` | The lattices mismatch at ≤ ~105 atoms. Raise `interface_max_strain`, pick other `surface` indices, or accept that a large coincidence cell does not fit the 120-atom limit. Ask the user which. |
| `A 2-layer ... slab already exceeds max_atoms` | The template's surface cell is too large; use a smaller conventional/primitive template. |
| `Could not generate N unique '<family>' structures` | Too many duplicates or rejected builds; usually a family with almost no freedom. Lower its weight. |
| `calorine is required` (rounds ≥ 1) | Install `requirements.txt` in the `conda_env` the SLURM job uses. |
