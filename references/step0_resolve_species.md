# Step 0 · Resolve species (once)

**Script:** `scripts/resolve_species.py` · **Reads:** `config/system.yaml` · **Writes:** `runs/species.json`

Turns the INPUT (film, substrate, additional gas) into every species the training data
must cover. Runs once; every later step builds from `runs/species.json`.

## What it adds

| Kind | How it is resolved |
|---|---|
| Gas molecules | Each `gas.molecules` entry: a name `ase.build.molecule` knows (e.g. `NH3`, `N2`, `SiH4`), or `{name, structure_file}`. |
| Gas fragments | Every connected fragment reachable by removing atoms one at a time (NH3 → NH2, NH, N, H), using ASE natural cutoffs × 1.2 for bonds. |
| Inert gas | `gas.inert` as listed (e.g. `Ar` for sputtering). |
| Film, substrate | Their crystal templates; Step 0 checks each template's elements match its `formula`. |
| Elemental crystals | One per non-gas element. Diatomic/atomic elements (H, N, O, F, Cl, noble gases) stay gas-only. ASE builds fcc/bcc/hcp/diamond/sc; any other element (e.g. orthorhombic Ga) needs `element_structures`. |
| Interfacial compounds | Only the ones listed in `interfacial_compounds`, each with a template. Step 0 lists every film/gas × substrate element pair and notes pairs with no compound, but cannot predict which compounds form. |
| Melting points | Film, substrate, every compound and every elemental crystal (`melting_point_K`, `element_melting_points_K`). Step 1 scales the thermal-rattling temperature by them. |

## Claude's job here

- Fill missing templates and melting points only from the user or a cited source. Look
  melting points up (web search) and say which source each came from. For a compound that
  decomposes before melting (GaN ≈ 1119 K at 1 atm, Si3N4 ≈ 2173 K), use the
  decomposition temperature and say so.
- Read the "no interfacial compound listed for X-Y" notes back to the user and ask whether
  any of those compounds can form; never add one on your own.

## Common stops

| Message | Fix |
|---|---|
| `structure_file is not set` / `does not exist` | Point the field at a CIF/POSCAR/extxyz under the repo. |
| `ASE has no simple crystal for element X` | Add `element_structures: {X: path}`. |
| `element_melting_points_K is missing [...]` | Add the listed elements' melting points. |
| `contains [...] but formula ... has [...]` | The template and `formula` disagree; fix whichever is wrong. |
| `ase.build.molecule doesn't know gas 'X'` | Give it as `{name: X, structure_file: ...}`. |
