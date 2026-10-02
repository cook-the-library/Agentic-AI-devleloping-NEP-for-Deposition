"""Shared helpers for the nep-deposition-workflow skill scripts.

pyyaml + numpy are the only hard requirements for the helpers themselves;
ase (structures) and calorine (running a trained NEP on CPU) are imported
where needed, and every script stops with a clear message telling the user
what to install rather than silently no-op'ing.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

SKILL_ROOT = Path(__file__).resolve().parent.parent
# NEP_CONFIG_DIR / NEP_RUNS_DIR let one checkout serve several material systems.
CONFIG_DIR = Path(os.environ.get("NEP_CONFIG_DIR", SKILL_ROOT / "config"))
TEMPLATES_DIR = SKILL_ROOT / "templates"
RUNS_DIR = Path(os.environ.get("NEP_RUNS_DIR", SKILL_ROOT / "runs"))
SPECIES_PATH = RUNS_DIR / "species.json"
FINAL_NEP_DIR = RUNS_DIR / "final_nep"
CHECK_B_DIR = RUNS_DIR / "check_b"


class ConfigError(RuntimeError):
    """Raised when required config is missing/unfilled. Not caught silently --
    this should stop the pipeline and surface to the human."""


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_clusters_config() -> dict:
    return load_yaml(CONFIG_DIR / "clusters.yaml")


def load_criteria_config() -> dict:
    return load_yaml(CONFIG_DIR / "criteria.yaml")


def load_system_config() -> dict:
    return load_yaml(CONFIG_DIR / "system.yaml")


def load_sampling_config() -> dict:
    return load_yaml(CONFIG_DIR / "sampling.yaml")


def _find_unfilled(obj, path=""):
    unfilled = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            unfilled.extend(_find_unfilled(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            unfilled.extend(_find_unfilled(v, f"{path}[{i}]"))
    elif isinstance(obj, str) and "FILL_ME_IN" in obj:
        unfilled.append(path)
    return unfilled


def require_filled(cfg: dict, *, context: str) -> None:
    """Raise ConfigError listing every FILL_ME_IN field still present in cfg.

    Call this on the specific sub-config a stage actually needs (e.g. the
    single cluster's block from clusters.yaml) rather than the whole file, so
    a stage that doesn't touch ACES doesn't block on unfilled ACES fields.
    """
    unfilled = _find_unfilled(cfg)
    if unfilled:
        raise ConfigError(
            f"Cannot proceed with {context}: the following config values are still "
            f"placeholders and must be filled in by a human before this stage can "
            f"run: {', '.join(unfilled)}. Edit config/clusters.yaml (or the relevant "
            f"config file) and re-run."
        )


def cluster_config(cluster: str) -> dict:
    if cluster not in ("anvil", "aces"):
        raise ValueError(f"Unknown cluster '{cluster}', expected 'anvil' or 'aces'")
    all_cfg = load_clusters_config()
    return all_cfg[cluster]


def require_ase():
    try:
        import ase  # noqa: F401
    except ImportError as e:
        raise ConfigError("ASE is required (`pip install ase`).") from e


def nep_calculator(nep_txt: Path):
    """ASE calculator for a trained NEP, run on CPU through calorine. Used for
    the MD sampling in rounds 1..N and for Check A / Check B."""
    try:
        from calorine.calculators import CPUNEP
    except ImportError as e:
        raise ConfigError(
            "calorine is required to run a trained NEP from Python (`pip install calorine`)."
        ) from e
    if not Path(nep_txt).exists():
        raise ConfigError(f"{nep_txt} not found -- has NEP training finished?")
    return CPUNEP(str(nep_txt))


def read_template(path_str: str, context: str):
    """Read a crystal template (CIF/POSCAR/extxyz) given relative to the repo root."""
    from ase.io import read

    if not path_str or "FILL_ME_IN" in str(path_str):
        raise ConfigError(f"{context}: structure_file is not set in config/system.yaml.")
    path = Path(path_str)
    if not path.is_absolute():
        path = SKILL_ROOT / path
    if not path.exists():
        raise ConfigError(f"{context}: structure file {path} does not exist.")
    return read(path)


def atoms_to_dict(atoms) -> dict:
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().round(6).tolist(),
        "cell": atoms.get_cell().round(6).tolist(),
        "pbc": [bool(x) for x in atoms.get_pbc()],
    }


def dict_to_atoms(d: dict):
    from ase import Atoms

    return Atoms(symbols=d["symbols"], positions=d["positions"], cell=d["cell"], pbc=d["pbc"])


def structure_hash(atoms, decimals: int = 4) -> str:
    """Hash for the exact-duplicate check: identical species, cell and
    (wrapped) positions to `decimals` Angstrom, independent of atom order."""
    import numpy as np

    a = atoms.copy()
    if a.get_pbc().any() and a.get_cell().rank == 3:
        a.wrap()
    pos = np.round(a.get_positions(), decimals) + 0.0  # +0.0 turns -0.0 into 0.0
    rows = sorted(zip(a.get_atomic_numbers().tolist(), map(tuple, pos.tolist())))
    cell = (np.round(a.get_cell().array, decimals) + 0.0).tolist()
    payload = json.dumps([rows, cell, a.get_pbc().tolist()])
    return hashlib.sha1(payload.encode()).hexdigest()


def round_dir(round_num: int, create: bool = True) -> Path:
    d = RUNS_DIR / f"round_{round_num:03d}"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def render_template(template_name: str, context: dict) -> str:
    """Render a .sbatch.template with Jinja2 if available, else a minimal
    fallback that supports the {{var}}, {% if x %}...{% endif %}, and
    {% for x in y %}...{% endfor %} constructs actually used in
    templates/*.sbatch.template (nothing fancier)."""
    template_path = TEMPLATES_DIR / template_name
    text = template_path.read_text()
    try:
        import jinja2

        return jinja2.Template(text).render(**context)
    except ImportError:
        return _minimal_render(text, context)


def _minimal_render(text: str, context: dict) -> str:
    import re

    # {% for line in modules %}...{{line}}\n{% endfor %}
    def render_for(match):
        var, iterable_name, body = match.group(1), match.group(2), match.group(3)
        items = context.get(iterable_name, [])
        out = []
        for item in items:
            out.append(body.replace("{{" + var + "}}", str(item)))
        return "".join(out)

    text = re.sub(
        r"\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}",
        render_for,
        text,
        flags=re.DOTALL,
    )

    # {% if cond %}...{% endif %}  (only supports a bare truthy variable name)
    def render_if(match):
        cond_name, body = match.group(1), match.group(2)
        return body if context.get(cond_name) else ""

    text = re.sub(
        r"\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}",
        render_if,
        text,
        flags=re.DOTALL,
    )

    for key, val in context.items():
        if isinstance(val, (str, int, float)):
            text = text.replace("{{" + key + "}}", str(val))

    return text


def sbatch_context(cluster_cfg: dict, *, job_name: str, workdir: Path, kind: str) -> dict:
    """kind: 'vasp' (DFT labels and Check B AIMD) | 'nep' (training) |
    'python' (Step 1 MD sampling with the previous NEP, on CPU)."""
    walltime = cluster_cfg[f"walltime_{kind}"]
    if kind == "vasp":
        partition = cluster_cfg["partition_cpu"]
        nodes = cluster_cfg["nodes_vasp"]
        ntasks = cluster_cfg["ntasks_vasp"]
    elif kind == "nep":
        partition = cluster_cfg["partition_gpu"]
        nodes = 1
        ntasks = 1
    else:  # python
        partition = cluster_cfg["partition_cpu"]
        nodes = 1
        ntasks = cluster_cfg.get("ntasks_python", 1)

    common = load_clusters_config().get("common", {})
    return {
        "job_name": job_name,
        "account": cluster_cfg["account"],
        "partition": partition,
        "nodes": nodes,
        "ntasks": ntasks,
        "walltime": walltime,
        "workdir": str(workdir),
        "qos": cluster_cfg.get("qos"),
        "modules": cluster_cfg["modules"].get(kind, []),
        "vasp_exe": cluster_cfg.get("executables", {}).get("vasp_std", "vasp_std"),
        "nep_exe": cluster_cfg.get("executables", {}).get("nep", "nep"),
        "conda_env": common.get("conda_env"),
        "repo_root": str(SKILL_ROOT),
    }


def submit_job(sbatch_path: Path) -> str:
    """Submit via sbatch, return the SLURM job id. Raises if sbatch isn't
    available (e.g. running this off-cluster to just inspect the rendered
    script) -- caller should catch and report, not swallow."""
    if not _has_sbatch():
        raise RuntimeError(
            f"'sbatch' not found on PATH -- {sbatch_path} was rendered but not "
            f"submitted. Run this on the login node of the target cluster, or "
            f"submit it yourself with `sbatch {sbatch_path}`."
        )
    result = subprocess.run(
        ["sbatch", "--parsable", str(sbatch_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = result.stdout.strip().splitlines()[-1]
    return job_id


def _has_sbatch() -> bool:
    from shutil import which

    return which("sbatch") is not None


def job_state(job_id: str) -> str:
    """Return a coarse state: PENDING, RUNNING, COMPLETED, FAILED, or UNKNOWN."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
            check=True,
        )
        states = [s.strip() for s in result.stdout.splitlines() if s.strip()]
        if not states:
            return "UNKNOWN"
        state = states[0]
        if state.startswith("COMPLETED"):
            return "COMPLETED"
        if any(s in state for s in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL")):
            return "FAILED"
        if "RUNNING" in state:
            return "RUNNING"
        if "PENDING" in state:
            return "PENDING"
        return state
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"


def wait_for_job(job_id: str, *, poll_interval: int, max_wait_hours: float) -> str:
    deadline = time.time() + max_wait_hours * 3600
    while time.time() < deadline:
        state = job_state(job_id)
        if state in ("COMPLETED", "FAILED"):
            return state
        time.sleep(poll_interval)
    return "TIMED_OUT_WAITING"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
