#!/usr/bin/env python3
"""
Download task-specific structures from Materials Project for prior parent pools.
Uses config/tasks.yaml to derive element filters and pool defaults.
"""
import sys
from pathlib import Path

# Ensure project root (MatAgent/) is on sys.path so "src" imports work.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.env_loader import load_env_file
load_env_file()

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import yaml
from mp_api.client import MPRester
from tqdm import tqdm


def _parse_elements(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [e.strip() for e in raw.split(",") if e.strip()]


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("vrh", "value"):
            if key in value and value[key] is not None:
                return float(value[key])
    return None


def _first_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def _chunked(items: List[str], size: int):
    for idx in range(0, len(items), size):
        yield items[idx: idx + size]


def _dielectric_from_e_total(e_total):
    if e_total is None:
        return None
    if isinstance(e_total, (int, float)):
        return float(e_total)
    try:
        return float(e_total[0][0] + e_total[1][1] + e_total[2][2]) / 3.0
    except (IndexError, TypeError, ValueError):
        return None


def _search_piezoelectric(mpr, material_ids: List[str]):
    kwargs = {"fields": ["material_id", "e_ij_max"], "material_ids": material_ids}
    try:
        return mpr.materials.piezoelectric.search(**kwargs)
    except TypeError:
        kwargs["material_id"] = kwargs.pop("material_ids")
        return mpr.materials.piezoelectric.search(**kwargs)


def _search_dielectric(mpr, material_ids: List[str]):
    kwargs = {"fields": ["material_id", "e_total"], "material_ids": material_ids}
    try:
        return mpr.materials.dielectric.search(**kwargs)
    except TypeError:
        kwargs["material_id"] = kwargs.pop("material_ids")
        return mpr.materials.dielectric.search(**kwargs)


def _search_with_chunking(search_fn, material_ids: List[str], chunk_size: int):
    if not material_ids:
        return []
    results = []
    for chunk in _chunked(material_ids, chunk_size):
        try:
            results.extend(search_fn(chunk))
        except ValueError as exc:
            if "too long" in str(exc) and len(chunk) > 1:
                # MP API rejects long ID lists; retry with smaller chunks.
                smaller = max(1, chunk_size // 2)
                results.extend(_search_with_chunking(search_fn, chunk, smaller))
            else:
                raise
    return results


def _fetch_piezoelectric_map(mpr, material_ids: List[str]) -> Dict[str, float]:
    if not material_ids:
        return {}
    docs = _search_with_chunking(
        lambda ids: _search_piezoelectric(mpr, ids),
        list(material_ids),
        chunk_size=500,
    )
    results = {}
    for doc in docs:
        mp_id = getattr(doc, "material_id", None)
        if not mp_id:
            continue
        piezo = _as_float(getattr(doc, "e_ij_max", None))
        if piezo is None:
            piezo = _as_float(_first_attr(doc, ["e_ij_maximum", "e_ij_max_value", "e_ij"]))
        if piezo is not None:
            results[str(mp_id)] = piezo
    return results


def _fetch_dielectric_map(mpr, material_ids: List[str]) -> Dict[str, float]:
    if not material_ids:
        return {}
    docs = _search_with_chunking(
        lambda ids: _search_dielectric(mpr, ids),
        list(material_ids),
        chunk_size=500,
    )
    results = {}
    for doc in docs:
        mp_id = getattr(doc, "material_id", None)
        if not mp_id:
            continue
        e_total = _first_attr(doc, ["e_total", "epsilon_total", "eps_total"])
        dielectric = _dielectric_from_e_total(e_total)
        if dielectric is None:
            dielectric = _as_float(
                _first_attr(doc, ["dielectric_constant", "epsilon_avg", "eps_total_avg"])
            )
        if dielectric is not None:
            results[str(mp_id)] = dielectric
    return results


def _load_tasks(tasks_path: Path) -> Dict:
    with open(tasks_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("tasks", {})


def _resolve_task(tasks: Dict, key_or_name: str) -> Tuple[str, Dict]:
    if key_or_name in tasks:
        return key_or_name, tasks[key_or_name]
    lowered = key_or_name.strip().lower()
    for key, cfg in tasks.items():
        name = (cfg.get("name") or "").strip().lower()
        if name == lowered:
            return key, cfg
    raise KeyError(f"Task '{key_or_name}' not found in tasks.yaml")


def _resolve_min_ed(cli_min_ed: Optional[float]) -> float:
    if cli_min_ed is not None:
        return float(cli_min_ed)
    return 0.0


def _resolve_max_ed(task_cfg: Dict, cli_max_ed: Optional[float]) -> float:
    if cli_max_ed is not None:
        return float(cli_max_ed)
    decomp = task_cfg.get("constraints", {}).get("decomposition_energy", {}) or {}
    if decomp.get("enabled") and decomp.get("max") is not None:
        return float(decomp["max"])
    return 0.2


def _resolve_defaults(task_cfg: Dict, args: argparse.Namespace) -> Dict:
    init_pool = task_cfg.get("initial_parent_pool", {}) or {}
    task_oriented = init_pool.get("task_oriented", {}) or {}

    include_elements = _parse_elements(args.include_elements) or task_oriented.get("include_elements", [])
    include_mode = task_oriented.get("include_mode", "subset")
    exclude_elements = _parse_elements(args.exclude_elements) or task_oriented.get("exclude_elements", [])
    min_elements = args.min_elements if args.min_elements is not None else task_oriented.get("min_elements")
    max_elements = args.max_elements if args.max_elements is not None else task_oriented.get("max_elements")
    max_sites = args.max_sites if args.max_sites is not None else task_oriented.get("max_sites")

    target_count = args.count if args.count is not None else init_pool.get("max_count", 300)
    min_ed = _resolve_min_ed(args.min_ed)
    max_ed = _resolve_max_ed(task_cfg, args.max_ed)
    constraints = task_cfg.get("constraints", {}) or {}
    bulk_cfg = constraints.get("bulk_modulus", {}) or {}
    shear_cfg = constraints.get("shear_modulus", {}) or {}
    density_cfg = constraints.get("density", {}) or {}
    band_gap_cfg = constraints.get("band_gap", {}) or {}
    formation_cfg = constraints.get("formation_energy", {}) or {}
    piezo_cfg = constraints.get("piezoelectric_coefficient", {}) or {}
    dielectric_cfg = constraints.get("dielectric_constant", {}) or {}
    bulk_min = None
    bulk_max = None
    shear_min = None
    shear_max = None
    density_min = None
    density_max = None
    band_gap_min = None
    band_gap_max = None
    formation_min = None
    formation_max = None
    bulk_range = None
    shear_range = None
    piezo_min = None
    piezo_max = None
    dielectric_min = None
    dielectric_max = None
    if args.use_task_constraints:
        if bulk_cfg.get("enabled"):
            if bulk_cfg.get("min") is not None:
                bulk_min = float(bulk_cfg["min"])
            if bulk_cfg.get("max") is not None:
                bulk_max = float(bulk_cfg["max"])
        if shear_cfg.get("enabled"):
            if shear_cfg.get("min") is not None:
                shear_min = float(shear_cfg["min"])
            if shear_cfg.get("max") is not None:
                shear_max = float(shear_cfg["max"])
        if density_cfg.get("enabled"):
            if density_cfg.get("min") is not None:
                density_min = float(density_cfg["min"])
            if density_cfg.get("max") is not None:
                density_max = float(density_cfg["max"])
        if band_gap_cfg.get("enabled"):
            if band_gap_cfg.get("min") is not None:
                band_gap_min = float(band_gap_cfg["min"])
            if band_gap_cfg.get("max") is not None:
                band_gap_max = float(band_gap_cfg["max"])
        if formation_cfg.get("enabled"):
            if formation_cfg.get("min") is not None:
                formation_min = float(formation_cfg["min"])
            if formation_cfg.get("max") is not None:
                formation_max = float(formation_cfg["max"])
        if bulk_min is not None and bulk_max is not None:
            bulk_range = (bulk_min, bulk_max)
        if shear_min is not None and shear_max is not None:
            shear_range = (shear_min, shear_max)

    if piezo_cfg.get("enabled"):
        if piezo_cfg.get("min") is not None:
            piezo_min = float(piezo_cfg["min"])
        if piezo_cfg.get("max") is not None:
            piezo_max = float(piezo_cfg["max"])

    if dielectric_cfg.get("enabled"):
        if dielectric_cfg.get("min") is not None:
            dielectric_min = float(dielectric_cfg["min"])
        if dielectric_cfg.get("max") is not None:
            dielectric_max = float(dielectric_cfg["max"])

    # Explicit CLI overrides for off-target / weak parent-pool construction.
    if args.band_gap_min is not None:
        band_gap_min = float(args.band_gap_min)
    if args.band_gap_max is not None:
        band_gap_max = float(args.band_gap_max)
    if args.dielectric_min is not None:
        dielectric_min = float(args.dielectric_min)
    if args.dielectric_max is not None:
        dielectric_max = float(args.dielectric_max)

    return {
        "include_elements": include_elements,
        "include_mode": include_mode,
        "exclude_elements": exclude_elements,
        "min_elements": min_elements,
        "max_elements": max_elements,
        "max_sites": max_sites,
        "target_count": target_count,
        "min_ed": min_ed,
        "max_ed": max_ed,
        "query_mode": args.query_mode,
        "bulk_range": bulk_range,
        "shear_range": shear_range,
        "bulk_min": bulk_min,
        "bulk_max": bulk_max,
        "shear_min": shear_min,
        "shear_max": shear_max,
        "density_min": density_min,
        "density_max": density_max,
        "band_gap_min": band_gap_min,
        "band_gap_max": band_gap_max,
        "formation_min": formation_min,
        "formation_max": formation_max,
        "require_moduli": args.use_task_constraints and not args.allow_missing_moduli,
        "require_piezo": bool(piezo_cfg.get("enabled")),
        "require_dielectric": bool(dielectric_cfg.get("enabled")),
        "piezo_min": piezo_min,
        "piezo_max": piezo_max,
        "dielectric_min": dielectric_min,
        "dielectric_max": dielectric_max,
    }


def _filter_docs(
    docs: List,
    include_elements: List[str],
    exclude_elements: List[str],
    include_mode: str = "subset",
    min_elements: Optional[int] = None,
    max_elements: Optional[int] = None,
    max_sites: Optional[int] = None,
    bulk_min: Optional[float] = None,
    bulk_max: Optional[float] = None,
    shear_min: Optional[float] = None,
    shear_max: Optional[float] = None,
    density_min: Optional[float] = None,
    density_max: Optional[float] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    formation_min: Optional[float] = None,
    formation_max: Optional[float] = None,
    require_moduli: bool = False,
    require_piezo: bool = False,
    require_dielectric: bool = False,
    piezo_map: Optional[Dict[str, float]] = None,
    dielectric_map: Optional[Dict[str, float]] = None,
    piezo_min: Optional[float] = None,
    piezo_max: Optional[float] = None,
    dielectric_min: Optional[float] = None,
    dielectric_max: Optional[float] = None,
) -> List:
    include_set = set(include_elements or [])
    exclude_set = set(exclude_elements or [])
    filtered = []
    missing_elements = 0
    for doc in docs:
        doc_elements = {str(e) for e in getattr(doc, "elements", [])}
        if (include_set or exclude_set) and not doc_elements:
            missing_elements += 1
            continue
        if include_set:
            if include_mode == "any":
                if not doc_elements & include_set:
                    continue
            else:
                if not doc_elements.issubset(include_set):
                    continue
        if exclude_set and doc_elements.intersection(exclude_set):
            continue
        if min_elements is not None and doc.nelements < min_elements:
            continue
        if max_elements is not None and doc.nelements > max_elements:
            continue
        if max_sites is not None and doc.nsites > max_sites:
            continue
        if require_piezo or piezo_min is not None or piezo_max is not None:
            piezo_value = None
            if piezo_map is not None:
                piezo_value = piezo_map.get(str(doc.material_id))
            else:
                piezo_value = _as_float(getattr(doc, "e_ij_max", None))
            if piezo_value is None:
                continue
            if piezo_min is not None and piezo_value < piezo_min:
                continue
            if piezo_max is not None and piezo_value > piezo_max:
                continue
        if require_dielectric or dielectric_min is not None or dielectric_max is not None:
            dielectric_value = None
            if dielectric_map is not None:
                dielectric_value = dielectric_map.get(str(doc.material_id))
            else:
                dielectric_value = _dielectric_from_e_total(getattr(doc, "e_total", None))
            if dielectric_value is None:
                continue
            if dielectric_min is not None and dielectric_value < dielectric_min:
                continue
            if dielectric_max is not None and dielectric_value > dielectric_max:
                continue
        if bulk_min is not None or bulk_max is not None:
            bulk_mod = _as_float(getattr(doc, "bulk_modulus", None))
            if bulk_mod is None:
                if require_moduli:
                    continue
            else:
                if bulk_min is not None and bulk_mod < bulk_min:
                    continue
                if bulk_max is not None and bulk_mod > bulk_max:
                    continue
        if shear_min is not None or shear_max is not None:
            shear_mod = _as_float(getattr(doc, "shear_modulus", None))
            if shear_mod is None:
                if require_moduli:
                    continue
            else:
                if shear_min is not None and shear_mod < shear_min:
                    continue
                if shear_max is not None and shear_mod > shear_max:
                    continue
        if density_min is not None or density_max is not None:
            density = _as_float(getattr(doc, "density", None))
            if density is None:
                continue
            if density_min is not None and density < density_min:
                continue
            if density_max is not None and density > density_max:
                continue
        if band_gap_min is not None or band_gap_max is not None:
            band_gap = _as_float(getattr(doc, "band_gap", None))
            if band_gap is None:
                continue
            if band_gap_min is not None and band_gap < band_gap_min:
                continue
            if band_gap_max is not None and band_gap > band_gap_max:
                continue
        if formation_min is not None or formation_max is not None:
            formation_energy = _as_float(getattr(doc, "formation_energy_per_atom", None))
            if formation_energy is None:
                continue
            if formation_min is not None and formation_energy < formation_min:
                continue
            if formation_max is not None and formation_energy > formation_max:
                continue
        filtered.append(doc)
    if missing_elements:
        print(f"Note: {missing_elements} entries missing elements field, skipped")
    return filtered


def download_task_specific_structures(
    task_key: str,
    task_cfg: Dict,
    output_dir: Path,
    target_count: int,
    min_ed: float,
    max_ed: float,
    include_elements: List[str],
    include_mode: str,
    exclude_elements: List[str],
    min_elements: Optional[int],
    max_elements: Optional[int],
    max_sites: Optional[int],
    theoretical: bool,
    query_mode: str,
    bulk_range: Optional[Tuple[float, float]],
    shear_range: Optional[Tuple[float, float]],
    bulk_min: Optional[float],
    bulk_max: Optional[float],
    shear_min: Optional[float],
    shear_max: Optional[float],
    density_min: Optional[float],
    density_max: Optional[float],
    band_gap_min: Optional[float],
    band_gap_max: Optional[float],
    formation_min: Optional[float],
    formation_max: Optional[float],
    require_moduli: bool,
    require_piezo: bool,
    require_dielectric: bool,
    piezo_min: Optional[float],
    piezo_max: Optional[float],
    dielectric_min: Optional[float],
    dielectric_max: Optional[float],
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp_api_key = os.getenv("MP_API_KEY")
    if not mp_api_key:
        print("Error: MP_API_KEY not found")
        return

    print(f"Task: {task_key} ({task_cfg.get('name')})")
    if include_elements and query_mode == "elements" and (include_mode == "any" or len(include_elements) > 4):
        print(
            "Note: MP elements filter incompatible with include_mode or element set too large. "
            "Switching to query_mode=none."
        )
        query_mode = "none"

    print(f"Criteria: {min_ed} <= Ed <= {max_ed} eV/atom, theoretical={theoretical}")
    print(f"Target count: {target_count}")
    print(f"Output dir: {output_dir}")
    print(f"Query mode: {query_mode}")
    if include_elements:
        if include_mode == "any":
            print(f"Elements (at least one): {sorted(include_elements)}")
        else:
            print(f"Elements (allowed set): {sorted(include_elements)}")
    if exclude_elements:
        print(f"Elements (excluded): {sorted(exclude_elements)}")
    if min_elements is not None or max_elements is not None:
        print(f"Element count range: {min_elements} - {max_elements}")
    if max_sites is not None:
        print(f"Max sites: {max_sites}")
    if bulk_min is not None or bulk_max is not None:
        print(f"Bulk modulus range: {bulk_min} - {bulk_max} GPa")
    if shear_min is not None or shear_max is not None:
        print(f"Shear modulus range: {shear_min} - {shear_max} GPa")
    if density_min is not None or density_max is not None:
        print(f"Density range: {density_min} - {density_max} g/cm^3")
    if band_gap_min is not None or band_gap_max is not None:
        print(f"Band gap range: {band_gap_min} - {band_gap_max} eV")
    if formation_min is not None or formation_max is not None:
        print(f"Formation energy range: {formation_min} - {formation_max} eV/atom")
    if piezo_min is not None or piezo_max is not None:
        print(f"Piezoelectric range: {piezo_min} - {piezo_max}")
    if dielectric_min is not None or dielectric_max is not None:
        print(f"Dielectric range: {dielectric_min} - {dielectric_max}")
    print("=" * 60)

    examples = []
    with MPRester(mp_api_key) as mpr:
        print("\nQuerying Materials Project...")
        query = dict(
            energy_above_hull=(min_ed, max_ed),
            theoretical=theoretical,
            fields=[
                "material_id",
                "formula_pretty",
                "elements",
                "energy_per_atom",
                "formation_energy_per_atom",
                "energy_above_hull",
                "density",
                "nsites",
                "nelements",
                "symmetry",
                "bulk_modulus",
                "shear_modulus",
                "band_gap",
                "e_ij_max",
                "e_total",
            ],
        )

        if include_elements:
            if query_mode == "chemsys":
                query["chemsys"] = "-".join(sorted(include_elements))
            elif query_mode == "elements":
                query["elements"] = include_elements
            elif query_mode == "none":
                pass
            else:
                raise ValueError(f"Unsupported query_mode: {query_mode}")

        if min_elements is not None or max_elements is not None:
            query["num_elements"] = (min_elements or 1, max_elements or 6)
        if bulk_range:
            query["k_vrh"] = bulk_range
        if shear_range:
            query["g_vrh"] = shear_range

        docs = mpr.materials.summary.search(**query)
        print(f"Query returned {len(docs)} structures")

        piezo_map = {}
        dielectric_map = {}
        candidate_ids = {str(doc.material_id) for doc in docs if getattr(doc, "material_id", None)}
        if require_piezo:
            piezo_map = _fetch_piezoelectric_map(mpr, list(candidate_ids))
            print(f"Piezoelectric available: {len(piezo_map)} / {len(candidate_ids)}")
        if require_dielectric:
            dielectric_map = _fetch_dielectric_map(mpr, list(candidate_ids))
            print(f"Dielectric available: {len(dielectric_map)} / {len(candidate_ids)}")

        docs = _filter_docs(
            docs,
            include_elements=include_elements,
            exclude_elements=exclude_elements,
            include_mode=include_mode,
            min_elements=min_elements,
            max_elements=max_elements,
            max_sites=max_sites,
            bulk_min=bulk_min,
            bulk_max=bulk_max,
            shear_min=shear_min,
            shear_max=shear_max,
            density_min=density_min,
            density_max=density_max,
            band_gap_min=band_gap_min,
            band_gap_max=band_gap_max,
            formation_min=formation_min,
            formation_max=formation_max,
            require_moduli=require_moduli,
            require_piezo=require_piezo,
            require_dielectric=require_dielectric,
            piezo_map=piezo_map,
            dielectric_map=dielectric_map,
            piezo_min=piezo_min,
            piezo_max=piezo_max,
            dielectric_min=dielectric_min,
            dielectric_max=dielectric_max,
        )
        print(f"After filtering: {len(docs)} structures remaining")

        if len(docs) > target_count:
            print(f"Random sampling {target_count}...")
            docs = random.sample(docs, target_count)

        print(f"\nDownloading {len(docs)} structures...\n")
        for doc in tqdm(docs, desc="Downloading"):
            try:
                mp_id = doc.material_id
                structure = mpr.get_structure_by_material_id(mp_id)
                filename = f"{mp_id}.vasp"
                filepath = output_dir / filename
                structure.to(filename=str(filepath), fmt="poscar")

                crystal_system = "unknown"
                space_group = "unknown"
                if hasattr(doc, "symmetry") and doc.symmetry:
                    if hasattr(doc.symmetry, "crystal_system"):
                        crystal_system = str(doc.symmetry.crystal_system)
                    if hasattr(doc.symmetry, "symbol"):
                        space_group = str(doc.symmetry.symbol)

                elements = [str(e) for e in getattr(doc, "elements", [])]
                bulk_mod = _as_float(getattr(doc, "bulk_modulus", None))
                shear_mod = _as_float(getattr(doc, "shear_modulus", None))
                density = _as_float(getattr(doc, "density", None))
                band_gap = _as_float(getattr(doc, "band_gap", None))

                # Extract piezoelectric data (prefer piezoelectric endpoint)
                if piezo_map:
                    piezo_max = piezo_map.get(str(mp_id))
                else:
                    piezo_max = _as_float(getattr(doc, "e_ij_max", None))

                # Extract dielectric data (prefer dielectric endpoint)
                if dielectric_map:
                    dielectric_const = dielectric_map.get(str(mp_id))
                else:
                    dielectric_const = _dielectric_from_e_total(getattr(doc, "e_total", None))

                example = {
                    "mp_id": mp_id,
                    "formula": doc.formula_pretty,
                    "file": filename,
                    "elements": elements,
                    "energy_per_atom": float(doc.energy_per_atom),
                    "formation_energy_per_atom": float(doc.formation_energy_per_atom),
                    "energy_above_hull": float(doc.energy_above_hull),
                    "nsites": doc.nsites,
                    "nelements": doc.nelements,
                    "crystal_system": crystal_system,
                    "space_group": space_group,
                    "bulk_modulus": bulk_mod,
                    "shear_modulus": shear_mod,
                    "density": density,
                    "band_gap": band_gap,
                    "piezoelectric_coefficient": piezo_max,
                    "dielectric_constant": dielectric_const,
                }
                examples.append(example)
            except Exception as e:
                print(f"\nWarning: failed to download {doc.material_id}: {e}")
                continue

    index_file = output_dir / "index.json"
    index_data = {
        "description": f"Task-specific structures for {task_cfg.get('name', task_key)}",
        "criteria": {
            "energy_above_hull": {"min": min_ed, "max": max_ed},
            "theoretical": theoretical,
            "include_elements": include_elements,
            "exclude_elements": exclude_elements,
            "min_elements": min_elements,
            "max_elements": max_elements,
            "max_sites": max_sites,
            "query_mode": query_mode,
            "bulk_modulus": {"min": bulk_min, "max": bulk_max},
            "shear_modulus": {"min": shear_min, "max": shear_max},
            "density": {"min": density_min, "max": density_max},
            "band_gap": {"min": band_gap_min, "max": band_gap_max},
            "formation_energy": {"min": formation_min, "max": formation_max},
            "piezoelectric_coefficient": {"min": piezo_min, "max": piezo_max},
            "dielectric_constant": {"min": dielectric_min, "max": dielectric_max},
            "require_moduli": require_moduli,
            "require_piezo": require_piezo,
            "require_dielectric": require_dielectric,
            "source": "Materials Project",
        },
        "count": len(examples),
        "examples": examples,
    }

    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"\nIndex saved to: {index_file}")
    print(f"Structure files saved in: {output_dir}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Download task-specific structures from Materials Project")
    parser.add_argument("--task", "-t", required=True, help="Task key or name (config/tasks.yaml)")
    parser.add_argument(
        "--tasks-config",
        type=str,
        default="config/tasks.yaml",
        help="Path to tasks config file (default: config/tasks.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Output directory (default: prior_database/<Task Name>)",
    )
    parser.add_argument("--count", "-n", type=int, default=None, help="Target number of structures")
    parser.add_argument("--min-ed", type=float, default=None, help="Min energy above hull threshold")
    parser.add_argument("--max-ed", type=float, default=None, help="Max energy above hull threshold")
    parser.add_argument("--band-gap-min", type=float, default=None, help="Band gap lower bound override")
    parser.add_argument("--band-gap-max", type=float, default=None, help="Band gap upper bound override")
    parser.add_argument("--dielectric-min", type=float, default=None, help="Dielectric constant lower bound override")
    parser.add_argument("--dielectric-max", type=float, default=None, help="Dielectric constant upper bound override")
    parser.add_argument(
        "--include-elements",
        type=str,
        default="",
        help="Allowed elements, comma-separated",
    )
    parser.add_argument(
        "--exclude-elements",
        type=str,
        default="",
        help="Excluded elements, comma-separated",
    )
    parser.add_argument("--min-elements", type=int, default=None, help="Min number of elements")
    parser.add_argument("--max-elements", type=int, default=None, help="Max number of elements")
    parser.add_argument("--max-sites", type=int, default=None, help="Max number of sites")
    parser.add_argument("--no-theoretical", action="store_true", help="Include experimental structures")
    parser.add_argument(
        "--use-task-constraints",
        action="store_true",
        help="Use task bulk/shear constraints for filtering",
    )
    parser.add_argument(
        "--allow-missing-moduli",
        action="store_true",
        help="Allow structures with missing bulk/shear data",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default="elements",
        choices=["elements", "chemsys", "none"],
        help="MP query mode: elements (default) / chemsys / none",
    )

    args = parser.parse_args()

    tasks_path = Path(args.tasks_config)
    tasks = _load_tasks(tasks_path)
    task_key, task_cfg = _resolve_task(tasks, args.task)
    defaults = _resolve_defaults(task_cfg, args)

    output_dir = args.output or f"prior_database/{task_key}"

    download_task_specific_structures(
        task_key=task_key,
        task_cfg=task_cfg,
        output_dir=Path(output_dir),
        target_count=defaults["target_count"],
        min_ed=defaults["min_ed"],
        max_ed=defaults["max_ed"],
        include_elements=defaults["include_elements"],
        include_mode=defaults["include_mode"],
        exclude_elements=defaults["exclude_elements"],
        min_elements=defaults["min_elements"],
        max_elements=defaults["max_elements"],
        max_sites=defaults["max_sites"],
        theoretical=not args.no_theoretical,
        query_mode=defaults["query_mode"],
        bulk_range=defaults["bulk_range"],
        shear_range=defaults["shear_range"],
        bulk_min=defaults["bulk_min"],
        bulk_max=defaults["bulk_max"],
        shear_min=defaults["shear_min"],
        shear_max=defaults["shear_max"],
        density_min=defaults["density_min"],
        density_max=defaults["density_max"],
        band_gap_min=defaults["band_gap_min"],
        band_gap_max=defaults["band_gap_max"],
        formation_min=defaults["formation_min"],
        formation_max=defaults["formation_max"],
        require_moduli=defaults["require_moduli"],
        require_piezo=defaults["require_piezo"],
        require_dielectric=defaults["require_dielectric"],
        piezo_min=defaults["piezo_min"],
        piezo_max=defaults["piezo_max"],
        dielectric_min=defaults["dielectric_min"],
        dielectric_max=defaults["dielectric_max"],
    )


if __name__ == "__main__":
    main()
