"""Structure operators for material generation

Provides atomic operations for structure manipulation:
- Substitution: Replace elements
- Mutation: Perturb structure
- Mix: Combine parent structures
- Prototype fill: Build from standard templates
- Crossover: Exchange structural fragments between parents
- Doping: Trace impurity substitution (0.1%-10%)
"""

import numpy as np
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict
from src.core.structure import CrystalStructure

logger = logging.getLogger(__name__)

# Prototype library for fill_prototype tool (fractional coordinates in cubic cell)
_PROTOTYPE_LIBRARY: Dict[str, Dict] = {
    "rock-salt": {
        "display_name": "Rock-salt",
        "min_elements": 2,
        "max_elements": 2,
        "nn_factor": 0.5,  # nearest neighbor distance = 0.5 * a
        "nn_pair": (0, 1),
        "sites": [
            (0, [0.0, 0.0, 0.0]),
            (0, [0.0, 0.5, 0.5]),
            (0, [0.5, 0.0, 0.5]),
            (0, [0.5, 0.5, 0.0]),
            (1, [0.5, 0.5, 0.5]),
            (1, [0.0, 0.0, 0.5]),
            (1, [0.0, 0.5, 0.0]),
            (1, [0.5, 0.0, 0.0]),
        ],
    },
    "diamond": {
        "display_name": "Diamond",
        "min_elements": 1,
        "max_elements": 2,
        "nn_factor": 0.4330127019,  # sqrt(3)/4
        "nn_pair": (0, 0),
        "sites": [
            (0, [0.0, 0.0, 0.0]),
            (0, [0.0, 0.5, 0.5]),
            (0, [0.5, 0.0, 0.5]),
            (0, [0.5, 0.5, 0.0]),
            (1, [0.25, 0.25, 0.25]),
            (1, [0.25, 0.75, 0.75]),
            (1, [0.75, 0.25, 0.75]),
            (1, [0.75, 0.75, 0.25]),
        ],
    },
    "zinc-blende": {
        "display_name": "Zinc-blende",
        "min_elements": 2,
        "max_elements": 2,
        "nn_factor": 0.4330127019,  # sqrt(3)/4
        "nn_pair": (0, 1),
        "sites": [
            (0, [0.0, 0.0, 0.0]),
            (0, [0.0, 0.5, 0.5]),
            (0, [0.5, 0.0, 0.5]),
            (0, [0.5, 0.5, 0.0]),
            (1, [0.25, 0.25, 0.25]),
            (1, [0.25, 0.75, 0.75]),
            (1, [0.75, 0.25, 0.75]),
            (1, [0.75, 0.75, 0.25]),
        ],
    },
    "fluorite": {
        "display_name": "Fluorite",
        "min_elements": 2,
        "max_elements": 2,
        "nn_factor": 0.4330127019,  # sqrt(3)/4
        "nn_pair": (0, 1),
        "sites": [
            (0, [0.0, 0.0, 0.0]),
            (0, [0.0, 0.5, 0.5]),
            (0, [0.5, 0.0, 0.5]),
            (0, [0.5, 0.5, 0.0]),
            (1, [0.25, 0.25, 0.25]),
            (1, [0.25, 0.25, 0.75]),
            (1, [0.25, 0.75, 0.25]),
            (1, [0.25, 0.75, 0.75]),
            (1, [0.75, 0.25, 0.25]),
            (1, [0.75, 0.25, 0.75]),
            (1, [0.75, 0.75, 0.25]),
            (1, [0.75, 0.75, 0.75]),
        ],
    },
    "perovskite": {
        "display_name": "Perovskite",
        "min_elements": 3,
        "max_elements": 3,
        "nn_factor": 0.5,  # B-O bond distance = 0.5 * a
        "nn_pair": (1, 2),
        "sites": [
            (0, [0.0, 0.0, 0.0]),  # A
            (1, [0.5, 0.5, 0.5]),  # B
            (2, [0.5, 0.5, 0.0]),  # O
            (2, [0.5, 0.0, 0.5]),
            (2, [0.0, 0.5, 0.5]),
        ],
    },
}

# Alternate names for prototypes
_PROTOTYPE_ALIASES = {
    "rocksalt": "rock-salt",
    "nacl": "rock-salt",
    "zincblende": "zinc-blende",
    "zb": "zinc-blende",
    "perov": "perovskite",
}


def _normalize_template_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-").replace(" ", "-")
    return _PROTOTYPE_ALIASES.get(key, key)


def _atomic_radius(symbol: str) -> float:
    from pymatgen.core import Element
    elem = Element(symbol)
    radius = elem.atomic_radius or elem.covalent_radius or 1.5
    return float(radius)


class StructureOperators:
    """
    Atomic operations for structure generation

    These are the "tools" that agents can call to generate new structures
    """

    @classmethod
    def register_prior_templates(
        cls,
        index_path: Path,
        seed_path: Path,
        max_templates: int = 10,
        max_elements: int = 3
    ) -> List[Dict]:
        """
        Register task-specific templates from a prior database index.

        Returns:
            List of template metadata for prompt usage.
        """
        index_path = Path(index_path)
        seed_path = Path(seed_path)
        if not index_path.exists():
            return []

        try:
            with index_path.open() as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read prior index {index_path}: {e}")
            return []

        examples = data.get("examples", [])
        def _safe_float(value: Optional[float]) -> Optional[float]:
            try:
                return float(value)
            except Exception:
                return None

        candidates = []
        for ex in examples:
            nelements = ex.get("nelements", 99)
            if nelements > max_elements:
                continue
            file_name = ex.get("file")
            if not file_name:
                continue
            file_path = seed_path / file_name
            if not file_path.exists():
                continue
            ed = ex.get("energy_above_hull")
            ed_val = ed if ed is not None else float("inf")
            candidates.append((ed_val, ex, file_path))

        candidates.sort(key=lambda x: x[0])

        templates = []
        used_formulas = set()
        template_idx = 1
        for _, ex, file_path in candidates:
            if len(templates) >= max_templates:
                break
            formula = ex.get("formula")
            if formula in used_formulas:
                continue

            try:
                poscar_text = file_path.read_text()
                struct = CrystalStructure.from_poscar(poscar_text)
            except Exception as e:
                logger.warning(f"Failed to parse template {file_path.name}: {e}")
                continue

            species_order = []
            for s in struct.species:
                if s not in species_order:
                    species_order.append(s)

            if len(species_order) > max_elements:
                continue

            species_index = {s: i for i, s in enumerate(species_order)}
            sites = []
            for s, pos in zip(struct.species, struct.positions):
                sites.append((species_index[s], [float(x) for x in pos]))

            reference_radii = [_atomic_radius(s) for s in species_order]
            template_name = f"prior_{template_idx:02d}"
            template_key = _normalize_template_name(template_name)
            if template_key in _PROTOTYPE_LIBRARY:
                template_idx += 1
                continue

            _PROTOTYPE_LIBRARY[template_key] = {
                "display_name": template_name,
                "min_elements": len(species_order),
                "max_elements": len(species_order),
                "lattice": struct.lattice.tolist(),
                "sites": sites,
                "reference_radii": reference_radii,
                "species_order": species_order,
                "source": "prior",
            }

            ratio = None
            reduced_formula = None
            try:
                from collections import Counter
                from math import gcd
                from functools import reduce
                from pymatgen.core import Composition
                counts = Counter(struct.species)
                ratio_vals = [counts[s] for s in species_order]
                ratio_gcd = reduce(gcd, ratio_vals) if ratio_vals else 1
                ratio = ":".join(str(v // ratio_gcd) for v in ratio_vals)
                reduced_formula = Composition(struct.species).reduced_formula
            except Exception:
                pass

            templates.append({
                "name": template_name,
                "reference_formula": formula or struct.formula,
                "elements": species_order,
                "crystal_system": ex.get("crystal_system"),
                "space_group": ex.get("space_group"),
                "nsites": ex.get("nsites"),
                "ratio": ratio,
                "reduced_formula": reduced_formula,
                "decomposition_energy": _safe_float(ex.get("energy_above_hull")),
                "density": _safe_float(ex.get("density")),
                "bulk_modulus": _safe_float(ex.get("bulk_modulus")),
                "shear_modulus": _safe_float(ex.get("shear_modulus")),
                "piezoelectric_coefficient": _safe_float(ex.get("piezoelectric_coefficient")),
                "dielectric_constant": _safe_float(ex.get("dielectric_constant")),
            })
            used_formulas.add(formula or struct.formula)
            template_idx += 1

        return templates

    @classmethod
    def register_dynamic_templates_from_pool(
        cls,
        parent_pool: List[CrystalStructure],
        max_templates: int = 10,
        max_elements: int = 4,
        score_key: str = 'weighted_score'
    ) -> List[Dict]:
        """
        Dynamically register templates from current parent pool.

        This extracts high-quality structures from the parent pool and registers
        them as templates for fill_prototype. These templates reflect current
        search progress and are guaranteed to satisfy constraints.

        Args:
            parent_pool: Current parent pool structures
            max_templates: Maximum number of templates to extract
            max_elements: Maximum elements per template
            score_key: Property to sort by ('weighted_score', 'decomposition_energy', etc.)

        Returns:
            List of template metadata for prompt usage
        """
        if not parent_pool:
            logger.warning("Parent pool is empty, cannot extract dynamic templates")
            return []

        def _safe_float(value: Optional[float]) -> Optional[float]:
            try:
                return float(value)
            except Exception:
                return None

        # Filter candidates by element count
        candidates = []
        for struct in parent_pool:
            unique_elements = list(set(struct.species))
            if len(unique_elements) > max_elements:
                continue

            # Get sorting metric
            if score_key == 'weighted_score' and hasattr(struct, 'weighted_score'):
                sort_val = -struct.weighted_score  # Higher is better, so negate for ascending sort
            elif score_key == 'decomposition_energy':
                sort_val = struct.decomposition_energy if struct.decomposition_energy is not None else float('inf')
            else:
                sort_val = struct.properties.get(score_key, float('inf'))

            candidates.append((sort_val, struct))

        # Sort by metric (ascending, so best first for Ed, worst first for score due to negation)
        candidates.sort(key=lambda x: x[0])

        templates = []
        template_idx = 1

        for _, struct in candidates:
            if len(templates) >= max_templates:
                break

            # Get unique elements in order
            species_order = []
            for s in struct.species:
                if s not in species_order:
                    species_order.append(s)

            # Build species index
            species_index = {s: i for i, s in enumerate(species_order)}
            sites = []
            for s, pos in zip(struct.species, struct.positions):
                sites.append((species_index[s], [float(x) for x in pos]))

            # Calculate reference radii
            reference_radii = [_atomic_radius(s) for s in species_order]

            # Register in global prototype library with 'parent_' prefix
            template_name = f"parent_{template_idx:02d}"
            template_key = _normalize_template_name(template_name)

            # Overwrite if exists (allows dynamic updates)
            _PROTOTYPE_LIBRARY[template_key] = {
                "display_name": template_name,
                "min_elements": len(species_order),
                "max_elements": len(species_order),
                "lattice": struct.lattice.tolist(),
                "sites": sites,
                "reference_radii": reference_radii,
                "species_order": species_order,
                "source": "parent_pool",  # Mark as dynamic
            }

            # Build stoichiometry ratio
            ratio = None
            reduced_formula = None
            crystal_system = None
            try:
                from collections import Counter
                from math import gcd
                from functools import reduce
                from pymatgen.core import Composition

                counts = Counter(struct.species)
                ratio_vals = [counts[s] for s in species_order]
                ratio_gcd = reduce(gcd, ratio_vals) if ratio_vals else 1
                ratio = ":".join(str(v // ratio_gcd) for v in ratio_vals)
                reduced_formula = Composition(struct.species).reduced_formula

                # Try to get crystal system from pymatgen
                pmg_struct = struct.to_pymatgen()
                crystal_system = pmg_struct.get_space_group_info()[1]
            except Exception:
                pass

            # Metadata for prompt
            templates.append({
                "name": template_name,
                "reference_formula": struct.formula,
                "elements": species_order,
                "crystal_system": crystal_system,
                "ratio": ratio,
                "reduced_formula": reduced_formula or struct.formula,
                "weighted_score": getattr(struct, 'weighted_score', None),
                "decomposition_energy": struct.decomposition_energy,
                "density": _safe_float(
                    struct.properties.get("density")
                    if struct.properties.get("density") is not None
                    else (struct.density if hasattr(struct, "density") else None)
                ),
                "bulk_modulus": _safe_float(struct.properties.get("bulk_modulus")),
                "shear_modulus": _safe_float(struct.properties.get("shear_modulus")),
                "piezoelectric_coefficient": _safe_float(struct.properties.get("piezoelectric_coefficient")),
                "dielectric_constant": _safe_float(struct.properties.get("dielectric_constant")),
                "band_gap": _safe_float(struct.properties.get("band_gap")),
                "formation_energy": _safe_float(struct.properties.get("formation_energy")),
                "nsites": len(struct.species),
            })
            template_idx += 1

        logger.info(
            f"Registered {len(templates)} dynamic templates from parent pool "
            f"(sorted by {score_key})"
        )

        return templates

    @staticmethod
    def substitute_element(
        parent: CrystalStructure,
        old_element: str,
        new_element: str
    ) -> CrystalStructure:
        """
        Substitute element(s) in a structure. Supports comma-separated
        multi-element substitution (e.g. old='Ti,O', new='Zr,S').

        Args:
            parent: Parent structure
            old_element: Element(s) to replace (comma-separated for multi)
            new_element: Replacement element(s) (comma-separated for multi)

        Returns:
            New structure with substituted element(s)

        Examples:
            ZnO + substitute(O→S) → ZnS
            BaTiO3 + substitute(Ba,Ti → Sr,Zr) → SrZrO3
        """
        old_list = [e.strip() for e in old_element.split(",")]
        new_list = [e.strip() for e in new_element.split(",")]
        if len(old_list) != len(new_list):
            raise ValueError(
                f"Mismatched substitution count: {old_list} → {new_list}"
            )

        sub_map = dict(zip(old_list, new_list))
        sub_desc = " + ".join(f"{o}→{n}" for o, n in sub_map.items())
        logger.info(f"Substituting {sub_desc} in {parent.formula}")

        # Replace elements in species list
        new_species = [sub_map.get(s, s) for s in parent.species]

        # Update formula
        new_formula = parent.formula
        for old_e, new_e in sub_map.items():
            new_formula = new_formula.replace(old_e, new_e)

        # Create new structure
        child = CrystalStructure(
            formula=new_formula,
            lattice=parent.lattice.copy(),
            positions=parent.positions.copy(),
            species=new_species
        )

        # Track provenance
        child.metadata['source'] = 'substitute'
        child.metadata['parent'] = parent.formula
        child.metadata['substitution'] = sub_desc

        return child

    @staticmethod
    def mutate_structure(
        parent: CrystalStructure,
        strength: float = 0.1
    ) -> CrystalStructure:
        """
        Mutate a structure with random perturbations

        Args:
            parent: Parent structure
            strength: Mutation strength (0.01-0.1 recommended)
                     Controls magnitude of random changes

        Returns:
            Mutated structure

        Example:
            NaCl + mutate(0.1) → NaCl' (slightly perturbed)
        """
        logger.info(f"Mutating {parent.formula} with strength={strength}")

        # Perturb atomic positions
        new_positions = parent.positions.copy()
        perturbation = np.random.uniform(-strength, strength, new_positions.shape)
        new_positions += perturbation

        # Wrap to unit cell [0, 1)
        new_positions = np.mod(new_positions, 1.0)

        # Optionally perturb lattice (smaller magnitude)
        new_lattice = parent.lattice.copy()
        if np.random.random() < 0.5:  # 50% chance to also perturb lattice
            lattice_scale = 1.0 + np.random.uniform(-strength/2, strength/2)
            new_lattice *= lattice_scale

        # Create new structure
        child = CrystalStructure(
            formula=parent.formula,
            lattice=new_lattice,
            positions=new_positions,
            species=parent.species.copy()
        )

        # Track provenance
        child.metadata['source'] = 'mutate'
        child.metadata['parent'] = parent.formula
        child.metadata['mutation_strength'] = strength

        return child

    @staticmethod
    def mix_structures(
        parent1: CrystalStructure,
        parent2: CrystalStructure,
        ratio: float = 0.5,
        min_distance_factor: float = 0.75,
        max_collision_iters: int = 3
    ) -> CrystalStructure:
        """
        Composition-dominant global recombination of two parent structures.

        Primary intent: expand chemical system by recombining parent compositions.
        Coupled effect: lattice/topology are also reorganized during mixing.

        Args:
            parent1: First parent
            parent2: Second parent
            ratio: Mixing ratio (fraction of parent1, 0-1)

        Returns:
            Mixed structure

        Example:
            LiCoO2 + LiNiO2 + mix(ratio=0.7) → 70/30 mixed lattice
        """
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            ratio = 0.5
        ratio = max(0.0, min(1.0, ratio))
        logger.info(
            f"Mixing {parent1.formula} × {parent2.formula} (ratio={ratio:.2f})"
        )

        # Interpolate lattice parameters
        new_lattice = ratio * parent1.lattice + (1 - ratio) * parent2.lattice

        # Simple 50/50 atom mix: take half of each parent's sites
        import random
        idx1 = list(range(len(parent1.species)))
        idx2 = list(range(len(parent2.species)))
        random.shuffle(idx1)
        random.shuffle(idx2)
        n1_keep = max(1, int(round(len(idx1) * ratio)))
        n2_keep = max(1, int(round(len(idx2) * (1 - ratio))))
        idx1 = idx1[:n1_keep]
        idx2 = idx2[:n2_keep]

        new_positions = np.vstack([
            parent1.positions[idx1],
            parent2.positions[idx2],
        ])
        new_species = [parent1.species[i] for i in idx1] + [parent2.species[i] for i in idx2]

        removed_atoms = 0
        if min_distance_factor and min_distance_factor > 0:
            try:
                from pymatgen.core import Structure

                positions = new_positions
                species = new_species
                for _ in range(max_collision_iters):
                    if len(species) < 2:
                        break
                    struct = Structure(new_lattice, species, positions, coords_are_cartesian=False)
                    dist_mat = struct.distance_matrix
                    radii = [_atomic_radius(s) for s in species]
                    to_remove = set()
                    n = len(species)
                    for i in range(n):
                        if i in to_remove:
                            continue
                        for j in range(i + 1, n):
                            if j in to_remove:
                                continue
                            min_dist = min_distance_factor * (radii[i] + radii[j])
                            if dist_mat[i][j] < min_dist:
                                to_remove.add(j)
                    if not to_remove:
                        break
                    keep = [idx for idx in range(n) if idx not in to_remove]
                    if not keep:
                        break
                    positions = positions[keep]
                    species = [species[idx] for idx in keep]
                    removed_atoms += len(to_remove)

                new_positions = positions
                new_species = species
                if removed_atoms:
                    logger.info(
                        f"Mix collision cleanup removed {removed_atoms} atoms "
                        f"(min_distance_factor={min_distance_factor})"
                    )
            except Exception as e:
                logger.warning(f"Mix collision cleanup failed: {e}")

        # Generate mixed formula
        from collections import Counter
        from pymatgen.core import Composition
        counts = Counter(new_species)
        new_formula = Composition(counts).reduced_formula

        # Create new structure
        child = CrystalStructure(
            formula=new_formula,
            lattice=new_lattice,
            positions=new_positions,
            species=new_species
        )

        # Track provenance
        child.metadata['source'] = 'mix'
        child.metadata['parent1'] = parent1.formula
        child.metadata['parent2'] = parent2.formula
        child.metadata['mix_ratio'] = ratio
        if removed_atoms:
            child.metadata['collision_removals'] = removed_atoms

        return child

    @staticmethod
    def fill_prototype(
        template_name: str,
        elements: List[str]
    ) -> CrystalStructure:
        """
        Structure-dominant global jump via prototype templating.

        Primary intent: switch to a target structural topology (template).
        Coupled effect: composition adapts to template slot requirements.

        Args:
            template_name: Prototype name (e.g., Rock-salt, Diamond, Perovskite)
            elements: Element symbols ordered by prototype definition

        Returns:
            CrystalStructure built from the prototype
        """
        if not template_name:
            raise ValueError("template_name is required for fill_prototype")

        key = _normalize_template_name(template_name)
        if key not in _PROTOTYPE_LIBRARY:
            raise ValueError(f"Unknown prototype template: {template_name}")

        proto = _PROTOTYPE_LIBRARY[key]
        min_elems = proto["min_elements"]
        max_elems = proto["max_elements"]
        elements = [str(e).strip() for e in elements if str(e).strip()]
        from pymatgen.core import Element
        cleaned = []
        for symbol in elements:
            try:
                cleaned.append(Element(symbol).symbol)
            except Exception:
                cleaned.append(symbol)
        elements = cleaned
        if len(elements) < min_elems:
            raise ValueError(
                f"{proto['display_name']} requires at least {min_elems} elements"
            )
        if len(elements) > max_elems:
            logger.warning(
                f"{proto['display_name']} expects {max_elems} elements; "
                f"truncating from {len(elements)}"
            )
            elements = elements[:max_elems]

        # Compute lattice parameter from atomic radii.
        # Keep existing radius-based scaling behavior unchanged for reproducibility.
        lattice_a = None
        if "lattice" in proto:
            lattice = np.array(proto["lattice"], dtype=float)
            scale = 1.0
            ref_radii = proto.get("reference_radii")
            if ref_radii:
                ratios = []
                for i in range(min(len(ref_radii), len(elements))):
                    if ref_radii[i] > 0:
                        ratios.append(_atomic_radius(elements[i]) / ref_radii[i])
                if ratios:
                    scale = sum(ratios) / len(ratios)
            lattice = lattice * scale
            lattice_a = float(np.linalg.norm(lattice[0]))
        else:
            idx_a, idx_b = proto["nn_pair"]
            if key == "diamond" and len(elements) >= 2:
                idx_b = 1
            elem_a = elements[idx_a] if idx_a < len(elements) else elements[0]
            elem_b = elements[idx_b] if idx_b < len(elements) else elements[0]
            target_nn = _atomic_radius(elem_a) + _atomic_radius(elem_b)
            nn_factor = proto["nn_factor"]
            a = max(target_nn / nn_factor, 2.5)

            lattice = np.eye(3) * a
            lattice_a = float(a)
        positions = []
        species = []
        for sub_idx, frac in proto["sites"]:
            elem_idx = min(sub_idx, len(elements) - 1)
            species.append(elements[elem_idx])
            positions.append(frac)

        positions = np.array(positions, dtype=float)

        # Build formula
        from collections import Counter
        from pymatgen.core import Composition
        counts = Counter(species)
        formula = Composition(counts).reduced_formula

        child = CrystalStructure(
            formula=formula,
            lattice=lattice,
            positions=positions,
            species=species
        )

        child.metadata["source"] = "fill_prototype"
        child.metadata["prototype"] = proto["display_name"]
        child.metadata["elements"] = elements
        if lattice_a is not None:
            child.metadata["lattice_a"] = lattice_a

        return child

    @staticmethod
    def crossover_structures(
        parent1: CrystalStructure,
        parent2: CrystalStructure,
        cut_point: Optional[float] = None,
        axis: int = 0
    ) -> CrystalStructure:
        """
        Crossover two parent structures using cut-and-splice method

        Splits both parents along a specified axis and combines atoms from each half.
        This implements classical genetic algorithm crossover for crystal structures.

        Args:
            parent1: First parent structure
            parent2: Second parent structure
            cut_point: Cut position along axis (0-1). If None, random cut at 0.3-0.7
            axis: Axis to cut along (0=a, 1=b, 2=c)

        Returns:
            Child structure with atoms from both parents

        Example:
            NaCl + LiF + crossover(cut=0.5, axis=2)
            → Bottom half from NaCl, top half from LiF

        Note:
            - Different from mix_structures which only interpolates lattice
            - This exchanges actual atomic fragments between parents
            - More similar to biological crossover/recombination
        """
        # Random cut point if not specified
        if cut_point is None:
            cut_point = np.random.uniform(0.3, 0.7)

        logger.info(
            f"Crossover {parent1.formula} × {parent2.formula} "
            f"(axis={axis}, cut={cut_point:.2f})"
        )

        # Interpolate lattice (similar to mix, but could also use parent1's lattice)
        # Using 50-50 mix as default for crossover
        new_lattice = 0.5 * parent1.lattice + 0.5 * parent2.lattice

        # Get atomic positions and species from both parents
        pos1 = parent1.positions.copy()
        pos2 = parent2.positions.copy()
        species1 = parent1.species.copy()
        species2 = parent2.species.copy()

        # Select atoms based on position along the cut axis
        # Atoms from parent1 if position < cut_point, else from parent2
        mask1 = pos1[:, axis] < cut_point
        mask2 = pos2[:, axis] >= cut_point

        # Combine selected atoms
        selected_pos1 = pos1[mask1]
        selected_species1 = [species1[i] for i in range(len(species1)) if mask1[i]]

        selected_pos2 = pos2[mask2]
        selected_species2 = [species2[i] for i in range(len(species2)) if mask2[i]]

        # Merge atoms from both parents
        new_positions = np.vstack([selected_pos1, selected_pos2])
        new_species = selected_species1 + selected_species2

        # Generate descriptive formula
        # Count atoms from each parent
        n1 = len(selected_species1)
        n2 = len(selected_species2)
        new_formula = f"{parent1.formula}[{n1}]x{parent2.formula}[{n2}]"

        # Create child structure
        child = CrystalStructure(
            formula=new_formula,
            lattice=new_lattice,
            positions=new_positions,
            species=new_species
        )

        # Track provenance
        child.metadata['source'] = 'crossover'
        child.metadata['parent1'] = parent1.formula
        child.metadata['parent2'] = parent2.formula
        child.metadata['cut_point'] = cut_point
        child.metadata['cut_axis'] = axis
        child.metadata['atoms_from_parent1'] = n1
        child.metadata['atoms_from_parent2'] = n2

        return child

    @staticmethod
    def dope_structure(
        parent: CrystalStructure,
        dopant: Optional[str] = None,
        concentration: Optional[float] = None,
        host_element: Optional[str] = None
    ) -> CrystalStructure:
        """
        Dope structure with trace impurities (0.1% - 10%)

        Introduces small amounts of foreign elements to modify electronic/defect structure.
        This is a key technique in semiconductor and materials engineering.

        Args:
            parent: Parent structure to dope
            dopant: Dopant element (if None, auto-select from adjacent groups)
            concentration: Doping concentration 0-1 (if None, random 0.01-0.10)
            host_element: Element to replace (if None, auto-select metallic element)

        Returns:
            Doped structure

        Example:
            GaN + dope(dopant='Mg', concentration=0.05)
            → Ga_{0.95}Mg_{0.05}N (p-type semiconductor)
        """
        import random
        from collections import Counter

        logger.info(f"Doping {parent.formula}")

        # Import here to avoid dependency issues if pymatgen not available
        try:
            from pymatgen.core.periodic_table import Element
            use_pymatgen = True
        except ImportError:
            logger.warning("pymatgen not available, using simplified doping logic")
            use_pymatgen = False

        # 1. Auto-select host element if not specified
        if host_element is None:
            unique_species = list(set(parent.species))

            if use_pymatgen:
                # Prefer metallic elements (common for doping)
                metallic = [s for s in unique_species if Element(s).is_metal]
                if metallic:
                    host_element = random.choice(metallic)
                else:
                    host_element = random.choice(unique_species)
            else:
                # Simple fallback: choose most common element
                counts = Counter(parent.species)
                host_element = counts.most_common(1)[0][0]

            logger.info(f"  Auto-selected host element: {host_element}")

        # Verify host element exists in structure
        if host_element not in parent.species:
            raise ValueError(f"Host element {host_element} not found in {parent.formula}")

        # 2. Auto-select dopant if not specified
        if dopant is None:
            candidates = []

            if use_pymatgen:
                host_elem = Element(host_element)

                # Try adjacent groups in periodic table
                try:
                    for delta_group in [-1, 1]:
                        new_group = host_elem.group + delta_group
                        if 1 <= new_group <= 18:
                            # Find elements in same/adjacent period, adjacent group
                            for elem_symbol in ['Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                                               'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                                               'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                                               'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I']:
                                try:
                                    elem = Element(elem_symbol)
                                    if (elem.group == new_group and
                                        abs(elem.row - host_elem.row) <= 1 and
                                        elem_symbol != host_element and
                                        elem_symbol not in parent.species):
                                        candidates.append(elem_symbol)
                                except:
                                    pass
                except:
                    pass

            # Common dopants for semiconductors and functional materials
            common_dopants = ['B', 'Al', 'Ga', 'In', 'N', 'P', 'As', 'Sb',
                            'Mg', 'Zn', 'Cd', 'Si', 'Ge', 'Sn']
            candidates.extend([d for d in common_dopants if d != host_element and d not in parent.species])

            if candidates:
                dopant = random.choice(list(set(candidates)))
            else:
                # Fall back to light elements
                fallback = ['B', 'C', 'N', 'Si', 'P', 'S']
                dopant = random.choice([f for f in fallback if f != host_element])

            logger.info(f"  Auto-selected dopant: {dopant}")

        # 3. Auto-select concentration if not specified
        if concentration is None:
            # Typical doping range: 1% - 10%
            concentration = random.uniform(0.01, 0.10)
            logger.info(f"  Auto-selected concentration: {concentration:.3f} ({concentration*100:.1f}%)")

        # Validate concentration
        if not 0 < concentration < 1:
            raise ValueError(f"Concentration must be between 0 and 1, got {concentration}")

        # 4. Partial replacement based on concentration
        new_species = parent.species.copy()
        host_indices = [i for i, s in enumerate(new_species) if s == host_element]

        if not host_indices:
            raise ValueError(f"No {host_element} atoms found to dope")

        # Calculate number of atoms to replace
        n_dope = max(1, int(len(host_indices) * concentration))
        n_dope = min(n_dope, len(host_indices) - 1)  # Keep at least 1 host atom

        # Randomly select which atoms to dope
        doped_indices = random.sample(host_indices, n_dope)
        for idx in doped_indices:
            new_species[idx] = dopant

        # 5. Update formula to reflect doping
        new_formula = f"{parent.formula}_doped_{dopant}{concentration*100:.0f}pct"

        # Create doped structure
        child = CrystalStructure(
            formula=new_formula,
            lattice=parent.lattice.copy(),
            positions=parent.positions.copy(),
            species=new_species
        )

        # 6. Track metadata
        child.metadata['source'] = 'dope'
        child.metadata['parent'] = parent.formula
        child.metadata['host_element'] = host_element
        child.metadata['dopant'] = dopant
        child.metadata['concentration'] = concentration
        child.metadata['n_doped_atoms'] = n_dope
        child.metadata['n_total_host_atoms'] = len(host_indices)

        logger.info(f"  Doped {n_dope}/{len(host_indices)} {host_element} atoms with {dopant}")
        logger.info(f"  Result: {new_formula}")

        return child
