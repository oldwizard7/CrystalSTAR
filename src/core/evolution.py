"""Evolutionary algorithm for crystal structure search"""

from typing import List, Dict, Optional
import math
import numpy as np
from pathlib import Path
import logging

from .structure import CrystalStructure

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """
    Evolutionary algorithm engine
    
    Implements:
    - Population initialization
    - Selection
    - Ranking
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evolution engine

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.population_size = config['evolution']['population_size']
        self.parent_size = config['evolution']['parent_size']

        # Selection strategy
        self.objective = config['objective'].get('type', 'stability')
        if (
            self.objective == 'stability'
            and 'targets' in config.get('objective', {})
            and 'weights' in config.get('objective', {})
        ):
            # Default to multi-objective when targets/weights are provided.
            self.objective = 'multi_objective'

        # Support both old and new objective formats
        if 'targets' in config['objective']:
            # New format: targets dictionary
            self.target_ed = config['objective']['targets'].get('decomposition_energy', 0.0)
        else:
            # Old format: direct target_ed
            self.target_ed = config['objective'].get('target_ed', 0.0)

        # Load hard constraints from config
        self.hard_constraints = config.get('constraints', {})
    
    def initialize_population(
        self,
        data_path: str,
        filters: Optional[Dict] = None
    ) -> List[CrystalStructure]:
        """
        Initialize parent population from database
        
        Args:
            data_path: Path to structure database
            filters: Optional filters (e.g., element types, num_elements)
            
        Returns:
            Initial population of structures
        """
        logger.info(f"Initializing population from {data_path}")
        
        # Load structures from data path
        structures = self._load_structures(data_path, filters)
        
        # Sample initial population
        if len(structures) < self.population_size * self.parent_size:
            logger.warning(
                f"Not enough structures. Found {len(structures)}, "
                f"need {self.population_size * self.parent_size}"
            )
            return structures
        
        # Sort by stability (or other criteria)
        sorted_structures = sorted(
            structures,
            key=lambda s: s.decomposition_energy if s.decomposition_energy else float('inf')
        )
        
        # Select top structures
        selected = sorted_structures[:self.population_size * self.parent_size]
        
        logger.info(f"Selected {len(selected)} structures for initial population")
        
        return selected
    
    def _load_structures(
        self,
        data_path: str,
        filters: Optional[Dict]
    ) -> List[CrystalStructure]:
        """Load structures from database"""
        structures = []
        data_dir = Path(data_path)
        
        if not data_dir.exists():
            logger.error(f"Data path does not exist: {data_path}")
            return structures
        
        # Load POSCAR files
        for file_path in data_dir.glob("*.vasp"):
            try:
                with open(file_path, 'r') as f:
                    poscar_str = f.read()
                struct = CrystalStructure.from_poscar(poscar_str)
                
                # Apply filters
                if filters and not self._passes_filters(struct, filters):
                    continue
                
                structures.append(struct)
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        # Load JSON files
        for file_path in data_dir.glob("*.json"):
            try:
                struct = CrystalStructure.load(str(file_path))
                
                if filters and not self._passes_filters(struct, filters):
                    continue
                
                structures.append(struct)
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(structures)} structures")
        return structures
    
    def _passes_filters(self, structure: CrystalStructure, filters: Dict) -> bool:
        """Check if structure passes filters"""
        # Number of elements
        if 'num_elements' in filters:
            min_elem, max_elem = filters['num_elements']
            num_unique = len(set(structure.species))
            if not (min_elem <= num_unique <= max_elem):
                return False
        
        # Exclude elements
        if 'exclude_elements' in filters:
            if any(elem in structure.species for elem in filters['exclude_elements']):
                return False
        
        # Include only certain elements
        if 'include_elements' in filters:
            if not all(elem in filters['include_elements'] for elem in structure.species):
                return False
        
        return True
    
    def select(
        self,
        parents: List[CrystalStructure],
        children: List[CrystalStructure],
        extra_pool: Optional[List[CrystalStructure]] = None
    ) -> List[CrystalStructure]:
        """
        Select next generation
        
        Args:
            parents: Current parent pool
            children: Newly generated children
            extra_pool: Optional extra structures
            
        Returns:
            Next generation parent pool
        """
        # Combine all candidates
        candidates = parents + children
        if extra_pool:
            candidates += extra_pool
        
        # Remove duplicates
        candidates = self._remove_duplicates(candidates)
        
        # Rank by objective
        ranked = self._rank_structures(candidates)
        
        # Select top K*P
        top_k = self.population_size * self.parent_size
        selected = ranked[:top_k]
        
        logger.info(
            f"Selected {len(selected)} structures from {len(candidates)} candidates"
        )
        
        return selected
    
    def _remove_duplicates(
        self,
        structures: List[CrystalStructure]
    ) -> List[CrystalStructure]:
        """
        Remove duplicate structures using StructureMatcher

        Uses pymatgen's StructureMatcher to identify truly identical structures,
        not just same chemical formula (which can have different polymorphs).
        """
        if not structures:
            return []

        try:
            from pymatgen.analysis.structure_matcher import StructureMatcher

            # Use StructureMatcher with standard tolerances
            matcher = StructureMatcher(
                ltol=0.2,      # Lattice tolerance (Å)
                stol=0.3,      # Site tolerance
                angle_tol=5,   # Angle tolerance (degrees)
            )

            unique = []

            for struct in structures:
                is_duplicate = False

                try:
                    # Convert to pymatgen Structure
                    pmg_struct = struct.to_pymatgen()

                    # Compare with all unique structures found so far
                    for unique_struct in unique:
                        unique_pmg = unique_struct.to_pymatgen()

                        if matcher.fit(pmg_struct, unique_pmg):
                            # Found a duplicate!
                            is_duplicate = True
                            logger.debug(f"Duplicate found: {struct.formula} matches existing structure")
                            break

                    if not is_duplicate:
                        unique.append(struct)

                except Exception as e:
                    # If structure matching fails, fall back to keeping it
                    logger.warning(f"Structure matching failed for {struct.formula}: {e}")
                    unique.append(struct)

            duplicates_removed = len(structures) - len(unique)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate structures (kept {len(unique)} unique)")

            return unique

        except ImportError:
            # Fallback to formula-based deduplication if pymatgen not available
            logger.warning("pymatgen not available, using formula-based deduplication")
            unique = []
            seen_formulas = set()

            for struct in structures:
                if struct.formula not in seen_formulas:
                    unique.append(struct)
                    seen_formulas.add(struct.formula)

            return unique
    
    def _rank_structures(
        self,
        structures: List[CrystalStructure]
    ) -> List[CrystalStructure]:
        """
        Rank structures by objective
        
        Args:
            structures: List of structures to rank
            
        Returns:
            Sorted list (best first)
        """
        if self.objective == "stability":
            # Rank by decomposition energy (lower is better)
            return sorted(
                structures,
                key=lambda s: s.decomposition_energy if s.decomposition_energy is not None else float('inf')
            )
        
        elif self.objective == "bulk_modulus":
            # Rank by bulk modulus (higher is better)
            return sorted(
                structures,
                key=lambda s: s.properties.get('bulk_modulus', 0),
                reverse=True
            )
        
        elif self.objective == "multi_objective":
            # Weighted sum of objectives (score in [0, 1], higher is better)
            obj_cfg = self.config.get('objective', {})
            targets = obj_cfg.get('targets', {}) or {}
            weights = obj_cfg.get('weights', {}) or {}
            if not targets or not weights:
                logger.warning("Multi-objective requested but targets/weights are missing; using stability ranking")
                return sorted(
                    structures,
                    key=lambda s: s.decomposition_energy if s.decomposition_energy is not None else float('inf')
                )

            weight_map = self._resolve_property_weights(targets, weights)
            weight_sum = sum(weight_map.values())
            if weight_sum <= 0:
                logger.warning("Multi-objective weights sum to 0; using stability ranking")
                return sorted(
                    structures,
                    key=lambda s: s.decomposition_energy if s.decomposition_energy is not None else float('inf')
                )

            # Normalize weights
            norm_weights = {k: v / weight_sum for k, v in weight_map.items() if v > 0}

            # Scale for Ed when target == 0
            default_ed_scale = self.config.get('thresholds', {}).get('metastability', 0.1) or 0.1
            scales = obj_cfg.get('scales', {}) or {}

            def multi_objective_score(s: CrystalStructure) -> float:
                # 🔥 HARD CONSTRAINT CHECK - violations get zero score
                if not self._satisfies_hard_constraints(s):
                    return 0.0

                total = 0.0
                for prop, target in targets.items():
                    weight = norm_weights.get(prop, 0.0)
                    if weight <= 0:
                        continue
                    value = self._get_property_value(s, prop)
                    score = self._score_property(prop, value, target, scales, default_ed_scale)
                    total += weight * score
                return total

            # 🔥 Log multi-objective ranking details
            logger.info(f"Multi-objective ranking: {len(structures)} structures using weighted sum")
            logger.info(f"  Targets: {targets}")
            logger.info(f"  Normalized weights: {norm_weights}")

            # Compute scores and sort
            sorted_with_scores = sorted(
                [(s, multi_objective_score(s)) for s in structures],
                key=lambda x: x[1],
                reverse=True
            )

            # Log top 5 structures with their scores and properties
            logger.info(f"  Top 5 structures by weighted score:")
            for i, (s, score) in enumerate(sorted_with_scores[:5]):
                props_str = ", ".join([
                    f"{prop}={self._get_property_value(s, prop):.3f}" if self._get_property_value(s, prop) is not None else f"{prop}=None"
                    for prop in targets.keys()
                ])
                logger.info(f"    #{i+1}: {s.formula:15s} score={score:.4f} | {props_str}")

            # 🔥 Store weighted_score in structure for later use (e.g., dynamic templates)
            for s, score in sorted_with_scores:
                s.weighted_score = score

            return [s for s, _ in sorted_with_scores]
        
        else:
            logger.warning(f"Unknown objective: {self.objective}, using stability")
            return sorted(
                structures,
                key=lambda s: s.decomposition_energy if s.decomposition_energy else float('inf')
            )
    
    def calculate_diversity(self, structures: List[CrystalStructure]) -> float:
        """
        Calculate structural diversity metric
        
        Args:
            structures: List of structures
            
        Returns:
            Diversity score (0-1)
        """
        if len(structures) <= 1:
            return 0.0
        
        # Simple diversity: unique formulas / total structures
        unique_formulas = len(set(s.formula for s in structures))
        diversity = unique_formulas / len(structures)
        
        return diversity

    def _get_property_value(self, structure: CrystalStructure, prop: str) -> Optional[float]:
        """Fetch a property value from structure attributes or properties dict."""
        if prop in structure.properties:
            return structure.properties.get(prop)
        if hasattr(structure, prop):
            return getattr(structure, prop)
        return None

    def _score_property(
        self,
        prop: str,
        value: Optional[float],
        target: Optional[float],
        scales: Dict[str, float],
        default_ed_scale: float
    ) -> float:
        """Compute a 0-1 score using 'target-saturation' normalization."""
        if value is None:
            return 0.0

        minimize_props = {
            'density',
            'decomposition_energy',
            'energy_above_hull',
            'formation_energy',
            'energy_per_atom'
        }

        if prop in minimize_props:
            if target is None:
                return 0.0
            if target < 0:
                # Negative target (e.g., formation_energy=-1.5)
                # More negative value = better; score=1.0 when value <= target
                if value >= 0:
                    return 0.0
                return max(min(value / target, 1.0), 0.0)
            if target > 0:
                if value <= 0:
                    return 0.0
                return min(target / value, 1.0)
            # target == 0: use exponential decay
            scale = scales.get(prop, default_ed_scale)
            if scale <= 0:
                scale = default_ed_scale
            return math.exp(-value / scale)

        # Default: maximize
        if target is None or target <= 0:
            return 0.0
        if value <= 0:
            return 0.0
        return min(value / target, 1.0)

    def _satisfies_hard_constraints(self, structure: CrystalStructure) -> bool:
        """
        Check if structure satisfies all hard constraints.

        Returns:
            True if all constraints satisfied, False otherwise
        """
        if not self.hard_constraints:
            return True

        for constraint_name, constraint_spec in self.hard_constraints.items():
            # Skip disabled constraints
            if isinstance(constraint_spec, dict) and not constraint_spec.get('enabled', True):
                continue

            # Special case: is_valid constraint
            if constraint_name == 'is_valid':
                if not structure.is_valid:
                    logger.debug(f"Structure {structure.formula} violates is_valid constraint")
                    return False
                continue

            # Get property value
            value = self._get_property_value(structure, constraint_name)
            if value is None:
                # If property not available, can't check constraint - allow it
                continue

            # Check min/max constraints
            if isinstance(constraint_spec, dict):
                min_val = constraint_spec.get('min')
                max_val = constraint_spec.get('max')

                if min_val is not None and value < min_val:
                    logger.debug(
                        f"Structure {structure.formula} violates {constraint_name} min constraint: "
                        f"{value:.3f} < {min_val}"
                    )
                    return False

                if max_val is not None and value > max_val:
                    logger.debug(
                        f"Structure {structure.formula} violates {constraint_name} max constraint: "
                        f"{value:.3f} > {max_val}"
                    )
                    return False

        return True

    def _resolve_property_weights(self, targets: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
        """Map objective weights to target properties (supports common aliases)."""
        alias_map = {
            'bulk_modulus': ['bulk', 'stiffness_bulk'],
            'shear_modulus': ['shear', 'stiffness_shear'],
            'density': ['lightness', 'lightweight'],
            'decomposition_energy': ['stability', 'stability_hull', 'stability_ed'],
            'formation_energy': ['formation', 'stability_formation'],
            'band_gap': ['insulation', 'optical'],
            'dielectric_constant': ['dielectric'],
            'piezoelectric_coefficient': ['piezoelectric'],
        }

        resolved = {}
        unused = set(weights.keys())
        for prop in targets.keys():
            candidates = [prop] + alias_map.get(prop, [])
            for key in candidates:
                if key in weights:
                    resolved[prop] = weights[key]
                    unused.discard(key)
                    break

        if unused:
            logger.debug(f"Multi-objective weights not mapped to targets: {sorted(unused)}")
        return resolved
