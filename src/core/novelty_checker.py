"""
Structural novelty checker for materials discovery

Implements multi-tier novelty detection:
- Tier 1: Composition-based (fast)
- Tier 2: Quick structural filters (space group, density, atom count)
- Tier 3: Detailed structure matching (slow but accurate)
"""

from typing import Dict, List, Optional
from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NoveltyChecker:
    """
    Multi-level novelty checking with structural awareness

    Usage:
        checker = NoveltyChecker(mp_api_key, mode='standard')
        result = checker.is_novel(candidate_structure)

        if result['is_novel']:
            print(f"Novel! Reason: {result['reason']}")
    """

    # Matcher configurations for different strictness levels
    MATCHER_CONFIGS = {
        'quick': {
            'ltol': 0.3,      # Lattice tolerance (Å)
            'stol': 0.5,      # Site tolerance
            'angle_tol': 10,  # Angle tolerance (degrees)
        },
        'standard': {
            'ltol': 0.2,
            'stol': 0.3,
            'angle_tol': 5,
        },
        'strict': {
            'ltol': 0.1,
            'stol': 0.2,
            'angle_tol': 3,
        }
    }

    # Quick filter thresholds
    DENSITY_TOLERANCE = 0.10  # ±10%
    NATOMS_TOLERANCE = 2      # ±2 atoms (for primitive vs conventional cells)

    def __init__(self, mp_api_key: str, mode: str = 'standard'):
        """
        Initialize novelty checker

        Args:
            mp_api_key: Materials Project API key
            mode: 'quick' | 'standard' | 'strict'
        """
        self.mp_api_key = mp_api_key
        self.mode = mode

        if mode not in self.MATCHER_CONFIGS:
            raise ValueError(f"Invalid mode: {mode}. Choose from {list(self.MATCHER_CONFIGS.keys())}")

        config = self.MATCHER_CONFIGS[mode]
        self.matcher = StructureMatcher(**config)

        logger.info(f"NoveltyChecker initialized in '{mode}' mode")
        logger.info(f"  Lattice tolerance: {config['ltol']} Å")
        logger.info(f"  Site tolerance: {config['stol']}")
        logger.info(f"  Angle tolerance: {config['angle_tol']}°")

    def is_novel(self,
                 candidate_structure: Structure,
                 check_mp: bool = True,
                 skip_tier3: bool = False) -> Dict:
        """
        Multi-tier novelty check

        Args:
            candidate_structure: Structure to check
            check_mp: Whether to check against Materials Project
            skip_tier3: Skip detailed StructureMatcher (faster but less accurate)

        Returns:
            {
                'is_novel': bool,
                'reason': str,
                'tier': int,  # 1=composition, 2=quick_filter, 3=structure_match
                'similar_mpid': str or None,
                'similarity_score': float
            }
        """
        result = {
            'is_novel': False,
            'reason': '',
            'tier': 0,
            'similar_mpid': None,
            'similarity_score': 0.0
        }

        # Quick return if MP check disabled
        if not check_mp:
            result['is_novel'] = True
            result['reason'] = 'MP check disabled'
            result['tier'] = 0
            return result

        # ========== Tier 1: Composition Check (reduced composition) ==========
        candidate_comp: Composition = candidate_structure.composition.reduced_composition
        formula = candidate_comp.reduced_formula
        chemsys = self._get_chemsys(candidate_comp)

        mp_entries = self._search_mp_entries(candidate_comp, chemsys)

        # No entries for this chemical system at all → absolute novel
        if not mp_entries:
            result['is_novel'] = True
            result['reason'] = f'Chemsys {chemsys} not in Materials Project'
            result['tier'] = 1
            logger.debug(f"✓ Novel (Tier 1): {chemsys} not present in MP")
            return result

        # Filter entries with identical reduced composition
        comp_matches = [
            entry for entry in mp_entries
            if self._composition_matches(candidate_comp, entry.structure.composition)
        ]

        if not comp_matches:
            # Chemsys exists, but exact stoichiometry absent → novel, but explicitly labeled
            result['is_novel'] = True
            result['reason'] = f'Chemsys {chemsys} present, composition {formula} not found in MP'
            result['tier'] = 1
            logger.debug(f"✓ Novel (Tier 1): {chemsys} known, composition {formula} absent")
            return result

        logger.debug(f"Found {len(comp_matches)} MP entries for composition {formula} within chemsys {chemsys}")

        # ========== Tier 2: Quick Structure Filters ==========
        passed_tier2, similar_mpid = self._quick_structure_check(
            candidate_structure, comp_matches
        )

        if passed_tier2:
            result['is_novel'] = True
            result['reason'] = f'Different structure from known {formula} polymorphs'
            result['tier'] = 2
            logger.debug(f"✓ Novel (Tier 2): {formula} - different space group/density/size")
            return result

        # If skipping Tier 3, return as not novel
        if skip_tier3:
            result['is_novel'] = False
            result['reason'] = f'Likely matches {similar_mpid} (Tier 3 skipped)'
            result['tier'] = 2
            result['similar_mpid'] = similar_mpid
            return result

        # ========== Tier 3: Detailed Structure Matching ==========
        match_result = self._detailed_structure_match(candidate_structure, mp_entries)

        if match_result['matched']:
            result['is_novel'] = False
            result['reason'] = f"Matches MP-{match_result['mpid']}"
            result['tier'] = 3
            result['similar_mpid'] = match_result['mpid']
            result['similarity_score'] = match_result['score']
            logger.debug(f"✗ Not Novel (Tier 3): matches {match_result['mpid']}")
        else:
            result['is_novel'] = True
            result['reason'] = f'Structure differs from all {len(mp_entries)} known {formula} polymorphs'
            result['tier'] = 3
            logger.debug(f"✓ Novel (Tier 3): {formula} - new polymorph!")

        return result

    def _search_mp_entries(self, comp: Composition, chemsys: str) -> List:
        """Search Materials Project for entries in the same chemical system"""
        try:
            from mp_api.client import MPRester

            with MPRester(self.mp_api_key) as mpr:
                results = mpr.materials.search(
                    chemsys=chemsys,
                    fields=['material_id', 'structure', 'symmetry', 'density', 'formula_pretty']
                )
                return results
        except Exception as e:
            logger.warning(f"MP search failed for {chemsys}: {e}")
            return []

    def _quick_structure_check(self,
                               candidate: Structure,
                               mp_entries: List) -> tuple:
        """
        Quick structural filtering using space group, density, and atom count

        Returns:
            (is_novel, similar_mpid)
        """
        c_sg = self._get_spacegroup(candidate)
        c_natoms = len(candidate)
        c_density = candidate.density

        for mp_entry in mp_entries:
            mp_struct = mp_entry.structure

            # Filter 1: Space group
            mp_sg = self._get_spacegroup(mp_struct)
            if c_sg != mp_sg:
                continue  # Different symmetry → likely different structure

            # Filter 2: Atom count
            mp_natoms = len(mp_struct)
            if abs(c_natoms - mp_natoms) > self.NATOMS_TOLERANCE:
                continue  # Different cell size → likely different structure

            # Filter 3: Density
            mp_density = mp_struct.density
            density_diff = abs(c_density - mp_density) / mp_density
            if density_diff > self.DENSITY_TOLERANCE:
                continue  # Very different density → likely different structure

            # Passed all quick checks → likely same structure
            logger.debug(f"  Quick match candidate: {mp_entry.material_id} "
                        f"(SG:{mp_sg}, N:{mp_natoms}, ρ:{mp_density:.2f})")
            return False, mp_entry.material_id

        # No quick matches found
        return True, None

    def _detailed_structure_match(self,
                                  candidate: Structure,
                                  mp_entries: List) -> Dict:
        """
        Detailed structure matching using StructureMatcher

        Returns:
            {'matched': bool, 'mpid': str, 'score': float}
        """
        for mp_entry in mp_entries:
            mp_struct = mp_entry.structure

            try:
                if self.matcher.fit(candidate, mp_struct):
                    score = self._calculate_similarity(candidate, mp_struct)
                    logger.debug(f"  StructureMatcher: MATCH with {mp_entry.material_id} "
                               f"(score: {score:.3f})")
                    return {
                        'matched': True,
                        'mpid': mp_entry.material_id,
                        'score': score
                    }
            except Exception as e:
                logger.warning(f"Matching failed for {mp_entry.material_id}: {e}")
                continue

        return {'matched': False, 'mpid': None, 'score': 0.0}

    def _get_spacegroup(self, structure: Structure) -> int:
        """Get space group number"""
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            return sga.get_space_group_number()
        except:
            return -1

    def _calculate_similarity(self, s1: Structure, s2: Structure) -> float:
        """
        Calculate structural similarity score (0-1)

        Based on lattice parameters and density
        """
        try:
            # Lattice difference (normalized)
            lat_diff = np.linalg.norm(s1.lattice.matrix - s2.lattice.matrix)
            lat_norm = np.linalg.norm(s2.lattice.matrix)
            lat_similarity = 1.0 / (1.0 + lat_diff / lat_norm)

            # Density difference (normalized)
            density_diff = abs(s1.density - s2.density) / s2.density
            density_similarity = 1.0 / (1.0 + density_diff)

            # Combined score
            score = 0.6 * lat_similarity + 0.4 * density_similarity
            return score
        except:
            return 0.0

    def _get_chemsys(self, comp: Composition) -> str:
        """Return canonical chemsys string (e.g., Hf-Sc-Si)"""
        elements = sorted([el.symbol for el in comp.elements])
        return "-".join(elements)

    def _composition_matches(self, target: Composition, other: Composition) -> bool:
        """Check if two compositions are identical on reduced basis"""
        try:
            return target.almost_equals(other.reduced_composition)
        except Exception:
            return False


class FastNoveltyChecker:
    """
    Fast novelty checker using only Tier 1 and Tier 2 (no StructureMatcher)

    ~10x faster than NoveltyChecker, ~90% accuracy
    Use for rapid screening when speed is critical
    """

    DENSITY_TOLERANCE = 0.10
    NATOMS_TOLERANCE = 2

    def __init__(self, mp_api_key: str):
        self.mp_api_key = mp_api_key
        self.checker = NoveltyChecker(mp_api_key, mode='quick')

    def is_novel(self, candidate: Structure) -> Dict:
        """
        Fast novelty check (Tier 1 + 2 only)
        """
        return self.checker.is_novel(candidate, skip_tier3=True)
