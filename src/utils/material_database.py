"""Material database for filtering known materials with MP API integration"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

KNOWN_MATERIALS = {
    'LiCoO2', 'LiFePO4', 'LiMn2O4', 'NaCoO2', 'LiNiO2',
    'LiMnO2', 'NaFePO4',
    
    'CaTiO3', 'BaTiO3', 'SrTiO3', 'BaZrO3', 'ZnTiO3',
    'PbTiO3', 'LaAlO3', 'SrRuO3',
    
    'MgO', 'Al2O3', 'SiO2', 'TiO2', 'Fe2O3', 'Fe3O4',
    'ZnO', 'CuO', 'Cu2O', 'SnO2', 'In2O3',
    
    'Mg2SiO4', 'MgSiO3', 'CaSiO3', 'Ca2SiO4',
    'Al2SiO5', 'Mg3Al2Si3O12',
    
    'GaN', 'AlN', 'InN', 'GaAs', 'InP', 'GaP',
    'ZnS', 'ZnSe', 'CdS', 'CdSe', 'CdTe',
    'Si', 'Ge', 'SiC',
    
    'NaCl', 'KCl', 'LiF', 'NaF', 'MgF2', 'CaF2',
    'LiCl', 'KBr', 'NaBr',
    
    'FeS2', 'MoS2', 'WS2', 'Bi2S3', 'Sb2S3',
    'PbS', 'Ag2S', 'Cu2S',
    
    'BN', 'Si3N4', 'AlN', 'TiN',
}


def normalize_formula(formula: str) -> str:
    """
    ，
    
    :
        "LiCoO2 - Layered Structure" → "LiCoO2"
        "Li1Co1O2" → "LiCoO2"
        "Mg 2 Si O 4" → "Mg2SiO4"
    
    Args:
        formula: 
        
    Returns:
    """
    import re
    
    if ' - ' in formula:
        formula = formula.split(' - ')[0].strip()
    
    formula = re.sub(r'\s+', '', formula)
    
    # formula = re.sub(r'([A-Z][a-z]?)1(?![0-9])', r'\1', formula)
    
    return formula


def check_materials_project(formula: str, api_key: Optional[str] = None) -> bool:
    """
     Materials Project API 
    
    Args:
        formula: 
        api_key: MP API key ( None，)
        
    Returns:
        True =  MP ，False = 
    """
    try:
        from mp_api.client import MPRester
        
        if api_key is None:
            api_key = os.getenv('MATERIALS_PROJECT_API_KEY') or os.getenv('MP_API_KEY')

        if not api_key:
            logger.warning("MP API key not found (tried MATERIALS_PROJECT_API_KEY and MP_API_KEY), skipping MP query")
            return False
        
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty"]
            )
            
            if docs:
                logger.debug(f"Found {len(docs)} entries in MP for {formula}")
                return True
            else:
                logger.debug(f"No MP entries found for {formula}")
                return False
                
    except ImportError:
        logger.warning("mp-api not installed. Install with: pip install mp-api")
        return False
    except Exception as e:
        logger.warning(f"MP API query failed for {formula}: {e}")
        return False


def is_known_material(
    formula: str, 
    use_mp: bool = True,
    strict: bool = False
) -> bool:
    """
    （ + MP API）
    
    Args:
        formula: （， "LiCoO2 - Layered"）
        use_mp:  Materials Project API（ True）
        strict: （True）（False）
        
    Returns:
        True = ，False = 
    """
    clean_formula = normalize_formula(formula)
    
    if clean_formula in KNOWN_MATERIALS:
        logger.debug(f"{formula} found in local database")
        return True
    
    if use_mp:
        if check_materials_project(clean_formula):
            KNOWN_MATERIALS.add(clean_formula)
            return True
    
    return False


def categorize_materials(
    structures: list,
    use_mp: bool = True,
    progress_callback: Optional[callable] = None,
    use_structure_matching: bool = True,
    novelty_mode: str = 'standard'
) -> dict:
    """

    NEW: Now supports structure-aware novelty detection!
    - Checks space group, density, atom count
    - Uses StructureMatcher for accurate comparison
    - Can distinguish polymorphs of same composition

    Args:
        structures: CrystalStructure 
        use_mp:  Materials Project API
        progress_callback:  callback(current, total)
        use_structure_matching: （！）
        novelty_mode: 'quick' | 'standard' | 'strict'

    Returns:
        {
            'known': [],
            'novel': [],
            'all': [],
            'novelty_details': [{novelty}],  # NEW!
            'novelty_tier_counts': {tier: count}
        }
    """
    known = []
    novel = []
    novelty_details = []
    tier_counts = {}

    total = len(structures)

    def record_novelty(struct, result: dict, method: str):
        """Attach novelty metadata and keep tier counts in sync."""
        detail = {
            'formula': struct.formula,
            'is_novel': result.get('is_novel', False),
            'reason': result.get('reason', ''),
            'tier': result.get('tier', 0),
            'similar_mpid': result.get('similar_mpid'),
            'similarity_score': result.get('similarity_score', 0.0),
            'mode': novelty_mode if use_structure_matching and use_mp else 'formula_only',
            'method': method
        }
        novelty_details.append(detail)

        tier = detail.get('tier', 0)
        if tier:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        struct.metadata['novelty'] = detail

    # Use structure-aware novelty checker if enabled
    if use_structure_matching and use_mp:
        try:
            from src.core.novelty_checker import NoveltyChecker
            import os

            # Support both environment variable names for compatibility
            mp_api_key = os.environ.get('MATERIALS_PROJECT_API_KEY') or os.environ.get('MP_API_KEY')
            if not mp_api_key:
                logger.warning("MP API key not found (tried MATERIALS_PROJECT_API_KEY and MP_API_KEY). Falling back to formula-only check.")
                use_structure_matching = False
            else:
                checker = NoveltyChecker(mp_api_key, mode=novelty_mode)
                logger.info(f"Using structure-aware novelty detection (mode: {novelty_mode})")
        except Exception as e:
            logger.warning(f"Failed to initialize NoveltyChecker: {e}")
            logger.warning("Falling back to formula-only check")
            use_structure_matching = False

    for i, struct in enumerate(structures):
        if progress_callback:
            progress_callback(i + 1, total)

        if use_structure_matching and use_mp:
            # Structure-aware check (NEW!)
            try:
                # Convert CrystalStructure to pymatgen Structure
                pmg_structure = struct.to_pymatgen()
                result = checker.is_novel(pmg_structure, check_mp=True)

                record_novelty(struct, result, method="structure_match")

                if result['is_novel']:
                    novel.append(struct)
                    logger.debug(f"  ✓ Novel: {struct.formula} - {result['reason']}")
                else:
                    known.append(struct)
                    logger.debug(f"  ✗ Known: {struct.formula} - {result['reason']}")

            except Exception as e:
                logger.warning(f"Structure matching failed for {struct.formula}: {e}")
                # Fallback to formula-only check
                is_known = is_known_material(struct.formula, use_mp=use_mp)
                fallback_result = {
                    'is_novel': not is_known,
                    'reason': 'Fallback to formula-only check',
                    'tier': 1,
                    'similar_mpid': None,
                    'similarity_score': 0.0
                }
                record_novelty(struct, fallback_result, method="formula_only_fallback")

                if is_known:
                    known.append(struct)
                else:
                    novel.append(struct)
        else:
            # Formula-only check (legacy)
            is_known = is_known_material(struct.formula, use_mp=use_mp)
            result = {
                'is_novel': not is_known,
                'reason': 'Composition not in Materials Project' if not is_known else 'Composition found in Materials Project',
                'tier': 1,
                'similar_mpid': None,
                'similarity_score': 0.0
            }
            record_novelty(struct, result, method="formula_only")

            if is_known:
                known.append(struct)
            else:
                novel.append(struct)

    logger.info(f"Categorized {total} structures: {len(known)} known, {len(novel)} novel")

    # Log tier distribution if using structure matching
    if novelty_details and tier_counts:
        logger.info("Novelty detection tier breakdown:")
        tier_names = {1: "Composition", 2: "Quick Structure", 3: "Detailed Match"}
        for tier in sorted(tier_counts.keys()):
            logger.info(f"  Tier {tier} ({tier_names.get(tier, 'Unknown')}): {tier_counts[tier]} structures")

    return {
        'known': known,
        'novel': novel,
        'all': structures,
        'novelty_details': novelty_details,  # NEW: detailed info for each structure
        'novelty_tier_counts': tier_counts
    }


def add_known_material(formula: str):
    """
    
    Args:
        formula: 
    """
    clean_formula = normalize_formula(formula)
    KNOWN_MATERIALS.add(clean_formula)
    logger.info(f"Added {clean_formula} to known materials database")


def get_statistics(categorized: dict) -> dict:
    """
    
    Args:
        categorized: categorize_materials() 
        
    Returns:
    """
    all_structures = categorized['all']
    known = categorized['known']
    novel = categorized['novel']
    
    novel_stable = [s for s in novel if s.is_valid and s.decomposition_energy < 0]
    known_stable = [s for s in known if s.is_valid and s.decomposition_energy < 0]
    
    stats = {
        'total': len(all_structures),
        'known_count': len(known),
        'novel_count': len(novel),
        'known_ratio': len(known) / len(all_structures) if all_structures else 0,
        'novel_ratio': len(novel) / len(all_structures) if all_structures else 0,
        'novel_stable_count': len(novel_stable),
        'known_stable_count': len(known_stable),
    }
    
    if novel_stable:
        best_novel = min(novel_stable, key=lambda s: s.decomposition_energy)
        stats['best_novel'] = {
            'formula': best_novel.formula,
            'decomposition_energy': best_novel.decomposition_energy
        }
    
    return stats


if __name__ == "__main__":
    import sys
    
    test_cases = [
        "LiCoO2 - Layered Structure",
        "BaZrO3 - Perovskite Structure",
        "BeNO2 - Nitridoberyllate Structure",
        "AlPSe2",
        "MgSiO4 - Disilicate Structure",
        "CuGaN2",
        "Sc2O3",
    ]
    
    print("Material Classification Test")
    print("=" * 70)
    print(f"{'Formula':<40} {'Local':<10} {'MP API':<10} {'Status':<10}")
    print("=" * 70)
    
    for formula in test_cases:
        clean = normalize_formula(formula)
        in_local = clean in KNOWN_MATERIALS
        in_mp = check_materials_project(clean)
        is_known = is_known_material(formula, use_mp=True)
        
        status = "KNOWN" if is_known else "NOVEL"
        local_str = "✓" if in_local else "✗"
        mp_str = "✓" if in_mp else "✗"
        
        print(f"{formula:<40} {local_str:<10} {mp_str:<10} {status:<10}")
