from typing import List, Dict
import logging
import numpy as np

from src.core.structure import CrystalStructure

logger = logging.getLogger(__name__)


class MotifScorer:
    """
    Motif fidelity scorer
    
    Evaluates how well child structures preserve desired
    crystallographic motifs from parent structures
    """
    
    def __init__(self, config: Dict):
        """
        Initialize motif scorer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['validation']
        self.symmetry_tolerance = self.config.get('symmetry_tolerance', 0.1)
    
    def score(
        self,
        children: List[CrystalStructure],
        parents: List[CrystalStructure]
    ) -> List[CrystalStructure]:
        """
        Score child structures based on motif fidelity
        
        Args:
            children: Child structures to score
            parents: Parent structures for reference
            
        Returns:
            Children with motif_score added to metadata
        """
        logger.info(f"Scoring {len(children)} structures for motif fidelity")
        
        for child in children:
            # Calculate fidelity to closest parent
            fidelity = self._calculate_fidelity(child, parents)
            child.metadata['motif_score'] = fidelity
            
            logger.debug(f"{child.formula}: motif_score = {fidelity:.3f}")
        
        return children
    
    def _calculate_fidelity(
        self,
        child: CrystalStructure,
        parents: List[CrystalStructure]
    ) -> float:
        """
        Calculate fidelity score (0-1)
        
        Higher score = better preservation of parent motifs
        
        Args:
            child: Child structure
            parents: Parent structures
            
        Returns:
            Fidelity score
        """
        scores = []
        
        for parent in parents:
            score = 0.0
            
            # 1. Crystal system similarity (weight: 0.3)
            if self._same_crystal_system(child, parent):
                score += 0.3
            
            # 2. Space group similarity (weight: 0.3)
            if self._similar_space_group(child, parent):
                score += 0.3
            
            # 3. Composition similarity (weight: 0.2)
            comp_sim = self._composition_similarity(child, parent)
            score += 0.2 * comp_sim
            
            # 4. Density similarity (weight: 0.2)
            density_sim = self._density_similarity(child, parent)
            score += 0.2 * density_sim
            
            scores.append(score)
        
        # Return max similarity to any parent
        return max(scores) if scores else 0.0
    
    def _same_crystal_system(
        self,
        struct1: CrystalStructure,
        struct2: CrystalStructure
    ) -> bool:
        """Check if structures belong to same crystal system"""
        try:
            pmg1 = struct1.to_pymatgen()
            pmg2 = struct2.to_pymatgen()
            
            sys1 = pmg1.get_space_group_info()[0]
            sys2 = pmg2.get_space_group_info()[0]
            
            return sys1 == sys2
            
        except Exception:
            return False
    
    def _similar_space_group(
        self,
        struct1: CrystalStructure,
        struct2: CrystalStructure
    ) -> bool:
        """Check if structures have similar space groups"""
        try:
            pmg1 = struct1.to_pymatgen()
            pmg2 = struct2.to_pymatgen()
            
            sg1 = pmg1.get_space_group_info()[1]
            sg2 = pmg2.get_space_group_info()[1]
            
            # Same space group or within tolerance
            return abs(sg1 - sg2) <= 5  # Allow small variations
            
        except Exception:
            return False
    
    def _composition_similarity(
        self,
        struct1: CrystalStructure,
        struct2: CrystalStructure
    ) -> float:
        """Calculate composition similarity (Jaccard index)"""
        from collections import Counter
        
        comp1 = Counter(struct1.species)
        comp2 = Counter(struct2.species)
        
        # Normalize
        total1 = sum(comp1.values())
        total2 = sum(comp2.values())
        
        for key in comp1:
            comp1[key] /= total1
        for key in comp2:
            comp2[key] /= total2
        
        # Jaccard similarity
        all_elements = set(comp1.keys()) | set(comp2.keys())
        
        intersection = sum(min(comp1.get(e, 0), comp2.get(e, 0)) for e in all_elements)
        union = sum(max(comp1.get(e, 0), comp2.get(e, 0)) for e in all_elements)
        
        return intersection / union if union > 0 else 0.0
    
    def _density_similarity(
        self,
        struct1: CrystalStructure,
        struct2: CrystalStructure
    ) -> float:
        """Calculate density similarity"""
        d1 = struct1.density
        d2 = struct2.density
        
        # Normalized difference
        diff = abs(d1 - d2) / max(d1, d2)
        
        return max(0.0, 1.0 - diff)