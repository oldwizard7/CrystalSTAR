"""Structure Parser Agent - Validation and parsing"""

from typing import List, Dict
import logging
import numpy as np

from src.core.structure import CrystalStructure
from src.utils.parser import StructureParser as ParserUtils

logger = logging.getLogger(__name__)


class StructureParser:
    """
    Structure parsing and validation agent
    
    Validates:
    - Three-dimensional periodicity
    - Physical connectivity (interatomic distances)
    - Chemical validity (charge balance)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize parser
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['validation']
        self.min_distance_factor = self.config.get('min_distance_factor', 0.6)
        self.max_distance_factor = self.config.get('max_distance_factor', 1.3)
        self.check_charge = self.config.get('check_charge_balance', True)
    
    def parse(self, raw_responses: List[str]) -> List[CrystalStructure]:
        """
        Parse and validate structures from LLM responses
        
        Args:
            raw_responses: List of raw LLM response strings
            
        Returns:
            List of valid CrystalStructure objects
        """
        valid_structures = []
        
        for i, response in enumerate(raw_responses):
            logger.debug(f"Parsing response {i+1}/{len(raw_responses)}")
            
            # Extract POSCAR blocks
            structures = ParserUtils.parse_structures(response)
            
            # Validate each structure
            for struct in structures:
                if self._validate(struct):
                    struct.is_valid = True
                    valid_structures.append(struct)
                    logger.info(f"✓ Valid structure: {struct.formula}")
                else:
                    logger.warning(f"✗ Invalid structure: {struct.formula}")
        
        logger.info(f"Parsed {len(valid_structures)} valid structures from {len(raw_responses)} responses")
        
        return valid_structures
    
    def _validate(self, structure: CrystalStructure) -> bool:
        """
        Validate structure constraints
        
        Args:
            structure: Crystal structure to validate
            
        Returns:
            True if structure passes all validations
        """
        # 1. Check periodicity (lattice vectors not degenerate)
        if not self._check_periodicity(structure):
            logger.debug(f"Failed periodicity check: {structure.formula}")
            return False
        
        # 2. Check interatomic distances
        if not self._check_distances(structure):
            logger.debug(f"Failed distance check: {structure.formula}")
            return False
        
        # 3. Check charge balance (optional)
        if self.check_charge and not self._check_charge_balance(structure):
            logger.debug(f"Failed charge balance: {structure.formula}")
            return False
        
        return True
    
    def _check_periodicity(self, structure: CrystalStructure) -> bool:
        """Check that lattice vectors form valid 3D cell"""
        det = np.linalg.det(structure.lattice)
        return abs(det) > 1e-6  # Non-degenerate
    
    def _check_distances(self, structure: CrystalStructure) -> bool:
        """Check interatomic distances are physical"""
        from pymatgen.core import Element
        
        try:
            pmg_struct = structure.to_pymatgen()
            
            # Get all pairwise distances
            for i, site1 in enumerate(pmg_struct):
                for j, site2 in enumerate(pmg_struct):
                    if i >= j:
                        continue
                    
                    distance = site1.distance(site2)
                    
                    # Expected distance (sum of covalent radii)
                    elem1 = Element(site1.specie.symbol)
                    elem2 = Element(site2.specie.symbol)
                    expected = elem1.atomic_radius + elem2.atomic_radius
                    
                    # Check if within acceptable range
                    min_dist = expected * self.min_distance_factor
                    max_dist = expected * self.max_distance_factor
                    
                    if distance < min_dist or distance > max_dist:
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Distance check failed: {e}")
            return False
    
    def _check_charge_balance(self, structure: CrystalStructure) -> bool:
        """Check that structure is charge balanced"""
        from pymatgen.core import Element
        
        try:
            # Simple check: use common oxidation states
            pmg_struct = structure.to_pymatgen()
            pmg_struct.add_oxidation_state_by_guess()
            
            total_charge = sum(site.specie.oxi_state for site in pmg_struct)
            
            return abs(total_charge) < 0.1  # Allow small numerical error
            
        except Exception:
            # If oxidation states can't be assigned, assume valid
            return True