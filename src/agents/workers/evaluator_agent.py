"""Evaluator Agent - MLIP-based evaluation"""

from typing import List, Dict
import logging

from src.core.structure import CrystalStructure
from src.core.evaluator import StructureEvaluator

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """
    Structure evaluation agent using MLIPs
    
    Performs:
    - Structure relaxation
    - Energy calculation
    - Decomposition energy calculation
    - Property calculation (bulk modulus, etc.)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator agent
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluator = StructureEvaluator(config['evaluator'])
    
    def evaluate(self, structures: List[CrystalStructure]) -> List[CrystalStructure]:
        """
        Evaluate structures using MLIP
        
        Args:
            structures: List of structures to evaluate
            
        Returns:
            List of evaluated structures with computed properties
        """
        logger.info(f"Evaluating {len(structures)} structures")
        
        evaluated = self.evaluator.batch_evaluate(structures)
        
        # Log statistics
        valid_count = sum(1 for s in evaluated if s.is_valid)
        metastable_count = sum(1 for s in evaluated 
                               if s.decomposition_energy is not None 
                               and s.decomposition_energy < 0.1)
        
        logger.info(f"Evaluation complete: {valid_count}/{len(structures)} valid, "
                   f"{metastable_count} metastable")
        
        return evaluated