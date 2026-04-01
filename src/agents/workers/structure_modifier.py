"""Structure Modifier Agent - LLM-based structure generation"""

from typing import List, Dict
import logging

from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class StructureModifier:
    """
    LLM-driven structure modification agent
    
    Generates new crystal structures by:
    - Implicit crossover of parent structures
    - Mutation through LLM reasoning
    - Maintaining chemical validity
    """
    
    def __init__(self, config: Dict):
        """
        Initialize structure modifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm = LLMClient(config['llm'])
        self.num_children = config['evolution']['children_size']
    
    def generate(
        self,
        strategy: Dict,
        parents: List[CrystalStructure]
    ) -> List[str]:
        """
        Generate new structures using LLM
        
        Args:
            strategy: Search strategy with prompt template
            parents: Parent structures for reference
            
        Returns:
            List of raw LLM responses
        """
        logger.info(f"Generating {self.num_children} structures from {len(parents)} parents")
        
        # Get prompt template
        prompt_template = strategy.get('prompt_template', self._get_default_prompt())
        
        # Generate structures
        responses = self.llm.generate_structures(
            parent_structures=parents,
            prompt_template=prompt_template,
            num_children=self.num_children,
            **strategy.get('template_vars', {})
        )
        
        logger.info(f"Received {len(responses)} responses from LLM")
        
        return responses
    
    def _get_default_prompt(self) -> str:
        """Get default prompt template"""
        return """You are an expert materials scientist. Your task is to propose {reproduction_size} new crystalline materials with valid stable structures.

Requirements:
1. No isolated or overlapping atoms
2. Structures should be thermodynamically stable
3. Maintain proper atomic coordination

The proposed new materials can be modifications or combinations of the base materials given below.

Base materials for reference:
{reference_structures}

Format: Output each structure in POSCAR format with 12 decimal precision.

Output your {reproduction_size} proposed structures:"""