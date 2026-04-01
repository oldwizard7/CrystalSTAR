"""Worker Agents package"""

from .structure_modifier import StructureModifier
from .structure_parser import StructureParser
from .evaluator_agent import EvaluatorAgent
from .motif_scorer import MotifScorer


class WorkerAgents:
    """
    Worker Agents pipeline coordinator
    
    Executes the complete Generate-Parse-Evaluate-Score workflow
    """
    
    def __init__(self, config: dict):
        """Initialize all worker agents"""
        self.modifier = StructureModifier(config)
        self.parser = StructureParser(config)
        self.evaluator = EvaluatorAgent(config)
        self.scorer = MotifScorer(config)
    
    def execute(self, strategy: dict, parents: list) -> list:
        """
        Execute complete worker pipeline
        
        Args:
            strategy: Search strategy dictionary
            parents: Parent structures
            
        Returns:
            Evaluated and scored child structures
        """
        # 1. Generate: LLM proposes new structures
        raw_structures = self.modifier.generate(strategy, parents)
        
        # 2. Parse: Extract and validate POSCAR
        parsed_structures = self.parser.parse(raw_structures)
        
        # 3. Evaluate: Calculate energy and properties
        evaluated_structures = self.evaluator.evaluate(parsed_structures)
        
        # 4. Score: Assess motif fidelity
        scored_structures = self.scorer.score(evaluated_structures, parents)
        
        return scored_structures


__all__ = [
    'WorkerAgents',
    'StructureModifier',
    'StructureParser',
    'EvaluatorAgent',
    'MotifScorer'
]