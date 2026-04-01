"""Hypothesis Generation Agent - Strategy optimization using GEPA"""
import numpy as np
from typing import List, Dict, Optional
import logging

from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class HypothesisGenerationAgent:
    """
    Hypothesis Generation Agent
    
    Implements GEPA (Genetic Evolution of Prompt Alternatives)
    to optimize search strategies
    """
    
    def __init__(self, config: Dict, prompts: Dict):
        """
        Initialize hypothesis generation agent
        
        Args:
            config: Configuration dictionary
            prompts: Prompt templates dictionary
        """
        self.config = config
        self.prompts = prompts
        self.llm = LLMClient(config['llm'])
        
        # Strategy history for evolution
        self.strategy_history = []
        self.performance_history = []

    def _get_ablation_mode(self) -> str:
        mode = str(self.config.get("ablation_mode", "crystal_forge")).strip().lower()
        if mode not in {"memory_less", "vanilla_react", "flat_memory", "crystal_forge"}:
            mode = "crystal_forge"
        return mode
    
    def propose_strategy(
        self,
        parent_pool: List[CrystalStructure],
        current_plan: Optional[str] = None,
        reflection: Optional[Dict] = None
    ) -> Dict:
        """
        Propose search strategy for current iteration
        
        Uses ReAct (Reasoning-Action-Observation) pattern
        
        Args:
            parent_pool: Current parent structures
            current_plan: Current search plan
            reflection: Performance reflection from previous iteration
            
        Returns:
            Strategy dictionary with prompt template and parameters
        """
        logger.info("Generating hypothesis for next iteration")
        
        # Analyze current state
        analysis = self._analyze_state(parent_pool, reflection)
        
        # Generate strategy using ReAct
        strategy = self._react_loop(analysis, current_plan)
        
        # Store for evolution
        self.strategy_history.append(strategy)
        
        return strategy
    
    def _analyze_state(
        self,
        parents: List[CrystalStructure],
        reflection: Optional[Dict]
    ) -> Dict:
        """
        Analyze current search state
        
        Returns:
            Analysis dictionary with key observations
        """
        num_parents = len(parents)

        if num_parents == 0:
            analysis = {
                'num_parents': 0,
                'best_ed': None,
                'avg_ed': None,
                'composition_diversity': 0.0,
            }
        else:
            valid_eds = [
                p.decomposition_energy for p in parents
                if getattr(p, 'decomposition_energy', None) is not None
            ]

            best_ed = min(valid_eds) if valid_eds else None
            avg_ed = float(np.mean(valid_eds)) if valid_eds else None

            composition_diversity = (
                len(set(getattr(p, 'formula', None) for p in parents)) / num_parents
                if num_parents else 0.0
            )

            analysis = {
                'num_parents': num_parents,
                'best_ed': best_ed,
                'avg_ed': avg_ed,
                'composition_diversity': composition_diversity,
            }
        
        if reflection:
            analysis['previous_valid_rate'] = reflection.get('valid_rate', 0)
            analysis['previous_metastable_rate'] = reflection.get('metastable_rate', 0)
            # NEW: Include LLM reflection analysis for strategy generation
            analysis['llm_analysis'] = reflection.get('llm_analysis', None)

        return analysis
    
    def _react_loop(self, analysis: Dict, current_plan: Optional[str]) -> Dict:
        """
        ReAct reasoning loop
        
        Thought → Action → Observation
        
        Args:
            analysis: Current state analysis
            current_plan: Current strategy
            
        Returns:
            New strategy
        """
        # Thought: Analyze what needs improvement
        thought = self._generate_thought(analysis, current_plan)
        logger.info(f"Thought: {thought}")
        
        # Action: Propose modification strategy
        action = self._generate_action(thought, analysis)
        logger.info(f"Action: {action}")
        
        # Observation: Expected outcome (simulated)
        observation = self._generate_observation(action, analysis)
        logger.info(f"Expected outcome: {observation}")
        
        # Compile into strategy
        strategy = {
            'thought': thought,
            'action': action,
            'expected_outcome': observation,
            'prompt_template': self._select_prompt_template(action),
            'template_vars': self._generate_template_vars(action, analysis)
        }
        
        return strategy
    
    def _generate_thought(self, analysis: Dict, current_plan: Optional[str]) -> str:
        """Generate reasoning about current state"""
        best_ed = analysis.get('best_ed')
        avg_ed = analysis.get('avg_ed')

        if best_ed is None:
            return ("Insufficient energy data from parents. "
                   "Focus on generating diverse candidates and obtaining valid energies.")
        if best_ed > 0.1:
            return (f"Current structures are far from stability (best Ed = {best_ed:.3f}). "
                   f"Need to explore lower energy configurations.")
        elif best_ed > 0:
            return (f"Close to stability (best Ed = {best_ed:.3f}). "
                   f"Need fine-tuning to reach convex hull.")
        else:
            return (f"Achieved stable structures (best Ed = {best_ed:.3f}). "
                   f"Can optimize for additional properties.")
    
    def _generate_action(self, thought: str, analysis: Dict) -> str:
        """Generate action based on thought and LLM analysis"""
        llm_analysis = analysis.get('llm_analysis')
        best_ed = analysis.get('best_ed')

        # If we have LLM analysis, incorporate its recommendations
        if llm_analysis and llm_analysis != "LLM analysis failed - using statistical reflection only":
            # Extract recommendations from LLM analysis
            action = f"Based on LLM materials science analysis:\n{llm_analysis}\n\n"
            action += "Implement these chemistry-guided recommendations in structure generation."
            return action

        # Fallback to rule-based strategy
        if best_ed is None:
            return "Generate diverse initial structures to gather energy estimates"
        if best_ed > 0.1:
            return "Apply isotropic compression to increase cohesive energy"
        elif best_ed > 0:
            return "Apply local perturbations to optimize atomic positions"
        else:
            return "Explore compositional substitutions for property enhancement"
    
    def _generate_observation(self, action: str, analysis: Dict) -> str:
        """Generate expected observation"""
        return f"Expected to generate structures with improved stability based on {action}"
    
    def _select_prompt_template(self, action: str) -> str:
        """Select appropriate prompt template based on action"""
        objective = self.config['objective']['type']
        
        if 'compression' in action.lower():
            return self.prompts.get('csg_gepa_optimized', self.prompts['csg_basic'])
        elif 'multi' in objective or 'property' in action.lower():
            return self.prompts.get('multi_objective', self.prompts['csg_basic'])
        else:
            return self.prompts['csg_basic']
    
    def _generate_template_vars(self, action: str, analysis: Dict) -> Dict:
        """Generate template variables for prompt"""
        # Support both old and new objective formats
        if 'targets' in self.config['objective']:
            # New format: targets dictionary
            target_value = self.config['objective']['targets'].get('decomposition_energy', 0.0)
        else:
            # Old format: direct target_ed
            target_value = self.config['objective'].get('target_ed', 0.0)

        return {
            'target_property': self.config['objective']['type'],
            'target_value': target_value,
            'current_best': analysis['best_ed'],
            'observation': action
        }
    
    def evolve_prompts(self, performance_data: Dict):
        """
        Evolve prompts based on performance (GEPA)

        This is a placeholder for genetic algorithm on prompts

        Args:
            performance_data: Performance metrics
        """
        self.performance_history.append(performance_data)

        # TODO: Implement genetic algorithm for prompt evolution
        # - Selection: Choose best performing prompts
        # - Crossover: Combine elements from successful prompts
        # - Mutation: Modify prompt components

        logger.info("Prompt evolution not yet implemented")

    def generate_strategies(self, reflection: Dict, hmrr_context: Optional[Dict] = None) -> Dict:
        """
        Generate multiple search strategies based on reflection (Phase 2B)

        Let LLM decide how many strategies (2-5) and allocate weights

        Args:
            reflection: Reflection dict with individual_reflections
            hmrr_context: Optional HMRR prompt context from orchestrator

        Returns:
            Dict with strategies list and reasoning
        """
        logger.info("="*60)
        logger.info("GENERATING MULTI-STRATEGY PLAN...")
        logger.info("="*60)

        # Build meta-prompt for strategy generation
        meta_prompt = self._build_strategy_meta_prompt(reflection, hmrr_context=hmrr_context)

        # Call LLM
        try:
            import json
            response = self.llm.generate(meta_prompt, n=1)[0]

            # Parse JSON response
            response_clean = response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_clean

            strategies_data = json.loads(response_clean)

            # Validate structure
            if 'strategies' not in strategies_data or not isinstance(strategies_data['strategies'], list):
                raise ValueError("Invalid strategies format: missing 'strategies' list")

            if len(strategies_data['strategies']) < 2 or len(strategies_data['strategies']) > 5:
                logger.warning(f"LLM generated {len(strategies_data['strategies'])} strategies, expected 2-5")

            # Print for transparency
            logger.info("="*60)
            logger.info("LLM-GENERATED STRATEGIES:")
            logger.info("="*60)
            logger.info(f"Number of strategies: {len(strategies_data['strategies'])}")
            logger.info(f"Reasoning: {strategies_data.get('reasoning', 'N/A')}")
            logger.info("")

            for i, strat in enumerate(strategies_data['strategies'], 1):
                logger.info(f"Strategy #{i}: {strat['strategy_name']}")
                logger.info(f"  Type: {strat['strategy_type']}")
                logger.info(f"  Weight: {strat['allocation_weight']}")
                logger.info(f"  Description: {strat['description']}")
                logger.info(f"  Target Ed: {strat['target_ed_range']}")
                logger.info("")
            logger.info("="*60)

            return strategies_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategy JSON: {e}")
            logger.error(f"Response: {response[:500]}...")
            return self._fallback_strategies(reflection)
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            return self._fallback_strategies(reflection)

    def _build_strategy_meta_prompt(self, reflection: Dict, hmrr_context: Optional[Dict] = None) -> str:
        """Build meta-prompt for LLM to generate strategies"""

        # Format reflection summary
        stats = reflection
        summary_parts = [
            f"Total generated: {stats['total_generated']} structures",
            f"Valid: {stats['valid_count']} ({stats['valid_rate']:.1f}%)",
            f"Metastable (Ed<0.1): {stats['metastable_count']} ({stats['metastable_rate']:.1f}%)",
            f"Stable (Ed<0): {stats['stable_count']} ({stats['stable_rate']:.1f}%)",
        ]

        if stats['best_ed'] != float('inf'):
            summary_parts.extend([
                f"Best Ed: {stats['best_ed']:.3f} eV/atom",
                f"Average Ed: {stats['avg_ed']:.3f} eV/atom",
            ])

        summary = "\n".join(summary_parts)

        # Format individual reflections
        individual_refs = reflection.get('individual_reflections', [])
        ind_summary_parts = []

        for i, ref in enumerate(individual_refs, 1):
            ind_summary_parts.append(f"\nStructure #{i}: {ref['formula']} [{ref['category']}]")
            ind_summary_parts.append(f"  Ed: {ref['ed']:.3f} eV/atom")

            analysis = ref.get('analysis', {})
            if 'reason' in analysis:
                ind_summary_parts.append(f"  Reason: {analysis['reason']}")
            if 'improvement_suggestions' in analysis:
                suggestions = analysis['improvement_suggestions'][:2]  # Top 2
                for sug in suggestions:
                    ind_summary_parts.append(f"  - Suggestion: {sug}")

        ind_summary = "\n".join(ind_summary_parts) if ind_summary_parts else "No individual reflections available"

        hmrr_lines = []
        if self._get_ablation_mode() == "crystal_forge" and hmrr_context and hmrr_context.get("enabled"):
            macro_rules = hmrr_context.get("macro_rules", []) or []
            meso_tips = hmrr_context.get("meso_tips", []) or []
            if macro_rules:
                hmrr_lines.extend(["[Macro Prior]"])
                for idx, rec in enumerate(macro_rules, 1):
                    hmrr_lines.append(f"{idx}. {rec.get('message', '')}")
                hmrr_lines.append("")
            if meso_tips:
                hmrr_lines.extend(["[Meso Tactical Policy]"])
                for idx, rec in enumerate(meso_tips, 1):
                    hmrr_lines.append(f"{idx}. {rec.get('message', '')}")
                hmrr_lines.append("")
        hmrr_block = ""
        if hmrr_lines:
            hmrr_text = "\n".join(hmrr_lines).strip()
            hmrr_block = f"\nHIERARCHICAL MEMORY CONTEXT:\n{hmrr_text}\n"

        # Build prompt
        prompt = f"""You are a research strategist designing a materials discovery campaign.

CURRENT ITERATION RESULTS:
{summary}

INDIVIDUAL STRUCTURE ANALYSES:
{ind_summary}

GLOBAL ANALYSIS:
{reflection.get('llm_analysis', 'N/A')[:500]}...{hmrr_block}

TASK:
Based on these results, design 2-5 distinct search strategies for the next iteration.

GUIDELINES:
- If you have clear successes (Ed < 0.1): create exploitation strategies to build on them
- If you have moderate performers (0.1 < Ed < 0.5): create improvement strategies
- Always include at least one exploration strategy to discover new chemical spaces
- Avoid redundant strategies - each should have a distinct goal and approach
- Strategies should learn from both successes AND failures

For EACH strategy, provide:
1. **strategy_name**: Short identifier (e.g., "exploit_simple_binaries", "improve_perovskites")
2. **strategy_type**: One of ["exploit", "improve", "explore"]
3. **description**: What this strategy aims to do (1-2 sentences)
4. **instructions**: Specific instructions for structure generation (what elements, structures, compositions to try)
5. **target_ed_range**: Expected Ed range [min, max] in eV/atom
6. **allocation_weight**: Relative priority 1-10 (higher = more structures allocated)

Respond in JSON format:
{{
  "strategies": [
    {{
      "strategy_name": "descriptive_name",
      "strategy_type": "exploit",
      "description": "Brief description of what this strategy does",
      "instructions": "Detailed instructions for generating structures following this strategy",
      "target_ed_range": [0.0, 0.1],
      "allocation_weight": 8
    }},
    ...
  ],
  "reasoning": "Brief explanation (2-3 sentences) of why you chose these strategies and their weights"
}}

IMPORTANT:
- Generate 2-5 strategies (your choice based on results)
- Higher allocation_weight = more structures will use this strategy
- Be specific in instructions (mention element families, structure types, etc.)
- Provide ONLY the JSON output, no additional text
"""

        return prompt

    def _fallback_strategies(self, reflection: Dict) -> Dict:
        """Fallback strategies when LLM fails"""
        logger.warning("Using fallback strategies due to LLM failure")

        stats = reflection
        best_ed = stats.get('best_ed', float('inf'))

        # Simple rule-based fallback
        if best_ed < 0.1:
            # Has success
            strategies = [
                {
                    'strategy_name': 'exploit_successful',
                    'strategy_type': 'exploit',
                    'description': 'Build on successful structures',
                    'instructions': 'Generate structures similar to the successful ones. Use similar elements and compositions.',
                    'target_ed_range': [0.0, 0.1],
                    'allocation_weight': 7
                },
                {
                    'strategy_name': 'explore_new',
                    'strategy_type': 'explore',
                    'description': 'Explore new chemical spaces',
                    'instructions': 'Try different element combinations and structure types.',
                    'target_ed_range': [0.0, 0.3],
                    'allocation_weight': 3
                }
            ]
        else:
            # No clear success
            strategies = [
                {
                    'strategy_name': 'explore_simple_binaries',
                    'strategy_type': 'explore',
                    'description': 'Focus on simple binary compounds',
                    'instructions': 'Generate simple binary compounds with common elements.',
                    'target_ed_range': [0.0, 0.2],
                    'allocation_weight': 5
                },
                {
                    'strategy_name': 'explore_ternaries',
                    'strategy_type': 'explore',
                    'description': 'Try ternary compounds',
                    'instructions': 'Generate ternary compounds with well-known structure types.',
                    'target_ed_range': [0.0, 0.3],
                    'allocation_weight': 5
                }
            ]

        return {
            'strategies': strategies,
            'reasoning': 'Fallback strategies used due to LLM error'
        }
