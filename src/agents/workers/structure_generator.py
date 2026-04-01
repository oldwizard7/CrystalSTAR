"""Structure Generator Agent - Generate structures from scratch"""

import logging
import random
from typing import List, Dict
from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient
from src.utils.example_selector import ExampleSelector

logger = logging.getLogger(__name__)


class StructureGeneratorAgent:
    """
    Generate new crystal structures from scratch
    
    Uses reference examples to show format, but generates entirely new structures.
    Does NOT modify existing structures (no crossover/mutation of parents).
    """
    
    def __init__(self, config: Dict):
        """
        Initialize structure generator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm = LLMClient(config['llm'])

        # Example selector for random sampling from MP novel structures
        self.example_selector = ExampleSelector()

        # Number of examples to sample each iteration
        self.num_examples = config.get('generation', {}).get('num_examples', 3)

        # Reference examples (3-5 structures showing format) - legacy support
        self.reference_examples = []
        self.max_reference_examples = (
            config.get('generation', {})
            .get('max_reference_examples', 5)  # allow sampling more if desired
        )

        # Success examples (previously generated good structures)
        self.success_examples = []
        self.max_success_examples = 10

        # Failure examples (previously generated unstable structures) - NEW!
        self.failure_examples = []
        self.max_failure_examples = 3  # Keep only recent failures to avoid token bloat

        # All generated structures (for deduplication feedback to LLM)
        self.generated_structures = []
        self.max_generated_feedback = 10  # Show last N generated structures to LLM
    
    def set_reference_examples(self, examples: List[CrystalStructure]):
        """
        Set reference examples for format demonstration
        
        Args:
            examples: List of reference structures (3-5 examples)
        """
        self.reference_examples = examples
        logger.info(f"Set {len(examples)} reference examples for format")
    
    def add_success_example(self, structure: CrystalStructure):
        """
        Add a successful structure to example pool

        Args:
            structure: A successfully generated structure
        """
        self.success_examples.append(structure)

        # Keep pool size limited (only recent successes)
        if len(self.success_examples) > self.max_success_examples:
            self.success_examples.pop(0)

        logger.debug(f"Added success example: {structure.formula}")

    def add_failure_example(self, structure: CrystalStructure):
        """
        Add a failed structure to negative example pool

        Args:
            structure: A failed (highly unstable) structure
        """
        self.failure_examples.append(structure)

        # Keep pool size limited (only recent failures)
        if len(self.failure_examples) > self.max_failure_examples:
            self.failure_examples.pop(0)

        logger.debug(f"Added failure example: {structure.formula} (Ed={structure.decomposition_energy:.3f})")

    def add_generated_structure(self, structure: CrystalStructure):
        """
        Track all generated structures for deduplication feedback

        Args:
            structure: Any generated structure (successful or failed)
        """
        self.generated_structures.append(structure)

        # Keep pool size limited
        if len(self.generated_structures) > self.max_generated_feedback:
            self.generated_structures.pop(0)

        logger.debug(f"Tracked generated structure: {structure.formula}")
    
    def generate(
        self,
        task_description: str,
        constraints: Dict,
        n_structures: int = 5,
        strategy = None,  # Can be str (Phase 1) or Dict (Phase 2B)
        output_dir: str = None,
        iteration: int = None,
        global_formula_stats: Dict = None,  # Global formula statistics from orchestrator
        use_reference_examples: bool = None  # Explicitly control reference examples (None = auto-decide)
    ) -> List[str]:
        """
        Generate new structures from scratch

        Args:
            task_description: Description of what to generate
                             e.g., "stable wide-bandgap semiconductors"
            constraints: Property constraints
                        e.g., {'band_gap': '>2.5 eV', 'formation_energy': '<-1.0 eV/atom'}
            n_structures: Number of structures to generate
            strategy: Search strategy from curate phase (optional)
                     Can be:
                     - str: Single strategy text (Phase 1)
                     - Dict: Strategy dict with instructions (Phase 2B)
            output_dir: Output directory for saving prompt (optional)
            iteration: Current iteration number (optional)
            global_formula_stats: Global formula statistics (unique formulas, repetition counts)
            use_reference_examples: Whether to include reference examples in prompt
                                   None = auto-decide based on iteration
                                   True = always include
                                   False = never include

        Returns:
            List of generated structures as POSCAR strings
        """
        logger.info(f"Generating {n_structures} new structures for: {task_description}")
        if strategy:
            if isinstance(strategy, dict):
                logger.info(f"Using strategy: {strategy.get('strategy_name', 'unknown')}")
            else:
                logger.info(f"Using strategy: {str(strategy)[:100]}...")

        # Decide whether to include reference examples
        # Auto-decide: use reference examples only in early iterations (1-2)
        # After that, rely on actual success/failure examples from generation
        if use_reference_examples is None:
            use_reference_examples = (iteration is None or iteration <= 2)

        # Build prompt
        ref_examples = self._sample_reference_examples() if use_reference_examples else None
        prompt = self._build_generation_prompt(
            task_description,
            constraints,
            n_structures,
            strategy,
            reference_examples=ref_examples,
            global_formula_stats=global_formula_stats,
            iteration=iteration
        )

        logger.debug(f"Prompt length: {len(prompt)} characters")

        # Save prompt to file
        if output_dir and iteration is not None:
            from pathlib import Path
            prompt_file = Path(output_dir) / f"iteration_{iteration}" / "prompt.txt"
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(prompt)
            logger.debug(f"Prompt saved to {prompt_file}")
        
        # Generate using LLM
        try:
            responses = self.llm.generate(prompt, n=1)
            logger.info(f"LLM generated {len(responses)} response(s)")
            return responses
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return []
    
    def _build_generation_prompt(
        self,
        task_desc: str,
        constraints: Dict,
        n: int,
        strategy = None,  # Can be str or Dict
        reference_examples: str = None,
        global_formula_stats: Dict = None,
        iteration: int = None
    ) -> str:
        """
        Build prompt for structure generation

        Args:
            task_desc: Task description
            constraints: Property constraints
            n: Number of structures to generate
            strategy: Search strategy - can be str (Phase 1) or Dict (Phase 2B)
            reference_examples: Formatted reference examples text (optional)

        Returns:
            Complete prompt string
        """
        # reference_examples is now a formatted string, not a list
        # If not provided, it will be generated by _sample_reference_examples()

        prompt_parts = [
            "You are an expert materials scientist with deep knowledge of crystal structures.",
            "",
            f"TASK: Generate {n} ENTIRELY NEW crystal structures for: {task_desc}",
            "",
            "IMPORTANT:",
            "- Generate COMPLETELY NEW structures (do NOT copy the examples)",
            "- Use CREATIVE element combinations",
            "- Ensure structures are CHEMICALLY PLAUSIBLE",
            "- Follow the POSCAR format exactly as shown",
            "- **PREFER SIMPLE STRUCTURES**: 2-8 atoms total (easier to ensure completeness)",
            "- Binary compounds (AB, AB2) are preferred over complex multi-element systems",
            "",
            "CONSTRAINTS:"
        ]

        # Add constraints
        for key, value in constraints.items():
            if value is not None:
                prompt_parts.append(f"  - {key}: {value}")

        # Add global formula diversity statistics (CRITICAL: prevents repetition)
        if global_formula_stats:
            total_structures = global_formula_stats.get('total_structures', 0)
            unique_formulas = global_formula_stats.get('unique_formulas', 0)
            global_diversity_rate = 100 * unique_formulas / total_structures if total_structures > 0 else 0
            most_repeated = global_formula_stats.get('most_repeated_formulas', {})

            if total_structures > 0 and most_repeated:
                prompt_parts.extend([
                    "",
                    "="*60,
                    "⚠️  AVOID THESE FORMULAS (already generated):",
                    "="*60,
                    ""
                ])
                # Show only top 5 to keep it concise
                for idx, (formula, count) in enumerate(list(most_repeated.items())[:5], 1):
                    prompt_parts.append(f"  ✗ {formula} ({count}×)")

                prompt_parts.extend([
                    "",
                    "Generate NEW formulas not in this list!",
                    "",
                ])

        # Add task-specific element selection guidance
        element_guidance = self._get_task_specific_element_guidance(task_desc)
        if element_guidance:
            prompt_parts.extend([
                "",
                "="*60,
                "ELEMENT SELECTION GUIDANCE (Task-Specific):",
                "="*60,
                ""
            ])
            prompt_parts.extend(element_guidance)
            prompt_parts.append("")

        # Add strategy guidance from curate phase (KEY ADDITION!)
        if strategy:
            prompt_parts.extend([
                "",
                "="*60,
                "SEARCH STRATEGY (Based on previous iteration feedback):",
                "="*60,
                ""
            ])

            # Phase 2B: Handle dict strategy
            if isinstance(strategy, dict):
                prompt_parts.extend([
                    f"Strategy Name: {strategy.get('strategy_name', 'N/A')}",
                    f"Strategy Type: {strategy.get('strategy_type', 'N/A')}",
                    f"Goal: {strategy.get('description', 'N/A')}",
                    "",
                    "SPECIFIC INSTRUCTIONS:",
                    strategy.get('instructions', ''),
                    "",
                    f"TARGET: Decomposition energy in range {strategy.get('target_ed_range', [0.0, 0.3])[0]:.2f} - {strategy.get('target_ed_range', [0.0, 0.3])[1]:.2f} eV/atom",
                    "",
                    "APPLY THE ABOVE STRATEGY when generating new structures.",
                    ""
                ])
            else:
                # Phase 1: Handle string strategy
                prompt_parts.extend([
                    str(strategy),
                    "",
                    "APPLY THE ABOVE STRATEGY when generating new structures.",
                    ""
                ])

        # Add reference examples ONLY in early iterations
        # After iteration 2, we rely on actual generated success/failure examples
        if reference_examples:
            prompt_parts.extend([
                "",
                "="*60,
                "REFERENCE EXAMPLES (Format demonstration - DO NOT COPY):",
                "="*60,
                "",
                "These are known materials from Materials Project that meet the task requirements.",
                "Use them as INSPIRATION for element combinations and structure types.",
                ""
            ])
            prompt_parts.append(reference_examples)
            prompt_parts.append("")
        elif iteration and iteration <= 2:
            # First iterations but no reference examples - show placeholder
            prompt_parts.extend([
                "",
                "="*60,
                "REFERENCE EXAMPLES:",
                "="*60,
                "",
                "(No reference examples available - Materials Project query may have failed)",
                "Focus on generating chemically plausible simple compounds.",
                ""
            ])
        # else: iteration > 2, don't show reference examples section at all

        # Add success/failure examples (keep it concise)
        if self.success_examples:
            prompt_parts.extend([
                "="*60,
                "✓ SUCCESSFUL STRUCTURES (for inspiration):",
                "="*60,
                ""
            ])
            for succ in self.success_examples[-2:]:  # Only show 2 instead of 3
                ed = succ.decomposition_energy if succ.decomposition_energy else "N/A"
                prompt_parts.append(f"  ✓ {succ.formula} (Ed={ed})")
            prompt_parts.append("")

        if self.failure_examples:
            prompt_parts.extend([
                "="*60,
                "✗ FAILED STRUCTURES TO AVOID:",
                "="*60,
                ""
            ])
            for fail in self.failure_examples[-2:]:  # Only show 2 instead of 3
                ed = fail.decomposition_energy if fail.decomposition_energy else "N/A"
                prompt_parts.append(f"  ✗ {fail.formula} (Ed={ed}, too unstable)")
            prompt_parts.append("")

        # Add previously generated structures (simplified)
        if self.generated_structures and len(self.generated_structures) > 3:
            formula_counts = {}
            for struct in self.generated_structures:
                formula_counts[struct.formula] = formula_counts.get(struct.formula, 0) + 1

            if formula_counts:
                prompt_parts.extend([
                    "="*60,
                    "🚫 ALREADY GENERATED (don't repeat):",
                    "="*60,
                    ""
                ])
                # Show only top 5 most repeated
                for formula, count in sorted(formula_counts.items(), key=lambda x: -x[1])[:5]:
                    prompt_parts.append(f"  ✗ {formula} ({count}×)")
                prompt_parts.extend([
                    "",
                    "Generate NEW formulas with different elements!",
                    ""
                ])

        # Add generation instructions (simplified)
        prompt_parts.extend([
            "="*60,
            "YOUR TASK:",
            "="*60,
            "",
            f"Generate {n} NEW structures (2-8 atoms each, SIMPLE is better!)",
            "",
            "="*60,
            "POSCAR FORMAT REQUIREMENTS (STRICTLY REQUIRED):",
            "="*60,
            "",
            "Line 1: Chemical formula (comment)",
            "Line 2: Scaling factor (use 1.0)",
            "Lines 3-5: Three lattice vectors (3 numbers per line, 12 decimal precision)",
            "  Example: 5.640000000000 0.000000000000 0.000000000000",
            "Line 6: Element symbols (space-separated, alphabetically sorted)",
            "  Example: Ga N",
            "Line 7: Atom counts for each element (space-separated, matching order above)",
            "  Example: 2 2",
            "Line 8: Coordinate type (must be 'direct' for fractional coordinates)",
            "Lines 9+: Atomic positions (ONLY 3 fractional coordinates, 12 decimal precision)",
            "  Example: 0.000000000000 0.000000000000 0.000000000000",
            "",
            "CRITICAL REQUIREMENTS:",
            "- Use EXACTLY 12 decimal places for all numbers",
            "- Element symbols on Line 6 MUST match order of atoms in position lines",
            "- Number of position lines MUST equal sum of atom counts",
            "- Each position line MUST have ONLY: <x> <y> <z> (NO element symbols!)",
            "- Fractional coordinates must be between 0.0 and 1.0",
            "- DO NOT use markdown code blocks (```), output raw POSCAR text only",
            "",
            "COMPLETE FORMAT EXAMPLE (4 atoms):",
            "",
            "GaN",
            "1.0",
            "3.189000000000 0.000000000000 0.000000000000",
            "0.000000000000 3.189000000000 0.000000000000",
            "0.000000000000 0.000000000000 5.185000000000",
            "Ga N",
            "2 2",
            "direct",
            "0.333333333333 0.666666666667 0.000000000000",
            "0.666666666667 0.333333333333 0.500000000000",
            "0.333333333333 0.666666666667 0.375000000000",
            "0.666666666667 0.333333333333 0.875000000000",
            "",
            "COMPLETE FORMAT EXAMPLE (8 atoms):",
            "",
            "TiC",
            "1.0",
            "4.328000000000 0.000000000000 0.000000000000",
            "0.000000000000 4.328000000000 0.000000000000",
            "0.000000000000 0.000000000000 4.328000000000",
            "C Ti",
            "4 4",
            "direct",
            "0.000000000000 0.000000000000 0.000000000000",
            "0.500000000000 0.500000000000 0.000000000000",
            "0.500000000000 0.000000000000 0.500000000000",
            "0.000000000000 0.500000000000 0.500000000000",
            "0.500000000000 0.500000000000 0.500000000000",
            "0.000000000000 0.000000000000 0.500000000000",
            "0.000000000000 0.500000000000 0.000000000000",
            "0.500000000000 0.000000000000 0.000000000000",
            "",
            "="*60,
            f"NOW GENERATE EXACTLY {n} DIFFERENT STRUCTURES:",
            "="*60,
            "",
            "⚠️⚠️⚠️  CRITICAL OUTPUT REQUIREMENTS - READ CAREFULLY:",
            "",
            "BEFORE you output each structure:",
            "1. COUNT the number of atoms (sum of atom counts on Line 7)",
            "2. OUTPUT EXACTLY that many coordinate lines (no more, no less)",
            "3. DO NOT stop early - complete ALL coordinate lines",
            "",
            "VERIFICATION CHECKLIST for EACH structure:",
            "☑ Line 7 shows atom counts (e.g., '4 4' means 8 atoms total)",
            "☑ After 'direct', you MUST output EXACTLY 8 coordinate lines",
            "☑ Each coordinate line has 3 numbers with 12 decimal places",
            "☑ DO NOT truncate or stop early - finish ALL coordinates",
            "",
            "BAD EXAMPLE (INCOMPLETE - will be REJECTED):",
            "ZrC2",
            "1.0",
            "...",
            "C Zr",
            "8 4       ← This means 12 atoms total!",
            "direct",
            "0.300... 0.300... 0.300...",
            "0.700... 0.700... 0.700...",
            "...only 8 lines... ← WRONG! Need 12 lines!",
            "",
            "GOOD EXAMPLE (COMPLETE):",
            "TiC",
            "1.0",
            "...",
            "C Ti",
            "4 4       ← This means 8 atoms total",
            "direct",
            "0.000... 0.000... 0.000...",
            "0.500... 0.500... 0.000...",
            "0.500... 0.000... 0.500...",
            "0.000... 0.500... 0.500...",
            "0.500... 0.500... 0.500...",
            "0.000... 0.000... 0.500...",
            "0.000... 0.500... 0.000...",
            "0.500... 0.000... 0.000...  ← Exactly 8 lines! CORRECT!",
            "",
            "NOW OUTPUT YOUR STRUCTURES:",
            "- Start each with 'Structure N:'",
            "- Complete ALL coordinate lines for EACH structure",
            "- Double-check atom count matches coordinate lines"
        ])
        
        return "\n".join(prompt_parts)

    def _get_task_specific_element_guidance(self, task_desc: str) -> list:
        """
        Generate task-specific element selection guidance

        Args:
            task_desc: Task description string

        Returns:
            List of guidance strings (one per line)
        """
        # Normalize task description for matching
        task_lower = task_desc.lower()

        # Hard Ceramics / Stiff Materials
        if any(keyword in task_lower for keyword in ['ceramic', 'stiff', 'hard', 'bulk modulus', 'shear modulus']):
            return [
                "🎯 HARD CERAMICS - Recommended Element Combinations:",
                "",
                "⭐ PRIORITY 1 - SIMPLE Binary Compounds (2-8 atoms, EASIEST to generate correctly):",
                "  * Carbides: TiC, ZrC, HfC, WC (rocksalt or simple cubic, 2-8 atoms)",
                "  * Nitrides: TiN, AlN, BN (zinc-blende or wurtzite, 2-8 atoms)",
                "  * Borides: TiB2, ZrB2 (simple hexagonal, 3-6 atoms)",
                "  * Oxides: MgO, ZrO2 (simple cubic or tetragonal, 2-6 atoms)",
                "",
                "💡 KEY: Keep total atom count ≤8 for reliable generation!",
                "",
                "PRIORITY 2 - Ternary Compounds (if binary doesn't work):",
                "  * Sialons: Si-Al-O-N systems",
                "  * Mixed carbides: (Ti,W)C, (Zr,Hf)C",
                "  * Oxynitrides: AlON, SiAlON",
                "",
                "✓ PREFER: Ti, Zr, Hf, W, Mo, Cr, V, Si, Al, B, C, N, O",
                "✓ Keep it SIMPLE: Binary (AB) or ternary (AB2, A2B3) compounds work best!",
                "✓ Avoid complex multi-element oxides (>4 elements)",
                "",
                "❌ AVOID for this task:",
                "  * Semiconductors (Ga, In, As, Sb) - not relevant for ceramics",
                "  * Soft elements (Mg, Zn, Ca) - too low modulus",
                "  * Rare earths (La, Ce, Nd, etc.) - limited data",
                "  * Complex 5-6 element oxides - usually softer",
                "",
                "💡 STRATEGY: Start with known hard ceramics (SiC, TiC, Al2O3) and make small variations!",
            ]

        # Wide Bandgap Semiconductors
        elif any(keyword in task_lower for keyword in ['semiconductor', 'bandgap', 'band gap', 'electronic']):
            return [
                "🎯 WIDE BANDGAP SEMICONDUCTORS - Recommended Elements:",
                "",
                "PRIORITY 1 - III-V Nitrides:",
                "  * GaN, AlN, InN, AlGaN, InGaN",
                "",
                "PRIORITY 2 - II-VI Compounds:",
                "  * ZnO, ZnS, ZnSe, CdS (Eg > 2.5 eV)",
                "",
                "PRIORITY 3 - Others:",
                "  * SiC (various polytypes), Diamond, c-BN",
                "",
                "✓ PREFER: Ga, Al, In, N, Zn, O, S, Se, Si, C",
                "✓ Binary or simple ternary compounds",
                "",
                "❌ AVOID:",
                "  * Narrow bandgap elements: Ge, Sn, Pb, As, Sb",
                "  * Transition metals (unless forming oxides like ZnO)",
            ]

        # High Refractive Index Materials
        elif any(keyword in task_lower for keyword in ['refractive', 'optical', 'photonic']):
            return [
                "🎯 HIGH REFRACTIVE INDEX - Recommended Elements:",
                "",
                "PRIORITY - Heavy element oxides/sulfides:",
                "  * TiO2 (rutile, n~2.6), ZrO2, HfO2, Ta2O5",
                "  * PbS, PbSe, CdTe (chalcogenides)",
                "",
                "✓ PREFER: Ti, Zr, Hf, Ta, Pb, S, Se, Te, O",
                "✓ Heavy atoms → higher refractive index",
                "",
                "❌ AVOID: Light elements (Li, Be, Mg, Al) - low n",
            ]

        # Thermoelectric Materials
        elif any(keyword in task_lower for keyword in ['thermoelectric', 'seebeck', 'zte']):
            return [
                "🎯 THERMOELECTRIC MATERIALS - Recommended Elements:",
                "",
                "PRIORITY - Heavy element chalcogenides/skutterudites:",
                "  * PbTe, Bi2Te3, Sb2Te3, SnSe",
                "  * CoSb3 (skutterudite)",
                "",
                "✓ PREFER: Pb, Bi, Sb, Te, Se, Sn, Co",
                "✓ Heavy atoms with complex structures",
                "",
                "❌ AVOID: Light elements, simple structures",
            ]

        # Default fallback - Generic guidance
        else:
            return [
                "GENERAL ELEMENT SELECTION GUIDANCE:",
                "",
                "✓ PREFER elements with good Materials Project coverage:",
                "  * Common metals: Al, Ti, Zr, Hf, Mg, Zn, Fe, Co, Ni, Cu, Mn, Cr, V, W, Mo",
                "  * Main group: Si, Ge, Sn, B, C, N, P, O, S, Se",
                "",
                "✓ Keep structures SIMPLE:",
                "  * Binary (AB, AB2, A2B3) is better than quaternary or higher",
                "  * Prefer 2-3 elements over 4-6 elements",
                "",
                "❌ AVOID:",
                "  * Rare earth elements (La, Ce, Pr, Nd, etc.) - limited MP coverage",
                "  * Radioactive elements (U, Th, Pu)",
                "  * Noble gases and alkali metals (unless specifically required)",
                "",
                "💡 Focus on the task description to choose appropriate elements!",
            ]

    def _sample_reference_examples(self) -> str:
        """
        Randomly sample reference examples from MP novel structures.
        Returns formatted text ready to insert into prompt.
        """
        # If example_selector has examples, use it
        if self.example_selector and self.example_selector.examples:
            examples = self.example_selector.random_sample(k=self.num_examples)
            return self.example_selector.format_examples_for_prompt(examples)

        # Legacy: If manual reference_examples were set, use those
        if self.reference_examples:
            sampled = self.reference_examples[:self.max_reference_examples]
            parts = []
            for i, ref in enumerate(sampled, 1):
                parts.append(f"Example {i}: {ref.formula}\n{ref.to_poscar()}\n")
            return "\n".join(parts)

        # No examples available
        return "No reference examples available."


# Convenience function for testing
def test_generator():
    """Test the structure generator"""
    config = {
        'llm': {
            'provider': 'openai',
            'model': 'gpt-4',
            'temperature': 0.9,
            'max_tokens': 4096
        }
    }
    
    generator = StructureGeneratorAgent(config)
    
    # Create dummy reference
    from src.core.structure import CrystalStructure
    import numpy as np
    
    ref = CrystalStructure(
        formula="NaCl",
        lattice=np.eye(3) * 5.64,
        positions=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
        species=['Na', 'Cl']
    )
    
    generator.set_reference_examples([ref])
    
    # Generate
    results = generator.generate(
        task_description="stable wide-bandgap semiconductors",
        constraints={'band_gap': '>2.5 eV', 'formation_energy': '<-1.0 eV/atom'},
        n_structures=2
    )
    
    print(f"Generated {len(results)} structures")
    for i, result in enumerate(results):
        print(f"\nStructure {i+1}:")
        print(result[:200] + "...")


if __name__ == "__main__":
    test_generator()
