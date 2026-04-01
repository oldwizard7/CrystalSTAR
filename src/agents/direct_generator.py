"""Direct Generator Agent - Ablation baseline for tool_calling

Instead of selecting tools to modify parent structures, this agent
directly prompts the LLM to output new crystal structures in CIF format.

Used for ablation experiments: ablation/direct_llm/
"""

import logging
import json
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class DirectGenerationAgent:
    """
    Ablation baseline agent: LLM directly generates crystal structures in CIF format.

    Replaces the ToolSelectionAgent + operator execution in the generate phase.
    All other ACE loop components (reflect, curate, select) remain unchanged.

    Interface mirrors ToolSelectionAgent.select_tools() so orchestrator.py
    only needs a single if/else branch.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.llm = LLMClient(config['llm'])

    def generate_structures(
        self,
        example_pool: List[CrystalStructure],
        strategy: Optional[str],
        reflection: Dict,
        n_structures: int,
        output_dir: str = None,
        iteration: int = None,
        all_structures: List[CrystalStructure] = None,
        strategies_data: Dict = None,
        max_iterations: int = None,
        hmrr_context: Optional[Dict] = None,
    ) -> List[CrystalStructure]:
        """
        Generate new crystal structures by prompting the LLM to output CIF text.

        Args:
            example_pool: Current parent pool structures
            strategy: Strategy description from curate phase
            reflection: Performance reflection from last iteration
            n_structures: Number of structures to generate
            output_dir: Output directory for saving prompts/outputs
            iteration: Current iteration number
            all_structures: All generated structures (for diversity tracking)
            strategies_data: Multi-strategy data from hypothesis agent
            max_iterations: Total iterations (for context)
            hmrr_context: Hierarchical memory context dict

        Returns:
            List of CrystalStructure objects with metadata['source'] = 'direct_llm'
        """
        prompt = self._build_cif_prompt(
            example_pool=example_pool,
            strategy=strategy,
            reflection=reflection,
            n_structures=n_structures,
            all_structures=all_structures,
            strategies_data=strategies_data,
            iteration=iteration,
            max_iterations=max_iterations,
            hmrr_context=hmrr_context,
        )

        logger.info("=" * 60)
        logger.info("DIRECT GENERATOR PROMPT:")
        logger.info("=" * 60)
        logger.info(prompt)

        # Save prompt to file
        if output_dir is not None and iteration is not None:
            log_dir = Path(output_dir) / f"iteration_{iteration}" / "direct_generator"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        # Call LLM (single response containing all N CIF blocks)
        responses = self.llm.generate(prompt, n=1)
        raw_text = responses[0] if responses else ""

        logger.info("=" * 60)
        logger.info("DIRECT GENERATOR RAW OUTPUT:")
        logger.info("=" * 60)
        logger.info(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))

        # Save raw output
        if output_dir is not None and iteration is not None:
            (log_dir / "output.txt").write_text(raw_text, encoding="utf-8")

        # Parse CIF blocks → CrystalStructure objects
        structures = self._parse_cif_response(raw_text, n_structures)

        # Log and save stats
        stats = {
            "total_requested": n_structures,
            "parse_success": len(structures),
            "parse_failure": n_structures - len(structures),
            "parse_rate": len(structures) / n_structures if n_structures > 0 else 0.0,
        }
        logger.info(
            f"Direct LLM generation: {len(structures)}/{n_structures} CIF blocks "
            f"parsed successfully (rate={stats['parse_rate']:.1%})"
        )
        if output_dir is not None and iteration is not None:
            (log_dir / "stats.json").write_text(
                json.dumps(stats, indent=2), encoding="utf-8"
            )

        return structures

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_cif_prompt(
        self,
        example_pool: List[CrystalStructure],
        strategy: Optional[str],
        reflection: Dict,
        n_structures: int,
        all_structures: Optional[List[CrystalStructure]],
        strategies_data: Optional[Dict],
        iteration: Optional[int],
        max_iterations: Optional[int],
        hmrr_context: Optional[Dict],
    ) -> str:
        task_name = self.config.get("task_description", "novel stable materials")
        task_constraints = self.config.get("task_constraints", {}) or {}

        # --- Constraint summary ---
        constraint_lines = []
        for prop, defn in task_constraints.items():
            if prop == "is_valid" or not isinstance(defn, dict):
                continue
            if not defn.get("enabled", True):
                continue
            if "exclude_elements" in defn:
                elems = defn["exclude_elements"]
                constraint_lines.append(f"  - Exclude elements: {elems}")
            elif "min" in defn and "max" in defn:
                constraint_lines.append(f"  - {prop}: {defn['min']} to {defn['max']}")
            elif "min" in defn:
                constraint_lines.append(f"  - {prop}: >= {defn['min']}")
            elif "max" in defn:
                constraint_lines.append(f"  - {prop}: <= {defn['max']}")
        constraint_summary = (
            "\n".join(constraint_lines) if constraint_lines else "  - Thermodynamic stability (low decomposition energy)"
        )

        # --- Parent pool summary ---
        _PROP_LABEL = {
            "band_gap": ("band_gap", 3),
            "formation_energy": ("Ef", 3),
            "bulk_modulus": ("bulk_mod", 1),
            "shear_modulus": ("shear_mod", 1),
            "density": ("density", 2),
            "piezoelectric_coefficient": ("piezo", 2),
            "dielectric_constant": ("dielectric", 1),
            "decomposition_energy": ("Ed", 4),
        }
        tracked_props = {
            p for p, d in task_constraints.items()
            if p != "is_valid" and isinstance(d, dict) and d.get("enabled", True)
            and "exclude_elements" not in d
        }

        parent_lines = []
        for i, cs in enumerate(example_pool):
            parts = [f"Parent {i}: {cs.formula}"]

            # Space group
            sg = cs.metadata.get("space_group") or cs.metadata.get("spacegroup")
            if sg:
                parts.append(f"sg={sg}")

            # Lattice parameters
            if cs.lattice is not None:
                from pymatgen.core import Lattice
                latt = Lattice(cs.lattice)
                parts.append(
                    f"a={latt.a:.2f} b={latt.b:.2f} c={latt.c:.2f} Å"
                )

            # Ed
            if cs.decomposition_energy is not None:
                parts.append(f"Ed={cs.decomposition_energy:.4f} eV/atom")

            # Task-relevant properties
            for prop in sorted(tracked_props):
                val = cs.properties.get(prop)
                if val is None:
                    val = cs.metadata.get(prop)
                if val is not None and prop in _PROP_LABEL:
                    label, digits = _PROP_LABEL[prop]
                    parts.append(f"{label}={val:.{digits}f}")

            parent_lines.append(" | ".join(parts))
        parent_summary = (
            "\n".join(parent_lines)
            if parent_lines
            else "  (no parent structures available - generate from scratch)"
        )

        # --- Diversity: overused formulas ---
        repeated_formulas: List[str] = []
        if all_structures:
            from collections import Counter
            formula_counts = Counter(s.formula for s in all_structures)
            repeated_formulas = [f for f, cnt in formula_counts.most_common(10) if cnt >= 3]
        avoid_block = (
            f"Avoid overusing these formulas (already seen >= 3 times): {repeated_formulas}"
            if repeated_formulas
            else ""
        )

        # --- Reflection summary ---
        refl_block = ""
        if reflection:
            valid_rate = reflection.get("valid_rate", 0.0)
            hit_rate = reflection.get("hit_rate", 0.0)
            best_ed = reflection.get("best_ed")
            refl_parts = [f"valid_rate={valid_rate:.1f}%", f"hit_rate={hit_rate:.1f}%"]
            if best_ed is not None:
                refl_parts.append(f"best_Ed={best_ed:.4f} eV/atom")
            refl_block = f"Previous iteration: {', '.join(refl_parts)}"

        # --- Strategy ---
        strategy_block = ""
        if strategies_data and isinstance(strategies_data, dict):
            strats = strategies_data.get("strategies", [])
            if strats:
                strategy_block = "Current strategies:\n"
                for s in strats[:3]:
                    strategy_block += f"  [{s.get('strategy_type','?')}] {s.get('description','')}\n"
        elif strategy:
            strategy_block = f"Current strategy: {strategy}"

        # --- HMRR context (macro rules) ---
        hmrr_block = ""
        if hmrr_context:
            macro = hmrr_context.get("macro_rules", [])
            if macro:
                hmrr_block = "Key findings from previous runs:\n"
                for rule in macro[:3]:
                    hmrr_block += f"  - {rule}\n"

        # --- Iteration context ---
        iter_block = ""
        if iteration is not None and max_iterations is not None:
            iter_block = f"Iteration {iteration}/{max_iterations}."

        # --- Assemble prompt ---
        prompt = f"""You are an expert materials scientist specializing in crystal structure design.

TASK: Design novel {task_name}.

HARD CONSTRAINTS (structures must satisfy ALL of these):
{constraint_summary}

SOFT GOAL: Minimize decomposition energy (thermodynamic stability). Target: Ed < 0.1 eV/atom.

{iter_block}
{refl_block}

{strategy_block}

{hmrr_block}

PARENT POOL (use as inspiration - do NOT copy directly):
{parent_summary}

{avoid_block}

YOUR TASK:
Generate EXACTLY {n_structures} new crystal structures in CIF format.

REQUIREMENTS:
- Each structure must plausibly satisfy the hard constraints listed above
- Use realistic, physically meaningful lattice parameters
- Keep total atom count between 2 and 30 per unit cell
- Generate compounds with at least 2 distinct elements
- Each structure should be meaningfully different from the parents and from each other
- Do NOT repeat formulas from the avoid list above

OUTPUT FORMAT (STRICTLY FOLLOW):
- Output EXACTLY {n_structures} CIF blocks
- Separate each block with the exact marker: ### STRUCTURE_END ###
- Each CIF block must begin with "data_structure_N" (N = 1, 2, ...)
- Do NOT output markdown code fences (no backticks ```)
- Do NOT include any explanation text between blocks

CRITICAL: Atom sites MUST use CIF loop_ syntax (NOT key-value pairs). Follow this example exactly:

data_example
_cell_length_a 3.99
_cell_length_b 3.99
_cell_length_c 4.03
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 4 m m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ba1 Ba 0.0 0.0 0.0
Ti1 Ti 0.5 0.5 0.5
O1  O  0.5 0.5 0.0
O2  O  0.5 0.0 0.5
O3  O  0.0 0.5 0.5
### STRUCTURE_END ###

BEGIN OUTPUT:
"""
        return prompt.strip()

    # ------------------------------------------------------------------
    # CIF parsing
    # ------------------------------------------------------------------

    def _parse_cif_response(
        self, raw_text: str, expected_n: int
    ) -> List[CrystalStructure]:
        """Split raw LLM output into CIF blocks and parse each one."""
        blocks = self._split_cif_blocks(raw_text)
        structures = []
        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue
            try:
                pmg_structs = self._parse_single_cif(block)
                if not pmg_structs:
                    logger.warning(f"CIF block {i+1}: CifParser returned empty list")
                    continue
                pmg_struct = pmg_structs[0]
                cs = self._pmg_to_crystal_structure(pmg_struct, block_index=i + 1)
                structures.append(cs)
                logger.debug(f"CIF block {i+1}: parsed OK → {cs.formula}")
            except Exception as e:
                logger.warning(f"CIF block {i+1}: parse failed: {e}")
        return structures

    def _split_cif_blocks(self, text: str) -> List[str]:
        """Split LLM output into individual CIF blocks."""
        # Strip markdown code fences (LLM often wraps output in ```...```)
        text = re.sub(r'```[^\n]*\n?', '', text)

        # Strategy 1: split on ### marker (accepts both ### and ### STRUCTURE_END ###)
        if "###" in text:
            blocks = re.split(r'###(?:\s*STRUCTURE_END\s*###)?', text)
            filtered = [b for b in blocks if b.strip().startswith("data_")]
            if filtered:
                return filtered

        # Strategy 2: split on "data_" headers (standard CIF convention)
        parts = re.split(r'(?=^data_)', text, flags=re.MULTILINE)
        return [p for p in parts if p.strip().startswith("data_")]

    def _parse_single_cif(self, cif_text: str):
        """Parse a single CIF block string → list of pymatgen Structures."""
        from pymatgen.io.cif import CifParser

        def _try_parse(parser):
            """Try primitive=True first (no spacegroup lookup needed), then primitive=False."""
            try:
                structs = parser.parse_structures(primitive=True)
                if structs:
                    return structs
            except Exception:
                pass
            return parser.parse_structures(primitive=False)

        # Try from_str (pymatgen >= 2022.x) with lenient occupancy tolerance
        if hasattr(CifParser, 'from_str'):
            try:
                parser = CifParser.from_str(cif_text, occupancy_tolerance=100.0)
                return _try_parse(parser)
            except Exception:
                pass  # Fall through to tempfile

        # Fallback: write to tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cif", delete=False, encoding="utf-8"
        ) as f:
            f.write(cif_text)
            tmp_path = f.name
        try:
            parser = CifParser(tmp_path, occupancy_tolerance=100.0)
            return _try_parse(parser)
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def _pmg_to_crystal_structure(self, pmg_struct, block_index: int) -> CrystalStructure:
        """Convert a pymatgen Structure to CrystalStructure."""
        # Reject over-expanded structures (space group symmetry can blow up atom count)
        if len(pmg_struct) > 30:
            raise ValueError(
                f"Structure has {len(pmg_struct)} atoms after symmetry expansion "
                f"(limit: 30). Likely unphysical (atoms overlapping)."
            )

        lattice_matrix = pmg_struct.lattice.matrix.copy()
        positions = pmg_struct.frac_coords.copy()
        species = []
        for site in pmg_struct:
            try:
                species.append(str(site.specie))
            except AttributeError:
                # pymatgen version compat: site.specie removed; fall back to species_string
                sp_str = site.species_string
                # For disordered sites (mixed occupancy), take dominant species
                if ":" in sp_str:
                    dominant = max(site.species.keys(), key=lambda el: site.species[el])
                    species.append(str(dominant))
                else:
                    species.append(sp_str)
        formula = pmg_struct.formula

        cs = CrystalStructure(
            formula=formula,
            lattice=lattice_matrix,
            positions=positions,
            species=species,
            is_valid=False,
            decomposition_energy=None,
        )
        cs.metadata["source"] = "direct_llm"
        cs.metadata["cif_block_index"] = block_index

        # Try to get space group for logging/prompt context in future iterations
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(pmg_struct, symprec=0.1)
            cs.metadata["space_group"] = sga.get_space_group_symbol()
        except Exception:
            pass

        return cs
