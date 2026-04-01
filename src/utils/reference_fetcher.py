"""
Task-specific reference example fetcher from Materials Project

Dynamically queries MP database for relevant reference structures based on task constraints.
Provides both positive examples (meeting constraints) and negative examples (failing constraints).
"""

import logging
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskReferenceProvider:
    """
    Provides task-specific reference examples from Materials Project

    Features:
    - Dynamically queries MP based on task constraints
    - Returns both success and failure examples
    - Caches results to avoid repeated API calls
    - Supports fallback to hardcoded examples if API unavailable
    """

    # Hardcoded fallback examples for each task
    FALLBACK_REFERENCES = {
        "hard_stiff_ceramics": {
            "positive_formulas": ["SiC", "TiC", "Al2O3", "Si3N4", "B4C"],
            "positive_mp_ids": ["mp-8062", "mp-1519", "mp-1143", "mp-2400", "mp-160"],
            "negative_formulas": ["MgO", "CaO"],  # Softer oxides
            "negative_mp_ids": ["mp-1265", "mp-2605"]
        },
        "wide_bandgap_semiconductors": {
            "positive_formulas": ["GaN", "AlN", "SiC", "ZnO"],
            "positive_mp_ids": ["mp-804", "mp-661", "mp-8062", "mp-2133"],
            "negative_formulas": ["Si", "Ge"],  # Narrow bandgap
            "negative_mp_ids": ["mp-149", "mp-32"]
        },
        # Add more tasks as needed
    }

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None, llm_client=None):
        """
        Initialize reference provider

        Args:
            api_key: Materials Project API key (if None, reads from env)
            cache_dir: Directory to cache downloaded structures
            llm_client: Optional LLM client for intelligent reference selection
        """
        self.api_key = api_key or os.getenv('MATERIALS_PROJECT_API_KEY') or os.getenv('MP_API_KEY')
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/mp_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client

        if not self.api_key:
            logger.warning("MP API key not found - will use fallback hardcoded examples only")

    def get_task_references(
        self,
        task_name: str,
        task_constraints: Dict,
        n_positive: int = 3,
        n_negative: int = 2,
        use_api: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Get task-specific reference examples

        Args:
            task_name: Name of the task (e.g., "hard_stiff_ceramics")
            task_constraints: Task constraint definitions
            n_positive: Number of positive examples (structures meeting constraints)
            n_negative: Number of negative examples (structures failing constraints)
            use_api: Whether to use MP API (if False, use only fallback examples)

        Returns:
            (positive_examples, negative_examples) - Lists of structure dicts
        """
        logger.info(f"Fetching reference examples for task: {task_name}")
        logger.info(f"  Requesting {n_positive} positive + {n_negative} negative examples")

        # Try LLM-assisted selection first (if LLM client available)
        if self.llm_client and use_api and self.api_key:
            try:
                logger.info("  Using LLM-assisted reference selection...")
                positive, negative = self._llm_assisted_selection(
                    task_name, task_constraints, n_positive, n_negative
                )
                if positive:
                    logger.info(f"  ✓ LLM selected {len(positive)} positive + {len(negative)} negative examples")
                    return positive, negative
            except Exception as e:
                logger.warning(f"  LLM-assisted selection failed: {e}")
                logger.info("  Falling back to constraint-based query...")

        # Try API-based query
        if use_api and self.api_key:
            try:
                positive, negative = self._query_mp_by_constraints(
                    task_constraints, n_positive, n_negative
                )
                if positive:
                    logger.info(f"  ✓ Retrieved {len(positive)} positive + {len(negative)} negative examples from MP API")
                    return positive, negative
            except Exception as e:
                logger.warning(f"  MP API query failed: {e}")
                logger.info("  Falling back to hardcoded examples...")

        # Fallback to hardcoded examples
        return self._get_fallback_examples(task_name, n_positive, n_negative)

    def _query_mp_by_constraints(
        self,
        task_constraints: Dict,
        n_positive: int,
        n_negative: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Query Materials Project based on task constraints

        Args:
            task_constraints: Constraint definitions from task config
            n_positive: Number of positive examples
            n_negative: Number of negative examples

        Returns:
            (positive_examples, negative_examples)
        """
        from mp_api.client import MPRester

        positive_examples = []
        negative_examples = []

        with MPRester(self.api_key) as mpr:
            # Build query for positive examples (structures meeting constraints)
            positive_query = self._build_mp_query(task_constraints, target="positive")

            # Build query for negative examples (structures failing constraints)
            negative_query = self._build_mp_query(task_constraints, target="negative")

            # Query positive examples
            if positive_query:
                logger.debug(f"  Positive query: {positive_query}")
                positive_docs = mpr.materials.summary.search(
                    **positive_query,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "structure",
                        "formation_energy_per_atom",
                        "energy_above_hull",
                        "bulk_modulus",
                        "shear_modulus",
                    ],
                    num_chunks=1,
                    chunk_size=n_positive * 3  # Get extra for filtering
                )

                # Convert to our format and filter
                for doc in positive_docs[:n_positive]:
                    positive_examples.append(self._convert_mp_doc(doc))

            # Query negative examples
            if negative_query:
                logger.debug(f"  Negative query: {negative_query}")
                negative_docs = mpr.materials.summary.search(
                    **negative_query,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "structure",
                        "formation_energy_per_atom",
                        "energy_above_hull",
                        "bulk_modulus",
                        "shear_modulus",
                    ],
                    num_chunks=1,
                    chunk_size=n_negative * 3
                )

                for doc in negative_docs[:n_negative]:
                    negative_examples.append(self._convert_mp_doc(doc))

        return positive_examples, negative_examples

    def _build_mp_query(self, task_constraints: Dict, target: str = "positive") -> Dict:
        """
        Build MP API query from task constraints

        Args:
            task_constraints: Task constraint definitions
            target: "positive" (meet constraints) or "negative" (fail constraints)

        Returns:
            Query dict for MPRester.search()
        """
        query = {}

        # Always prefer simple compounds for reference examples
        query["num_elements"] = {"$lte": 3}

        # Only query stable/near-stable materials
        query["energy_above_hull"] = {"$lt": 0.05}  # Within 50 meV/atom of hull

        for prop_name, prop_def in task_constraints.items():
            if prop_name == 'is_valid' or not isinstance(prop_def, dict):
                continue

            if not prop_def.get('enabled', True):
                continue

            # Map to MP field names
            mp_field = self._map_to_mp_field(prop_name)
            if not mp_field:
                continue

            # Build constraint
            if target == "positive":
                # Structures that meet the constraint
                if 'min' in prop_def and 'max' in prop_def:
                    query[mp_field] = {
                        "$gte": prop_def['min'],
                        "$lte": prop_def['max']
                    }
                elif 'min' in prop_def:
                    query[mp_field] = {"$gte": prop_def['min']}
                elif 'max' in prop_def:
                    query[mp_field] = {"$lte": prop_def['max']}

            else:  # target == "negative"
                # Structures that fail the constraint (useful as "what to avoid")
                if 'min' in prop_def:
                    # Below minimum threshold
                    query[mp_field] = {"$lt": prop_def['min'] * 0.5}  # Significantly below

        return query

    def _map_to_mp_field(self, property_name: str) -> Optional[str]:
        """
        Map our property names to Materials Project field names

        Args:
            property_name: Our property name

        Returns:
            MP field name or None if not supported
        """
        mapping = {
            # Summary search does not support nested elasticity.*; use flat fields
            "bulk_modulus": "bulk_modulus",
            "shear_modulus": "shear_modulus",
            "band_gap": "band_gap",
            "density": "density",
            "formation_energy": "formation_energy_per_atom"
        }
        return mapping.get(property_name)

    def _convert_mp_doc(self, doc) -> Dict:
        """
        Convert MP document to our structure format

        Args:
            doc: MP summary document

        Returns:
            Structure dict compatible with CrystalStructure
        """
        from src.core.structure import CrystalStructure
        import numpy as np

        # Get pymatgen structure
        pmg_structure = doc.structure

        # Convert to our format
        structure = CrystalStructure(
            formula=doc.formula_pretty,
            lattice=np.array(pmg_structure.lattice.matrix),
            positions=np.array([site.frac_coords for site in pmg_structure]),
            species=[str(site.specie) for site in pmg_structure],
            energy=doc.formation_energy_per_atom if hasattr(doc, 'formation_energy_per_atom') else None,
            decomposition_energy=doc.energy_above_hull if hasattr(doc, 'energy_above_hull') else None,
            metadata={
                'material_id': doc.material_id,
                'source': 'materials_project'
            }
        )

        # Add elastic properties if available
        if hasattr(doc, 'bulk_modulus') and doc.bulk_modulus is not None:
            bm = doc.bulk_modulus
            structure.properties['bulk_modulus'] = bm.vrh if hasattr(bm, "vrh") else bm
        if hasattr(doc, 'shear_modulus') and doc.shear_modulus is not None:
            gm = doc.shear_modulus
            structure.properties['shear_modulus'] = gm.vrh if hasattr(gm, "vrh") else gm

        return structure

    def _get_fallback_examples(
        self,
        task_name: str,
        n_positive: int,
        n_negative: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Get hardcoded fallback examples when API is unavailable

        Args:
            task_name: Task name
            n_positive: Number of positive examples
            n_negative: Number of negative examples

        Returns:
            (positive_examples, negative_examples)
        """
        fallback = self.FALLBACK_REFERENCES.get(task_name)

        if not fallback:
            logger.warning(f"  No fallback examples defined for task '{task_name}'")
            return [], []

        positive = []
        negative = []

        # Try to load from MP IDs
        if self.api_key:
            try:
                from mp_api.client import MPRester
                with MPRester(self.api_key) as mpr:
                    # Load positive examples
                    for mp_id in fallback['positive_mp_ids'][:n_positive]:
                        try:
                            doc = mpr.materials.summary.get_data_by_id(
                                mp_id,
                                fields=[
                                    "material_id",
                                    "formula_pretty",
                                    "structure",
                                    "formation_energy_per_atom",
                                    "energy_above_hull",
                                    "bulk_modulus",
                                    "shear_modulus",
                                ]
                            )
                            positive.append(self._convert_mp_doc(doc))
                        except Exception as e:
                            logger.warning(f"  Failed to load MP ID {mp_id}: {e}")

                    # Load negative examples
                    for mp_id in fallback['negative_mp_ids'][:n_negative]:
                        try:
                            doc = mpr.materials.summary.get_data_by_id(
                                mp_id,
                                fields=[
                                    "material_id",
                                    "formula_pretty",
                                    "structure",
                                    "formation_energy_per_atom",
                                    "energy_above_hull",
                                    "bulk_modulus",
                                    "shear_modulus",
                                ]
                            )
                            negative.append(self._convert_mp_doc(doc))
                        except Exception as e:
                            logger.warning(f"  Failed to load MP ID {mp_id}: {e}")

                if positive:
                    logger.info(f"  ✓ Loaded {len(positive)} positive + {len(negative)} negative examples from fallback MP IDs")
                    return positive, negative

            except Exception as e:
                logger.warning(f"  Failed to load fallback examples from MP: {e}")

        # Last resort: return empty (will use whatever examples are in data/ directory)
        logger.warning(f"  Could not load reference examples - generation will proceed without MP-based references")
        return [], []

    def _llm_assisted_selection(
        self,
        task_name: str,
        task_constraints: Dict,
        n_positive: int,
        n_negative: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Use LLM to intelligently select relevant reference materials

        Args:
            task_name: Task name
            task_constraints: Task constraints
            n_positive: Number of positive examples
            n_negative: Number of negative examples

        Returns:
            (positive_examples, negative_examples)
        """
        # Build LLM prompt to suggest materials
        prompt = self._build_llm_selection_prompt(task_name, task_constraints, n_positive, n_negative)

        # Call LLM
        responses = self.llm_client.generate(prompt, n=1)
        if not responses or not responses[0]:
            raise RuntimeError("LLM returned empty response for reference selection")
        response = responses[0]

        # Parse LLM response to extract material formulas
        positive_formulas, negative_formulas = self._parse_llm_material_suggestions(response)

        logger.debug(f"  LLM suggested positive: {positive_formulas}")
        logger.debug(f"  LLM suggested negative: {negative_formulas}")

        # Query MP for these specific materials
        from mp_api.client import MPRester

        positive_examples = []
        negative_examples = []

        with MPRester(self.api_key) as mpr:
            # Query positive examples
            for formula in positive_formulas[:n_positive]:
                try:
                    docs = mpr.materials.summary.search(
                        formula=formula,
                        fields=[
                            "material_id",
                            "formula_pretty",
                            "structure",
                            "formation_energy_per_atom",
                            "energy_above_hull",
                            "bulk_modulus",
                            "shear_modulus",
                        ],
                        num_chunks=1,
                        chunk_size=1
                    )
                    if docs:
                        positive_examples.append(self._convert_mp_doc(docs[0]))
                except Exception as e:
                    logger.warning(f"  Failed to query {formula}: {e}")

            # Query negative examples
            for formula in negative_formulas[:n_negative]:
                try:
                    docs = mpr.materials.summary.search(
                        formula=formula,
                        fields=[
                            "material_id",
                            "formula_pretty",
                            "structure",
                            "formation_energy_per_atom",
                            "energy_above_hull",
                            "bulk_modulus",
                            "shear_modulus",
                        ],
                        num_chunks=1,
                        chunk_size=1
                    )
                    if docs:
                        negative_examples.append(self._convert_mp_doc(docs[0]))
                except Exception as e:
                    logger.warning(f"  Failed to query {formula}: {e}")

        return positive_examples, negative_examples

    def _build_llm_selection_prompt(
        self,
        task_name: str,
        task_constraints: Dict,
        n_positive: int,
        n_negative: int
    ) -> str:
        """
        Build prompt for LLM to suggest relevant materials

        Args:
            task_name: Task name
            task_constraints: Task constraints
            n_positive: Number of positive examples needed
            n_negative: Number of negative examples needed

        Returns:
            Prompt string
        """
        # Format constraints
        constraints_text = []
        for prop_name, prop_def in task_constraints.items():
            if prop_name == 'is_valid' or not isinstance(prop_def, dict):
                continue
            if not prop_def.get('enabled', True):
                continue

            if 'min' in prop_def and 'max' in prop_def:
                constraints_text.append(f"  - {prop_name}: {prop_def['min']} to {prop_def['max']}")
            elif 'min' in prop_def:
                constraints_text.append(f"  - {prop_name}: ≥ {prop_def['min']}")
            elif 'max' in prop_def:
                constraints_text.append(f"  - {prop_name}: ≤ {prop_def['max']}")

        prompt = f"""You are a materials science expert. Select the most relevant reference materials for the following task.

TASK: {task_name}

CONSTRAINTS:
{chr(10).join(constraints_text) if constraints_text else "  (No specific constraints)"}

Your task is to suggest:
1. {n_positive} well-known materials that MEET these constraints (positive examples)
2. {n_negative} well-known materials that FAIL these constraints (negative examples)

Requirements:
- Choose SIMPLE, well-studied materials (binary or ternary compounds preferred)
- Positive examples should be archetypal materials for this task
- Negative examples should clearly fail the constraints but be chemically similar
- Only suggest materials that exist in Materials Project database

Output format (JSON):
{{
  "positive": ["Formula1", "Formula2", ...],
  "negative": ["Formula3", "Formula4", ...],
  "reasoning": "Brief explanation of your choices"
}}

Example for "Hard Ceramics" task:
{{
  "positive": ["SiC", "TiC", "Al2O3"],
  "negative": ["MgO", "CaO"],
  "reasoning": "SiC, TiC, Al2O3 are archetypal hard ceramics with high bulk/shear modulus. MgO and CaO are structurally similar oxides but much softer."
}}

Provide ONLY the JSON output, no additional text."""

        return prompt

    def _parse_llm_material_suggestions(self, response: str) -> Tuple[List[str], List[str]]:
        """
        Parse LLM response to extract material formulas

        Args:
            response: LLM response text

        Returns:
            (positive_formulas, negative_formulas)
        """
        import json
        import re

        # Try to extract JSON from response
        # Look for JSON block (might be wrapped in markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("Could not find JSON in LLM response")

        # Parse JSON
        data = json.loads(json_str)

        positive = data.get('positive', [])
        negative = data.get('negative', [])

        logger.debug(f"  LLM reasoning: {data.get('reasoning', 'N/A')}")

        return positive, negative


def get_task_references(
    task_name: str,
    task_constraints: Dict,
    n_positive: int = 3,
    n_negative: int = 2,
    api_key: Optional[str] = None
) -> Tuple[List, List]:
    """
    Convenience function to get task-specific references

    Args:
        task_name: Task name
        task_constraints: Task constraint definitions
        n_positive: Number of positive examples
        n_negative: Number of negative examples
        api_key: MP API key (optional)

    Returns:
        (positive_examples, negative_examples)
    """
    provider = TaskReferenceProvider(api_key=api_key)
    return provider.get_task_references(
        task_name=task_name,
        task_constraints=task_constraints,
        n_positive=n_positive,
        n_negative=n_negative
    )
