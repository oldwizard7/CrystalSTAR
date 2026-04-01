"""
LLM-driven Materials Project retriever for task-oriented initial parent pool selection.

This module uses LLM to understand task requirements and automatically query
Materials Project database for relevant seed structures.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter

from pymatgen.core import Structure
from mp_api.client import MPRester

from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class TaskOrientedMPRetriever:
    """
    LLM，Materials Project

    Features:
    - LLM，
    - MP
    - 
    - 
    """

    def __init__(self, config: Dict, llm_client: LLMClient = None):
        """
        Args:
            config: Configuration dict containing:
                - mp_api_key: Materials Project API key
                - cache_dir: Directory for caching query results
                - max_results: Maximum number of structures to query
                - llm_analyzer: LLM configuration for query generation
            llm_client: Optional LLMClient instance (creates new if None)
        """
        self.config = config
        self.cache_dir = Path(config.get('cache_dir', 'data/mp_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # LLM client for query generation
        if llm_client is None:
            llm_config = config.get('llm_analyzer', config.get('llm', {}))
            # Ensure we have a proper config dict for LLMClient
            if 'provider' not in llm_config and 'llm_provider' in config:
                llm_config = {
                    'provider': config.get('llm_provider', 'openai'),
                    'model': config.get('llm_model', 'gpt-4o-mini'),
                    'temperature': llm_config.get('temperature', 0.3),
                    'max_tokens': config.get('max_tokens', 4096)
                }
            self.llm_client = LLMClient(llm_config)
        else:
            self.llm_client = llm_client

        # MP API configuration
        self.mp_api_key = config.get('mp_api_key')
        self.max_results = config.get('max_results', 200)

    def retrieve_initial_pool(
        self,
        task_config: Dict,
        max_count: int = 20,
        force_refresh: bool = False
    ) -> List[CrystalStructure]:
        """
        ：

        Args:
            task_config: Task configuration containing name, description, constraints
            max_count: Maximum number of structures to return
            force_refresh: If True, ignore cache and re-query MP

        Returns:
            List of CrystalStructure objects for initial parent pool
        """
        task_name = task_config.get('name', 'unknown_task')

        # Check cache first
        cache_file = self.cache_dir / f"{task_name.replace(' ', '_').lower()}.json"
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached MP query results from {cache_file}")
            structures = self._load_from_cache(cache_file)
            if structures:
                selected = self._diverse_selection(structures, max_count)
                logger.info(f"Selected {len(selected)} structures from cache")
                return selected

        logger.info("Analyzing task with LLM to generate MP query strategy...")
        query_strategy = self._analyze_task_with_llm(task_config)

        logger.info("Querying Materials Project database...")
        candidates = self._query_mp(query_strategy)

        if not candidates:
            logger.warning("No structures found from MP query")
            return []

        # Save to cache
        self._save_to_cache(cache_file, candidates, query_strategy)

        selected = self._diverse_selection(candidates, max_count)

        logger.info(f"Successfully retrieved {len(selected)} structures for initial parent pool")
        return selected

    def _analyze_task_with_llm(self, task_config: Dict) -> Dict:
        """
        LLM，MP

        Returns:
            Dict containing:
                - elements: List of relevant elements
                - num_elements: [min, max] element count range
                - property_filters: Dict of property constraints
                - stability: Energy above hull threshold
                - reasoning: LLM's explanation
        """
        # Format constraints for prompt
        constraints_str = self._format_constraints(task_config.get('constraints', {}))

        prompt = f"""You are a materials scientist helping to select reference structures from Materials Project database.

TASK INFORMATION:
Name: {task_config.get('name', 'Unknown Task')}
Description: {task_config.get('description', 'No description provided').strip()}

Task Constraints:
{constraints_str}

YOUR TASK:
Based on the task requirements above, suggest optimal query parameters for Materials Project database:

1. **Chemical Elements**: Which 5-10 elements are most likely to produce materials satisfying these constraints?
   - Consider element combinations known for the required properties
   - Prioritize elements commonly used for this type of material

2. **Property Filters**: What min/max values should we use for properties?
   - Use the task constraints as guidance
   - Be slightly more permissive than exact constraints to get diverse candidates

3. **Stability Requirements**: What energy_above_hull threshold is appropriate?
   - For metastable material search: 0.1-0.3 eV/atom
   - For stable material search: 0.0-0.1 eV/atom

4. **Structural Preferences**: Any crystal system or symmetry preferences?

Output ONLY valid JSON in this exact format:
{{
  "elements": ["Ti", "C", "N", "Si", "B", "Al", "Zr", "Hf"],
  "num_elements": [1, 3],
  "property_filters": {{
    "bulk_modulus": {{"min": 80, "max": 350}},
    "shear_modulus": {{"min": 50, "max": 220}}
  }},
  "stability": {{
    "energy_above_hull_max": 0.2
  }},
  "crystal_system": null,
  "reasoning": "Brief explanation of element choices and why they suit this task"
}}

IMPORTANT:
- Only include property_filters for properties mentioned in task constraints
- Be specific about element choices based on material science knowledge
- Keep num_elements small (1-3 or 2-4) for focused search
"""

        response = None
        try:
            response_list = self.llm_client.generate(prompt, n=1)
            if not response_list:
                raise ValueError("Empty LLM response")
            response = response_list[0]

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            query_strategy = json.loads(response)

            logger.info("LLM-generated query strategy:")
            logger.info(f"  Elements: {query_strategy.get('elements', [])}")
            logger.info(f"  Num elements: {query_strategy.get('num_elements', [])}")
            logger.info(f"  Property filters: {query_strategy.get('property_filters', {})}")
            logger.info(f"  Reasoning: {query_strategy.get('reasoning', 'N/A')}")

            return query_strategy

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            if response is not None:
                logger.error(f"Raw response: {response}")
            # Fallback to conservative defaults
            return self._get_default_strategy(task_config)

    def _query_mp(self, query_strategy: Dict) -> List[CrystalStructure]:
        """
        Materials Project

        Args:
            query_strategy: Query parameters from LLM analysis

        Returns:
            List of CrystalStructure objects
        """
        if not self.mp_api_key:
            logger.error("MP_API_KEY not provided in config")
            return []

        try:
            with MPRester(self.mp_api_key) as mpr:
                # Build query parameters
                # Convert element list to chemsys string (e.g., ['Ti', 'C', 'Al'] -> 'Al-C-Ti')
                elements_list = query_strategy.get('elements', [])
                chemsys_str = '-'.join(sorted(elements_list)) if elements_list else None

                query_params = {}
                if chemsys_str:
                    query_params['chemsys'] = chemsys_str
                    logger.debug(f"Converted elements {elements_list} to chemsys '{chemsys_str}'")
                query_params['num_elements'] = tuple(query_strategy.get('num_elements', [1, 3]))

                # Add property filters (map to MP summary field names)
                prop_filters = query_strategy.get('property_filters', {})
                prop_aliases = {
                    "bulk_modulus": "k_vrh",
                    "shear_modulus": "g_vrh",
                }
                prop_field_aliases = {
                    "bulk_modulus": "bulk_modulus",
                    "shear_modulus": "shear_modulus",
                }
                mp_prop_fields = []
                for prop, constraints in prop_filters.items():
                    mp_prop = prop_aliases.get(prop, prop)
                    min_val = constraints.get('min')
                    max_val = constraints.get('max')
                    if min_val is None or max_val is None:
                        logger.warning(f"Skipping property filter '{prop}' missing min/max bounds")
                        continue
                    if mp_prop not in {"k_vrh", "k_voigt", "k_reuss", "g_vrh", "g_voigt", "g_reuss"}:
                        logger.warning(f"Skipping unsupported MP property filter: {mp_prop}")
                        continue
                    query_params[mp_prop] = (min_val, max_val)
                    field_name = prop_field_aliases.get(prop)
                    if field_name:
                        mp_prop_fields.append(field_name)

                # Add stability filter
                stability = query_strategy.get('stability', {})
                if 'energy_above_hull_max' in stability:
                    query_params['energy_above_hull'] = (0, stability['energy_above_hull_max'])

                # Crystal system filter (optional)
                crystal_system = query_strategy.get('crystal_system')
                if crystal_system:
                    query_params['crystal_system'] = crystal_system

                # Define fields to retrieve
                fields = [
                    'structure', 'material_id', 'formula_pretty',
                    'energy_above_hull', 'formation_energy_per_atom',
                    'band_gap', 'density', 'nsites',
                    'symmetry.crystal_system', 'symmetry.number'
                ]

                # Add property fields
                for prop in mp_prop_fields:
                    if prop not in fields:
                        fields.append(prop)

                logger.info(f"Executing MP query with parameters: {query_params}")

                # Execute query
                docs = mpr.summary.search(
                    **query_params,
                    fields=fields
                )

                logger.info(f"Found {len(docs)} structures from Materials Project")

                # Convert to CrystalStructure objects
                structures = []
                for doc in docs[:self.max_results]:
                    try:
                        cs = self._mp_doc_to_crystal_structure(doc)
                        structures.append(cs)
                    except Exception as e:
                        logger.warning(f"Failed to convert MP doc {doc.material_id}: {e}")
                        continue

                return structures

        except Exception as e:
            logger.error(f"Materials Project query failed: {e}")
            return []

    def _mp_doc_to_crystal_structure(self, doc) -> CrystalStructure:
        """Convert MP document to CrystalStructure"""
        structure = doc.structure

        cs = CrystalStructure(
            structure=structure,
            formula=doc.formula_pretty,
            metadata={
                'source': 'materials_project',
                'material_id': doc.material_id,
                'energy_above_hull': doc.energy_above_hull,
                'formation_energy': doc.formation_energy_per_atom,
                'band_gap': getattr(doc, 'band_gap', None),
                'density': getattr(doc, 'density', None),
                'crystal_system': getattr(doc.symmetry, 'crystal_system', None),
                'spacegroup': getattr(doc.symmetry, 'number', None),
            }
        )

        # Set properties if available (map MP elastic fields to standard keys)
        if getattr(doc, 'bulk_modulus', None) is not None:
            cs.properties["bulk_modulus"] = doc.bulk_modulus
        if getattr(doc, 'shear_modulus', None) is not None:
            cs.properties["shear_modulus"] = doc.shear_modulus

        cs.energy_above_hull = doc.energy_above_hull
        cs.decomposition_energy = doc.energy_above_hull  # Use as proxy

        return cs

    def _diverse_selection(
        self,
        candidates: List[CrystalStructure],
        max_count: int
    ) -> List[CrystalStructure]:
        """

        Strategy:
        1. Diversify by crystal system
        2. Diversify by number of elements
        3. Diversify by composition
        4. Sort by stability (energy_above_hull)
        """
        if len(candidates) <= max_count:
            return candidates

        logger.info(f"Selecting {max_count} diverse structures from {len(candidates)} candidates")

        # Strategy: Stratified sampling by crystal system and num_elements
        selected = []

        # Group by (crystal_system, num_elements)
        groups = {}
        for cs in candidates:
            crystal_sys = cs.metadata.get('crystal_system', 'unknown')
            num_elem = len(set(cs.species))
            key = (crystal_sys, num_elem)

            if key not in groups:
                groups[key] = []
            groups[key].append(cs)

        logger.info(f"Found {len(groups)} diversity groups")

        # Sort each group by stability
        for key, group in groups.items():
            groups[key] = sorted(group, key=lambda x: x.energy_above_hull or 999)

        # Round-robin selection from groups
        group_keys = list(groups.keys())
        idx = 0

        while len(selected) < max_count and any(groups.values()):
            key = group_keys[idx % len(group_keys)]

            if groups[key]:
                selected.append(groups[key].pop(0))

            idx += 1

            # Remove empty groups
            if idx % len(group_keys) == 0:
                groups = {k: v for k, v in groups.items() if v}
                group_keys = list(groups.keys())
                if not group_keys:
                    break

        logger.info(f"Selected {len(selected)} structures with diversity:")
        crystal_systems = Counter([s.metadata.get('crystal_system') for s in selected])
        num_elements = Counter([len(set(s.species)) for s in selected])
        logger.info(f"  Crystal systems: {dict(crystal_systems)}")
        logger.info(f"  Num elements: {dict(num_elements)}")

        return selected[:max_count]

    def _save_to_cache(
        self,
        cache_file: Path,
        structures: List[CrystalStructure],
        query_strategy: Dict
    ):
        """Save query results to cache"""
        try:
            cache_data = {
                'query_strategy': query_strategy,
                'num_structures': len(structures),
                'structures': []
            }

            for cs in structures:
                cache_data['structures'].append({
                    'material_id': cs.metadata.get('material_id'),
                    'formula': cs.formula,
                    'energy_above_hull': cs.energy_above_hull,
                    'crystal_system': cs.metadata.get('crystal_system'),
                    'metadata': cs.metadata
                })

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Saved {len(structures)} structures to cache: {cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_cache(self, cache_file: Path) -> List[CrystalStructure]:
        """Load structures from cache and fetch from MP"""
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            material_ids = [s['material_id'] for s in cache_data['structures']]

            logger.info(f"Loading {len(material_ids)} structures from MP by IDs...")

            with MPRester(self.mp_api_key) as mpr:
                structures = []

                for mpid in material_ids:
                    try:
                        doc = mpr.summary.get_data_by_id(mpid)
                        cs = self._mp_doc_to_crystal_structure(doc)
                        structures.append(cs)
                    except Exception as e:
                        logger.warning(f"Failed to load {mpid}: {e}")
                        continue

                return structures

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return []

    def _format_constraints(self, constraints: Dict) -> str:
        """Format constraints dict to readable string"""
        lines = []
        for prop, constraint in constraints.items():
            if isinstance(constraint, dict):
                if constraint.get('enabled', True):
                    min_val = constraint.get('min', '-∞')
                    max_val = constraint.get('max', '∞')
                    lines.append(f"  - {prop}: {min_val} to {max_val}")
            elif isinstance(constraint, (int, float)):
                lines.append(f"  - {prop}: {constraint}")

        return '\n'.join(lines) if lines else "  (No constraints specified)"

    def _get_default_strategy(self, task_config: Dict) -> Dict:
        """Fallback strategy when LLM analysis fails"""
        logger.warning("Using default query strategy as fallback")

        # Extract elements from constraints if available
        constraints = task_config.get('constraints', {})

        return {
            'elements': ['Ti', 'C', 'N', 'Si', 'B', 'Al', 'O'],
            'num_elements': [1, 3],
            'property_filters': {},
            'stability': {'energy_above_hull_max': 0.2},
            'crystal_system': None,
            'reasoning': 'Default conservative strategy (LLM analysis failed)'
        }
