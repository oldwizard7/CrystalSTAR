#!/usr/bin/env python3
"""
Initial Parent Pool Initializer

Supports multiple initialization strategies:
- random: Random sampling (original behavior)
- quality: Top-K by stability (lowest Ed)
- diversity: Sample across crystal systems/element counts
- task_oriented: Filter by task-relevant elements/properties
- smart: Hybrid (task_oriented + diversity)
"""

import logging
import random
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


class ParentPoolInitializer:
    """Initialize parent pool with different strategies"""

    def __init__(self, config: Dict):
        """
        Initialize with configuration

        Args:
            config: Configuration dict from config.yaml['data']['initial_parent_pool']
        """
        self.config = config
        self.strategy = config.get('strategy', 'random')
        self.seed = config.get('seed', None)
        self.max_count = config.get('max_count', 20)

        # Set random seed if provided
        if self.seed is not None:
            random.seed(self.seed)
            logger.info(f"Set random seed for parent pool initialization: {self.seed}")

    def select_from_index(self, index_path: Path) -> List[Dict]:
        """
        Select structures from index.json using configured strategy

        Args:
            index_path: Path to index.json

        Returns:
            List of selected example dicts (not CrystalStructure objects yet)
        """
        # Load index
        if not index_path.exists():
            logger.warning(f"Index file not found: {index_path}")
            return []

        try:
            with open(index_path) as f:
                data = json.load(f)
            examples = data.get('examples', [])
            logger.info(f"Loaded {len(examples)} structures from index")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return []

        if not examples:
            return []

        # Apply strategy
        if self.strategy == 'random':
            selected = self._select_random(examples)
        elif self.strategy == 'quality':
            selected = self._select_quality(examples)
        elif self.strategy == 'diversity':
            selected = self._select_diversity(examples)
        elif self.strategy == 'task_oriented':
            selected = self._select_task_oriented(examples)
        elif self.strategy == 'smart':
            selected = self._select_smart(examples)
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', falling back to random")
            selected = self._select_random(examples)

        logger.info(f"Selected {len(selected)} structures using '{self.strategy}' strategy")
        return selected

    def _select_random(self, examples: List[Dict]) -> List[Dict]:
        """
        Random sampling (original behavior)

        Args:
            examples: All available examples

        Returns:
            Randomly sampled examples
        """
        n = min(self.max_count, len(examples))
        selected = random.sample(examples, n)
        logger.info(f"Random sampling: selected {n}/{len(examples)} structures (seed={self.seed})")
        return selected

    def _select_quality(self, examples: List[Dict]) -> List[Dict]:
        """
        Select top-K most stable structures (lowest energy_above_hull)

        Args:
            examples: All available examples

        Returns:
            Top-K most stable examples
        """
        # Sort by energy_above_hull
        sorted_examples = sorted(examples,
                                key=lambda x: x.get('energy_above_hull', float('inf')))

        selected = sorted_examples[:self.max_count]

        if selected:
            ed_range = (selected[0]['energy_above_hull'], selected[-1]['energy_above_hull'])
            logger.info(f"Quality selection: Ed range {ed_range[0]:.4f} - {ed_range[1]:.4f} eV/atom")

        return selected

    def _select_diversity(self, examples: List[Dict]) -> List[Dict]:
        """
        Sample across crystal systems and element counts for diversity

        Args:
            examples: All available examples

        Returns:
            Diverse set of examples
        """
        # Group by (crystal_system, nelements)
        groups = {}
        for ex in examples:
            key = (ex.get('crystal_system', 'unknown'), ex.get('nelements', 0))
            groups.setdefault(key, []).append(ex)

        logger.info(f"Found {len(groups)} diversity groups")

        # Sample from each group
        selected = []
        per_group = max(1, self.max_count // len(groups))

        for (cs, n_elem), group in groups.items():
            # Within each group, select lowest Ed
            group_sorted = sorted(group, key=lambda x: x.get('energy_above_hull', float('inf')))
            selected.extend(group_sorted[:per_group])

        # If not enough, add more from best remaining
        if len(selected) < self.max_count:
            remaining = [ex for ex in examples if ex not in selected]
            remaining_sorted = sorted(remaining, key=lambda x: x.get('energy_above_hull', float('inf')))
            selected.extend(remaining_sorted[:self.max_count - len(selected)])

        selected = selected[:self.max_count]

        # Log diversity stats
        systems = set(ex.get('crystal_system', 'unknown') for ex in selected)
        n_elements = set(ex.get('nelements', 0) for ex in selected)
        logger.info(f"Diversity selection: {len(systems)} crystal systems, {len(n_elements)} element counts")

        return selected

    def _select_task_oriented(self, examples: List[Dict]) -> List[Dict]:
        """
        Filter by task-relevant elements/properties

        Args:
            examples: All available examples

        Returns:
            Task-oriented filtered examples
        """
        task_cfg = self.config.get('task_oriented', {})

        if not task_cfg.get('enabled', True):
            logger.warning("task_oriented strategy selected but not enabled in config, using random")
            return self._select_random(examples)

        # Get filter criteria
        include_elements = set(task_cfg.get('include_elements', []))
        include_mode = task_cfg.get('include_mode', 'any')
        exclude_elements = set(task_cfg.get('exclude_elements', []))
        max_elements = task_cfg.get('max_elements', None)
        max_sites = task_cfg.get('max_sites', None)
        fill_general_pool = task_cfg.get('fill_general_pool', True)

        logger.info(f"Task-oriented filters:")
        if include_elements:
            logger.info(f"  Include elements ({include_mode}): {include_elements}")
        if exclude_elements:
            logger.info(f"  Exclude elements: {exclude_elements}")
        if max_elements:
            logger.info(f"  Max elements: {max_elements}")
        if max_sites:
            logger.info(f"  Max sites: {max_sites}")
        logger.info(f"  Fill general pool: {fill_general_pool}")

        # Filter candidates
        candidates = []
        for ex in examples:
            # Extract elements from formula
            formula = ex.get('formula', '')
            elements = self._extract_elements(formula)

            # Apply filters
            if include_elements:
                if include_mode == 'subset':
                    # All elements must be within include_elements
                    if not elements.issubset(include_elements):
                        continue
                else:
                    # Default: must have at least one included element
                    if not (elements & include_elements):
                        continue

            if exclude_elements and (elements & exclude_elements):
                continue  # Must not have any excluded element

            if max_elements and ex.get('nelements', 0) > max_elements:
                continue

            if max_sites and ex.get('nsites', 0) > max_sites:
                continue

            candidates.append(ex)

        logger.info(f"Task-oriented filter: {len(candidates)}/{len(examples)} candidates passed")

        # Sort by stability and take top-K
        candidates_sorted = sorted(candidates,
                                  key=lambda x: x.get('energy_above_hull', float('inf')))
        selected = candidates_sorted[:self.max_count]

        # If not enough candidates, fill with best remaining
        if len(selected) < self.max_count:
            if fill_general_pool:
                logger.warning(f"Only {len(selected)} task-oriented candidates, filling with general pool")
                remaining = [ex for ex in examples if ex not in selected]
                remaining_sorted = sorted(remaining, key=lambda x: x.get('energy_above_hull', float('inf')))
                selected.extend(remaining_sorted[:self.max_count - len(selected)])
            else:
                logger.warning(f"Only {len(selected)} task-oriented candidates, keeping strict filter")

        return selected[:self.max_count]

    def _select_smart(self, examples: List[Dict]) -> List[Dict]:
        """
        Smart hybrid: task_oriented (70%) + diversity (30%)

        Args:
            examples: All available examples

        Returns:
            Smart mixed selection
        """
        task_cfg = self.config.get('task_oriented', {})
        task_ratio = task_cfg.get('ratio', 0.7)
        diversity_ratio = self.config.get('diversity', {}).get('ratio', 0.3)

        n_task = int(self.max_count * task_ratio)
        n_diversity = self.max_count - n_task

        logger.info(f"Smart strategy: {n_task} task-oriented + {n_diversity} diversity")

        # Step 1: Task-oriented selection
        task_cfg_copy = self.config.copy()
        task_cfg_copy['max_count'] = n_task
        task_initializer = ParentPoolInitializer(task_cfg_copy)
        task_initializer.strategy = 'task_oriented'
        task_selected = task_initializer._select_task_oriented(examples)

        # Step 2: Diversity selection from remaining
        remaining = [ex for ex in examples if ex not in task_selected]
        if remaining:
            diversity_cfg_copy = self.config.copy()
            diversity_cfg_copy['max_count'] = n_diversity
            diversity_initializer = ParentPoolInitializer(diversity_cfg_copy)
            diversity_initializer.strategy = 'diversity'
            diversity_selected = diversity_initializer._select_diversity(remaining)
        else:
            diversity_selected = []
            logger.warning("No remaining structures for diversity selection")

        selected = task_selected + diversity_selected

        logger.info(f"Smart selection complete: {len(task_selected)} task + {len(diversity_selected)} diverse = {len(selected)} total")

        return selected

    def _extract_elements(self, formula: str) -> Set[str]:
        """
        Extract element symbols from chemical formula

        Args:
            formula: Chemical formula (e.g., "BaAl2O4")

        Returns:
            Set of element symbols (e.g., {"Ba", "Al", "O"})
        """
        # Match element symbols (capital letter optionally followed by lowercase)
        elements = set(re.findall(r'[A-Z][a-z]?', formula))
        return elements


# Convenience function for testing
if __name__ == "__main__":
    # Test with example config
    config = {
        'strategy': 'smart',
        'max_count': 10,
        'seed': 42,
        'task_oriented': {
            'enabled': True,
            'ratio': 0.7,
            'include_elements': ['Si', 'Ge', 'Ga', 'N', 'O', 'S', 'Al', 'Zn'],
            'exclude_elements': ['Fe', 'Co', 'Ni', 'Cu', 'Mn'],
            'max_elements': 4,
            'max_sites': 50
        },
        'diversity': {
            'ratio': 0.3
        }
    }

    initializer = ParentPoolInitializer(config)

    # Test with real index
    index_path = Path(__file__).parent.parent.parent / "data" / "reference_examples" / "novel_stable" / "index.json"

    if index_path.exists():
        selected = initializer.select_from_index(index_path)

        print(f"\nSelected {len(selected)} structures:")
        for i, ex in enumerate(selected[:5], 1):
            print(f"{i}. {ex['formula']} ({ex['mp_id']})")
            print(f"   Ed={ex['energy_above_hull']:.4f} eV/atom, {ex['nelements']} elements, {ex['crystal_system']}")
    else:
        print(f"Index not found: {index_path}")
