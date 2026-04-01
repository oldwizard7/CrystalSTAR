#!/usr/bin/env python3
"""Random example selector for few-shot learning"""
from pathlib import Path
import json
import random
from typing import List, Dict, Optional


class ExampleSelector:
    """reference examplesfew-shot learning"""

    def __init__(self, examples_dir: Path = None):
        """

        Args:
            examples_dir: examples，index.jsonVASP
                          data/reference_examples/novel_stable/
        """
        if examples_dir is None:
            examples_dir = Path(__file__).parent.parent.parent / "data" / "reference_examples" / "novel_stable"

        self.examples_dir = Path(examples_dir)
        self.examples = self._load_index()

    def _load_index(self) -> List[Dict]:
        """examples"""
        index_file = self.examples_dir / "index.json"

        if not index_file.exists():
            print(f":  {index_file}")
            print(" scripts/download_novel_structures.py examples")
            return []

        try:
            with open(index_file) as f:
                data = json.load(f)
                examples = data.get('examples', [])
                print(f" {len(examples)} reference examples")
                return examples
        except Exception as e:
            print(f": : {e}")
            return []

    def random_sample(self, k: int = 3) -> List[Dict]:
        """
        kexamples

        Args:
            k: ，3

        Returns:
            List of example dicts
        """
        if not self.examples:
            print(": examples")
            return []

        k = min(k, len(self.examples))

        return random.sample(self.examples, k)

    def get_poscar_content(self, example: Dict) -> Optional[str]:
        """
        examplePOSCAR

        Args:
            example: example dict，'file'

        Returns:
            POSCAR，None
        """
        filepath = self.examples_dir / example['file']

        if not filepath.exists():
            print(f":  {filepath}")
            return None

        try:
            return filepath.read_text()
        except Exception as e:
            print(f":  {filepath}: {e}")
            return None

    def format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """
        examplesprompt

        Args:
            examples: example dicts

        Returns:
            ，prompt
        """
        if not examples:
            return ""

        formatted_parts = []

        for i, example in enumerate(examples, 1):
            poscar_content = self.get_poscar_content(example)

            if not poscar_content:
                continue

            part = f"""Example {i}: {example['formula']}
Material ID: {example['mp_id']}
Crystal System: {example.get('crystal_system', 'unknown')}
Space Group: {example.get('space_group', 'unknown')}
Atoms: {example.get('nsites', 'unknown')}
Energy above hull: {example.get('energy_above_hull', 'unknown')} eV/atom

POSCAR:
{poscar_content}
"""
            formatted_parts.append(part)

        if not formatted_parts:
            return ""

        header = f"""Here are {len(formatted_parts)} reference examples of stable crystal structures:

"""

        return header + "\n---\n\n".join(formatted_parts)


if __name__ == "__main__":
    selector = ExampleSelector()

    print(f"\nexamples: {len(selector.examples)}")

    if selector.examples:
        print("\n3examples:")
        samples = selector.random_sample(k=3)

        for i, ex in enumerate(samples, 1):
            print(f"\n{i}. {ex['formula']} ({ex['mp_id']})")
            print(f"   Elements: {ex['nelements']}")
            print(f"   Atoms: {ex['nsites']}")
            print(f"   Ed: {ex['energy_above_hull']:.4f} eV/atom")

        print("\n" + "="*60)
        print("prompt:")
        print("="*60)
        formatted = selector.format_examples_for_prompt(samples)
        print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
