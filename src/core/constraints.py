"""
Property Constraints Checker
Validates if structures satisfy all defined property constraints
"""
from typing import Dict, Any
from src.core.structure import CrystalStructure
import logging

logger = logging.getLogger(__name__)


class ConstraintChecker:
    """
    Check if crystal structures satisfy property constraints

    Supports dynamic constraint types:
        - Boolean constraints (is_valid)
        - Min/max constraints (decomposition_energy, band_gap, etc.)
        - Range constraints (min and max)
    """

    def __init__(self, constraints_config: Dict[str, Any]):
        """
        Initialize ConstraintChecker

        Args:
            constraints_config: Dict of constraint definitions
                Example:
                {
                    'is_valid': {'value': True, 'description': '...'},
                    'decomposition_energy': {'max': 0.1, 'description': '...'},
                    'band_gap': {'min': 2.5, 'max': 5.0, 'description': '...'}
                }
        """
        self.constraints = constraints_config
        logger.info(f"ConstraintChecker initialized with {len(constraints_config)} constraint(s)")

    def check(self, structure: CrystalStructure) -> bool:
        """
        Check if structure satisfies ALL constraints

        Args:
            structure: Crystal structure to validate

        Returns:
            True if all constraints satisfied, False otherwise
        """
        for constraint_name, constraint_def in self.constraints.items():
            # Skip disabled constraints
            if isinstance(constraint_def, dict) and not constraint_def.get('enabled', True):
                logger.debug(f"Skipping disabled constraint '{constraint_name}'")
                continue

            # Special handling for boolean constraints
            if constraint_name == 'is_valid':
                required = constraint_def.get('value', True)
                if structure.is_valid != required:
                    logger.debug(
                        f"Structure failed constraint '{constraint_name}': "
                        f"expected {required}, got {structure.is_valid}"
                    )
                    return False
                continue

            # Element containment constraint (e.g. must contain Li/Na/K)
            if constraint_name == 'contains_mobile_species':
                required = constraint_def.get('elements', [])
                if required:
                    structure_elements = set(structure.species)
                    if not structure_elements & set(required):
                        logger.debug(
                            f"Structure failed constraint '{constraint_name}': "
                            f"none of {required} found in {structure_elements}"
                        )
                        return False
                continue

            # Check if structure has this property
            # First try as direct attribute, then fall back to properties dict
            if hasattr(structure, constraint_name):
                value = getattr(structure, constraint_name)
            elif constraint_name in structure.properties:
                value = structure.properties[constraint_name]
            else:
                # Property not computed yet - consider as failure
                logger.debug(
                    f"Structure missing property '{constraint_name}' "
                    f"required for constraint check (checked both attribute and properties dict)"
                )
                return False

            # Check min constraint
            if 'min' in constraint_def:
                min_val = constraint_def['min']
                if value < min_val:
                    logger.debug(
                        f"Structure failed constraint '{constraint_name}': "
                        f"{value} < {min_val} (min)"
                    )
                    return False

            # Check max constraint
            if 'max' in constraint_def:
                max_val = constraint_def['max']
                if value > max_val:
                    logger.debug(
                        f"Structure failed constraint '{constraint_name}': "
                        f"{value} > {max_val} (max)"
                    )
                    return False

            # Check exact value constraint (for non-boolean properties)
            if 'value' in constraint_def and constraint_name != 'is_valid':
                expected = constraint_def['value']
                if value != expected:
                    logger.debug(
                        f"Structure failed constraint '{constraint_name}': "
                        f"{value} != {expected} (expected)"
                    )
                    return False

        # All constraints satisfied
        return True

    def get_constraints_text(self) -> str:
        """
        Generate human-readable constraints text for LLM prompts

        Returns:
            Formatted string describing all constraints
        """
        lines = []

        for name, definition in self.constraints.items():
            desc = definition.get('description', name)

            # Range constraint (min and max)
            if 'min' in definition and 'max' in definition:
                lines.append(
                    f"- {desc}: {definition['min']} ≤ {name} ≤ {definition['max']}"
                )

            # Min only
            elif 'min' in definition:
                lines.append(f"- {desc}: {name} ≥ {definition['min']}")

            # Max only
            elif 'max' in definition:
                lines.append(f"- {desc}: {name} ≤ {definition['max']}")

            # Element containment
            elif name == 'contains_mobile_species':
                elements = definition.get('elements', [])
                lines.append(f"- Must contain at least one of: {', '.join(elements)}")

            # Boolean or description only
            elif 'value' in definition and name == 'is_valid':
                lines.append(f"- {desc}")

            # Other
            else:
                lines.append(f"- {desc}")

        return '\n'.join(lines)

    def get_failed_constraints(self, structure: CrystalStructure) -> list:
        """
        Get list of constraints that the structure fails

        Args:
            structure: Crystal structure to check

        Returns:
            List of failed constraint names
        """
        failed = []

        for constraint_name, constraint_def in self.constraints.items():
            # Skip disabled constraints
            if isinstance(constraint_def, dict) and not constraint_def.get('enabled', True):
                continue

            # Check is_valid
            if constraint_name == 'is_valid':
                required = constraint_def.get('value', True)
                if structure.is_valid != required:
                    failed.append(constraint_name)
                continue

            # Element containment constraint
            if constraint_name == 'contains_mobile_species':
                required = constraint_def.get('elements', [])
                if required:
                    structure_elements = set(structure.species)
                    if not structure_elements & set(required):
                        failed.append(f"{constraint_name} (missing {required})")
                continue

            # Check if property exists
            if not hasattr(structure, constraint_name):
                failed.append(f"{constraint_name} (missing)")
                continue

            value = getattr(structure, constraint_name)

            # Check constraints
            if 'min' in constraint_def and value < constraint_def['min']:
                failed.append(f"{constraint_name} (< {constraint_def['min']})")
            if 'max' in constraint_def and value > constraint_def['max']:
                failed.append(f"{constraint_name} (> {constraint_def['max']})")
            if 'value' in constraint_def and constraint_name != 'is_valid':
                if value != constraint_def['value']:
                    failed.append(f"{constraint_name} (!= {constraint_def['value']})")

        return failed

    def count_satisfied(self, structures: list) -> int:
        """
        Count how many structures satisfy all constraints

        Args:
            structures: List of CrystalStructure objects

        Returns:
            Number of structures that satisfy all constraints
        """
        return sum(1 for s in structures if self.check(s))

    def filter_satisfied(self, structures: list) -> list:
        """
        Filter structures to only those satisfying all constraints

        Args:
            structures: List of CrystalStructure objects

        Returns:
            List of structures that satisfy all constraints
        """
        return [s for s in structures if self.check(s)]

    def __repr__(self) -> str:
        """String representation"""
        constraint_names = ', '.join(self.constraints.keys())
        return f"ConstraintChecker({len(self.constraints)} constraints: {constraint_names})"
