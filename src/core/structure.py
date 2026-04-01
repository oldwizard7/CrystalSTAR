"""Crystal structure data class and utilities"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.io.vasp import Poscar
import json

@dataclass
class CrystalStructure:
    """
    Crystal structure representation
    
    Attributes:
        formula: Chemical formula (e.g., "Na4Cl4")
        lattice: 3x3 lattice matrix (Å)
        positions: Nx3 fractional atomic positions
        species: List of element symbols
        energy: Total energy (eV/atom)
        decomposition_energy: Energy above convex hull (eV/atom)
        properties: Additional properties (bulk modulus, etc.)
        metadata: Extra information
    """
    formula: str
    lattice: np.ndarray  # 3x3
    positions: np.ndarray  # Nx3
    species: List[str]
    
    # Computed properties
    energy: Optional[float] = None
    decomposition_energy: Optional[float] = None
    properties: Dict[str, float] = field(default_factory=dict)
    
    # Validation flags
    is_valid: bool = False
    is_relaxed: bool = False
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate inputs"""
        if self.lattice.shape != (3, 3):
            raise ValueError(f"Lattice must be 3x3, got {self.lattice.shape}")
        if len(self.positions) != len(self.species):
            raise ValueError("Number of positions must match number of species")
    
    @property
    def num_atoms(self) -> int:
        """Number of atoms in the structure"""
        return len(self.species)
    
    @property
    def volume(self) -> float:
        """Unit cell volume (Å³)"""
        return abs(np.linalg.det(self.lattice))
    
    @property
    def density(self) -> float:
        """Density (g/cm³)"""
        from pymatgen.core import Element
        # atomic_mass is in amu; convert to g using amu_to_g and Å³ to cm³.
        mass_amu = sum(Element(s).atomic_mass for s in self.species)
        if self.volume <= 0:
            return 0.0
        amu_to_g = 1.66053906660e-24
        volume_cm3 = self.volume * 1e-24
        return (mass_amu * amu_to_g) / volume_cm3
    
    def to_poscar(self, significant_figures: int = 12) -> str:
        """
        Convert to POSCAR format string
        
        Args:
            significant_figures: Decimal precision
            
        Returns:
            POSCAR format string
        """
        lines = []
        
        # Comment line
        lines.append(self.formula)
        
        # Scaling factor
        lines.append("1.0")
        
        # Lattice vectors
        for vec in self.lattice:
            formatted = " ".join(f"{v:.{significant_figures}f}" for v in vec)
            lines.append(formatted)
        
        # Species and counts
        from collections import Counter
        species_counts = Counter(self.species)
        unique_species = sorted(species_counts.keys())
        
        lines.append(" ".join(unique_species))
        lines.append(" ".join(str(species_counts[s]) for s in unique_species))
        
        # Coordinate type
        lines.append("direct")
        
        # Atomic positions
        # Sort by species for POSCAR format
        sorted_indices = sorted(range(len(self.species)), 
                               key=lambda i: unique_species.index(self.species[i]))
        
        for idx in sorted_indices:
            pos = self.positions[idx]
            species = self.species[idx]
            formatted_pos = " ".join(f"{p:.{significant_figures}f}" for p in pos)
            lines.append(f"{formatted_pos} {species}")
        
        return "\n".join(lines)
    
    @classmethod
    def from_poscar(cls, poscar_str: str) -> 'CrystalStructure':
        """
        Parse POSCAR format string with robust error checking

        Args:
            poscar_str: POSCAR format string

        Returns:
            CrystalStructure instance

        Raises:
            ValueError: If POSCAR format is invalid or incomplete
        """
        lines = [line.strip() for line in poscar_str.strip().split('\n') if line.strip()]

        # Validate minimum number of lines
        if len(lines) < 9:
            raise ValueError(
                f"POSCAR format incomplete: expected at least 9 lines, got {len(lines)}"
            )

        try:
            # Parse comment (formula)
            formula = lines[0]

            # Parse scaling factor
            scale = float(lines[1])

            # Parse lattice vectors (lines 2-4)
            if len(lines) < 5:
                raise ValueError("Missing lattice vectors")

            lattice = np.array([
                [float(x) for x in lines[2].split()],
                [float(x) for x in lines[3].split()],
                [float(x) for x in lines[4].split()]
            ]) * scale

            # Parse species (line 5)
            if len(lines) < 6:
                raise ValueError("Missing species line")
            species_line = lines[5].split()

            # Parse counts (line 6)
            if len(lines) < 7:
                raise ValueError("Missing atom counts line")
            counts_line = [int(x) for x in lines[6].split()]

            # Validate species and counts match
            if len(species_line) != len(counts_line):
                raise ValueError(
                    f"Species ({len(species_line)}) and counts ({len(counts_line)}) mismatch"
                )

            # Parse coordinate type (line 7)
            if len(lines) < 8:
                raise ValueError("Missing coordinate type line")
            coord_type = lines[7].lower()
            is_direct = coord_type.startswith('d')

            # Calculate expected number of position lines
            total_atoms = sum(counts_line)
            expected_lines = 8 + total_atoms

            if len(lines) < expected_lines:
                raise ValueError(
                    f"Incomplete POSCAR: expected {expected_lines} lines for {total_atoms} atoms, "
                    f"got {len(lines)}"
                )

            # Parse positions
            positions = []
            species = []

            line_idx = 8
            for spec, count in zip(species_line, counts_line):
                for _ in range(count):
                    if line_idx >= len(lines):
                        raise ValueError(
                            f"Ran out of position lines at atom {len(positions)+1}/{total_atoms}"
                        )

                    pos_line = lines[line_idx].split()
                    if len(pos_line) < 3:
                        raise ValueError(
                            f"Invalid position line {line_idx}: expected 3 coordinates, "
                            f"got {len(pos_line)}"
                        )

                    pos = [float(x) for x in pos_line[:3]]
                    positions.append(pos)
                    species.append(spec)
                    line_idx += 1

            positions = np.array(positions)

            # Derive reduced formula from parsed species and replace generic placeholders.
            placeholder_formulas = {
                "", "system", "poscar", "structure", "generated", "unknown"
            }
            try:
                from collections import Counter
                derived_formula = Composition(Counter(species)).reduced_formula
            except Exception:
                derived_formula = None

            if formula.strip().lower() in placeholder_formulas and derived_formula:
                formula = derived_formula

            # Convert to fractional if cartesian
            if not is_direct:
                positions = positions @ np.linalg.inv(lattice)

            return cls(
                formula=formula,
                lattice=lattice,
                positions=positions,
                species=species
            )

        except (IndexError, ValueError) as e:
            # Provide helpful error message
            raise ValueError(
                f"Failed to parse POSCAR format: {str(e)}\n"
                f"POSCAR had {len(lines)} lines:\n"
                f"  Line 1 (comment): {lines[0] if len(lines) > 0 else 'MISSING'}\n"
                f"  Line 2 (scale): {lines[1] if len(lines) > 1 else 'MISSING'}\n"
                f"  Lines 3-5 (lattice): {'OK' if len(lines) >= 5 else 'MISSING'}\n"
                f"  Line 6 (species): {lines[5] if len(lines) > 5 else 'MISSING'}\n"
                f"  Line 7 (counts): {lines[6] if len(lines) > 6 else 'MISSING'}\n"
                f"  Line 8 (coord type): {lines[7] if len(lines) > 7 else 'MISSING'}"
            )
    
    @classmethod
    def from_pymatgen(cls, structure: Structure) -> 'CrystalStructure':
        """Create from pymatgen Structure"""
        return cls(
            formula=structure.composition.reduced_formula,
            lattice=structure.lattice.matrix,
            positions=structure.frac_coords,
            species=[str(site.specie) for site in structure]
        )
    
    def to_pymatgen(self) -> Structure:
        """Convert to pymatgen Structure"""
        return Structure(
            Lattice(self.lattice),
            self.species,
            self.positions
        )
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary with safe Python-native types"""
        import numpy as np

        def _to_float(x):
            """Convert numpy/tensor floats to Python float"""
            if x is None:
                return None
            if isinstance(x, (np.floating, float)):
                return float(x)
            if hasattr(x, "item"):  # torch or numpy scalar
                try:
                    return float(x.item())
                except Exception:
                    return x
            return x

        def _to_list(arr):
            """Convert numpy arrays to Python lists"""
            if arr is None:
                return None
            if hasattr(arr, "tolist"):
                return arr.tolist()
            # fallback for iterables
            try:
                return list(arr)
            except Exception:
                return arr

        # Clean up properties dict
        clean_props = {k: _to_float(v) for k, v in self.properties.items()}

        return {
            'formula': self.formula,
            'lattice': _to_list(self.lattice),
            'positions': _to_list(self.positions),
            'species': list(self.species),
            'energy': _to_float(self.energy),
            'decomposition_energy': _to_float(self.decomposition_energy),
            'properties': clean_props,
            'is_valid': bool(self.is_valid),
            'is_relaxed': bool(self.is_relaxed),
            'metadata': self.metadata
        }

    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CrystalStructure':
        """Deserialize from dictionary"""
        return cls(
            formula=data['formula'],
            lattice=np.array(data['lattice']),
            positions=np.array(data['positions']),
            species=data['species'],
            energy=data.get('energy'),
            decomposition_energy=data.get('decomposition_energy'),
            properties=data.get('properties', {}),
            is_valid=data.get('is_valid', False),
            is_relaxed=data.get('is_relaxed', False),
            metadata=data.get('metadata', {})
        )
    
    def save(self, filepath: str):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CrystalStructure':
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        status = "valid" if self.is_valid else "invalid"
        relaxed = "relaxed" if self.is_relaxed else "unrelaxed"
        ed_str = f"Ed={self.decomposition_energy:.3f}" if self.decomposition_energy is not None else "Ed=N/A"
        return f"CrystalStructure({self.formula}, {status}, {relaxed}, {ed_str})"
