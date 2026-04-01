"""Data management utilities for MatLLMSearch"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import logging

from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from tqdm import tqdm

from src.core.structure import CrystalStructure

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manage crystal structure data
    
    Features:
    - Download from Materials Project
    - Cache management
    - Data filtering and preprocessing
    """
    
    def __init__(self, data_dir: str = "data/"):
        """
        Initialize data manager
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.initial_structures_dir = self.data_dir / "initial_structures"
        self.results_dir = self.data_dir / "results"
        self.cache_dir = self.data_dir / "cache"
        self.examples_dir = self.data_dir / "examples"
        
        # Create directories
        self.initial_structures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_materials_project(
        self,
        api_key: str,
        max_structures: int = 1000,
        filters: Optional[Dict] = None
    ) -> List[CrystalStructure]:
        """
        Download structures from Materials Project
        
        Args:
            api_key: Materials Project API key
            max_structures: Maximum number of structures to download
            filters: Query filters (e.g., elements, num_elements)
            
        Returns:
            List of downloaded structures
        """
        logger.info(f"Downloading structures from Materials Project (max: {max_structures})")
        
        with MPRester(api_key) as mpr:
            # Default query: stable structures with 3-6 elements
            query = {
                "nelements": {"$gte": 3, "$lte": 6},
                "e_above_hull": {"$lte": 0.05},  # Stable or near-stable
                "is_stable": True
            }
            
            # Apply custom filters
            if filters:
                if 'elements' in filters:
                    query["elements"] = {"$all": filters['elements']}
                if 'exclude_elements' in filters:
                    query["elements"] = {"$nin": filters['exclude_elements']}
            
            # Query structures
            docs = mpr.summary.search(
                **query,
                fields=["material_id", "structure", "formation_energy_per_atom", 
                       "energy_above_hull", "band_gap"],
                num_chunks=10
            )
            
            structures = []
            
            for i, doc in enumerate(tqdm(docs[:max_structures], desc="Processing")):
                try:
                    # Convert to CrystalStructure
                    pmg_struct = doc.structure
                    struct = CrystalStructure.from_pymatgen(pmg_struct)
                    
                    # Add metadata
                    struct.metadata['material_id'] = doc.material_id
                    struct.metadata['formation_energy'] = doc.formation_energy_per_atom
                    struct.metadata['e_above_hull'] = doc.energy_above_hull
                    struct.metadata['band_gap'] = doc.band_gap
                    struct.decomposition_energy = doc.energy_above_hull
                    struct.is_valid = True
                    
                    # Save to disk
                    filename = f"mp-{doc.material_id}.json"
                    struct.save(str(self.initial_structures_dir / filename))
                    
                    structures.append(struct)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {doc.material_id}: {e}")
            
            logger.info(f"Downloaded and saved {len(structures)} structures")
            
            # Save summary
            self._save_download_summary(structures)
            
            return structures
    
    def _save_download_summary(self, structures: List[CrystalStructure]):
        """Save summary of downloaded structures"""
        summary = {
            'total_structures': len(structures),
            'formulas': [s.formula for s in structures],
            'avg_num_atoms': sum(s.num_atoms for s in structures) / len(structures),
            'unique_compositions': len(set(s.formula for s in structures))
        }
        
        summary_path = self.initial_structures_dir / "download_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")
    
    def create_example_dataset(self):
        """
        Create example dataset for testing
        
        Generates simple cubic structures
        """
        logger.info("Creating example dataset...")
        
        example_dir = self.examples_dir / "simple_structures"
        example_dir.mkdir(parents=True, exist_ok=True)
        
        # Example 1: NaCl (rock salt)
        nacl = self._create_nacl_structure()
        nacl.save(str(example_dir / "NaCl.json"))
        
        # Example 2: CsCl (cesium chloride)
        cscl = self._create_cscl_structure()
        cscl.save(str(example_dir / "CsCl.json"))
        
        # Example 3: Perovskite (CaTiO3)
        perovskite = self._create_perovskite_structure()
        perovskite.save(str(example_dir / "CaTiO3.json"))
        
        logger.info(f"Created 3 example structures in {example_dir}")
        
        # Create README
        self._create_example_readme(example_dir)
    
    def _create_nacl_structure(self) -> CrystalStructure:
        """Create NaCl structure"""
        import numpy as np
        
        a = 5.64  # Lattice parameter (Angstrom)
        lattice = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
        
        # FCC lattice with Na and Cl
        positions = np.array([
            [0.0, 0.0, 0.0],  # Na
            [0.5, 0.5, 0.0],  # Na
            [0.5, 0.0, 0.5],  # Na
            [0.0, 0.5, 0.5],  # Na
            [0.5, 0.5, 0.5],  # Cl
            [0.0, 0.0, 0.5],  # Cl
            [0.0, 0.5, 0.0],  # Cl
            [0.5, 0.0, 0.0]   # Cl
        ])
        
        species = ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl']
        
        return CrystalStructure(
            formula="Na4Cl4",
            lattice=lattice,
            positions=positions,
            species=species
        )
    
    def _create_cscl_structure(self) -> CrystalStructure:
        """Create CsCl structure"""
        import numpy as np
        
        a = 4.12
        lattice = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
        
        positions = np.array([
            [0.0, 0.0, 0.0],  # Cs
            [0.5, 0.5, 0.5]   # Cl
        ])
        
        species = ['Cs', 'Cl']
        
        return CrystalStructure(
            formula="CsCl",
            lattice=lattice,
            positions=positions,
            species=species
        )
    
    def _create_perovskite_structure(self) -> CrystalStructure:
        """Create perovskite CaTiO3 structure"""
        import numpy as np
        
        a = 3.84
        lattice = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
        
        positions = np.array([
            [0.0, 0.0, 0.0],  # Ca
            [0.5, 0.5, 0.5],  # Ti
            [0.5, 0.5, 0.0],  # O
            [0.5, 0.0, 0.5],  # O
            [0.0, 0.5, 0.5]   # O
        ])
        
        species = ['Ca', 'Ti', 'O', 'O', 'O']
        
        return CrystalStructure(
            formula="CaTiO3",
            lattice=lattice,
            positions=positions,
            species=species
        )
    
    def _create_example_readme(self, example_dir: Path):
        """Create README for examples"""
        readme_content = """# Example Structures

This directory contains simple example structures for testing MatLLMSearch.

## Structures

1. **NaCl.json** - Sodium Chloride (Rock Salt)
   - Space group: Fm-3m (225)
   - Lattice: FCC
   - Common ionic compound

2. **CsCl.json** - Cesium Chloride
   - Space group: Pm-3m (221)
   - Lattice: Simple cubic
   - Alternative ionic structure

3. **CaTiO3.json** - Calcium Titanate (Perovskite)
   - Space group: Pm-3m (221)
   - Lattice: Cubic perovskite
   - Oxide structure

## Usage
```python
from src.utils.data_manager import DataManager

dm = DataManager()
structures = dm.load_structures('data/examples/simple_structures/')
```
"""
        
        with open(example_dir / "README.md", 'w') as f:
            f.write(readme_content)
    
    def load_structures(self, directory: str) -> List[CrystalStructure]:
        """
        Load all structures from a directory
        
        Args:
            directory: Path to structure directory
            
        Returns:
            List of CrystalStructure objects
        """
        structures = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return structures
        
        # Load JSON files
        json_loaded_basenames = set()
        for file_path in dir_path.glob("*.json"):
            if file_path.name in {"download_summary.json", "index.json"}:
                continue
            
            try:
                struct = CrystalStructure.load(str(file_path))
                struct.metadata.setdefault("source_file", file_path.name)
                structures.append(struct)
                json_loaded_basenames.add(file_path.stem)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        # Load VASP files
        for file_path in dir_path.glob("*.vasp"):
            if file_path.stem in json_loaded_basenames:
                # Prefer json version (may contain metadata/energies)
                continue
            try:
                with open(file_path, 'r') as f:
                    poscar_str = f.read()
                struct = CrystalStructure.from_poscar(poscar_str)
                struct.metadata["source_file"] = file_path.name
                structures.append(struct)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(structures)} structures from {directory}")
        
        return structures
    
    def filter_structures(
        self,
        structures: List[CrystalStructure],
        filters: Dict
    ) -> List[CrystalStructure]:
        """
        Filter structures by criteria
        
        Args:
            structures: List of structures to filter
            filters: Filter criteria
            
        Returns:
            Filtered structures
        """
        filtered = structures.copy()
        
        # Filter by number of elements
        if 'num_elements' in filters:
            min_elem, max_elem = filters['num_elements']
            filtered = [s for s in filtered 
                       if min_elem <= len(set(s.species)) <= max_elem]
        
        # Filter by specific elements
        if 'include_elements' in filters:
            required = set(filters['include_elements'])
            filtered = [s for s in filtered 
                       if required.issubset(set(s.species))]
        
        # Exclude elements
        if 'exclude_elements' in filters:
            excluded = set(filters['exclude_elements'])
            filtered = [s for s in filtered 
                       if not any(elem in excluded for elem in s.species)]
        
        # Filter by stability
        if 'max_ed' in filters:
            max_ed = filters['max_ed']
            filtered = [s for s in filtered 
                       if s.decomposition_energy is not None 
                       and s.decomposition_energy <= max_ed]
        
        logger.info(f"Filtered {len(structures)} → {len(filtered)} structures")
        
        return filtered
    
    def cache_phase_diagram(self, api_key: str):
        """Cache Materials Project phase diagram for Ed calculation"""
        logger.info("Caching phase diagram data...")
        
        with MPRester(api_key) as mpr:
            # Download all stable entries
            docs = mpr.summary.search(
                is_stable=True,
                fields=["material_id", "composition", "energy_per_atom"]
            )
            
            phase_data = {
                'entries': [
                    {
                        'material_id': doc.material_id,
                        'composition': str(doc.composition),
                        'energy': doc.energy_per_atom
                    }
                    for doc in docs
                ]
            }
            
            cache_path = self.cache_dir / "phase_diagram.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(phase_data, f)
            
            logger.info(f"Cached {len(phase_data['entries'])} phase diagram entries")
