"""ALIGNN (Atomistic Line Graph Neural Network) wrapper for property prediction

Uses pretrained models from JARVIS-DFT to predict various materials properties.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class ALIGNNPredictor:
    """
    Wrapper for ALIGNN pretrained models from JARVIS-DFT

    Supported properties:
    - formation_energy_peratom: Formation energy (eV/atom)
    - optb88vdw_bandgap: Band gap (eV)
    - bulk_modulus_kv: Bulk modulus (GPa)
    - shear_modulus_gv: Shear modulus (GPa)
    - mbj_bandgap: MBJ band gap (eV)
    - slme: Solar cell efficiency (%)
    - magmom_oszicar: Magnetic moment
    - spillage: Spillage
    - kpoint_length_unit: K-point length
    - encut: Plane wave cutoff
    - optb88vdw_total_energy: Total energy (eV)
    - epsx, epsy, epsz: Dielectric constants
    - mepsx, mepsy, mepsz: Ionic contributions
    - max_ir_mode, min_ir_mode: IR modes
    - n-Seebeck, n-powerfactor, p-Seebeck, p-powerfactor: Thermoelectric properties
    - avg_elec_mass, avg_hole_mass: Effective masses
    - exfoliation_energy: 2D exfoliation energy (meV/atom)
    """

    # Mapping of property names to ALIGNN model names
    PROPERTY_MODELS = {
        'formation_energy': 'jv_formation_energy_peratom_alignn',
        'energy_above_hull': 'jv_ehull_alignn',  # Energy above hull (≈ decomposition energy)
        'ehull': 'jv_ehull_alignn',  # Alias
        'decomposition_energy': 'jv_ehull_alignn',  # Use Ehull as Ed proxy
        'bandgap': 'jv_optb88vdw_bandgap_alignn',
        'band_gap': 'jv_optb88vdw_bandgap_alignn',  # alias for task constraints
        'bulk_modulus': 'jv_bulk_modulus_kv_alignn',
        'shear_modulus': 'jv_shear_modulus_gv_alignn',
        'mbj_bandgap': 'jv_mbj_bandgap_alignn',
        'total_energy': 'jv_optb88vdw_total_energy_alignn',
        'magmom': 'jv_magmom_oszicar_alignn',
        'exfoliation_energy': 'jv_exfoliation_energy_alignn',
        # Dielectric properties
        'eps_total_x': 'jv_epsx_alignn',
        'eps_total_y': 'jv_epsy_alignn',
        'eps_total_z': 'jv_epsz_alignn',
        # Thermoelectric
        'n_seebeck': 'jv_n-Seebeck_alignn',
        'p_seebeck': 'jv_p-Seebeck_alignn',
        'n_powerfactor': 'jv_n-powerfactor_alignn',
        'p_powerfactor': 'jv_p-powerfactor_alignn',
        # Piezoelectric and dielectric (DFPT max values, JARVIS-DFT)
        'piezoelectric_coefficient': 'jv_dfpt_piezo_max_dij_alignn',
        'dielectric_constant': 'jv_dfpt_piezo_max_dielectric_alignn',
    }

    def __init__(self, model_dir: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize ALIGNN predictor

        Args:
            model_dir: Directory containing ALIGNN pretrained models
                      Default: src/surrogate_models/alignn/pretrained_models/
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device

        # Set model directory
        if model_dir is None:
            # Default to project's surrogate_models directory
            project_root = Path(__file__).resolve().parents[2]
            model_dir = project_root / 'src' / 'surrogate_models' / 'alignn' / 'pretrained_models'

        self.model_dir = Path(model_dir)

        # Cache for loaded models
        self._models = {}

        # Check if ALIGNN is installed
        try:
            import alignn
            # Try to import pretrained module (this is what we actually use)
            try:
                from alignn.pretrained import get_prediction
                self.alignn_available = True
                logger.info(f"ALIGNN loaded successfully (device: {device})")
            except ImportError:
                # Fallback: ALIGNN is installed but pretrained module missing
                self.alignn_available = True
                logger.info(f"ALIGNN installed (device: {device})")
        except ImportError:
            self.alignn_available = False
            logger.warning(
                "ALIGNN not installed. Install with: "
                "pip install alignn"
            )

    def _load_model(self, property_name: str):
        """Load ALIGNN model for specific property"""
        if not self.alignn_available:
            raise ImportError("ALIGNN is not installed")

        # Check if model already loaded
        if property_name in self._models:
            return self._models[property_name]

        # Get model name
        model_name = self.PROPERTY_MODELS.get(property_name)
        if model_name is None:
            raise ValueError(f"Unknown property: {property_name}. Supported: {list(self.PROPERTY_MODELS.keys())}")

        # Load model
        try:
            from alignn.pretrained import get_prediction
            logger.info(f"Loading ALIGNN model for {property_name}...")

            # ALIGNN uses a different API - we'll use get_prediction directly
            # No need to explicitly load model, it's done in get_prediction
            self._models[property_name] = model_name
            logger.info(f"Model {model_name} ready")

            return model_name

        except Exception as e:
            logger.error(f"Failed to load ALIGNN model for {property_name}: {e}")
            raise

    def predict(self, structure: Structure, property_name: str) -> float:
        """
        Predict a single property for a structure

        Args:
            structure: Pymatgen Structure object
            property_name: Name of property to predict (e.g., 'formation_energy', 'bulk_modulus')

        Returns:
            Predicted value
        """
        if not self.alignn_available:
            raise ImportError("ALIGNN is not installed")

        # Load model
        model_name = self._load_model(property_name)

        try:
            from alignn.pretrained import get_prediction

            # ALIGNN expects either a CIF file path or atoms object
            # We'll convert structure to atoms
            from jarvis.core.atoms import Atoms as JarvisAtoms
            from jarvis.core.specie import chem_data, get_node_attributes

            # Convert Pymatgen Structure to JARVIS Atoms
            lattice_mat = structure.lattice.matrix.tolist()
            coords = structure.frac_coords.tolist()
            elements = [str(site.specie) for site in structure.sites]

            jarvis_atoms = JarvisAtoms(
                lattice_mat=lattice_mat,
                coords=coords,
                elements=elements,
                cartesian=False
            )

            # Get prediction
            result = get_prediction(
                atoms=jarvis_atoms,
                model_name=model_name
            )

            # Result is typically a dict with 'prediction' key
            if isinstance(result, dict):
                prediction = result.get('prediction', result.get('predict', None))
            else:
                prediction = result

            if prediction is None:
                raise ValueError(f"Could not extract prediction from result: {result}")

            # Handle array outputs (take mean if needed)
            if isinstance(prediction, (list, np.ndarray)):
                prediction = float(np.mean(prediction))
            else:
                prediction = float(prediction)

            logger.debug(f"Predicted {property_name} = {prediction:.6f}")
            return prediction

        except Exception as e:
            logger.error(f"ALIGNN prediction failed for {property_name}: {e}")
            raise

    def predict_multiple(self, structure: Structure, property_names: List[str]) -> Dict[str, float]:
        """
        Predict multiple properties for a structure

        Args:
            structure: Pymatgen Structure object
            property_names: List of property names to predict

        Returns:
            Dictionary mapping property names to predicted values
        """
        results = {}
        for prop in property_names:
            try:
                results[prop] = self.predict(structure, prop)
            except Exception as e:
                logger.warning(f"Failed to predict {prop}: {e}")
                results[prop] = None

        return results

    def get_available_properties(self) -> List[str]:
        """Get list of available property names"""
        return list(self.PROPERTY_MODELS.keys())

    @staticmethod
    def download_models():
        """
        Download pretrained ALIGNN models from JARVIS

        Note: Models are automatically downloaded on first use by ALIGNN.
        This method is provided for convenience to pre-download all models.
        """
        logger.info("ALIGNN models are downloaded automatically on first use")
        logger.info("No manual download required")

        # The first time get_prediction is called for each model,
        # ALIGNN will automatically download it to ~/.cache/alignn/
        return True


class CGCNNPredictor:
    """
    Wrapper for CGCNN (Crystal Graph Convolutional Neural Network)

    Note: CGCNN requires custom trained models.
    This is a placeholder for future implementation.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """Initialize CGCNN predictor"""
        self.device = device
        self.model_path = model_path

        try:
            import cgcnn
            self.cgcnn_available = True
            logger.info("CGCNN loaded successfully")
        except ImportError:
            self.cgcnn_available = False
            logger.warning(
                "CGCNN not installed. Install from: "
                "https://github.com/txie-93/cgcnn"
            )

    def predict(self, structure: Structure, property_name: str) -> float:
        """Predict property using CGCNN"""
        if not self.cgcnn_available:
            raise ImportError("CGCNN is not installed")

        # TODO: Implement CGCNN prediction
        raise NotImplementedError("CGCNN prediction not yet implemented")
