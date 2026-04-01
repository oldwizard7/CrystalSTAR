"""Structure evaluation using Machine Learning Interatomic Potentials - Multi-Backend Support"""

from typing import List, Optional, Dict
import os
import json
import shlex
import subprocess
import tempfile
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.entries.computed_entries import ComputedEntry
from mp_api.client import MPRester
import ase
from ase import constraints as ase_constraints
from ase.optimize import FIRE, BFGS, LBFGS
import logging

from .structure import CrystalStructure

logger = logging.getLogger(__name__)


class StructureEvaluator:
    """
    Evaluate crystal structures using Machine Learning Interatomic Potentials

    Supported backends:
    - CHGNet: Universal ML potential from Materials Project
    - M3GNet: Graph network potential
    - ORB: Orbital Materials foundation model (v1, v2, v3)
    - ALIGNN: Pretrained graph neural network from JARVIS-DFT (fast surrogate)
    - CGCNN: Crystal Graph Convolutional Neural Network (fast surrogate)

    Features:
    - Structure relaxation (MLIP backends only)
    - Energy calculation
    - Decomposition energy (distance to convex hull)
    - Property calculation (bulk modulus, band gap, etc.)

    Note: ALIGNN/CGCNN are fast surrogate models that predict properties directly
          without relaxation. Use them for rapid screening, then refine with MLIP.
    """

    def __init__(self, config: Dict, task_constraints: Dict = None):
        """
        Initialize evaluator

        Args:
            config: Configuration dictionary with 'evaluator' section
            task_constraints: Task-specific property constraints (optional)
                            Used to determine which properties to calculate
        """
        self.config = config
        eval_config = config.get('evaluator', {})

        # Parse task constraints to determine required properties
        task_constraints = task_constraints or {}
        self.required_properties = self._parse_required_properties(task_constraints)

        # Get backend type
        self.backend = eval_config.get('backend', 'chgnet').lower()

        # Determine if backend is a surrogate model (no relaxation)
        self.is_surrogate = self.backend in ['alignn', 'cgcnn']

        # Optional property-specific surrogate backends (e.g., bulk_modulus via ALIGNN)
        self.property_backends = {}
        raw_property_backends = eval_config.get('property_backends', {})
        if raw_property_backends:
            if isinstance(raw_property_backends, dict):
                for prop, backend in raw_property_backends.items():
                    if not backend:
                        continue
                    if not isinstance(backend, str):
                        logger.warning("Property backend for %s must be a string", prop)
                        continue
                    self.property_backends[prop] = backend.lower()
            else:
                logger.warning("evaluator.property_backends must be a dict of property -> backend")

        self.property_predictors = {}
        self.property_backend_device = eval_config.get('property_backend_device', eval_config.get('device', 'cpu'))
        self.property_backend_model_dir = eval_config.get('property_backend_model_dir', eval_config.get('model_dir', None))
        self.property_backend_model_path = eval_config.get('property_backend_model_path', eval_config.get('model_path', None))
        self.property_backends_external = {}
        raw_external = eval_config.get('property_backend_external', {})
        if raw_external:
            if isinstance(raw_external, dict):
                self.property_backends_external = raw_external
            else:
                logger.warning("evaluator.property_backend_external must be a dict")
        self.property_calibration = {}
        raw_calibration = eval_config.get('property_calibration', {})
        if raw_calibration:
            if isinstance(raw_calibration, dict):
                self.property_calibration = raw_calibration
            else:
                logger.warning("evaluator.property_calibration must be a dict")

        if self.property_backends and self.is_surrogate:
            logger.info("Property-specific backends are ignored for surrogate main backend")
        elif self.property_backends:
            logger.info(f"Property-specific backends enabled: {self.property_backends}")

        # Load appropriate model and calculator
        logger.info(f"Loading {self.backend.upper()} model...")
        self.model, self.calculator = self._load_model(eval_config)

        if self.is_surrogate:
            logger.info(f"Using surrogate model backend - direct property prediction without relaxation")

        # Materials Project query controls (prevent runaway recomputation)
        self.mp_timeout = eval_config.get('mp_timeout', 60)  # seconds
        self.mp_max_entries = eval_config.get('mp_max_entries', 300)
        self.mp_recompute_mlip = eval_config.get('mp_recompute_mlip', True)
        self.mp_offline = eval_config.get('mp_offline', False) or os.getenv('MP_OFFLINE') == '1'

        # Relaxation parameters
        relax_config = eval_config.get('relax', {})
        self.fmax = relax_config.get('fmax', 0.1)
        self.max_steps = relax_config.get('steps', 500)
        self.optimizer_name = relax_config.get('optimizer', 'FIRE')
        self.cell_filter_name = relax_config.get('cell_filter', None)

        # Phase diagram cache for stability calculation
        # Key: frozenset of elements, Value: list of MLIP-recomputed competing entries
        self.phase_diagram_cache = {}

        # MLIP energy cache for MP structures
        self.mlip_energy_cache = {}

        self._load_phase_diagram()

    def _load_model(self, eval_config: Dict):
        """
        Load ML potential model based on backend

        Args:
            eval_config: Evaluator configuration

        Returns:
            (model, calculator) tuple
        """
        if self.backend == 'chgnet':
            try:
                from chgnet.model import CHGNet
                from chgnet.model.dynamics import CHGNetCalculator

                device = eval_config.get('device', 'cpu')
                check_cuda_mem = eval_config.get('check_cuda_mem', False)
                if device == 'cuda':
                    try:
                        import torch
                        if not torch.cuda.is_available():
                            logger.warning("CUDA requested but not available; falling back to CPU")
                            device = 'cpu'
                    except Exception as e:
                        logger.warning(f"CUDA check failed ({e}); falling back to CPU")
                        device = 'cpu'

                model = CHGNet.load(use_device=device, check_cuda_mem=check_cuda_mem)
                calculator = CHGNetCalculator(model, use_device=device, check_cuda_mem=check_cuda_mem)
                logger.info(f"CHGNet loaded successfully on {device}")
                return model, calculator

            except ImportError:
                logger.error("CHGNet not installed. Install with: pip install chgnet")
                raise

        elif self.backend == 'm3gnet':
            try:
                from matgl.ext.ase import M3GNetCalculator
                import matgl

                model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
                calculator = M3GNetCalculator(potential=model)
                logger.info("M3GNet loaded successfully")
                return model, calculator

            except ImportError:
                logger.error("M3GNet not installed. Install with: pip install matgl")
                raise

        elif self.backend == 'orb':
            try:
                from orb_models.forcefield import pretrained
                from orb_models.forcefield.calculator import ORBCalculator

                model_name = eval_config.get('model_name', 'orb-v3')
                device = eval_config.get('device', 'cpu')  # 'cpu' or 'cuda'

                # Load ORB model
                logger.info(f"Loading ORB model: {model_name} on {device}")

                if model_name == 'orb-v1' or model_name == 'orb_v1':
                    orbff = pretrained.orb_v1(device=device)
                elif model_name == 'orb-v2' or model_name == 'orb_v2':
                    orbff = pretrained.orb_v2(device=device)
                elif model_name in ['orb-v3', 'orb_v3', 'orb']:
                    # Use conservative model with 20Å cutoff trained on Materials Project-like data
                    # This is best for crystal structure evaluation
                    orbff = pretrained.orb_v3_conservative_20_mpa(device=device)
                elif model_name == 'orb_v3_direct':
                    orbff = pretrained.orb_v3_direct_20_mpa(device=device)
                elif hasattr(pretrained, model_name):
                    # Allow direct specification of any pretrained model
                    orbff = getattr(pretrained, model_name)(device=device)
                else:
                    raise ValueError(f"Unknown ORB model: {model_name}")

                calculator = ORBCalculator(orbff, device=device)
                logger.info(f"{model_name.upper()} loaded successfully on {device}")
                return orbff, calculator

            except ImportError as e:
                logger.error(f"ORB not installed: {e}")
                logger.error("Install with: pip install orb-models")
                raise

        elif self.backend == 'alignn':
            try:
                from ..surrogate_models.alignn_wrapper import ALIGNNPredictor

                device = eval_config.get('device', 'cpu')
                model_dir = eval_config.get('model_dir', None)

                predictor = ALIGNNPredictor(model_dir=model_dir, device=device)
                logger.info(f"ALIGNN loaded successfully")

                # Return predictor as both model and calculator (they're the same for surrogates)
                return predictor, predictor

            except ImportError as e:
                logger.error(f"ALIGNN not installed: {e}")
                logger.error("Install with: pip install alignn jarvis-tools")
                raise

        elif self.backend == 'cgcnn':
            try:
                from ..surrogate_models.alignn_wrapper import CGCNNPredictor

                device = eval_config.get('device', 'cpu')
                model_path = eval_config.get('model_path', None)

                predictor = CGCNNPredictor(model_path=model_path, device=device)
                logger.info(f"CGCNN loaded successfully")

                return predictor, predictor

            except ImportError as e:
                logger.error(f"CGCNN not installed: {e}")
                logger.error("Install from: https://github.com/txie-93/cgcnn")
                raise

        else:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                f"Supported: chgnet, m3gnet, orb, alignn, cgcnn"
            )

    def _get_property_predictor(self, backend: str):
        """
        Load (or return cached) property-specific surrogate predictor.
        """
        backend = backend.lower()
        if backend in self.property_predictors:
            return self.property_predictors[backend]

        if backend == 'alignn':
            from ..surrogate_models.alignn_wrapper import ALIGNNPredictor

            predictor = ALIGNNPredictor(
                model_dir=self.property_backend_model_dir,
                device=self.property_backend_device
            )
        elif backend == 'cgcnn':
            from ..surrogate_models.alignn_wrapper import CGCNNPredictor

            predictor = CGCNNPredictor(
                model_path=self.property_backend_model_path,
                device=self.property_backend_device
            )
        else:
            raise ValueError(
                f"Unsupported property backend: {backend}. "
                "Supported: alignn, cgcnn"
            )

        self.property_predictors[backend] = predictor
        return predictor

    def _predict_with_external_backend(self, backend: str, structure: CrystalStructure, properties: List[str]) -> Dict[str, Optional[float]]:
        config = self.property_backends_external.get(backend, {})
        command = config.get('command')
        if not command:
            raise ValueError(f"property_backend_external.{backend}.command is required")

        if isinstance(command, str):
            cmd = shlex.split(command)
        else:
            cmd = list(command)

        payload = {
            'structure': structure.to_dict(),
            'properties': properties,
            'device': config.get('device'),
            'model_dir': config.get('model_dir'),
        }
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        timeout = config.get('timeout', 300)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name

        cmd = cmd + ["--input", tmp_path]
        try:
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                cwd=project_root,
                timeout=timeout,
                check=False,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "unknown error"
            raise RuntimeError(f"External backend {backend} failed: {detail}")

        raw_stdout = result.stdout.strip()
        if not raw_stdout:
            raise RuntimeError(f"External backend {backend} returned empty output")

        try:
            response = json.loads(raw_stdout)
        except json.JSONDecodeError as exc:
            response = None
            for line in reversed(raw_stdout.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    response = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            if response is None:
                snippet = raw_stdout[-1000:]
                raise RuntimeError(
                    f"External backend {backend} returned invalid JSON: {exc}; "
                    f"stdout tail: {snippet}"
                ) from exc

        # Check if the external backend returned an error response
        if 'error' in response:
            error_msg = response.get('error', 'Unknown error')
            detail = response.get('detail', '')
            detail_str = f"; detail: {detail[:500]}" if detail else ""
            raise RuntimeError(f"External backend {backend} returned error: {error_msg}{detail_str}")

        predictions = response.get('predictions', {})

        # Log warning for None values (indicates internal ALIGNN failure)
        none_props = [p for p in properties if predictions.get(p) is None]
        if none_props:
            debug_info = response.get('debug', '')
            debug_str = f"; debug: {debug_info[:500]}" if debug_info else ""
            logger.warning(f"External backend {backend} returned None for: {none_props}{debug_str} (will fallback to CHGNet)")

        return {prop: predictions.get(prop) for prop in properties}

    def _predict_properties_with_backends(self, structure: CrystalStructure, properties: set) -> Dict[str, Optional[float]]:
        """
        Predict selected properties using property-specific backends.
        """
        if not self.property_backends or self.is_surrogate:
            return {}

        props = [p for p in properties if p in self.property_backends and p != 'decomposition_energy']
        if not props:
            return {}

        pmg_struct = structure.to_pymatgen()
        predictions = {}
        grouped = {}
        for prop in props:
            grouped.setdefault(self.property_backends[prop], []).append(prop)

        for backend, backend_props in grouped.items():
            if backend in self.property_backends_external:
                try:
                    backend_predictions = self._predict_with_external_backend(backend, structure, backend_props)
                except Exception as e:
                    logger.warning(f"Property prediction failed with external backend {backend}: {e}")
                    backend_predictions = {prop: None for prop in backend_props}
            else:
                try:
                    predictor = self._get_property_predictor(backend)
                except Exception as e:
                    logger.warning(f"Property backend {backend} unavailable: {e}")
                    for prop in backend_props:
                        predictions[prop] = None
                    continue

                try:
                    if hasattr(predictor, 'predict_multiple'):
                        backend_predictions = predictor.predict_multiple(pmg_struct, backend_props)
                    else:
                        backend_predictions = {prop: predictor.predict(pmg_struct, prop) for prop in backend_props}
                except Exception as e:
                    logger.warning(f"Property prediction failed with {backend}: {e}")
                    backend_predictions = {prop: None for prop in backend_props}

            predictions.update(backend_predictions)

        if self.property_calibration:
            for prop, value in list(predictions.items()):
                if value is None:
                    continue
                calib = self.property_calibration.get(prop)
                if not calib:
                    continue
                try:
                    if isinstance(calib, dict) and calib.get('type', 'linear') == 'linear':
                        a = float(calib.get('a', 1.0))
                        b = float(calib.get('b', 0.0))
                        predictions[prop] = a * float(value) + b
                    elif isinstance(calib, (list, tuple)) and len(calib) == 2:
                        a, b = calib
                        predictions[prop] = float(a) * float(value) + float(b)
                except Exception as exc:
                    logger.warning(f"Failed to apply calibration for {prop}: {exc}")

        return predictions

    def _load_phase_diagram(self):
        """Load Materials Project phase diagram"""
        try:
            logger.info("Loading Materials Project phase diagram...")
            # This would load from MP API or cached data
            # For now, we'll compute it on demand
            pass
        except Exception as e:
            logger.warning(f"Could not load phase diagram: {e}")

    def _parse_required_properties(self, task_constraints: Dict) -> set:
        """
        Parse task constraints to determine which properties need to be calculated

        Args:
            task_constraints: Task constraint dictionary

        Returns:
            Set of property names that need to be calculated
        """
        required = set()

        for prop_name, prop_def in task_constraints.items():
            # Skip non-computable constraints
            if prop_name in ['is_valid', 'contains_mobile_species']:
                continue

            if not isinstance(prop_def, dict):
                continue

            # Only include enabled constraints
            if not prop_def.get('enabled', True):
                continue

            # Add property to required set
            required.add(prop_name)

        if required:
            logger.info(f"Task requires these properties to be calculated: {sorted(required)}")

        return required

    def _wrap_with_cell_filter(self, atoms):
        """
        Optionally wrap atoms with an ASE cell filter for cell relaxation.
        """
        name = self.cell_filter_name
        if not name:
            return None

        if not isinstance(name, str):
            logger.warning(f"cell_filter must be a string, got {type(name).__name__}")
            return None

        key = name.strip()
        if not key or key.lower() in ("none", "off", "false", "no"):
            return None

        alias_map = {
            "unit": "UnitCellFilter",
            "unitcell": "UnitCellFilter",
            "unitcellfilter": "UnitCellFilter",
            "exp": "ExpCellFilter",
            "expcell": "ExpCellFilter",
            "expcellfilter": "ExpCellFilter",
            "frechet": "FrechetCellFilter",
            "frechetcellfilter": "FrechetCellFilter",
        }

        class_name = alias_map.get(key.lower(), key)
        cls = getattr(ase_constraints, class_name, None)

        if cls is None:
            if class_name == "FrechetCellFilter":
                fallback = getattr(ase_constraints, "UnitCellFilter", None)
                if fallback is None:
                    logger.warning("FrechetCellFilter not available and UnitCellFilter missing; skipping cell relaxation")
                    return None
                logger.warning(
                    "FrechetCellFilter not available in ASE %s; falling back to UnitCellFilter",
                    ase.__version__,
                )
                return fallback(atoms)

            logger.warning(f"Unknown cell_filter '{key}'; skipping cell relaxation")
            return None

        try:
            return cls(atoms)
        except Exception as e:
            logger.warning(f"Failed to apply cell_filter '{class_name}': {e}")
            return None

    def _validate_and_fix_structure(self, structure: CrystalStructure) -> CrystalStructure:
        """
        Validate and fix common structure issues

        Args:
            structure: Input structure

        Returns:
            Fixed structure (or original if no issues)
        """
        try:
            pmg_struct = structure.to_pymatgen()

            # 1. Wrap fractional coordinates to [0, 1)
            fixed = False
            for site in pmg_struct:
                for i, coord in enumerate(site.frac_coords):
                    if coord < 0 or coord >= 1.0:
                        fixed = True
                        break
                if fixed:
                    break

            if fixed:
                logger.debug("Wrapping fractional coordinates to [0, 1) range")
                # Pymatgen should auto-wrap, but ensure it
                from pymatgen.core import Structure
                new_struct = Structure.from_sites(pmg_struct.sites)
                # Update CrystalStructure with fixed coordinates
                structure.positions = new_struct.frac_coords
                structure.lattice = new_struct.lattice.matrix

            # 2. Check minimum inter-atomic distance
            try:
                if len(pmg_struct) > 1:
                    # Get all distances
                    all_neighbors = pmg_struct.get_all_neighbors(r=10.0)
                    if all_neighbors and len(all_neighbors[0]) > 0:
                        min_dist = min(neighbor[1] for atom_neighbors in all_neighbors for neighbor in atom_neighbors)
                        if min_dist < 0.5:  # Less than 0.5 Å is unphysical
                            logger.warning(f"Very small inter-atomic distance detected: {min_dist:.3f} Å - structure may be problematic")
                            structure.metadata['min_distance_warning'] = min_dist
            except Exception as dist_error:
                logger.debug(f"Inter-atomic distance check failed: {dist_error}")

            return structure

        except Exception as e:
            logger.warning(f"Structure validation failed: {e}")
            return structure  # Return original if validation fails

    def evaluate(self, structure: CrystalStructure) -> CrystalStructure:
        """
        Complete evaluation pipeline

        Args:
            structure: Input structure

        Returns:
            Evaluated structure with computed properties
        """
        base_metadata = dict(getattr(structure, "metadata", {}) or {})

        # 0. Validate and fix structure
        structure = self._validate_and_fix_structure(structure)
        if getattr(structure, "metadata", None):
            base_metadata.update(structure.metadata)

        # Surrogate models use a different evaluation path (no relaxation)
        if self.is_surrogate:
            return self._evaluate_surrogate(structure, base_metadata)

        # 1. Relax structure (MLIP backends only)
        relaxed = self.relax_structure(structure)
        if base_metadata:
            if not getattr(relaxed, "metadata", None):
                relaxed.metadata = {}
            relaxed.metadata.update(base_metadata)

        if not relaxed.is_valid:
            # Relaxation failed, return invalid structure
            return relaxed

        # 2. Calculate energy
        relaxed.energy = self.calculate_energy(relaxed)

        # 3. Calculate decomposition energy
        relaxed.decomposition_energy = self.calculate_decomposition_energy(relaxed)

        # 4. Calculate task-required properties dynamically
        property_predictions = self._predict_properties_with_backends(relaxed, self.required_properties)
        handled_properties = set()

        if property_predictions:
            if not getattr(relaxed, "metadata", None):
                relaxed.metadata = {}
            relaxed.metadata['property_backends'] = self.property_backends

        for prop_name, value in property_predictions.items():
            if value is None:
                continue

            if prop_name == 'bulk_modulus':
                if value > 0:
                    relaxed.bulk_modulus = value
                    relaxed.properties['bulk_modulus'] = value
                    handled_properties.add('bulk_modulus')
            elif prop_name == 'shear_modulus':
                if value > 0:
                    relaxed.shear_modulus = value
                    relaxed.properties['shear_modulus'] = value
                    handled_properties.add('shear_modulus')
            else:
                relaxed.properties[prop_name] = value
                handled_properties.add(prop_name)

        for prop_name in self.required_properties:
            if prop_name == 'decomposition_energy':
                # Already calculated above
                continue
            if prop_name in handled_properties:
                continue

            # Calculate property based on name
            if prop_name == 'bulk_modulus':
                bulk_mod = self.calculate_bulk_modulus(relaxed)
                setattr(relaxed, 'bulk_modulus', bulk_mod)
                relaxed.properties['bulk_modulus'] = bulk_mod

            elif prop_name == 'shear_modulus':
                shear_mod = self.calculate_shear_modulus(relaxed)
                setattr(relaxed, 'shear_modulus', shear_mod)
                relaxed.properties['shear_modulus'] = shear_mod

            elif prop_name == 'density':
                density = self.calculate_density(relaxed)
                relaxed.properties['density'] = density

            elif prop_name == 'formation_energy':
                # Formation energy requires additional reference data
                # For now, skip or use a placeholder
                logger.debug(f"Property '{prop_name}' calculation not yet implemented")

            else:
                logger.warning(f"Unknown property '{prop_name}' requested by task")

        relaxed.is_valid = True
        relaxed.is_relaxed = True

        return relaxed

    def _evaluate_surrogate(self, structure: CrystalStructure, base_metadata: Dict) -> CrystalStructure:
        """
        Evaluate structure using surrogate models (ALIGNN/CGCNN)

        Surrogate models predict properties directly without relaxation.
        Much faster but less accurate than MLIP.

        Args:
            structure: Input structure
            base_metadata: Metadata to preserve

        Returns:
            Evaluated structure with predicted properties
        """
        try:
            pmg_struct = structure.to_pymatgen()

            # Map property names to surrogate model names
            property_mapping = {
                'formation_energy': 'formation_energy',
                'decomposition_energy': 'energy_above_hull',  # Use Ehull directly!
                'bulk_modulus': 'bulk_modulus',
                'shear_modulus': 'shear_modulus',
                'bandgap': 'bandgap',
                'energy': 'formation_energy'  # Use formation energy for total energy
            }

            # Determine which properties to calculate
            props_to_calc = set()

            # Always calculate basic properties
            props_to_calc.add('energy_above_hull')  # Ehull = decomposition energy
            props_to_calc.add('bulk_modulus')

            # Add task-required properties
            for prop_name in self.required_properties:
                if prop_name in property_mapping:
                    props_to_calc.add(property_mapping[prop_name])

            # Predict properties
            logger.info(f"    Predicting properties with {self.backend.upper()}: {props_to_calc}")

            predicted = {}
            for prop in props_to_calc:
                try:
                    value = self.model.predict(pmg_struct, prop)
                    predicted[prop] = value
                    logger.debug(f"      {prop} = {value:.6f}")
                except Exception as e:
                    logger.warning(f"      Failed to predict {prop}: {e}")
                    predicted[prop] = None

            # Set structure properties
            # Use energy above hull (Ehull) as decomposition energy
            ehull = predicted.get('energy_above_hull')
            if ehull is not None:
                # Ehull is energy above convex hull in eV/atom
                # This is exactly what we want for decomposition energy!
                structure.decomposition_energy = max(ehull, 0.0)  # Ensure non-negative
                structure.energy = ehull  # Also store as energy
            else:
                structure.decomposition_energy = float('inf')
                structure.energy = float('inf')

            # Set bulk modulus
            bulk_mod = predicted.get('bulk_modulus')
            if bulk_mod is not None and bulk_mod > 0:
                structure.bulk_modulus = bulk_mod
                structure.properties['bulk_modulus'] = bulk_mod
            else:
                structure.bulk_modulus = 0.0
                structure.properties['bulk_modulus'] = 0.0

            # Set shear modulus
            shear_mod = predicted.get('shear_modulus')
            if shear_mod is not None and shear_mod > 0:
                structure.shear_modulus = shear_mod
                structure.properties['shear_modulus'] = shear_mod
            elif bulk_mod is not None and bulk_mod > 0:
                # Estimate from bulk modulus
                structure.shear_modulus = 0.6 * bulk_mod
                structure.properties['shear_modulus'] = structure.shear_modulus
            else:
                structure.shear_modulus = 0.0
                structure.properties['shear_modulus'] = 0.0

            # Set bandgap if predicted
            bandgap = predicted.get('bandgap')
            if bandgap is not None:
                structure.properties['bandgap'] = bandgap

            # Set density
            structure.properties['density'] = self.calculate_density(structure)

            # Update metadata
            if not getattr(structure, "metadata", None):
                structure.metadata = {}
            structure.metadata.update(base_metadata)
            structure.metadata['backend'] = self.backend
            structure.metadata['surrogate_model'] = True
            structure.metadata['predicted_properties'] = list(predicted.keys())

            structure.is_valid = True
            structure.is_relaxed = False  # Surrogate models don't relax

            logger.info(f"    Surrogate prediction complete: Ed={structure.decomposition_energy:.3f} eV/atom, B={structure.bulk_modulus:.1f} GPa")

            return structure

        except Exception as e:
            logger.error(f"Surrogate evaluation failed: {e}")
            logger.debug(f"Error details: {e}", exc_info=True)
            structure.is_valid = False
            return structure

    def relax_structure(self, structure: CrystalStructure) -> CrystalStructure:
        """
        Relax structure using the selected MLIP

        Args:
            structure: Input structure

        Returns:
            Relaxed structure
        """
        try:
            # Convert to ASE atoms
            pmg_struct = structure.to_pymatgen()
            atoms = ase.Atoms(
                symbols=[str(s) for s in pmg_struct.species],  # Convert Element objects to strings
                positions=pmg_struct.cart_coords,
                cell=pmg_struct.lattice.matrix,
                pbc=True
            )

            # Set calculator
            atoms.calc = self.calculator

            # Optional cell relaxation
            opt_target = atoms
            cell_filter = self._wrap_with_cell_filter(atoms)
            if cell_filter is not None:
                opt_target = cell_filter
                logger.info(f"    Using cell filter: {self.cell_filter_name}")

            # Choose optimizer
            if self.optimizer_name.upper() == 'FIRE':
                optimizer = FIRE(opt_target, logfile=None)
            elif self.optimizer_name.upper() == 'BFGS':
                optimizer = BFGS(opt_target, logfile=None)
            elif self.optimizer_name.upper() == 'LBFGS':
                optimizer = LBFGS(opt_target, logfile=None)
            else:
                logger.warning(f"Unknown optimizer {self.optimizer_name}, using FIRE")
                optimizer = FIRE(opt_target, logfile=None)

            # Run relaxation
            logger.info(f"    Starting {self.backend.upper()} relaxation (max_steps={self.max_steps}, fmax={self.fmax})...")
            optimizer.run(fmax=self.fmax, steps=self.max_steps)
            logger.info(f"    Relaxation completed in {optimizer.get_number_of_steps()} steps")

            # Convert back to CrystalStructure
            relaxed_struct = Structure(
                lattice=atoms.cell[:],
                species=atoms.get_chemical_symbols(),
                coords=atoms.get_positions(),
                coords_are_cartesian=True
            )

            relaxed = CrystalStructure.from_pymatgen(relaxed_struct)
            relaxed.metadata['relaxation_steps'] = optimizer.get_number_of_steps()
            relaxed.metadata['backend'] = self.backend
            relaxed.is_valid = True
            relaxed.is_relaxed = True

            return relaxed

        except Exception as e:
            logger.error(f"Relaxation failed with {self.backend}: {e}")
            structure.is_valid = False
            return structure

    def calculate_energy(self, structure: CrystalStructure) -> float:
        """
        Calculate total energy

        Args:
            structure: Crystal structure

        Returns:
            Energy per atom (eV/atom)
        """
        try:
            pmg_struct = structure.to_pymatgen()
            
            if self.backend in ['chgnet', 'm3gnet']:
                # CHGNet and M3GNet use predict_structure
                prediction = self.model.predict_structure(pmg_struct)
                # CHGNet predict_structure()['e'] is already in eV/atom.
                # Do not divide by reduced-formula atom count again.
                e_value = prediction['e']
                if hasattr(e_value, 'item'):
                    e_value = e_value.item()
                energy_per_atom = float(e_value)

            elif self.backend == 'orb':
                # ORB uses ASE calculator
                atoms = ase.Atoms(
                    symbols=[str(s) for s in pmg_struct.species],
                    positions=pmg_struct.cart_coords,
                    cell=pmg_struct.lattice.matrix,
                    pbc=True
                )
                atoms.calc = self.calculator
                total_energy = atoms.get_potential_energy()  # eV
                energy_per_atom = total_energy / len(atoms)

            else:
                raise ValueError(f"Energy calculation not implemented for {self.backend}")

            return energy_per_atom

        except Exception as e:
            logger.error(f"Energy calculation failed: {e}")
            return float('inf')

    def calculate_decomposition_energy(self, structure: CrystalStructure) -> float:
        """
        Calculate decomposition energy (distance to convex hull)

        Args:
            structure: Crystal structure

        Returns:
            Decomposition energy (eV/atom)
        """
        try:
            pmg_struct = structure.to_pymatgen()
            composition = pmg_struct.composition

            # Get competing phase entries from MP (cached)
            competing_entries = self._get_competing_phases_entries(composition)

            if competing_entries is None or len(competing_entries) == 0:
                logger.warning("No phase diagram data; marking Ed as inf")
                return float("inf")

            # Create ComputedEntry for our structure
            # CRITICAL: composition.num_atoms is for REDUCED formula (e.g., Na1Cl1)
            # but structure.num_atoms is actual atom count (e.g., 8 for Na4Cl4)
            # Energy must be scaled to match composition's atom count

            total_energy_actual = structure.energy * structure.num_atoms  # Total energy for actual structure

            # Scale energy to match reduced composition
            # e.g., Na4Cl4 (8 atoms, E=-3.666 eV) → Na1Cl1 (2 atoms, E=-0.917 eV)
            energy_for_reduced = total_energy_actual * (composition.num_atoms / structure.num_atoms)

            # Debug logging (can be disabled in production)
            logger.debug(f"Structure atoms: {structure.num_atoms}, Composition: {composition.reduced_formula}")
            logger.debug(f"Energy scaling: {structure.energy:.4f} eV/atom × {structure.num_atoms} atoms × ({composition.num_atoms}/{structure.num_atoms}) = {energy_for_reduced:.4f} eV")

            our_entry = ComputedEntry(
                composition=composition,  # Reduced formula (e.g., Na1Cl1, num_atoms=2)
                energy=energy_for_reduced  # Energy for reduced formula
            )

            # Build phase diagram with all entries
            all_entries = competing_entries + [our_entry]

            try:
                phase_diagram = PhaseDiagram(all_entries)
            except ValueError as e:
                # Handle missing terminal entries (e.g., rare earth elements)
                if "Missing terminal entries" in str(e):
                    logger.warning(f"Cannot build phase diagram for {composition.reduced_formula}: {e}")
                    logger.warning("  MP data insufficient - using formation energy as fallback")

                    # FALLBACK: Use formation energy per atom as rough stability indicator
                    # Formation energy = (E_total - sum(E_elements)) / n_atoms
                    # Negative = stable, positive = unstable
                    # This is less accurate than Ed but better than inf

                    # Get elemental reference energies from MP (if available)
                    formation_energy_per_atom = None
                    try:
                        mp_api_key = os.getenv('MATERIALS_PROJECT_API_KEY') or os.getenv('MP_API_KEY')
                        if not mp_api_key:
                            logger.warning("  MP API key not found - cannot compute formation energy fallback")
                        elif self.mp_offline:
                            logger.warning("  MP_OFFLINE set - skipping formation energy fallback")
                        else:
                            elemental_energies = {}
                            with MPRester(mp_api_key) as mpr:
                                for element in composition.elements:
                                    # Query MP for elemental reference
                                    elem_results = mpr.materials.summary.search(
                                        chemsys=str(element),
                                        num_elements=(1, 1),
                                        fields=["energy_per_atom"]
                                    )
                                    if elem_results:
                                        # Use lowest energy polymorph
                                        elemental_energies[str(element)] = min(r.energy_per_atom for r in elem_results)

                            if len(elemental_energies) == len(composition.elements):
                                # Calculate formation energy
                                total_atoms = sum(composition.get_atomic_fraction(el) for el in composition.elements)
                                ref_energy = sum(
                                    composition.get_atomic_fraction(el) * elemental_energies[str(el)]
                                    for el in composition.elements
                                )
                                formation_energy_per_atom = (energy_for_reduced - ref_energy) / total_atoms

                                # Use absolute value + penalty as Ed proxy
                                # Stable materials: -3 to 0 eV/atom formation energy → Ed proxy: 0-3
                                # Unstable materials: >0 eV/atom → Ed proxy: higher
                                ed_proxy = abs(formation_energy_per_atom) + 2.0  # Add penalty for uncertainty

                                logger.warning(f"  Using Ed proxy = {ed_proxy:.3f} eV/atom (from formation energy)")
                                return ed_proxy

                    except Exception as fe:
                        logger.warning(f"  Formation energy calculation failed: {fe}")

                    # Last resort: return moderate penalty instead of infinity
                    # Lower penalty (5.0 vs 10.0) to avoid overly penalizing rare earth elements
                    logger.warning("  Using moderate penalty (5.0 eV/atom) for missing MP data")
                    return 5.0  # Moderate penalty - allows rare earth structures to participate
                else:
                    raise

            # Calculate decomposition energy (distance to convex hull)
            # This properly accounts for decomposition into multiple phases
            # get_e_above_hull returns energy per atom above the convex hull
            ed_per_atom = phase_diagram.get_e_above_hull(our_entry)

            # Log the Ed value for debugging
            logger.debug(f"  {composition.reduced_formula}: Ed = {ed_per_atom:.6f} eV/atom")

            return ed_per_atom

        except Exception as e:
            logger.error(f"Ed calculation failed for {structure.formula}: {e}")
            logger.debug(f"Error details: {e}", exc_info=True)
            return float('inf')

    def _get_competing_phases_entries(self, composition) -> Optional[List[ComputedEntry]]:
        """
        Get competing phase entries from Materials Project with MLIP-recomputed energies

        CRITICAL FIX: To ensure consistent energy reference, we recompute all MP structures
        using the same MLIP backend. This fixes the energy scale mismatch between DFT (MP)
        and MLIP predictions.

        Args:
            composition: Pymatgen Composition object

        Returns:
            List of ComputedEntry objects with MLIP energies, or None if unavailable
        """
        try:
            # Create cache key from sorted elements
            elements = sorted([str(e) for e in composition.elements])
            cache_key = frozenset(elements)

            # Check cache first
            if cache_key in self.phase_diagram_cache:
                logger.debug(f"Using cached MLIP phase diagram for {'-'.join(elements)}")
                return self.phase_diagram_cache[cache_key]

            import os
            mp_api_key = os.getenv('MATERIALS_PROJECT_API_KEY') or os.getenv('MP_API_KEY')

            if not mp_api_key:
                logger.warning("MP API key not found (tried MATERIALS_PROJECT_API_KEY and MP_API_KEY) - cannot query phase diagram")
                # Cache the None result to avoid repeated warnings
                self.phase_diagram_cache[cache_key] = None
                return None

            # Respect offline mode
            if self.mp_offline:
                logger.warning("MP_OFFLINE set - skipping Materials Project query for phase diagram")
                self.phase_diagram_cache[cache_key] = None
                return None

            # Query Materials Project for entries in this chemical system
            logger.info(f"Querying Materials Project for {'-'.join(elements)} phase diagram...")

            # mp_api <=0.39 does not support timeout kwarg; rely on default session timeout
            with MPRester(mp_api_key) as mpr:
                # Get all entries for this chemical system
                mp_entries = mpr.get_entries_in_chemsys(elements)

                if not mp_entries:
                    logger.warning(f"No MP entries found for system {'-'.join(elements)}")
                    # Cache the None result
                    self.phase_diagram_cache[cache_key] = None
                    return None

                logger.info(f"Found {len(mp_entries)} MP entries for {'-'.join(elements)}")

                # Optional downsampling to avoid huge recomputation cost.
                # Always keep elemental endpoints to ensure phase diagram can be built.
                if len(mp_entries) > self.mp_max_entries:
                    elemental_entries = [e for e in mp_entries if len(e.composition.elements) == 1]
                    non_elemental = [e for e in mp_entries if len(e.composition.elements) > 1]
                    keep_non_elemental = max(self.mp_max_entries - len(elemental_entries), 0)
                    non_elemental = sorted(non_elemental, key=lambda e: e.energy_per_atom)[: keep_non_elemental]
                    mp_entries = elemental_entries + non_elemental
                    logger.warning(
                        f"Truncated MP entries to {len(mp_entries)} (kept all elemental endpoints, "
                        f"capped non-elemental to {keep_non_elemental}; mp_max_entries={self.mp_max_entries})"
                    )

                # Skip MLIP recomputation if disabled (faster but mixes energy scales)
                if not self.mp_recompute_mlip:
                    logger.info("Using MP DFT energies directly (mp_recompute_mlip=False)")
                    self.phase_diagram_cache[cache_key] = mp_entries
                    return mp_entries

                # CRITICAL FIX: Recompute all energies using MLIP to ensure consistent energy scale
                logger.info(f"Recomputing energies with {self.backend.upper()} for energy consistency...")

                mlip_entries = []
                failed_count = 0

                for i, mp_entry in enumerate(mp_entries):
                    try:
                        # Get structure from MP entry
                        # MP entries have structure attribute for some entry types
                        if hasattr(mp_entry, 'structure') and mp_entry.structure is not None:
                            mp_struct = mp_entry.structure
                        else:
                            # If no structure in entry, fetch it from MP
                            entry_id = mp_entry.entry_id if hasattr(mp_entry, 'entry_id') else None
                            if entry_id:
                                mp_struct = mpr.get_structure_by_material_id(entry_id)
                            else:
                                logger.warning(f"Cannot get structure for entry {i+1}")
                                failed_count += 1
                                continue

                        # Check MLIP energy cache
                        entry_id = mp_entry.entry_id if hasattr(mp_entry, 'entry_id') else str(mp_entry.composition)

                        if entry_id in self.mlip_energy_cache:
                            mlip_energy_per_atom = self.mlip_energy_cache[entry_id]
                        else:
                            # Compute MLIP energy
                            temp_struct = CrystalStructure.from_pymatgen(mp_struct)
                            mlip_energy_per_atom = self.calculate_energy(temp_struct)

                            # Cache the result
                            self.mlip_energy_cache[entry_id] = mlip_energy_per_atom

                        # Create new entry with MLIP energy
                        # CRITICAL: Must use actual structure atom count, not reduced composition!
                        # mp_struct.num_sites is the actual atom count (e.g., 8 for Na4Cl4)
                        # mp_entry.composition.num_atoms is the reduced count (e.g., 2 for Na1Cl1)
                        mlip_total_energy = mlip_energy_per_atom * mp_struct.num_sites

                        # Scale to match reduced composition
                        energy_for_reduced = mlip_total_energy * (mp_entry.composition.num_atoms / mp_struct.num_sites)

                        mlip_entry = ComputedEntry(
                            composition=mp_entry.composition,
                            energy=energy_for_reduced
                        )

                        mlip_entries.append(mlip_entry)

                        # Progress logging
                        if (i + 1) % 10 == 0:
                            logger.debug(f"  Processed {i+1}/{len(mp_entries)} entries...")

                    except Exception as e:
                        logger.warning(f"Failed to compute MLIP energy for entry {i+1}: {e}")
                        failed_count += 1
                        continue

                if failed_count > 0:
                    logger.warning(f"Failed to compute MLIP energies for {failed_count}/{len(mp_entries)} entries")

                # Log ground state for pure elements (useful for debugging MLIP vs DFT differences)
                if len(elements) == 1 and len(mlip_entries) > 0:
                    min_entry = min(mlip_entries, key=lambda e: e.energy_per_atom)
                    min_mp_entry = mp_entries[mlip_entries.index(min_entry)]
                    entry_id = min_mp_entry.entry_id if hasattr(min_mp_entry, 'entry_id') else 'unknown'
                    logger.debug(f"{elements[0]} ground state: {entry_id} (E={min_entry.energy_per_atom:.4f} eV/atom)")

                if not mlip_entries:
                    logger.warning(f"No valid MLIP entries computed for {'-'.join(elements)}")
                    self.phase_diagram_cache[cache_key] = None
                    return None

                logger.info(f"Successfully computed MLIP energies for {len(mlip_entries)}/{len(mp_entries)} entries")

                # Cache the MLIP-recomputed entries
                self.phase_diagram_cache[cache_key] = mlip_entries

                return mlip_entries

        except Exception as e:
            logger.warning(f"Failed to query Materials Project: {e}")
            logger.debug(f"Error details: {e}", exc_info=True)
            # Cache the None result to avoid repeated failures
            cache_key = frozenset(sorted([str(e) for e in composition.elements]))
            self.phase_diagram_cache[cache_key] = None
            return None

    def calculate_bulk_modulus(self, structure: CrystalStructure) -> float:
        """
        Calculate bulk modulus from second derivative of energy-volume curve

        Uses finite difference method: B = V * d²E/dV²
        This is more robust than fitting Birch-Murnaghan for CHGNet energies

        Args:
            structure: Crystal structure

        Returns:
            Bulk modulus (GPa)
        """
        try:
            # Apply volumetric strains and compute energies
            # Use 7 points for good numerical derivative accuracy
            strains = np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
            volumes = []
            energies = []

            for strain in strains:
                # Scale lattice uniformly (volumetric strain)
                scale = (1 + strain) ** (1/3)
                strained_lattice = structure.lattice * scale

                strained = CrystalStructure(
                    formula=structure.formula,
                    lattice=strained_lattice,
                    positions=structure.positions.copy(),  # Keep fractional coords
                    species=structure.species.copy()
                )

                energy = self.calculate_energy(strained)  # eV/atom
                volume = strained.volume  # Å³

                volumes.append(volume)
                energies.append(energy * structure.num_atoms)  # Convert to total energy (eV)

            # Calculate second derivative using finite differences
            # Fit a parabola: E = a*V^2 + b*V + c
            # Then: d²E/dV² = 2*a
            volumes = np.array(volumes)
            energies = np.array(energies)

            # Polynomial fit (2nd order)
            coeffs = np.polyfit(volumes, energies, 2)
            a = coeffs[0]  # Coefficient of V^2 term

            # Second derivative
            d2E_dV2 = 2 * a  # eV/Å^6

            # Bulk modulus: B = V * d²E/dV²
            V0 = structure.volume  # Å³
            bulk_modulus_eV_per_A3 = V0 * d2E_dV2  # eV/Å³

            # Convert to GPa: 1 eV/Å³ = 160.21766 GPa
            bulk_modulus = bulk_modulus_eV_per_A3 * 160.21766

            # Sanity check: bulk modulus should be positive
            if bulk_modulus < 0:
                logger.warning(f"Negative bulk modulus ({bulk_modulus:.2f} GPa) from parabolic fit")

                # Try alternative: finite difference at equilibrium
                try:
                    # Use central difference: (E(V+dV) - E(V-dV)) / (2*dV)
                    idx_mid = len(volumes) // 2
                    if idx_mid > 0 and idx_mid < len(volumes) - 1:
                        dV = volumes[idx_mid + 1] - volumes[idx_mid]
                        dE = energies[idx_mid + 1] - energies[idx_mid - 1]
                        d2E_dV2_fd = dE / (dV * dV)
                        bulk_mod_fd = V0 * d2E_dV2_fd * 160.21766

                        if bulk_mod_fd > 0:
                            # Sanity check: ceramic materials typically have B < 300 GPa
                            # Values significantly above this are likely numerical errors from noisy curves
                            # Even diamond (~450 GPa) would fail our constraints (max 300 GPa)
                            MAX_REASONABLE_BULK = 350.0  # GPa - conservative threshold above constraint max

                            if bulk_mod_fd > MAX_REASONABLE_BULK:
                                logger.warning(f"  Finite difference gave unreasonably high value: {bulk_mod_fd:.2f} GPa (likely numerical error)")
                                logger.warning(f"  Structure likely has unstable energy landscape")
                                # Don't trust this result - treat as mechanically unstable
                                # Return small value to signal instability (will likely fail constraints)
                                return 10.0

                            logger.info(f"  Using finite difference method: B = {bulk_mod_fd:.2f} GPa")
                            return bulk_mod_fd
                except Exception as fd_error:
                    logger.debug(f"  Finite difference failed: {fd_error}")

                # If still negative, structure is likely mechanically unstable
                # Return small positive value to allow evaluation but signal instability
                logger.warning(f"  Structure appears mechanically unstable - using small positive value")
                return 10.0  # Minimal value to avoid division by zero in downstream calcs

            # Final sanity check: even for positive values from parabolic fit
            # ceramic materials typically have B < 300 GPa (per task constraints)
            # Values > 350 GPa are likely numerical artifacts
            MAX_REASONABLE_BULK = 350.0  # GPa

            if bulk_modulus > MAX_REASONABLE_BULK:
                logger.warning(f"Parabolic fit gave unreasonably high bulk modulus: {bulk_modulus:.2f} GPa")
                logger.warning(f"  This exceeds physical expectations for ceramics (max ~300 GPa)")
                logger.warning(f"  Likely numerical error - treating as mechanically unstable")
                return 10.0

            logger.debug(f"Bulk modulus: {bulk_modulus:.1f} GPa (V0={V0:.1f} Å³, d²E/dV²={d2E_dV2:.4e} eV/Å⁶)")

            return bulk_modulus

        except Exception as e:
            logger.error(f"Bulk modulus calculation failed: {e}")
            logger.debug(f"Error details: {e}", exc_info=True)
            return 0.0

    def calculate_shear_modulus(self, structure: CrystalStructure) -> float:
        """
        Calculate shear modulus using approximation from bulk modulus

        NOTE: Full elastic tensor calculation with CHGNet is unstable and causes crashes.
        Using empirical relationship instead: G ≈ 0.6 * B for ceramics

        Args:
            structure: Crystal structure

        Returns:
            Shear modulus (GPa)
        """
        try:
            # Get bulk modulus first
            bulk_mod = self.calculate_bulk_modulus(structure)

            if bulk_mod > 0:
                # Empirical relationship for ceramics
                # Poisson ratio ν ≈ 0.25 → G = 3B(1-2ν)/(2(1+ν)) ≈ 0.6B
                shear_modulus = 0.6 * bulk_mod
                logger.debug(f"Estimated shear modulus from bulk: {shear_modulus:.1f} GPa (B={bulk_mod:.1f} GPa)")
                return shear_modulus
            else:
                logger.warning("Bulk modulus is zero, cannot estimate shear modulus")
                return 0.0

        except Exception as e:
            logger.error(f"Shear modulus calculation failed: {e}")
            return 0.0

    def calculate_density(self, structure: CrystalStructure) -> float:
        """
        Calculate density of the structure

        Args:
            structure: Crystal structure

        Returns:
            Density (g/cm^3)
        """
        try:
            pmg_struct = structure.to_pymatgen()
            density = pmg_struct.density  # g/cm^3
            return density

        except Exception as e:
            logger.error(f"Density calculation failed: {e}")
            return 0.0

    def batch_evaluate(self, structures: List[CrystalStructure]) -> List[CrystalStructure]:
        """
        Evaluate multiple structures in batch

        Args:
            structures: List of structures

        Returns:
            List of evaluated structures
        """
        evaluated = []
        for i, struct in enumerate(structures):
            logger.info(f"Evaluating structure {i+1}/{len(structures)}: {struct.formula}")
            evaluated_struct = self.evaluate(struct)
            evaluated.append(evaluated_struct)

        return evaluated
