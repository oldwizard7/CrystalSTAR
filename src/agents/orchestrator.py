"""Orchestrator Agent - ACE loop controller"""

from typing import List, Dict, Optional
import logging
import math
import numpy as np
import yaml
import json
import platform
from datetime import datetime
from pathlib import Path

from src.core.structure import CrystalStructure
from src.core.evolution import EvolutionEngine
from src.agents.hypothesis_gen import HypothesisGenerationAgent
from src.agents.workers import WorkerAgents
from src.utils.hier_memory import HierMemoryManager

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Orchestrator Agent - Strategic leader of the search system
    
    Implements ACE (Agentic Context Engineering):
    - Generate: Execute current search phase
    - Reflect: Analyze results
    - Curate: Evolve strategy
    """
    
    def __init__(self, config: Dict, prompts: Dict, task_config: Dict = None, output_dir: str = None, constraint_checker = None):
        """
        Initialize orchestrator

        Args:
            config: Global configuration dictionary
            prompts: Prompt templates
            task_config: Task-specific configuration (name, description, constraints, objective)
            output_dir: Output directory for saving logs/strategies (optional)
            constraint_checker: ConstraintChecker for calculating hit rates (optional)
        """
        self.config = config
        self.prompts = prompts
        self.task_config = task_config or {}
        self.output_dir = output_dir
        self.constraint_checker = constraint_checker
        task_key = self.task_config.get('task_key') or self.task_config.get('name')
        self.hier_memory = HierMemoryManager(config, task_key=task_key, task_name=self.task_config.get('name'))

        # Stability thresholds (MASTER CONTROL - highest priority)
        thresholds = config.get('thresholds', {})
        self.metastability_threshold = thresholds.get('metastability', 0.1)
        self.failure_multiplier = thresholds.get('failure_multiplier', 5)
        self.failure_threshold = self.metastability_threshold * self.failure_multiplier

        logger.info(f"Stability thresholds initialized:")
        logger.info(f"  Metastability threshold: {self.metastability_threshold} eV/atom")
        logger.info(f"  Failure threshold: {self.failure_threshold} eV/atom")
        logger.info(f"Hierarchical memory enabled: {self.hier_memory.enabled}")

        # Pre-evaluation guardrail:
        # Avoid known high-risk elemental candidates (e.g., O/F single-element)
        # that can trigger expensive decomposition-energy recomputation paths.
        elem_skip_cfg = (self.config.get('evaluator', {}) or {}).get('elemental_skip_filter', {}) or {}
        self.elemental_skip_enabled = bool(elem_skip_cfg.get('enabled', True))
        symbols = elem_skip_cfg.get('symbols', ['O', 'F'])
        self.elemental_skip_symbols = {
            str(sym).strip() for sym in symbols if str(sym).strip()
        }
        if self.elemental_skip_enabled and self.elemental_skip_symbols:
            logger.info(
                "Elemental pre-eval filter enabled for single-element candidates: "
                f"{sorted(self.elemental_skip_symbols)}"
            )

        # Sub-agents
        self.hypothesis_agent = HypothesisGenerationAgent(config, prompts)
        # Worker agents
        self.worker_agents = WorkerAgents(config)

        # Structure generator (generate from scratch, not modify)
        from src.agents.workers.structure_generator import StructureGeneratorAgent
        self.structure_generator = StructureGeneratorAgent(config)
        self.llm_client = self.structure_generator.llm
        self.hier_memory.set_llm_client(self.llm_client)

        # Tool-based generation system
        from src.core.operators import StructureOperators
        from src.agents.tool_selector import ToolSelectionAgent
        self.operators = StructureOperators()
        self.tool_selector = ToolSelectionAgent(config)

        self.evolution = EvolutionEngine(config)


        # Search state
        self.current_plan = None  # Legacy single strategy (Phase 1)
        self.current_strategies = None  # Phase 2B: Multi-strategy dict
        self.search_history = []
        self.iteration = 0
        self._prior_templates_loaded = False

        # Track all generated structures for global statistics
        self.all_structures = []
        self.gen_requested_total = 0
        self.gen_parse_success_total = 0
        self.gen_parse_failure_total = 0
        self.gen_requested_current_iter = 0
        self.gen_parse_success_current_iter = 0
        self.gen_parse_failure_current_iter = 0

        # 🔥 NEW: Track tool performance statistics
        self.tool_stats = {
            'substitute': {'total': 0, 'valid': 0, 'metastable': 0, 'hits': 0, 'energies': [], 'scores': []},
            'mutate': {'total': 0, 'valid': 0, 'metastable': 0, 'hits': 0, 'energies': [], 'scores': []},
            'mix': {'total': 0, 'valid': 0, 'metastable': 0, 'hits': 0, 'energies': [], 'scores': []},
            'crossover': {'total': 0, 'valid': 0, 'metastable': 0, 'hits': 0, 'energies': [], 'scores': []},
            'dope': {'total': 0, 'valid': 0, 'metastable': 0, 'hits': 0, 'energies': [], 'scores': []},
            'fill_prototype': {'total': 0, 'valid': 0, 'metastable': 0, 'hits': 0, 'energies': [], 'scores': []},
            'new': {'total': 0, 'valid': 0, 'metastable': 0, 'hits': 0, 'energies': [], 'scores': []},
            'direct_llm': {'total': 0, 'valid': 0, 'metastable': 0, 'hits': 0, 'energies': [], 'scores': []},
        }

        # Budget tracking
        self.budget_remaining = config.get('budget', float('inf'))

        # Ablation mode (tool-calling only): memory_less | vanilla_react | flat_memory | crystal_forge
        self.ablation_mode = str(config.get('ablation_mode', 'crystal_forge')).strip().lower()
        if self.ablation_mode not in {'memory_less', 'vanilla_react', 'flat_memory', 'crystal_forge'}:
            logger.warning(
                "Unknown ablation_mode '%s', falling back to 'crystal_forge'",
                self.ablation_mode,
            )
            self.ablation_mode = 'crystal_forge'

        # Generation mode (ablation: "tool_calling" | "direct_llm")
        self.generation_mode = config.get('generation_mode', 'tool_calling')
        if self.generation_mode == 'direct_llm':
            from src.agents.direct_generator import DirectGenerationAgent
            self.direct_generator = DirectGenerationAgent(config)
            logger.info("Generation mode: direct_llm (ablation baseline)")
        else:
            logger.info("Generation mode: tool_calling (default)")

        self.effective_ablation_mode = (
            self.ablation_mode if self.generation_mode == 'tool_calling' else 'crystal_forge'
        )
        if self.generation_mode != 'tool_calling' and self.ablation_mode != 'crystal_forge':
            logger.info(
                "ablation_mode=%s ignored because generation_mode=%s",
                self.ablation_mode,
                self.generation_mode,
            )
        self.hmrr_runtime_enabled = self.effective_ablation_mode == 'crystal_forge'
        logger.info("Effective ablation mode: %s", self.effective_ablation_mode)
        if not self.hmrr_runtime_enabled:
            logger.info("HMRR runtime disabled by ablation_mode")
    
    def initialize(self, reference_path: str, filters: Optional[Dict] = None) -> List[CrystalStructure]:
        """
        Initialize search with reference examples

        Args:
            reference_path: Path to reference examples (3-5 structures for format)
            filters: Not used (kept for compatibility)

        Returns:
            Empty example pool (will be filled during search)
        """
        logger.info("Initializing search...")
        logger.info(f"Loading reference examples from {reference_path}")

        # Load reference examples
        from src.utils.data_manager import DataManager
        dm = DataManager()
        reference_examples = dm.load_structures(reference_path)

        if not reference_examples:
            logger.error(f"No reference examples found in {reference_path}")
            raise ValueError("Need at least 1 reference example")

        logger.info(f"Loaded {len(reference_examples)} reference examples:")
        for ref in reference_examples:
            logger.info(f"  - {ref.formula}")

        # Set reference examples for generator
        self.structure_generator.set_reference_examples(reference_examples)

        example_pool = self.load_initial_parent_pool()

        logger.info("Initialization complete. Ready to generate structures.")

        return example_pool

    def load_initial_parent_pool(self) -> List[CrystalStructure]:
        """
        Load initial parent pool seeds based on config and task-specific overrides.

        Returns:
            Seed structures for the initial parent pool (may be empty).
        """
        # Start with empty example pool (will be populated with successful generations)
        example_pool: List[CrystalStructure] = []

        def _merge_dict(base: Dict, override: Dict) -> Dict:
            merged = base.copy()
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = _merge_dict(merged[key], value)
                else:
                    merged[key] = value
            return merged

        def _register_templates_from_seed(seed_path: Path, template_cfg: Dict) -> None:
            """Load task templates for fill_prototype without requiring a parent pool."""
            if self._prior_templates_loaded:
                return
            index_path = seed_path / "index.json"
            if not index_path.exists():
                return
            if not (template_cfg or {}).get("enabled", True):
                self._prior_templates_loaded = True
                return

            # Sync per-task template preferences to tool selector config.
            data_cfg = self.tool_selector.config.setdefault("data", {})
            init_cfg = data_cfg.setdefault("initial_parent_pool", {})
            init_cfg["template_library"] = {
                **init_cfg.get("template_library", {}),
                **(template_cfg or {}),
            }
            max_templates = int((template_cfg or {}).get("max_templates", 10))
            max_elements = int((template_cfg or {}).get("max_elements", 3))
            templates = self.operators.register_prior_templates(
                index_path=index_path,
                seed_path=seed_path,
                max_templates=max_templates,
                max_elements=max_elements,
            )
            if templates:
                self.tool_selector.set_template_catalog(templates)
                logger.info(f"Registered {len(templates)} task templates from {seed_path}")
            self._prior_templates_loaded = True

        # Optionally seed initial parent pool from config (supports per-task override)
        init_pool_cfg = self.config.get('data', {}).get('initial_parent_pool', {}) or {}
        pool_cfg = init_pool_cfg.copy()

        per_task_cfg = (init_pool_cfg.get('per_task') or {}).get(
            self.task_config.get('task_key') or self.task_config.get('name')
        )
        if per_task_cfg:
            pool_cfg = _merge_dict(pool_cfg, per_task_cfg)

        task_pool_cfg = self.task_config.get('initial_parent_pool') or {}
        if task_pool_cfg:
            pool_cfg = _merge_dict(pool_cfg, task_pool_cfg)

        if not pool_cfg.get('enabled'):
            return example_pool

        evolution_cfg = self.config.get("evolution", {}) or {}
        try:
            pop_size = int(evolution_cfg.get("population_size", 0) or 0)
            parent_size = int(evolution_cfg.get("parent_size", 0) or 0)
        except (TypeError, ValueError):
            pop_size = 0
            parent_size = 0
        parent_capacity = pop_size * parent_size
        if parent_capacity <= 0:
            if pool_cfg.get('path'):
                seed_path = Path(pool_cfg['path'])
                template_cfg = pool_cfg.get("template_library", {}) or {}
                _register_templates_from_seed(seed_path, template_cfg)
            logger.info(
                "Parent pool disabled by evolution config "
                f"(population_size={pop_size}, parent_size={parent_size})."
            )
            return example_pool

        # 🔥 NEW: Check if LLM-driven MP retrieval is enabled
        use_llm_retrieval = pool_cfg.get('use_llm_retrieval', False)
        llm_retrieval_cfg = self.config.get('data', {}).get('initial_parent_pool', {}).get('llm_retrieval', {})

        if use_llm_retrieval and llm_retrieval_cfg.get('enabled', False):
            logger.info("="*60)
            logger.info("Using LLM-driven Materials Project retrieval")
            logger.info("="*60)

            try:
                from src.utils.llm_mp_retriever import TaskOrientedMPRetriever

                # Get MP API key from environment or config
                import os
                mp_api_key = (
                    os.getenv('MATERIALS_PROJECT_API_KEY')
                    or os.getenv('MP_API_KEY')
                    or self.config.get('mp_api_key')
                )

                if not mp_api_key:
                    logger.error("MP_API_KEY not found in environment or config. Falling back to file-based selection.")
                else:
                    # Build retriever config
                    retriever_config = {
                        'mp_api_key': mp_api_key,
                        'cache_dir': llm_retrieval_cfg.get('cache_dir', 'data/mp_cache'),
                        'max_results': llm_retrieval_cfg.get('mp_api', {}).get('max_results', 200),
                        'llm_analyzer': llm_retrieval_cfg.get('llm_analyzer', {})
                    }

                    retriever = TaskOrientedMPRetriever(retriever_config, llm_client=self.llm_client)

                    # Retrieve structures using LLM analysis
                    max_count = pool_cfg.get('max_count', 20)
                    seed_structs = retriever.retrieve_initial_pool(
                        self.task_config,
                        max_count=max_count,
                        force_refresh=pool_cfg.get('force_refresh', False)
                    )

                    if seed_structs:
                        logger.info(f"✓ Successfully retrieved {len(seed_structs)} structures from MP")
                        return seed_structs
                    else:
                        logger.warning("LLM retrieval returned no structures, falling back to file-based selection")

            except Exception as e:
                logger.error(f"LLM-driven MP retrieval failed: {e}", exc_info=True)
                logger.warning("Falling back to file-based selection")

        # Continue with file-based selection if LLM retrieval is disabled or failed
        if not pool_cfg.get('path'):
            return example_pool

        seed_path = Path(pool_cfg['path'])
        strategy = pool_cfg.get('strategy', 'random')
        logger.info(f"Loading initial parent pool from {seed_path} (strategy={strategy})")
        template_cfg = pool_cfg.get("template_library", {}) or {}
        _register_templates_from_seed(seed_path, template_cfg)

        # Use ParentPoolInitializer to select structures
        from src.utils.data_manager import DataManager
        from src.utils.parent_pool_initializer import ParentPoolInitializer

        dm = DataManager()
        initializer = ParentPoolInitializer(pool_cfg)

        # Check if index.json exists
        index_path = seed_path / "index.json"
        if index_path.exists():
            # Use smart selection from index
            selected_examples = initializer.select_from_index(index_path)

            # Convert selected examples to file paths
            selected_files = [ex['file'] for ex in selected_examples]

            # Load only selected structures
            all_structs = dm.load_structures(seed_path)
            seed_structs = []
            for struct in all_structs:
                src_file = struct.metadata.get('source_file', '')
                if src_file in selected_files:
                    seed_structs.append(struct)

            logger.info(f"  Loaded {len(seed_structs)} strategically selected seeds")
        else:
            # Fallback: load all and use legacy random sampling
            logger.warning(f"  No index.json found at {index_path}, using legacy random sampling")
            seed_structs = dm.load_structures(seed_path)
            max_count = pool_cfg.get('max_count', len(seed_structs))
            seed = pool_cfg.get('seed', None)
            if len(seed_structs) > max_count:
                import random
                rng = random.Random(seed) if seed is not None else random
                seed_structs = rng.sample(seed_structs, max_count)
                logger.info(f"  Loaded {max_count}/{len(seed_structs)} seeds (random, seed={seed})")

        if seed_structs:
            # Attach energy_above_hull/metadata to seeds
            if index_path.exists():
                # If using index-based selection, metadata is already selected
                if 'selected_examples' in locals():
                    by_file = {ex.get("file"): ex for ex in selected_examples}
                else:
                    # Legacy path: read all examples
                    try:
                        with open(index_path, "r") as f:
                            idx = json.load(f)
                        examples = idx.get("examples", [])
                        by_file = {ex.get("file"): ex for ex in examples if "file" in ex}
                    except Exception as e:
                        logger.warning(f"  Failed to read seed index at {index_path}: {e}")
                        by_file = {}

                by_formula = {}
                for ex in (selected_examples if 'selected_examples' in locals() else []):
                    if "formula" in ex:
                        by_formula.setdefault(ex["formula"], []).append(ex)

                attached = 0
                unmatched = 0
                for s in seed_structs:
                    src = s.metadata.get("source_file")
                    if src and src in by_file:
                        ex = by_file[src]
                        if ex.get("formula"):
                            s.formula = str(ex["formula"])
                        if "energy_above_hull" in ex:
                            s.decomposition_energy = ex["energy_above_hull"]
                        props = {}
                        for key in ("bulk_modulus", "shear_modulus", "density", "piezoelectric_coefficient", "dielectric_constant", "band_gap"):
                            if key in ex and ex[key] is not None:
                                props[key] = float(ex[key])
                        if "formation_energy_per_atom" in ex and ex["formation_energy_per_atom"] is not None:
                            props["formation_energy"] = float(ex["formation_energy_per_atom"])
                        if props:
                            s.properties.update(props)
                        # store useful metadata
                        s.metadata.update({
                            "mp_id": ex.get("mp_id"),
                            "energy_above_hull": ex.get("energy_above_hull"),
                            "formation_energy_per_atom": ex.get("formation_energy_per_atom"),
                            "energy_per_atom": ex.get("energy_per_atom"),
                            "crystal_system": ex.get("crystal_system"),
                            "space_group": ex.get("space_group"),
                        })
                        s.is_valid = True
                        attached += 1
                        continue

                    # Fallback: match by formula if file name not found
                    if s.formula in by_formula:
                        ex = by_formula[s.formula][0]
                        if ex.get("formula"):
                            s.formula = str(ex["formula"])
                        if "energy_above_hull" in ex:
                            s.decomposition_energy = ex["energy_above_hull"]
                        props = {}
                        for key in ("bulk_modulus", "shear_modulus", "density", "piezoelectric_coefficient", "dielectric_constant", "band_gap"):
                            if key in ex and ex[key] is not None:
                                props[key] = float(ex[key])
                        if "formation_energy_per_atom" in ex and ex["formation_energy_per_atom"] is not None:
                            props["formation_energy"] = float(ex["formation_energy_per_atom"])
                        if props:
                            s.properties.update(props)
                        s.metadata.update({
                            "mp_id": ex.get("mp_id"),
                            "energy_above_hull": ex.get("energy_above_hull"),
                            "formation_energy_per_atom": ex.get("formation_energy_per_atom"),
                            "energy_per_atom": ex.get("energy_per_atom"),
                            "crystal_system": ex.get("crystal_system"),
                            "space_group": ex.get("space_group"),
                        })
                        s.is_valid = True
                        attached += 1
                    else:
                        unmatched += 1
                        s.decomposition_energy = float("inf")
                logger.info(f"  Attached energy metadata to {attached}/{len(seed_structs)} seeds from index.json")
                if unmatched:
                    logger.warning(f"  {unmatched} seeds not found in index; marked Ed=inf")

            example_pool.extend(seed_structs)
        else:
            logger.warning(f"No structures found at {seed_path}; starting with empty parent pool.")

        return example_pool
    
    def generate(self, example_pool: List[CrystalStructure]) -> List[CrystalStructure]:
        """
        Generate: Create new structures using tool-based approach

        Args:
            example_pool: Previously successful structures

        Returns:
            Newly generated structures
        """
        logger.info(f"=== Generate Phase (Iteration {self.iteration}) ===")

        # Get task description and constraints from task_config
        task_name = self.task_config.get('name', 'Materials Discovery')
        task_desc = self.task_config.get('description', 'stable materials').strip()

        # Build constraints dict from task_config
        constraints = {}
        task_constraints = self.task_config.get('constraints', {})

        # Convert task constraints to natural language format for LLM prompt
        for prop_name, prop_def in task_constraints.items():
            if prop_name == 'is_valid':
                continue  # Skip is_valid (implicit requirement)

            if not isinstance(prop_def, dict):
                continue

            # Only include enabled constraints
            if not prop_def.get('enabled', True):
                continue

            # Format constraints based on type
            if 'min' in prop_def and 'max' in prop_def:
                constraints[prop_name] = f"{prop_def['min']}-{prop_def['max']}"
            elif 'min' in prop_def:
                constraints[prop_name] = f">{prop_def['min']}"
            elif 'max' in prop_def:
                constraints[prop_name] = f"<{prop_def['max']}"
            elif 'value' in prop_def:
                constraints[prop_name] = f"={prop_def['value']}"

        logger.info(f"Task: {task_name}")
        logger.info(f"Description: {task_desc}")
        logger.info(f"Constraints: {constraints}")

        # 🔥 DYNAMIC TEMPLATE UPDATE: Extract templates from current parent pool
        if example_pool:
            logger.info("Updating templates from parent pool...")
            dynamic_templates = self.operators.register_dynamic_templates_from_pool(
                parent_pool=example_pool,
                max_templates=len(example_pool),
                max_elements=4,
                score_key='weighted_score'  # Use best structures
            )
            if dynamic_templates:
                # Update tool selector with fresh templates
                self.tool_selector.set_template_catalog(dynamic_templates)
                logger.info(f"✓ Updated {len(dynamic_templates)} dynamic templates from parent pool")
                # Log top 3 templates with task-relevant properties
                _PROP_LABEL = {
                    "band_gap": ("gap", 2), "formation_energy": ("Ef", 3),
                    "bulk_modulus": ("bulk", 1), "shear_modulus": ("shear", 1),
                    "density": ("rho", 2), "piezoelectric_coefficient": ("piezo", 2),
                    "dielectric_constant": ("dielectric", 1),
                }
                task_constraints = self.config.get("task_constraints", {}) or {}
                tracked = {
                    p for p, d in task_constraints.items()
                    if p != "is_valid" and isinstance(d, dict) and d.get("enabled", True)
                }
                for i, tmpl in enumerate(dynamic_templates[:3], 1):
                    ed_val = tmpl.get('decomposition_energy')
                    parts = [f"Ed={ed_val:.3f}" if ed_val is not None else "Ed=N/A"]
                    for prop in tracked:
                        if prop in _PROP_LABEL:
                            label, digits = _PROP_LABEL[prop]
                            val = tmpl.get(prop)
                            if val is not None:
                                parts.append(f"{label}={val:.{digits}f}")
                    logger.info(
                        f"  #{i}: {tmpl['name']} ({tmpl['reference_formula']}) - "
                        f"{', '.join(parts)}"
                    )

        # Get number of structures to generate
        n_structures = self.config.get('evolution', {}).get('children_size', 5)
        logger.info(f"Generating {n_structures} new structures...")
        self.gen_requested_current_iter = 0
        self.gen_parse_success_current_iter = 0
        self.gen_parse_failure_current_iter = 0

        # Get previous reflection/context for prompt construction
        prev_reflection = self.search_history[-1] if self.search_history else {}
        ablation_context = self._ablation_prompt_context(prev_reflection)

        # ===== GENERATE STRUCTURES (mode-switched) =====
        if self.generation_mode == 'direct_llm':
            # Ablation baseline: LLM directly outputs CIF structures
            self.direct_generator.config["task_constraints"] = self.task_config.get("constraints", {})
            self.direct_generator.config["task_description"] = task_desc
            children = self.direct_generator.generate_structures(
                example_pool=example_pool,
                strategy=self.current_plan,
                reflection=prev_reflection,
                n_structures=n_structures,
                output_dir=self.output_dir,
                iteration=self.iteration,
                all_structures=self.all_structures,
                strategies_data=self.current_strategies,
                max_iterations=self.config['evolution']['max_iterations'],
                hmrr_context=self.hier_memory.get_prompt_context(),
            )
            logger.info(f"Generated {len(children)} structures from direct LLM")
            requested = n_structures
            parse_success = len(children)
            parse_failure = max(0, requested - parse_success)

        else:
            # ===== TOOL-BASED GENERATION =====
            # Step 1: Agent decides which tools to use
            logger.info("Tool selection agent analyzing state...")
            # Provide task constraints so the tool selector can show task-relevant metrics.
            self.tool_selector.config["task_constraints"] = self.task_config.get("constraints", {})
            self.tool_selector.config["task_description"] = task_desc
            tool_actions = self.tool_selector.select_tools(
                example_pool=example_pool,
                strategy=self.current_plan,
                reflection=ablation_context['prev_reflection'],
                n_structures=n_structures,
                output_dir=self.output_dir,
                iteration=self.iteration,
                all_structures=ablation_context['all_structures'],  # 🔥 NEW: For diversity tracking
                strategies_data=ablation_context['strategies_data'],  # 🔥 NEW: For target Ed ranges
                max_iterations=self.config['evolution']['max_iterations'],  # 🔥 NEW: For context
                hmrr_context=ablation_context['hmrr_context'],
            )

            # Step 1.5 (Phase 2B): Allocate strategies to tool actions
            strategy_allocation = self._allocate_strategies_to_actions(tool_actions)

            # Step 2: Execute tool actions
            logger.info(f"Executing {len(tool_actions)} tool actions...")
            children = []
            mix_used = 0
            disabled_tools = {
                str(tool).strip().lower()
                for tool in (self.config.get("tool_selector", {}).get("disabled_tools", []) or [])
            }

            def _fallback_fill_prototype_from_parents(parent_indices: List[int]) -> CrystalStructure:
                if "fill_prototype" in disabled_tools:
                    parent = example_pool[parent_indices[0]]
                    return self.operators.mutate_structure(parent=parent, strength=0.05)
                elements = []
                for idx in parent_indices:
                    elements.extend(example_pool[idx].species)
                unique = []
                for element in elements:
                    if element not in unique:
                        unique.append(element)
                if len(unique) >= 3:
                    template = "Perovskite"
                    chosen = unique[:3]
                elif len(unique) == 2:
                    template = "Rock-salt"
                    chosen = unique[:2]
                else:
                    template = "Diamond"
                    chosen = unique[:1] if unique else ["Si"]
                return self.operators.fill_prototype(
                    template_name=template,
                    elements=chosen
                )

            for i, action in enumerate(tool_actions):
                tool_name = action['tool']
                logger.info(f"  Action {i+1}/{len(tool_actions)}: {tool_name}")

                try:
                    if str(tool_name).strip().lower() in disabled_tools:
                        logger.warning(f"Tool '{tool_name}' is disabled; replacing with mutate")
                        if example_pool:
                            parent = example_pool[action.get('parent', i % len(example_pool))]
                            child = self.operators.mutate_structure(parent=parent, strength=0.05)
                            children.append(child)
                        else:
                            logger.error("No parents available to replace disabled tool action")
                        continue

                    if tool_name == 'substitute':
                        # Element substitution
                        parent = example_pool[action['parent']]
                        child = self.operators.substitute_element(
                            parent=parent,
                            old_element=action['old_element'],
                            new_element=action['new_element']
                        )
                        children.append(child)

                    elif tool_name == 'mutate':
                        # Structure mutation
                        parent = example_pool[action['parent']]
                        child = self.operators.mutate_structure(
                            parent=parent,
                            strength=action['strength']
                        )
                        children.append(child)

                    elif tool_name == 'mix':
                        parent1_idx = action['parent1']
                        parent2_idx = action['parent2']
                        assigned_strategy = strategy_allocation[i] if i < len(strategy_allocation) else None

                        if mix_used >= 1:
                            if "fill_prototype" in disabled_tools:
                                logger.warning("Mix skipped: limit reached and fill_prototype disabled; using mutate instead")
                                parent = example_pool[parent1_idx]
                                child = self.operators.mutate_structure(parent=parent, strength=0.05)
                            else:
                                logger.warning("Mix skipped: per-iteration limit reached; using fill_prototype instead")
                                child = _fallback_fill_prototype_from_parents([parent1_idx, parent2_idx])
                        else:
                            parent1 = example_pool[parent1_idx]
                            parent2 = example_pool[parent2_idx]
                            min_distance_factor = self.config.get('validation', {}).get('min_distance_factor', 0.75)
                            child = self.operators.mix_structures(
                                parent1=parent1,
                                parent2=parent2,
                                ratio=action.get('ratio', 0.5),
                                min_distance_factor=min_distance_factor
                            )
                            mix_used += 1

                        if assigned_strategy:
                            child.metadata['strategy_used'] = assigned_strategy['strategy_name']
                            child.metadata['strategy_type'] = assigned_strategy['strategy_type']

                        children.append(child)

                    elif tool_name == 'crossover':
                        logger.warning("Crossover is disabled; skipping action")
                        continue

                    elif tool_name == 'dope':
                        # Dope structure with trace impurities
                        parent = example_pool[action['parent']]
                        # Use optional parameters if provided
                        kwargs = {}
                        if 'dopant' in action:
                            kwargs['dopant'] = action['dopant']
                        if 'concentration' in action:
                            kwargs['concentration'] = action['concentration']
                        if 'host_element' in action:
                            kwargs['host_element'] = action['host_element']
                        child = self.operators.dope_structure(
                            parent=parent,
                            **kwargs
                        )
                        children.append(child)

                    elif tool_name == 'fill_prototype':
                        assigned_strategy = strategy_allocation[i] if i < len(strategy_allocation) else None

                        template = action.get('template') or action.get('template_name')
                        elements = action.get('elements') or action.get('elements_list')
                        if not template or not elements:
                            raise ValueError("fill_prototype requires template and elements")

                        child = self.operators.fill_prototype(
                            template_name=template,
                            elements=elements
                        )

                        if assigned_strategy:
                            child.metadata['strategy_used'] = assigned_strategy['strategy_name']
                            child.metadata['strategy_type'] = assigned_strategy['strategy_type']

                        children.append(child)

                    elif tool_name == 'new':
                        logger.warning("Tool 'new' is deprecated; use fill_prototype instead")
                        continue

                except Exception as e:
                    logger.error(f"  Tool execution failed: {e}")
                    continue

            logger.info(f"Generated {len(children)} structures from tools")
            requested = len(children)
            parse_success = len(children)
            parse_failure = 0

        self.gen_requested_current_iter = requested
        self.gen_parse_success_current_iter = parse_success
        self.gen_parse_failure_current_iter = parse_failure
        self.gen_requested_total += requested
        self.gen_parse_success_total += parse_success
        self.gen_parse_failure_total += parse_failure

        # Step 3: Evaluate structures
        if children:
            logger.info("Evaluating structures...")
            from src.core.evaluator import StructureEvaluator
            # Pass task constraints to evaluator so it knows which properties to calculate
            task_constraints = self.task_config.get('constraints', {})
            evaluator = StructureEvaluator(self.config, task_constraints=task_constraints)

            evaluated = []
            for i, child in enumerate(children):
                logger.info(f"  Evaluating {i+1}/{len(children)}: {child.formula}")
                skip_reason = self._get_pre_eval_skip_reason(child)
                if skip_reason:
                    logger.warning(f"  Skipping {child.formula}: {skip_reason}")
                    child.is_valid = False
                    child.decomposition_energy = float('inf')
                    child.metadata['pre_eval_filtered'] = True
                    child.metadata['pre_eval_filter_reason'] = skip_reason
                    evaluated.append(child)
                    if self.hmrr_runtime_enabled:
                        self.hier_memory.add_micro_event(
                            iteration=self.iteration,
                            structure=child,
                            constraint_hit=False,
                            trigger="pre_eval_filtered",
                        )
                    continue
                try:
                    evaluated_child = evaluator.evaluate(child)
                    evaluated.append(evaluated_child)
                    hit = self._check_constraints(evaluated_child) if evaluated_child.is_valid else False
                    if self.hmrr_runtime_enabled:
                        self.hier_memory.add_micro_event(
                            iteration=self.iteration,
                            structure=evaluated_child,
                            constraint_hit=hit,
                            trigger="evaluation_result",
                        )
                except Exception as e:
                    logger.warning(f"  Evaluation failed for {child.formula}: {e}")
                    child.is_valid = False
                    evaluated.append(child)
                    if self.hmrr_runtime_enabled:
                        self.hier_memory.add_micro_event(
                            iteration=self.iteration,
                            structure=child,
                            constraint_hit=False,
                            trigger="evaluation_exception",
                        )

            children = evaluated

            # Track all evaluated structures for deduplication feedback
            for child in children:
                self.structure_generator.add_generated_structure(child)

        logger.info(f"Generated {len(children)} structures in this iteration")

        return children

    def _get_pre_eval_skip_reason(self, structure: CrystalStructure) -> Optional[str]:
        """
        Return a skip reason when structure should bypass expensive evaluation.

        Current policy: block single-element O/F candidates (configurable) because
        they frequently trigger unstable/high-memory Ed recomputation workloads.
        """
        if not self.elemental_skip_enabled or not self.elemental_skip_symbols:
            return None

        unique_elems = {
            str(elem).strip() for elem in structure.species if str(elem).strip()
        }
        if len(unique_elems) != 1:
            return None

        only_elem = next(iter(unique_elems))
        if only_elem not in self.elemental_skip_symbols:
            return None

        return (
            f"single-element {only_elem} candidate blocked by "
            "evaluator.elemental_skip_filter"
        )
    
    def reflect(self, children: List[CrystalStructure], example_pool: List[CrystalStructure] = None) -> Dict:
        """
        Reflect: Analyze results

        Args:
            children: Generated child structures
            example_pool: Parent pool structures (for baseline comparison)

        Returns:
            Reflection dictionary with statistics
        """
        logger.info("=== Reflect Phase ===")

        # Add children to global history
        self.all_structures.extend(children)

        # Calculate statistics
        valid_children = [c for c in children if c.is_valid]
        metastable_children = [c for c in valid_children
                               if c.decomposition_energy is not None
                               and c.decomposition_energy <= self.metastability_threshold]
        stable_children = [c for c in metastable_children
                          if c.decomposition_energy < 0]
        parse_requested = self.gen_requested_current_iter
        parse_success = self.gen_parse_success_current_iter or len(children)
        parse_failure = self.gen_parse_failure_current_iter
        if parse_requested <= 0:
            parse_requested = parse_success
            parse_failure = 0

        reflection = {
            'iteration': self.iteration,
            'total_generated': len(children),
            'parse_requested': parse_requested,
            'parse_success': parse_success,
            'parse_failure': parse_failure,
            'parse_success_rate': 100 * parse_success / parse_requested if parse_requested else 0,
            'valid_count': len(valid_children),
            'metastable_count': len(metastable_children),
            'stable_count': len(stable_children),
            'valid_rate': 100 * len(valid_children) / parse_success if parse_success else 0,
            'valid_rate_parsed': 100 * len(valid_children) / parse_success if parse_success else 0,
            'valid_rate_strict': 100 * len(valid_children) / parse_requested if parse_requested else 0,
            'metastable_rate': 100 * len(metastable_children) / len(children) if children else 0,
            'stable_rate': 100 * len(stable_children) / len(children) if children else 0
        }
        
        # Energy statistics
        # Prefer metastable set; if none, fall back to all valid energies to avoid inf
        energies_metastable = [c.decomposition_energy for c in metastable_children]
        energies_all_valid = [c.decomposition_energy for c in valid_children
                              if c.decomposition_energy is not None]

        if energies_metastable:
            energies = energies_metastable
        elif energies_all_valid:
            energies = energies_all_valid
        else:
            energies = []

        if energies:
            reflection['best_ed'] = min(energies)
            reflection['avg_ed'] = np.mean(energies)
            reflection['worst_ed'] = max(energies)
        else:
            reflection['best_ed'] = float('inf')
            reflection['avg_ed'] = float('inf')
            reflection['worst_ed'] = float('inf')
        
        # 🔥 NEW: Weighted Score statistics (multi-objective)
        weighted_scores = [getattr(c, 'weighted_score', None) for c in valid_children]
        weighted_scores = [s for s in weighted_scores if s is not None]

        if weighted_scores:
            reflection['best_weighted_score'] = max(weighted_scores)  # Higher is better
            reflection['avg_weighted_score'] = np.mean(weighted_scores)
            reflection['worst_weighted_score'] = min(weighted_scores)
        else:
            reflection['best_weighted_score'] = 0.0
            reflection['avg_weighted_score'] = 0.0
            reflection['worst_weighted_score'] = 0.0

        # Diversity
        reflection['diversity'] = self.evolution.calculate_diversity(valid_children)

        # 🔥 NEW: Tool performance statistics
        tool_performance = self._calculate_tool_statistics(children)
        reflection['tool_statistics'] = tool_performance

        # Hit Rate (structures satisfying ALL task constraints)
        if self.constraint_checker:
            satisfied_children = self.constraint_checker.filter_satisfied(children)
            reflection['hit_count'] = len(satisfied_children)
            reflection['hit_rate'] = 100 * len(satisfied_children) / parse_success if parse_success else 0
            reflection['hit_rate_parsed'] = 100 * len(satisfied_children) / parse_success if parse_success else 0
            reflection['hit_rate_strict'] = 100 * len(satisfied_children) / parse_requested if parse_requested else 0
        else:
            reflection['hit_count'] = 0
            reflection['hit_rate'] = 0
            reflection['hit_rate_parsed'] = 0
            reflection['hit_rate_strict'] = 0

        # Global formula statistics (for diversity tracking)
        from collections import Counter
        all_formulas = [s.formula for s in self.all_structures if hasattr(s, 'formula')]
        formula_counts = Counter(all_formulas)
        reflection['global_formula_stats'] = {
            'total_structures': len(self.all_structures),
            'unique_formulas': len(formula_counts),
            'formula_counts': dict(formula_counts.most_common()),  # All formulas sorted by frequency
            'most_repeated_formulas': dict(formula_counts.most_common(10))  # Top 10 for prompt
        }

        # Log complete reflection summary
        logger.info("="*60)
        logger.info("ITERATION SUMMARY:")
        logger.info("="*60)
        logger.info(f"Generated: {reflection['total_generated']} structures")
        if self.generation_mode == 'direct_llm':
            logger.info(
                f"Parse success: {reflection['parse_success']} / {reflection['parse_requested']} "
                f"({reflection['parse_success_rate']:.1f}%)"
            )
        logger.info(f"Valid: {reflection['valid_count']} ({reflection['valid_rate']:.1f}%)")
        if self.generation_mode == 'direct_llm':
            logger.info(
                f"Valid (strict): {reflection['valid_count']} / {reflection['parse_requested']} "
                f"({reflection['valid_rate_strict']:.1f}%)"
            )
        logger.info(f"Metastable (Ed≤{self.metastability_threshold}): {reflection['metastable_count']} ({reflection['metastable_rate']:.1f}%)")
        logger.info(f"Stable (Ed<0): {reflection['stable_count']} ({reflection['stable_rate']:.1f}%)")
        if self.constraint_checker:
            logger.info(f"Hit (satisfy ALL constraints): {reflection['hit_count']} ({reflection['hit_rate']:.1f}%)")
            if self.generation_mode == 'direct_llm':
                logger.info(
                    f"Hit (strict): {reflection['hit_count']} / {reflection['parse_requested']} "
                    f"({reflection['hit_rate_strict']:.1f}%)"
                )
        if energies:
            logger.info(f"Ed range: {reflection['best_ed']:.3f} - {reflection['worst_ed']:.3f} eV/atom")
            logger.info(f"Ed average: {reflection['avg_ed']:.3f} eV/atom")
        if weighted_scores:
            logger.info(f"Weighted Score range: {reflection['worst_weighted_score']:.4f} - {reflection['best_weighted_score']:.4f}")
            logger.info(f"Weighted Score average: {reflection['avg_weighted_score']:.4f}")
        logger.info(f"Diversity: {reflection['diversity']:.2f}")

        # Log global formula statistics
        global_stats = reflection['global_formula_stats']
        logger.info(f"Global diversity: {global_stats['unique_formulas']} unique formulas out of {global_stats['total_structures']} total structures")
        if global_stats['most_repeated_formulas']:
            logger.info("Most repeated formulas:")
            for formula, count in list(global_stats['most_repeated_formulas'].items())[:5]:
                logger.info(f"  {formula}: {count}x")
        logger.info("="*60)

        llm_analysis = self._generate_llm_reflection(children, reflection, example_pool)
        reflection['llm_analysis'] = llm_analysis

        # NEW (Phase 2): Individual structure reflections
        individual_reflections = self._generate_individual_reflections(children, reflection)
        reflection['individual_reflections'] = individual_reflections
        if self.hmrr_runtime_enabled:
            try:
                self.hier_memory.update_after_reflection(
                    self.iteration,
                    reflection,
                    output_dir=self.output_dir,
                )
            except Exception as exc:
                logger.error(
                    "HMRR reflection failed at iteration %s (strict_mode=%s): %s",
                    self.iteration,
                    self.hier_memory.strict_mode,
                    exc,
                )
                if self.output_dir:
                    logger.error(
                        "Inspect HMRR traces under %s/iteration_%s/reflection/",
                        self.output_dir,
                        self.iteration,
                    )
                raise
            reflection['hmrr_metrics'] = self.hier_memory.get_metrics()
            reflection['hmrr_context'] = self.hier_memory.get_prompt_context()
            self.hier_memory.save_snapshot(self.output_dir, self.iteration)
        else:
            reflection['hmrr_metrics'] = self._disabled_hmrr_metrics()
            reflection['hmrr_context'] = self._disabled_hmrr_context()

        # Save reflection to file (after adding llm_analysis and individual_reflections)
        if self.output_dir:
            from pathlib import Path
            import json
            reflection_file = Path(self.output_dir) / f"iteration_{self.iteration}" / "reflection.json"
            reflection_file.parent.mkdir(parents=True, exist_ok=True)
            # Sanitize numpy/torch scalars before dumping
            def _safe(x):
                import numpy as np
                if isinstance(x, (np.floating, np.float32, np.float64)):
                    return float(x)
                if hasattr(x, "item"):  # numpy scalar or torch tensor
                    try:
                        return float(x.item())
                    except Exception:
                        return x
                return x

            def _sanitize(obj):
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_sanitize(v) for v in obj]
                return _safe(obj)

            with open(reflection_file, 'w') as f:
                json.dump(_sanitize(reflection), f, indent=2)
            logger.debug(f"Reflection saved to {reflection_file}")

        # Store in history
        self.search_history.append(reflection)

        return reflection

    def _calculate_tool_statistics(self, children: List[CrystalStructure]) -> Dict:
        """
        Calculate performance statistics for each tool type

        Args:
            children: Generated structures from current iteration

        Returns:
            Dictionary with per-tool statistics and rankings
        """
        # Update cumulative tool statistics
        for child in children:
            tool = child.metadata.get('source', 'unknown')
            if tool not in self.tool_stats:
                continue

            self.tool_stats[tool]['total'] += 1

            if child.is_valid:
                self.tool_stats[tool]['valid'] += 1

            if child.decomposition_energy is not None:
                self.tool_stats[tool]['energies'].append(child.decomposition_energy)

                if child.decomposition_energy <= self.metastability_threshold:
                    self.tool_stats[tool]['metastable'] += 1
            score = self._compute_weighted_score(child)
            self.tool_stats[tool]['scores'].append(score)
            if self._check_constraints(child):
                self.tool_stats[tool]['hits'] += 1

        # Calculate rates and averages
        tool_performance = {}
        for tool, stats in self.tool_stats.items():
            total = stats['total']
            if total == 0:
                tool_performance[tool] = {
                    'total': 0,
                    'hit_count': 0,
                    'hit_rate': 0.0,
                    'valid_rate': 0.0,
                    'metastable_rate': 0.0,
                    'avg_weighted_score': 0.0,
                    'best_weighted_score': 0.0,
                    'worst_weighted_score': 0.0,
                    'avg_ed': float('inf'),
                    'min_ed': float('inf')
                }
                continue

            valid_rate = 100 * stats['valid'] / total
            metastable_rate = 100 * stats['metastable'] / total
            hit_rate = 100 * stats['hits'] / total

            energies = stats['energies']
            # Keep Ed=inf as sentinel in raw records, but exclude non-finite
            # values from aggregate statistics to avoid polluting avg/min.
            finite_energies = [
                ed for ed in energies
                if ed is not None and np.isfinite(ed)
            ]
            avg_ed = np.mean(finite_energies) if finite_energies else float('inf')
            min_ed = min(finite_energies) if finite_energies else float('inf')
            scores = stats['scores']
            avg_score = float(np.mean(scores)) if scores else 0.0
            best_score = max(scores) if scores else 0.0
            worst_score = min(scores) if scores else 0.0

            tool_performance[tool] = {
                'total': total,
                'valid': stats['valid'],
                'metastable': stats['metastable'],
                'hit_count': stats['hits'],
                'valid_rate': valid_rate,
                'metastable_rate': metastable_rate,
                'hit_rate': hit_rate,
                'avg_weighted_score': avg_score,
                'best_weighted_score': best_score,
                'worst_weighted_score': worst_score,
                'avg_ed': avg_ed,
                'min_ed': min_ed
            }

        # Rank tools by primary task metric
        constraints = self.task_config.get('constraints', {}) if hasattr(self, "task_config") else {}
        has_target_constraints = False
        for name, cfg in constraints.items():
            if name == 'is_valid':
                continue
            if isinstance(cfg, dict) and cfg.get('enabled', True):
                has_target_constraints = True
                break

        if has_target_constraints:
            rank_metric = 'hit_rate'
            rank_label = 'hit rate'
            ranked_tools = sorted(
                [(tool, perf) for tool, perf in tool_performance.items() if perf['total'] > 0],
                key=lambda x: (x[1]['hit_rate'], x[1]['avg_weighted_score'], -x[1]['avg_ed']),
                reverse=True
            )
        else:
            rank_metric = 'metastable_rate'
            rank_label = 'metastable rate'
            ranked_tools = sorted(
                [(tool, perf) for tool, perf in tool_performance.items() if perf['total'] > 0],
                key=lambda x: (x[1]['metastable_rate'], x[1]['avg_weighted_score'], -x[1]['avg_ed']),
                reverse=True
            )

        # Log tool performance
        logger.info("="*60)
        logger.info("TOOL PERFORMANCE STATISTICS:")
        logger.info("="*60)
        for tool, perf in ranked_tools:
            logger.info(
                f"  {tool:12s}: {perf['total']:3d} used | "
                f"Valid: {perf['valid_rate']:5.1f}% | "
                f"Hit: {perf['hit_rate']:5.1f}% | "
                f"Metastable: {perf['metastable_rate']:5.1f}% | "
                f"Avg Score: {perf['avg_weighted_score']:6.3f} | "
                f"Best Score: {perf['best_weighted_score']:6.3f} | "
                f"Avg Ed: {perf['avg_ed']:6.3f} | "
                f"Min Ed: {perf['min_ed']:6.3f}"
            )
        logger.info("="*60)

        return {
            'per_tool': tool_performance,
            'ranked': [(tool, perf) for tool, perf in ranked_tools],
            'rank_metric': rank_metric,
            'rank_label': rank_label
        }

    def curate(self, reflection: Dict, example_pool: List[CrystalStructure]) -> Dict:
        """
        Curate: Evolve strategy using hypothesis agent (Phase 2B: Multi-strategy)

        Args:
            reflection: Performance reflection with individual_reflections
            example_pool: Current example pool (parent structures)

        Returns:
            Strategies dictionary with multiple strategies
        """
        logger.info("=== Curate Phase ===")

        # Vanilla ReAct and memory-less baselines should not receive
        # synthesized multi-strategy planning from the curate phase.
        if self.effective_ablation_mode in {"vanilla_react", "memory_less", "flat_memory"}:
            logger.info(
                "Skipping multi-strategy curation for ablation_mode=%s",
                self.effective_ablation_mode,
            )
            self.current_strategies = None
            return {
                "enabled": False,
                "mode": self.effective_ablation_mode,
                "reason": "multi-strategy curation disabled for baseline mode",
            }

        # Phase 2B: Generate multiple strategies using LLM
        hmrr_context = (
            self.hier_memory.get_prompt_context()
            if self.hmrr_runtime_enabled
            else self._disabled_hmrr_context()
        )
        strategies_data = self.hypothesis_agent.generate_strategies(
            reflection,
            hmrr_context=hmrr_context,
        )

        # Save strategies to file
        if self.output_dir:
            from pathlib import Path
            import json
            strategy_file = Path(self.output_dir) / f"iteration_{self.iteration}" / "strategies.json"
            strategy_file.parent.mkdir(parents=True, exist_ok=True)
            with open(strategy_file, 'w') as f:
                json.dump(strategies_data, f, indent=2)
            logger.debug(f"Strategies saved to {strategy_file}")

        # Store strategies for generate phase
        self.current_strategies = strategies_data

        return strategies_data

    def finalize_hier_memory(self) -> None:
        """Persist macro memory when HMRR is enabled."""
        if self.hmrr_runtime_enabled:
            self.hier_memory.finalize()

    def reset_step_memory(self) -> None:
        """Reset transient strategy memory for memory-less ablations."""
        self.current_plan = None
        self.current_strategies = None

    def get_hmrr_metrics(self) -> Dict:
        """Expose HMRR metrics for run-level summaries."""
        if self.hmrr_runtime_enabled:
            return self.hier_memory.get_metrics()
        return self._disabled_hmrr_metrics()

    def get_generation_metrics(self) -> Dict:
        """Expose generation/parse accounting for run-level summaries."""
        requested = self.gen_requested_total
        parse_success = self.gen_parse_success_total
        parse_failure = self.gen_parse_failure_total
        return {
            'requested_total': requested,
            'parse_success_total': parse_success,
            'parse_failure_total': parse_failure,
            'parse_success_rate': (parse_success / requested) if requested > 0 else 0.0,
        }

    def _disabled_hmrr_context(self) -> Dict:
        return {"enabled": False, "macro_rules": [], "meso_tips": [], "micro_events": []}

    def _disabled_hmrr_metrics(self) -> Dict:
        return {
            "enabled": False,
            "micro_events_count": 0,
            "meso_rules_count": 0,
            "macro_rules_count": 0,
            "meso_trigger_count": 0,
            "macro_trigger_count": 0,
            "macro_rules_loaded_count": 0,
            "macro_rules_generated_count": 0,
            "stagnation_count": 0,
            "stagnation_patience": int(
                (((self.config.get("reflection", {}) or {}).get("hierarchical", {}) or {})
                 .get("meso", {}) or {}).get("stagnation_patience", 3)
            ),
            "macro_trigger_every_n_iters": int(
                (((self.config.get("reflection", {}) or {}).get("hierarchical", {}) or {})
                 .get("macro", {}) or {}).get("trigger_every_n_iters", 10)
            ),
            "macro_persist_across_runs": False,
            "llm_reflection_enabled": False,
            "llm_reflection_success_count": 0,
            "llm_reflection_failure_count": 0,
            "strict_mode_enabled": False,
            "meso_trigger_policy": "disabled",
            "retry_max_attempts": 0,
            "retry_backoff_seconds": 0.0,
            "trace_io": False,
            "meso_llm_attempt_count": 0,
            "meso_llm_success_count": 0,
            "meso_llm_failure_count": 0,
            "macro_llm_attempt_count": 0,
            "macro_llm_success_count": 0,
            "macro_llm_failure_count": 0,
        }

    def _ablation_prompt_context(self, prev_reflection: Dict) -> Dict:
        """
        Build prompt-time context according to ablation_mode.

        crystal_forge: full context
        vanilla_react: keep only recent reflection summary, no HMRR/strategy memory
        flat_memory: raw long-context only, no synthesized history or HMRR
        memory_less: only current observation, no historical evidence
        """
        mode = self.effective_ablation_mode
        if mode in {'memory_less', 'flat_memory'}:
            return {
                'prev_reflection': {},
                'strategies_data': None,
                'all_structures': None,
                'hmrr_context': self._disabled_hmrr_context(),
            }
        if mode == 'vanilla_react':
            return {
                'prev_reflection': prev_reflection,
                'strategies_data': None,
                'all_structures': None,
                'hmrr_context': self._disabled_hmrr_context(),
            }
        return {
            'prev_reflection': prev_reflection,
            'strategies_data': self.current_strategies,
            'all_structures': self.all_structures,
            'hmrr_context': self.hier_memory.get_prompt_context(),
        }

    def _format_strategy(self, strategy_dict: Dict) -> str:
        """
        Format strategy dictionary into readable text for LLM prompt

        Args:
            strategy_dict: Strategy from hypothesis agent

        Returns:
            Formatted strategy text
        """
        parts = [
            "REASONING:",
            f"{strategy_dict['thought']}",
            "",
            "RECOMMENDED ACTION:",
            f"{strategy_dict['action']}",
            "",
            "EXPECTED OUTCOME:",
            f"{strategy_dict['expected_outcome']}"
        ]

        return "\n".join(parts)
    
    def _allocate_strategies_to_actions(self, tool_actions: List[Dict]) -> List[Optional[Dict]]:
        """
        Allocate strategies to tool actions (Phase 2B)

        Args:
            tool_actions: List of tool actions from tool_selector

        Returns:
            List of strategy dicts (one per action), None if no strategies available
        """
        # If no multi-strategy available, return None for all
        if not self.current_strategies or 'strategies' not in self.current_strategies:
            logger.debug("No multi-strategy available, using legacy mode")
            return [None] * len(tool_actions)

        strategies_list = self.current_strategies['strategies']
        n_actions = len(tool_actions)

        # Normalize weights
        total_weight = sum(s['allocation_weight'] for s in strategies_list)
        if total_weight == 0:
            logger.warning("All strategy weights are 0, using equal allocation")
            normalized_weights = [1.0 / len(strategies_list)] * len(strategies_list)
        else:
            normalized_weights = [s['allocation_weight'] / total_weight for s in strategies_list]

        # Allocate based on weights
        allocation = []
        for strat, weight in zip(strategies_list, normalized_weights):
            count = max(1, round(weight * n_actions))  # At least 1
            allocation.extend([strat] * count)

        # Adjust if over
        while len(allocation) > n_actions:
            # Remove from lowest priority (only consider strategies still in allocation)
            # Count occurrences of each strategy in allocation
            from collections import Counter
            allocation_counts = Counter(s['strategy_name'] for s in allocation)

            # Find the strategy with lowest weight that's still in allocation
            min_idx = None
            min_weight = float('inf')
            for i, strat in enumerate(strategies_list):
                if allocation_counts[strat['strategy_name']] > 0:
                    if strat['allocation_weight'] < min_weight:
                        min_weight = strat['allocation_weight']
                        min_idx = i

            # Remove one instance of this strategy
            if min_idx is not None:
                for i in range(len(allocation)-1, -1, -1):
                    if allocation[i]['strategy_name'] == strategies_list[min_idx]['strategy_name']:
                        allocation.pop(i)
                        break
            else:
                # Safety: if no strategy found, just pop the last one
                allocation.pop()
                break

        # Adjust if under
        while len(allocation) < n_actions:
            # Add to highest priority
            max_idx = max(range(len(strategies_list)),
                         key=lambda i: strategies_list[i]['allocation_weight'])
            allocation.append(strategies_list[max_idx])

        # Log allocation
        logger.info("="*60)
        logger.info("STRATEGY ALLOCATION TO ACTIONS:")
        logger.info("="*60)
        from collections import Counter
        strategy_counts = Counter(s['strategy_name'] for s in allocation)
        for strat in strategies_list:
            count = strategy_counts[strat['strategy_name']]
            percentage = 100 * count / n_actions
            logger.info(f"{strat['strategy_name']}: {count} actions ({percentage:.1f}%)")
        logger.info("="*60)

        return allocation

    def should_terminate(self, reflection: Dict) -> bool:
        """
        Check termination conditions

        Args:
            reflection: Current iteration reflection

        Returns:
            True if search should terminate
        """
        # Budget exhausted
        if self.budget_remaining <= 0:
            logger.info("Budget exhausted")
            return True

        # Target achieved
        # Support both old and new objective formats
        obj_config = self.config.get('objective', {})
        if 'targets' in obj_config:
            target_ed = obj_config['targets'].get('decomposition_energy', 0.0)
        else:
            target_ed = obj_config.get('target_ed', 0.0)

        if reflection['best_ed'] < target_ed:
            logger.info(f"Target achieved: {reflection['best_ed']:.3f} < {target_ed}")
            return True
        
        # Max iterations reached
        max_iter = self.config['evolution']['max_iterations']
        if self.iteration >= max_iter:
            logger.info(f"Max iterations reached: {self.iteration} >= {max_iter}")
            return True
        
        return False
    
    def save_checkpoint(self, parent_pool: List[CrystalStructure], output_dir: str):
        """Save search checkpoint"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save structures to a tidy checkpoints folder: structures/checkpoints/iter_<n>/structure_<i>.json
        checkpoint_struct_dir = output_path / "structures" / "checkpoints" / f"iter_{self.iteration}"
        checkpoint_struct_dir.mkdir(parents=True, exist_ok=True)
        for i, struct in enumerate(parent_pool):
            struct.save(str(checkpoint_struct_dir / f"structure_{i}.json"))

        # Save history
        # ---- Fix numpy.float32 serialization ----
        def _safe(x):
            import numpy as np
            if isinstance(x, (np.floating, np.float32, np.float64)):
                return float(x)
            if hasattr(x, "item"):  # numpy scalar or torch tensor
                return float(x.item())
            return x

        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return _safe(obj)

        clean_history = _sanitize(self.search_history)

        # Add metadata to history
        history_with_metadata = {
            'metadata': {
                'experiment_type': 'matagent_search',
                'iteration': self.iteration,
                'timestamp': datetime.now().isoformat(),
                'hostname': platform.node(),
                'python_version': platform.python_version(),
                'config': {
                    'population_size': self.config['evolution']['population_size'],
                    'parent_size': self.config['evolution']['parent_size'],
                    'children_size': self.config['evolution']['children_size'],
                    'max_iterations': self.config['evolution']['max_iterations'],
                    'llm_provider': self.config['llm']['provider'],
                    'llm_model': self.config['llm']['model'],
                    'llm_temperature': self.config['llm']['temperature'],
                    'evaluator_backend': self.config['evaluator']['backend'],
                    'evaluator_model': self.config['evaluator'].get('model_name', 'N/A'),
                    'relax_fmax': self.config['evaluator']['relax']['fmax'],
                    'relax_steps': self.config['evaluator']['relax']['steps'],
                    'energy_cutoff': self.config['evaluator']['energy_cutoff'],
                    'metastability_threshold': self.metastability_threshold,
                    'failure_threshold': self.failure_threshold
                },
                'pool_stats': {
                    'pool_size': len(parent_pool),
                    'valid_structures': sum(1 for s in parent_pool if s.is_valid),
                    'metastable_structures': sum(
                        1
                        for s in parent_pool
                        if getattr(s, 'decomposition_energy', None) is not None
                        and s.decomposition_energy <= self.metastability_threshold
                    )
                }
            },
            'history': clean_history
        }

        # Save history under a dedicated history folder
        history_dir = output_path / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        with open(history_dir / f"history_{self.iteration}.json", 'w') as f:
            json.dump(history_with_metadata, f, indent=2)
        # ---- End fix ----


        logger.info(f"Checkpoint saved to {output_dir}")

    def _compute_weighted_score(self, structure: CrystalStructure) -> float:
        """
        Calculate weighted score for a structure (consistent with evolution.select)

        Args:
            structure: Structure to score

        Returns:
            Weighted score in [0, 1]
        """
        obj_cfg = self.config.get('objective', {})
        targets = obj_cfg.get('targets', {})
        weights = obj_cfg.get('weights', {})

        if not targets or not weights:
            # Fallback: use Ed if no multi-objective config
            ed = structure.decomposition_energy
            if ed is None:
                return 0.0
            return max(0.0, 1.0 - ed / self.metastability_threshold)

        # Resolve property weights (same logic as evolution.py)
        from src.core.evolution import EvolutionEngine
        engine = EvolutionEngine(self.config)
        weight_map = engine._resolve_property_weights(targets, weights)

        # Normalize weights
        weight_sum = sum(weight_map.values())
        if weight_sum <= 0:
            return 0.0
        norm_weights = {k: v/weight_sum for k, v in weight_map.items() if v > 0}

        # Calculate weighted score
        total = 0.0
        default_ed_scale = self.metastability_threshold
        scales = obj_cfg.get('scales', {}) or {}

        for prop, target in targets.items():
            weight = norm_weights.get(prop, 0.0)
            if weight <= 0:
                continue

            # Get property value
            if prop in structure.properties:
                value = structure.properties.get(prop)
            elif hasattr(structure, prop):
                value = getattr(structure, prop)
            else:
                continue

            if value is None:
                continue

            # Score property (reuse engine's logic)
            score = engine._score_property(
                prop, value, target, scales, default_ed_scale
            )
            total += weight * score

        return total

    def _select_by_weighted_score(
        self,
        children: List[CrystalStructure],
        reflection_config: Dict
    ) -> List[tuple]:
        """
        Select structures for reflection based on weighted multi-objective score.

        Selects top-K by weighted score + bottom-M for failure analysis.

        Args:
            children: All generated structures
            reflection_config: Reflection configuration

        Returns:
            List of (structure, category, score) tuples
        """
        # Filter valid structures with Ed
        valid_structures = [
            c for c in children
            if c.is_valid and c.decomposition_energy is not None
        ]

        if not valid_structures:
            logger.warning("No valid structures for weighted score reflection")
            return []

        # Compute weighted scores for all structures
        scored_structures = []
        for struct in valid_structures:
            score = self._compute_weighted_score(struct)
            scored_structures.append((struct, score))

        # Sort by score (descending - higher score is better)
        scored_structures.sort(key=lambda x: x[1], reverse=True)

        # Get configuration
        analyze_count = reflection_config.get('analyze_count', 5)
        top_count = max(3, analyze_count - 2)  # At least 3 top structures
        bottom_count = min(2, analyze_count - top_count)  # At least 2 bottom structures

        # Select top and bottom structures
        selected = []

        # Top structures (high weighted score)
        for i in range(min(top_count, len(scored_structures))):
            struct, score = scored_structures[i]
            selected.append((struct, "TOP", score))

        # Bottom structures (low weighted score - to learn from failures)
        if len(scored_structures) > top_count:
            for i in range(max(0, len(scored_structures) - bottom_count), len(scored_structures)):
                struct, score = scored_structures[i]
                selected.append((struct, "BOTTOM", score))

        logger.info(f"Selected {len(selected)} structures by weighted score:")
        logger.info(f"  - Top {min(top_count, len(scored_structures))} by weighted score (TOP rank)")
        logger.info(f"  - Bottom {min(bottom_count, max(0, len(scored_structures) - top_count))} by weighted score (BOTTOM rank)")

        return selected

    def _select_by_legacy_mode(self, children: List[CrystalStructure]) -> List[tuple]:
        """
        Select structures using legacy Ed threshold + constraint satisfaction.

        Legacy categorization:
        - PERFECT: Ed ≤ threshold AND satisfies constraints
        - PARTIAL_SUCCESS: Ed > threshold BUT satisfies constraints
        - STABLE_BUT_WEAK: Ed ≤ threshold BUT violates constraints
        - FAILED: Ed > threshold AND violates constraints (or invalid)

        Args:
            children: All generated structures

        Returns:
            List of (structure, category, Ed) tuples
        """
        # Get threshold from config
        ed_threshold = self.metastability_threshold  # e.g., 0.1 eV/atom

        # Filter valid structures
        valid_structures = [
            c for c in children
            if c.is_valid and c.decomposition_energy is not None
        ]

        if not valid_structures:
            logger.warning("No valid structures for legacy mode reflection")
            return []

        # Categorize structures
        perfect = []
        partial_success = []
        stable_but_weak = []
        failed = []

        for struct in valid_structures:
            ed = struct.decomposition_energy
            satisfies_constraints = self._check_constraints(struct)

            if ed <= ed_threshold and satisfies_constraints:
                perfect.append(struct)
            elif ed > ed_threshold and satisfies_constraints:
                partial_success.append(struct)
            elif ed <= ed_threshold and not satisfies_constraints:
                stable_but_weak.append(struct)
            else:
                failed.append(struct)

        # Sort each category by Ed (ascending - lower is better)
        perfect.sort(key=lambda s: s.decomposition_energy)
        partial_success.sort(key=lambda s: s.decomposition_energy)
        stable_but_weak.sort(key=lambda s: s.decomposition_energy)
        failed.sort(key=lambda s: s.decomposition_energy)

        # Select structures to analyze
        # Strategy: top-2 PERFECT + top-1 PARTIAL_SUCCESS + bottom-2 FAILED
        selected = []

        # Top PERFECT structures
        for i in range(min(2, len(perfect))):
            struct = perfect[i]
            selected.append((struct, "PERFECT", struct.decomposition_energy))

        # Top PARTIAL_SUCCESS structures
        for i in range(min(1, len(partial_success))):
            struct = partial_success[i]
            selected.append((struct, "PARTIAL_SUCCESS", struct.decomposition_energy))

        # STABLE_BUT_WEAK structures (if any)
        for i in range(min(1, len(stable_but_weak))):
            struct = stable_but_weak[i]
            selected.append((struct, "STABLE_BUT_WEAK", struct.decomposition_energy))

        # Worst FAILED structures (to learn what not to do)
        for i in range(max(0, len(failed) - 2), len(failed)):
            struct = failed[i]
            selected.append((struct, "FAILED", struct.decomposition_energy))

        # Fallback: if no structures selected, take best available
        if not selected and valid_structures:
            sorted_by_ed = sorted(valid_structures, key=lambda s: s.decomposition_energy)
            for i in range(min(3, len(sorted_by_ed))):
                struct = sorted_by_ed[i]
                category = "MODERATE"  # Neutral category for fallback
                selected.append((struct, category, struct.decomposition_energy))
            logger.warning(f"Legacy mode fallback: selected {len(selected)} best-Ed structures")

        logger.info(f"Legacy mode categorization:")
        logger.info(f"  - {len(perfect)} PERFECT (Ed ≤ {ed_threshold} AND satisfies constraints)")
        logger.info(f"  - {len(partial_success)} PARTIAL_SUCCESS (Ed > {ed_threshold} BUT satisfies constraints)")
        logger.info(f"  - {len(stable_but_weak)} STABLE_BUT_WEAK (Ed ≤ {ed_threshold} BUT violates constraints)")
        logger.info(f"  - {len(failed)} FAILED (Ed > {ed_threshold} AND violates constraints)")
        logger.info(f"Selected {len(selected)} structures for individual reflection")

        return selected

    def _check_constraints(self, structure: CrystalStructure) -> bool:
        """
        Check if structure satisfies task constraints.

        Args:
            structure: Structure to check

        Returns:
            True if all enabled constraints are satisfied
        """
        if not self.constraint_checker:
            # No constraint checker available - assume satisfied
            return True

        try:
            # Use constraint checker to evaluate
            result = self.constraint_checker.check(structure)
            return bool(result)
        except Exception as e:
            logger.warning(f"Constraint check failed for {structure.formula}: {e}")
            return False

    def _generate_individual_reflections(
        self,
        children: List[CrystalStructure],
        context: Dict
    ) -> List[Dict]:
        """
        Generate individual reflections for key structures

        Args:
            children: All generated structures
            context: Context including stats

        Returns:
            List of individual reflection dictionaries
        """
        logger.info("="*60)
        logger.info("GENERATING INDIVIDUAL STRUCTURE REFLECTIONS...")
        logger.info("="*60)

        # 🔥 NEW: Check reflection mode
        reflection_config = self.config.get('reflection', {})
        selection_mode = reflection_config.get('selection_mode', 'weighted_score')

        logger.info(f"Reflection selection mode: {selection_mode}")

        # Select structures to analyze based on mode
        if selection_mode == "weighted_score":
            # 🔥 NEW: Use weighted score (consistent with evolution.select)
            selected_tuples = self._select_by_weighted_score(children, reflection_config)
        else:
            # Legacy mode: Ed threshold + constraint satisfaction
            selected_tuples = self._select_by_legacy_mode(children)

        if not selected_tuples:
            logger.warning("No structures selected for individual reflection")
            return []

        # Count structures by category
        category_counts = {}
        for _, category, _ in selected_tuples:
            category_counts[category] = category_counts.get(category, 0) + 1

        logger.info(f"Analyzing {len(selected_tuples)} structures individually:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"  - {count} {category}")
        hit_count = sum(1 for struct, _, _ in selected_tuples if self._check_constraints(struct))
        logger.info(f"  - Hit: {hit_count}/{len(selected_tuples)}")

        individual_reflections = []

        for idx, (struct, category, metric_value) in enumerate(selected_tuples, 1):
            # Log structure with its category and metric
            hit_label = "HIT" if self._check_constraints(struct) else "NO-HIT"
            if selection_mode == "weighted_score":
                logger.info(
                    f"\nAnalyzing structure {idx}/{len(selected_tuples)}: {struct.formula} "
                    f"[{category} | {hit_label}] (weighted_score={metric_value:.4f})"
                )
            else:
                logger.info(
                    f"\nAnalyzing structure {idx}/{len(selected_tuples)}: {struct.formula} "
                    f"[{category} | {hit_label}] (Ed={metric_value:.4f})"
                )

            # Build individual prompt
            reflection = self._generate_single_structure_reflection(
                struct=struct,
                category=category,
                rank=idx,
                context=context
            )

            if reflection:
                individual_reflections.append(reflection)

        logger.info("="*60)
        logger.info(f"Generated {len(individual_reflections)} individual reflections")
        logger.info("="*60)

        return individual_reflections

    def _generate_single_structure_reflection(
        self,
        struct: CrystalStructure,
        category: str,
        rank: int,
        context: Dict
    ) -> Optional[Dict]:
        """
        Generate reflection for a single structure

        Args:
            struct: Structure to analyze
            category: TOP/BOTTOM or legacy categories (PERFECT/PARTIAL_SUCCESS/FAILED)
            rank: Rank among analyzed structures
            context: Context with stats

        Returns:
            Reflection dictionary or None if failed
        """
        # Get unique elements and stoichiometry
        unique_elements = sorted(set(struct.species))
        element_counts = {elem: struct.species.count(elem) for elem in unique_elements}
        stoichiometry = ', '.join(f'{elem}:{count}' for elem, count in element_counts.items())

        # Get task-specific properties from task_config
        task_constraints = self.task_config.get('constraints', {})
        task_name = self.task_config.get('name', 'Materials Discovery')
        task_desc_analysis = self.task_config.get('description', '').strip()

        # 🔥 NEW: Build property information with satisfaction indicators
        property_lines = []
        analysis_questions = []
        satisfied_props = []
        unsatisfied_props = []

        for prop_name, prop_def in task_constraints.items():
            if prop_name == 'is_valid' or not isinstance(prop_def, dict):
                continue

            # Only include enabled constraints
            if not prop_def.get('enabled', True):
                continue

            # Get property value from structure
            if hasattr(struct, prop_name):
                prop_value = getattr(struct, prop_name)

                # 🔥 NEW: Check if property satisfies constraints
                is_satisfied = True
                if 'min' in prop_def and prop_value < prop_def['min']:
                    is_satisfied = False
                if 'max' in prop_def and prop_value > prop_def['max']:
                    is_satisfied = False

                # Format property display with satisfaction indicator
                if 'min' in prop_def and 'max' in prop_def:
                    constraint_str = f"[Target: {prop_def['min']}-{prop_def['max']}]"
                elif 'min' in prop_def:
                    constraint_str = f"[Min: {prop_def['min']}]"
                elif 'max' in prop_def:
                    constraint_str = f"[Max: {prop_def['max']}]"
                else:
                    constraint_str = ""

                # 🔥 NEW: Add ✅/❌ indicators
                indicator = "✅" if is_satisfied else "❌"
                property_lines.append(f"- {prop_name}: {prop_value:.2f} {constraint_str} {indicator}")
                analysis_questions.append(prop_name)

                if is_satisfied:
                    satisfied_props.append(prop_name)
                else:
                    unsatisfied_props.append(prop_name)

        # 🔥 NEW: Always include decomposition_energy with satisfaction check
        ed_satisfied = struct.decomposition_energy <= self.metastability_threshold
        ed_indicator = "✅" if ed_satisfied else "❌"
        ed_line = f"- Decomposition Energy: {struct.decomposition_energy:.3f} eV/atom [Target: ≤{self.metastability_threshold}] {ed_indicator}"
        property_lines.insert(0, ed_line)

        if ed_satisfied:
            satisfied_props.insert(0, "decomposition_energy")
        else:
            unsatisfied_props.insert(0, "decomposition_energy")

        properties_text = '\n'.join(property_lines)

        # 🔥 NEW: Category-specific analysis guidance
        if category == "PARTIAL_SUCCESS":
            category_note = f"""
⭐ PARTIAL SUCCESS: This structure achieved EXCELLENT mechanical properties but has poor thermodynamic stability!

✅ STRENGTHS (properties that satisfy constraints):
   {', '.join(satisfied_props) if satisfied_props else 'None'}

❌ WEAKNESSES (properties that don't satisfy constraints):
   {', '.join(unsatisfied_props) if unsatisfied_props else 'None'}

💡 KEY INSIGHT: The element combination and bonding produced the RIGHT mechanical properties!
   Focus your analysis on:
   - WHAT element/structure features led to good mechanical properties (bulk/shear modulus)?
   - WHY is the stability poor (high Ed)?
   - HOW to maintain the mechanical properties while improving stability?
"""
            prop_analysis_text = """
1. **What chemical/structural features produced the EXCELLENT mechanical properties?**
   (Identify what worked well - element choice, bonding type, coordination, etc.)

2. **Why is the thermodynamic stability poor despite good mechanical properties?**
   (What causes the high decomposition energy?)
"""
        elif category == "PERFECT":
            category_note = "🌟 PERFECT: This structure satisfies ALL requirements!"
            prop_analysis_text = """
1. **What makes this structure successful?**
   (Identify all key factors - chemistry, bonding, structure)
"""
        else:
            category_note = ""
            if analysis_questions:
                prop_analysis_text = f"""
1. **Why did this structure achieve these property values?**
   Focus on: {', '.join(analysis_questions)}
   (Discuss chemical bonding, valence matching, ionic radii, electronegativity, coordination, crystal structure, etc.)
"""
            else:
                prop_analysis_text = """
1. **Why did this structure achieve this Ed value?**
   (Discuss chemical bonding, valence matching, ionic radii, electronegativity, coordination, etc.)
"""

        hit_label = "HIT" if self._check_constraints(struct) else "NO-HIT"

        # Build prompt
        prompt = f"""You are a materials science expert analyzing a crystal structure for the task: {task_name}

TASK BACKGROUND:
{task_desc_analysis}

STRUCTURE INFORMATION:
- Formula: {struct.formula}
{properties_text}
- Rank group: {category}
- Hit: {hit_label}
- Total atoms: {len(struct.species)}
- Elements: {', '.join(unique_elements)}
- Stoichiometry: {stoichiometry}
{category_note}

CONTEXT:
- Total structures generated this iteration: {context['total_generated']}
- This structure's rank group: {category}
- This structure's hit status: {hit_label}

TASK:
Analyze this specific structure and answer:
{prop_analysis_text}
2. **Key chemical/structural features** (both positive and negative aspects)

3. **Specific suggestions to improve THIS structure** (2-3 actionable ideas to better meet the task requirements)
   {f"CRITICAL: Suggest how to KEEP the good mechanical properties while improving stability!" if category == "PARTIAL_SUCCESS" else ""}

Respond in JSON format:
{{
  "reason": "Brief explanation of why these property values were achieved",
  "key_features": ["feature 1", "feature 2", "feature 3"],
  "improvement_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}}

Provide ONLY the JSON output, no additional text.
"""

        # Save prompt if output_dir exists
        if self.output_dir:
            from pathlib import Path
            reflection_dir = Path(self.output_dir) / f"iteration_{self.iteration}" / "individual_reflections"
            reflection_dir.mkdir(parents=True, exist_ok=True)

            prompt_file = reflection_dir / f"{struct.formula}_{category}_prompt.txt"
            prompt_file.write_text(prompt)

        # Call LLM
        try:
            import json
            response = self.hypothesis_agent.llm.generate(prompt, n=1)[0]

            # Try to parse JSON
            # Remove markdown code blocks if present
            response_clean = response.strip()
            if response_clean.startswith('```'):
                # Extract JSON from markdown code block
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_clean

            analysis = json.loads(response_clean)

            # Construct reflection dictionary
            reflection = {
                'structure_id': struct.metadata.get('id', f'struct_{rank}'),
                'formula': struct.formula,
                'ed': float(struct.decomposition_energy),
                'category': category,
                'analysis': analysis
            }

            # Save individual reflection
            if self.output_dir:
                reflection_file = reflection_dir / f"{struct.formula}_{category}.json"
                with open(reflection_file, 'w') as f:
                    json.dump(reflection, f, indent=2)

            logger.info(f"  ✓ Individual reflection generated for {struct.formula}")

            return reflection

        except json.JSONDecodeError as e:
            logger.error(f"  ✗ Failed to parse JSON for {struct.formula}: {e}")
            logger.error(f"    Response: {response[:200]}...")
            return None
        except Exception as e:
            logger.error(f"  ✗ Failed to generate individual reflection for {struct.formula}: {e}")
            return None

    def _generate_llm_reflection(self, children: List[CrystalStructure], stats: Dict, example_pool: List[CrystalStructure] = None) -> str:
        """
        Generate LLM-based deep reflection analysis

        Args:
            children: Generated child structures
            stats: Statistical reflection data
            example_pool: Parent pool structures (for baseline comparison)

        Returns:
            LLM analysis text
        """
        logger.info("="*60)
        logger.info("GENERATING LLM REFLECTION ANALYSIS...")
        logger.info("="*60)

        # Build prompt (with parent pool baseline if available)
        prompt = self._build_reflection_prompt(children, stats, example_pool)

        # Save prompt
        if self.output_dir:
            from pathlib import Path
            reflection_dir = Path(self.output_dir) / f"iteration_{self.iteration}" / "reflection"
            reflection_dir.mkdir(parents=True, exist_ok=True)
            
            # reflection prompt
            prompt_file = reflection_dir / "reflection_prompt.txt"
            prompt_file.write_text(prompt)
            logger.debug(f"Reflection prompt saved to {prompt_file}")

        # Print prompt to console
        logger.info("="*60)
        logger.info("REFLECTION PROMPT:")
        logger.info("="*60)
        logger.info(prompt)
        logger.info("="*60)

        # Call LLM
        try:
            analysis = self.hypothesis_agent.llm.generate(prompt, n=1)[0]

            # Print analysis to console
            logger.info("="*60)
            logger.info("LLM REFLECTION ANALYSIS:")
            logger.info("="*60)
            logger.info(analysis)
            logger.info("="*60)

            # Save analysis
            if self.output_dir:
                analysis_file = reflection_dir / "analysis.txt"
                analysis_file.write_text(analysis)
                logger.debug(f"Reflection analysis saved to {analysis_file}")

            return analysis

        except Exception as e:
            logger.error(f"LLM reflection analysis failed: {e}")
            return "LLM analysis failed - using statistical reflection only"

    def _build_reflection_prompt(self, children: List[CrystalStructure], stats: Dict, example_pool: List[CrystalStructure] = None) -> str:
        """
        Build prompt for LLM reflection analysis (supports dual reflection modes)

        Args:
            children: Generated child structures
            stats: Statistical reflection data
            example_pool: Parent pool structures (for baseline comparison)

        Returns:
            Prompt string
        """
        # Get task information
        task_name = self.task_config.get('name', 'Materials Discovery')
        task_desc_reflect = self.task_config.get('description', '').strip()
        task_constraints = self.task_config.get('constraints', {})

        # Determine which properties to track
        tracked_properties = []
        for prop_name, prop_def in task_constraints.items():
            if prop_name == 'is_valid' or not isinstance(prop_def, dict):
                continue
            if prop_def.get('enabled', True):
                tracked_properties.append(prop_name)

        # 🔥 NEW: Check reflection mode
        reflection_config = self.config.get('reflection', {})
        selection_mode = reflection_config.get('selection_mode', 'weighted_score')
        # Property labels used in weighted-score prompt sections. Must be available
        # even when parent pool is empty (population_size=0).
        _PLB = {
            "band_gap": ("gap", 2), "formation_energy": ("Ef", 3),
            "bulk_modulus": ("bulk", 1), "shear_modulus": ("shear", 1),
            "density": ("rho", 2), "piezoelectric_coefficient": ("piezo", 2),
            "dielectric_constant": ("dielectric", 1),
        }
        _tracked = {
            p for p, d in task_constraints.items()
            if p != "is_valid" and isinstance(d, dict) and d.get("enabled", True)
        }

        # Categorization depends on mode
        if selection_mode == "weighted_score":
            # 🔥 NEW: Weighted score mode - compute scores for all structures
            valid_structures = [c for c in children if c.is_valid and c.decomposition_energy is not None]
            scored_structures = []
            for struct in valid_structures:
                score = self._compute_weighted_score(struct)
                scored_structures.append((struct, score))

            # Sort by score (descending - higher is better)
            scored_structures.sort(key=lambda x: x[1], reverse=True)

            # Define top and bottom structures
            # 🔥 FIX: Only show BOTTOM when we have enough structures to differentiate
            if len(scored_structures) > 5:
                # Enough structures: show distinct TOP and BOTTOM
                top_structures = scored_structures[:5]
                bottom_structures = scored_structures[-3:]
            else:
                # Few structures: just show all as TOP, no BOTTOM
                top_structures = scored_structures
                bottom_structures = []

            # For compatibility with prompt construction below
            perfect = []
            partial_success = []
            successful = []
            failed = []

        else:
            # Legacy mode: Multi-dimensional categorization
            satisfied_constraints = []
            if hasattr(self, 'constraint_checker') and self.constraint_checker:
                satisfied_constraints = self.constraint_checker.filter_satisfied(children)
            satisfied_set = set(id(s) for s in satisfied_constraints)

            # Categorize with BOTH stability AND task constraints
            perfect = [c for c in children if c.is_valid and c.decomposition_energy is not None
                       and c.decomposition_energy <= self.metastability_threshold
                       and id(c) in satisfied_set]
            partial_success = [c for c in children if c.is_valid and c.decomposition_energy is not None
                              and c.decomposition_energy > self.metastability_threshold
                              and id(c) in satisfied_set]

            # Legacy categories
            successful = perfect  # Only truly perfect structures
            failed = [c for c in children if c.is_valid and c.decomposition_energy is not None
                     and c.decomposition_energy > self.failure_threshold
                     and id(c) not in satisfied_set]

            # For compatibility
            top_structures = []
            bottom_structures = []
            scored_structures = []

        prompt_parts = [
            f"You are a materials science expert analyzing crystal structure generation results for: {task_name}",
            "",
            f"Task background: {task_desc_reflect}",
            "",
            "="*60,
            "STATISTICAL SUMMARY:",
            "="*60,
            "",
            f"Total generated: {stats['total_generated']} structures",
            f"Valid structures: {stats['valid_count']} ({stats['valid_rate']:.1f}%)",
            f"Metastable (Ed≤{self.metastability_threshold}): {stats['metastable_count']} ({stats['metastable_rate']:.1f}%)",
            f"Stable (Ed<0): {stats['stable_count']} ({stats['stable_rate']:.1f}%)",
        ]

        if self.generation_mode == 'direct_llm':
            prompt_parts.extend([
                f"Parse success: {stats.get('parse_success', stats['total_generated'])} / {stats.get('parse_requested', stats['total_generated'])} = {stats.get('parse_success_rate', 0.0):.1f}%",
                f"Valid structures (strict): {stats.get('valid_count', 0)} / {stats.get('parse_requested', stats['total_generated'])} = {stats.get('valid_rate_strict', stats['valid_rate']):.1f}%",
                "",
            ])

        # Add Hit Rate if available
        if 'hit_rate' in stats and stats['hit_rate'] >= 0:
            prompt_parts.extend([
                f"✓ HIT RATE (satisfy ALL task constraints): {stats['hit_count']} / {stats['total_generated']} = {stats['hit_rate']:.1f}%",
            ])
            if self.generation_mode == 'direct_llm':
                prompt_parts.extend([
                    f"✓ HIT RATE strict (requested denominator): {stats['hit_count']} / {stats.get('parse_requested', stats['total_generated'])} = {stats.get('hit_rate_strict', stats['hit_rate']):.1f}%",
                ])
            prompt_parts.append("")
        else:
            prompt_parts.append("")

        if stats['best_ed'] != float('inf'):
            prompt_parts.extend([
                f"Best Ed: {stats['best_ed']:.3f} eV/atom",
                f"Average Ed: {stats['avg_ed']:.3f} eV/atom",
                f"Worst Ed: {stats['worst_ed']:.3f} eV/atom",
                "",
            ])

        # Add weighted score statistics (multi-objective)
        if 'best_weighted_score' in stats and stats.get('best_weighted_score', 0) > 0:
            prompt_parts.extend([
                f"Multi-objective Weighted Score (higher is better):",
                f"  Best: {stats['best_weighted_score']:.4f}",
                f"  Average: {stats.get('avg_weighted_score', 0):.4f}",
                f"  Worst: {stats.get('worst_weighted_score', 0):.4f}",
                "",
            ])

        prompt_parts.extend([
            f"Diversity score (current iteration): {stats['diversity']:.2f}",
            "",
        ])

        # Add global formula diversity statistics
        if 'global_formula_stats' in stats:
            global_stats = stats['global_formula_stats']
            global_diversity_rate = 100 * global_stats['unique_formulas'] / global_stats['total_structures'] if global_stats['total_structures'] > 0 else 0

            prompt_parts.extend([
                "="*60,
                "GLOBAL FORMULA DIVERSITY (ALL ITERATIONS):",
                "="*60,
                "",
                f"Total structures generated so far: {global_stats['total_structures']}",
                f"Unique formulas: {global_stats['unique_formulas']}",
                f"Global diversity rate: {global_diversity_rate:.1f}%",
                "",
            ])

            # Show most repeated formulas
            most_repeated = global_stats.get('most_repeated_formulas', {})
            if most_repeated:
                prompt_parts.extend([
                    "⚠️  MOST REPEATED FORMULAS (avoid generating these again):",
                    ""
                ])
                for idx, (formula, count) in enumerate(list(most_repeated.items())[:10], 1):
                    percentage = 100 * count / global_stats['total_structures']
                    prompt_parts.append(f"  {idx}. {formula}: {count}× ({percentage:.1f}%)")
                prompt_parts.extend([
                    "",
                    "NOTE: You should prioritize generating NEW formulas that are NOT in the above list!",
                    "",
                ])

        # 🔥 NEW: Add parent pool baseline comparison (weighted_score mode only)
        if example_pool and selection_mode == "weighted_score":
            # Compute scores for parent pool
            parent_scores = []
            for struct in example_pool:
                if struct.is_valid and struct.decomposition_energy is not None:
                    score = self._compute_weighted_score(struct)
                    parent_scores.append((struct, score))

            if parent_scores:
                parent_scores.sort(key=lambda x: x[1], reverse=True)
                best_parent = parent_scores[0]
                avg_parent_score = sum(s for _, s in parent_scores) / len(parent_scores)

                # Compute scores for children
                child_scores = []
                for struct in children:
                    if struct.is_valid and struct.decomposition_energy is not None:
                        score = self._compute_weighted_score(struct)
                        child_scores.append((struct, score))

                best_child = max(child_scores, key=lambda x: x[1]) if child_scores else None

                prompt_parts.extend([
                    "="*60,
                    "PARENT POOL CONTEXT (Historical Best Structures):",
                    "="*60,
                    "",
                    "The parent pool represents the best structures discovered across all previous iterations.",
                    "These structures serve as the baseline for comparison.",
                    "",
                    f"Parent Pool Statistics:",
                    f"  - Total Structures: {len(example_pool)}",
                    f"  - Best Score: {best_parent[1]:.4f} ({best_parent[0].formula})",
                    f"  - Average Score: {avg_parent_score:.4f}",
                    "",
                    "Parent Pool Structures:",
                    ""
                ])
                for i, (struct, score) in enumerate(parent_scores, 1):
                    ed = struct.decomposition_energy if struct.decomposition_energy is not None else float('inf')
                    extra = ""
                    for prop in _tracked:
                        if prop in _PLB:
                            lbl, dg = _PLB[prop]
                            val = struct.properties.get(prop)
                            if val is not None:
                                extra += f", {lbl}={val:.{dg}f}"
                    prompt_parts.append(
                        f"  {i}. {struct.formula} "
                        f"(score={score:.4f}, Ed={ed:.3f} eV/atom{extra}, {len(struct.species)} atoms)"
                    )
                prompt_parts.append("")

                # Add iteration results (objective data only, no judgement)
                if best_child:
                    avg_child_score = sum(s for _, s in child_scores) / len(child_scores) if child_scores else 0

                    prompt_parts.extend([
                        "THIS ITERATION RESULTS:",
                        f"  - Total Children: {len(child_scores)}",
                        f"  - Best Child Score: {best_child[1]:.4f} ({best_child[0].formula})",
                        f"  - Average Child Score: {avg_child_score:.4f}",
                        "",
                        "Comparison to Parent Pool:",
                        f"  - Best: {best_child[1]:.4f} (child) vs {best_parent[1]:.4f} (parent)  →  Δ = {best_child[1] - best_parent[1]:+.4f}",
                        f"  - Avg:  {avg_child_score:.4f} (child) vs {avg_parent_score:.4f} (parent)  →  Δ = {avg_child_score - avg_parent_score:+.4f}",
                        ""
                    ])
                else:
                    prompt_parts.extend([
                        "THIS ITERATION RESULTS:",
                        "  - No valid children generated",
                        ""
                    ])

        # 🔥 NEW: Structure list display depends on mode
        if selection_mode == "weighted_score":
            # ========== WEIGHTED SCORE MODE ==========
            if top_structures:
                prompt_parts.extend([
                    "="*60,
                    "🏆 TOP STRUCTURES (by weighted multi-objective score):",
                    "="*60,
                    ""
                ])
                for i, (struct, score) in enumerate(top_structures, 1):
                    extra = ""
                    for prop in _tracked:
                        if prop in _PLB:
                            lbl, dg = _PLB[prop]
                            val = struct.properties.get(prop)
                            if val is not None:
                                extra += f", {lbl}={val:.{dg}f}"
                    prompt_parts.append(f"  {i}. {struct.formula} (weighted_score={score:.4f}, Ed={struct.decomposition_energy:.3f} eV/atom{extra}, {len(struct.species)} atoms)")
                prompt_parts.append("")

            if bottom_structures:
                prompt_parts.extend([
                    "="*60,
                    "⚠️  BOTTOM STRUCTURES (by weighted multi-objective score):",
                    "="*60,
                    ""
                ])
                for i, (struct, score) in enumerate(bottom_structures, 1):
                    extra = ""
                    for prop in _tracked:
                        if prop in _PLB:
                            lbl, dg = _PLB[prop]
                            val = struct.properties.get(prop)
                            if val is not None:
                                extra += f", {lbl}={val:.{dg}f}"
                    prompt_parts.append(f"  {i}. {struct.formula} (weighted_score={score:.4f}, Ed={struct.decomposition_energy:.3f} eV/atom{extra}, {len(struct.species)} atoms)")
                prompt_parts.append("")

            # Detailed breakdown for weighted score mode
            # 🔥 FIX: Avoid duplicates when selecting structures
            seen_ids = set()
            all_analyzed_pairs = []

            # Add top-3 structures
            for pair in top_structures[:3]:
                struct, score = pair
                struct_id = id(struct)
                if struct_id not in seen_ids:
                    all_analyzed_pairs.append((pair, "TOP"))
                    seen_ids.add(struct_id)

            # Add bottom-2 structures (skip if already in top-3)
            bottom_count = 0
            for pair in reversed(bottom_structures):  # Start from worst
                if bottom_count >= 2:
                    break  # Already have 2 bottom structures
                struct, score = pair
                struct_id = id(struct)
                if struct_id not in seen_ids:
                    all_analyzed_pairs.append((pair, "BOTTOM"))
                    seen_ids.add(struct_id)
                    bottom_count += 1

            if all_analyzed_pairs:
                prompt_parts.extend([
                    "="*60,
                    "DETAILED PER-STRUCTURE BREAKDOWN:",
                    "="*60,
                    "",
                    f"Below are detailed profiles for {len(all_analyzed_pairs)} key structures (ranked by weighted score).",
                    "Please analyze EACH structure individually in your response.",
                    ""
                ])

                for idx, ((struct, score), category) in enumerate(all_analyzed_pairs, 1):

                    # Get unique elements
                    unique_elements = sorted(set(struct.species))
                    element_counts = {elem: struct.species.count(elem) for elem in unique_elements}

                    # Build structure info
                    struct_info = [
                        f"Structure #{idx} [{category}]:",
                        f"  Formula: {struct.formula}",
                        f"  Weighted Score: {score:.4f}",
                    ]

                    # Add decomposition energy with satisfaction indicator
                    ed_satisfied = struct.decomposition_energy <= self.metastability_threshold
                    ed_indicator = "✅" if ed_satisfied else "❌"
                    struct_info.append(f"  Decomposition Energy: {struct.decomposition_energy:.3f} eV/atom [Target: ≤{self.metastability_threshold}] {ed_indicator}")

                    hit_label = "HIT" if self._check_constraints(struct) else "NO-HIT"
                    struct_info.append(f"  Hit: {hit_label}")

                    # Add task-specific properties with satisfaction indicators
                    for prop_name in tracked_properties:
                        prop_value = None
                        if hasattr(struct, prop_name):
                            prop_value = getattr(struct, prop_name)
                        elif prop_name in struct.properties:
                            prop_value = struct.properties.get(prop_name)

                        if prop_value is None:
                            continue

                        prop_def = task_constraints.get(prop_name, {})

                        # Check if property satisfies constraints
                        is_satisfied = True
                        if 'min' in prop_def and prop_value < prop_def['min']:
                            is_satisfied = False
                        if 'max' in prop_def and prop_value > prop_def['max']:
                            is_satisfied = False

                        # Format constraint info
                        if 'min' in prop_def and 'max' in prop_def:
                            constraint_str = f" [Target: {prop_def['min']}-{prop_def['max']}]"
                        elif 'min' in prop_def:
                            constraint_str = f" [Min: {prop_def['min']}]"
                        elif 'max' in prop_def:
                            constraint_str = f" [Max: {prop_def['max']}]"
                        else:
                            constraint_str = ""

                        # Add satisfaction indicator
                        indicator = "✅" if is_satisfied else "❌"
                        struct_info.append(f"  {prop_name}: {prop_value:.2f}{constraint_str} {indicator}")

                    struct_info.extend([
                        f"  Total atoms: {len(struct.species)}",
                        f"  Elements: {', '.join(unique_elements)}",
                        f"  Stoichiometry: {', '.join(f'{elem}:{count}' for elem, count in element_counts.items())}",
                        ""
                    ])

                    prompt_parts.extend(struct_info)

        else:
            # ========== LEGACY MODE ==========
            # Add PERFECT structures (Ed ≤ threshold AND satisfies constraints)
            if perfect:
                prompt_parts.extend([
                    "="*60,
                    f"🌟 PERFECT STRUCTURES (Ed ≤ {self.metastability_threshold} AND satisfies ALL constraints):",
                    "="*60,
                    ""
                ])
                for i, struct in enumerate(perfect[:5], 1):
                    prompt_parts.append(f"  {i}. {struct.formula} (Ed = {struct.decomposition_energy:.3f} eV/atom, {len(struct.species)} atoms)")
                prompt_parts.append("")

            # Add PARTIAL_SUCCESS structures (unstable BUT satisfies mechanical constraints)
            if partial_success:
                prompt_parts.extend([
                    "="*60,
                    f"⭐ PARTIAL SUCCESS (satisfies mechanical constraints BUT Ed > {self.metastability_threshold}):",
                    "="*60,
                    "These structures achieved the RIGHT mechanical properties (bulk/shear modulus)",
                    "but have poor thermodynamic stability. KEY INSIGHT: The element combinations work!",
                    ""
                ])
                for i, struct in enumerate(partial_success[:5], 1):
                    prompt_parts.append(f"  {i}. {struct.formula} (Ed = {struct.decomposition_energy:.3f} eV/atom, {len(struct.species)} atoms)")
                prompt_parts.append("")

            # Add failed structures (neither stable nor satisfies constraints)
            if failed:
                prompt_parts.extend([
                    "="*60,
                    f"❌ FAILED STRUCTURES (Ed > {self.failure_threshold} AND doesn't satisfy constraints):",
                    "="*60,
                    ""
                ])
                # Limit to top 3 worst failures to keep prompt concise
                for i, struct in enumerate(failed[:3], 1):
                    prompt_parts.append(f"  {i}. {struct.formula} (Ed = {struct.decomposition_energy:.3f} eV/atom, {len(struct.species)} atoms)")
                prompt_parts.append("")

            # Add middle-range structures (between metastability and failure thresholds)
            middle = [c for c in children if c.is_valid and c.decomposition_energy is not None
                      and self.metastability_threshold < c.decomposition_energy <= self.failure_threshold]
            if middle:
                prompt_parts.extend([
                    "="*60,
                    f"MODERATE STRUCTURES ({self.metastability_threshold} < Ed ≤ {self.failure_threshold} eV/atom):",
                    "="*60,
                    ""
                ])
                for i, struct in enumerate(middle[:5], 1):
                    prompt_parts.append(f"  {i}. {struct.formula} (Ed = {struct.decomposition_energy:.3f} eV/atom, {len(struct.species)} atoms)")
                prompt_parts.append("")

            # Per-structure detailed breakdown with legacy categories
            all_analyzed = perfect[:2] + partial_success[:2] + failed[:3]  # Top 2 perfect, 2 partial, 3 worst
            if all_analyzed:
                prompt_parts.extend([
                    "="*60,
                    "DETAILED PER-STRUCTURE BREAKDOWN:",
                    "="*60,
                    "",
                    "Below are detailed profiles for key structures (perfect, partial success, and failed).",
                    "Please analyze EACH structure individually in your response.",
                    ""
                ])

                for idx, struct in enumerate(all_analyzed, 1):
                    # Multi-dimensional category determination
                    if any(s is struct for s in perfect):
                        category = "PERFECT"
                    elif any(s is struct for s in partial_success):
                        category = "PARTIAL_SUCCESS"
                    else:
                        category = "FAILED"

                    # Get unique elements
                    unique_elements = sorted(set(struct.species))
                    element_counts = {elem: struct.species.count(elem) for elem in unique_elements}

                    # Build structure info with satisfaction indicators
                    struct_info = [
                        f"Structure #{idx} [{category}]:",
                        f"  Formula: {struct.formula}",
                    ]

                    # Add decomposition energy with satisfaction indicator
                    ed_satisfied = struct.decomposition_energy <= self.metastability_threshold
                    ed_indicator = "✅" if ed_satisfied else "❌"
                    struct_info.append(f"  Decomposition Energy: {struct.decomposition_energy:.3f} eV/atom [Target: ≤{self.metastability_threshold}] {ed_indicator}")

                    # Add task-specific properties with satisfaction indicators
                    for prop_name in tracked_properties:
                        if hasattr(struct, prop_name):
                            prop_value = getattr(struct, prop_name)
                            prop_def = task_constraints.get(prop_name, {})

                            # Check if property satisfies constraints
                            is_satisfied = True
                            if 'min' in prop_def and prop_value < prop_def['min']:
                                is_satisfied = False
                            if 'max' in prop_def and prop_value > prop_def['max']:
                                is_satisfied = False

                            # Format constraint info
                            if 'min' in prop_def and 'max' in prop_def:
                                constraint_str = f" [Target: {prop_def['min']}-{prop_def['max']}]"
                            elif 'min' in prop_def:
                                constraint_str = f" [Min: {prop_def['min']}]"
                            elif 'max' in prop_def:
                                constraint_str = f" [Max: {prop_def['max']}]"
                            else:
                                constraint_str = ""

                            # Add satisfaction indicator
                            indicator = "✅" if is_satisfied else "❌"
                            struct_info.append(f"  {prop_name}: {prop_value:.2f}{constraint_str} {indicator}")

                    struct_info.extend([
                        f"  Total atoms: {len(struct.species)}",
                        f"  Elements: {', '.join(unique_elements)}",
                        f"  Stoichiometry: {', '.join(f'{elem}:{count}' for elem, count in element_counts.items())}",
                        ""
                    ])

                    prompt_parts.extend(struct_info)

        # Task instruction - MODIFIED to require per-structure analysis with task-specific properties
        properties_to_analyze = ', '.join(tracked_properties) if tracked_properties else 'decomposition_energy'

        prompt_parts.extend([
            "="*60,
            "YOUR TASK:",
            "="*60,
            "",
            "Provide a concise, data-grounded materials analysis with TWO parts:",
            "",
            "PART A: INDIVIDUAL STRUCTURE ANALYSIS",
            "For EACH structure in the 'Detailed Per-Structure Breakdown' above, provide:",
            f"   - Why did this structure achieve these property values? (Focus on: {properties_to_analyze})",
            "   - What are the key chemical/structural features that lead to these property values (good or bad)?",
            "   - Specific suggestion to improve THIS structure to better meet task requirements (1-2 sentences)",
            "",
            "PART B: OVERALL PATTERNS & STRATEGY",
            "",
            "1. **Success Patterns** (if any successful structures):",
            "   - What chemical patterns do successful structures share?",
            "   - Element combinations that lead to desired property values",
            "   - Common structural motifs or coordination environments",
            "",
            "2. **Failure Patterns** (if any failed structures):",
            "   - Why don't these structures meet the property requirements?",
            "   - Chemical incompatibilities (valence mismatch, poor bonding, etc.)",
            "   - Structural issues (poor coordination, unfavorable packing, etc.)",
            "",
            "3. **Strategic Recommendations** (3-5 specific actionable suggestions for next iteration):",
            "   - Recommended element combinations to achieve target properties",
            "   - Element combinations to AVOID",
            "   - Structural strategies (coordination preferences, crystal symmetries, etc.)",
            "   - Composition strategies that favor the desired properties",
            "",
            "Format your response with clear sections:",
            "### Individual Structure Analysis",
            "(Analyze each structure separately)",
            "",
            "### Overall Success Patterns",
            "### Overall Failure Patterns",
            "### Strategic Recommendations for Next Iteration",
            "",
            "Be specific and chemistry-focused. Avoid generic research-process advice (e.g., \"use DFT\"),",
            "avoid re-stating the per-structure tables, and tie each pattern or recommendation to the data above."
        ])

        return "\n".join(prompt_parts)
