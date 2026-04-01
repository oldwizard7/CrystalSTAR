"""Main entry point for MatLLMSearch"""

import logging
import yaml
import argparse
import os
from pathlib import Path
import numpy as np
import sys
import json
import platform
from datetime import datetime

# Ensure the project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from src.utils.env_loader import load_env_file, check_environment
load_env_file()

from src.agents.orchestrator import OrchestratorAgent
from src.utils.visualization import ResultsVisualizer
from src.utils.logger import LoggerSetup

# Temporary logger (will be reconfigured after parsing args)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _merge_dict(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_profile(config: dict, profile_name: str) -> dict:
    profiles = config.get('profiles', {})
    if not profile_name or profile_name not in profiles:
        return config
    merged = _merge_dict(config, profiles[profile_name])
    merged.setdefault('profile', {})
    merged['profile']['active'] = profile_name
    return merged


def load_prompts(prompts_path: str) -> dict:
    """Load prompt templates from YAML"""
    with open(prompts_path, 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts


def save_results(structures: list, output_dir: str, name: str):
    """Save structures to output directory"""
    output_path = Path(output_dir) / "structures" / name
    output_path.mkdir(parents=True, exist_ok=True)

    for i, struct in enumerate(structures):
        struct.save(str(output_path / f"{name}_{i}.json"))


def main():
    """Main execution loop"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='MatAgent - LLM-assisted materials discovery')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--prompts', type=str, default='config/prompts.yaml',
                       help='Path to prompts file')
    parser.add_argument('--data', type=str, default='data/reference_examples/',
                       help='Path to reference examples (format demonstration)')
    parser.add_argument('--output', type=str, default='output/',
                       help='Output directory')
    parser.add_argument('--task', type=str, default=None,
                       help='Task name from tasks.yaml (see TASKS.md)')
    parser.add_argument('--profile', type=str, default=None,
                       help='Config profile override (e.g., login, gpu)')
    parser.add_argument('--list-tasks', action='store_true',
                       help='List all available tasks and exit')
    args = parser.parse_args()

    # ========== Handle --list-tasks early (before logging setup) ==========
    if args.list_tasks:
        from src.utils.task_loader import TaskLoader
        task_file = 'config/tasks.yaml'
        loader = TaskLoader(task_file)

        print("="*80)
        print("Available Tasks")
        print("="*80)
        print()

        for task_name in loader.list_tasks():
            task = loader.get_task(task_name)
            print(f"  {task_name}:")
            print(f"    Name: {task['name']}")
            print(f"    Description: {task['description'][:80]}...")
            constraints = loader.get_constraints(task_name)
            print(f"    Active Constraints: {len(constraints)}")
            print()

        print("Use: python src/main.py --task <task_name>")
        print("See: TASKS.md for detailed documentation")
        return
    # ========== End list-tasks handler ==========

    # ========== Setup logging with timestamped output directory ==========
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to both console and file
    LoggerSetup.setup_logger(
        output_dir=str(run_dir),
        level=logging.INFO,
        log_filename="run.log"
    )

    # Log system information
    LoggerSetup.log_system_info(logger)

    # Update output directory to use timestamped run directory
    args.output = str(run_dir)
    # ========== End logging setup ==========

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    profile_env = config.get('profile', {}).get('env_var', 'MATAGENT_PROFILE')
    profile_name = args.profile or os.getenv(profile_env) or config.get('profile', {}).get('active')
    if profile_name:
        if profile_name not in config.get('profiles', {}):
            logger.warning(f"Unknown profile '{profile_name}', using base config")
        else:
            config = apply_profile(config, profile_name)
            logger.info(f"Using config profile: {profile_name}")

    # Normalize ablation mode
    ablation_mode = str(config.get("ablation_mode", "crystal_forge")).strip().lower()
    valid_modes = {"memory_less", "vanilla_react", "flat_memory", "crystal_forge"}
    if ablation_mode not in valid_modes:
        logger.warning("Unknown ablation_mode=%s, fallback to crystal_forge", ablation_mode)
        ablation_mode = "crystal_forge"
    config["ablation_mode"] = ablation_mode
    logger.info("Ablation mode: %s", ablation_mode)
    if config.get("generation_mode", "tool_calling") != "tool_calling":
        logger.info(
            "ablation_mode is ignored for generation_mode=%s",
            config.get("generation_mode"),
        )
    # Route prompt logs to run-specific file
    config.setdefault('llm', {})
    config['llm']['prompt_log_path'] = str(run_dir / "prompts.log")

    prompts = load_prompts(args.prompts)

    # ========== Load Task Configuration ==========
    from src.utils.task_loader import TaskLoader
    from src.core.constraints import ConstraintChecker

    # Determine which task to use
    task_file = config.get('task', {}).get('config_file', 'config/tasks.yaml')
    task_name = args.task or config.get('task', {}).get('active', 'wide_bandgap_semiconductors')

    logger.info(f"Loading task configuration from {task_file}")
    task_loader = TaskLoader(task_file)
    task_config = task_loader.get_task(task_name)
    # Keep the machine-readable key for downstream per-task logic
    task_config['task_key'] = task_name

    logger.info(f"Selected task: {task_config['name']}")
    logger.info(f"Description: {task_config['description'][:100]}...")

    # Initialize constraint checker
    constraints = task_loader.get_constraints(task_name)
    constraint_checker = ConstraintChecker(constraints)

    logger.info(f"\nTask Constraints (for Hit Rate calculation):")
    constraints_text = constraint_checker.get_constraints_text()
    for line in constraints_text.split('\n'):
        logger.info(f"  {line}")

    # Update config with task objective (override if specified in task)
    task_objective = task_loader.get_objective(task_name)
    if task_objective:
        config['objective'] = task_objective
        # If task provides targets/weights but no explicit type, default to multi-objective.
        if 'type' not in config['objective'] and 'targets' in config['objective'] and 'weights' in config['objective']:
            config['objective']['type'] = 'multi_objective'
        logger.info(f"\nUsing task-specific objective")

    # Update prompts with task information
    # (Will be used by hypothesis generation)
    config['task_info'] = {
        'name': task_config['name'],
        'description': task_config['description'],
        'constraints_text': constraints_text
    }
    # ========== End Task Loading ==========

    # Initialize orchestrator with task configuration and constraint checker
    logger.info("\nInitializing Orchestrator Agent...")
    orchestrator = OrchestratorAgent(
        config,
        prompts,
        task_config=task_config,
        output_dir=args.output,
        constraint_checker=constraint_checker
    )
    
    # Initialize reference examples dynamically from Materials Project
    logger.info("Fetching task-specific reference examples from Materials Project...")
    from src.utils.reference_fetcher import TaskReferenceProvider

    try:
        # Create provider with LLM assistance
        ref_provider = TaskReferenceProvider(
            llm_client=orchestrator.structure_generator.llm  # Use same LLM client
        )

        positive_refs, negative_refs = ref_provider.get_task_references(
            task_name=task_name,
            task_constraints=task_config.get('constraints', {}),
            n_positive=3,  # 3 success examples
            n_negative=2   # 2 failure examples
        )

        logger.info(f"  Retrieved {len(positive_refs)} positive + {len(negative_refs)} negative reference examples")

        # Set reference examples in structure generator
        if positive_refs:
            # Disable example_selector to use our task-specific references instead
            orchestrator.structure_generator.example_selector = None

            # Set reference examples
            orchestrator.structure_generator.set_reference_examples(positive_refs)
            logger.info(f"  ✓ Set {len(positive_refs)} task-specific reference examples:")
            for ref in positive_refs:
                logger.info(f"      - {ref.formula}")

        if negative_refs:
            logger.info(f"  ✗ Set {len(negative_refs)} failure examples to avoid:")
            for ref in negative_refs:
                orchestrator.structure_generator.add_failure_example(ref)
                logger.info(f"      - {ref.formula}")

    except Exception as e:
        logger.warning(f"Failed to fetch MP reference examples: {e}")
        logger.info(f"Falling back to loading from {args.data}")
        # Fallback: use original loading logic
        filters = {
            'num_elements': (3, 6),
            'exclude_elements': []
        }
        example_pool = orchestrator.initialize(args.data, filters)
    else:
        # Load task-specific initial parent pool seeds (if configured)
        example_pool = orchestrator.load_initial_parent_pool()
    
    # if not example_pool:
    #     logger.error("No initial structures loaded. Exiting.")
    #     return
    
    # ACE Main Loop
    logger.info("\n" + "="*60)
    logger.info("Starting MatLLMSearch ACE Loop")
    logger.info("="*60 + "\n")

    all_structures = []
    consecutive_failures = 0  # Track consecutive failures
    max_consecutive_failures = 2  # Allow up to 2 consecutive failures

    while True:
        orchestrator.iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {orchestrator.iteration}")
        logger.info(f"{'='*60}\n")

        if (
            config.get("generation_mode", "tool_calling") == "tool_calling"
            and config.get("ablation_mode") == "memory_less"
        ):
            orchestrator.reset_step_memory()

        # 1. GENERATE: Create new structures
        children = orchestrator.generate(example_pool)

        if not children:
            consecutive_failures += 1
            logger.warning(
                f"No children generated in iteration {orchestrator.iteration}. "
                f"Consecutive failures: {consecutive_failures}/{max_consecutive_failures}"
            )

            # Check if we should terminate
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    f"Reached maximum consecutive failures ({max_consecutive_failures}). "
                    f"Terminating search."
                )
                break

            # Continue to next iteration (retry)
            logger.info("Retrying in next iteration...")
            continue

        # Reset failure counter on success
        consecutive_failures = 0

        # 2. REFLECT: Analyze results (with parent pool baseline for comparison)
        reflection = orchestrator.reflect(children, example_pool)

        # 3. CURATE: Evolve strategy (pass example_pool for analysis)
        new_plan = orchestrator.curate(reflection, example_pool)

        # 4. SELECT: Form next generation
        from src.core.evolution import EvolutionEngine
        evolution = EvolutionEngine(config)
        example_pool = evolution.select(example_pool, children)
        
        # Store all structures
        all_structures.extend(children)
        
        # Save checkpoint
        if orchestrator.iteration % config['logging']['save_frequency'] == 0:
            orchestrator.save_checkpoint(example_pool, args.output)
        
        # Check termination
        if orchestrator.should_terminate(reflection):
            logger.info("\nTermination condition met.")
            break

    # Persist HMRR macro memory (if enabled)
    try:
        orchestrator.finalize_hier_memory()
    except Exception as e:
        logger.warning(f"Failed to finalize hierarchical memory: {e}")
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("SEARCH COMPLETE")
    logger.info("="*60)
    
    # Save all structures
    logger.info("\n" + "="*60)
    logger.info("FILTERING KNOWN MATERIALS")
    logger.info("="*60)
    
    from src.utils.material_database import categorize_materials, get_statistics

    def progress_callback(current, total):
        if current % 5 == 0 or current == total:
            logger.info(f"  Checking materials: {current}/{total}")
    
    logger.info("Querying Materials Project database...")

    # Get novelty detection config
    novelty_config = config.get('novelty', {})
    use_structure_matching = novelty_config.get('use_structure_matching', True)
    novelty_mode = novelty_config.get('mode', 'standard')

    if use_structure_matching:
        logger.info(f"Using structure-aware novelty detection (mode: {novelty_mode})")
    else:
        logger.info("Using formula-only novelty detection (legacy mode)")

    categorized = categorize_materials(
        all_structures,
        use_mp=True,
        progress_callback=progress_callback,
        use_structure_matching=use_structure_matching,  # NEW!
        novelty_mode=novelty_mode  # NEW!
    )
    
    known_structures = categorized['known']
    novel_structures = categorized['novel']
    novelty_details = categorized.get('novelty_details', [])
    novelty_tier_counts = categorized.get('novelty_tier_counts', {})

    if novelty_details:
        novelty_report = {
            'config': {
                'use_structure_matching': use_structure_matching,
                'mode': novelty_mode
            },
            'tier_counts': novelty_tier_counts,
            'details': novelty_details
        }
        novelty_report_path = Path(args.output) / "novelty_report.json"
        with open(novelty_report_path, 'w') as f:
            json.dump(novelty_report, f, indent=2)
        logger.info(f"Saved novelty detection report to {novelty_report_path}")
    
    stats = get_statistics(categorized)
    
    logger.info(f"\n{'='*60}")
    logger.info("MATERIAL CATEGORIZATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total structures: {stats['total']}")
    logger.info(f"Known materials: {stats['known_count']} ({100*stats['known_ratio']:.1f}%)")
    logger.info(f"Novel materials: {stats['novel_count']} ({100*stats['novel_ratio']:.1f}%)")
    
    if known_structures:
        logger.info(f"\n{'Known materials found':}")
        for s in sorted(known_structures, key=lambda x: x.decomposition_energy or 0):
            ed = s.decomposition_energy if s.decomposition_energy else float('inf')
            logger.info(f"  - {s.formula}: Ed = {ed:.3f} eV/atom")
    
    if novel_structures:
        logger.info(f"\n{'Novel materials found':}")
        for s in sorted(novel_structures, key=lambda x: x.decomposition_energy or 0):
            ed = s.decomposition_energy if s.decomposition_energy else float('inf')
            logger.info(f"  - {s.formula}: Ed = {ed:.3f} eV/atom")
    
    logger.info(f"\n{'='*60}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*60}")
    
    if novel_structures:
        logger.info(f"Saving {len(novel_structures)} novel structures...")
        save_results(novel_structures, args.output, "novel")
    
    if known_structures:
        logger.info(f"Saving {len(known_structures)} known structures...")
        save_results(known_structures, args.output, "known")
    
    logger.info(f"Saving all {len(all_structures)} structures...")
    save_results(all_structures, args.output, "all")
    
    
    
    logger.info(f"\nSaving {len(all_structures)} structures to {args.output}")
    save_results(all_structures, args.output, "final")
    
    # Generate visualizations
    logger.info("\n" + "="*60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*60)
    
    visualizer = ResultsVisualizer()
    
    if novel_structures:
        logger.info("Creating energy distribution for novel materials...")
        visualizer.plot_energy_distribution(
            novel_structures,
            save_path=f"{args.output}/energy_distribution_novel.png"
        )
    
    logger.info("Creating energy distribution for all materials...")
    visualizer.plot_energy_distribution(
        all_structures,
        save_path=f"{args.output}/energy_distribution_all.png"
    )
    
    logger.info("Creating convergence plot...")
    visualizer.plot_convergence(
        orchestrator.search_history,
        save_path=f"{args.output}/convergence.png"
    )

    # ========== Calculate Hit Rate Metrics ==========
    logger.info("\n" + "="*60)
    logger.info("HIT RATE METRICS")
    logger.info("="*60)

    # Calculate Hit Rates using constraint checker
    N_total = len(all_structures)
    N_novel = len(novel_structures)
    generation_metrics = orchestrator.get_generation_metrics()
    N_requested_total = generation_metrics.get('requested_total', N_total)
    N_parse_success_total = generation_metrics.get('parse_success_total', N_total)

    # N_valid_all: structures satisfying ALL constraints
    valid_all = constraint_checker.filter_satisfied(all_structures)
    N_valid_all = len(valid_all)

    # N_valid_novel: novel AND satisfy constraints
    valid_novel = constraint_checker.filter_satisfied(novel_structures)
    N_valid_novel = len(valid_novel)

    # Calculate Hit Rates
    HitRate_all = N_valid_all / N_total if N_total > 0 else 0
    HitRate_all_strict = N_valid_all / N_requested_total if N_requested_total > 0 else 0
    HitRate_novel = N_valid_novel / N_novel if N_novel > 0 else 0
    NovelHitRate = N_valid_novel / N_total if N_total > 0 else 0
    ValidRate_all_parsed = N_valid_all / N_parse_success_total if N_parse_success_total > 0 else 0
    ValidRate_all_strict = N_valid_all / N_requested_total if N_requested_total > 0 else 0

    logger.info(f"\nConstraints: {len(constraints)} active")
    for line in constraints_text.split('\n'):
        logger.info(f"  {line}")

    logger.info(f"\nCounts:")
    logger.info(f"  Total structures (N_total): {N_total}")
    logger.info(f"  Valid structures (N_valid_all): {N_valid_all}")
    logger.info(f"  Novel structures (N_novel): {N_novel}")
    logger.info(f"  Valid novel (N_valid_novel): {N_valid_novel}")

    logger.info(f"\nHit Rates:")
    logger.info(f"  HitRate_all:    {HitRate_all:6.1%}  (overall success rate)")
    logger.info(f"  HitRate_all_strict: {HitRate_all_strict:6.1%}  (requested denominator)")
    logger.info(f"  HitRate_novel:  {HitRate_novel:6.1%}  (novel material quality)")
    logger.info(f"  NovelHitRate:   {NovelHitRate:6.1%}  (valuable discoveries ⭐)")
    logger.info(f"  ValidRate_all_parsed: {ValidRate_all_parsed:6.1%}")
    logger.info(f"  ValidRate_all_strict: {ValidRate_all_strict:6.1%}")

    # Show some examples of Hit vs Miss
    if valid_all:
        logger.info(f"\nTop 3 Hits (satisfy all constraints):")
        for s in sorted(valid_all, key=lambda x: x.decomposition_energy)[:3]:
            logger.info(f"  ✓ {s.formula}: Ed = {s.decomposition_energy:.3f} eV/atom")

    # ========== End Hit Rate Calculation ==========

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("FINAL STATISTICS - ALL STRUCTURES")
    logger.info("="*60)

    # Get metastability threshold from config (use same threshold as orchestrator)
    metastability_threshold = config.get('thresholds', {}).get('metastability', 0.1)

    valid = [s for s in all_structures if s.is_valid]
    metastable = [s for s in valid if s.decomposition_energy <= metastability_threshold]
    stable = [s for s in metastable if s.decomposition_energy < 0]

    logger.info(f"Total generated: {len(all_structures)}")

    if len(all_structures) > 0:
        logger.info(f"Valid: {len(valid)} ({100*len(valid)/len(all_structures):.1f}%)")
        logger.info(f"Metastable (Ed≤{metastability_threshold} eV/atom): {len(metastable)} ({100*len(metastable)/len(all_structures):.1f}%)")
        logger.info(f"Stable (Ed<0): {len(stable)} ({100*len(stable)/len(all_structures):.1f}%)")
    
    logger.info("\n" + "="*60)
    logger.info("FINAL STATISTICS - NOVEL MATERIALS ONLY")
    logger.info("="*60)
    
    if novel_structures:
        novel_valid = [s for s in novel_structures if s.is_valid]
        novel_stable = [s for s in novel_valid if s.decomposition_energy < 0]
        
        logger.info(f"Novel structures: {len(novel_structures)}")
        logger.info(f"Novel valid: {len(novel_valid)} ({100*len(novel_valid)/len(novel_structures):.1f}%)")
        logger.info(f"Novel stable: {len(novel_stable)} ({100*len(novel_stable)/len(novel_structures):.1f}%)")
        
        if novel_stable:
            best_novel = min(novel_stable, key=lambda s: s.decomposition_energy)
            logger.info(f"\n🌟 BEST NOVEL STRUCTURE:")
            logger.info(f"   Formula: {best_novel.formula}")
            logger.info(f"   Decomposition energy: {best_novel.decomposition_energy:.3f} eV/atom")
            logger.info(f"   This material is NOT in Materials Project database!")
    else:
        logger.warning("⚠️  No novel materials discovered in this run!")
        logger.info("All generated structures are known materials.")
        logger.info("Consider adjusting the search strategy or prompt.")
    
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"  - novel_*.json: {len(novel_structures)} novel materials")
    logger.info(f"  - known_*.json: {len(known_structures)} known materials")
    logger.info(f"  - all_*.json: {len(all_structures)} total materials")
    if novelty_details:
        logger.info("  - novelty_report.json: Novelty detection details")
    logger.info(f"  - *.png: Visualization plots")


    

    logger.info("\n" + "="*60)
    logger.info("FINAL STATISTICS")
    logger.info("="*60)

    # Get metastability threshold from config
    metastability_threshold = config.get('thresholds', {}).get('metastability', 0.1)

    valid = [s for s in all_structures if s.is_valid]
    metastable = [s for s in valid if s.decomposition_energy <= metastability_threshold]
    stable = [s for s in metastable if s.decomposition_energy < 0]
    
    logger.info(f"Total generated: {len(all_structures)}")


    if len(all_structures) > 0:
        logger.info(f"Valid: {len(valid)} ({100*len(valid)/len(all_structures):.1f}%)")
        logger.info(f"Metastable (Ed≤{metastability_threshold} eV/atom): {len(metastable)} ({100*len(metastable)/len(all_structures):.1f}%)")
        logger.info(f"Stable (Ed<0): {len(stable)} ({100*len(stable)/len(all_structures):.1f}%)")
        
        if stable:
            best = min(stable, key=lambda s: s.decomposition_energy)
            logger.info(f"\nBest structure: {best.formula}")
            logger.info(f"Decomposition energy: {best.decomposition_energy:.3f} eV/atom")
    else:
        logger.warning("No structures were generated in this run.")
    
    logger.info(f"\nResults saved to: {args.output}")

    # Calculate diversity metrics
    unique_formulas_count = len(set(s.formula for s in all_structures)) if all_structures else 0

    # Calculate unique structures using StructureMatcher
    unique_structures_count = len(all_structures)  # Default: all structures are unique
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
        unique_structures = []

        for struct in all_structures:
            is_duplicate = False
            try:
                pmg_struct = struct.to_pymatgen()
                for unique_struct in unique_structures:
                    unique_pmg = unique_struct.to_pymatgen()
                    if matcher.fit(pmg_struct, unique_pmg):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_structures.append(struct)
            except Exception as e:
                logger.debug(f"Structure matching failed for {struct.formula}: {e}")
                unique_structures.append(struct)  # Keep it if matching fails

        unique_structures_count = len(unique_structures)
        logger.info(f"\nDiversity Analysis:")
        logger.info(f"  Total structures: {len(all_structures)}")
        logger.info(f"  Unique structures (by StructureMatcher): {unique_structures_count}")
        logger.info(f"  Unique formulas: {unique_formulas_count}")
        logger.info(f"  Diversity rate: {100 * unique_structures_count / len(all_structures):.1f}%" if all_structures else "  Diversity rate: N/A")

    except ImportError:
        logger.warning("pymatgen not available for diversity calculation, using formula-based count")
        unique_structures_count = unique_formulas_count

    diversity_rate = unique_structures_count / len(all_structures) if all_structures else 0.0

    # Save comprehensive summary with metadata
    summary_file = Path(args.output) / "run_summary.json"

    summary_data = {
        'metadata': {
            'experiment_type': 'matagent_full_run',
            'timestamp': datetime.now().isoformat(),
            'hostname': platform.node(),
            'python_version': platform.python_version(),
            'task': {
                'name': task_name,
                'full_name': task_config['name'],
                'description': task_config['description'],
                'constraints': constraints
            },
            'config': {
                'population_size': config['evolution']['population_size'],
                'parent_size': config['evolution']['parent_size'],
                'children_size': config['evolution']['children_size'],
                'max_iterations': config['evolution']['max_iterations'],
                'llm_provider': config['llm']['provider'],
                'llm_model': config['llm']['model'],
                'llm_temperature': config['llm']['temperature'],
                'ablation_mode': config.get('ablation_mode', 'crystal_forge'),
                'evaluator_backend': config['evaluator']['backend'],
                'evaluator_model': config['evaluator'].get('model_name', 'N/A'),
                'relax_fmax': config['evaluator']['relax']['fmax'],
                'relax_steps': config['evaluator']['relax']['steps'],
                'energy_cutoff': config['evaluator']['energy_cutoff'],
                'hierarchical_enabled': bool(
                    (config.get('reflection', {}).get('hierarchical')
                     or config.get('reflection', {}).get('hierachical')
                     or {}).get('enabled', False)
                )
            }
        },
        'novelty_detection': {
            'use_structure_matching': use_structure_matching,
            'mode': novelty_mode,
            'tier_counts': novelty_tier_counts,
            'details_file': "novelty_report.json" if novelty_details else None
        },
        'results': {
            'total_structures': len(all_structures),
            'valid_structures': len(valid),
            'metastable_structures': len(metastable),
            'stable_structures': len(stable),
            'novel_structures': len(novel_structures),
            'known_structures': len(known_structures),
            'metastable_rate': 100 * len(metastable) / len(all_structures) if all_structures else 0,
            'stable_rate': 100 * len(stable) / len(all_structures) if all_structures else 0,
            'novel_rate': 100 * len(novel_structures) / len(all_structures) if all_structures else 0
        },
        'generation_metrics': {
            'requested_total': int(N_requested_total),
            'parse_success_total': int(N_parse_success_total),
            'parse_failure_total': int(generation_metrics.get('parse_failure_total', max(0, N_requested_total - N_parse_success_total))),
            'parse_success_rate': float(generation_metrics.get('parse_success_rate', (N_parse_success_total / N_requested_total) if N_requested_total > 0 else 0.0))
        },
        'valid_rate_metrics': {
            'N_valid_all': N_valid_all,
            'ValidRate_all_parsed': float(ValidRate_all_parsed),
            'ValidRate_all_strict': float(ValidRate_all_strict)
        },
        'diversity_metrics': {
            'unique_structures_count': unique_structures_count,
            'unique_formulas_count': unique_formulas_count,
            'diversity_rate': float(diversity_rate),
            'diversity_rate_percent': float(100 * diversity_rate),
            'description': 'Diversity metrics measure internal uniqueness (vs novelty which compares to Materials Project)'
        },
        'hit_rate_metrics': {
            'N_total': N_total,
            'N_requested_total': int(N_requested_total),
            'N_valid_all': N_valid_all,
            'N_novel': N_novel,
            'N_valid_novel': N_valid_novel,
            'HitRate_all': float(HitRate_all),
            'HitRate_all_strict': float(HitRate_all_strict),
            'HitRate_novel': float(HitRate_novel),
            'NovelHitRate': float(NovelHitRate)
        },
        'best_structures': {
            'best_overall': {
                'formula': best.formula if stable else None,
                'decomposition_energy': float(best.decomposition_energy) if stable else None,
                'energy': float(best.energy) if stable else None
            } if stable else None,
            'best_novel': {
                'formula': best_novel.formula if novel_structures and 'best_novel' in locals() else None,
                'decomposition_energy': float(best_novel.decomposition_energy) if novel_structures and 'best_novel' in locals() else None,
                'energy': float(best_novel.energy) if novel_structures and 'best_novel' in locals() else None
            } if novel_structures and 'best_novel' in locals() else None
        },
        'hmrr_metrics': orchestrator.get_hmrr_metrics(),
        'files': {
            'novel_structures': f"novel_*.json ({len(novel_structures)} files)",
            'known_structures': f"known_*.json ({len(known_structures)} files)",
            'all_structures': f"all_*.json ({len(all_structures)} files)",
            'novelty_report': "novelty_report.json" if novelty_details else None,
            'visualizations': "*.png",
            'history': f"history_*.json",
            'log': "run.log"
        }
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
