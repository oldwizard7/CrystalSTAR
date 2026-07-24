# [COLM 2026] CrystalSTAR

### Structured Action Orchestration with Trio-Reflection for Constrained Novel Crystal Discovery

## Overview

CrystalSTAR combines LLM-based reasoning with domain-specific crystal manipulation tools and a multi-scale memory module (M3) to discover novel crystal structures satisfying user-defined property constraints.

Key features:
- **Tool-augmented generation**: Four crystal manipulation tools (Substitute, Mutate, Mix, Prototype) ground LLM actions in physically valid transformations
- **Multi-scale Memory (M3)**: Hierarchical memory (Micro/Meso/Macro) that accumulates search experience across iterations and runs
- **Surrogate evaluation**: CHGNet for stability + ALIGNN for target properties (band gap, bulk modulus, etc.)
- **14 benchmark tasks** spanning electronic, mechanical, optical, and thermal property domains

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key (for LLM backbone)
- Materials Project API key (for novelty checking and parent pool retrieval)

### Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate crystalstar

# Or install manually
pip install pymatgen chgnet mp-api openai

# Set API keys
export OPENAI_API_KEY="your-key"
export MP_API_KEY="your-key"
```

## Quick Start

```bash
# Run on a single task (e.g., High-k Dielectrics)
python -m src.main --config config/config.yaml --task high_k_dielectrics

# Run with CPU-only profile (fewer iterations, for testing)
python -m src.main --config config/config.yaml --task high_k_dielectrics --profile login

# List available tasks
python -m src.main --list-tasks
```

## Benchmark Tasks

All 14 tasks are defined in `config/tasks.yaml`:

| Task | Hard Constraints |
|------|------------------|
| Wide-Bandgap Semiconductors | Band gap >= 2.5 eV; formation energy <= -1.0 eV/atom |
| SAW/BAW Acoustic Substrates | Shear modulus 25-150 GPa; dielectric constant 3.7-95 |
| High-k Dielectrics | Dielectric constant 10-90; band gap 2.5-6.5 eV |
| Solid-State Electrolytes | Band gap >= 2.0 eV; formation energy <= -1.0 eV/atom |
| Piezoelectric Harvesters | Piezoelectric coefficient >= 8 pC/N; dielectric constant 10-8000 |
| Photovoltaic Absorbers | Band gap 0.7-2.0 eV; formation energy <= 0 eV/atom |
| Hard Coatings | Bulk modulus 200-500 GPa; shear modulus 100-300 GPa |
| Hard, Stiff Ceramics | Bulk modulus 100-300 GPa; shear modulus 60-200 GPa |
| Aerospace Materials | Bulk modulus >= 100 GPa; shear modulus >= 40 GPa; density <= 5.0 g/cm3 |
| Acousto-Optic Hybrids | Piezoelectric coefficient 2-9 pC/N; dielectric constant 8-95 |
| Low-Density Structural Materials | Density <= 3.5 g/cm3; shear modulus 65-195 GPa |
| Toxic-Free Perovskite Oxides | Band gap >= 2.0 eV; bulk modulus 90-135 GPa |
| Insulating Dielectrics | Dielectric constant >= 8.0; band gap >= 2.5 eV |
| Transparent Conductors | Band gap >= 3.0 eV |

## Configuration

Main configuration: `config/config.yaml`

Key settings:
- `reflection.hierarchical.enabled`: Enable/disable M3 memory module
- `ablation_mode`: `crystal_forge` (full M3) / `vanilla_react` (no memory) / `flat_memory` (unstructured memory)
- `generation_mode`: `tool_calling` (default) / `direct_llm` (ablation baseline)
- `evaluator.backend`: `chgnet` (recommended) / `m3gnet` / `orb`

## Project Structure

```
CrystalSTAR/
├── src/
│   ├── main.py                 # Entry point
│   ├── agents/
│   │   ├── orchestrator.py     # Main search loop
│   │   ├── tool_selector.py    # LLM tool selection with M3 context
│   │   ├── hypothesis_gen.py   # Strategy generation
│   │   └── workers/            # Structure generation/parsing agents
│   ├── core/
│   │   ├── evaluator.py        # Surrogate evaluation pipeline
│   │   ├── operators.py        # Crystal manipulation tools
│   │   ├── evolution.py        # Parent pool management
│   │   ├── constraints.py      # Property constraint checking
│   │   ├── novelty_checker.py  # MP-based novelty detection
│   │   └── structure.py        # Crystal structure dataclass
│   ├── utils/
│   │   ├── hier_memory.py      # M3 hierarchical memory implementation
│   │   ├── llm_client.py       # LLM API wrapper
│   │   └── ...
│   └── surrogate_models/
│       └── alignn_wrapper.py   # ALIGNN property prediction
├── config/
│   ├── config.yaml             # Main configuration
│   ├── tasks.yaml              # 14 benchmark task definitions
│   └── prompts.yaml            # LLM prompt templates
└── scripts/
    └── initial_parent_pool/    # Parent pool construction from MP
```

## Reproducing Results

1. Configure API keys (OpenAI + Materials Project)
2. Build parent pools: `python scripts/initial_parent_pool/download_task_specific_structures.py --task <task_name>`
3. Run search: `python -m src.main --config config/config.yaml --task <task_name> --profile gpu`
4. Results are saved to `output/`

## License

MIT
