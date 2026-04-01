# CrystalSTAR

**Crystal Structure Tool-Augmented Reasoning** -- an agentic framework for autonomous crystal structure discovery using Large Language Models.

*Paper under review.*

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

| Task | Target Properties |
|------|-------------------|
| Wide-Bandgap Semiconductors | Band gap 3.0-6.0 eV |
| SAW/BAW Acoustic Substrates | Bulk modulus 100-250 GPa, Shear 50-150 GPa |
| High-k Dielectrics | Dielectric constant 10-90, Band gap 2.5-6.5 eV |
| Solid-State Electrolytes | Band gap 4.0-8.0 eV |
| Piezoelectric Harvesters | Piezo coeff. 2-20 C/m2, Band gap >= 2.0 eV |
| Photovoltaic Absorbers | Band gap 1.0-1.8 eV |
| Hard Coatings | Bulk 200-500 GPa, Shear 100-300 GPa |
| Hard, Stiff Ceramics | Bulk 200-400 GPa, Shear 120-250 GPa |
| Aerospace Materials | Bulk 80-160 GPa, Shear 40-100 GPa |
| Acousto-Optic Hybrids | Bulk 50-150 GPa, Band gap 2.0-5.0 eV |
| Low-Density Structural | Bulk 30-120 GPa, Density 1.5-4.0 g/cm3 |
| Toxic-Free Perovskite Oxides | Band gap >= 2.0, Bulk 90-135 GPa |
| Insulating Dielectrics | Dielectric 5-50, Band gap >= 5.0 eV |
| Transparent Conductors | Band gap 2.5-4.5 eV |

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
