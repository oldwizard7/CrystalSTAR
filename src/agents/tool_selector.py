"""Tool Selection Agent - Decides which tools to use for structure generation

Uses LLM to intelligently select and parameterize tools based on:
- Current search strategy
- Parent structures
- Historical performance
"""

import logging
import json
import re
import math
from typing import List, Dict, Optional
from src.core.structure import CrystalStructure
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ToolSelectionAgent:
    """
    Agent that decides which tools to use for generation

    Implements intelligent tool selection using LLM reasoning
    """

    def __init__(self, config: Dict):
        """
        Initialize tool selection agent

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm = LLMClient(config['llm'])
        self.template_catalog: List[Dict] = []

    def set_template_catalog(self, templates: List[Dict]) -> None:
        self.template_catalog = templates or []

    def _get_disabled_tools(self) -> set:
        disabled = self.config.get("tool_selector", {}).get("disabled_tools", []) or []
        return {str(tool).strip().lower() for tool in disabled}

    def _filter_disabled_tools(self, actions: List[Dict]) -> List[Dict]:
        disabled = self._get_disabled_tools()
        if not disabled:
            return actions
        filtered = [
            action for action in actions
            if str(action.get("tool", "")).strip().lower() not in disabled
        ]
        dropped = len(actions) - len(filtered)
        if dropped:
            logger.warning(f"Dropped {dropped} disabled tool actions: {sorted(disabled)}")
        return filtered

    def _persist_tool_selector_artifacts(
        self,
        output_dir: Optional[str],
        iteration: Optional[int],
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        actions: Optional[List[Dict]] = None,
    ) -> None:
        if not output_dir or iteration is None:
            return

        from pathlib import Path

        tool_selector_dir = Path(output_dir) / f"iteration_{iteration}" / "tool_selector"
        tool_selector_dir.mkdir(parents=True, exist_ok=True)

        if prompt is not None:
            (tool_selector_dir / "prompt.txt").write_text(prompt)
        if response is not None:
            (tool_selector_dir / "output.txt").write_text(response)
        if actions is not None:
            with open(tool_selector_dir / "actions.json", "w", encoding="utf-8") as f:
                json.dump(actions, f, indent=2)

    def _build_empty_pool_fallback_prompt(
        self,
        n_structures: int,
        reflection: Optional[Dict],
        iteration: Optional[int],
        max_iterations: Optional[int],
    ) -> str:
        task_description = self.config.get("task_description", "")
        disabled_tools = sorted(self._get_disabled_tools())

        prompt_parts = [
            "EMPTY_PARENT_POOL_FALLBACK",
            "",
            "Tool selector LLM prompt was skipped because the parent pool is empty.",
            "Fallback prototype-based action generation was used instead.",
            "",
        ]

        if task_description:
            prompt_parts.extend([
                "=" * 60,
                "TASK BACKGROUND:",
                "=" * 60,
                task_description,
                "",
            ])

        if iteration is not None and max_iterations is not None:
            prompt_parts.extend([
                f"ITERATION: {iteration}/{max_iterations}",
                "",
            ])

        if reflection:
            prompt_parts.extend([
                "=" * 60,
                "AVAILABLE CONTEXT:",
                "=" * 60,
                f"Previous valid rate: {reflection.get('valid_rate', 0):.1f}%",
                f"Previous hit rate: {reflection.get('hit_rate', 0):.1f}%",
                f"Previous diversity: {reflection.get('diversity', 0):.2f}",
                "",
            ])

        prompt_parts.extend([
            "=" * 60,
            "AVAILABLE TOOLS:",
            "=" * 60,
            "Only fill_prototype is usable without parent structures.",
            "",
        ])
        if disabled_tools:
            prompt_parts.extend([
                f"Disabled tools: {', '.join(disabled_tools)}",
                "",
            ])

        prompt_parts.extend([
            "=" * 60,
            "TEMPLATE CATALOG USED FOR FALLBACK:",
            "=" * 60,
        ])
        if self.template_catalog:
            for idx, tpl in enumerate(self.template_catalog, 1):
                name = tpl.get("name")
                elements = tpl.get("elements") or []
                formula = tpl.get("reference_formula") or tpl.get("best_for") or "N/A"
                if not name or not elements:
                    continue
                prompt_parts.append(
                    f"  {idx}. {name} (reference={formula}, required_elements={len(elements)}, elements={elements})"
                )
        else:
            prompt_parts.append("  No templates available.")
        prompt_parts.extend([
            "",
            "=" * 60,
            "FALLBACK REQUEST:",
            "=" * 60,
            f"Generate EXACTLY {n_structures} prototype-based actions using the available templates.",
            "No parent-conditioned tool selection was possible in this iteration.",
            "",
        ])

        return "\n".join(prompt_parts)

    def _build_fallback_output(self, reason: str, actions: List[Dict]) -> str:
        lines = [
            f"Fallback reason: {reason}",
            f"Generated actions: {len(actions)}",
            "",
        ]
        for idx, action in enumerate(actions, 1):
            lines.append(f"{idx}. {action}")
        return "\n".join(lines)

    def select_tools(
        self,
        example_pool: List[CrystalStructure],
        strategy: Optional[str],
        reflection: Dict,
        n_structures: int,
        output_dir: str = None,  # Output directory for saving prompts
        iteration: int = None,     # Current iteration number
        all_structures: List[CrystalStructure] = None,  # All structures for diversity tracking
        strategies_data: Dict = None,  # Strategies dict with target_ed_range etc.
        max_iterations: int = None,  # Total iterations for context
        hmrr_context: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Select tools and parameters for structure generation

        Args:
            example_pool: Available parent structures
            strategy: Current search strategy from curate phase
            reflection: Performance reflection from last iteration
            n_structures: Number of structures to generate

        Returns:
            List of tool actions:
            [
                {"tool": "substitute", "parent": 0, "old_element": "O", "new_element": "S"},
                {"tool": "mutate", "parent": 1, "strength": 0.1},
                {"tool": "fill_prototype", "template": "Rock-salt", "elements": ["Na", "Cl"]}
            ]
        """
        logger.info("Tool selector analyzing state and choosing tools...")
        disabled_tools = self._get_disabled_tools()

        # Handle empty pool case
        if len(example_pool) == 0:
            prompt = self._build_empty_pool_fallback_prompt(
                n_structures=n_structures,
                reflection=reflection,
                iteration=iteration,
                max_iterations=max_iterations,
            )
            if "fill_prototype" in disabled_tools:
                logger.error("fill_prototype disabled but no parents available; cannot generate structures.")
                self._persist_tool_selector_artifacts(
                    output_dir=output_dir,
                    iteration=iteration,
                    prompt=prompt,
                    response="Fallback failed: fill_prototype is disabled and no parents are available.\n",
                    actions=[],
                )
                return []
            logger.info("Empty parent pool - using prototype-based generation")
            actions = self._fallback_selection(example_pool, n_structures)
            logger.info(f"Selected {len(actions)} fallback tool actions (empty parent pool)")
            for action in actions:
                logger.info(f"  - {action['tool']}: {action}")
            self._persist_tool_selector_artifacts(
                output_dir=output_dir,
                iteration=iteration,
                prompt=prompt,
                response=self._build_fallback_output(
                    reason="empty parent pool; used prototype-based generation",
                    actions=actions,
                ),
                actions=actions,
            )
            return actions

        # Build decision prompt
        mode = self._get_ablation_mode()
        previous_reflection_prompt = None
        past_raw_trajectories = None
        if mode in {"vanilla_react", "crystal_forge"}:
            previous_reflection_prompt = self._load_previous_reflection_prompt(output_dir, iteration)
        elif mode == "flat_memory":
            past_raw_trajectories = self._load_past_raw_trajectories(output_dir, iteration)
        prompt = self._build_decision_prompt(
            example_pool, strategy, reflection, n_structures,
            all_structures, strategies_data, iteration, max_iterations,
            previous_reflection_prompt, hmrr_context, past_raw_trajectories
        )

        # Print complete prompt to console
        logger.info("="*60)
        logger.info("TOOL SELECTOR PROMPT:")
        logger.info("="*60)
        logger.info(prompt)
        logger.info("="*60)

        # Save prompt to file
        if output_dir and iteration is not None:
            self._persist_tool_selector_artifacts(
                output_dir=output_dir,
                iteration=iteration,
                prompt=prompt,
            )

        # Get LLM decision
        try:
            responses = self.llm.generate(prompt, n=1)
            if not responses or not responses[0]:
                raise RuntimeError("LLM returned empty response for tool selection")
            response = responses[0]

            # Print complete response to console
            logger.info("="*60)
            logger.info("TOOL SELECTOR LLM OUTPUT:")
            logger.info("="*60)
            logger.info(response)
            logger.info("="*60)

            # Save LLM response to file
            if output_dir and iteration is not None:
                self._persist_tool_selector_artifacts(
                    output_dir=output_dir,
                    iteration=iteration,
                    response=response,
                )

            # Parse response
            actions = self._parse_tool_actions(response, len(example_pool))
            actions = self._enforce_template_element_count(actions)
            actions = self._filter_disabled_tools(actions)

            # Validate number of actions
            if len(actions) != n_structures:
                logger.warning(
                    f"LLM returned {len(actions)} actions, expected {n_structures}. "
                    f"Using fallback to fill remaining {n_structures - len(actions)} actions."
                )
                # Keep the valid actions and fill the rest with fallback
                if len(actions) < n_structures:
                    remaining = n_structures - len(actions)
                    fallback_actions = self._fallback_selection(example_pool, remaining)
                    actions.extend(fallback_actions)
                else:
                    # Too many actions, truncate
                    actions = actions[:n_structures]

            logger.info(f"Selected {len(actions)} tool actions:")
            for action in actions:
                logger.info(f"  - {action['tool']}: {action}")

            # Save parsed actions to file
            if output_dir and iteration is not None:
                self._persist_tool_selector_artifacts(
                    output_dir=output_dir,
                    iteration=iteration,
                    actions=actions,
                )

            return actions

        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            # Fallback: generate all actions
            logger.warning("Using fallback tool selection for all actions")
            actions = self._fallback_selection(example_pool, n_structures)
            # Persist fallback actions so runs remain auditable even when LLM calls fail.
            if output_dir and iteration is not None:
                self._persist_tool_selector_artifacts(
                    output_dir=output_dir,
                    iteration=iteration,
                    response=self._build_fallback_output(
                        reason=f"tool selection failed: {e}",
                        actions=actions,
                    ),
                    actions=actions,
                )
            return actions

    def _enforce_template_element_count(self, actions: List[Dict]) -> List[Dict]:
        if not self.template_catalog:
            return actions

        expected_counts: Dict[str, int] = {}
        for tpl in self.template_catalog:
            name = tpl.get("name")
            elements = tpl.get("elements")
            if not name or not elements:
                continue
            expected_counts[str(name).strip().lower()] = len(elements)

        filtered: List[Dict] = []
        for action in actions:
            if action.get("tool") != "fill_prototype":
                filtered.append(action)
                continue

            template = str(action.get("template", "")).strip()
            elements = action.get("elements") or []
            expected = expected_counts.get(template.lower())
            if expected is None:
                logger.warning(
                    f"Unknown template '{template}' for fill_prototype; dropping action"
                )
                continue
            if len(elements) != expected:
                logger.warning(
                    f"{template} expects {expected} elements; got {len(elements)}. "
                    f"Dropping action to avoid truncation"
                )
                continue
            filtered.append(action)

        return filtered

    def _get_prompt_layout(self) -> str:
        reflection_cfg = (self.config.get("reflection", {}) or {})
        hier_cfg = reflection_cfg.get("hierarchical")
        if hier_cfg is None and "hierachical" in reflection_cfg:
            hier_cfg = reflection_cfg.get("hierachical")
        hier_cfg = hier_cfg or {}
        layout = str(hier_cfg.get("prompt_layout", "m3_explicit")).strip().lower()
        if layout not in {"m3_explicit", "legacy"}:
            layout = "m3_explicit"
        return layout

    def _get_ablation_mode(self) -> str:
        mode = str(self.config.get("ablation_mode", "crystal_forge")).strip().lower()
        if mode not in {"memory_less", "vanilla_react", "flat_memory", "crystal_forge"}:
            mode = "crystal_forge"
        return mode

    def _build_m3_backbone(
        self,
        hmrr_context: Dict,
        reflection: Optional[Dict],
        strategies_data: Optional[Dict],
        all_structures: Optional[List[CrystalStructure]],
    ) -> List[str]:
        lines: List[str] = [
            "=" * 60,
            "M3 DECISION BACKBONE",
            "=" * 60,
            "",
            "=" * 60,
            "[Macro Prior]",
            "=" * 60,
        ]

        macro_rules = hmrr_context.get("macro_rules", []) or []
        if macro_rules:
            for idx, rec in enumerate(macro_rules, 1):
                confidence = rec.get("confidence")
                conf_text = f" (conf={confidence:.2f})" if isinstance(confidence, (int, float)) else ""
                lines.append(f"  {idx}. {rec.get('message', '')}{conf_text}")
        else:
            lines.append("  - No archived macro prior yet for this task.")
        lines.append("")

        lines.extend([
            "=" * 60,
            "[Meso Tactical Policy]",
            "=" * 60,
        ])
        meso_added = False
        meso_tips = hmrr_context.get("meso_tips", []) or []
        if meso_tips:
            for idx, rec in enumerate(meso_tips, 1):
                lines.append(f"  {idx}. {rec.get('message', '')}")
            meso_added = True

        if reflection:
            tool_stats = reflection.get("tool_statistics", {}) or {}
            ranked_tools = tool_stats.get("ranked", []) or []
            if ranked_tools:
                best_tool, best_perf = ranked_tools[0]
                lines.append(
                    f"  - Tool signal: prioritize {best_tool} (hit={best_perf.get('hit_rate', 0.0):.1f}%, valid={best_perf.get('valid_rate', 0.0):.1f}%)."
                )
                meso_added = True

        if strategies_data and isinstance(strategies_data.get("strategies"), list) and strategies_data["strategies"]:
            highest = max(strategies_data["strategies"], key=lambda s: s.get("allocation_weight", 0))
            lines.append(
                f"  - Strategic focus: {highest.get('strategy_name', 'N/A')} ({highest.get('strategy_type', 'N/A')})."
            )
            meso_added = True

        if all_structures:
            from collections import Counter
            formulas = [s.formula for s in all_structures if hasattr(s, "formula")]
            repeated = [(f, c) for f, c in Counter(formulas).most_common(1) if c >= 2]
            if repeated:
                formula, count = repeated[0]
                lines.append(f"  - Repetition control: avoid overusing {formula} ({count}x).")
                meso_added = True

        if not meso_added:
            lines.append("  - No meso tactical policy available yet.")
        lines.append("")

        lines.extend([
            "=" * 60,
            "[Micro Immediate Signals]",
            "=" * 60,
        ])
        micro_events = hmrr_context.get("micro_events", []) or []
        if micro_events:
            for idx, rec in enumerate(micro_events, 1):
                lines.append(f"  {idx}. {rec.get('message', '')}")
        else:
            lines.append("  - No immediate micro signals yet.")
        lines.append("")

        return lines

    def _split_llm_analysis_sections(self, llm_analysis: str) -> Dict[str, str]:
        """Split reflection analysis into Micro/Meso/Other buckets by heading."""
        if not llm_analysis:
            return {"micro": "", "meso": "", "other": ""}

        sections: List[tuple[str, List[str]]] = []
        current_heading = ""
        current_lines: List[str] = []

        for raw_line in llm_analysis.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("###"):
                if current_heading or current_lines:
                    sections.append((current_heading, current_lines))
                current_heading = stripped
                current_lines = []
                continue
            current_lines.append(line)

        if current_heading or current_lines:
            sections.append((current_heading, current_lines))

        micro_chunks: List[str] = []
        meso_chunks: List[str] = []
        other_chunks: List[str] = []

        for heading, body_lines in sections:
            heading_norm = heading.lstrip("#").strip().lower() if heading else ""
            body = "\n".join(body_lines).strip()
            chunk = heading.strip() if heading else ""
            if body:
                chunk = f"{chunk}\n\n{body}" if chunk else body
            chunk = chunk.strip()
            if not chunk:
                continue

            if "individual structure analysis" in heading_norm:
                micro_chunks.append(chunk)
                continue

            if (
                "overall success patterns" in heading_norm
                or "overall failure patterns" in heading_norm
                or "strategic recommendations for next iteration" in heading_norm
            ):
                meso_chunks.append(chunk)
                continue

            other_chunks.append(chunk)

        return {
            "micro": "\n\n".join(micro_chunks).strip(),
            "meso": "\n\n".join(meso_chunks).strip(),
            "other": "\n\n".join(other_chunks).strip(),
        }

    def _prefer_prior_templates(self, actions: List[Dict]) -> None:
        if not self.template_catalog:
            return
        template_cfg = (
            self.config.get("data", {})
            .get("initial_parent_pool", {})
            .get("template_library", {})
        )
        prefer_prior_templates = template_cfg.get("prefer_prior_templates", False)
        force_prior_templates = template_cfg.get("force_prior_templates", False)
        if not prefer_prior_templates and not force_prior_templates:
            return

        by_count: Dict[int, List[str]] = {}
        by_name: Dict[str, Dict] = {}
        for tpl in self.template_catalog:
            name = tpl.get("name")
            elements = tpl.get("elements")
            if not name or not elements:
                continue
            count = len(elements)
            by_count.setdefault(count, []).append(name)
            by_name[name] = tpl

        fill_idx = 0
        for action in actions:
            if action.get("tool") != "fill_prototype":
                continue
            template = str(action.get("template", ""))
            if template.lower().startswith("prior"):
                continue
            elements = action.get("elements") or []
            count = len(elements)
            candidates = by_count.get(count)
            if not candidates:
                if not force_prior_templates:
                    continue
                all_templates = [
                    (len(tpl.get("elements", [])), tpl.get("name"), tpl)
                    for tpl in self.template_catalog
                    if tpl.get("name")
                ]
                if not all_templates:
                    continue
                all_templates.sort(key=lambda x: abs(x[0] - count))
                _, chosen_name, chosen_tpl = all_templates[0]
                action["template"] = chosen_name
                ref_elements = chosen_tpl.get("elements") or []
                if ref_elements:
                    action["elements"] = ref_elements
                continue
            action["template"] = candidates[fill_idx % len(candidates)]
            fill_idx += 1

    def _build_decision_prompt(
        self,
        parents: List[CrystalStructure],
        strategy: Optional[str],
        reflection: Dict,
        n: int,
        all_structures: List[CrystalStructure] = None,
        strategies_data: Dict = None,
        iteration: int = None,
        max_iterations: int = None,
        previous_reflection_prompt: Optional[str] = None,
        hmrr_context: Optional[Dict] = None,
        past_raw_trajectories: Optional[str] = None,
    ) -> str:
        """Build prompt for tool selection"""

        disable_defaults = True
        disabled_tools = self._get_disabled_tools()
        template_example = "template_01"
        if self.template_catalog:
            for tpl in self.template_catalog:
                name = tpl.get("name")
                if name:
                    template_example = name
                    break
        enabled_tools = [
            tool for tool in ("substitute", "fill_prototype", "mutate", "mix")
            if tool not in disabled_tools
        ]
        prompt_parts = [
            "You are a materials scientist deciding how to generate new crystal structures.",
            f"You have {len(enabled_tools)} specialized tools.",
            "",
        ]
        task_description = self.config.get("task_description", "")
        if task_description:
            prompt_parts.extend([
                "="*60,
                "TASK BACKGROUND:",
                "="*60,
                task_description,
                "",
            ])
        prompt_parts.extend([
            "="*60,
            "AVAILABLE TOOLS:",
            "="*60,
            "Classification rule: follow the dominant control variable (primary intent);",
            "coupled composition/structure side-effects are expected.",
            "",
        ])
        if "substitute" not in disabled_tools:
            prompt_parts.extend([
                "1. substitute(parent, old_element, new_element)",
                "   - Replace element(s) in a parent structure (supports multi-element: comma-separated)",
                "   - Examples: substitute(parent=0, old='Ti', new='Zr')",
                "              substitute(parent=0, old='Ti,O', new='Zr,S')",
                "   - Use when: composition-local refinement (tune elements while mostly preserving scaffold)",
                "",
            ])
        if "fill_prototype" not in disabled_tools:
            prompt_parts.extend([
                "2. fill_prototype(template, elements)",
                "   - Build a structure from a task-specific template with stable coordinates",
                "   - Templates: use ONLY the task templates listed below",
                "   - IMPORTANT: The number of elements MUST exactly match the template's required count",
                "   - Use when: structure-global jump (switch topology and adapt composition to template slots)",
                "",
            ])
        if "mutate" not in disabled_tools:
            prompt_parts.extend([
                "3. mutate(parent, strength)",
                "   - Randomly perturb structure (strength: 0.01-0.1)",
                "   - Use when: structure-local refinement (perturb coordinates/lattice while preserving topology)",
                "",
            ])
        if "mix" not in disabled_tools:
            prompt_parts.extend([
                "4. mix(parent1, parent2, ratio)",
                "   - Combine two parent structures with a chosen ratio (ratio=0-1, default 0.5)",
                "   - At most ONE mix per iteration",
                "   - Use when: composition-global recombination (expand chemical system with coupled structure changes)",
                "",
            ])
        if disabled_tools:
            prompt_parts.extend([
                f"DISABLED TOOLS: {', '.join(sorted(disabled_tools))}",
                "",
            ])
        prompt_parts.extend([
            "CHEMICAL SANITY CHECK (brief):",
            "  - Prefer charge-balanced ratios for ionic/covalent systems",
            "  - Use small integer ratios (1:1, 1:2, 2:3)",
            "  - Avoid large atomic size mismatch in substitutions",
            "",
            "="*60,
            "CURRENT SITUATION:",
            "="*60,
            ""
        ])

        if self.template_catalog and "fill_prototype" not in disabled_tools:
            template_lines = ["   - Task templates (from parent pool / prior database):"]
            task_constraints = self.config.get("task_constraints", {}) or {}
            tracked_props = {
                prop
                for prop, prop_def in task_constraints.items()
                if prop != "is_valid"
                and isinstance(prop_def, dict)
                and prop_def.get("enabled", True)
            }
            show_all_metrics = not tracked_props

            def _format_metric(value: Optional[float], digits: int) -> Optional[str]:
                if value is None:
                    return None
                try:
                    return f"{float(value):.{digits}f}"
                except Exception:
                    return None

            def _add_metric(meta_bits: List[str], label: str, value: Optional[float], digits: int) -> None:
                formatted = _format_metric(value, digits)
                if formatted is not None:
                    meta_bits.append(f"{label}={formatted}")

            for tpl in self.template_catalog:
                name = tpl.get("name")
                formula = tpl.get("reference_formula")
                crystal_system = tpl.get("crystal_system")
                space_group = tpl.get("space_group")
                ratio = tpl.get("ratio")
                elements = tpl.get("elements")
                best_for = tpl.get("reduced_formula") or formula
                ed = tpl.get("decomposition_energy", tpl.get("energy_above_hull"))
                piezoelectric_coefficient = tpl.get("piezoelectric_coefficient")
                dielectric_constant = tpl.get("dielectric_constant")
                bulk_modulus = tpl.get("bulk_modulus")
                shear_modulus = tpl.get("shear_modulus")
                density = tpl.get("density")
                band_gap = tpl.get("band_gap")
                nsites = tpl.get("nsites")
                if not name:
                    continue
                meta_bits = []
                if crystal_system:
                    meta_bits.append(f"system={crystal_system}")
                if space_group:
                    meta_bits.append(f"sg={space_group}")
                if ratio:
                    meta_bits.append(f"ratio={ratio}")
                if elements:
                    slots = [chr(65 + j) for j in range(len(elements))]
                    meta_bits.append(f"elements=[{','.join(slots)}] ({len(elements)} required)")
                if best_for:
                    meta_bits.append(f"best_for={best_for}")
                if show_all_metrics:
                    _add_metric(meta_bits, "Ed", ed, 3)
                    _add_metric(meta_bits, "bulk", bulk_modulus, 1)
                    _add_metric(meta_bits, "shear", shear_modulus, 1)
                    _add_metric(meta_bits, "rho", density, 2)
                    _add_metric(meta_bits, "gap", band_gap, 2)
                    _add_metric(meta_bits, "Ef", tpl.get("formation_energy"), 3)
                    _add_metric(meta_bits, "piezo", piezoelectric_coefficient, 2)
                    _add_metric(meta_bits, "dielectric", dielectric_constant, 1)
                else:
                    # Always show Ed for context even if it's not a hard constraint.
                    _add_metric(meta_bits, "Ed", ed, 3)
                    if "bulk_modulus" in tracked_props:
                        _add_metric(meta_bits, "bulk", bulk_modulus, 1)
                    if "shear_modulus" in tracked_props:
                        _add_metric(meta_bits, "shear", shear_modulus, 1)
                    if "density" in tracked_props:
                        _add_metric(meta_bits, "rho", density, 2)
                    if "band_gap" in tracked_props:
                        _add_metric(meta_bits, "gap", band_gap, 2)
                    if "formation_energy" in tracked_props:
                        _add_metric(meta_bits, "Ef", tpl.get("formation_energy"), 3)
                    if "piezoelectric_coefficient" in tracked_props:
                        _add_metric(meta_bits, "piezo", piezoelectric_coefficient, 2)
                    if "dielectric_constant" in tracked_props:
                        _add_metric(meta_bits, "dielectric", dielectric_constant, 1)
                if nsites:
                    try:
                        meta_bits.append(f"nsites={int(nsites)}")
                    except Exception:
                        pass
                meta = f"; {'; '.join(meta_bits)}" if meta_bits else ""
                if formula:
                    template_lines.append(f"     • {name} (from {formula}{meta})")
                else:
                    template_lines.append(f"     • {name}{meta}")
            template_lines.append(
                "   - HARD REQUIREMENT: When using fill_prototype, you MUST use templates listed above."
            )
            template_lines.append("")
            try:
                insert_at = prompt_parts.index("3. mutate(parent, strength)")
                prompt_parts[insert_at:insert_at] = template_lines
            except ValueError:
                prompt_parts.extend(template_lines)

        # Add iteration context
        if iteration is not None and max_iterations is not None:
            prompt_parts.extend([
                f"ITERATION: {iteration}/{max_iterations}",
                ""
            ])

        mode = self._get_ablation_mode()
        is_crystal = mode == "crystal_forge"
        is_memory_less = mode == "memory_less"
        is_vanilla = mode == "vanilla_react"
        is_flat = mode == "flat_memory"
        include_history = mode in {"vanilla_react", "crystal_forge"}
        include_reflective_analysis = is_crystal
        include_tool_policy = is_crystal
        include_strategy_guidance = is_crystal
        include_diversity_signals = is_crystal

        prompt_layout = self._get_prompt_layout()
        include_hmrr = bool(is_crystal and hmrr_context and hmrr_context.get("enabled"))
        use_m3_layout = bool(include_hmrr and prompt_layout == "m3_explicit")
        m3_evidence_blocks: set[str] = set()

        def _ensure_m3_evidence_block(title: str) -> None:
            if not use_m3_layout or title in m3_evidence_blocks:
                return
            prompt_parts.extend([
                "=" * 60,
                title,
                "=" * 60,
                "",
            ])
            m3_evidence_blocks.add(title)

        # Inject hierarchical memory context (Macro -> Meso -> Micro)
        if include_hmrr:
            if use_m3_layout:
                prompt_parts.extend(
                    self._build_m3_backbone(
                        hmrr_context=hmrr_context,
                        reflection=reflection,
                        strategies_data=strategies_data,
                        all_structures=all_structures,
                    )
                )
            else:
                macro_rules = hmrr_context.get("macro_rules", []) or []
                meso_tips = hmrr_context.get("meso_tips", []) or []
                micro_events = hmrr_context.get("micro_events", []) or []

                if macro_rules:
                    prompt_parts.extend([
                        "="*60,
                        "[Macro Rules]",
                        "="*60,
                    ])
                    for idx, rec in enumerate(macro_rules, 1):
                        confidence = rec.get("confidence")
                        conf_text = f" (conf={confidence:.2f})" if isinstance(confidence, (int, float)) else ""
                        prompt_parts.append(f"  {idx}. {rec.get('message', '')}{conf_text}")
                    prompt_parts.append("")

                if meso_tips:
                    prompt_parts.extend([
                        "="*60,
                        "[Meso Tips]",
                        "="*60,
                    ])
                    for idx, rec in enumerate(meso_tips, 1):
                        prompt_parts.append(f"  {idx}. {rec.get('message', '')}")
                    prompt_parts.append("")

                if micro_events:
                    prompt_parts.extend([
                        "="*60,
                        "[Micro Latest Events]",
                        "="*60,
                    ])
                    for idx, rec in enumerate(micro_events, 1):
                        prompt_parts.append(f"  {idx}. {rec.get('message', '')}")
                    prompt_parts.append("")

        if use_m3_layout:
            prompt_parts.extend([
                "=" * 60,
                "EVIDENCE ZONE (from existing reflections)",
                "=" * 60,
                "",
            ])

        # Add reflection
        if include_history and reflection:
            if previous_reflection_prompt:
                _ensure_m3_evidence_block("EVIDENCE - MICRO (Recent Raw Outcomes)")
                prompt_parts.extend([
                    "="*60,
                    "📌 PREVIOUS ITERATION DATA (RAW SUMMARY):",
                    "="*60,
                    previous_reflection_prompt,
                    ""
                ])
            stats_lines = [
                "PREVIOUS ITERATION RESULTS:",
                f"  - Best decomposition energy: {reflection.get('best_ed', 'N/A')} eV/atom",
            ]

            # Add weighted score if available (multi-objective)
            if 'best_weighted_score' in reflection and reflection.get('best_weighted_score', 0) > 0:
                stats_lines.append(f"  - Best weighted score: {reflection['best_weighted_score']:.4f} (multi-objective)")
                stats_lines.append(f"  - Avg weighted score: {reflection.get('avg_weighted_score', 0):.4f}")

            stats_lines.extend([
                f"  - Hit rate: {reflection.get('hit_rate', 0):.1f}% ({reflection.get('hit_count', 0)}/{reflection.get('total_generated', 0)})",
                f"  - Valid rate: {reflection.get('valid_rate', 0):.1f}%",
                f"  - Diversity: {reflection.get('diversity', 0):.2f}",
                ""
                ])

            if not previous_reflection_prompt:
                _ensure_m3_evidence_block("EVIDENCE - MICRO (Recent Raw Outcomes)")
                prompt_parts.extend(stats_lines)

            # 🔥 NEW: Add LLM reflection analysis (deep insights about why structures succeeded/failed)
            if include_reflective_analysis and 'llm_analysis' in reflection and reflection['llm_analysis']:
                llm_analysis = reflection['llm_analysis']
                if use_m3_layout:
                    split_analysis = self._split_llm_analysis_sections(llm_analysis)
                    if split_analysis["micro"]:
                        _ensure_m3_evidence_block("EVIDENCE - MICRO (Per-Structure Diagnostics)")
                        prompt_parts.extend([
                            "=" * 60,
                            "🧠 DEEP ANALYSIS FROM PREVIOUS ITERATION:",
                            "=" * 60,
                            split_analysis["micro"],
                            ""
                        ])

                    if split_analysis["meso"]:
                        _ensure_m3_evidence_block("EVIDENCE - MESO (Patterns & Tactical Recommendations)")
                        prompt_parts.extend([
                            "=" * 60,
                            "🧠 DEEP ANALYSIS FROM PREVIOUS ITERATION:",
                            "=" * 60,
                            split_analysis["meso"],
                            ""
                        ])

                    if split_analysis["other"]:
                        _ensure_m3_evidence_block("EVIDENCE - OTHER (Supplementary Analysis)")
                        prompt_parts.extend([
                            "=" * 60,
                            "🧠 DEEP ANALYSIS FROM PREVIOUS ITERATION:",
                            "=" * 60,
                            split_analysis["other"],
                            ""
                        ])
                else:
                    prompt_parts.extend([
                        "="*60,
                        "🧠 DEEP ANALYSIS FROM PREVIOUS ITERATION:",
                        "="*60,
                        llm_analysis,
                        ""
                    ])

            # 🔥 NEW: Add tool performance statistics
            if include_tool_policy and 'tool_statistics' in reflection:
                tool_stats = reflection['tool_statistics']
                ranked_tools = tool_stats.get('ranked', [])
                rank_metric = tool_stats.get('rank_metric', 'metastable_rate')
                rank_label = tool_stats.get('rank_label', 'metastable rate')

                if ranked_tools:
                    _ensure_m3_evidence_block("EVIDENCE - MESO (Tool Policy Signals)")
                    prompt_parts.extend([
                        "="*60,
                        "📊 TOOL PERFORMANCE HISTORY (LEARN FROM THIS!):",
                        "="*60,
                        ""
                    ])

                    # Show ranked tools with performance metrics
                    for i, (tool, perf) in enumerate(ranked_tools, 1):
                        total = perf['total']
                        metastable_rate = perf['metastable_rate']
                        valid_rate = perf['valid_rate']
                        hit_rate = perf.get('hit_rate', 0.0)
                        avg_score = perf.get('avg_weighted_score', 0.0)
                        best_score = perf.get('best_weighted_score', 0.0)
                        avg_ed = perf['avg_ed']
                        min_ed = perf['min_ed']
                        rank_value = perf.get(rank_metric, 0.0)

                        # Add emoji indicators
                        if rank_metric == 'hit_rate':
                            if rank_value >= 50:
                                indicator = "✅ EXCELLENT"
                            elif rank_value >= 20:
                                indicator = "👍 GOOD"
                            elif rank_value >= 5:
                                indicator = "⚠️  OKAY"
                            else:
                                indicator = "❌ POOR"
                        else:
                            if rank_value >= 20:
                                indicator = "✅ EXCELLENT"
                            elif rank_value >= 10:
                                indicator = "👍 GOOD"
                            elif rank_value >= 5:
                                indicator = "⚠️  OKAY"
                            else:
                                indicator = "❌ POOR"

                        prompt_parts.append(
                            f"  {i}. {tool:12s} [{indicator}]: "
                            f"{total:3d} used | "
                            f"Valid: {valid_rate:5.1f}% | "
                            f"Hit: {hit_rate:5.1f}% | "
                            f"Metastable: {metastable_rate:5.1f}% | "
                            f"Avg Score: {avg_score:6.3f} | "
                            f"Best Score: {best_score:6.3f} | "
                            f"Best Ed: {min_ed:6.3f} | "
                            f"Avg Ed: {avg_ed:6.3f}"
                        )

                    prompt_parts.append("")

        # 🔥 NEW: Add critical strategic guidance
        if include_strategy_guidance and include_history and strategies_data and 'strategies' in strategies_data:
            strategies = strategies_data['strategies']
            if strategies:
                _ensure_m3_evidence_block("EVIDENCE - MESO (Strategic Guidance)")
                # Extract target Ed ranges from all strategies
                ed_ranges = []
                for s in strategies:
                    if 'target_ed_range' in s:
                        ed_ranges.append(s['target_ed_range'])

                if ed_ranges:
                    min_ed = min(r[0] for r in ed_ranges)
                    max_ed = max(r[1] for r in ed_ranges)

                    prompt_parts.extend([
                        "="*60,
                        "🎯 STRATEGIC GUIDANCE (CRITICAL!):",
                        "="*60,
                        f"TARGET: Generate structures with decomposition energy (Ed) in range {min_ed}-{max_ed} eV/atom",
                        f"CURRENT BEST: {reflection.get('best_ed', 'N/A')} eV/atom",
                        ""
                    ])

                    # Add specific strategy instructions
                    highest_priority = max(strategies, key=lambda s: s.get('allocation_weight', 0))
                    prompt_parts.extend([
                        "TOP PRIORITY STRATEGY:",
                        f"  Name: {highest_priority.get('strategy_name', 'N/A')}",
                        f"  Type: {highest_priority.get('strategy_type', 'N/A')}",
                        f"  Target Ed: {highest_priority.get('target_ed_range', 'N/A')} eV/atom",
                        f"  Instructions: {highest_priority.get('instructions', 'N/A')}",
                        ""
                    ])

        # 🔥 NEW: Add formula diversity tracking and avoid list
        if include_diversity_signals and include_history and all_structures:
            from collections import Counter
            formulas = [s.formula for s in all_structures if hasattr(s, 'formula')]
            formula_counts = Counter(formulas)

            if formula_counts:
                # Show most repeated formulas that should be avoided
                most_common = formula_counts.most_common(5)
                repeated = [(f, c) for f, c in most_common if c >= 2]

                if repeated:
                    _ensure_m3_evidence_block("EVIDENCE - MESO (Diversity Control Signals)")
                    prompt_parts.extend([
                        "="*60,
                        "⚠️  FORMULA DIVERSITY - AVOID REPETITION:",
                        "="*60,
                        f"Total structures generated so far: {len(all_structures)}",
                        f"Unique formulas: {len(formula_counts)}",
                        "",
                        "🚨 MOST REPEATED FORMULAS (AVOID GENERATING THESE!):",
                        ""
                    ])

                    for formula, count in repeated:
                        percentage = (count / len(formulas)) * 100
                        prompt_parts.append(f"  - {formula}: {count}× ({percentage:.1f}%)")

                    prompt_parts.extend([
                        "",
                        "IMPORTANT:",
                        "- When using mutate/mix, check if parents are in the repeated list",
                        "- If a parent is repeated, prefer using fill_prototype or choose different parents",
                        "- Prioritize generating NOVEL formulas not in the above list",
                        ""
                    ])

        # Add parent structures
        prompt_parts.extend([
            "="*60,
            "AVAILABLE PARENT STRUCTURES:",
            "="*60,
            ""
        ])

        task_constraints = self.config.get("task_constraints", {}) or {}
        tracked_props = {
            prop
            for prop, prop_def in task_constraints.items()
            if prop != "is_valid"
            and isinstance(prop_def, dict)
            and prop_def.get("enabled", True)
        }
        show_all_metrics = not tracked_props

        def _format_metric(value: Optional[float], digits: int) -> Optional[str]:
            if value is None:
                return None
            try:
                return f"{float(value):.{digits}f}"
            except Exception:
                return None

        def _add_metric_if_value(items: List[str], label: str, value: Optional[float], digits: int, suffix: str = "") -> None:
            formatted = _format_metric(value, digits)
            if formatted is not None:
                items.append(f"{label}={formatted}{suffix}")

        def _add_tracked_metric(items: List[str], label: str, value: Optional[float], digits: int, suffix: str = "") -> None:
            formatted = _format_metric(value, digits)
            if formatted is None:
                items.append(f"{label}=N/A")
            else:
                items.append(f"{label}={formatted}{suffix}")

        for i, parent in enumerate(parents):
            ed = parent.decomposition_energy if parent.decomposition_energy is not None else None
            formula = parent.formula if hasattr(parent, 'formula') else "Unknown"
            props = parent.properties if hasattr(parent, "properties") and parent.properties else {}

            piezo = props.get("piezoelectric_coefficient")
            dielectric = props.get("dielectric_constant")
            bulk_modulus = props.get("bulk_modulus")
            shear_modulus = props.get("shear_modulus")
            density = props.get("density")
            band_gap = props.get("band_gap")

            formation_energy = props.get("formation_energy")

            metric_bits: List[str] = []
            if show_all_metrics:
                _add_metric_if_value(metric_bits, "Ed", ed, 3, " eV/atom")
                _add_metric_if_value(metric_bits, "bulk", bulk_modulus, 1)
                _add_metric_if_value(metric_bits, "shear", shear_modulus, 1)
                _add_metric_if_value(metric_bits, "rho", density, 2)
                _add_metric_if_value(metric_bits, "gap", band_gap, 2)
                _add_metric_if_value(metric_bits, "Ef", formation_energy, 3)
                _add_metric_if_value(metric_bits, "piezo", piezo, 2)
                _add_metric_if_value(metric_bits, "dielectric", dielectric, 1)
            else:
                # Always show Ed for context even if it's not a hard constraint.
                _add_tracked_metric(metric_bits, "Ed", ed, 3, " eV/atom")
                if "bulk_modulus" in tracked_props:
                    _add_tracked_metric(metric_bits, "bulk", bulk_modulus, 1)
                if "shear_modulus" in tracked_props:
                    _add_tracked_metric(metric_bits, "shear", shear_modulus, 1)
                if "density" in tracked_props:
                    _add_tracked_metric(metric_bits, "rho", density, 2)
                if "band_gap" in tracked_props:
                    _add_tracked_metric(metric_bits, "gap", band_gap, 2)
                if "formation_energy" in tracked_props:
                    _add_tracked_metric(metric_bits, "Ef", formation_energy, 3)
                if "piezoelectric_coefficient" in tracked_props:
                    _add_tracked_metric(metric_bits, "piezo", piezo, 2)
                if "dielectric_constant" in tracked_props:
                    _add_tracked_metric(metric_bits, "dielectric", dielectric, 1)

            # 🔥 NEW: Mark if this parent is repeated
            warning = ""
            if include_diversity_signals and include_history and all_structures:
                formulas = [s.formula for s in all_structures if hasattr(s, 'formula')]
                count = formulas.count(formula)
                if count >= 2:
                    warning = f" ⚠️ REPEATED"

            metrics = ", ".join(metric_bits) if metric_bits else "metrics=N/A"
            prompt_parts.append(
                f"  Parent {i}: {formula:<15} "
                f"({metrics}){warning}"
            )

        # Add POSCAR structural details for a random sample of parents
        import random as _rnd
        sample_size = min(5, len(parents))
        if sample_size > 0:
            sampled_indices = sorted(_rnd.sample(range(len(parents)), sample_size))
            prompt_parts.extend([
                "",
                "STRUCTURAL DETAILS (POSCAR format for sampled parents):",
            ])
            for idx in sampled_indices:
                p = parents[idx]
                poscar = p.to_poscar(significant_figures=6)
                prompt_parts.extend([
                    f"--- Parent {idx} ({p.formula}) ---",
                    poscar,
                    "",
                ])

        # Add strategy (legacy single strategy support)
        if strategy:
            prompt_parts.extend([
                "",
                "LEGACY STRATEGY:",
                strategy,
                ""
            ])

        if is_crystal:
            main_task_hint = "Consider the STRATEGIC GUIDANCE and avoid repeated formulas."
        elif is_vanilla:
            main_task_hint = (
                "Use the recent raw execution summary and the current parent pool only. "
                "Do not rely on synthesized strategy hints."
            )
        elif is_flat:
            main_task_hint = (
                "Use the current parent pool together with [Past Raw Trajectories] only. "
                "Do not rely on synthesized summaries, strategies, or memory abstractions."
            )
        else:
            main_task_hint = "Focus on current parent pool evidence only; do not rely on prior-iteration history."

        if is_flat:
            prompt_parts.extend([
                "",
                "=" * 60,
                "[Past Raw Trajectories]",
                "=" * 60,
                past_raw_trajectories or "No past raw trajectories yet.",
                "",
            ])

        task_requirements = [
            f"- You MUST provide EXACTLY {n} tool actions (no more, no less)",
            "- Each action must be on a separate line",
            "- Do NOT provide explanations or comments",
            f"- Just output {n} tool calls directly",
        ]
        if include_diversity_signals and include_history:
            task_requirements.append("- AVOID operations on parents marked with ⚠️ REPEATED")
        task_requirements.extend([
            "- The total number of atoms in a structure MUST be less than 30 for ceramics",
            "- Always generate compounds (at least 2 elements). Pure elements like Zr or N2 are invalid for ceramic tasks",
            "- At most ONE mix action per iteration",
        ])

        prompt_parts.extend([
            "",
            "="*60,
            "YOUR TASK:",
            "="*60,
            "",
            f"Generate EXACTLY {n} new structures using the tools above.",
            main_task_hint,
            "Prioritize action diversity across composition-local / structure-local / global-jump intents.",
            "",
            "CRITICAL REQUIREMENTS:",
            *task_requirements,
            "",
            "Output format (one action per line):",
            "",
        ])
        if "substitute" not in disabled_tools:
            prompt_parts.append("substitute(parent=0, old='Ti', new='Zr')  # or old='Ti,O', new='Zr,S' for multi")
        if "mutate" not in disabled_tools:
            prompt_parts.append("mutate(parent=1, strength=0.05)")
        if "mix" not in disabled_tools:
            prompt_parts.append("mix(parent1=0, parent2=1, ratio=0.7)")
        if "fill_prototype" not in disabled_tools:
            prompt_parts.append(
                f"fill_prototype(template='{template_example}', elements=['A','B'])"
                if disable_defaults
                else "fill_prototype(template='Rock-salt', elements=['Na','Cl'])"
            )
        prompt_parts.extend([
            "",
            f"NOW PROVIDE EXACTLY {n} ACTIONS:",
            ""
        ])

        return "\n".join(prompt_parts)

    def _load_previous_reflection_prompt(self, output_dir: Optional[str], iteration: Optional[int]) -> Optional[str]:
        if not output_dir or not iteration or iteration <= 1:
            return None
        from pathlib import Path
        prompt_path = Path(output_dir) / f"iteration_{iteration - 1}" / "reflection" / "reflection_prompt.txt"
        if not prompt_path.exists():
            return None
        text = prompt_path.read_text()
        start = text.find("STATISTICAL SUMMARY:")
        if start == -1:
            start = 0
        end = text.find("DETAILED PER-STRUCTURE BREAKDOWN:")
        if end == -1:
            end = text.find("YOUR TASK:")
        if end != -1:
            text = text[start:end]
        else:
            text = text[start:]
        return text.strip()

    def _load_past_raw_trajectories(
        self,
        output_dir: Optional[str],
        iteration: Optional[int],
    ) -> str:
        if not output_dir or not iteration or iteration <= 1:
            return "No past raw trajectories yet."

        flat_cfg = self.config.get("flat_memory", {}) or {}
        same_run_only = bool(flat_cfg.get("same_run_only", True))
        try:
            window_size = max(1, int(flat_cfg.get("window_size", 5)))
        except Exception:
            window_size = 5
        try:
            char_cap = max(1000, int(flat_cfg.get("char_cap", 12000)))
        except Exception:
            char_cap = 12000

        if not same_run_only:
            logger.warning("flat_memory.same_run_only=false is not implemented; using current-run history only")

        from pathlib import Path

        base_dir = Path(output_dir)
        blocks: List[str] = []
        total_chars = 0
        start_iteration = max(1, iteration - window_size)

        for past_iteration in range(iteration - 1, start_iteration - 1, -1):
            block = self._build_raw_trajectory_block(base_dir, past_iteration)
            if not block:
                continue
            projected_chars = total_chars + len(block) + (2 if blocks else 0)
            if projected_chars > char_cap:
                break
            blocks.append(block)
            total_chars = projected_chars

        if not blocks:
            return "No past raw trajectories yet."
        return "\n\n".join(blocks)

    def _build_raw_trajectory_block(self, base_dir, past_iteration: int) -> Optional[str]:
        tool_dir = base_dir / f"iteration_{past_iteration}" / "tool_selector"
        reflection_path = base_dir / f"iteration_{past_iteration}" / "reflection.json"
        output_path = tool_dir / "output.txt"
        actions_path = tool_dir / "actions.json"

        action_lines: List[str] = []
        if output_path.exists():
            raw_actions = output_path.read_text(encoding="utf-8").strip()
            if raw_actions:
                action_lines = [line.rstrip() for line in raw_actions.splitlines() if line.strip()]
        elif actions_path.exists():
            try:
                actions = json.loads(actions_path.read_text(encoding="utf-8"))
            except Exception:
                actions = []
            if isinstance(actions, list):
                action_lines = [
                    f"{idx}. {json.dumps(action, ensure_ascii=True, sort_keys=True)}"
                    for idx, action in enumerate(actions, 1)
                ]

        observation_lines: List[str] = []
        raw_observation = self._load_raw_observation_text(base_dir, past_iteration)
        if raw_observation:
            observation_lines = raw_observation.splitlines()
        elif reflection_path.exists():
            try:
                reflection = json.loads(reflection_path.read_text(encoding="utf-8"))
            except Exception:
                reflection = {}
            if isinstance(reflection, dict):
                observation_lines.extend([
                    f"total_generated: {reflection.get('total_generated', 0)}",
                    f"valid: {reflection.get('valid_count', 0)} ({reflection.get('valid_rate', 0):.1f}%)",
                    f"hit: {reflection.get('hit_count', 0)} ({reflection.get('hit_rate', 0):.1f}%)",
                    f"metastable: {reflection.get('metastable_count', 0)} ({reflection.get('metastable_rate', 0):.1f}%)",
                    f"best_ed: {self._format_flat_number(reflection.get('best_ed'))}",
                    f"avg_ed: {self._format_flat_number(reflection.get('avg_ed'))}",
                ])
                snippets = self._select_raw_structure_snippets(reflection.get("individual_reflections", []))
                if snippets:
                    observation_lines.append("structure_snippets:")
                    observation_lines.extend(snippets)

        if not action_lines and not observation_lines:
            return None

        lines = [
            f"Trajectory: iteration {past_iteration}",
            "[Action]",
            *(action_lines or ["No recorded actions."]),
            "[Observation]",
            *(observation_lines or ["No recorded observation."]),
        ]
        return "\n".join(lines)

    def _load_raw_observation_text(self, base_dir, past_iteration: int) -> Optional[str]:
        reflection_prompt_path = (
            base_dir / f"iteration_{past_iteration}" / "reflection" / "reflection_prompt.txt"
        )
        if not reflection_prompt_path.exists():
            return None

        text = reflection_prompt_path.read_text(encoding="utf-8").strip()
        if not text:
            return None

        start = text.find("STATISTICAL SUMMARY:")
        if start == -1:
            start = 0
        end = text.find("YOUR TASK:")
        if end == -1:
            end = len(text)
        block = text[start:end].strip()
        return block or None

    def _select_raw_structure_snippets(self, individual_reflections: List[Dict]) -> List[str]:
        if not isinstance(individual_reflections, list):
            return []

        def _priority(item: Dict) -> tuple:
            category = str(item.get("category", "")).upper()
            category_rank = 0 if category in {"FAILED", "BOTTOM", "PARTIAL"} else 1
            ed = item.get("ed")
            if not isinstance(ed, (int, float)):
                ed = float("-inf")
            return (category_rank, -float(ed))

        selected = sorted(individual_reflections, key=_priority)[:3]
        lines: List[str] = []
        for ref in selected:
            formula = ref.get("formula", "N/A")
            category = ref.get("category", "N/A")
            ed = self._format_flat_number(ref.get("ed"))
            lines.append(f"- formula={formula}; category={category}; ed={ed}")
            analysis = ref.get("analysis", {}) or {}
            reason = str(analysis.get("reason", "")).strip()
            if reason:
                lines.append(f"  reason: {reason[:240]}")
            suggestions = analysis.get("improvement_suggestions", []) or []
            if suggestions:
                first_suggestion = str(suggestions[0]).strip()
                if first_suggestion:
                    lines.append(f"  suggestion: {first_suggestion[:200]}")
        return lines

    def _format_flat_number(self, value) -> str:
        if isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                return f"{float(value):.3f}"
        return "N/A"

    def _parse_tool_actions(self, response: str, num_parents: int) -> List[Dict]:
        """
        Parse LLM response into tool actions.

        The parser is tolerant to:
        - parent / parent_idx, parent1_idx / parent1, etc.
        - single or double quotes
        - optional spaces / trailing comments
        """
        actions: List[Dict] = []

        import ast

        def _coerce_number(val: str):
            try:
                return float(val) if "." in val else int(val)
            except Exception:
                return val

        def _strip_quotes(val: str) -> str:
            return val.strip().strip("'\"")

        def _split_top_level_args(arg_str: str) -> List[str]:
            parts = []
            buf = []
            depth = 0
            quote = None
            for ch in arg_str:
                if quote:
                    buf.append(ch)
                    if ch == quote:
                        quote = None
                    continue

                if ch in ("'", '"'):
                    buf.append(ch)
                    quote = ch
                    continue

                if ch in ("[", "("):
                    depth += 1
                    buf.append(ch)
                    continue

                if ch in ("]", ")"):
                    depth = max(depth - 1, 0)
                    buf.append(ch)
                    continue

                if ch == "," and depth == 0:
                    part = "".join(buf).strip()
                    if part:
                        parts.append(part)
                    buf = []
                    continue

                buf.append(ch)

            if buf:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
            return parts

        def _parse_args(arg_str: str) -> Dict[str, str]:
            parsed: Dict[str, str] = {}
            for k, v in re.findall(r"(\w+)\s*=\s*([^,\)]+)", arg_str):
                parsed[k] = _coerce_number(v.strip().strip("'\""))
            return parsed

        def _parse_positional(arg_str: str) -> List:
            tokens = []
            for part in _split_top_level_args(arg_str):
                token = part.strip()
                if token:
                    tokens.append(_coerce_number(token.strip().strip("'\"")))
            return tokens

        def _parse_elements_value(val) -> List[str]:
            if val is None:
                return []
            if isinstance(val, list):
                return [str(v).strip().strip("'\"") for v in val if str(v).strip()]
            if isinstance(val, tuple):
                return [str(v).strip().strip("'\"") for v in val if str(v).strip()]
            if isinstance(val, str):
                raw = val.strip()
                if raw.startswith("[") and raw.endswith("]"):
                    try:
                        parsed = ast.literal_eval(raw)
                        if isinstance(parsed, (list, tuple)):
                            return [str(v).strip().strip("'\"") for v in parsed if str(v).strip()]
                    except Exception:
                        pass
                raw = raw.strip("[]")
                parts = [p.strip().strip("'\"") for p in raw.split(",")]
                return [p for p in parts if p]
            return []

        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            tool_match = re.match(r"(\w+)\s*\((.*)\)", line)
            if not tool_match:
                continue

            tool = tool_match.group(1).lower()
            if tool == "fillprototype":
                tool = "fill_prototype"
            arg_str = tool_match.group(2)
            args = _parse_args(arg_str)
            positional = [] if args else _parse_positional(arg_str)

            # Normalize argument names
            parent = args.get("parent", args.get("parent_idx"))
            parent1 = args.get("parent1", args.get("parent1_idx"))
            parent2 = args.get("parent2", args.get("parent2_idx"))

            if tool == "fill_prototype":
                parts = _split_top_level_args(arg_str)
                kwargs = {}
                positional_parts = []
                for part in parts:
                    if "=" in part:
                        key, val = part.split("=", 1)
                        kwargs[key.strip()] = val.strip()
                    else:
                        positional_parts.append(part.strip())

                template_raw = kwargs.get("template") or kwargs.get("template_name")
                elements_raw = kwargs.get("elements") or kwargs.get("elements_list")

                if template_raw is None and positional_parts:
                    template_raw = positional_parts[0]
                if elements_raw is None and len(positional_parts) > 1:
                    elements_raw = positional_parts[1:]

                template = _strip_quotes(str(template_raw)) if template_raw is not None else None
                elements = _parse_elements_value(elements_raw)
                if template and elements:
                    actions.append({
                        "tool": "fill_prototype",
                        "template": template,
                        "elements": elements,
                    })
                continue

            if tool == "substitute":
                if parent is None and len(positional) >= 3:
                    parent, old_el, new_el = positional[:3]
                else:
                    old_el = args.get("old", args.get("old_element"))
                    new_el = args.get("new", args.get("new_element"))
                if parent is None or parent >= num_parents:
                    continue
                if old_el and new_el:
                    actions.append({
                        "tool": "substitute",
                        "parent": int(parent),
                        "old_element": str(old_el),
                        "new_element": str(new_el),
                    })

            elif tool == "mutate":
                if parent is None and len(positional) >= 2:
                    parent, strength = positional[:2]
                else:
                    strength = args.get("strength")
                if parent is None or parent >= num_parents:
                    continue
                if strength is not None:
                    actions.append({
                        "tool": "mutate",
                        "parent": int(parent),
                        "strength": float(strength),
                    })

            elif tool == "mix":
                if (parent1 is None or parent2 is None) and len(positional) >= 2:
                    parent1, parent2 = positional[:2]
                    ratio = 0.5
                else:
                    ratio = args.get("ratio", 0.5)
                if parent1 is None or parent2 is None:
                    continue
                if parent1 >= num_parents or parent2 >= num_parents:
                    continue
                if ratio is not None:
                    actions.append({
                        "tool": "mix",
                        "parent1": int(parent1),
                        "parent2": int(parent2),
                        "ratio": float(ratio),
                    })

            elif tool == "crossover":
                continue

            elif tool == "dope":
                if parent is None and len(positional) >= 1:
                    parent = positional[0]
                    if len(positional) >= 2:
                        args["dopant"] = positional[1]
                    if len(positional) >= 3:
                        args["concentration"] = positional[2]
                    if len(positional) >= 4:
                        args["host_element"] = positional[3]
                if parent is None or parent >= num_parents:
                    continue
                action = {
                    "tool": "dope",
                    "parent": int(parent),
                }
                if "dopant" in args:
                    action["dopant"] = str(args["dopant"])
                if "concentration" in args:
                    action["concentration"] = float(args["concentration"])
                host = args.get("host", args.get("host_element"))
                if host:
                    action["host_element"] = str(host)
                actions.append(action)

            elif tool == "new":
                continue

        return actions

    def _fallback_selection(self, parents: List, n: int) -> List[Dict]:
        """
        Fallback tool selection if LLM fails

        Uses a simple heuristic to generate diverse actions:
        - If no parents: prototype-based generation
        - If parents exist: mix of mutation, prototype, and mix
        """
        actions = []
        disabled_tools = self._get_disabled_tools()
        allow_fill = "fill_prototype" not in disabled_tools
        allow_mutate = "mutate" not in disabled_tools
        allow_mix = "mix" not in disabled_tools
        default_templates = []
        if self.template_catalog:
            for tpl in self.template_catalog:
                name = tpl.get("name")
                elements = tpl.get("elements")
                if name and elements:
                    default_templates.append({"template": name, "elements": elements})
        if not default_templates and not parents:
            return []
        mix_used = False
        top30_list = []

        if parents:
            ed_pairs = []
            for i, parent in enumerate(parents):
                ed_val = parent.decomposition_energy
                ed_pairs.append((i, ed_val if ed_val is not None else float('inf')))
            ed_pairs.sort(key=lambda x: x[1])
            top_n = max(1, math.ceil(0.3 * len(ed_pairs)))
            top30_list = [idx for idx, _ in ed_pairs[:top_n]]

        if len(parents) == 0:
            # No parents: generate from prototypes
            if not allow_fill:
                logger.warning("fill_prototype disabled; cannot generate without parents.")
                return []
            for i in range(n):
                proto = default_templates[i % len(default_templates)]
                actions.append({
                    "tool": "fill_prototype",
                    "template": proto["template"],
                    "elements": proto["elements"],
                })
            return actions

        can_use_templates = bool(default_templates) and allow_fill

        def _add_mutate(idx: int) -> None:
            import random
            actions.append({
                "tool": "mutate",
                "parent": idx % len(parents),
                "strength": random.uniform(0.01, 0.1)
            })

        def _add_fill(idx: int) -> None:
            proto = default_templates[idx % len(default_templates)]
            actions.append({
                "tool": "fill_prototype",
                "template": proto["template"],
                "elements": proto["elements"],
            })

        def _add_mix() -> bool:
            nonlocal mix_used
            if len(parents) >= 2 and not mix_used and len(top30_list) >= 2:
                actions.append({
                    "tool": "mix",
                    "parent1": top30_list[0],
                    "parent2": top30_list[1],
                    "ratio": 0.5,
                })
                mix_used = True
                return True
            return False

        def _add_best_available(idx: int) -> None:
            if allow_mutate:
                _add_mutate(idx)
                return
            if can_use_templates:
                _add_fill(idx)
                return
            if allow_mix and _add_mix():
                return

        tool_sequence = []
        if allow_mutate:
            tool_sequence.append("mutate")
        if can_use_templates:
            tool_sequence.append("fill_prototype")
        if allow_mix:
            tool_sequence.append("mix")
        if not tool_sequence:
            logger.warning("All fallback tools are disabled; no actions generated.")
            return []

        # With parents: use diverse tools
        # Pattern: mutate, fill_prototype, mix, mutate, fill_prototype, mix, ...
        for i in range(n):
            tool_type = tool_sequence[i % len(tool_sequence)]
            if tool_type == "mutate":
                _add_mutate(i)
            elif tool_type == "fill_prototype":
                if can_use_templates:
                    _add_fill(i)
                else:
                    _add_best_available(i)
            elif tool_type == "mix":
                if not _add_mix():
                    _add_best_available(i)

        return actions
