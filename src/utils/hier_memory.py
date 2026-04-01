"""Hierarchical memory manager for HMRR (Micro/Meso/Macro)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_message(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _is_low_information_macro_message(value: Any) -> bool:
    msg = _normalize_message(value)
    if not msg:
        return True
    lower = msg.lower()
    if lower.startswith("###"):
        return True
    banned_exact = {
        "individual structure analysis",
        "overall success patterns",
        "overall failure patterns",
        "strategic recommendations for next iteration",
        "summary",
    }
    if lower in banned_exact:
        return True
    if len(msg) < 24:
        return True
    return False


@dataclass
class LayerConfig:
    enabled: bool
    max_items_for_prompt: int
    max_items_stored: int


class HierMemoryManager:
    """Maintains hierarchical memory and exposes prompt-ready context."""

    def __init__(
        self,
        config: Dict[str, Any],
        task_key: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> None:
        reflection_cfg = (config.get("reflection", {}) or {})
        hier_cfg = reflection_cfg.get("hierarchical")
        if hier_cfg is None and "hierachical" in reflection_cfg:
            hier_cfg = reflection_cfg.get("hierachical")
            logger.warning("Using deprecated config key reflection.hierachical; please rename to reflection.hierarchical")
        hier_cfg = hier_cfg or {}

        self.enabled = bool(hier_cfg.get("enabled", False))

        micro_cfg = hier_cfg.get("micro", {}) or {}
        meso_cfg = hier_cfg.get("meso", {}) or {}
        macro_cfg = hier_cfg.get("macro", {}) or {}

        self.micro = LayerConfig(
            enabled=bool(micro_cfg.get("enabled", True)),
            max_items_for_prompt=int(micro_cfg.get("max_items_for_prompt", 3)),
            max_items_stored=int(micro_cfg.get("max_items_stored", 100)),
        )
        self.meso = LayerConfig(
            enabled=bool(meso_cfg.get("enabled", True)),
            max_items_for_prompt=int(meso_cfg.get("max_items_for_prompt", 3)),
            max_items_stored=int(meso_cfg.get("max_items_stored", 30)),
        )
        self.macro = LayerConfig(
            enabled=bool(macro_cfg.get("enabled", True)),
            max_items_for_prompt=int(macro_cfg.get("max_items_for_prompt", 2)),
            max_items_stored=int(macro_cfg.get("max_items_stored", 20)),
        )

        self.stagnation_patience = int(meso_cfg.get("stagnation_patience", 3))
        self.stagnation_epsilon = float(meso_cfg.get("stagnation_epsilon", 1e-6))

        self.trigger_every_n_iters = int(macro_cfg.get("trigger_every_n_iters", 10))
        self.macro_min_confidence = float(macro_cfg.get("min_confidence", 0.55))
        self.persist_macro_across_runs = bool(macro_cfg.get("persist_across_runs", True))
        self.macro_store_dir = str(macro_cfg.get("store_dir", "data/memory/macro_rules"))
        llm_reflect_cfg = hier_cfg.get("llm_reflection", {}) or {}
        self.llm_reflection_enabled = bool(llm_reflect_cfg.get("enabled", True))
        self.llm_reflection_meso_enabled = bool(llm_reflect_cfg.get("meso_enabled", True))
        self.llm_reflection_macro_enabled = bool(llm_reflect_cfg.get("macro_enabled", True))
        self.llm_reflection_require_client = bool(llm_reflect_cfg.get("require_llm_client", False))
        self.meso_prompt_micro_k = int(llm_reflect_cfg.get("meso_prompt_micro_k", 6))
        self.meso_prompt_history_k = int(llm_reflect_cfg.get("meso_prompt_history_k", 4))
        self.macro_prompt_meso_k = int(llm_reflect_cfg.get("macro_prompt_meso_k", 6))
        self.macro_prompt_micro_k = int(llm_reflect_cfg.get("macro_prompt_micro_k", 4))
        self.macro_prompt_history_k = int(llm_reflect_cfg.get("macro_prompt_history_k", 4))
        self.strict_mode = bool(llm_reflect_cfg.get("strict_mode", False))
        self.retry_max_attempts = max(1, int(llm_reflect_cfg.get("retry_max_attempts", 1)))
        self.retry_backoff_seconds = max(0.0, float(llm_reflect_cfg.get("retry_backoff_seconds", 0.0)))
        meso_trigger_policy = str(llm_reflect_cfg.get("meso_trigger_policy", "stagnation")).strip().lower()
        if meso_trigger_policy not in {"stagnation", "every_iteration"}:
            meso_trigger_policy = "stagnation"
        self.meso_trigger_policy = meso_trigger_policy
        self.trace_io = bool(llm_reflect_cfg.get("trace_io", False))
        self.trace_dirname = str(llm_reflect_cfg.get("trace_dirname", "reflection")).strip() or "reflection"

        self.task_key = self._sanitize_task_key(task_key or task_name or "unknown_task")

        self.micro_events: List[Dict[str, Any]] = []
        self.meso_tips: List[Dict[str, Any]] = []
        self.macro_rules: List[Dict[str, Any]] = []

        self.best_metric: Optional[float] = None
        self.stagnation_count = 0
        self.best_metric_source = "best_ed"

        self.meso_trigger_count = 0
        self.macro_trigger_count = 0
        self.macro_rules_loaded_count = 0
        self.macro_rules_generated_count = 0
        self.llm_reflection_success_count = 0
        self.llm_reflection_failure_count = 0
        self.meso_llm_attempt_count = 0
        self.meso_llm_success_count = 0
        self.meso_llm_failure_count = 0
        self.macro_llm_attempt_count = 0
        self.macro_llm_success_count = 0
        self.macro_llm_failure_count = 0
        self.llm_client: Optional[Any] = None
        self._llm_missing_warned = False

        self._project_root = Path(__file__).resolve().parents[2]
        self._macro_file_path = (self._project_root / self.macro_store_dir / f"{self.task_key}.json").resolve()

        if self.enabled and self.macro.enabled and self.persist_macro_across_runs:
            self._load_macro_rules()

    def set_llm_client(self, llm_client: Any) -> None:
        """Attach shared LLM client from orchestrator."""
        self.llm_client = llm_client

    @staticmethod
    def _sanitize_task_key(task_key: str) -> str:
        clean = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in str(task_key))
        clean = clean.strip("._")
        return clean or "unknown_task"

    def _load_macro_rules(self) -> None:
        if not self._macro_file_path.exists():
            return
        try:
            obj = json.loads(self._macro_file_path.read_text())
        except Exception as exc:
            logger.warning("Failed to parse macro memory file %s: %s", self._macro_file_path, exc)
            return

        records = obj.get("macro_rules", [])
        if not isinstance(records, list):
            logger.warning("Macro memory file %s has invalid schema", self._macro_file_path)
            return

        loaded = []
        dropped = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if _is_low_information_macro_message(rec.get("message")):
                dropped += 1
                continue
            rec["message"] = _normalize_message(rec.get("message"))
            loaded.append(rec)

        loaded = self._rank_and_trim(loaded, self.macro.max_items_stored)
        self.macro_rules = loaded
        self.macro_rules_loaded_count = len(loaded)
        if dropped:
            logger.warning(
                "Dropped %d low-information macro rules from %s during load",
                dropped,
                self._macro_file_path,
            )
        if loaded:
            logger.info("Loaded %d macro rules for task %s", len(loaded), self.task_key)

    def finalize(self) -> None:
        if not (self.enabled and self.macro.enabled and self.persist_macro_across_runs):
            return
        try:
            self._macro_file_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "task_key": self.task_key,
                "updated_at": _utc_now_iso(),
                "macro_rules": self._rank_and_trim(self.macro_rules, self.macro.max_items_stored),
            }
            self._macro_file_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.warning("Failed to persist macro rules to %s: %s", self._macro_file_path, exc)

    def _record(
        self,
        layer: str,
        iteration: int,
        trigger: str,
        message: str,
        evidence: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        support: Optional[int] = None,
        score_delta: Optional[float] = None,
    ) -> Dict[str, Any]:
        rec = {
            "layer": layer,
            "iteration": int(iteration),
            "trigger": str(trigger),
            "message": str(message),
            "evidence": evidence or {},
            "confidence": _safe_float(confidence),
            "support": int(support) if support is not None else None,
            "score_delta": _safe_float(score_delta),
            "timestamp": _utc_now_iso(),
        }
        return rec

    @staticmethod
    def _importance(record: Dict[str, Any]) -> float:
        confidence = _safe_float(record.get("confidence")) or 0.0
        support = _safe_float(record.get("support")) or 0.0
        delta = abs(_safe_float(record.get("score_delta")) or 0.0)
        return confidence + 0.05 * support + delta

    def _rank_and_trim(self, records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        ranked = sorted(
            records,
            key=lambda r: (
                self._importance(r),
                int(r.get("iteration", 0)),
                str(r.get("timestamp", "")),
            ),
            reverse=True,
        )
        return ranked[:limit]

    def add_micro_event(self, iteration: int, structure: Any, constraint_hit: Optional[bool], trigger: str = "evaluation_result") -> None:
        if not (self.enabled and self.micro.enabled):
            return

        formula = getattr(structure, "formula", "unknown")
        ed = _safe_float(getattr(structure, "decomposition_energy", None))
        is_valid = bool(getattr(structure, "is_valid", False))
        source = ""
        try:
            source = str(getattr(structure, "metadata", {}).get("source", ""))
        except Exception:
            source = ""

        if not is_valid:
            status = "invalid"
        elif constraint_hit is True:
            status = "hit"
        else:
            status = "no_hit"

        message = f"{formula}: {status}, Ed={ed if ed is not None else 'N/A'}"
        evidence = {
            "formula": formula,
            "is_valid": is_valid,
            "ed": ed,
            "source": source,
            "constraint_hit": constraint_hit,
        }

        rec = self._record(
            layer="micro",
            iteration=iteration,
            trigger=trigger,
            message=message,
            evidence=evidence,
            confidence=0.95,
            support=1,
        )
        self.micro_events.append(rec)
        self.micro_events = self._rank_and_trim(self.micro_events, self.micro.max_items_stored)

    def update_after_reflection(
        self,
        iteration: int,
        reflection: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        metric, source = self._current_metric(reflection)
        improved = False
        delta = None
        if metric is not None:
            if self.best_metric is None:
                improved = True
                self.best_metric = metric
            else:
                delta = metric - self.best_metric
                if delta > self.stagnation_epsilon:
                    improved = True
                    self.best_metric = metric

            self.best_metric_source = source

        if improved:
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        meso_trigger: Optional[str] = None
        if self.meso.enabled:
            if self.meso_trigger_policy == "every_iteration":
                meso_trigger = "every_iteration"
            elif self.stagnation_count >= self.stagnation_patience:
                meso_trigger = "stagnation"

        if meso_trigger is not None:
            support_now = max(1, self.stagnation_count)
            tip, tip_confidence, tip_support, tip_evidence = self._derive_meso_tip(
                reflection=reflection,
                iteration=iteration,
                score_delta=delta,
                stagnation_count=support_now,
                output_dir=output_dir,
                trigger=meso_trigger,
            )
            rec = self._record(
                layer="meso",
                iteration=iteration,
                trigger=meso_trigger,
                message=tip,
                evidence={
                    "best_ed": reflection.get("best_ed"),
                    "hit_rate": reflection.get("hit_rate"),
                    "valid_rate": reflection.get("valid_rate"),
                    **tip_evidence,
                },
                confidence=tip_confidence,
                support=tip_support if tip_support is not None else support_now,
                score_delta=delta,
            )
            self.meso_tips.append(rec)
            self.meso_tips = self._rank_and_trim(self.meso_tips, self.meso.max_items_stored)
            if meso_trigger == "stagnation":
                self.stagnation_count = 0
            self.meso_trigger_count += 1

        if (
            self.macro.enabled
            and self.trigger_every_n_iters > 0
            and iteration % self.trigger_every_n_iters == 0
        ):
            rule, confidence, rule_evidence = self._derive_macro_rule(
                reflection=reflection,
                iteration=iteration,
                output_dir=output_dir,
                trigger="periodic",
            )
            if confidence >= self.macro_min_confidence:
                rec = self._record(
                    layer="macro",
                    iteration=iteration,
                    trigger="periodic",
                    message=rule,
                    evidence={
                        "best_ed": reflection.get("best_ed"),
                        "hit_rate": reflection.get("hit_rate"),
                        "meso_count": len(self.meso_tips),
                        **rule_evidence,
                    },
                    confidence=confidence,
                    support=max(1, len(self.meso_tips)),
                )
                self.macro_rules.append(rec)
                self.macro_rules = self._rank_and_trim(self.macro_rules, self.macro.max_items_stored)
                self.macro_trigger_count += 1
                self.macro_rules_generated_count += 1
                if self.persist_macro_across_runs:
                    self.finalize()

    def _current_metric(self, reflection: Dict[str, Any]) -> tuple[Optional[float], str]:
        weighted = _safe_float(reflection.get("best_weighted_score"))
        if weighted is not None and weighted > 0:
            return weighted, "best_weighted_score"

        best_ed = _safe_float(reflection.get("best_ed"))
        if best_ed is None:
            return None, "best_ed"
        # Lower Ed is better; convert to maximize-style metric.
        return -best_ed, "best_ed"

    def _derive_meso_tip(
        self,
        reflection: Dict[str, Any],
        iteration: int,
        score_delta: Optional[float],
        stagnation_count: int,
        output_dir: Optional[str],
        trigger: str,
    ) -> tuple[str, float, Optional[int], Dict[str, Any]]:
        if self.strict_mode and not self.llm_reflection_meso_enabled:
            raise RuntimeError("Strict HMRR mode requires llm_reflection.meso_enabled=true")
        llm_result = self._derive_meso_tip_with_llm(
            reflection=reflection,
            iteration=iteration,
            score_delta=score_delta,
            stagnation_count=stagnation_count,
            output_dir=output_dir,
            trigger=trigger,
        )
        if llm_result is not None:
            return llm_result
        if self.strict_mode:
            raise RuntimeError("Strict HMRR mode requires successful meso LLM reflection")
        return self._derive_meso_tip_fallback(reflection), 0.75, None, {"source": "fallback_template"}

    def _derive_meso_tip_fallback(self, reflection: Dict[str, Any]) -> str:
        hit_rate = _safe_float(reflection.get("hit_rate")) or 0.0
        valid_rate = _safe_float(reflection.get("valid_rate")) or 0.0
        diversity = _safe_float(reflection.get("diversity")) or 0.0
        if hit_rate < 10:
            return "Hits remain very low; prioritize structure-global jumps (fill_prototype) and reduce local substitutions."
        if valid_rate < 50:
            return "Validity is weak; constrain chemistry and favor conservative templates over aggressive mix/mutate."
        if diversity < 0.4:
            return "Diversity is collapsing; avoid repeated parents and increase exploration in new formula spaces."
        return "Progress is stagnant; rebalance tool usage and prioritize higher-performing operators from last iteration."

    def _derive_macro_rule(
        self,
        reflection: Dict[str, Any],
        iteration: int,
        output_dir: Optional[str],
        trigger: str,
    ) -> tuple[str, float, Dict[str, Any]]:
        if self.strict_mode and not self.llm_reflection_macro_enabled:
            raise RuntimeError("Strict HMRR mode requires llm_reflection.macro_enabled=true")
        llm_result = self._derive_macro_rule_with_llm(
            reflection=reflection,
            iteration=iteration,
            output_dir=output_dir,
            trigger=trigger,
        )
        if llm_result is not None:
            return llm_result
        if self.strict_mode:
            raise RuntimeError("Strict HMRR mode requires successful macro LLM reflection")
        return self._derive_macro_rule_fallback(reflection)

    def _derive_macro_rule_fallback(self, reflection: Dict[str, Any]) -> tuple[str, float, Dict[str, Any]]:
        hit_rate = _safe_float(reflection.get("hit_rate")) or 0.0
        best_ed = _safe_float(reflection.get("best_ed"))
        llm_analysis = str(reflection.get("llm_analysis") or "").strip()

        if llm_analysis and "failed" not in llm_analysis.lower():
            candidate_lines = [_normalize_message(x) for x in llm_analysis.splitlines()]
            candidate_lines = [x for x in candidate_lines if x and not _is_low_information_macro_message(x)]
            if candidate_lines:
                first_sentence = candidate_lines[0]
                if len(first_sentence) > 240:
                    first_sentence = first_sentence[:240].rstrip() + "..."
                rule = first_sentence
            else:
                rule = "Balance exploitation and exploration by ranking tools with recent hit/validity trends each iteration."
        elif best_ed is not None and best_ed <= 0.1:
            rule = "Stable regions are emerging; preserve successful frameworks and apply conservative compositional refinements."
        elif hit_rate < 20:
            rule = "When hit rate is persistently low, prioritize template-driven global exploration before local edits."
        else:
            rule = "Balance exploitation and exploration by ranking tools with recent hit/validity trends each iteration."

        confidence = max(0.3, min(0.95, hit_rate / 100.0 + 0.35))
        return rule, confidence, {"source": "fallback_template"}

    def _llm_ready(self) -> bool:
        if not self.llm_reflection_enabled:
            return False
        if self.llm_client is not None and hasattr(self.llm_client, "generate"):
            return True
        if self.llm_reflection_require_client and not self._llm_missing_warned:
            logger.warning("HMRR llm_reflection enabled but no llm_client attached; falling back to templates.")
            self._llm_missing_warned = True
        return False

    @staticmethod
    def _compact_lines(records: List[Dict[str, Any]], max_items: int) -> str:
        if not records:
            return "None"
        lines: List[str] = []
        for idx, rec in enumerate(records[:max_items], 1):
            trigger = rec.get("trigger", "")
            msg = str(rec.get("message", "")).strip()
            confidence = _safe_float(rec.get("confidence"))
            conf = f", conf={confidence:.2f}" if confidence is not None else ""
            lines.append(f"{idx}. [{trigger}] {msg}{conf}")
        return "\n".join(lines)

    @staticmethod
    def _sanitize_message(text: str, max_len: int = 240) -> str:
        clean = " ".join(str(text or "").strip().split())
        if len(clean) > max_len:
            clean = clean[:max_len].rstrip() + "..."
        return clean

    @staticmethod
    def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        raw = str(text).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _validate_confidence_field(obj: Dict[str, Any], field: str = "confidence") -> Tuple[bool, str]:
        if field not in obj:
            return False, f"invalid confidence: missing field '{field}'"
        value = _safe_float(obj.get(field))
        if value is None:
            return False, f"invalid confidence: field '{field}' is not numeric"
        if not (0.0 < value <= 1.0):
            return False, f"invalid confidence: {value} not in (0, 1]"
        return True, ""

    def _stage_trace_dir(self, output_dir: Optional[str], iteration: int) -> Optional[Path]:
        if not output_dir:
            return None
        trace_dir = Path(output_dir) / f"iteration_{iteration}" / self.trace_dirname
        trace_dir.mkdir(parents=True, exist_ok=True)
        return trace_dir

    def _write_llm_trace(
        self,
        output_dir: Optional[str],
        iteration: int,
        stage: str,
        payload: Dict[str, Any],
    ) -> None:
        if not (self.trace_io and output_dir):
            return
        try:
            trace_dir = self._stage_trace_dir(output_dir, iteration)
            if trace_dir is None:
                return
            trace_path = trace_dir / f"hmrr_{stage}_llm_trace.json"
            trace_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.warning("Failed to write HMRR %s LLM trace (iter=%s): %s", stage, iteration, exc)

    def _inc_stage_attempt(self, stage: str) -> None:
        if stage == "meso":
            self.meso_llm_attempt_count += 1
        elif stage == "macro":
            self.macro_llm_attempt_count += 1

    def _inc_stage_success(self, stage: str) -> None:
        if stage == "meso":
            self.meso_llm_success_count += 1
        elif stage == "macro":
            self.macro_llm_success_count += 1

    def _inc_stage_failure(self, stage: str) -> None:
        if stage == "meso":
            self.meso_llm_failure_count += 1
        elif stage == "macro":
            self.macro_llm_failure_count += 1

    def _llm_generate_json(
        self,
        prompt: str,
        *,
        stage: str,
        iteration: int,
        trigger: str,
        output_dir: Optional[str],
        validator: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        trace_payload: Dict[str, Any] = {
            "iteration": int(iteration),
            "stage": stage,
            "trigger": trigger,
            "status": "failed",
            "attempts": 0,
            "final_error": None,
            "parsed_output": None,
            "files": [],
            "timestamp_start": _utc_now_iso(),
            "timestamp_end": None,
        }
        trace_dir = self._stage_trace_dir(output_dir, iteration) if self.trace_io else None

        if not self._llm_ready():
            err = "LLM client is unavailable or disabled for reflection generation."
            trace_payload["final_error"] = err
            trace_payload["timestamp_end"] = _utc_now_iso()
            self._write_llm_trace(output_dir, iteration, stage, trace_payload)
            self._inc_stage_failure(stage)
            self.llm_reflection_failure_count += 1
            if self.strict_mode:
                raise RuntimeError(err)
            return None

        last_error = "unknown_error"
        attempts = max(1, self.retry_max_attempts)
        for attempt in range(1, attempts + 1):
            self._inc_stage_attempt(stage)
            trace_payload["attempts"] = attempt

            prompt_name = f"hmrr_{stage}_llm_attempt_{attempt}_prompt.txt"
            response_name = f"hmrr_{stage}_llm_attempt_{attempt}_response.txt"
            attempt_file_meta: Dict[str, Any] = {
                "attempt": attempt,
                "prompt_file": prompt_name if trace_dir else None,
                "response_file": response_name if trace_dir else None,
                "error": None,
            }
            if trace_dir is not None:
                (trace_dir / prompt_name).write_text(prompt)

            raw_response = ""
            try:
                responses = self.llm_client.generate(prompt, n=1)
                if not responses:
                    raise RuntimeError("empty response")
                raw_response = str(responses[0] or "")
                if not raw_response.strip():
                    raise RuntimeError("empty response")
                obj = self._extract_json_obj(raw_response)
                if obj is None:
                    raise RuntimeError("non-json response")
                if validator is not None:
                    valid, err_msg = validator(obj)
                    if not valid:
                        raise RuntimeError(err_msg)

                if trace_dir is not None:
                    (trace_dir / response_name).write_text(raw_response)
                trace_payload["files"].append(attempt_file_meta)
                trace_payload["status"] = "success"
                trace_payload["final_error"] = None
                trace_payload["parsed_output"] = obj
                trace_payload["timestamp_end"] = _utc_now_iso()
                self._write_llm_trace(output_dir, iteration, stage, trace_payload)

                self._inc_stage_success(stage)
                self.llm_reflection_success_count += 1
                return obj
            except Exception as exc:
                last_error = str(exc)
                attempt_file_meta["error"] = last_error
                if trace_dir is not None:
                    failure_dump = raw_response if raw_response else f"[error] {last_error}"
                    (trace_dir / response_name).write_text(failure_dump)
                trace_payload["files"].append(attempt_file_meta)

                if attempt < attempts and self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds)

        trace_payload["status"] = "failed"
        trace_payload["final_error"] = last_error
        trace_payload["timestamp_end"] = _utc_now_iso()
        self._write_llm_trace(output_dir, iteration, stage, trace_payload)
        self._inc_stage_failure(stage)
        self.llm_reflection_failure_count += 1
        if self.strict_mode:
            raise RuntimeError(
                f"HMRR {stage} LLM reflection failed after {attempts} attempts: {last_error}"
            )
        logger.warning("HMRR %s LLM reflection generation failed: %s", stage, last_error)
        return None

    def _derive_meso_tip_with_llm(
        self,
        reflection: Dict[str, Any],
        iteration: int,
        score_delta: Optional[float],
        stagnation_count: int,
        output_dir: Optional[str],
        trigger: str,
    ) -> Optional[tuple[str, float, Optional[int], Dict[str, Any]]]:
        if not self.llm_reflection_meso_enabled:
            return None

        recent_micro = self._rank_and_trim(self.micro_events, max(1, self.meso_prompt_micro_k))
        recent_meso = self._rank_and_trim(self.meso_tips, max(1, self.meso_prompt_history_k))
        current_metric, metric_source = self._current_metric(reflection)

        prompt = f"""You are the Meso Reflection Agent for iterative materials search.

Task key: {self.task_key}
Current iteration: {iteration}
Trigger: {trigger} (count={stagnation_count}, patience={self.stagnation_patience})
Metric source: {metric_source}
Current metric value: {current_metric}
Score delta vs best: {score_delta}
Best Ed: {reflection.get('best_ed')}
Hit rate: {reflection.get('hit_rate')}
Valid rate: {reflection.get('valid_rate')}
Diversity: {reflection.get('diversity')}

[Recent Micro Events]
{self._compact_lines(recent_micro, self.meso_prompt_micro_k)}

[Recent Meso Tips]
{self._compact_lines(recent_meso, self.meso_prompt_history_k)}

[Global Reflection Excerpt]
{str(reflection.get('llm_analysis') or '')[:800]}

Produce ONE tactical meso reflection that is specific and non-redundant.
Avoid repeating previous meso wording exactly.
confidence must be a float in (0, 1].
Return strict JSON only:
{{
  "message": "one sentence tactical guidance (<=180 chars)",
  "confidence": 0.72,
  "support": 1,
  "evidence": ["short reason 1", "short reason 2"]
}}
"""
        obj = self._llm_generate_json(
            prompt,
            stage="meso",
            iteration=iteration,
            trigger=trigger,
            output_dir=output_dir,
            validator=lambda x: self._validate_confidence_field(x, "confidence"),
        )
        if not obj:
            return None

        message = self._sanitize_message(obj.get("message", ""), max_len=180)
        if not message:
            return None

        confidence = _safe_float(obj.get("confidence"))
        if confidence is None:
            confidence = 0.75
        confidence = max(0.0, min(1.0, confidence))

        support = obj.get("support")
        try:
            support_val = int(support) if support is not None else None
        except Exception:
            support_val = None

        evidence = obj.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        evidence_text = [self._sanitize_message(x, max_len=120) for x in evidence[:4] if str(x).strip()]

        return message, confidence, support_val, {"source": "llm", "evidence_lines": evidence_text}

    def _derive_macro_rule_with_llm(
        self,
        reflection: Dict[str, Any],
        iteration: int,
        output_dir: Optional[str],
        trigger: str,
    ) -> Optional[tuple[str, float, Dict[str, Any]]]:
        if not self.llm_reflection_macro_enabled:
            return None

        recent_meso = self._rank_and_trim(self.meso_tips, max(1, self.macro_prompt_meso_k))
        recent_micro = self._rank_and_trim(self.micro_events, max(1, self.macro_prompt_micro_k))
        prior_macro = self._rank_and_trim(self.macro_rules, max(1, self.macro_prompt_history_k))

        prompt = f"""You are the Macro Reflection Agent for materials discovery.

Task key: {self.task_key}
Current iteration: {iteration}
Best Ed: {reflection.get('best_ed')}
Hit rate: {reflection.get('hit_rate')}
Valid rate: {reflection.get('valid_rate')}
Diversity: {reflection.get('diversity')}
Current macro min confidence threshold: {self.macro_min_confidence}

[Meso Tactical Memory]
{self._compact_lines(recent_meso, self.macro_prompt_meso_k)}

[Recent Micro Events]
{self._compact_lines(recent_micro, self.macro_prompt_micro_k)}

[Existing Macro Rules]
{self._compact_lines(prior_macro, self.macro_prompt_history_k)}

[Global Reflection Excerpt]
{str(reflection.get('llm_analysis') or '')[:1200]}

Produce ONE macro rule that is transferable across future runs for this task.
Prefer novel insight over rephrasing existing macro rules.
confidence must be a float in (0, 1].
Return strict JSON only:
{{
  "rule": "one sentence strategic rule (<=200 chars)",
  "confidence": 0.72,
  "evidence": ["short reason 1", "short reason 2"]
}}
"""
        obj = self._llm_generate_json(
            prompt,
            stage="macro",
            iteration=iteration,
            trigger=trigger,
            output_dir=output_dir,
            validator=lambda x: self._validate_confidence_field(x, "confidence"),
        )
        if not obj:
            return None

        rule = self._sanitize_message(obj.get("rule", ""), max_len=200)
        if not rule:
            return None
        if _is_low_information_macro_message(rule):
            logger.warning("HMRR macro rule rejected as low-information: %s", rule)
            return None

        confidence = _safe_float(obj.get("confidence"))
        if confidence is None:
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))

        evidence = obj.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        evidence_text = [self._sanitize_message(x, max_len=120) for x in evidence[:4] if str(x).strip()]

        return rule, confidence, {"source": "llm", "evidence_lines": evidence_text}

    def get_prompt_context(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "macro_rules": [], "meso_tips": [], "micro_events": []}

        macro = self._rank_and_trim(self.macro_rules, self.macro.max_items_for_prompt) if self.macro.enabled else []
        meso = self._rank_and_trim(self.meso_tips, self.meso.max_items_for_prompt) if self.meso.enabled else []
        micro = self._rank_and_trim(self.micro_events, self.micro.max_items_for_prompt) if self.micro.enabled else []
        return {
            "enabled": True,
            "macro_rules": macro,
            "meso_tips": meso,
            "micro_events": micro,
        }

    def save_snapshot(self, output_dir: Optional[str], iteration: int) -> None:
        if not (self.enabled and output_dir):
            return
        try:
            snapshot_path = Path(output_dir) / f"iteration_{iteration}" / "hmrr_snapshot.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "iteration": int(iteration),
                "task_key": self.task_key,
                "enabled": self.enabled,
                "best_metric": self.best_metric,
                "best_metric_source": self.best_metric_source,
                "stagnation_count": self.stagnation_count,
                "counts": {
                    "micro_events": len(self.micro_events),
                    "meso_tips": len(self.meso_tips),
                    "macro_rules": len(self.macro_rules),
                },
                "prompt_context": self.get_prompt_context(),
                "metrics": self.get_metrics(),
            }
            snapshot_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.warning("Failed to save HMRR snapshot for iteration %s: %s", iteration, exc)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "micro_events_count": len(self.micro_events),
            "meso_rules_count": len(self.meso_tips),
            "macro_rules_count": len(self.macro_rules),
            "meso_trigger_count": self.meso_trigger_count,
            "macro_trigger_count": self.macro_trigger_count,
            "macro_rules_loaded_count": self.macro_rules_loaded_count,
            "macro_rules_generated_count": self.macro_rules_generated_count,
            "stagnation_count": self.stagnation_count,
            "stagnation_patience": self.stagnation_patience,
            "macro_trigger_every_n_iters": self.trigger_every_n_iters,
            "macro_persist_across_runs": self.persist_macro_across_runs,
            "llm_reflection_enabled": self.llm_reflection_enabled,
            "llm_reflection_success_count": self.llm_reflection_success_count,
            "llm_reflection_failure_count": self.llm_reflection_failure_count,
            "strict_mode_enabled": self.strict_mode,
            "meso_trigger_policy": self.meso_trigger_policy,
            "retry_max_attempts": self.retry_max_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "trace_io": self.trace_io,
            "meso_llm_attempt_count": self.meso_llm_attempt_count,
            "meso_llm_success_count": self.meso_llm_success_count,
            "meso_llm_failure_count": self.meso_llm_failure_count,
            "macro_llm_attempt_count": self.macro_llm_attempt_count,
            "macro_llm_success_count": self.macro_llm_success_count,
            "macro_llm_failure_count": self.macro_llm_failure_count,
        }
