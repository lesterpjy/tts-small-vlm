"""RunContext - the single entry point for all three logging layers.

Pipeline code never touches Phoenix, OpenTelemetry, or wandb directly. It
goes through the `stage_span` / `llm_span` / `write_question` helpers on
`RunContext`, which fans out to:

  1. JSONL on disk (`runs/{run_id}/candidates.jsonl`) - source of truth
  2. Arize Phoenix (OTel span tree) - trace inspection
  3. Weights & Biases - scalars & config

The three layers are linked by a single `run_id` (see `make_run_id`).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml
from loguru import logger

from src.utils.tracing import init_phoenix
from src.utils.wandb_logger import init_wandb


def _set_error_status(span: Any) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode
        span.set_status(Status(StatusCode.ERROR))
    except Exception:
        pass


# OpenInference semantic convention attribute names.
# OpenInference semantic conventions - Phoenix's UI columns are wired to
# these exact keys. Custom attrs still land in the raw-attributes list but
# won't populate the Input/Output/Tokens/Model/Metadata columns.
_OI_SPAN_KIND = "openinference.span.kind"
_OI_INPUT_VALUE = "input.value"
_OI_OUTPUT_VALUE = "output.value"
_OI_SESSION_ID = "session.id"   # Phoenix "Session" grouping on the root span
_OI_METADATA = "metadata"       # JSON string -> Phoenix "Metadata" column
_OI_LLM_MODEL_NAME = "llm.model_name"
_OI_LLM_INVOCATION_PARAMS = "llm.invocation_parameters"     # JSON string
_OI_LLM_INPUT_MSG_ROLE = "llm.input_messages.0.message.role"
_OI_LLM_INPUT_MSG_CONTENT = "llm.input_messages.0.message.content"
_OI_LLM_OUTPUT_MSG_ROLE = "llm.output_messages.0.message.role"
_OI_LLM_OUTPUT_MSG_CONTENT = "llm.output_messages.0.message.content"
_OI_LLM_TOKEN_PROMPT = "llm.token_count.prompt"
_OI_LLM_TOKEN_COMPLETION = "llm.token_count.completion"
_OI_LLM_TOKEN_TOTAL = "llm.token_count.total"

# Sampling-param keys that `llm_span` will lift out of **attrs and pack into
# `llm.invocation_parameters` JSON (so Phoenix's Parameters view renders them).
_INVOCATION_KEYS = ("temperature", "top_p", "max_tokens", "max_new_tokens", "seed")


# ---------------------------------------------------------------------------
# run_id
# ---------------------------------------------------------------------------


def _model_short(model_name: str) -> str:
    """`Qwen/Qwen2.5-VL-7B-Instruct` -> `qwen25-vl-7b-instruct`."""
    tail = model_name.rsplit("/", 1)[-1]
    return tail.lower().replace(".", "").replace("_", "-")


def describe_strategy(config: dict) -> str:
    """Human-readable TTS-strategy label. Shows up on the W&B dashboard
    ("Strategy" column) and in Phoenix metadata."""
    runner = config.get("runner", "unknown")
    n = config.get("describe", {}).get("n_samples", 1)
    m = config.get("reason", {}).get("m_samples", 1)
    verifier = config.get("verify", {}).get("method", "majority_vote")
    sc_n = config.get("self_consistency", {}).get("n", 1)
    temp = config.get("generation", {}).get("temperature", 0.0)

    if runner == "zero_shot":
        style = config.get("generation", {}).get("prompt_style", "default")
        return f"Zero-shot MCQ (guided_choice A-E, prompt={style})"
    if runner == "cot":
        return f"Chain-of-Thought (single sample, T={temp})"
    if runner == "self_consistency":
        return f"Best-of-N Self-Consistency (N={sc_n}, T={temp}, majority vote)"
    if runner == "dtr":
        return (
            f"Describe-then-Reason (N_describe={n} x M_reason={m}, "
            f"verify={verifier})"
        )
    return runner


def experiment_metadata(config: dict, *, variant_name: str) -> dict[str, Any]:
    """Flat key->value dict of experiment provenance. Goes to W&B summary +
    Phoenix `metadata` blob. Kept flat so the W&B UI can use each as a
    filterable column."""
    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})
    return {
        "experiment/variant": variant_name,
        "experiment/pipeline": config.get("pipeline", "unknown"),
        "experiment/runner": config.get("runner", "unknown"),
        "experiment/strategy": describe_strategy(config),
        "experiment/model": model_cfg.get("name", "unknown"),
        "experiment/dtype": model_cfg.get("dtype", "unknown"),
        "experiment/max_model_len": model_cfg.get("max_model_len", 0),
        "experiment/verifier": config.get("verify", {}).get("method", "n/a"),
        "experiment/n_describe": config.get("describe", {}).get("n_samples", 1),
        "experiment/m_reason": config.get("reason", {}).get("m_samples", 1),
        "experiment/temperature": (
            config.get("generation", {}).get("temperature")
            or config.get("reason", {}).get("temperature", 0.0)
        ),
        "experiment/seed": eval_cfg.get("seed", 0),
        "experiment/subset_size": eval_cfg.get("subset_size", 0),
        "experiment/self_consistency_n": config.get("self_consistency", {}).get("n", 0),
    }


def make_run_id(
    config: dict,
    *,
    pipeline: str | None = None,
    extra: str | None = None,
    clock: float | None = None,
) -> str:
    """Build a deterministic run identifier.

    Format: `{pipeline}-{model_short}-{verifier}-n{N}m{M}-t{temp}-seed{seed}-{YYYYMMDDhhmm}`

    Baselines that have no `describe.n_samples` / `reason.m_samples` fall back
    to `n1m1` and `pipeline="baseline"`. Pass `pipeline="msa"` or a custom
    string for DTR variants.
    """
    model_name = config.get("model", {}).get("name", "unknown")
    model_short = _model_short(model_name)
    verifier = config.get("verify", {}).get("method", "majvote")
    verifier = {"majority_vote": "majvote"}.get(verifier, verifier)

    n = config.get("describe", {}).get("n_samples", 1)
    m = config.get("reason", {}).get("m_samples", 1)
    temp = config.get("reason", {}).get("temperature",
                                        config.get("describe", {}).get("temperature", 0.0))
    temp_tag = f"t{int(round(temp * 10)):02d}"
    seed = config.get("evaluation", {}).get("seed", 0)

    if pipeline is None:
        pipeline = "msa" if (n > 1 or m > 1) else "baseline"

    ts = time.strftime("%Y%m%d%H%M", time.localtime(clock))
    parts = [pipeline, model_short, verifier, f"n{n}m{m}", temp_tag, f"seed{seed}", ts]
    if extra:
        parts.insert(1, extra)
    return "-".join(parts)


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------


class _NullRun:
    """Drop-in replacement when wandb is unavailable."""
    summary: dict = {}

    def log(self, *_a, **_kw) -> None:
        pass

    def finish(self, *_a, **_kw) -> None:
        pass


class RunContext:
    def __init__(
        self,
        run_id: str,
        config: dict,
        out_dir: Path,
        tags: list[str] | None = None,
        group: str | None = None,
        *,
        variant_name: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.variant_name = variant_name or config.get("variant", "unknown")
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "candidates.jsonl"
        self._tags = tags or []
        self._group = group

        self._jsonl_fp = None
        self._tracer: Any = None
        self._wandb_run: Any = _NullRun()
        self._n_written = 0
        self._start_time = 0.0
        self._exp_meta: dict[str, Any] = {}
        self._exp_meta_json: str = "{}"

        # Aggregates for the end-of-run summary + stratified tables. These
        # stay O(#keys) not O(#questions), so they're safe for 4k-Q runs.
        self._n_correct = 0
        self._n_total_chains = 0
        self._n_extract_fail_chains = 0
        self._total_vlm_calls = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_latency_s = 0.0
        self._confidence_counts: dict[str, int] = {}
        # Stratified slots carry compute *and* accuracy so one W&B table
        # answers both "is accuracy bucket X worse?" and "does it cost more?".
        self._by_language: dict[str, dict[str, float]] = {}
        self._by_subject: dict[str, dict[str, float]] = {}
        self._by_content_type: dict[str, dict[str, float]] = {}

    # -- context manager --

    def __enter__(self) -> "RunContext":
        self._jsonl_fp = open(self.jsonl_path, "a", buffering=1, encoding="utf-8")
        self._tracer = init_phoenix(self.run_id)
        wb = init_wandb(self.run_id, self.config, self._tags, group=self._group)
        if wb is not None:
            self._wandb_run = wb
        self._start_time = time.time()

        # Flat experiment provenance - filterable in W&B, serialised into
        # Phoenix `metadata` JSON on every question root span.
        self._exp_meta = experiment_metadata(self.config, variant_name=self.variant_name)
        self._exp_meta_json = json.dumps(self._exp_meta, ensure_ascii=False, default=str)
        try:
            # Push to both summary (table columns) and config (run-level filters).
            for k, v in self._exp_meta.items():
                self._wandb_run.summary[k] = v
            self._wandb_run.config.update(self._exp_meta, allow_val_change=True)
        except Exception:
            pass

        (self.out_dir / "config.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info(
            "RunContext started: {} [{}] (out={})",
            self.run_id, self._exp_meta["experiment/strategy"], self.out_dir,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.time() - self._start_time

        self._log_final_summary(elapsed)

        try:
            self._wandb_run.finish(exit_code=0 if exc_type is None else 1)
        except Exception:
            pass

        if self._jsonl_fp is not None:
            self._jsonl_fp.close()
            self._jsonl_fp = None

        logger.info(
            "RunContext finished: {} questions in {:.1f}s",
            self._n_written,
            elapsed,
        )

    # -- span helpers --

    def _run_attrs(self) -> dict:
        """Attrs that must appear on every span so Phoenix UI can filter by
        run/variant without having to dig into the root span."""
        return {
            "run_id": self.run_id,
            "variant": self.variant_name,
        }

    def _common_attrs(self, question: dict) -> dict:
        return {
            **self._run_attrs(),
            "question_id": question.get("sample_id") or question.get("id") or "",
            "subject": question.get("subject") or "",
            "language": question.get("language") or "",
            "content_type": question.get("content_type") or "",
        }

    @contextmanager
    def question_span(self, question: dict) -> Iterator[Any]:
        attrs = self._common_attrs(question)
        attrs[_OI_SPAN_KIND] = "CHAIN"
        # session.id groups all spans of a run in one Phoenix "Session" pane.
        attrs[_OI_SESSION_ID] = self.run_id
        # Experiment provenance for Phoenix's Metadata column.
        attrs[_OI_METADATA] = self._exp_meta_json
        with self._tracer.start_as_current_span("question", attributes=attrs) as span:
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                _set_error_status(span)
                raise

    @contextmanager
    def stage_span(self, name: str, **attrs) -> Iterator[Any]:
        attrs.setdefault(_OI_SPAN_KIND, "CHAIN")
        for k, v in self._run_attrs().items():
            attrs.setdefault(k, v)
        with self._tracer.start_as_current_span(name, attributes=attrs) as span:
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                _set_error_status(span)
                raise

    @contextmanager
    def llm_span(
        self,
        name: str,
        *,
        prompt: str | None = None,
        **attrs,
    ) -> Iterator[Any]:
        """LLM span with full OpenInference semantic conventions.

        Recognized kwargs are promoted to the keys Phoenix's UI expects:
          - `model`       -> `llm.model_name`
          - `temperature`, `top_p`, `max_tokens`, `max_new_tokens`, `seed`
                          -> packed into `llm.invocation_parameters` JSON
          - `prompt`      -> `llm.input_messages.0.message.{role,content}`
                            and `input.value` (role defaults to "user")
        Unknown kwargs become plain span attributes.
        """
        attrs.setdefault(_OI_SPAN_KIND, "LLM")
        for k, v in self._run_attrs().items():
            attrs.setdefault(k, v)

        model_name = attrs.pop("model", None)
        if model_name:
            attrs[_OI_LLM_MODEL_NAME] = model_name

        invocation = {}
        for k in _INVOCATION_KEYS:
            if k in attrs and attrs[k] is not None:
                invocation[k] = attrs[k]
        if invocation:
            attrs[_OI_LLM_INVOCATION_PARAMS] = json.dumps(invocation, default=str)

        if prompt is not None:
            truncated = prompt[:4000]
            attrs[_OI_INPUT_VALUE] = truncated
            attrs[_OI_LLM_INPUT_MSG_ROLE] = "user"
            attrs[_OI_LLM_INPUT_MSG_CONTENT] = truncated

        with self._tracer.start_as_current_span(name, attributes=attrs) as span:
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                _set_error_status(span)
                raise

    @staticmethod
    def set_output(span: Any, output: str) -> None:
        """Backward-compatible: attach completion text only. Prefer
        `set_llm_output` when tokens/logprob are available."""
        if span is None:
            return
        try:
            truncated = (output or "")[:4000]
            span.set_attribute(_OI_OUTPUT_VALUE, truncated)
            span.set_attribute(_OI_LLM_OUTPUT_MSG_ROLE, "assistant")
            span.set_attribute(_OI_LLM_OUTPUT_MSG_CONTENT, truncated)
        except Exception:
            pass

    @staticmethod
    def set_llm_output(
        span: Any,
        output: str,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_s: float | None = None,
        logprob: float | None = None,
        extracted_answer: str | None = None,
    ) -> None:
        """Attach completion + token counts + latency to an LLM span using
        OpenInference conventions, so Phoenix's Output/Tokens columns render."""
        if span is None:
            return
        try:
            truncated = (output or "")[:4000]
            span.set_attribute(_OI_OUTPUT_VALUE, truncated)
            span.set_attribute(_OI_LLM_OUTPUT_MSG_ROLE, "assistant")
            span.set_attribute(_OI_LLM_OUTPUT_MSG_CONTENT, truncated)

            if prompt_tokens is not None:
                span.set_attribute(_OI_LLM_TOKEN_PROMPT, int(prompt_tokens))
            if completion_tokens is not None:
                span.set_attribute(_OI_LLM_TOKEN_COMPLETION, int(completion_tokens))
            if prompt_tokens is not None and completion_tokens is not None:
                span.set_attribute(_OI_LLM_TOKEN_TOTAL,
                                   int(prompt_tokens) + int(completion_tokens))

            if latency_s is not None:
                span.set_attribute("latency_s", float(latency_s))
            if logprob is not None:
                span.set_attribute("logprob_mean", float(logprob))
            if extracted_answer is not None:
                span.set_attribute("extracted_answer", extracted_answer)
        except Exception:
            pass

    # -- end-of-run summary --

    def _log_final_summary(self, elapsed: float) -> None:
        """Write scalar summary + stratified tables to W&B. Safe no-op when
        W&B is unavailable (self._wandb_run is _NullRun)."""
        n_q = max(self._n_written, 1)
        accuracy = self._n_correct / n_q if self._n_written else 0.0
        extract_fail_rate = (
            self._n_extract_fail_chains / self._n_total_chains
            if self._n_total_chains else 0.0
        )
        total_tokens = self._total_prompt_tokens + self._total_completion_tokens

        try:
            s = self._wandb_run.summary
            s["wall_clock_s"] = elapsed
            s["questions"] = self._n_written
            s["accuracy"] = accuracy
            s["n_correct"] = self._n_correct
            s["extract_fail_rate"] = extract_fail_rate
            s["n_total_chains"] = self._n_total_chains
            s["n_extract_fail_chains"] = self._n_extract_fail_chains
            # Compute block for accuracy vs VLM calls analysis. Sums are
            # useful for a run-level headline number; means for comparing
            # variants at matched Q count.
            s["compute/vlm_calls_total"] = self._total_vlm_calls
            s["compute/prompt_tokens_total"] = self._total_prompt_tokens
            s["compute/completion_tokens_total"] = self._total_completion_tokens
            s["compute/tokens_total"] = total_tokens
            s["compute/vlm_calls_per_q_mean"] = self._total_vlm_calls / n_q
            s["compute/tokens_per_q_mean"] = total_tokens / n_q
            s["compute/latency_per_q_mean"] = self._total_latency_s / n_q
            s["compute/chains_per_q_mean"] = self._n_total_chains / n_q
            for conf, cnt in self._confidence_counts.items():
                s[f"confidence/{conf}"] = cnt
        except Exception:
            pass

        # Stratified tables go through wandb.log (not summary); they require
        # the wandb module to construct wandb.Table, so the _NullRun branch
        # has to skip them entirely. Columns cover both accuracy and compute
        # so each stratum answers both questions in one view.
        try:
            import wandb

            def _table(bucket: dict[str, dict[str, float]]) -> Any:
                rows = []
                for k, v in sorted(bucket.items()):
                    t = max(int(v["total"]), 1)
                    rows.append([
                        k, int(v["total"]), int(v["correct"]),
                        v["correct"] / t,
                        v.get("vlm_calls_sum", 0) / t,
                        v.get("tokens_sum", 0) / t,
                        v.get("latency_sum", 0.0) / t,
                    ])
                return wandb.Table(
                    columns=[
                        "key", "total", "correct", "accuracy",
                        "mean_vlm_calls", "mean_tokens", "mean_latency_s",
                    ],
                    data=rows,
                )

            self._wandb_run.log({
                "stratified/by_language": _table(self._by_language),
                "stratified/by_subject": _table(self._by_subject),
                "stratified/by_content_type": _table(self._by_content_type),
                "final/accuracy": accuracy,
                "final/extract_fail_rate": extract_fail_rate,
                "final/tokens_per_q_mean": total_tokens / n_q,
                "final/vlm_calls_per_q_mean": self._total_vlm_calls / n_q,
            })
        except Exception:
            # wandb not installed, or _NullRun (no .log table support).
            # JSONL + loguru still have the full picture.
            pass

    # -- JSONL writes --

    def _update_aggregates(self, record: dict) -> None:
        correct = bool(record.get("correct"))
        if correct:
            self._n_correct += 1

        chains = record.get("chains") or []
        descriptions = record.get("descriptions") or []
        prompt_toks = sum(int(c.get("prompt_tokens") or 0) for c in chains)
        prompt_toks += sum(int(d.get("prompt_tokens") or 0) for d in descriptions)
        completion_toks = sum(int(c.get("completion_tokens") or 0) for c in chains)
        completion_toks += sum(int(d.get("completion_tokens") or 0) for d in descriptions)
        vlm_calls = int(record.get("total_vlm_calls") or (len(chains) + len(descriptions)))
        latency = float(record.get("total_latency_s") or 0.0)

        self._n_total_chains += len(chains)
        self._n_extract_fail_chains += sum(
            1 for c in chains if c.get("extracted_answer") is None
        )
        self._total_vlm_calls += vlm_calls
        self._total_prompt_tokens += prompt_toks
        self._total_completion_tokens += completion_toks
        self._total_latency_s += latency

        for bucket_name, bucket in (
            ("language", self._by_language),
            ("subject", self._by_subject),
            ("content_type", self._by_content_type),
        ):
            key = record.get(bucket_name) or "unknown"
            slot = bucket.setdefault(key, {
                "total": 0, "correct": 0,
                "vlm_calls_sum": 0, "tokens_sum": 0, "latency_sum": 0.0,
            })
            slot["total"] += 1
            if correct:
                slot["correct"] += 1
            slot["vlm_calls_sum"] += vlm_calls
            slot["tokens_sum"] += prompt_toks + completion_toks
            slot["latency_sum"] += latency

        conf = (record.get("verifier") or {}).get("confidence") or "unknown"
        self._confidence_counts[conf] = self._confidence_counts.get(conf, 0) + 1

    def write_question(self, record: dict) -> None:
        """Append one question record to JSONL + stream per-question scalars
        to W&B (accuracy, latency, compute). Aggregates are updated here so
        __exit__ can emit a final summary + stratified tables."""
        assert self._jsonl_fp is not None, "RunContext must be entered before write_question"
        self._jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl_fp.flush()
        self._n_written += 1

        self._update_aggregates(record)

        try:
            chains = record.get("chains") or []
            descriptions = record.get("descriptions") or []
            q_prompt_toks = (
                sum(int(c.get("prompt_tokens") or 0) for c in chains)
                + sum(int(d.get("prompt_tokens") or 0) for d in descriptions)
            )
            q_completion_toks = (
                sum(int(c.get("completion_tokens") or 0) for c in chains)
                + sum(int(d.get("completion_tokens") or 0) for d in descriptions)
            )
            running_acc = self._n_correct / self._n_written
            fail_rate = (
                self._n_extract_fail_chains / self._n_total_chains
                if self._n_total_chains else 0.0
            )
            self._wandb_run.log({
                "question/correct": int(bool(record.get("correct"))),
                "question/latency_s": record.get("total_latency_s", 0.0),
                "question/vlm_calls": record.get("total_vlm_calls", 0),
                "question/n_chains": len(chains),
                "question/n_descriptions": len(descriptions),
                "question/tokens_prompt": q_prompt_toks,
                "question/tokens_completion": q_completion_toks,
                "question/tokens_total": q_prompt_toks + q_completion_toks,
                "running/accuracy": running_acc,
                "running/extract_fail_rate": fail_rate,
                "running/tokens_per_q_mean": (
                    (self._total_prompt_tokens + self._total_completion_tokens)
                    / self._n_written
                ),
                "progress/questions_done": self._n_written,
                "progress/elapsed_s": time.time() - self._start_time,
            })
        except Exception:
            pass


@contextmanager
def run_context(
    run_id: str,
    config: dict,
    out_dir: Path,
    tags: list[str] | None = None,
    group: str | None = None,
    *,
    variant_name: str | None = None,
) -> Iterator[RunContext]:
    ctx = RunContext(
        run_id=run_id,
        config=config,
        out_dir=out_dir,
        tags=tags,
        group=group,
        variant_name=variant_name,
    )
    with ctx as active:
        yield active
