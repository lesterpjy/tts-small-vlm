"""W&B initialization. Called once per run by RunContext.

Kept thin: just the `wandb.init` call plus env-var wiring. All actual logging
(scalars, summary, scaling curves) is issued by RunContext or analyze.py.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger


def init_wandb(
    run_id: str,
    config: dict,
    tags: list[str] | None = None,
    group: str | None = None,
) -> Any | None:
    """Start a W&B run whose display name equals `run_id`.

    Reads `WANDB_PROJECT` and `WANDB_ENTITY` from env (defaults match
    `.env.example`). Set `WANDB_MODE=offline` on clusters without internet.

    Returns the W&B Run object, or `None` if `wandb` is not importable or
    initialization fails, in which case runs continue without W&B tracking
    (the JSONL on disk is still the source of truth).
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping W&B logging")
        return None

    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "tts-small-vlm"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=run_id,
            config={**config, "run_id": run_id},
            tags=tags or [],
            group=group,
            job_type="eval",
            reinit=False,
        )
        logger.info("W&B run started: {}", run_id)
        return run
    except Exception as exc:
        logger.warning("W&B init failed ({}); continuing without W&B", exc)
        return None
