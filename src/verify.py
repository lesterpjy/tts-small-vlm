"""Stage 3: Verification and selection.

Five strategies:
1. Majority vote - group candidates by answer, pick the most common
2. Agentic verification - TIM-PRM-inspired independent question asking
3. Generative critic - GM-PRM-style (Zhang et al. 2508.04088) training-free
   prompted rubric over step-intent / visual-alignment / logical-soundness
4. Qwen-VL-PRM-7B - external process reward model
5. Qwen-VL-PRM-7B DTR-segmented - perception/reasoning attribution via
   [Perception]/[Reasoning] step tagging

When called through `RunContext`, emits one parent `verify` span and a
`verifier.{method}` child span.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import TYPE_CHECKING

from loguru import logger
from PIL import Image

from src.backend import generate
from src.utils import Candidate, SelectionResult

if TYPE_CHECKING:
    from src.utils.logger import RunContext

# --- Majority vote ---


def majority_vote(candidates: list[Candidate]) -> SelectionResult:
    """Select answer by majority vote across all candidates.

    Tie-breaking:
    - Clear majority (>50%): high confidence
    - Plurality (<50%): medium confidence
    - Exact tie: break by highest mean logprob, low confidence
    - All disagree: pick best logprob candidate, very_low confidence
    """
    valid = [c for c in candidates if c.answer is not None]
    if not valid:
        return SelectionResult(
            answer=None, method="majority_vote", confidence="very_low",
            metadata={"reason": "no_valid_candidates"},
        )

    vote_counts = Counter(c.answer for c in valid)
    total_votes = len(valid)
    top_count = vote_counts.most_common(1)[0][1]
    tied_answers = [ans for ans, cnt in vote_counts.items() if cnt == top_count]

    if len(tied_answers) == 1:
        # Single winner
        winner = tied_answers[0]
        fraction = top_count / total_votes
        confidence = "high" if fraction > 0.5 else "medium"
        best = _best_candidate_for_answer(valid, winner)
        return SelectionResult(
            answer=winner,
            method="majority_vote",
            confidence=confidence,
            vote_counts=dict(vote_counts),
            selected_candidate=best,
            metadata={"fraction": fraction},
        )

    # Tie: break by logprob
    winner, best = _break_tie_by_logprob(valid, tied_answers)
    return SelectionResult(
        answer=winner,
        method="majority_vote",
        confidence="low",
        vote_counts=dict(vote_counts),
        selected_candidate=best,
        metadata={"tied_answers": tied_answers, "tiebreak": "logprob"},
    )


def _best_candidate_for_answer(
    candidates: list[Candidate], answer: str
) -> Candidate | None:
    """Return the candidate with the highest logprob for a given answer."""
    matching = [c for c in candidates if c.answer == answer]
    if not matching:
        return None
    return max(matching, key=lambda c: c.logprob if c.logprob is not None else float("-inf"))


def _break_tie_by_logprob(
    candidates: list[Candidate], tied_answers: list[str]
) -> tuple[str, Candidate | None]:
    """Break a tie by selecting the answer with the best individual logprob."""
    best_candidate = None
    best_logprob = float("-inf")

    for ans in tied_answers:
        for c in candidates:
            if c.answer == ans and c.logprob is not None and c.logprob > best_logprob:
                best_logprob = c.logprob
                best_candidate = c

    if best_candidate:
        return best_candidate.answer, best_candidate
    # Fallback: pick first tied answer
    return tied_answers[0], _best_candidate_for_answer(candidates, tied_answers[0])


# --- Agentic verification (TIM-PRM-inspired) ---

EXTRACT_CLAIMS_PROMPT = (
    "List the specific visual facts stated in the description below. "
    "Each fact should be something verifiable by looking at the image "
    "(e.g., 'the graph shows a parabola', 'option A says 42', 'the table has 3 rows').\n"
    "Output as a numbered list. Maximum {max_claims} facts.\n\n"
    "Description:\n{description}"
)

GENERATE_QUESTIONS_PROMPT = (
    "For each fact below, write a short question that could verify it "
    "by looking at an image. Do NOT reference the original description.\n\n"
    "Facts:\n{claims}\n\n"
    "Output as a numbered list of questions, one per fact."
)


def agentic_verify(
    model,
    processor,
    image: Image.Image,
    candidates: list[Candidate],
    pre_filter_top_k: int = 8,
    skip_threshold: float = 0.75,
    max_claims: int = 5,
) -> SelectionResult:
    """Agentic verification: verify visual claims with independent questions.

    1. Run majority vote first as pre-filter
    2. If majority is strong (>skip_threshold), trust it
    3. Otherwise, verify top-K candidates by checking visual claims
    4. Select the candidate with highest verification score
    """
    # Step 1: majority vote pre-filter
    mv_result = majority_vote(candidates)
    valid = [c for c in candidates if c.answer is not None]

    if not valid:
        return mv_result

    total_votes = len(valid)
    top_count = max(Counter(c.answer for c in valid).values())
    majority_fraction = top_count / total_votes

    # Step 2: skip verification if majority is strong
    if majority_fraction >= skip_threshold:
        logger.info("Majority fraction {:.2f} >= {:.2f}, skipping agentic verification",
                     majority_fraction, skip_threshold)
        mv_result.method = "agentic_skip"
        return mv_result

    # Step 3: select candidates to verify
    vote_counts = Counter(c.answer for c in valid)
    scored = sorted(valid, key=lambda c: (
        vote_counts.get(c.answer, 0),
        c.logprob if c.logprob is not None else float("-inf"),
    ), reverse=True)
    to_verify = scored[:pre_filter_top_k]

    # Step 4: verify each candidate
    best_candidate = None
    best_score = -1.0
    scores = {}

    for candidate in to_verify:
        score = _verify_candidate(model, processor, image, candidate, max_claims)
        scores[candidate.uid] = score
        if score > best_score:
            best_score = score
            best_candidate = candidate

    logger.info("Agentic verification scores: {}", scores)

    if best_candidate:
        return SelectionResult(
            answer=best_candidate.answer,
            method="agentic",
            confidence="medium" if best_score > 0.5 else "low",
            vote_counts=dict(vote_counts),
            selected_candidate=best_candidate,
            metadata={"verification_scores": scores, "majority_fraction": majority_fraction},
        )

    # Fallback to majority vote
    return mv_result


def _verify_candidate(model, processor, image, candidate, max_claims) -> float:
    """Verify a single candidate by checking its visual claims.

    Returns a score in [0, 1] representing the fraction of claims verified.
    """
    # Step 1: extract claims from the description (text-only)
    prompt = EXTRACT_CLAIMS_PROMPT.format(description=candidate.description, max_claims=max_claims)
    claims_text = generate(model, processor, prompt, temperature=0.0, max_new_tokens=300).text
    if not claims_text.strip():
        return 0.0

    # Step 2: generate independent questions (text-only)
    prompt = GENERATE_QUESTIONS_PROMPT.format(claims=claims_text)
    questions = generate(model, processor, prompt, temperature=0.0, max_new_tokens=300).text
    if not questions.strip():
        return 0.0

    # Step 3: ask questions about the image independently (multimodal)
    prompt = f"Answer each question below briefly based on what you see in the image.\n\n{questions}"
    answers = generate(model, processor, prompt, image=image, temperature=0.0, max_new_tokens=500).text

    # Step 4: compare claims to independent answers (text-only)
    prompt = (
        "Compare the original claims with the independent answers below.\n"
        "For each claim, decide if the independent answer AGREES or DISAGREES.\n"
        "Output one line per claim: 'AGREE' or 'DISAGREE'.\n\n"
        f"Original claims:\n{claims_text}\n\n"
        f"Independent answers:\n{answers}"
    )
    result = generate(model, processor, prompt, temperature=0.0, max_new_tokens=200).text

    # Count agreements
    lines = result.strip().split("\n")
    agrees = len([l for l in lines if "AGREE" in l.upper() and "DISAGREE" not in l.upper()])
    total = len([l for l in lines if "AGREE" in l.upper() or "DISAGREE" in l.upper()])

    score = agrees / total if total > 0 else 0.0
    logger.debug("Candidate {} verification score: {:.2f}", candidate.uid, score)
    return score


# --- Generative critic (GM-PRM-style) ---

# Training-free, prompted-only rubric scoring. Derives from GM-PRM (Zhang et
# al. arXiv 2508.04088). The policy backbone rates a candidate's full reasoning
# chain on three axes (step-intent, visual-alignment, logical-soundness) with
# strict JSON output; per-axis scores are mean-aggregated to a scalar in [0, 1].
# Doubles as the step-scorer fallback for PRM-BAS when Qwen-VL-PRM is
# unavailable.

GENERATIVE_CRITIC_AXES: dict[str, str] = {
    "step-intent": (
        "Does each reasoning step address the question directly and build "
        "toward a commitment to one of the answer options? Penalise irrelevant "
        "tangents, repetition, self-doubt loops, or failure to commit to a letter."
    ),
    "visual-alignment": (
        "Does the chain's reasoning match what is actually visible in the image? "
        "Penalise hallucinated elements, misread values, missing visual evidence, "
        "or contradictions with the options rendered in the image."
    ),
    "logical-soundness": (
        "Does the final answer follow from the premises via valid logical, "
        "mathematical, or scientific steps? Penalise unjustified leaps, "
        "false equivalences, arithmetic errors, or a mismatch between the "
        "concluded letter and the reasoning that preceded it."
    ),
}

GENERATIVE_CRITIC_PROMPT = (
    "You are a rigorous evaluator reviewing a reasoning chain written by a "
    "student answering a multiple-choice exam question. The question and "
    "its answer options are shown in the image.\n\n"
    "Rate the chain on ONE axis: {axis_name}.\n\n"
    "Axis definition: {axis_definition}\n\n"
    "Scale: 1 (severely deficient), 2 (weak), 3 (adequate), 4 (strong), "
    "5 (excellent).\n\n"
    "Reasoning chain to evaluate:\n---\n{chain_text}\n---\n\n"
    "Output ONLY a JSON object on a single line with exactly two fields:\n"
    '{{"score": <integer 1-5>, "reason": "<brief one-sentence justification>"}}\n'
    "Do not output anything else. Do not wrap in code fences."
)


def _parse_critic_score(text: str) -> tuple[int | None, str | None]:
    """Parse one axis's critic output.

    Returns `(score, reason)` with `score` in {1..5} or `None` on failure.
    Strategy order:
      1. Fenced JSON block - despite the prompt forbidding fences, models
         sometimes add them; be forgiving.
      2. Bare JSON substring matching `{..."score"...}`.
      3. Regex fallback for `"score": N` with N in 1..5.
    """
    if not text:
        return None, None

    # (1) fenced JSON
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            s = obj.get("score")
            if isinstance(s, int) and 1 <= s <= 5:
                return s, obj.get("reason")
        except (json.JSONDecodeError, AttributeError):
            pass

    # (2) bare JSON object containing a "score" key
    m = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            s = obj.get("score")
            if isinstance(s, int) and 1 <= s <= 5:
                return s, obj.get("reason")
        except (json.JSONDecodeError, AttributeError):
            pass

    # (3) plain `"score": N` key/value regardless of wrapping
    m = re.search(r'"score"\s*:\s*([1-5])\b', text)
    if m:
        return int(m.group(1)), None

    return None, None


def generative_critic_score(
    model,
    processor,
    image: Image.Image,
    candidate: Candidate,
    axes: list[str] | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> float:
    """Score a candidate's reasoning chain via a GM-PRM-style generative critic.

    One `generate()` call per axis (three by default). Each axis's
    integer score (1..5) is mapped to [0, 1] via `(s - 1) / 4`, then the
    per-axis scores are mean-aggregated. Unparseable axes fall back to 0.5
    (neutral) and are logged; if every axis fails, the candidate score is
    0.5. Return value is clamped to [0, 1].
    """
    if axes is None:
        axes = list(GENERATIVE_CRITIC_AXES.keys())

    per_axis_scores: list[float] = []
    for axis in axes:
        definition = GENERATIVE_CRITIC_AXES.get(axis)
        if definition is None:
            logger.warning("Unknown critic axis '{}', skipping", axis)
            continue

        prompt = GENERATIVE_CRITIC_PROMPT.format(
            axis_name=axis,
            axis_definition=definition,
            chain_text=candidate.reasoning,
        )
        out = generate(
            model, processor, prompt, image=image,
            temperature=temperature, max_new_tokens=max_tokens,
        )
        score_raw, _reason = _parse_critic_score(out.text)

        if score_raw is None:
            logger.warning(
                "Critic axis '{}' unparseable for candidate {}: {!r}",
                axis, candidate.uid, (out.text or "")[:120],
            )
            per_axis_scores.append(0.5)
        else:
            per_axis_scores.append((score_raw - 1) / 4.0)

    if not per_axis_scores:
        return 0.5

    mean = sum(per_axis_scores) / len(per_axis_scores)
    return max(0.0, min(1.0, mean))


def generative_critic_rank(
    candidates: list[Candidate],
    model,
    processor,
    image: Image.Image,
    axes: list[str] | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> SelectionResult:
    """Score every candidate with `generative_critic_score`, pick the highest.

    Unparseable candidates (answer=None) are excluded from scoring. A critic
    scoring a chain that already failed the extractor is wasted compute. If
    every candidate is unparseable, returns a `very_low`-confidence `None`.
    """
    if not candidates:
        return SelectionResult(
            answer=None, method="generative", confidence="very_low",
            metadata={"reason": "no_candidates"},
        )

    valid = [c for c in candidates if c.answer is not None]
    if not valid:
        return SelectionResult(
            answer=None, method="generative", confidence="very_low",
            metadata={"reason": "no_parseable_candidates"},
        )

    scored: list[tuple[Candidate, float]] = []
    for c in valid:
        s = generative_critic_score(
            model, processor, image, c, axes=axes,
            temperature=temperature, max_tokens=max_tokens,
        )
        scored.append((c, s))

    scored.sort(key=lambda t: t[1], reverse=True)
    best_c, best_s = scored[0]

    # Confidence from top-two gap. Generous thresholds since critic scores
    # on a 3-axis mean over a 5-point scale are noisy at the 0.1 level.
    if len(scored) > 1:
        gap = best_s - scored[1][1]
        confidence = "high" if gap > 0.2 else "medium" if gap > 0.1 else "low"
    else:
        confidence = "medium"

    vote_counts = Counter(c.answer for c in valid)

    return SelectionResult(
        answer=best_c.answer,
        method="generative",
        confidence=confidence,
        vote_counts=dict(vote_counts),
        selected_candidate=best_c,
        metadata={
            "critic_scores": {c.uid: s for c, s in scored},
            "axes": list(axes or GENERATIVE_CRITIC_AXES.keys()),
            "top_score": best_s,
            "score_gap": (scored[0][1] - scored[1][1]) if len(scored) > 1 else None,
        },
    )


# --- Qwen-VL-PRM-7B wrapper (cross-model discriminative arm) ---
#
# Clean-room reimplementation from arXiv 2509.23250 + the public HF docs at
# ob11/Qwen-VL-PRM-7B.

QWEN_VL_PRM_MODEL_ID = "ob11/Qwen-VL-PRM-7B"

QWEN_VL_PRM_SYSTEM_PROMPT = (
    "You are an expert process reward model. You will be shown an image "
    "containing a multiple-choice question and a solution process. Judge "
    "whether the solution process so far is on a correct path to the right "
    "answer. Reply with exactly one token: + if the step is correct, - if not."
)


def load_qwen_vl_prm(model_id: str = QWEN_VL_PRM_MODEL_ID):
    """Load Qwen-VL-PRM-7B for inference. Returns (model, processor, tokenizer).

    Cached pos_id / neg_id can be retrieved with `_qwen_vl_prm_pm_token_ids`.
    Lazy torch / transformers import so callers without GPU still get the
    rest of verify.py.
    """
    import torch  # noqa: F401  (defer until call time)
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    logger.info("Loading Qwen-VL-PRM from {} (bf16, device_map=auto)...", model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor, tokenizer


def _qwen_vl_prm_pm_token_ids(tokenizer) -> tuple[int, int]:
    """Resolve the integer ids of the literal '+' and '-' tokens.

    Matches what the PRM was trained to emit at the final position; the
    softmax restricted to these two ids gives P(+).
    """
    pos = tokenizer.encode("+", add_special_tokens=False)
    neg = tokenizer.encode("-", add_special_tokens=False)
    if not pos or not neg:
        raise RuntimeError(
            f"Could not resolve +/- token ids; tokenizer.encode('+')={pos}, "
            f"encode('-')={neg}. Wrong tokenizer?"
        )
    return pos[0], neg[0]


def _split_into_steps(reasoning: str) -> list[str]:
    """Split a chain into reasoning steps for per-step PRM scoring.

    Heuristic: split on consecutive newlines; skip empties; if the chain
    contains no double-newline, treat the whole text as one step. The
    upstream PRM was trained on chains delimited by `Step N:` markers and
    blank lines; this gives both shapes a reasonable spelling. Pure
    whitespace / empty input returns []; a single-step chain returns the
    stripped text in a one-element list.
    """
    if not reasoning or not reasoning.strip():
        return []
    parts = [s.strip() for s in re.split(r"\n\s*\n", reasoning) if s.strip()]
    return parts or [reasoning.strip()]


def _qwen_vl_prm_forward_score(
    *, prm_model, prm_processor, prm_tokenizer,
    image: Image.Image, question: str, solution: str,
    system_prompt: str | None = None,
) -> float:
    """One forward pass over (system, image+question+solution) -> P(+).

    Reads the logits at the final position, softmax over the [+, -] pair only.
    """
    import math
    import torch

    sys_text = system_prompt or QWEN_VL_PRM_SYSTEM_PROMPT
    user_text = f"### Question:\n{question}\n\n### Solution Process:\n{solution}"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_text}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = prm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = prm_processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(prm_model.device)
    pos_id, neg_id = _qwen_vl_prm_pm_token_ids(prm_tokenizer)
    with torch.no_grad():
        out = prm_model(**inputs)
    last = out.logits[0, -1]
    pos_logit = float(last[pos_id].item())
    neg_logit = float(last[neg_id].item())
    m = max(pos_logit, neg_logit)
    z = math.exp(pos_logit - m) + math.exp(neg_logit - m)
    prob_pos = math.exp(pos_logit - m) / z
    return float(max(0.0, min(1.0, prob_pos)))


def qwen_vl_prm_score(
    *, prm_model, prm_processor, prm_tokenizer,
    image: Image.Image, question: str, steps: list[str],
    system_prompt: str | None = None,
) -> list[float]:
    """Score each step independently as a partial solution; return list[P(+)].

    The PRM is conditioned on the full prefix `steps[0..i]` when scoring
    `steps[i]`, matching the paper's Step-Score-Aggregation evaluation:
    you can mean these scores per chain, or take the prefix-monotonic
    minimum, depending on selection policy. Empty `steps` returns empty list.
    """
    if not steps:
        return []
    scores: list[float] = []
    prefix: list[str] = []
    for step in steps:
        partial = "\n\n".join([*prefix, step])
        prob_pos = _qwen_vl_prm_forward_score(
            prm_model=prm_model, prm_processor=prm_processor,
            prm_tokenizer=prm_tokenizer, image=image,
            question=question, solution=partial,
            system_prompt=system_prompt,
        )
        scores.append(prob_pos)
        prefix.append(step)
    return scores


def qwen_vl_prm_score_one_shot(
    *, prm_model, prm_processor, prm_tokenizer,
    image: Image.Image, question: str, solution: str,
    system_prompt: str | None = None,
) -> float:
    """Score the entire chain as one PRM forward pass (paper's One-shot mode).

    Cheaper than per-step scoring by a factor of n_steps. Reasonable when
    you only need a single P(+) for the whole chain; per-step scoring is
    needed for early-step beam pruning (PRM-BAS) or step-attribution.
    """
    return _qwen_vl_prm_forward_score(
        prm_model=prm_model, prm_processor=prm_processor,
        prm_tokenizer=prm_tokenizer, image=image,
        question=question, solution=solution,
        system_prompt=system_prompt,
    )


def qwen_vl_prm_rank(
    candidates: list[Candidate],
    *,
    image: Image.Image,
    question: str,
    prm_model,
    prm_processor,
    prm_tokenizer,
    system_prompt: str | None = None,
    mode: str = "step_mean",
) -> SelectionResult:
    """Score every candidate and pick the highest-mean P(+).

    `mode` in {"step_mean", "step_min", "one_shot"}.
    - step_mean: per-step P(+), averaged (paper's Step-Score-Aggregation).
    - step_min: per-step P(+), minimum (worst-step). Pessimistic; useful
      for catching chains that drift even if the final answer looks right.
    - one_shot: score the full solution in a single forward pass. ~n_steps
      times cheaper but loses per-step granularity.
    """
    if not candidates:
        return SelectionResult(
            answer=None, method="qwen_vl_prm", confidence="very_low",
            metadata={"reason": "no_candidates"},
        )
    valid = [c for c in candidates if c.answer is not None]
    if not valid:
        return SelectionResult(
            answer=None, method="qwen_vl_prm", confidence="very_low",
            metadata={"reason": "no_parseable_candidates"},
        )

    scored: list[tuple[Candidate, float]] = []
    per_chain_steps: dict[str, list[float]] = {}
    for c in valid:
        if mode == "one_shot":
            score = qwen_vl_prm_score_one_shot(
                prm_model=prm_model, prm_processor=prm_processor,
                prm_tokenizer=prm_tokenizer, image=image,
                question=question, solution=c.reasoning,
                system_prompt=system_prompt,
            )
            per_chain_steps[c.uid] = [score]
        else:
            steps = _split_into_steps(c.reasoning)
            step_scores = qwen_vl_prm_score(
                prm_model=prm_model, prm_processor=prm_processor,
                prm_tokenizer=prm_tokenizer, image=image,
                question=question, steps=steps,
                system_prompt=system_prompt,
            )
            per_chain_steps[c.uid] = step_scores
            if not step_scores:
                score = 0.0
            elif mode == "step_min":
                score = min(step_scores)
            else:  # step_mean (default)
                score = sum(step_scores) / len(step_scores)
        scored.append((c, score))

    scored.sort(key=lambda t: t[1], reverse=True)
    best_c, best_s = scored[0]

    if len(scored) > 1:
        gap = best_s - scored[1][1]
        confidence = "high" if gap > 0.2 else "medium" if gap > 0.1 else "low"
    else:
        confidence = "medium"

    vote_counts = Counter(c.answer for c in valid)

    return SelectionResult(
        answer=best_c.answer,
        method="qwen_vl_prm",
        confidence=confidence,
        vote_counts=dict(vote_counts),
        selected_candidate=best_c,
        metadata={
            "prm_scores": {c.uid: s for c, s in scored},
            "per_chain_step_scores": per_chain_steps,
            "mode": mode,
            "top_score": best_s,
            "score_gap": (scored[0][1] - scored[1][1]) if len(scored) > 1 else None,
        },
    )


# --- Qwen-VL-PRM DTR-segmented scoring ---


def qwen_vl_prm_score_dtr_segmented(
    *,
    prm_model,
    prm_processor,
    prm_tokenizer,
    image: Image.Image,
    question: str,
    description: str,
    reasoning: str,
    system_prompt: str | None = None,
) -> dict:
    """Score a DTR candidate with [Perception]/[Reasoning] segmentation.

    Formats the solution as tagged steps so the PRM (trained on VL-PRM300K
    with segment tags) can attribute errors to the perception or reasoning
    stage. The description is treated as a single perception step; the
    reasoning chain is split on double-newlines as usual.

    Returns a dict with overall, per-segment, and per-step scores.
    """
    if not description or not description.strip():
        desc_steps: list[str] = []
    else:
        desc_steps = [description.strip()]

    reason_steps = _split_into_steps(reasoning)

    tagged_steps = [f"[Perception]\n{s}" for s in desc_steps] + [
        f"[Reasoning]\n{s}" for s in reason_steps
    ]

    if not tagged_steps:
        return {
            "overall": 0.0,
            "perception_score": 0.0,
            "reasoning_mean": 0.0,
            "reasoning_min": 0.0,
            "perception_steps": [],
            "reasoning_steps": [],
            "n_perception_steps": 0,
            "n_reasoning_steps": 0,
        }

    all_scores = qwen_vl_prm_score(
        prm_model=prm_model,
        prm_processor=prm_processor,
        prm_tokenizer=prm_tokenizer,
        image=image,
        question=question,
        steps=tagged_steps,
        system_prompt=system_prompt,
    )

    n_desc = len(desc_steps)
    perception_scores = all_scores[:n_desc]
    reasoning_scores = all_scores[n_desc:]

    perception_mean = (
        sum(perception_scores) / len(perception_scores)
        if perception_scores
        else 0.0
    )
    reasoning_mean = (
        sum(reasoning_scores) / len(reasoning_scores)
        if reasoning_scores
        else 0.0
    )
    overall = sum(all_scores) / len(all_scores)

    return {
        "overall": overall,
        "perception_score": perception_mean,
        "reasoning_mean": reasoning_mean,
        "reasoning_min": min(reasoning_scores) if reasoning_scores else 0.0,
        "perception_steps": perception_scores,
        "reasoning_steps": reasoning_scores,
        "n_perception_steps": n_desc,
        "n_reasoning_steps": len(reason_steps),
    }


def qwen_vl_prm_rank_dtr(
    candidates: list[Candidate],
    *,
    image: Image.Image,
    question: str,
    prm_model,
    prm_processor,
    prm_tokenizer,
    system_prompt: str | None = None,
    aggregation: str = "overall",
) -> SelectionResult:
    """Rank DTR candidates using segmented PRM scoring.

    `aggregation` controls which score to rank by:
      - "overall": mean P(+) across all (perception + reasoning) steps
      - "reasoning_mean": mean P(+) over reasoning steps only
      - "reasoning_min": min P(+) over reasoning steps (pessimistic)

    All three aggregations are always computed; `aggregation` selects which
    one drives the ranking. Per-candidate perception/reasoning attribution
    is preserved in metadata for error analysis.
    """
    if not candidates:
        return SelectionResult(
            answer=None, method="qwen_vl_prm_dtr", confidence="very_low",
            metadata={"reason": "no_candidates"},
        )
    valid = [c for c in candidates if c.answer is not None]
    if not valid:
        return SelectionResult(
            answer=None, method="qwen_vl_prm_dtr", confidence="very_low",
            metadata={"reason": "no_parseable_candidates"},
        )

    scored: list[tuple[Candidate, float, dict]] = []
    for c in valid:
        seg = qwen_vl_prm_score_dtr_segmented(
            prm_model=prm_model,
            prm_processor=prm_processor,
            prm_tokenizer=prm_tokenizer,
            image=image,
            question=question,
            description=c.description,
            reasoning=c.reasoning,
            system_prompt=system_prompt,
        )
        rank_score = seg.get(aggregation, seg["overall"])
        scored.append((c, rank_score, seg))

    scored.sort(key=lambda t: t[1], reverse=True)
    best_c, best_s, _best_seg = scored[0]

    if len(scored) > 1:
        gap = best_s - scored[1][1]
        confidence = "high" if gap > 0.2 else "medium" if gap > 0.1 else "low"
    else:
        confidence = "medium"

    vote_counts = Counter(c.answer for c in valid)

    return SelectionResult(
        answer=best_c.answer,
        method="qwen_vl_prm_dtr",
        confidence=confidence,
        vote_counts=dict(vote_counts),
        selected_candidate=best_c,
        metadata={
            "prm_scores": {c.uid: s for c, s, _ in scored},
            "segmented_scores": {c.uid: seg for c, _, seg in scored},
            "aggregation": aggregation,
            "top_score": best_s,
            "score_gap": (scored[0][1] - scored[1][1]) if len(scored) > 1 else None,
        },
    )


# --- PRM-decomposed verification (perception filter + reasoning rank) ---


def prm_decomposed_verify(
    candidates: list[Candidate],
    *,
    image: "Image.Image",
    question: str,
    prm_model,
    prm_processor,
    prm_tokenizer,
    system_prompt: str | None = None,
    perception_threshold: float = 0.3,
    reasoning_weight: float = 0.7,
) -> SelectionResult:
    """PRM verification with perception/reasoning decomposition.

    Unlike flat PRM ranking, this method:
    1. Scores all chains with DTR-segmented PRM (perception vs reasoning)
    2. Filters chains where perception_score < threshold (bad OCR)
    3. Ranks remaining by weighted combo of perception + reasoning scores
    4. Falls back to majority vote on filtered set if scores are flat

    This targets EXAMS-V's failure mode: bad OCR propagates into all reasoning
    chains derived from a single description, making reasoning scores
    unreliable. By gating on perception first, we discard chains built on
    incorrect visual extraction before they corrupt the vote.
    """
    if not candidates:
        return SelectionResult(
            answer=None, method="prm_decomposed", confidence="very_low",
            metadata={"reason": "no_candidates"},
        )
    valid = [c for c in candidates if c.answer is not None]
    if not valid:
        return SelectionResult(
            answer=None, method="prm_decomposed", confidence="very_low",
            metadata={"reason": "no_parseable_candidates"},
        )

    scored: list[tuple[Candidate, float, dict]] = []
    for c in valid:
        seg = qwen_vl_prm_score_dtr_segmented(
            prm_model=prm_model,
            prm_processor=prm_processor,
            prm_tokenizer=prm_tokenizer,
            image=image,
            question=question,
            description=c.description,
            reasoning=c.reasoning,
            system_prompt=system_prompt,
        )
        composite = (
            (1 - reasoning_weight) * seg["perception_score"]
            + reasoning_weight * seg["reasoning_mean"]
        )
        scored.append((c, composite, seg))

    # Filter by perception threshold
    before_filter = len(scored)
    filtered = [(c, s, seg) for c, s, seg in scored
                if seg["perception_score"] >= perception_threshold]

    if not filtered:
        logger.warning(
            "prm_decomposed: all {} chains below perception threshold {}, "
            "keeping all", before_filter, perception_threshold,
        )
        filtered = scored

    n_dropped = before_filter - len(filtered)
    if n_dropped > 0:
        logger.info(
            "prm_decomposed: filtered {}/{} chains with perception < {}",
            n_dropped, before_filter, perception_threshold,
        )

    # Rank by composite score
    filtered.sort(key=lambda t: t[1], reverse=True)
    best_c, best_s, best_seg = filtered[0]

    if len(filtered) > 1:
        gap = best_s - filtered[1][1]
        confidence = "high" if gap > 0.2 else "medium" if gap > 0.1 else "low"
    else:
        confidence = "medium"

    vote_counts = Counter(c.answer for c, _, _ in filtered)

    return SelectionResult(
        answer=best_c.answer,
        method="prm_decomposed",
        confidence=confidence,
        vote_counts=dict(vote_counts),
        selected_candidate=best_c,
        metadata={
            "composite_scores": {c.uid: s for c, s, _ in scored},
            "segmented_scores": {c.uid: seg for c, _, seg in scored},
            "perception_threshold": perception_threshold,
            "reasoning_weight": reasoning_weight,
            "n_filtered": n_dropped,
            "n_remaining": len(filtered),
            "top_score": best_s,
            "top_perception": best_seg["perception_score"],
            "top_reasoning_mean": best_seg["reasoning_mean"],
            "score_gap": (filtered[0][1] - filtered[1][1]) if len(filtered) > 1 else None,
        },
    )


# --- Top-level selection ---


def _dispatch(
    candidates: list[Candidate],
    method: str,
    model,
    processor,
    image: Image.Image | None,
    verify_config: dict | None,
) -> SelectionResult:
    if method == "majority_vote":
        return majority_vote(candidates)

    if method == "agentic":
        if model is None or processor is None or image is None:
            raise ValueError("agentic verification requires model, processor, and image")
        cfg = verify_config or {}
        return agentic_verify(
            model, processor, image, candidates,
            pre_filter_top_k=cfg.get("pre_filter_top_k", 8),
            skip_threshold=cfg.get("skip_threshold", 0.75),
            max_claims=cfg.get("max_claims", 5),
        )

    if method == "generative":
        if model is None or processor is None or image is None:
            raise ValueError("generative verification requires model, processor, and image")
        cfg = verify_config or {}
        return generative_critic_rank(
            candidates, model, processor, image,
            axes=cfg.get("axes"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 128),
        )

    if method == "qwen_vl_prm":
        if model is None or processor is None or image is None:
            raise ValueError(
                "qwen_vl_prm verification requires (prm_model, prm_processor, image); "
                "pass the loaded PRM tuple via `model`/`processor` and the original "
                "exam image via `image`."
            )
        cfg = verify_config or {}
        tokenizer = cfg.get("tokenizer") or getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "qwen_vl_prm verification requires verify_config['tokenizer'] or "
                "processor.tokenizer to be set"
            )
        question = cfg.get("question") or ""
        return qwen_vl_prm_rank(
            candidates,
            image=image,
            question=question,
            prm_model=model,
            prm_processor=processor,
            prm_tokenizer=tokenizer,
            system_prompt=cfg.get("system_prompt"),
            mode=cfg.get("mode", "step_mean"),
        )

    if method == "qwen_vl_prm_dtr":
        if model is None or processor is None or image is None:
            raise ValueError(
                "qwen_vl_prm_dtr verification requires (prm_model, prm_processor, image)"
            )
        cfg = verify_config or {}
        tokenizer = cfg.get("tokenizer") or getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "qwen_vl_prm_dtr verification requires verify_config['tokenizer'] or "
                "processor.tokenizer to be set"
            )
        question = cfg.get("question") or ""
        return qwen_vl_prm_rank_dtr(
            candidates,
            image=image,
            question=question,
            prm_model=model,
            prm_processor=processor,
            prm_tokenizer=tokenizer,
            system_prompt=cfg.get("system_prompt"),
            aggregation=cfg.get("aggregation", "overall"),
        )

    if method == "prm_decomposed":
        if model is None or processor is None or image is None:
            raise ValueError(
                "prm_decomposed verification requires (prm_model, prm_processor, image)"
            )
        cfg = verify_config or {}
        tokenizer = cfg.get("tokenizer") or getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "prm_decomposed verification requires verify_config['tokenizer']"
            )
        question = cfg.get("question") or ""
        return prm_decomposed_verify(
            candidates,
            image=image,
            question=question,
            prm_model=model,
            prm_processor=processor,
            prm_tokenizer=tokenizer,
            system_prompt=cfg.get("system_prompt"),
            perception_threshold=float(cfg.get("perception_threshold", 0.3)),
            reasoning_weight=float(cfg.get("reasoning_weight", 0.7)),
        )

    if method == "visualprm":
        logger.warning(
            "method='visualprm' is deprecated; use method='qwen_vl_prm' "
            "(ob11/Qwen-VL-PRM-7B). Falling back to majority vote.",
        )
        result = majority_vote(candidates)
        result.metadata["fallback_from"] = "visualprm"
        return result

    raise ValueError(f"Unknown verification method: {method}")


def select_answer(
    candidates: list[Candidate],
    method: str = "majority_vote",
    model=None,
    processor=None,
    image: Image.Image | None = None,
    verify_config: dict | None = None,
    *,
    ctx: "RunContext | None" = None,
) -> SelectionResult:
    """Select the best answer from candidates using the specified method.

    Args:
        candidates: All N*M candidates from the reason stage.
        method: "majority_vote" | "agentic" | "generative" | "visualprm".
        model: HF/vLLM model (required for agentic/generative/visualprm).
        processor: HF processor (required for agentic/generative/visualprm).
        image: Original exam image (required for agentic/generative/visualprm).
        verify_config: Verification-specific config dict.
        ctx: Optional `RunContext` for Phoenix span emission.
    """
    if ctx is None:
        return _dispatch(candidates, method, model, processor, image, verify_config)

    with ctx.stage_span("verify", method=method):
        child_name = f"verifier.{method}"
        # majority_vote is a pure CHAIN; the rest issue LLM calls.
        span_cm = (
            ctx.stage_span(child_name, method=method)
            if method == "majority_vote"
            else ctx.llm_span(child_name, method=method)
        )
        with span_cm as leaf:
            result = _dispatch(candidates, method, model, processor, image, verify_config)
            try:
                leaf.set_attribute("selected_answer", result.answer or "")
                leaf.set_attribute("confidence", result.confidence or "")
                leaf.set_attribute("cluster_sizes", str(dict(result.vote_counts or {})))
                tie_break = result.metadata.get("tiebreak") if result.metadata else None
                if tie_break:
                    leaf.set_attribute("tie_break", tie_break)
            except Exception:
                pass
            return result
