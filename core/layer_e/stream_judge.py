"""Qwen3Guard-Stream judge for verifying LLM outputs (issue #27).

Unlike the Layer E Gen judge (which classifies *inputs* by generating a
structured completion), this judge scores an assistant response with the
Qwen3Guard-Stream token-level classification heads. The whole conversation
is scored in a single forward pass: per-position verdicts from the initial
``stream_moderate_from_ids`` call are identical to feeding tokens one at a
time (causal attention guarantees it), at a fraction of the cost — the
reference implementation runs with ``use_cache: false`` and recomputes the
full sequence on every incremental call (~230 ms/token on CPU, vs one
~1.4 s pass for a 457-token conversation).

Response-level aggregation follows the Qwen3Guard technical report
(arXiv:2510.14276): the response is flagged at token ``i`` only when
``debounce_tokens`` consecutive positions (default 2, the report's rule)
carry a blocking risk level, and token ``i``'s category becomes the
response category. ``Controversial`` positions only count as blocking when
``block_controversial`` is set.

Like the Gen judge, inference failures fail closed (``decision="block"``).
Missing model artifacts fail loudly instead: the ``FileNotFoundError`` from
settings path resolution propagates to the caller.
"""

import logging
from typing import Any, cast

import torch
from transformers import AutoModel, AutoTokenizer

from .utils import StreamJudgeOutput


log = logging.getLogger(__name__)

RISK_LEVEL_ORDER = {"Safe": 0, "Controversial": 1, "Unsafe": 2}


def first_flagged_index(levels, blocking, debounce_tokens):
    """Index of the first token where `debounce_tokens` consecutive positions
    are in `blocking`, or None."""
    run = 0
    for index, level in enumerate(levels):
        run = run + 1 if level in blocking else 0
        if run >= debounce_tokens:
            return index
    return None


def worst_risk_level(levels):
    if not levels:
        return "Safe"
    return max(levels, key=lambda level: RISK_LEVEL_ORDER.get(level, 0))


class Qwen3GuardStreamJudge:
    def __init__(
        self,
        model_dir,
        model_name=None,
        block_controversial=False,
        debounce_tokens=2,
        max_seq_tokens=8192,
    ):
        self.model_dir = str(model_dir)
        self.model_name = str(model_name or model_dir)
        self.block_controversial = bool(block_controversial)
        self.debounce_tokens = max(int(debounce_tokens), 1)
        self.max_seq_tokens = int(max_seq_tokens)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # trust_remote_code is mandatory: Qwen3ForGuardModel and its streaming
        # API only exist in the checkpoint's bundled code, not in transformers.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_dir, dtype=self.dtype, trust_remote_code=True)
        self.model = cast(Any, model).to(self.device).eval()

    def _render(self, prompt_text, output_text):
        """Render the chat template; return (token_ids, response_start_index)."""
        tokenizer = cast(Any, self.tokenizer)
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": output_text},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        token_ids = tokenizer(text, return_tensors="pt").input_ids[0]
        ids = token_ids.tolist()
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_token = tokenizer.convert_tokens_to_ids("user")
        try:
            # First user span, searched forward: ChatML markers embedded in the
            # (attacker-controlled) prompt or output can then only move the
            # boundary earlier — scanning more tokens — never later, which
            # would let an output hide content from verification.
            user_start = next(
                i for i in range(len(ids) - 1) if ids[i : i + 2] == [im_start, user_token]
            )
            user_end = next(i for i in range(user_start + 2, len(ids)) if ids[i] == im_end)
        except StopIteration:
            raise ValueError(
                "Could not locate the user span in the rendered chat template; "
                f"unexpected template for tokenizer at {self.model_dir}"
            ) from None
        return token_ids, user_end + 1

    def _score_tokens(self, token_ids):
        """One forward pass over the full sequence; per-position response-head verdicts."""
        model = cast(Any, self.model)
        with torch.inference_mode():
            result, stream_state = model.stream_moderate_from_ids(
                token_ids.to(self.device), role="assistant", stream_state=None
            )
            model.close_stream(stream_state)
        return result

    def verify_output(self, output_text, prompt_text=""):
        """Score an output; `flagged_token_index` indexes the scored response
        span, which includes the chat template's assistant framing tokens."""
        token_ids, response_start = self._render(prompt_text, output_text)
        truncated = False
        if len(token_ids) > self.max_seq_tokens:
            # The response span is what must be covered; drop the prompt
            # context first, and only then truncate the response tail.
            token_ids, response_start = self._render("", output_text)
            if len(token_ids) > self.max_seq_tokens:
                token_ids = token_ids[: self.max_seq_tokens]
            truncated = True

        # Everything from scoring through verdict construction fails closed:
        # a malformed result must block, not surface as a server error.
        try:
            result = self._score_tokens(token_ids)
            token_risk_levels = list(result["risk_level"][response_start:])
            token_categories = list(result["category"][response_start:])

            blocking = {"Unsafe", "Controversial"} if self.block_controversial else {"Unsafe"}
            flagged_index = first_flagged_index(token_risk_levels, blocking, self.debounce_tokens)
            observed = worst_risk_level(token_risk_levels)

            if flagged_index is not None:
                decision = "block"
                category = token_categories[flagged_index]
                rationale = (
                    f"Qwen3Guard-Stream flagged the output as "
                    f"{token_risk_levels[flagged_index].lower()} ({category}) "
                    f"from response token {flagged_index}"
                )
            else:
                decision = "allow"
                category = None
                if observed == "Safe":
                    rationale = "Qwen3Guard-Stream classified the output as safe"
                else:
                    rationale = (
                        f"Qwen3Guard-Stream observed {observed.lower()} tokens "
                        f"below the blocking policy; output allowed"
                    )

            return StreamJudgeOutput(
                decision=decision,
                risk_level=observed,
                category=category,
                rationale=rationale,
                model=self.model_name,
                flagged_token_index=flagged_index,
                truncated=truncated,
                token_risk_levels=token_risk_levels,
                token_categories=token_categories,
            )
        except (
            ValueError,
            RuntimeError,
            TypeError,
            KeyError,
            IndexError,
            torch.cuda.OutOfMemoryError,
        ) as exc:
            log.exception("Qwen3Guard-Stream scoring failed; failing closed")
            return StreamJudgeOutput(
                decision="block",
                risk_level="Unsafe",
                category=None,
                rationale=f"Fallback block after stream judge failure: {str(exc)[:140]}",
                model=self.model_name,
                flagged_token_index=None,
                truncated=truncated,
                token_risk_levels=[],
                token_categories=[],
            )
