from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from core.layer_e.stream_judge import (
    Qwen3GuardStreamJudge,
    first_flagged_index,
    worst_risk_level,
)


# Token layout used by the fakes: [im_start, user, X, im_end, r0, r1, r2, r3]
# -> response span is the last four positions.
_IM_START, _USER, _IM_END = 1, 2, 3
_TOKEN_IDS = [_IM_START, _USER, 10, _IM_END, 20, 21, 22, 23]


class _FakeTokenizer:
    """Renders to a marker string so __call__ can tell the two renders apart."""

    def __init__(self, ids, promptless_ids=None):
        self.ids = ids
        self.promptless_ids = promptless_ids if promptless_ids is not None else ids
        self.render_count = 0

    def apply_chat_template(self, messages, **kwargs):
        self.render_count += 1
        return "with-prompt" if messages[0]["content"] else "promptless"

    def __call__(self, text, return_tensors="pt"):
        ids = self.ids if text == "with-prompt" else self.promptless_ids
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": _IM_START, "user": _USER, "<|im_end|>": _IM_END}[token]


class _FakeModel:
    """Per-position verdicts sized to the tokens actually fed, so a regression
    that feeds the wrong tensor (unsliced, untruncated) changes the output."""

    def __init__(self, risk_by_position=None, category_by_position=None, error=None, result=None):
        self.risk_by_position = risk_by_position or {}
        self.category_by_position = category_by_position or {}
        self.error = error
        self.result = result
        self.closed = False
        self.fed_lengths = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def stream_moderate_from_ids(self, token_ids, role, stream_state=None):
        assert role == "assistant"
        if self.error is not None:
            raise self.error
        self.fed_lengths.append(len(token_ids))
        if self.result is not None:
            return self.result, "state"
        n = len(token_ids)
        result = {
            "risk_level": [self.risk_by_position.get(i, "Safe") for i in range(n)],
            "risk_prob": [1.0] * n,
            "category": [self.category_by_position.get(i, "Political") for i in range(n)],
            "category_prob": [1.0] * n,
        }
        return result, "state"

    def close_stream(self, stream_state):
        self.closed = True


def _make_judge(monkeypatch, fake_model, tokenizer=None, **kwargs):
    tokenizer = tokenizer or _FakeTokenizer(_TOKEN_IDS)
    monkeypatch.setattr("core.layer_e.stream_judge.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "core.layer_e.stream_judge.AutoTokenizer.from_pretrained",
        lambda *args, **kw: tokenizer,
    )
    monkeypatch.setattr(
        "core.layer_e.stream_judge.AutoModel.from_pretrained",
        lambda *args, **kw: fake_model,
    )
    return Qwen3GuardStreamJudge(model_dir="/tmp/fake-stream", model_name="fake-stream", **kwargs)


def test_first_flagged_index_requires_consecutive_run():
    levels = ["Safe", "Unsafe", "Safe", "Unsafe", "Unsafe"]
    assert first_flagged_index(levels, {"Unsafe"}, 2) == 4
    assert first_flagged_index(levels, {"Unsafe"}, 1) == 1
    assert first_flagged_index(["Safe", "Safe"], {"Unsafe"}, 2) is None


def test_first_flagged_index_mixed_blocking_set():
    levels = ["Controversial", "Unsafe", "Safe"]
    assert first_flagged_index(levels, {"Unsafe", "Controversial"}, 2) == 1
    assert first_flagged_index(levels, {"Unsafe"}, 2) is None


def test_worst_risk_level_ordering():
    assert worst_risk_level([]) == "Safe"
    assert worst_risk_level(["Safe", "Controversial", "Safe"]) == "Controversial"
    assert worst_risk_level(["Controversial", "Unsafe"]) == "Unsafe"


def test_blocks_on_consecutive_unsafe_tokens(monkeypatch):
    # Response slice (positions 4-7): [Safe, Unsafe, Unsafe, Safe] -> flagged at 2.
    model = _FakeModel(
        risk_by_position={5: "Unsafe", 6: "Unsafe"},
        category_by_position={5: "Violent", 6: "Violent"},
    )
    judge = _make_judge(monkeypatch, model)
    out = judge.verify_output("some output", prompt_text="some prompt")

    assert out.decision == "block"
    assert out.risk_level == "Unsafe"
    assert out.category == "Violent"
    assert out.flagged_token_index == 2
    assert out.truncated is False
    assert out.token_risk_levels == ["Safe", "Unsafe", "Unsafe", "Safe"]
    assert "violent" in out.rationale.lower()
    assert model.closed


def test_single_unsafe_token_is_debounced(monkeypatch):
    model = _FakeModel(risk_by_position={5: "Unsafe"})
    judge = _make_judge(monkeypatch, model)
    out = judge.verify_output("some output")

    assert out.decision == "allow"
    assert out.risk_level == "Unsafe"  # worst observed is still reported
    assert out.category is None
    assert out.flagged_token_index is None
    assert "below the blocking policy" in out.rationale


def test_controversial_blocks_only_when_enabled(monkeypatch):
    risk = {4: "Controversial", 5: "Controversial"}

    lenient = _make_judge(monkeypatch, _FakeModel(risk_by_position=risk))
    assert lenient.verify_output("some output").decision == "allow"

    strict = _make_judge(
        monkeypatch, _FakeModel(risk_by_position=risk), block_controversial=True
    )
    out = strict.verify_output("some output")
    assert out.decision == "block"
    assert out.flagged_token_index == 1
    assert out.category == "Political"


def test_all_safe_allows(monkeypatch):
    model = _FakeModel()
    judge = _make_judge(monkeypatch, model)
    out = judge.verify_output("some output")

    assert out.decision == "allow"
    assert out.risk_level == "Safe"
    assert out.category is None
    assert "safe" in out.rationale.lower()


def test_within_limit_keeps_prompt_context_and_is_not_truncated(monkeypatch):
    tokenizer = _FakeTokenizer(_TOKEN_IDS)
    model = _FakeModel()
    judge = _make_judge(monkeypatch, model, tokenizer=tokenizer)
    out = judge.verify_output("some output", prompt_text="some prompt")

    assert out.truncated is False
    assert tokenizer.render_count == 1  # never re-rendered without the prompt
    assert model.fed_lengths == [len(_TOKEN_IDS)]
    assert len(out.token_risk_levels) == 4


def test_over_limit_drops_prompt_then_truncates_response_tail(monkeypatch):
    tokenizer = _FakeTokenizer(_TOKEN_IDS)
    model = _FakeModel()
    judge = _make_judge(monkeypatch, model, tokenizer=tokenizer, max_seq_tokens=6)
    out = judge.verify_output("some output", prompt_text="some prompt")

    assert out.truncated is True
    assert tokenizer.render_count == 2  # re-rendered without the prompt
    assert model.fed_lengths == [6]  # sliced to the cap
    # Response span within the 6 scored positions: indices 4-5.
    assert len(out.token_risk_levels) == 2


def test_chatml_markers_in_output_cannot_shrink_the_scored_span(monkeypatch):
    # An output embedding "<|im_start|>user ... <|im_end|>" must not move the
    # response boundary later (a verification bypass); the real user span is
    # always the first one. Layout: real span ends at 3; fake span at 5-8.
    ids = [_IM_START, _USER, 10, _IM_END, 20, _IM_START, _USER, 30, _IM_END, 21, 22]
    tokenizer = _FakeTokenizer(ids)
    model = _FakeModel(
        risk_by_position={5: "Unsafe", 6: "Unsafe"},
        category_by_position={5: "Unethical", 6: "Unethical"},
    )
    judge = _make_judge(monkeypatch, model, tokenizer=tokenizer)
    out = judge.verify_output("malicious output")

    assert len(out.token_risk_levels) == 7  # full span after the REAL user turn
    assert out.decision == "block"
    assert out.category == "Unethical"


def test_template_without_user_span_raises_value_error(monkeypatch):
    tokenizer = _FakeTokenizer([7, 8, 9])  # no <|im_start|>user pair at all
    judge = _make_judge(monkeypatch, _FakeModel(), tokenizer=tokenizer)

    with pytest.raises(ValueError, match="user span"):
        judge.verify_output("some output")


def test_malformed_model_result_fails_closed(monkeypatch):
    model = _FakeModel(result={"unexpected": []})
    judge = _make_judge(monkeypatch, model)
    out = judge.verify_output("some output")

    assert out.decision == "block"
    assert "Fallback block" in out.rationale


def test_inference_failure_fails_closed(monkeypatch):
    model = _FakeModel(error=RuntimeError("boom"))
    judge = _make_judge(monkeypatch, model)
    out = judge.verify_output("some output")

    assert out.decision == "block"
    assert "Fallback block" in out.rationale
    assert out.token_risk_levels == []
