from __future__ import annotations

from types import SimpleNamespace

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
    def apply_chat_template(self, messages, **kwargs):
        return "rendered"

    def __call__(self, text, return_tensors="pt"):
        return SimpleNamespace(input_ids=torch.tensor([_TOKEN_IDS]))

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": _IM_START, "user": _USER, "<|im_end|>": _IM_END}[token]


class _FakeModel:
    def __init__(self, risk_levels, categories, error=None):
        self.risk_levels = risk_levels
        self.categories = categories
        self.error = error
        self.closed = False

    def to(self, device):
        return self

    def eval(self):
        return self

    def stream_moderate_from_ids(self, token_ids, role, stream_state=None):
        assert role == "assistant"
        if self.error is not None:
            raise self.error
        result = {
            "risk_level": self.risk_levels,
            "risk_prob": [1.0] * len(self.risk_levels),
            "category": self.categories,
            "category_prob": [1.0] * len(self.categories),
        }
        return result, "state"

    def close_stream(self, stream_state):
        self.closed = True


def _make_judge(monkeypatch, fake_model, **kwargs):
    monkeypatch.setattr("core.layer_e.stream_judge.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "core.layer_e.stream_judge.AutoTokenizer.from_pretrained",
        lambda *args, **kw: _FakeTokenizer(),
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
    # Response slice: [Safe, Unsafe, Unsafe, Safe] -> flagged at index 2.
    model = _FakeModel(
        risk_levels=["Safe"] * 4 + ["Safe", "Unsafe", "Unsafe", "Safe"],
        categories=["Political"] * 4 + ["Political", "Violent", "Violent", "Political"],
    )
    judge = _make_judge(monkeypatch, model)
    out = judge.verify_output("some output", prompt_text="some prompt")

    assert out.decision == "block"
    assert out.risk_level == "Unsafe"
    assert out.category == "Violent"
    assert out.flagged_token_index == 2
    assert out.token_risk_levels == ["Safe", "Unsafe", "Unsafe", "Safe"]
    assert "violent" in out.rationale.lower()
    assert model.closed


def test_single_unsafe_token_is_debounced(monkeypatch):
    model = _FakeModel(
        risk_levels=["Safe"] * 4 + ["Safe", "Unsafe", "Safe", "Safe"],
        categories=["Political"] * 8,
    )
    judge = _make_judge(monkeypatch, model)
    out = judge.verify_output("some output")

    assert out.decision == "allow"
    assert out.risk_level == "Unsafe"  # worst observed is still reported
    assert out.category is None
    assert out.flagged_token_index is None
    assert "below the blocking policy" in out.rationale


def test_controversial_blocks_only_when_enabled(monkeypatch):
    risk = ["Safe"] * 4 + ["Controversial", "Controversial", "Safe", "Safe"]
    cats = ["Political"] * 8

    lenient = _make_judge(monkeypatch, _FakeModel(risk, cats))
    assert lenient.verify_output("some output").decision == "allow"

    strict = _make_judge(monkeypatch, _FakeModel(risk, cats), block_controversial=True)
    out = strict.verify_output("some output")
    assert out.decision == "block"
    assert out.flagged_token_index == 1
    assert out.category == "Political"


def test_all_safe_allows(monkeypatch):
    model = _FakeModel(risk_levels=["Safe"] * 8, categories=["Political"] * 8)
    judge = _make_judge(monkeypatch, model)
    out = judge.verify_output("some output")

    assert out.decision == "allow"
    assert out.risk_level == "Safe"
    assert out.category is None
    assert "safe" in out.rationale.lower()


def test_inference_failure_fails_closed(monkeypatch):
    model = _FakeModel([], [], error=RuntimeError("boom"))
    judge = _make_judge(monkeypatch, model)
    out = judge.verify_output("some output")

    assert out.decision == "block"
    assert "Fallback block" in out.rationale
    assert out.token_risk_levels == []
