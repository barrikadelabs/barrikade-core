from __future__ import annotations

import threading

from core.layer_e.utils import StreamJudgeOutput
from core.orchestrator import PIPipeline


class _FakeStreamJudge:
    instances = 0

    def __init__(self, **kwargs):
        type(self).instances += 1
        self.kwargs = kwargs

    def verify_output(self, output_text, prompt_text=""):
        return StreamJudgeOutput(
            decision="block",
            risk_level="Unsafe",
            category="Illegal Acts",
            rationale="flagged",
            model="fake-stream",
            flagged_token_index=3,
            truncated=False,
            token_risk_levels=["Safe", "Safe", "Unsafe", "Unsafe"],
            token_categories=["Political", "Political", "Illegal Acts", "Illegal Acts"],
        )


def _bare_pipeline() -> PIPipeline:
    # Bypass __init__: verify_output must not depend on the detection layers.
    pipeline = PIPipeline.__new__(PIPipeline)
    pipeline._stream_judge = None
    pipeline._stream_judge_lock = threading.Lock()
    return pipeline


def test_verify_output_lazily_builds_judge_once(monkeypatch, tmp_path):
    monkeypatch.setenv("BARRIKADA_LAYER_E_STREAM_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("core.layer_e.stream_judge.Qwen3GuardStreamJudge", _FakeStreamJudge)
    _FakeStreamJudge.instances = 0

    pipeline = _bare_pipeline()
    first = pipeline.verify_output("an output", prompt_text="a prompt")
    second = pipeline.verify_output("another output")

    assert _FakeStreamJudge.instances == 1
    assert first.verdict == "block"
    assert first.risk_level == "Unsafe"
    assert first.category == "Illegal Acts"
    assert first.flagged_token_index == 3
    assert first.processing_time_ms >= 0
    assert second.verdict == "block"
    assert first.to_dict()["token_risk_levels"] == ["Safe", "Safe", "Unsafe", "Unsafe"]
    assert first.get_risk_score() == 100.0

    # Settings actually reach the judge constructor.
    judge_kwargs = pipeline._stream_judge.kwargs
    assert judge_kwargs["block_controversial"] is False
    assert judge_kwargs["debounce_tokens"] == 2
    assert judge_kwargs["max_seq_tokens"] == 8192


def test_verify_output_missing_artifacts_fail_loudly(monkeypatch):
    monkeypatch.delenv("BARRIKADA_LAYER_E_STREAM_MODEL_DIR", raising=False)
    monkeypatch.setenv("BARRIKADA_CORE_MODELS_DIR", "/nonexistent/models")
    monkeypatch.setenv("BARRIKADA_BUNDLE_DIR", "/nonexistent/bundle")
    monkeypatch.setenv("BARRIKADA_ARTIFACTS_DIR", "/nonexistent/artifacts")

    pipeline = _bare_pipeline()
    try:
        pipeline.verify_output("an output")
        raised = False
    except FileNotFoundError as exc:
        raised = True
        assert "Layer E stream model" in str(exc)
    assert raised
