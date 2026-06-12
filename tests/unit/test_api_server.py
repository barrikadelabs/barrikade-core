from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.server import app, state


class _FakePipeline:
    def detect(self, text):
        assert isinstance(text, str)
        return SimpleNamespace(
            final_verdict=SimpleNamespace(value="allow"),
            decision_layer=SimpleNamespace(value="layer_c"),
            confidence_score=0.91,
            total_processing_time_ms=4.2,
            to_dict=lambda: {
                "final_verdict": "allow",
                "decision_layer": "layer_c",
                "confidence_score": 0.91,
                "total_processing_time_ms": 4.2,
            },
        )

    def verify_output(self, output, prompt_text=""):
        assert isinstance(output, str)
        assert isinstance(prompt_text, str)
        return SimpleNamespace(
            verdict="block",
            risk_level="Unsafe",
            category="Illegal Acts",
            rationale="flagged",
            truncated=False,
            processing_time_ms=12.5,
            to_dict=lambda: {
                "verdict": "block",
                "risk_level": "Unsafe",
                "category": "Illegal Acts",
                "rationale": "flagged",
                "truncated": False,
                "processing_time_ms": 12.5,
                "token_risk_levels": ["Safe", "Unsafe", "Unsafe"],
                "token_categories": ["Political", "Illegal Acts", "Illegal Acts"],
                "flagged_token_index": 2,
                "model": "fake-stream",
            },
        )


class _MissingStreamModelPipeline(_FakePipeline):
    def verify_output(self, output, prompt_text=""):
        raise FileNotFoundError("Could not locate Layer E stream model directory")


def test_live_healthcheck():
    client = TestClient(app)
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_detect_success_without_diagnostics(monkeypatch):
    state.pipeline = _FakePipeline() #type: ignore
    state.startup_error = None

    client = TestClient(app)
    resp = client.post("/v1/detect", json={"text": "hello", "include_diagnostics": False})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["final_verdict"] == "allow"
    assert payload["decision_layer"] == "layer_c"
    assert payload["result"] is None


def test_detect_success_with_diagnostics(monkeypatch):
    state.pipeline = _FakePipeline() #type: ignore
    state.startup_error = None

    client = TestClient(app)
    resp = client.post("/v1/detect", json={"text": "hello", "include_diagnostics": True})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["result"]["decision_layer"] == "layer_c"


def test_detect_unavailable_pipeline_returns_503():
    state.pipeline = None
    state.startup_error = "Pipeline boot failed"

    client = TestClient(app)
    resp = client.post("/v1/detect", json={"text": "hello"})

    assert resp.status_code == 503
    assert "Pipeline boot failed" in resp.json()["detail"]


def test_verify_output_success_without_diagnostics():
    state.pipeline = _FakePipeline()  # type: ignore
    state.startup_error = None

    client = TestClient(app)
    resp = client.post("/v1/verify-output", json={"output": "some llm response"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["verdict"] == "block"
    assert payload["risk_level"] == "Unsafe"
    assert payload["category"] == "Illegal Acts"
    assert payload["truncated"] is False
    assert payload["result"] is None


def test_verify_output_with_diagnostics():
    state.pipeline = _FakePipeline()  # type: ignore
    state.startup_error = None

    client = TestClient(app)
    resp = client.post(
        "/v1/verify-output",
        json={"output": "some llm response", "prompt": "a prompt", "include_diagnostics": True},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["result"]["token_risk_levels"] == ["Safe", "Unsafe", "Unsafe"]
    assert payload["result"]["flagged_token_index"] == 2


def test_verify_output_missing_artifacts_returns_503():
    state.pipeline = _MissingStreamModelPipeline()  # type: ignore
    state.startup_error = None

    client = TestClient(app)
    resp = client.post("/v1/verify-output", json={"output": "some llm response"})

    assert resp.status_code == 503
    assert "Layer E stream model" in resp.json()["detail"]


def test_verify_output_unavailable_pipeline_returns_503():
    state.pipeline = None
    state.startup_error = "Pipeline boot failed"

    client = TestClient(app)
    resp = client.post("/v1/verify-output", json={"output": "some llm response"})

    assert resp.status_code == 503
    assert "Pipeline boot failed" in resp.json()["detail"]


def test_ready_local_teacher_mode(monkeypatch):
    state.pipeline = _FakePipeline() #type: ignore
    state.startup_error = None

    monkeypatch.setattr(
        "api.server.Settings",
        lambda: SimpleNamespace(layer_e_judge_mode="qwen3guard"),
    )

    client = TestClient(app)
    resp = client.get("/health/ready")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ready"
    assert payload["pipeline_initialized"] is True
    assert "qwen3guard" in payload["details"].lower()
