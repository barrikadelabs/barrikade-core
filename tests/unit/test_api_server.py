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
