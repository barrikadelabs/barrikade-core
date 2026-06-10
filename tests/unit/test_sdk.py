import json

import pytest

import barrikade
from barrikade.__main__ import main
from core.artifacts import (
    ArtifactDownloadError,
    download_runtime_artifacts,
    ensure_runtime_artifacts,
    _extract_archive,
    _download_url_to_path,
)
from core.orchestrator import PIPipeline as CorePIPipeline
from core.settings import Settings


def test_public_sdk_exports_pipeline():
    assert barrikade.PIPipeline is CorePIPipeline


def test_ensure_runtime_artifacts_errors_when_auto_download_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("BARRIKADA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BARRIKADA_CORE_MODELS_DIR", str(tmp_path / "core-models"))

    with pytest.raises(ArtifactDownloadError, match="download-artifacts"):
        ensure_runtime_artifacts(auto_download=False)


def test_download_runtime_artifacts_fetches_missing_layers(monkeypatch, tmp_path):
    monkeypatch.setenv("BARRIKADA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BARRIKADA_CORE_MODELS_DIR", str(tmp_path / "core-models"))

    settings = Settings()
    existing_layer_b = tmp_path / "core-models" / "layer_b" / "embeddings"
    existing_layer_b.mkdir(parents=True)
    (existing_layer_b / "metadata.json").write_text("{}")

    listed_layers = []
    downloaded_files = []

    def fake_list(bucket_name, layer_name):
        listed_layers.append((bucket_name, layer_name))
        return [f"models/{layer_name}/artifact.bin"]

    def fake_download(bucket_name, blob_name, local_path, label=None):
        downloaded_files.append((bucket_name, blob_name, str(local_path)))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("ok")

    monkeypatch.setattr("core.artifacts._missing_layers", lambda _: ["layer_c", "layer_d", "layer_e"])
    monkeypatch.setattr("core.artifacts._list_gcs_layer_files", fake_list)
    monkeypatch.setattr("core.artifacts._download_gcs_file", fake_download)

    summary = download_runtime_artifacts(settings=settings, bucket_name="test-bucket")

    assert summary["bucket"] == "test-bucket"
    assert summary["downloaded_layers"] == ["layer_c", "layer_d", "layer_e"]
    assert listed_layers == [
        ("test-bucket", "layer_c"),
        ("test-bucket", "layer_d"),
        ("test-bucket", "layer_e"),
    ]
    assert any(path.endswith("layer_c/artifact.bin") for _, _, path in downloaded_files)
    assert any(path.endswith("layer_d/artifact.bin") for _, _, path in downloaded_files)
    assert any(path.endswith("layer_e/artifact.bin") for _, _, path in downloaded_files)


def test_cli_download_artifacts_invokes_downloader(monkeypatch, capsys):
    monkeypatch.setattr(
        "barrikade.__main__.download_runtime_bundle",
        lambda bucket_name, manifest_url, force: {
            "bucket": bucket_name,
            "manifest_url": manifest_url,
            "force": force,
            "downloaded_layers": ["layer_c"],
        },
    )

    exit_code = main(
        [
            "download-artifacts",
            "--bucket",
            "sdk-bucket",
            "--manifest-url",
            "https://example.com/manifest.json",
            "--force",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["bucket"] == "sdk-bucket"
    assert output["manifest_url"] == "https://example.com/manifest.json"
    assert output["force"] is True


def test_public_sdk_exports_stateful_security():
    from core.session_orchestrator import (
        SessionOrchestrator as CoreSessionOrchestrator,
        create_session_orchestrator as core_create_session_orchestrator,
        SessionDetectResult as CoreSessionDetectResult,
    )
    from core.session_settings import SessionSettings as CoreSessionSettings
    from core.session import (
        SessionEvent as CoreSessionEvent,
        SessionEventType as CoreSessionEventType,
        SessionNotActiveError as CoreSessionNotActiveError,
        SessionStatus as CoreSessionStatus,
        WorkloadSession as CoreWorkloadSession,
        SessionStoreBackend as CoreSessionStoreBackend,
        InMemorySessionStore as CoreInMemorySessionStore,
    )
    from models.verdicts import (
        InputProvenance as CoreInputProvenance,
        Intervention as CoreIntervention,
    )
    from models.incident_report import IncidentReport as CoreIncidentReport

    assert barrikade.SessionOrchestrator is CoreSessionOrchestrator
    assert barrikade.create_session_orchestrator is core_create_session_orchestrator
    assert barrikade.SessionDetectResult is CoreSessionDetectResult
    assert barrikade.SessionSettings is CoreSessionSettings
    assert barrikade.SessionEvent is CoreSessionEvent
    assert barrikade.SessionEventType is CoreSessionEventType
    assert barrikade.SessionNotActiveError is CoreSessionNotActiveError
    assert barrikade.SessionStatus is CoreSessionStatus
    assert barrikade.WorkloadSession is CoreWorkloadSession
    assert barrikade.SessionStoreBackend is CoreSessionStoreBackend
    assert barrikade.InMemorySessionStore is CoreInMemorySessionStore
    assert barrikade.InputProvenance is CoreInputProvenance
    assert barrikade.Intervention is CoreIntervention
    assert barrikade.IncidentReport is CoreIncidentReport


def test_extract_archive(tmp_path):
    import tarfile
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    
    # Create a mock tar.gz archive
    archive_path = tmp_path / "test.tar.gz"
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("hello archive")
    
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(test_file, arcname="test_file.txt")
        
    extracted_root = _extract_archive(archive_path, dest_dir)
    assert extracted_root == dest_dir
    assert (dest_dir / "test_file.txt").exists()
    assert (dest_dir / "test_file.txt").read_text() == "hello archive"


def test_download_url_to_path_resumable(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    local_path = tmp_path / "downloaded.bin"
    
    # Create a partial local file
    local_path.write_text("part1")
    assert local_path.stat().st_size == 5

    requested_headers = []

    def mock_http_get(url, stream=True, headers=None):
        requested_headers.append(headers)
        mock_resp = MagicMock()
        mock_resp.status_code = 206
        mock_resp.headers = {"Content-Length": "5"}
        mock_resp.iter_content.return_value = [b"part2"]
        return mock_resp

    monkeypatch.setattr("core.artifacts._http_get", mock_http_get)

    _download_url_to_path("https://example.com/file.bin", local_path, label="test")

    # Assert correct headers were sent
    assert len(requested_headers) == 1
    assert requested_headers[0]["Range"] == "bytes=5-"
    
    # Assert file was correctly appended to
    assert local_path.read_text() == "part1part2"

