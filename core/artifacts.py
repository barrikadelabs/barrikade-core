"""Runtime artifact bootstrap for Barrikade SDK installs."""

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.settings import Settings

log = logging.getLogger(__name__)

DEFAULT_GCS_BUCKET = "barrikade-bundles"
GCS_MODELS_PREFIX = "models"
DEFAULT_BUNDLE_MANIFEST_OBJECT = f"{GCS_MODELS_PREFIX}/manifest.json"
_BUNDLE_CHECKED = False
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_PROGRESS_LOG_INTERVAL_S = 5.0
_PROGRESS_LOG_STEP_PCT = 5


class ArtifactDownloadError(RuntimeError):
    """Raised when the runtime artifact bundle cannot be prepared."""


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def _use_tqdm() -> bool:
    if os.getenv("TQDM_DISABLE", "0") == "1":
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


class _DownloadProgress:
    def __init__(self, label: str, total_bytes: int | None, logger: logging.Logger) -> None:
        self.label = label
        self.total_bytes = total_bytes
        self.logger = logger
        self.downloaded = 0
        self._last_log_time = time.time()
        self._last_pct = -1
        self._tqdm = None

        if _use_tqdm():
            try:
                from tqdm import tqdm
            except Exception:
                self._tqdm = None
            else:
                self._tqdm = tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=label,
                    leave=False,
                    file=sys.stderr,
                )
        if self._tqdm is None:
            if total_bytes:
                self.logger.info("Downloading %s (%s)", label, _format_bytes(total_bytes))
            else:
                self.logger.info("Downloading %s (size unknown)", label)

    def update(self, byte_count: int) -> None:
        if byte_count <= 0:
            return
        if self._tqdm is not None:
            self._tqdm.update(byte_count)
            return

        self.downloaded += byte_count
        now = time.time()
        if self.total_bytes:
            pct = int(self.downloaded / self.total_bytes * 100)
            if pct >= self._last_pct + _PROGRESS_LOG_STEP_PCT or now - self._last_log_time >= _PROGRESS_LOG_INTERVAL_S:
                self._last_pct = pct
                self._last_log_time = now
                self.logger.info(
                    "Downloading %s: %s/%s (%d%%)",
                    self.label,
                    _format_bytes(self.downloaded),
                    _format_bytes(self.total_bytes),
                    pct,
                )
        elif now - self._last_log_time >= _PROGRESS_LOG_INTERVAL_S:
            self._last_log_time = now
            self.logger.info(
                "Downloading %s: %s",
                self.label,
                _format_bytes(self.downloaded),
            )

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()


def _artifact_bucket(bucket_name: str | None = None) -> str:
    return bucket_name or os.getenv("BARRIKADA_GCS_BUCKET") or DEFAULT_GCS_BUCKET


def _bundle_manifest_object() -> str:
    return os.getenv("BARRIKADA_BUNDLE_MANIFEST_OBJECT") or DEFAULT_BUNDLE_MANIFEST_OBJECT


def _bundle_manifest_url(bucket_name: str | None = None) -> str:
    override = os.getenv("BARRIKADA_BUNDLE_MANIFEST_URL")
    if override:
        return override
    bucket = _artifact_bucket(bucket_name)
    manifest_object = _bundle_manifest_object()
    return f"https://storage.googleapis.com/{bucket}/{manifest_object}"


def _bundle_root(settings: Settings) -> Path:
    return Path(settings.bundle_root_dir)


def _bundle_manifest_path(settings: Settings) -> Path:
    return Path(settings.bundle_manifest_path)


def _artifact_targets(settings: Settings) -> list[tuple[str, Path]]:
    return [
        ("layer_b", Path(settings.layer_b_signatures_dirname)),
        ("layer_c", Path(settings.layer_c_model_pathname)),
        ("layer_d", Path(settings.layer_d_model_dirname)),
        ("layer_e", Path(settings.layer_e_model_dirname)),
    ]


def _layer_candidates(settings: Settings) -> dict[str, list[Path]]:
    return {
        "layer_b": settings.layer_b_signatures_candidates,
        "layer_c": settings.layer_c_model_candidates,
        "layer_d": settings.layer_d_model_candidates,
        "layer_e": settings.layer_e_model_candidates,
    }


def _missing_layers(settings: Settings) -> list[str]:
    missing = []
    for layer_name, candidates in _layer_candidates(settings).items():
        if not any(candidate.exists() for candidate in candidates):
            missing.append(layer_name)
    return missing


def artifacts_available(settings: Settings | None = None) -> bool:
    runtime_settings = settings or Settings()
    return not _missing_layers(runtime_settings)


def _load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactDownloadError(
            f"Manifest at {path} is not valid JSON."
        ) from exc


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _fetch_manifest(url: str) -> dict:
    response = _http_get(url)
    if response.status_code != 200:
        raise ArtifactDownloadError(
            f"Failed to fetch bundle manifest from {url}: "
            f"HTTP {response.status_code} {response.reason}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise ArtifactDownloadError(
            f"Bundle manifest from {url} is not valid JSON."
        ) from exc


def _bundle_version(manifest: dict | None) -> str | None:
    if not manifest:
        return None
    for key in ("bundle_version", "version"):
        value = manifest.get(key)
        if value:
            return str(value)
    return None


def _parse_version(value: str) -> tuple[int, ...] | None:
    if not value:
        return None
    parts = re.split(r"[.+-]", value)
    numbers: list[int] = []
    for part in parts:
        if not part.isdigit():
            break
        numbers.append(int(part))
    return tuple(numbers) if numbers else None


def _bundle_update_required(
    *,
    local_manifest: dict | None,
    remote_manifest: dict | None,
    missing_layers: list[str],
) -> bool:
    if missing_layers:
        return True
    remote_version = _bundle_version(remote_manifest)
    local_version = _bundle_version(local_manifest)
    if remote_version is None:
        raise ArtifactDownloadError("Remote bundle manifest is missing bundle_version.")
    if local_version is None:
        return True
    remote_parsed = _parse_version(remote_version)
    local_parsed = _parse_version(local_version)
    if remote_parsed and local_parsed:
        return remote_parsed != local_parsed
    return remote_version != local_version


def _list_gcs_layer_files(bucket_name: str, layer_name: str) -> list[str]:
    try:
        from google.auth.credentials import AnonymousCredentials
        from google.cloud import storage
    except ImportError as exc:
        raise ArtifactDownloadError(
            "Artifact download requires google-cloud-storage. "
            "Install the SDK with its default dependencies."
        ) from exc

    client = storage.Client(
        credentials=AnonymousCredentials(),
        project="anonymous-project",
    )
    prefix = f"{GCS_MODELS_PREFIX}/{layer_name}/"
    files = []
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/") or "/archives/" in blob.name:
            continue
        files.append(blob.name)
    return files


def _download_gcs_file(
    bucket_name: str,
    blob_name: str,
    local_path: Path,
    *,
    label: str | None = None,
) -> None:
    url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    try:
        _download_url_to_path(url, local_path, label=label or blob_name)
    except ArtifactDownloadError as exc:
        raise ArtifactDownloadError(
            f"Failed to download {blob_name} from bucket {bucket_name}: {exc}"
        ) from exc


def _download_url_to_path(url: str, local_path: Path, *, label: str | None = None) -> None:
    headers = {}
    existing_size = 0
    mode = "wb"
    
    if local_path.exists():
        existing_size = local_path.stat().st_size
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

    response = _http_get(url, stream=True, headers=headers)
    
    # If range is not satisfiable, it usually means we already have the whole file.
    if response.status_code == 416:
        log.info("Range not satisfiable for %s; assuming file is fully downloaded.", local_path.name)
        return

    # If the server ignores the Range header (returns 200 instead of 206), fallback to full write
    if response.status_code == 200:
        mode = "wb"
        existing_size = 0
    elif response.status_code != 206:
        # Standard error handling
        raise ArtifactDownloadError(
            f"Failed to download {url}: HTTP {response.status_code} {response.reason}"
        )

    total_bytes = None
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            total_bytes = int(content_length)
            if response.status_code == 206:
                total_bytes += existing_size
        except ValueError:
            total_bytes = None

    local_path.parent.mkdir(parents=True, exist_ok=True)
    progress = _DownloadProgress(label or local_path.name, total_bytes, log)
    
    if existing_size > 0 and response.status_code == 206:
        progress.update(existing_size)
        
    try:
        with local_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    finally:
        progress.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_files(manifest: dict) -> list[dict]:
    files = manifest.get("files")
    if isinstance(files, list):
        return [item if isinstance(item, dict) else {"path": item} for item in files]

    layers = manifest.get("layers")
    if isinstance(layers, dict):
        expanded: list[dict] = []
        for layer_name, layer_files in layers.items():
            if not isinstance(layer_files, list):
                continue
            for item in layer_files:
                entry = item if isinstance(item, dict) else {"path": item}
                path = entry.get("path") or entry.get("name")
                if path:
                    entry = dict(entry)
                    entry["path"] = f"{layer_name}/{path}"
                expanded.append(entry)
        return expanded

    return []


def _http_get(url: str, *, stream: bool = False, headers: dict | None = None):
    try:
        import requests
    except ImportError as exc:
        raise ArtifactDownloadError(
            "Artifact download requires requests. Install the SDK with its default dependencies."
        ) from exc

    return requests.get(url, timeout=60, stream=stream, headers=headers)


def _safe_relative_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ArtifactDownloadError(f"Manifest path is not safe: {path_value}")
    return candidate


def _manifest_base_url(manifest: dict, bucket_name: str | None) -> str:
    base_url = manifest.get("base_url")
    if base_url:
        return str(base_url).rstrip("/")
    bucket = _artifact_bucket(bucket_name)
    return f"https://storage.googleapis.com/{bucket}/{GCS_MODELS_PREFIX}"


def _manifest_prefix(manifest: dict) -> str:
    prefix = manifest.get("prefix") or manifest.get("bundle_prefix") or ""
    return str(prefix).strip("/")


def _create_staging_dir(target_dir: Path) -> Path:
    staging_dir = target_dir.parent / f".bundle-staging-{uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def _swap_bundle_dir(staging_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        backup_dir = target_dir.parent / f".bundle-old-{uuid4().hex}"
        target_dir.rename(backup_dir)
        staging_dir.rename(target_dir)
        shutil.rmtree(backup_dir, ignore_errors=True)
    else:
        staging_dir.rename(target_dir)


def _extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    import tarfile
    import zipfile

    dest_dir_resolved = dest_dir.resolve()

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            # Check for directory traversal members
            safe_members = []
            for member in archive.getmembers():
                member_path = Path(member.name)
                # Resolve the target path safely
                target_path = dest_dir_resolved.joinpath(member_path).resolve()
                if not target_path.is_relative_to(dest_dir_resolved):
                    continue
                safe_members.append(member)
            archive.extractall(dest_dir_resolved, members=safe_members)  # nosec B202
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            # Check for directory traversal members
            safe_members = []
            for member in archive.namelist():
                member_path = Path(member)
                target_path = dest_dir_resolved.joinpath(member_path).resolve()
                if not target_path.is_relative_to(dest_dir_resolved):
                    continue
                safe_members.append(member)
            archive.extractall(dest_dir_resolved, members=safe_members)  # nosec B202
    else:
        raise ArtifactDownloadError(f"Unsupported bundle archive format: {archive_path}")

    manifest_subdir = None
    if (dest_dir / "manifest.json").exists():
        manifest_subdir = dest_dir
    else:
        entries = [entry for entry in dest_dir.iterdir() if entry.is_dir()]
        if len(entries) == 1:
            manifest_subdir = entries[0]
    return manifest_subdir or dest_dir


def _layer_root(layer_name: str, settings: Settings) -> Path:
    return Path(settings.artifacts_root_dir) / layer_name


def _iter_download_targets(
    layer_name: str,
    files: Iterable[str],
    settings: Settings,
) -> Iterable[tuple[str, Path, Path]]:
    layer_root = _layer_root(layer_name, settings)
    for blob_name in files:
        parts = blob_name.split("/")
        relative_path = Path(*parts[2:])
        yield blob_name, layer_root / relative_path, relative_path


def download_runtime_artifacts(
    *,
    settings: Settings | None = None,
    bucket_name: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    runtime_settings = settings or Settings()
    resolved_bucket = _artifact_bucket(bucket_name)
    summary: dict[str, object] = {"bucket": resolved_bucket, "downloaded_layers": []}
    downloaded_layers: list[str] = []

    for layer_name in _layers_to_download(runtime_settings, force):
        files = _list_gcs_layer_files(resolved_bucket, layer_name)
        if not files:
            raise ArtifactDownloadError(
                f"No runtime artifacts found for {layer_name} in gs://{resolved_bucket}/{GCS_MODELS_PREFIX}/"
            )

        total_files = len(files)
        max_workers = runtime_settings.max_download_workers
        log.info(
            "Downloading Barrikade artifacts for %s (%d files) using %d workers",
            layer_name,
            total_files,
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for index, (blob_name, local_path, relative_path) in enumerate(
                _iter_download_targets(layer_name, files, runtime_settings),
                start=1,
            ):
                label = f"{layer_name}/{relative_path.as_posix()} ({index}/{total_files})"
                futures.append(
                    pool.submit(
                        _download_gcs_file,
                        resolved_bucket,
                        blob_name,
                        local_path,
                        label=label,
                    )
                )
            try:
                for future in as_completed(futures):
                    future.result()
            except Exception:
                for f in futures:
                    f.cancel()
                raise

        downloaded_layers.append(layer_name)
        log.info("Layer %s artifacts ready", layer_name)

    summary["downloaded_layers"] = downloaded_layers
    summary["artifacts_dir"] = runtime_settings.artifacts_root_dir
    return summary


def _layers_to_download(settings: Settings, force: bool) -> list[str]:
    if force:
        return sorted({name for name, _ in _artifact_targets(settings)})
    return sorted(set(_missing_layers(settings)))


def download_runtime_bundle(
    *,
    settings: Settings | None = None,
    bucket_name: str | None = None,
    manifest_url: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    runtime_settings = settings or Settings()
    resolved_bucket = _artifact_bucket(bucket_name)
    resolved_manifest_url = manifest_url or _bundle_manifest_url(resolved_bucket)
    summary: dict[str, object] = {
        "bucket": resolved_bucket,
        "manifest_url": resolved_manifest_url,
        "bundle_dir": runtime_settings.bundle_root_dir,
        "updated": False,
    }

    local_manifest_path = _bundle_manifest_path(runtime_settings)
    local_manifest = _load_manifest(local_manifest_path)
    missing_layers = _missing_layers(runtime_settings)

    try:
        log.info("Fetching runtime bundle manifest from %s", resolved_manifest_url)
        remote_manifest = _fetch_manifest(resolved_manifest_url)
    except ArtifactDownloadError as exc:
        if "HTTP 404" in str(exc):
            summary["manifest_missing"] = True
            if missing_layers or force:
                summary.update(
                    download_runtime_artifacts(
                        settings=runtime_settings,
                        bucket_name=resolved_bucket,
                        force=force,
                    )
                )
                summary["updated"] = True
            return summary
        if local_manifest and not missing_layers and not force:
            log.warning("Using existing bundle; failed to fetch remote manifest: %s", exc)
            return summary
        raise

    summary["bundle_version"] = _bundle_version(remote_manifest)

    if not force and not _bundle_update_required(
        local_manifest=local_manifest,
        remote_manifest=remote_manifest,
        missing_layers=missing_layers,
    ):
        if summary.get("bundle_version"):
            log.info("Runtime bundle up to date (version %s)", summary["bundle_version"])
        else:
            log.info("Runtime bundle up to date")
        return summary

    archive_url = remote_manifest.get("bundle_url") or remote_manifest.get("archive_url")
    files = _resolve_manifest_files(remote_manifest)
    target_dir = _bundle_root(runtime_settings)

    if archive_url:
        log.info("Downloading runtime bundle archive")
        staging_dir = _create_staging_dir(target_dir)
        try:
            archive_path = staging_dir / "bundle-archive"
            _download_url_to_path(str(archive_url), archive_path, label="bundle archive")
            content_root = _extract_archive(archive_path, staging_dir)
            if content_root != staging_dir:
                _swap_bundle_dir(content_root, target_dir)
                shutil.rmtree(staging_dir, ignore_errors=True)
            else:
                _swap_bundle_dir(staging_dir, target_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    elif files:
        log.info("Downloading runtime bundle files (%d files)", len(files))
        base_url = _manifest_base_url(remote_manifest, resolved_bucket)
        prefix = _manifest_prefix(remote_manifest)
        staging_dir = _create_staging_dir(target_dir)
        max_workers = runtime_settings.max_download_workers

        def download_and_verify(entry_info, idx):
            path_value = entry_info.get("path") or entry_info.get("name")
            if not path_value:
                return
            relative_path = _safe_relative_path(str(path_value))
            if prefix:
                remote_path = f"{prefix}/{relative_path.as_posix()}"
            else:
                remote_path = relative_path.as_posix()
            
            destination = staging_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            expected_sha = entry_info.get("sha256")
            
            # Fast-path: Check if existing file in target_dir is up-to-date and reuse it
            existing_target_file = target_dir / relative_path
            if existing_target_file.exists() and expected_sha:
                try:
                    actual_sha = _sha256_file(existing_target_file)
                    if actual_sha.lower() == str(expected_sha).lower():
                        log.info("Reusing up-to-date local file for %s", relative_path.as_posix())
                        shutil.copy2(existing_target_file, destination)
                        return
                except Exception as e:
                    log.debug("Failed to reuse existing file %s: %s", relative_path, e)
            
            url = entry_info.get("url") or f"{base_url}/{remote_path}"
            label = f"bundle/{relative_path.as_posix()} ({idx}/{len(files)})"
            _download_url_to_path(str(url), destination, label=label)
            if expected_sha:
                actual_sha = _sha256_file(destination)
                if actual_sha.lower() != str(expected_sha).lower():
                    raise ArtifactDownloadError(
                        f"Checksum mismatch for {relative_path}"
                    )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = []
                for index, entry in enumerate(files, start=1):
                    futures.append(pool.submit(download_and_verify, entry, index))
                try:
                    for future in as_completed(futures):
                        future.result()
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise

            _swap_bundle_dir(staging_dir, target_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    else:
        summary.update(
            download_runtime_artifacts(
                settings=runtime_settings,
                bucket_name=resolved_bucket,
                force=force,
            )
        )
        summary["updated"] = True
        _write_manifest(local_manifest_path, remote_manifest)
        return summary

    _write_manifest(local_manifest_path, remote_manifest)
    summary["updated"] = True
    log.info("Runtime bundle ready at %s", runtime_settings.bundle_root_dir)
    return summary


def ensure_runtime_artifacts(
    *,
    settings: Settings | None = None,
    bucket_name: str | None = None,
    manifest_url: str | None = None,
    auto_download: bool | None = None,
    force: bool = False,
) -> None:
    ensure_runtime_bundle(
        settings=settings,
        bucket_name=bucket_name,
        manifest_url=manifest_url,
        auto_download=auto_download,
        force=force,
    )


def ensure_runtime_bundle(
    *,
    settings: Settings | None = None,
    bucket_name: str | None = None,
    manifest_url: str | None = None,
    auto_download: bool | None = None,
    force: bool = False,
) -> None:
    global _BUNDLE_CHECKED

    runtime_settings = settings or Settings()
    if _BUNDLE_CHECKED and not force:
        log.debug("Runtime bundle already checked; skipping.")
        return

    log.info("Checking Barrikade runtime bundle in %s", runtime_settings.bundle_root_dir)

    missing_layers = _missing_layers(runtime_settings)
    local_manifest_path = _bundle_manifest_path(runtime_settings)
    local_manifest = _load_manifest(local_manifest_path)

    if auto_download is None:
        auto_download = os.getenv("BARRIKADA_AUTO_DOWNLOAD_ARTIFACTS", "1") != "0"

    if not auto_download:
        log.info("Auto-download disabled; verifying local runtime artifacts")
        if missing_layers:
            missing = ", ".join(sorted(missing_layers))
            raise ArtifactDownloadError(
                "Barrikade runtime artifacts are missing for "
                f"{missing}. Run `python -m barrikade download-artifacts` or set "
                "BARRIKADA_AUTO_DOWNLOAD_ARTIFACTS=1."
            )
        if local_manifest is None:
            raise ArtifactDownloadError(
                "Barrikade runtime manifest is missing. Run `python -m barrikade download-artifacts` "
                "or set BARRIKADA_AUTO_DOWNLOAD_ARTIFACTS=1."
            )
        _BUNDLE_CHECKED = True
        return

    try:
        download_runtime_bundle(
            settings=runtime_settings,
            bucket_name=bucket_name,
            manifest_url=manifest_url,
            force=force,
        )
    except Exception as exc:
        missing = ", ".join(sorted(missing_layers))
        raise ArtifactDownloadError(
            "Barrikade could not prepare runtime artifacts for "
            f"{missing or 'bundle'}. Run `python -m barrikade download-artifacts` to retry."
        ) from exc

    _BUNDLE_CHECKED = True
