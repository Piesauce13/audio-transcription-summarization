from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import logging
import os
import shutil
from typing import Callable

log = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
STRICT = os.getenv("STRICT_MLFLOW", "0") == "1"


@dataclass
class ModelSource:
    path: Path
    version: str
    tags: dict[str, str] = field(default_factory=dict)


def dir_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        file_digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                file_digest.update(chunk)
        digest.update(f"{path.relative_to(root).as_posix()}:{file_digest.hexdigest()}\n".encode())
    return digest.hexdigest()


def verify(path: Path, expected: str | None) -> None:
    if not expected:
        log.warning("no sha256 pinned for %s; serving it unverified", path.name)
        return
    actual = dir_sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path.name}: sha256 {actual} != pinned {expected}")


def from_mlflow(name: str, alias: str, cache_dir: Path) -> ModelSource | None:
    try:
        import mlflow
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=TRACKING_URI)
        model_version = client.get_model_version_by_alias(name, alias)
        version = str(model_version.version)
        destination = cache_dir / name / version
        weights_dir = model_version.tags["weights_dir"]
        path = destination / "artifacts" / weights_dir
        if not path.exists():
            destination.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(TRACKING_URI)
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{name}/{version}", dst_path=str(destination))
        try:
            verify(path, model_version.tags.get("sha256"))
        except RuntimeError:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        return ModelSource(path, f"mlflow:{name}/{version}", dict(model_version.tags))
    except Exception:
        if STRICT:
            raise
        log.exception("MLflow lookup failed for %s@%s; falling back to local/HF weights", name, alias)
        return None


def download_local(path: Path, download: Callable[[Path], None]) -> ModelSource:
    if not (path / "config.json").exists():
        log.info("downloading weights from Hugging Face into %s", path)
        path.mkdir(parents=True, exist_ok=True)
        download(path)
    return ModelSource(path, f"local:{path.name}")


def resolve(name: str, alias: str, fallback: Path, download: Callable[[Path], None],
            cache_dir: Path) -> ModelSource:
    return from_mlflow(name, alias, cache_dir) or download_local(fallback, download)
