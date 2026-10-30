from dataclasses import dataclass
from pathlib import Path
import hashlib
import logging
from typing import Callable

log = logging.getLogger(__name__)

@dataclass
class ModelSource:
    path: Path
    version: str

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify(path: Path, expected: str | None) -> None:
    if not expected:
        log.warning("no sha256 pinned for %s; serving it unverified", path.name)
        return
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path.name}: sha256 {actual} != pinned {expected}")

def from_mlflow(name: str, alias: str, cache_dir: Path) -> ModelSource | None:
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri("http://localhost:5050")
        client = MlflowClient()
        mv = client.get_model_version_by_alias(name, alias)
        version = str(mv.version)
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{name}/{version}",
            dst_path=str(cache_dir / name / version)
        )
        path = Path(local) / "artifacts" / mv.tags["weights_file"]
        verify(path, mv.tags.get("sha256"))
        return ModelSource(path, f"mflow:{name}/{version}")
    except Exception as ex:
        log.warning("MLFlow lookup failed will use the model from untralytic github weight")
        return None

def download_local(path: Path, download: Callable[[Path], None]) -> ModelSource:
    if not path.exists():
        log.info("download new weight from github: %s", path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        download(path)
    return ModelSource(path, f"local:{path.name}")

def resolve(name: str, alias: str, fallback: Path, download: Callable[[Path], None]) -> ModelSource:
    cache = fallback.parent / "cache"
    return from_mlflow(name, alias, cache) or download_local(fallback, download)