from pathlib import Path
import argparse
import hashlib
import os
import shutil

import mlflow
from mlflow import MlflowClient

CATALOG = {
    "whisper-tiny": {"repo": "openai/whisper-tiny", "task": "automatic-speech-recognition"},
    "whisper-base": {"repo": "openai/whisper-base", "task": "automatic-speech-recognition"},
    "distilbart-cnn-12-6": {"repo": "sshleifer/distilbart-cnn-12-6", "task": "summarization"},
    "distilbart-cnn-6-6": {"repo": "sshleifer/distilbart-cnn-6-6", "task": "summarization"},
}
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
SKIP = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "*.tflite", "*.mlmodel", "*.gguf"]


def dir_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        file_digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                file_digest.update(chunk)
        digest.update(f"{path.relative_to(root).as_posix()}:{file_digest.hexdigest()}\n".encode())
    return digest.hexdigest()


def fetch(key: str) -> Path:
    dest = MODELS / key
    if not (dest / "config.json").exists():
        from huggingface_hub import list_repo_files, snapshot_download

        repo = CATALOG[key]["repo"]
        ignore = list(SKIP)
        if any(filename.endswith(".safetensors") for filename in list_repo_files(repo)):
            ignore.append("*.bin")
        print(f"downloading {repo} -> {dest}")
        snapshot_download(repo, local_dir=str(dest), ignore_patterns=ignore)
        shutil.rmtree(dest / ".cache", ignore_errors=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="registered model name, e.g. whisper-asr")
    parser.add_argument("key", choices=sorted(CATALOG))
    parser.add_argument("--alias", default="challenger")
    args = parser.parse_args()
    cfg = CATALOG[args.key]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("pretrained-meeting-notes")
    path = fetch(args.key)
    digest = dir_sha256(path)
    size_mb = round(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 2**20, 1)

    with mlflow.start_run(run_name=f"register-{args.key}"):
        mlflow.set_tags({"pipeline": "registration", "source_repo": cfg["repo"]})
        mlflow.log_params({"weights_dir": args.key, "size_mb": size_mb})
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=str(ROOT / "scripts" / "weights_model.py"),
            artifacts={"weights": str(path)},
            registered_model_name=args.name,
            pip_requirements=[f"mlflow=={mlflow.__version__}"],
        )
        client = MlflowClient()
        version = str(info.registered_model_version)
        for key, value in {"sha256": digest, "weights_dir": args.key,
                           "hf_repo": cfg["repo"], "task": cfg["task"]}.items():
            client.set_model_version_tag(args.name, version, key, value)
        client.set_registered_model_alias(args.name, args.alias, version)
        print(f"{args.name} v{version} -> @{args.alias}  sha256={digest[:12]}...")


if __name__ == "__main__":
    main()
