from pathlib import Path
import hashlib
from urllib.request import urlretrieve
import argparse
import mlflow
from mlflow import MlflowClient

CATALOG = {
    "yolo26n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "yolo26s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"
}
ROOT = Path(__file__).resolve().parent.parent
# folder /mlflow-lab/models
MODELS = ROOT / "models"

def sha256(path: Path) -> str:
    digest = hashlib.sha256();
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

# yolo26s.pt -> saved to mlflow-lab/models/yolo26s.pt
def fetch(filename: str) -> Path:
    path = MODELS / filename
    if not path.exists():
        MODELS.mkdir(exist_ok=True)
        print(f"downloading {CATALOG[filename]}")
        urlretrieve(CATALOG[filename], path)
    return path

# Commandlineapplication to push the pretrain model to mlflow
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="registerd model name, e.g. yolo-detector")
    ap.add_argument("filename", choices=sorted(CATALOG))
    ap.add_argument("--alias", default="challenger")
    args = ap.parse_args()

    mlflow.set_tracking_uri("http://localhost:5050")
    mlflow.set_experiment("pretrained-yolo-detection")
    path = fetch(args.filename)
    digest = sha256(path)

    with mlflow.start_run(run_name=f"register-{args.filename}"):
        mlflow.set_tags({"pipeline": "registeration", "source_url": CATALOG[args.filename]})
        mlflow.log_params({"weights_file": args.filename, "size_mb": round(path.stat().st_size / 2**20, 1)})
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=str(ROOT / "scripts" / "weights_model.py"),
            artifacts={"weights": str(path)},
            registered_model_name=args.name
        )

        client = MlflowClient()
        version = str(info.registered_model_version)
        for key, value in {"sha256": digest, "weights_file": args.filename, "source_url": CATALOG[args.filename]}.items():
            client.set_model_version_tag(args.name, version, key, value)
        client.set_registered_model_alias(args.name, args.alias, version)
        print(f"{args.name} v{version} -> @{args.alias} sha256={digest[:12]}...")


if __name__ == "__main__":
    main()