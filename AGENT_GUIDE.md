# Meeting Notes AI — Implementation Guide for the Coding Agent

**Goal:** Adapt our class MLflow template (YOLO) into a working prototype: upload/record audio → **Whisper** transcribes → **DistilBART** summarizes. Both models are downloaded from Hugging Face, **registered in an MLflow server running in Docker**, and the **API loads them back from MLflow** (by alias) when it starts. A simple web page is the frontend.

Deadline is tomorrow's presentation. **Prefer simple and working over clever.** Keep the template's structure and naming wherever possible: the professor already knows that pattern.

---

## 0. Rules for the agent

1. **Read the template first** (`docker-compose.yaml`, `app/main.py`, `app/model_source.py`, `scripts/register_pretrained.py`, `scripts/weights_model.py`, `scripts/requirements.txt`). Make minimal, explainable changes to it.
2. Do not commit model weights. Add `models/`, `__pycache__/`, `*.pyc` to `.gitignore`.
3. Keep `mlflow` client version identical to the server image (`3.17.2`) everywhere.
4. Dockerfiles and shell scripts must use **LF** line endings (the template files are CRLF because teammates are on Windows). Add a `.gitattributes` with `* text=auto eol=lf`.
5. After each phase, run the checks in section 7. Do not move on while one fails.
6. If something here contradicts what you observe, trust what you observe and tell the user.

---

## 1. How the template works (so you know what to preserve)

| Piece | Template (YOLO) | What it does |
|---|---|---|
| `docker-compose.yaml` | one `mlflow` service, image `ghcr.io/mlflow/mlflow:v3.17.2`, SQLite backend, `--serve-artifacts`, host port **5050** → 5000, `--allowed-hosts` list | MLflow server + UI. Artifacts are uploaded/downloaded **over HTTP through the server** (no shared file path needed) |
| `scripts/register_pretrained.py` | downloads weights, logs them with `mlflow.pyfunc.log_model(artifacts={"weights": path})` using a dummy `WeightOnly` PythonModel, sets tags (`sha256`, `weights_file`) and an **alias** | "Registration" = storing the raw weights in the MLflow model registry |
| `scripts/weights_model.py` | dummy PythonModel calling `mlflow.models.set_model(...)` | Just a placeholder so MLflow accepts the log call |
| `app/model_source.py` | `resolve(name, alias, fallback, download)`: ask MLflow for `models:/name/<version of alias>`, download, verify sha256; **if MLflow fails, fall back to downloading from the internet** | The "load it back from MLflow" step |
| `app/main.py` | FastAPI `lifespan` loads the model once into `_state` under a `_lock` | Model is loaded at startup, can be swapped later |

The template runs the API on the host. **We will put the API in Docker too**, because Whisper needs `ffmpeg` and that is painful to install on Windows.

---

## 2. What changes vs. the template

| Area | Template | Ours |
|---|---|---|
| Models | one YOLO `.pt` file | two **directories** from Hugging Face (`openai/whisper-base`, `sshleifer/distilbart-cnn-12-6`) |
| Registered names | `yolo-detector` | `whisper-asr`, `summarizer-distilbart` |
| Alias | register uses `challenger`, API loads `champion` | same; register first model with `--alias champion` so the API finds it |
| Tag `weights_file` | filename | **`weights_dir`** (directory name; MLflow stores it at `artifacts/<dir name>/`) |
| sha256 | hash of one file | hash of all files in the directory (sorted, path + hash) |
| Tracking URI | hard-coded `http://localhost:5050` | env var `MLFLOW_TRACKING_URI` (`http://mlflow:5000` inside Docker) |
| API container | none | new `api` service (ffmpeg + CPU PyTorch) |
| Extras | none | `/health`, `/admin/reload` (hot-swap), `STRICT_MLFLOW` switch, static frontend |

### Facts already verified against MLflow 3.17.2 (do not re-test)
- Registering a **directory** via `artifacts={"weights": <dir>}` stores it as `artifacts/<dir basename>/...`. After `download_artifacts("models:/name/<ver>")` the weights are at `<dst>/artifacts/<dir basename>/`. So the tag `weights_dir` must equal the directory's basename.
- A 1.2 GB upload through `--serve-artifacts` took ~7 s locally and the download ~4 s. No timeout tuning needed (the default server is uvicorn).
- `/health` on the MLflow server answers 200 for any Host header, so a Docker healthcheck on it is safe. Real API routes return **403** for a Host not in `--allowed-hosts`.
- `sshleifer/distilbart-cnn-12-6` has **only** `pytorch_model.bin` (1.22 GB), no safetensors. `openai/whisper-base` has safetensors plus other formats, so filter downloads (see `hf_fetch.py`).

### Facts about the libraries (important, easy to get wrong)
- **Current `transformers` is v5.x and has NO `"summarization"` pipeline.** Do not write `pipeline("summarization", ...)`. Use `AutoTokenizer` + `AutoModelForSeq2SeqLM` + `model.generate(...)` as in `summarizer.py` below.
- `pipeline("automatic-speech-recognition", ...)` still exists, and `from transformers.pipelines.audio_utils import ffmpeg_read` imports fine.
- Loading a `.bin` checkpoint needs **torch >= 2.6**.
- Use the **CPU** PyTorch wheel (`https://download.pytorch.org/whl/cpu`) in the Docker image; the default one is multi-GB.

> **Not yet tested by me:** the Whisper/DistilBART inference code, the Docker build of the `api` image, and the frontend. Treat those as drafts to run and fix.

---

## 3. Target layout

```
meeting-notes-ai/
├── docker-compose.yaml
├── .gitignore  .gitattributes  .dockerignore
├── README.md
├── models/                      # git-ignored; HF downloads + MLflow download cache (bind-mounted to /models)
├── scripts/                     # run on the HOST
│   ├── requirements.txt
│   ├── weights_model.py         # unchanged from template
│   ├── register_pretrained.py   # adapted
│   └── set_alias.py             # new: promote a version
└── app/                         # runs in the api container
    ├── Dockerfile
    ├── requirements.txt
    ├── __init__.py
    ├── main.py
    ├── model_source.py          # adapted
    ├── hf_fetch.py              # new (fallback download)
    ├── asr.py                   # new
    ├── summarizer.py            # new
    └── static/index.html        # new frontend
```

---

## 4. Files

### 4.1 `docker-compose.yaml`
```yaml
name: meeting-notes-ai

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v3.17.2
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --artifacts-destination /mlflow/artifacts
      --serve-artifacts
      --allowed-hosts "mlflow:5000,localhost:5050,127.0.0.1:5050"
    ports:
      - "5050:5000"
    volumes:
      - mlflow-data:/mlflow
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 5s
      timeout: 5s
      retries: 20

  api:
    build:
      context: .
      dockerfile: app/Dockerfile
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000   # service name, NOT localhost
      MODELS_DIR: /models
      STRICT_MLFLOW: "0"                        # set "1" for the final demo check
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    depends_on:
      mlflow:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 120s

volumes:
  mlflow-data:
```
If the MLflow UI is opened from a different hostname/IP, add it to `--allowed-hosts`.

### 4.2 `scripts/requirements.txt` (host side)
```
mlflow==3.17.2
huggingface_hub
```
(Template also listed `mlflow-skinny`, `ultralytics`, `fastapi`, etc. Not needed on the host any more. After the first successful run, pin `huggingface_hub`.)

### 4.3 `scripts/weights_model.py`
Keep as in the template (optionally fix the typo in the error message).
```python
import mlflow
from mlflow.pyfunc import PythonModel

class WeightOnly(PythonModel):
    def predict(self, context, model_input, params=None):
        raise NotImplementedError("Weights are stored here; the API serves the model.")

mlflow.models.set_model(WeightOnly())
```

### 4.4 `scripts/register_pretrained.py`
```python
from pathlib import Path
import argparse
import hashlib
import os
import shutil

import mlflow
from mlflow import MlflowClient

CATALOG = {
    "whisper-tiny":        {"repo": "openai/whisper-tiny",           "task": "automatic-speech-recognition"},
    "whisper-base":        {"repo": "openai/whisper-base",           "task": "automatic-speech-recognition"},
    "distilbart-cnn-12-6": {"repo": "sshleifer/distilbart-cnn-12-6", "task": "summarization"},
    "distilbart-cnn-6-6":  {"repo": "sshleifer/distilbart-cnn-6-6",  "task": "summarization"},
}
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"          # models/whisper-base/, models/distilbart-cnn-12-6/ ...
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
SKIP = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "*.tflite", "*.mlmodel", "*.gguf"]


def dir_sha256(root: Path) -> str:
    """Hash of every file (relative path + content hash), order-independent of the OS."""
    digest = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        fh = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                fh.update(chunk)
        digest.update(f"{p.relative_to(root).as_posix()}:{fh.hexdigest()}\n".encode())
    return digest.hexdigest()


def fetch(key: str) -> Path:
    """Download the HF repo into models/<key>/ (skips if config.json already there)."""
    dest = MODELS / key
    if not (dest / "config.json").exists():
        from huggingface_hub import list_repo_files, snapshot_download
        repo = CATALOG[key]["repo"]
        ignore = list(SKIP)
        if any(f.endswith(".safetensors") for f in list_repo_files(repo)):
            ignore.append("*.bin")          # prefer safetensors when the repo has them
        print(f"downloading {repo} -> {dest}")
        snapshot_download(repo, local_dir=str(dest), ignore_patterns=ignore)
        shutil.rmtree(dest / ".cache", ignore_errors=True)   # HF metadata; must not be hashed/uploaded
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="registered model name, e.g. whisper-asr or summarizer-distilbart")
    ap.add_argument("key", choices=sorted(CATALOG))
    ap.add_argument("--alias", default="challenger", help="use 'champion' for the version the API should serve")
    args = ap.parse_args()
    cfg = CATALOG[args.key]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("pretrained-meeting-notes")
    path = fetch(args.key)
    digest = dir_sha256(path)
    size_mb = round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20, 1)

    with mlflow.start_run(run_name=f"register-{args.key}"):
        mlflow.set_tags({"pipeline": "registration", "source_repo": cfg["repo"]})
        mlflow.log_params({"weights_dir": args.key, "size_mb": size_mb})
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=str(ROOT / "scripts" / "weights_model.py"),
            artifacts={"weights": str(path)},
            registered_model_name=args.name,
            pip_requirements=[f"mlflow=={mlflow.__version__}"],   # skips slow requirement inference
        )
        client = MlflowClient()
        version = str(info.registered_model_version)
        for k, v in {"sha256": digest, "weights_dir": args.key,
                     "hf_repo": cfg["repo"], "task": cfg["task"]}.items():
            client.set_model_version_tag(args.name, version, k, v)
        client.set_registered_model_alias(args.name, args.alias, version)
        print(f"{args.name} v{version} -> @{args.alias}  sha256={digest[:12]}...")


if __name__ == "__main__":
    main()
```

### 4.5 `scripts/set_alias.py`
```python
import argparse, os
from mlflow import MlflowClient

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("version"); ap.add_argument("alias")
a = ap.parse_args()
MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")) \
    .set_registered_model_alias(a.name, a.alias, a.version)
print(f"{a.name}@{a.alias} -> v{a.version}")
```
(The same can be done in the MLflow UI: Models → model → version → Aliases.)

### 4.6 `app/requirements.txt`
```
mlflow-skinny==3.17.2
fastapi==0.162.0
uvicorn[standard]==0.54.0
python-multipart==0.0.32
transformers        # PIN after first successful build (pip freeze)
numpy
# torch is installed separately in the Dockerfile (CPU wheel, >=2.6)
```

### 4.7 `app/Dockerfile`
```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
RUN pip install --no-cache-dir "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
COPY app app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# ONE worker: each worker would load its own ~1.6 GB copy of the models
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
`.dockerignore`: `models/`, `.git`, `**/__pycache__`, `*.md`.

### 4.8 `app/model_source.py` (adapted from the template)
```python
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import logging
import os
import shutil
from typing import Callable

log = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
STRICT = os.getenv("STRICT_MLFLOW", "0") == "1"   # 1 = never fall back; fail loudly


@dataclass
class ModelSource:
    path: Path
    version: str                       # "mlflow:<name>/<ver>" or "local:<dir>"
    tags: dict[str, str] = field(default_factory=dict)


def dir_sha256(root: Path) -> str:     # MUST match scripts/register_pretrained.py
    digest = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        fh = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                fh.update(chunk)
        digest.update(f"{p.relative_to(root).as_posix()}:{fh.hexdigest()}\n".encode())
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
        mv = client.get_model_version_by_alias(name, alias)
        version = str(mv.version)
        dst = cache_dir / name / version
        path = dst / "artifacts" / mv.tags["weights_dir"]
        if not path.exists():                       # versions are immutable -> cache per version
            dst.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(TRACKING_URI)
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{name}/{version}", dst_path=str(dst))
        try:
            verify(path, mv.tags.get("sha256"))
        except RuntimeError:
            shutil.rmtree(dst, ignore_errors=True)  # drop a corrupt/partial cache
            raise
        return ModelSource(path, f"mlflow:{name}/{version}", dict(mv.tags))
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
```

### 4.9 `app/hf_fetch.py` (fallback only, same filtering as the register script)
```python
import shutil
from pathlib import Path

SKIP = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "*.tflite", "*.mlmodel", "*.gguf"]


def fetch_repo(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import list_repo_files, snapshot_download
    ignore = list(SKIP)
    if any(f.endswith(".safetensors") for f in list_repo_files(repo_id)):
        ignore.append("*.bin")
    snapshot_download(repo_id, local_dir=str(dest), ignore_patterns=ignore)
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    return dest
```

### 4.10 `app/asr.py`
```python
from pathlib import Path
import numpy as np

SAMPLE_RATE = 16_000


def build(path: Path):
    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", model=str(path), device="cpu",
                    chunk_length_s=30)                       # needed for audio > 30 s
    pipe(np.zeros(SAMPLE_RATE, dtype=np.float32))            # warm-up
    return pipe


def decode(audio_bytes: bytes) -> np.ndarray:
    """Any format ffmpeg understands (mp3, m4a, wav, webm from the browser mic...) -> 16 kHz mono float32."""
    from transformers.pipelines.audio_utils import ffmpeg_read
    return ffmpeg_read(audio_bytes, SAMPLE_RATE)


def transcribe(pipe, audio: np.ndarray, language: str | None = None, translate: bool = False) -> str:
    kwargs = {}
    if language:
        kwargs["language"] = language
    if translate:
        kwargs["task"] = "translate"                          # any language -> English
    return pipe(audio, generate_kwargs=kwargs)["text"].strip()
```
If the installed transformers rejects `chunk_length_s` or `generate_kwargs`, read the warning and adapt; do not silently drop long-audio support.

### 4.11 `app/summarizer.py` (no pipeline; handles the 1024-token limit)
```python
from pathlib import Path
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Summarizer:
    def __init__(self, path: Path):
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(path)).eval()
        self.model_limit = getattr(self.model.config, "max_position_embeddings", 1024)  # 1024 for BART
        self.chunk_limit = self.model_limit - 124                                     # safety margin
        self._gen("Warm up the summarizer with a few ordinary words. " * 8, 20, 5)

    def _ntok(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def _gen(self, text: str, max_len: int = 130, min_len: int = 30) -> str:
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=self.model_limit)
        n_in = inputs["input_ids"].shape[1]
        max_len = max(20, min(max_len, int(n_in * 0.7)))      # never ask for more than the input has
        min_len = min(min_len, max_len // 2)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, num_beams=4, max_length=max_len, min_length=min_len,
                                      no_repeat_ngram_size=3, early_stopping=True, do_sample=False)
        return self.tok.decode(ids[0], skip_special_tokens=True).strip()

    def _pieces(self, text: str):
        for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
            ids = self.tok.encode(sent, add_special_tokens=False)
            if len(ids) <= self.chunk_limit:
                yield sent, len(ids)
            else:                                              # giant unpunctuated "sentence": hard split
                for i in range(0, len(ids), self.chunk_limit):
                    part = ids[i:i + self.chunk_limit]
                    yield self.tok.decode(part), len(part)

    def chunk(self, text: str) -> list[str]:
        chunks, cur, n = [], [], 0
        for piece, k in self._pieces(text):
            if cur and n + k > self.chunk_limit:
                chunks.append(" ".join(cur)); cur, n = [], 0
            cur.append(piece); n += k
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    def summarize(self, text: str) -> dict:
        text = text.strip()
        if len(text.split()) < 40:                             # too short: summarizing would invent text
            return {"summary": text, "chunks": 0}
        parts = self.chunk(text)
        summary = " ".join(self._gen(p) for p in parts)        # map
        for _ in range(4):                                     # reduce until it fits in one window
            if self._ntok(summary) <= self.chunk_limit:
                break
            summary = " ".join(self._gen(p) for p in self.chunk(summary))
        if len(parts) > 1:
            summary = self._gen(summary)                       # final pass to make it read as one summary
        return {"summary": summary, "chunks": len(parts)}
```

### 4.12 `app/main.py`
```python
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import asr
from app.hf_fetch import fetch_repo
from app.model_source import resolve
from app.summarizer import Summarizer

logging.basicConfig(level=logging.INFO)     # otherwise app logs are invisible under uvicorn
log = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "600"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

SPECS = {
    "asr": {"name": "whisper-asr", "alias": os.getenv("ASR_ALIAS", "champion"),
            "key": "whisper-base", "repo": "openai/whisper-base",
            "task": "automatic-speech-recognition", "build": asr.build},
    "summarizer": {"name": "summarizer-distilbart", "alias": os.getenv("SUMMARIZER_ALIAS", "champion"),
                   "key": "distilbart-cnn-12-6", "repo": "sshleifer/distilbart-cnn-12-6",
                   "task": "summarization", "build": Summarizer},
}

_lock = threading.Lock()
_state: dict[str, dict] = {}


def load(kind: str) -> str:
    spec = SPECS[kind]
    source = resolve(spec["name"], spec["alias"], MODELS_DIR / spec["key"],
                     lambda p: fetch_repo(spec["repo"], p), MODELS_DIR / "cache")
    task = source.tags.get("task")
    if task and task != spec["task"]:
        raise RuntimeError(f"{spec['name']}@{spec['alias']} is a '{task}' model, expected '{spec['task']}'")
    model = spec["build"](source.path)                  # slow part happens OUTSIDE the lock
    with _lock:
        _state[kind] = {"model": model, "version": source.version, "alias": spec["alias"]}
    log.info("serving %s: %s", kind, source.version)
    return source.version


def _get(kind: str) -> dict:
    with _lock:
        entry = _state.get(kind)
    if entry is None:
        raise HTTPException(503, f"{kind} model not loaded")
    return entry


@asynccontextmanager
async def lifespan(app: FastAPI):
    for kind in SPECS:
        load(kind)
    yield
    _state.clear()


app = FastAPI(title="Meeting Notes AI", lifespan=lifespan)


class SummarizeIn(BaseModel):
    text: str


def _sentences(text: str) -> list[str]:
    import re
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _read_audio(file: UploadFile):
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 2**20:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
    try:
        audio = asr.decode(data)
    except Exception:
        raise HTTPException(400, "could not decode audio (is it a valid audio/video file?)")
    seconds = len(audio) / asr.SAMPLE_RATE
    if seconds < 0.5:
        raise HTTPException(400, "audio is empty or too short")
    if seconds > MAX_AUDIO_SECONDS:
        raise HTTPException(413, f"audio longer than {MAX_AUDIO_SECONDS}s (CPU demo limit)")
    return audio, round(seconds, 1)


@app.get("/health")
def health():
    with _lock:
        models = {k: {"version": v["version"], "alias": v["alias"]} for k, v in _state.items()}
    return {"status": "ok" if len(models) == len(SPECS) else "loading", "models": models}


@app.post("/admin/reload")      # hot-swap: re-resolve the alias in MLflow without restarting
def reload(model: str = Query("all"), alias: str | None = Query(None)):
    kinds = list(SPECS) if model == "all" else [model]
    if any(k not in SPECS for k in kinds):
        raise HTTPException(400, f"model must be 'all' or one of {list(SPECS)}")
    if alias and len(kinds) != 1:
        raise HTTPException(400, "alias can only be set together with a single model")
    for k in kinds:
        old = SPECS[k]["alias"]
        if alias:
            SPECS[k]["alias"] = alias
        try:
            load(k)
        except Exception as ex:
            SPECS[k]["alias"] = old
            raise HTTPException(500, f"reload of {k} failed: {ex}")
    return health()


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...), language: str | None = Form(None),
               translate: bool = Form(False)):
    entry = _get("asr")
    audio, seconds = _read_audio(file)
    t = time.perf_counter()
    text = asr.transcribe(entry["model"], audio, language, translate)
    return {"text": text, "audio_seconds": seconds, "latency_s": round(time.perf_counter() - t, 2),
            "model": entry["version"]}


@app.post("/summarize")
def summarize(body: SummarizeIn):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    entry = _get("summarizer")
    t = time.perf_counter()
    out = entry["model"].summarize(body.text)
    return {**out, "bullets": _sentences(out["summary"]),
            "latency_s": round(time.perf_counter() - t, 2), "model": entry["version"]}


@app.post("/process")           # audio -> transcript -> summary
def process(file: UploadFile = File(...), language: str | None = Form(None),
            translate: bool = Form(False)):
    a, s = _get("asr"), _get("summarizer")
    audio, seconds = _read_audio(file)
    t0 = time.perf_counter()
    transcript = asr.transcribe(a["model"], audio, language, translate)
    t1 = time.perf_counter()
    out = s["model"].summarize(transcript)
    t2 = time.perf_counter()
    return {"transcript": transcript, "summary": out["summary"], "bullets": _sentences(out["summary"]),
            "chunks": out["chunks"], "audio_seconds": seconds,
            "timings": {"asr_s": round(t1 - t0, 2), "summarize_s": round(t2 - t1, 2)},
            "models": {"asr": a["version"], "summarizer": s["version"]}}


# mount LAST so API routes win
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
```
Endpoints are plain `def` on purpose: FastAPI runs them in a threadpool so the model call doesn't block the server.

### 4.13 `app/static/index.html` (provided: copy it as is)
The finished frontend is the separate file **`index.html`** delivered with this guide. Put it at `app/static/index.html`; do not redesign it. It is one self-contained page (no build step, no external assets, light and dark mode, responsive) that uses these endpoints on the same origin:
- `GET /health` for the status pill and the "Models in production" cards (green "Loaded from MLflow" badge, amber "Local fallback" badge when the version starts with `local:`).
- `POST /transcribe` (multipart field `file`), then `POST /summarize` (JSON `{"text": ...}`). The page calls these two in sequence instead of `/process` so the 3-step progress indicator is real.
- `POST /admin/reload` for the "Reload from MLflow" button (hot-swap demo).
- It is **English only**: no language or translate controls. The API still accepts the optional `language`/`translate` form fields; the UI just doesn't use them.
- Recording uses `MediaRecorder` (works on `http://localhost` only), auto-stops at 10 minutes to match `MAX_AUDIO_SECONDS`.
If you change an API response field name, update the page too. Fields it reads: `/health` -> `status`, `models.<kind>.version|alias`; `/transcribe` -> `text`, `audio_seconds`, `latency_s`; `/summarize` -> `summary`, `bullets`, `chunks`, `latency_s`; errors -> `detail`.

---

## 5. Run order (the exact demo flow)

```bash
# 1. MLflow only
docker compose up -d mlflow            # UI: http://localhost:5050

# 2. Register models from the host (downloads from Hugging Face, uploads into MLflow)
pip install -r scripts/requirements.txt
python scripts/register_pretrained.py whisper-asr whisper-base --alias champion
python scripts/register_pretrained.py summarizer-distilbart distilbart-cnn-12-6 --alias champion

# 3. API + frontend
docker compose up -d --build api
docker compose logs -f api             # expect: "serving asr: mlflow:whisper-asr/1"
# open http://localhost:8000
```
Disk: budget ~6 GB (HF copy + MLflow store + download cache). Docker Desktop memory: give it **at least 4 GB** (the two models need ~1.6 GB in RAM; OOM kills look like random container exits).

### Hot-swap / A-B demo (the part that impresses)
```bash
python scripts/register_pretrained.py whisper-asr whisper-tiny --alias challenger   # creates v2
curl -X POST "http://localhost:8000/admin/reload?model=asr&alias=challenger"        # serve v2 now, compare speed
python scripts/set_alias.py whisper-asr 2 champion                                  # "promote" v2 in MLflow
curl -X POST "http://localhost:8000/admin/reload?model=asr&alias=champion"          # API now serves v2, no restart
```

---

## 6. Gotchas to avoid

1. **Silent fallback hides failures.** If MLflow is unreachable the API quietly downloads from Hugging Face and still works, so "it works" proves nothing. Always check `/health` shows `mlflow:` versions, and do the strict check in section 7.
2. **`localhost` inside a container is the container itself.** The API must use `http://mlflow:5000`. The host scripts use `http://localhost:5050`.
3. **Version mismatch** between `mlflow` client and server causes odd errors; keep both at 3.17.2.
4. **Do not use `pipeline("summarization")`** (removed in transformers 5). Do not use multiple uvicorn workers.
5. **Whisper hallucinates on silence/music** ("Thank you."). Use clips with clear speech.
6. **Languages:** Whisper (multilingual `base`) accepts Thai and others, but DistilBART only summarizes **English**. For non-English audio, use the "translate to English" option or demo with English.
7. **Mic recording** only works on `http://localhost` or HTTPS, not on `http://<LAN-IP>`.
8. First request can be slow; the warm-up in `build()` handles most of it. Demo with 1–2 minute clips and keep a pre-recorded fallback clip.
9. `/admin/reload` has no authentication. Fine for a classroom demo; mention it in the presentation as a known limitation.

---

## 7. Verification checklist (definition of done)

- [ ] `docker compose up -d mlflow` → UI opens at `http://localhost:5050`, container is `healthy`.
- [ ] After the two register commands, the UI **Models** tab lists `whisper-asr` and `summarizer-distilbart`, each with version 1, alias `champion`, and tags `sha256`, `weights_dir`, `hf_repo`, `task`.
- [ ] `docker compose logs api` shows `serving asr: mlflow:whisper-asr/1` and `serving summarizer: mlflow:summarizer-distilbart/1`.
- [ ] `curl localhost:8000/health` → `status: ok`, both versions start with `mlflow:`.
- [ ] `curl -F file=@sample.wav localhost:8000/process` returns transcript, summary, bullets, timings.
- [ ] A transcript > ~1,000 words returns `chunks > 1` and does not crash (tests the chunking; use a long text on `/summarize`).
- [ ] **Strict check:** set `STRICT_MLFLOW: "1"`, delete `./models/whisper-base` and `./models/distilbart-cnn-12-6` (the local fallback copies), `docker compose up -d api`: it must still start. Then `docker compose stop mlflow` and restart `api`: it must **fail**. This proves the API really loads from MLflow. Restore `STRICT_MLFLOW: "0"` afterwards.
- [ ] Hot-swap sequence in section 5 changes the version shown in `/health` and in the UI badge.
- [ ] Uploading a `.txt` renamed to `.mp3` returns a clean 400, not a 500.
- [ ] Frontend works from a fresh browser tab: upload, record, error display, badges.
- [ ] `README.md` has the run order from section 5, an architecture sketch, and the model choices. This is the presentation backbone.

## 8. Stretch goals (only if everything above is green)

- Log every request to an MLflow experiment `inference` (latency, audio seconds, model version). Wrap in `try/except` and keep it off the critical path; the API must work if MLflow is down.
- "Action items" list: pick summary/transcript sentences containing words like *need to, should, will, must, deadline, by (day)*. No extra model needed.
- Segment timestamps from Whisper (`return_timestamps=True`) shown next to the transcript.
- Whisper size dropdown backed by the two registered versions (tiny vs base) with latency shown.
