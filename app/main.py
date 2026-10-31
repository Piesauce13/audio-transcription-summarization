import logging
import os
import re
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

logging.basicConfig(level=logging.INFO)
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
                     lambda path: fetch_repo(spec["repo"], path), MODELS_DIR / "cache")
    task = source.tags.get("task")
    if task and task != spec["task"]:
        raise RuntimeError(f"{spec['name']}@{spec['alias']} is a '{task}' model, expected '{spec['task']}'")
    model = spec["build"](source.path)
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
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]


def _read_audio(file: UploadFile):
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 2**20:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
    try:
        audio = asr.decode(data)
    except Exception as ex:
        log.warning("audio decoding failed: %s", ex)
        raise HTTPException(400, "could not decode audio (is it a valid audio/video file?)") from ex
    seconds = len(audio) / asr.SAMPLE_RATE
    if seconds < 0.5:
        raise HTTPException(400, "audio is empty or too short")
    if seconds > MAX_AUDIO_SECONDS:
        raise HTTPException(413, f"audio longer than {MAX_AUDIO_SECONDS}s (CPU demo limit)")
    return audio, round(seconds, 1)


@app.get("/health")
def health():
    with _lock:
        models = {kind: {"version": value["version"], "alias": value["alias"]}
                  for kind, value in _state.items()}
    return {"status": "ok" if len(models) == len(SPECS) else "loading", "models": models}


@app.post("/admin/reload")
def reload_models(model: str = Query("all"), alias: str | None = Query(None)):
    kinds = list(SPECS) if model == "all" else [model]
    if any(kind not in SPECS for kind in kinds):
        raise HTTPException(400, f"model must be 'all' or one of {list(SPECS)}")
    if alias and len(kinds) != 1:
        raise HTTPException(400, "alias can only be set together with a single model")
    for kind in kinds:
        old_alias = SPECS[kind]["alias"]
        if alias:
            SPECS[kind]["alias"] = alias
        try:
            load(kind)
        except Exception as ex:
            SPECS[kind]["alias"] = old_alias
            raise HTTPException(500, f"reload of {kind} failed: {ex}") from ex
    return health()


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...), language: str | None = Form(None),
               translate: bool = Form(False)):
    entry = _get("asr")
    audio, seconds = _read_audio(file)
    started = time.perf_counter()
    text = asr.transcribe(entry["model"], audio, language, translate)
    return {"text": text, "audio_seconds": seconds, "latency_s": round(time.perf_counter() - started, 2),
            "model": entry["version"]}


@app.post("/summarize")
def summarize(body: SummarizeIn):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    entry = _get("summarizer")
    started = time.perf_counter()
    output = entry["model"].summarize(body.text)
    return {**output, "bullets": _sentences(output["summary"]),
            "latency_s": round(time.perf_counter() - started, 2), "model": entry["version"]}


@app.post("/process")
def process(file: UploadFile = File(...), language: str | None = Form(None),
            translate: bool = Form(False)):
    asr_entry, summary_entry = _get("asr"), _get("summarizer")
    audio, seconds = _read_audio(file)
    started = time.perf_counter()
    transcript = asr.transcribe(asr_entry["model"], audio, language, translate)
    transcribed = time.perf_counter()
    output = summary_entry["model"].summarize(transcript)
    summarized = time.perf_counter()
    return {"transcript": transcript, "summary": output["summary"],
            "bullets": _sentences(output["summary"]), "chunks": output["chunks"],
            "audio_seconds": seconds,
            "timings": {"asr_s": round(transcribed - started, 2),
                        "summarize_s": round(summarized - transcribed, 2)},
            "models": {"asr": asr_entry["version"], "summarizer": summary_entry["version"]}}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
