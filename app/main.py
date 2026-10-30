from fastapi import FastAPI
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import threading
import logging
from contextlib import asynccontextmanager

from app.model_source import resolve

log = logging.getLogger(__name__)

_lock = threading.Lock()
_state: dict[str, object] = {}

def _download(path: Path) -> None:
    from ultralytics.utils.downloads import attempt_download_asset
    attempt_download_asset(path)
def load_model() -> str:
    source = resolve("yolo-detector", "champion", Path("/models") / "yolo26n.pt", _download)
    model = YOLO(str(source.path))
    model.predict(Image.new("RGB", (640, 640)), verbose=False) # test prediction will we load model to asynccontextmanager of fastapi
    with _lock:
        _state["model"], _state["version"] = model, source.version
    log.info("serving model %s", source.version)
    return source.version

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    _state.clear()

app = FastAPI(title="Vision Detection App", lifespan=lifespan)