import shutil
from pathlib import Path

SKIP = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "*.tflite", "*.mlmodel", "*.gguf"]


def fetch_repo(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import list_repo_files, snapshot_download

    ignore = list(SKIP)
    if any(filename.endswith(".safetensors") for filename in list_repo_files(repo_id)):
        ignore.append("*.bin")
    snapshot_download(repo_id, local_dir=str(dest), ignore_patterns=ignore)
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    return dest
