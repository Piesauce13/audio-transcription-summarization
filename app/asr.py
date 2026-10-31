from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def build(path: Path):
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model=str(path), device="cpu",
                    chunk_length_s=30)
    pipe(np.zeros(SAMPLE_RATE, dtype=np.float32))
    return pipe


def decode(audio_bytes: bytes) -> np.ndarray:
    from transformers.pipelines.audio_utils import ffmpeg_read

    return ffmpeg_read(audio_bytes, SAMPLE_RATE)


def transcribe(pipe, audio: np.ndarray, language: str | None = None,
               translate: bool = False) -> str:
    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language
    if translate:
        generate_kwargs["task"] = "translate"
    return pipe(audio, generate_kwargs=generate_kwargs)["text"].strip()
