from pathlib import Path
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Summarizer:
    def __init__(self, path: Path):
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(path)).eval()
        self.model_limit = getattr(self.model.config, "max_position_embeddings", 1024)
        self.chunk_limit = self.model_limit - 124
        self._gen("Warm up the summarizer with a few ordinary words. " * 8, 20, 5)

    def _ntok(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def _gen(self, text: str, max_len: int = 130, min_len: int = 30) -> str:
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=self.model_limit)
        input_length = inputs["input_ids"].shape[1]
        max_len = max(20, min(max_len, int(input_length * 0.7)))
        min_len = min(min_len, max_len // 2)
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs, num_beams=4, max_length=max_len, min_length=min_len,
                no_repeat_ngram_size=3, early_stopping=True, do_sample=False)
        return self.tok.decode(ids[0], skip_special_tokens=True).strip()

    def _pieces(self, text: str):
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            ids = self.tok.encode(sentence, add_special_tokens=False)
            if len(ids) <= self.chunk_limit:
                yield sentence, len(ids)
            else:
                for index in range(0, len(ids), self.chunk_limit):
                    part = ids[index:index + self.chunk_limit]
                    yield self.tok.decode(part), len(part)

    def chunk(self, text: str) -> list[str]:
        chunks, current, token_count = [], [], 0
        for piece, count in self._pieces(text):
            if current and token_count + count > self.chunk_limit:
                chunks.append(" ".join(current))
                current, token_count = [], 0
            current.append(piece)
            token_count += count
        if current:
            chunks.append(" ".join(current))
        return chunks

    def summarize(self, text: str) -> dict:
        text = text.strip()
        if len(text.split()) < 40:
            return {"summary": text, "chunks": 0}
        parts = self.chunk(text)
        summary = " ".join(self._gen(part) for part in parts)
        for _ in range(4):
            if self._ntok(summary) <= self.chunk_limit:
                break
            summary = " ".join(self._gen(part) for part in self.chunk(summary))
        if len(parts) > 1:
            summary = self._gen(summary)
        return {"summary": summary, "chunks": len(parts)}
