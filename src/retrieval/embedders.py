"""Text -> vector, behind one small interface, so the index does not care which model made them.

    NOTES_EMBEDDER=hash      offline, free, deterministic. Feature hashing of words and word pairs:
                             lexical, not semantic. For tests and for running without AWS.
    NOTES_EMBEDDER=bedrock   Amazon Titan Text Embeddings V2 on Bedrock (TITAN_EMBED_MODEL_ID).
                             Semantic: "breathing problems" lands near "dyspnea" and "wheezing".

An index records which embedder built it; querying with a different one is refused, because
vectors from two models are not comparable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
# Words that occur everywhere in notes carry no signal; "and" alone once matched every
# "Assessment and Plan" section.
STOPWORDS = frozenset(
    "a an and are as at be been by for from has have he her his in is it its of on or patient "
    "she that the their this to was were with who".split())
# Crude suffix stripping so "smoking"/"smoker" and "alcohol"/"alcoholic" meet. Keyword search only
# needs this much; the Titan vectors handle real paraphrase.
_SUFFIXES = ("ically", "ations", "ation", "ings", "ing", "ers", "er", "ics", "ic", "ies", "es", "ed", "ly", "s")


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    name: str  # recorded in the index; must match at query time
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...  # (n, dim), rows L2-normalised


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim, self.name = dim, f"hash-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text)
            for feat in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]:
                h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(out)


class TitanEmbedder:
    DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"

    def __init__(self, model_id: str | None = None, dim: int = 1024, workers: int = 8) -> None:
        import boto3
        from botocore.config import Config

        self.model_id = model_id or os.environ.get("TITAN_EMBED_MODEL_ID", self.DEFAULT_MODEL)
        self.dim, self.name, self._workers = dim, f"{self.model_id}:{dim}", workers
        self._client = boto3.client(
            "bedrock-runtime", region_name=os.environ.get("AWS_REGION") or None,
            config=Config(retries={"mode": "adaptive", "max_attempts": 6}, read_timeout=30))

    def _one(self, text: str) -> list[float]:
        # Titan V2 takes one text per call; up to 8,192 tokens. normalize=True returns unit vectors.
        body = json.dumps({"inputText": text[:20000], "dimensions": self.dim, "normalize": True})
        resp = self._client.invoke_model(modelId=self.model_id, body=body,
                                         contentType="application/json", accept="application/json")
        return json.loads(resp["body"].read())["embedding"]

    def embed(self, texts: list[str]) -> np.ndarray:
        with ThreadPoolExecutor(self._workers) as pool:
            rows = list(pool.map(self._one, texts))
        return _normalise(np.asarray(rows, dtype=np.float32).reshape(len(texts), self.dim))


def make_embedder(kind: str | None = None) -> Embedder:
    match kind or os.environ.get("NOTES_EMBEDDER", "hash"):
        case "hash":
            return HashEmbedder()
        case "bedrock":
            return TitanEmbedder()
        case other:
            raise ValueError(f"NOTES_EMBEDDER={other!r}: expected 'hash' or 'bedrock'")
