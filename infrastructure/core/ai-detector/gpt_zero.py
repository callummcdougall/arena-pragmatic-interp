import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
env_path = Path(__file__).resolve().parents[3] / "chapter4_alignment_science" / "exercises" / ".env"
load_dotenv(env_path)

API_KEY = os.getenv("GPTZERO_API_KEY")
URL = "https://api.gptzero.me/v2/predict/files"

CACHE_PATH = Path(__file__).resolve().parent / "gptzero_cache.json"
cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}


def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def key_for(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _needs_refetch(entry: dict) -> bool:
    """Return True if entry should be re-fetched (error, or missing sentence scores)."""
    if entry.get("skipped"):
        return False
    if entry.get("error"):
        return True
    return not entry.get("sentence_scores")


def _normalize_result(doc: dict) -> dict:
    """Convert GPTZero document response to our standard format (matching Sapling's shape)."""
    score = doc.get("class_probabilities", {}).get("ai", 0)
    sentence_scores = [
        {"score": s.get("generated_prob", 0), "sentence": s.get("sentence", "")}
        for s in doc.get("sentences", [])
    ]
    return {"score": score, "sentence_scores": sentence_scores}


def _score_batch(texts: list[str], max_retries=3, backoff_seconds=2.0) -> list[tuple[str, dict]]:
    """Score a batch of texts (up to 50) in a single API call."""
    keys = [key_for(t) for t in texts]

    # Build multipart file list
    files = []
    for i, t in enumerate(texts):
        buf = io.BytesIO(t.strip().encode("utf-8"))
        files.append(("files", (f"chunk_{i}.txt", buf, "text/plain")))

    for attempt in range(max_retries):
        try:
            r = requests.post(
                URL,
                headers={"x-api-key": API_KEY, "Accept": "application/json"},
                files=files,
                timeout=120,
            )
        except requests.RequestException as e:
            print(f"GPTZero request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (2**attempt))
                # Reset file pointers for retry
                for _, (_, buf, _) in files:
                    buf.seek(0)
                continue
            return [(k, {"score": 0, "error": True, "msg": str(e)}) for k in keys]

        if 200 <= r.status_code < 300:
            data = r.json()
            docs = data.get("documents", [])
            results = []
            for k, text, doc in zip(keys, texts, docs):
                out = _normalize_result(doc)
                cache[k] = out
                results.append((k, out))
            return results

        print(f"GPTZero error: status={r.status_code}, body={r.text[:500]}")

        if r.status_code == 429 and attempt < max_retries - 1:
            time.sleep(backoff_seconds * (2**attempt))
            for _, (_, buf, _) in files:
                buf.seek(0)
            continue

        if attempt < max_retries - 1:
            time.sleep(backoff_seconds * (2**attempt))
            for _, (_, buf, _) in files:
                buf.seek(0)
            continue

        return [
            (k, {"score": 0, "error": True, "status": r.status_code, "body": r.text})
            for k in keys
        ]

    return [(k, {"score": 0, "error": True, "msg": "max retries exceeded"}) for k in keys]


def score_one(text: str, max_retries=3, backoff_seconds=2.0):
    k = key_for(text)
    if k in cache and not _needs_refetch(cache[k]):
        return k, cache[k]

    t = text.strip()
    if not t:
        result = {"score": 0, "skipped": "empty"}
        cache[k] = result
        return k, result

    results = _score_batch([t], max_retries=max_retries, backoff_seconds=backoff_seconds)
    return results[0]


def score_many(texts, concurrency=50):
    """Score texts in batches of up to 50, using the cache for already-scored texts."""
    BATCH_SIZE = 50

    # Separate cached from uncached
    results = []
    uncached = []  # (original_index, text)
    for i, text in enumerate(texts):
        k = key_for(text)
        t = text.strip()
        if not t:
            entry = {"score": 0, "skipped": "empty"}
            cache[k] = entry
            results.append((k, entry))
        elif k in cache and not _needs_refetch(cache[k]):
            results.append((k, cache[k]))
        else:
            uncached.append((i, text))

    # Batch the uncached texts
    for batch_start in range(0, len(uncached), BATCH_SIZE):
        batch = uncached[batch_start : batch_start + BATCH_SIZE]
        batch_texts = [t for _, t in batch]
        print(f"GPTZero: scoring batch of {len(batch_texts)} chunks...")
        batch_results = _score_batch(batch_texts)
        results.extend(batch_results)

    return results


# Re-export split_into_chunks so nb.py can import it from either module
from sapling import split_into_chunks  # noqa: E402, F401
