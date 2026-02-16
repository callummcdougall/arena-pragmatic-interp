import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
env_path = Path(__file__).resolve().parents[3] / "chapter4_alignment_science" / "exercises" / ".env"
load_dotenv(env_path)

API_KEY = os.getenv("SAPLING_API_KEY")
URL = "https://api.sapling.ai/api/v1/aidetect"

CACHE_PATH = Path(__file__).resolve().parent / "aidetect_cache.json"
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


def score_one(text: str, max_retries=3, backoff_seconds=2.0):
    k = key_for(text)
    if k in cache and not _needs_refetch(cache[k]):
        return k, cache[k]

    t = text.strip()
    if not t:
        result = {"score": 0, "skipped": "empty"}
        cache[k] = result
        return k, result

    for attempt in range(max_retries):
        r = requests.post(URL, json={"key": API_KEY, "text": t, "sent_scores": True}, timeout=60)

        if 200 <= r.status_code < 300:
            out = r.json()
            cache[k] = out
            return k, out

        msg = None
        try:
            msg = r.json().get("msg")
            print("Sapling error JSON:", {"msg": msg})
        except Exception:
            print("Sapling error TEXT:", r.text[:500])

        print("Status:", r.status_code, "len(text):", len(t), "first100:", t[:100])

        if msg and "Unexpected error running classification" in msg and attempt < max_retries - 1:
            time.sleep(backoff_seconds * (2**attempt))
            continue

        out = {"score": 0, "error": True, "status": r.status_code, "msg": msg, "body": r.text, "text": t}
        cache[k] = out
        return k, out

    out = {"score": 0, "error": True, "status": r.status_code, "msg": msg, "body": r.text, "text": t}
    cache[k] = out
    return k, out


def score_many(texts, concurrency=50):
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(score_one, text): text for text in texts}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def split_into_chunks(text, max_chunk_length=2000, min_chunk_len=500):
    """Split text into chunks, preferring sentence boundaries, then merge short chunks forward."""
    if len(text) <= max_chunk_length:
        return [text]

    chunks = []
    sentences = re.split(r"([.!?]\s+)", text)

    current_chunk = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        separator = sentences[i + 1] if i + 1 < len(sentences) else ""

        if len(current_chunk) + len(sentence) + len(separator) <= max_chunk_length:
            current_chunk += sentence + separator
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + separator

    if current_chunk:
        chunks.append(current_chunk.strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_length:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_chunk_length):
                final_chunks.append(chunk[i : i + max_chunk_length])

    final_chunks = [c for c in final_chunks if c.strip()]
    if not final_chunks:
        return []

    merged_chunks = []
    i = 0
    while i < len(final_chunks):
        chunk = final_chunks[i]
        if len(chunk) < min_chunk_len and i + 1 < len(final_chunks):
            final_chunks[i + 1] = f"{chunk} {final_chunks[i + 1]}".strip()
            i += 1
            continue
        merged_chunks.append(chunk)
        i += 1

    if merged_chunks and len(merged_chunks[-1]) < min_chunk_len and len(merged_chunks) > 1:
        merged_chunks[-2] = f"{merged_chunks[-2]} {merged_chunks[-1]}".strip()
        merged_chunks.pop()

    return merged_chunks
