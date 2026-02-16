import re
from enum import StrEnum
from pathlib import Path

from html_template import generate_html

CHAPTERS_DIR = "/root/arena-pragmatic-interp/infrastructure/chapters"


class Detector(StrEnum):
    SAPLING = "sapling"
    GPTZERO = "gptzero"


def _load_detector(detector: Detector):
    """Import and return (key_for, save_cache, score_many, split_into_chunks) for the given detector."""
    if detector is Detector.SAPLING:
        from sapling import key_for, save_cache, score_many, split_into_chunks
    elif detector is Detector.GPTZERO:
        from gpt_zero import key_for, save_cache, score_many, split_into_chunks
    else:
        raise ValueError(f"Unknown detector: {detector}")
    return key_for, save_cache, score_many, split_into_chunks


# ── Cell parsing ──────────────────────────────────────────────────────────────


def parse_cells(
    file_path,
    skip_structure=True,
    skip_headers=True,
    remove_headers=True,
    skip_output=True,
    include_code=False,
    strip_quotes=True,
    skip_python_dropdowns=True,
):
    """Parse a master file and extract cell content.

    Args:
        include_code: If True, include code cells as well as markdown cells (default False).
        strip_quotes: If True (default), remove lines starting with ">" (blockquotes).
        skip_python_dropdowns: If True (default), remove <details>...</details> blocks that contain python code.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    cells = []
    lines = content.split("\n")
    i = 0
    cell_index = 0
    while i < len(lines):
        if "# ! CELL TYPE:" in lines[i]:
            cell_type = lines[i].split("# ! CELL TYPE:")[1].strip()
            i += 1

            if i < len(lines) and "# ! FILTERS:" in lines[i]:
                i += 1

            tags = []
            if i < len(lines) and "# ! TAGS:" in lines[i]:
                tags_str = lines[i].split("# ! TAGS:")[1].strip()
                if tags_str.startswith("[") and tags_str.endswith("]"):
                    tags_str = tags_str[1:-1]
                    tags = [tag.strip() for tag in re.split(r",(?![^\[]*\])", tags_str)]
                i += 1

            cell_start_line = i
            cell_content = []
            while i < len(lines) and "# ! CELL TYPE:" not in lines[i]:
                cell_content.append(lines[i])
                i += 1

            cells.append(
                {
                    "type": cell_type,
                    "content": "\n".join(cell_content),
                    "tags": tags,
                    "index": cell_index,
                    "line_start": cell_start_line,
                }
            )
            cell_index += 1
        else:
            i += 1

    # Filter by cell type
    allowed_types = {"markdown"}
    if include_code:
        allowed_types.add("code")
    filtered_cells = [cell for cell in cells if cell["type"] in allowed_types]

    # Skip first markdown cell if skip_structure is True
    if skip_structure and filtered_cells and filtered_cells[0]["type"] == "markdown":
        filtered_cells = filtered_cells[1:]

    processed_cells = []
    for cell in filtered_cells:
        content = cell["content"]
        tags = cell["tags"]

        # Strip triple-quote wrappers: r""", """, r''', '''
        content = content.strip()
        for prefix in ['r"""', "r'''", '"""', "'''"]:
            if content.startswith(prefix):
                content = content[len(prefix) :]
                break
        for suffix in ['"""', "'''"]:
            if content.endswith(suffix):
                content = content[: -len(suffix)]
                break
        content = content.strip()

        if not content:
            continue

        if skip_output:
            has_html_tag = any("html" in tag.lower() for tag in tags)
            if has_html_tag:
                continue
            if re.match(r"^\s*<img\s+src=.*>\s*$", content, re.IGNORECASE):
                continue

        if skip_headers:
            lines_list = content.split("\n")
            if len(lines_list) == 1 and lines_list[0].strip().startswith("#"):
                continue

        if remove_headers:
            lines_list = content.split("\n")
            if len(lines_list) > 1 and lines_list[0].strip().startswith("#"):
                content = "\n".join(lines_list[1:]).strip()

        if strip_quotes:
            lines_list = content.split("\n")
            content = "\n".join(line for line in lines_list if not line.startswith(">")).strip()

        if skip_python_dropdowns:
            content = re.sub(
                r"<details>.*?</details>",
                lambda m: "" if "```python" in m.group() else m.group(),
                content,
                flags=re.DOTALL,
            ).strip()

        if content:
            processed_cells.append({"content": content, "index": cell["index"], "line_start": cell["line_start"]})

    return processed_cells


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_cells(cells, detector: Detector = Detector.SAPLING, max_chunk_length=2000, min_chunk_len=500, concurrency=50):
    """Score all cells, returning a list of dicts with cell info + scores."""
    key_for, save_cache, score_many, split_into_chunks = _load_detector(detector)

    chunks = []
    chunk_to_cell_idx = []

    for cell_idx, cell in enumerate(cells):
        cell_chunks = split_into_chunks(cell["content"], max_chunk_length, min_chunk_len)
        for chunk in cell_chunks:
            chunks.append(chunk)
            chunk_to_cell_idx.append(cell_idx)

    results = score_many(chunks, concurrency=concurrency)

    chunk_scores = {}
    chunk_entries = {}
    for chunk_hash, entry in results:
        chunk_scores[chunk_hash] = entry.get("score", 0)
        chunk_entries[chunk_hash] = entry

    scored_cells = []
    for cell_idx, cell in enumerate(cells):
        cell_chunk_indices = [i for i, idx in enumerate(chunk_to_cell_idx) if idx == cell_idx]
        cell_chunk_hashes = [key_for(chunks[i]) for i in cell_chunk_indices]
        cell_chunk_scores_list = [chunk_scores[h] for h in cell_chunk_hashes]
        avg_score = sum(cell_chunk_scores_list) / len(cell_chunk_scores_list) if cell_chunk_scores_list else 0

        # Collect sentence scores from all chunks of this cell
        sentence_scores = []
        for h in cell_chunk_hashes:
            entry = chunk_entries[h]
            if entry.get("sentence_scores"):
                sentence_scores.extend(entry["sentence_scores"])

        scored_cells.append(
            {
                "content": cell["content"],
                "index": cell["index"],
                "line_start": cell["line_start"],
                "score": avg_score,
                "sentence_scores": sentence_scores,
            }
        )

    save_cache()
    return scored_cells


# ── Main ──────────────────────────────────────────────────────────────────────

detector = Detector.GPTZERO

file_path = Path(CHAPTERS_DIR) / "chapter4_alignment_science/master_4_1.py"
# file_path = Path(CHAPTERS_DIR) / "chapter1_transformer_interp/section_4_circuits/master_1_4_1.py"
cells = parse_cells(file_path, include_code=False)
scored = score_cells(cells, detector=detector)

reports_dir = Path(__file__).resolve().parent / "ai-reports"
reports_dir.mkdir(exist_ok=True)
stem = file_path.stem  # e.g. "master_4_1"
output_path = reports_dir / f"ai_report_{stem.removeprefix('master_')}_{detector}.html"
generate_html(scored, output_path)
