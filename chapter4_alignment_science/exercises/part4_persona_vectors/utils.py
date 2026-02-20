"""
Utility functions for Part 4: Persona Vectors exercises.

This module provides three helpers:

1. `load_transcript(path, max_assistant_turns)` — loads a JSON transcript from the
   assistant-axis repo, strips `<INTERNAL_STATE>` tags, and optionally truncates.

2. `get_turn_spans(messages, tokenizer)` — finds the (start, end) token index span
   for each assistant turn in a tokenized conversation. This is fiddly bookkeeping
   (the chat template adds special tokens that shift positions), so it's provided
   here rather than asked of students. Students call it from their ConversationAnalyzer.

3. `plot_similarity_line(cosine_sims, names, ...)` — the trait-axis scatter plot from
   the paper's `visualize_axis.ipynb`. Given an array of cosine similarities and
   corresponding trait names, it draws a horizontal scatter with a histogram overlay
   and labels the most extreme traits at each end.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Transcript loading
# ─────────────────────────────────────────────────────────────────────────────


def load_transcript(transcript_path: Path, max_assistant_turns: int | None = None) -> list[dict[str, str]]:
    """
    Load a JSON transcript from the assistant-axis repo and return a clean conversation.

    Strips ``<INTERNAL_STATE>...</INTERNAL_STATE>`` tags from user messages (these
    represent the simulated user's private thoughts and shouldn't be fed to a model).

    Args:
        transcript_path: Path to the JSON transcript file.
        max_assistant_turns: If given, truncate to this many assistant turns.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data["conversation"]

    # Strip <INTERNAL_STATE>...</INTERNAL_STATE> tags from user messages
    cleaned = []
    for msg in messages:
        content = msg["content"]
        if msg["role"] == "user":
            content = re.sub(r"<INTERNAL_STATE>.*?</INTERNAL_STATE>", "", content, flags=re.DOTALL).strip()
        cleaned.append({"role": msg["role"], "content": content})

    # Truncate by assistant turns if requested
    if max_assistant_turns is not None:
        result = []
        asst_count = 0
        for msg in cleaned:
            result.append(msg)
            if msg["role"] == "assistant":
                asst_count += 1
                if asst_count >= max_assistant_turns:
                    break
        return result

    return cleaned
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────────────────────────────────────
# Turn-span detection
# ─────────────────────────────────────────────────────────────────────────────


def get_turn_spans(messages: list[dict[str, str]], tokenizer) -> list[tuple[int, int]]:
    """
    Return the (start_token_idx, end_token_idx) span for every assistant turn in
    a tokenized conversation.

    **How it works:**

    We process assistant messages one at a time. For each assistant message at
    position `i` in `messages`, we compare two tokenized lengths:

    - "prompt up to here": `messages[:i]` formatted with `add_generation_prompt=True`
      (the model would have seen exactly this before generating the response).
      This tells us where the assistant response *starts*.

    - "conversation including this turn": `messages[:i+1]` formatted with
      `add_generation_prompt=False`.
      This tells us where the assistant response *ends*.

    Args:
        messages:  Full conversation as a list of {"role": ..., "content": ...} dicts.
        tokenizer: HuggingFace tokenizer with `apply_chat_template` support.

    Returns:
        List of (start_idx, end_idx) integer tuples, one per assistant turn.
        The slice `hidden_states[start_idx:end_idx]` gives the response tokens.

    Example::

        spans = get_turn_spans(transcript, tokenizer)
        # spans[0] = (55, 250) — first assistant turn occupies tokens 55..249
        # spans[1] = (310, 480) — second assistant turn occupies tokens 310..479
    """
    spans = []
    assistant_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]

    for asst_idx in assistant_indices:
        # Where the response starts: length of conversation up to (but not including)
        # this assistant message, formatted with generation prompt.
        prompt_before = tokenizer.apply_chat_template(
            messages[:asst_idx], tokenize=False, add_generation_prompt=True
        ).rstrip()
        response_start = tokenizer(prompt_before, return_tensors="pt").input_ids.shape[1]

        # Where the response ends: length of conversation including this message,
        # formatted without generation prompt (we don't want a trailing prompt token).
        full_up_to = tokenizer.apply_chat_template(
            messages[: asst_idx + 1], tokenize=False, add_generation_prompt=False
        )
        response_end = tokenizer(full_up_to, return_tensors="pt").input_ids.shape[1]

        spans.append((response_start, response_end))

    return spans


# ─────────────────────────────────────────────────────────────────────────────
# Trait-axis scatter visualization
# ─────────────────────────────────────────────────────────────────────────────


def plot_similarity_line(
    cosine_sims: np.ndarray,
    names: list[str],
    figsize: tuple[float, float] = (8, 3),
    n_extremes: int = 5,
    show_histogram: bool = True,
) -> plt.Figure:
    """
    Plot trait cosine similarities with the Assistant Axis on a horizontal scatter.

    Replicates the visualization from the paper's `visualize_axis.ipynb` notebook.
    Traits at the left (red) end are most "role-playing"; traits at the right (blue)
    end are most "assistant-like".

    Args:
        cosine_sims:    1-D array of cosine similarity values, one per trait.
        names:          List of trait names, same order as `cosine_sims`.
        figsize:        Figure size in inches.
        n_extremes:     How many extreme traits to label at each end.
        show_histogram: Whether to overlay a histogram of the distribution.

    Returns:
        The matplotlib Figure object (call `plt.show()` or `fig.savefig(...)` after).

    Example::

        from part4_persona_vectors.utils import plot_similarity_line
        fig = plot_similarity_line(trait_sims_array, trait_names, n_extremes=5)
        plt.title("Trait Vectors vs Assistant Axis (Gemma 2 27B, Layer 22)")
        plt.show()
    """
    projections = cosine_sims

    sorted_indices = np.argsort(projections)
    low_extreme_indices = list(sorted_indices[:n_extremes])
    high_extreme_indices = list(sorted_indices[-n_extremes:])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    custom_cmap = LinearSegmentedColormap.from_list("RedBlue1", ["#e63946", "#457b9d"])

    proj_norm = (projections - projections.min()) / (projections.max() - projections.min() + 1e-8)
    colors = custom_cmap(proj_norm)

    y_pos = np.zeros_like(projections)
    ax.scatter(projections, y_pos, c=colors, marker="D", s=40, alpha=0.6, edgecolors="none", zorder=3)

    if show_histogram:
        hist_counts, bin_edges = np.histogram(projections, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        hist_scale = 0.4
        scaled_heights = hist_counts * hist_scale
        bin_norm = (bin_centers - projections.min()) / (projections.max() - projections.min() + 1e-8)
        bin_colors = custom_cmap(bin_norm)
        ax.bar(bin_centers, scaled_heights, width=bin_width, alpha=0.3, color=bin_colors, edgecolor="none", zorder=1)

    y_above = [0.15, 0.25, 0.35]
    y_below = [-0.15, -0.25, -0.35]

    def _annotate_extreme(idx_list, reverse=False):
        for i, idx in enumerate(idx_list):
            label = names[idx].replace("_", " ").title()
            x_pos = projections[idx]
            i_iter = (len(idx_list) - 1 - i) if reverse else i
            if i_iter % 2 == 0:
                y_label = y_above[i_iter // 2] if i_iter // 2 < len(y_above) else y_above[-1]
                va = "bottom"
                line_end = y_label - 0.02
            else:
                y_label = y_below[i_iter // 2] if i_iter // 2 < len(y_below) else y_below[-1]
                va = "top"
                line_end = y_label + 0.02
            ax.plot(
                [x_pos, x_pos],
                [0.02 if y_label > 0 else -0.02, line_end],
                "-", color="gray", alpha=0.4, linewidth=0.8, zorder=1,
            )
            ax.text(x_pos, y_label, label, ha="center", va=va, fontsize=9, zorder=4)

    _annotate_extreme(low_extreme_indices, reverse=False)
    _annotate_extreme(list(reversed(high_extreme_indices)), reverse=True)

    max_abs = max(abs(projections.min()), abs(projections.max()))
    ax.annotate(
        "Role-playing",
        xy=(-max_abs, -0.45), xytext=(-max_abs + max_abs * 0.25, -0.45),
        arrowprops=dict(arrowstyle="->", color="#e63946", lw=2),
        fontsize=12, fontweight="bold", color="#e63946", ha="left", va="center",
    )
    ax.annotate(
        "Assistant-like",
        xy=(max_abs, -0.45), xytext=(max_abs - max_abs * 0.25, -0.45),
        arrowprops=dict(arrowstyle="->", color="#457b9d", lw=2),
        fontsize=12, fontweight="bold", color="#457b9d", ha="right", va="center",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=2, zorder=1)
    ax.tick_params(axis="x", length=12, width=1.5, pad=10)
    ax.tick_params(axis="y", length=0, width=0)
    ax.set_yticks([])
    ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_ylim(-0.55, 0.5)
    ax.set_xlim(-1, 1)
    ax.grid(False)
    plt.tight_layout()
    return fig
