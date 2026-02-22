"""Utility functions for probing exercises: per-token deception score visualization."""

import torch
from circuitsvis.tokens import colored_tokens
from IPython.display import HTML, display
from jaxtyping import Float
from torch import Tensor


def process_str_tokens(str_tokens: list[str]) -> list[str]:
    """Sanitize special tokens so they render as text rather than HTML tags."""
    replacements = {
        "<bos>": "[bos]",
        "<s>": "[s]",
        "</s>": "[/s]",
        "<start_of_turn>": "[start_of_turn]",
        "<end_of_turn>": "[end_of_turn]",
        "\n": "\\n",
    }
    for k, v in replacements.items():
        str_tokens = [tok.replace(k, v) for tok in str_tokens]
    return str_tokens


def visualize_token_scores(
    str_tokens: list[str],
    scores: list[float] | Float[Tensor, " seq"],
    title: str = "",
    label: str = "",
    score_value: float | None = None,
    min_value: float = -6.0,
    max_value: float = 6.0,
    positive_color: str = "red",
    negative_color: str = "blue",
    show: bool = True,
) -> str:
    """
    Render per-token colored scores using circuitsvis.

    Args:
        str_tokens: List of string tokens from the tokenizer.
        scores: Per-token scores (same length as str_tokens). Higher = more deceptive.
        title: Optional title displayed above the visualization.
        label: Optional label (e.g. "Honest", "Deceptive") shown next to the score.
        score_value: Optional aggregate score to display in the header.
        min_value: Lower bound of the color scale.
        max_value: Upper bound of the color scale.
        positive_color: Color for high scores (default red = deceptive).
        negative_color: Color for low scores (default blue = honest).
        show: If True, display the HTML inline via IPython.display.

    Returns:
        The HTML string for the visualization.
    """
    if isinstance(scores, Tensor):
        scores = scores.cpu().tolist()

    assert len(str_tokens) == len(scores), f"Token count ({len(str_tokens)}) != score count ({len(scores)})"

    tokens_clean = process_str_tokens(str_tokens)
    scores_rounded = [round(s, 3) for s in scores]

    cv_html = (
        str(
            colored_tokens(
                tokens_clean,
                scores_rounded,
                min_value=min_value,
                max_value=max_value,
                positive_color=positive_color,
                negative_color=negative_color,
            )
        )
        + "</div>"
    )

    # Build header
    header_parts = []
    if title:
        header_parts.append(f"<h3 style='font-family: Arial, sans-serif;'>{title}</h3>")
    if score_value is not None or label:
        subtitle_items = []
        if score_value is not None:
            subtitle_items.append(f"Score: {score_value:.2f}")
        if label:
            label_color = {"Honest": "#4c72b0", "Deceptive": "#c44e52"}.get(label, "#666")
            subtitle_items.append(f"Label: <span style='color: {label_color}; font-weight: bold;'>{label}</span>")
        header_parts.append(
            f"<p style='font-family: Arial, sans-serif; font-size: 14px;'>{', '.join(subtitle_items)}</p>"
        )

    html = "\n".join(header_parts) + cv_html

    if show:
        display(HTML(html))

    return html


def score_tokens_with_probe(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    probe_direction: Float[Tensor, " d"],
    layer: int,
    scaler=None,
) -> tuple[list[str], Float[Tensor, " seq"]]:
    """
    Tokenize text, extract activations at a given layer, and score each token with a probe direction.

    Args:
        text: The full text to tokenize and score.
        model: The language model.
        tokenizer: The tokenizer.
        probe_direction: The probe direction vector (d_model,).
        layer: Which layer to extract activations from.
        scaler: Optional sklearn StandardScaler (for LR probes trained with scaling).

    Returns:
        Tuple of (str_tokens, per_token_scores).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer + 1]  # [1, seq, d_model]
    acts = hidden[0].cpu().float()  # [seq, d_model]

    if scaler is not None:
        acts_scaled = torch.tensor(scaler.transform(acts.numpy()), dtype=torch.float32)
        scores = acts_scaled @ torch.tensor(probe_direction, dtype=torch.float32)
    else:
        scores = acts @ probe_direction.cpu().float()

    str_tokens = [tokenizer.decode(tok_id) for tok_id in inputs["input_ids"][0]]

    return str_tokens, scores


def logit_lens_on_direction(
    direction: Float[Tensor, " d"],
    model: torch.nn.Module,
    tokenizer,
    top_k: int = 15,
    label: str = "probe direction",
) -> None:
    """
    Project a probe direction through the model's unembedding (lm_head) to see what tokens
    are most/least predicted by this direction.

    Args:
        direction: The probe direction vector.
        model: The model (must have model.lm_head).
        tokenizer: The tokenizer for decoding token IDs.
        top_k: Number of top/bottom tokens to display.
        label: Label for the direction (for printing).
    """
    with torch.no_grad():
        direction_device = direction.to(model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
        logits = model.lm_head(direction_device)

    top_values, top_indices = logits.float().topk(top_k)
    bottom_values, bottom_indices = logits.float().topk(top_k, largest=False)

    top_tokens = [tokenizer.decode(idx.item()) for idx in top_indices]
    bottom_tokens = [tokenizer.decode(idx.item()) for idx in bottom_indices]

    print(f"\nLogit lens on {label}:")
    print(f"  Top-{top_k} tokens (positive end): {top_tokens}")
    print(f"  Bottom-{top_k} tokens (negative end): {bottom_tokens}")

    # Also print with values for more detail
    print("\n  Top tokens with logit values:")
    for tok, val in zip(top_tokens[:10], top_values[:10]):
        print(f"    {val.item():+.3f}  {repr(tok)}")
    print("  Bottom tokens with logit values:")
    for tok, val in zip(bottom_tokens[:10], bottom_values[:10]):
        print(f"    {val.item():+.3f}  {repr(tok)}")


def get_assistant_token_mask(messages, tokenizer):
    """Boolean mask: True for assistant-response tokens."""
    prefix_msgs = [m for m in messages if m["role"] != "assistant"]
    prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prefix_len = len(tokenizer(prefix_text)["input_ids"])
    full_len = len(tokenizer(full_text)["input_ids"])
    mask = torch.zeros(full_len, dtype=torch.bool)
    mask[prefix_len:] = True
    return mask
