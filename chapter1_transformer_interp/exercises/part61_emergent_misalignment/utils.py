"""Utility functions for Emergent Misalignment exercises."""

import re
import textwrap
from typing import Any

import pandas as pd
import plotly.express as px
from peft import PeftModel
from tabulate import tabulate


def wrap_text_for_hover(text: str, width: int = 80) -> str:
    """
    Wrap text with <br> tags for better hover display in plotly.

    Args:
        text: The text to wrap
        width: Maximum width per line

    Returns:
        Text with <br> tags inserted at appropriate line breaks
    """
    if not text or pd.isna(text):
        return ""
    lines = textwrap.wrap(str(text), width=width)
    return "<br>".join(lines)


def plot_steering_heatmaps(eval_results: pd.DataFrame):
    """
    Create interactive heatmaps showing misalignment and coherence scores.

    Args:
        eval_results: DataFrame with columns: prompt, steering_coef, response,
                     misalignment_score, coherence_score
    """
    # Create separate pivot tables for each metric
    misalignment_pivot = eval_results.pivot_table(
        index="prompt", columns="steering_coef", values="misalignment_score", aggfunc="first"
    )
    coherence_pivot = eval_results.pivot_table(
        index="prompt", columns="steering_coef", values="coherence_score", aggfunc="first"
    )

    # Create pivot table for responses (for hover data) and wrap text
    response_pivot = eval_results.pivot_table(
        index="prompt", columns="steering_coef", values="response", aggfunc="first"
    )
    # Wrap response text for better hover display
    response_pivot_wrapped = response_pivot.applymap(lambda x: wrap_text_for_hover(x, width=80))

    # Create misalignment heatmap (red colorscale)
    fig_misalignment = px.imshow(
        misalignment_pivot,
        color_continuous_scale="Reds",
        labels=dict(x="Steering Coefficient", y="Prompt", color="Misalignment Score"),
        title="Misalignment Scores by Prompt and Steering Coefficient",
        aspect="auto",
        zmin=0.0,
        zmax=1.0,
    )
    # Add response text to hover data
    fig_misalignment.update_traces(
        customdata=response_pivot_wrapped.values,
        hovertemplate="<b>Prompt:</b> %{y}<br>"
        + "<b>Steering Coef:</b> %{x}<br>"
        + "<b>Misalignment:</b> %{z:.3f}<br>"
        + "<b>Response:</b><br>%{customdata}<br>"
        + "<extra></extra>",
    )
    fig_misalignment.update_xaxes(side="bottom")
    fig_misalignment.update_layout(height=400 + 20 * len(misalignment_pivot))
    fig_misalignment.show()

    # Create coherence heatmap (blue colorscale, reversed so blue=1.0 and white=0.0)
    fig_coherence = px.imshow(
        coherence_pivot,
        color_continuous_scale="Blues",
        labels=dict(x="Steering Coefficient", y="Prompt", color="Coherence Score"),
        title="Coherence Scores by Prompt and Steering Coefficient",
        aspect="auto",
        zmin=0.0,
        zmax=1.0,
    )
    # Add response text to hover data
    fig_coherence.update_traces(
        customdata=response_pivot_wrapped.values,
        hovertemplate="<b>Prompt:</b> %{y}<br>"
        + "<b>Steering Coef:</b> %{x}<br>"
        + "<b>Coherence:</b> %{z:.3f}<br>"
        + "<b>Response:</b><br>%{customdata}<br>"
        + "<extra></extra>",
    )
    fig_coherence.update_xaxes(side="bottom")
    fig_coherence.update_layout(height=400 + 20 * len(coherence_pivot))
    fig_coherence.show()

    # Summary statistics
    print("\nSummary by steering coefficient:")
    summary_misalign = eval_results.groupby("steering_coef")["misalignment_score"].agg(["mean", "std"])
    summary_coherence = eval_results.groupby("steering_coef")["coherence_score"].agg(["mean", "std"])
    summary = pd.concat([summary_misalign, summary_coherence], axis=1, keys=["misalignment", "coherence"]).round(3)
    return summary


def _extract_lora_info(model: PeftModel) -> dict[str, Any]:
    """Extract LoRA architecture info from a PeftModel."""
    lora_layers = []
    lora_modules = set()
    total_params = 0
    rank = None
    first_A_shape = None
    first_B_shape = None

    for name, param in model.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            if match := re.search(r"layers\.(\d+)\.", name):
                layer_num = int(match.group(1))
                if layer_num not in lora_layers:
                    lora_layers.append(layer_num)

            if match := re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora", name):
                lora_modules.add(match.group(1))

            if "lora_A" in name and rank is None:
                rank = param.shape[0]
                first_A_shape = tuple(param.shape)

            if "lora_B" in name and first_B_shape is None:
                first_B_shape = tuple(param.shape)

            total_params += param.numel()

    lora_layers.sort()

    return {
        "layers": lora_layers,
        "num_layers": len(lora_layers),
        "total_model_layers": model.config.num_hidden_layers,
        "modules": sorted(lora_modules),
        "rank": rank,
        "A_shape": first_A_shape,
        "B_shape": first_B_shape,
        "total_params": total_params,
    }


def inspect_lora_adapters(low_rank_model: PeftModel, high_rank_model: PeftModel) -> None:
    """
    Inspect and compare two LoRA adapters side by side in a tabulate table.

    Prints a comparison table with columns: Property, Low-Rank LoRA, High-Rank LoRA.
    Properties shown: layers, modules, rank, A and B shapes, total trainable params.
    """
    low = _extract_lora_info(low_rank_model)
    high = _extract_lora_info(high_rank_model)

    def fmt_layers(info: dict[str, Any]) -> str:
        if info["num_layers"] == info["total_model_layers"]:
            return f"All {info['total_model_layers']}"
        return f"{info['num_layers']} layers: {info['layers']}"

    rows = [
        ["Layers", fmt_layers(low), fmt_layers(high)],
        ["Modules", ", ".join(low["modules"]), ", ".join(high["modules"])],
        ["Rank", low["rank"], high["rank"]],
        ["A shape", str(low["A_shape"]), str(high["A_shape"])],
        ["B shape", str(low["B_shape"]), str(high["B_shape"])],
        ["Total trainable params", f"{low['total_params']:,}", f"{high['total_params']:,}"],
    ]

    print(tabulate(rows, headers=["Property", "Low-Rank LoRA", "High-Rank LoRA"], tablefmt="simple_outline"))
