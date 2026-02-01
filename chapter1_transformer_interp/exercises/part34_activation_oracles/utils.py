"""Utility functions for Activation Oracle exercises."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from IPython.display import HTML, display
from transformers import AutoTokenizer

# Make sure exercises are in the path
if str(exercises_dir := Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(exercises_dir))

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def visualize_token_by_token_responses(
    tokens: list[str],
    responses: list[str | None],
    title: str = "Token-by-Token Oracle Responses",
    max_response_length: int = 100,
) -> HTML:
    """
    Create an HTML visualization of token-by-token oracle responses.

    Args:
        tokens: List of token strings
        responses: List of oracle responses (one per token, can be None)
        title: Title for the visualization
        max_response_length: Maximum length to display for each response

    Returns:
        HTML object that can be displayed in notebook
    """
    html_parts = [
        f"<h3>{title}</h3>",
        "<div style='font-family: monospace; line-height: 1.8;'>",
    ]

    for i, (token, response) in enumerate(zip(tokens, responses)):
        # Escape HTML special characters
        token_display = token.replace("<", "&lt;").replace(">", "&gt;").replace(" ", "·")

        if response and len(response.strip()) > 0:
            response_display = response[:max_response_length]
            if len(response) > max_response_length:
                response_display += "..."
            response_display = response_display.replace("<", "&lt;").replace(">", "&gt;")

            # Color code by position
            color = f"hsl({(i * 360 // len(tokens)) % 360}, 70%, 85%)"

            html_parts.append(
                f"<div style='margin: 5px 0; padding: 5px; background-color: {color}; border-radius: 3px;'>"
                f"<b>Token {i:3d}:</b> <code>{token_display:20s}</code> → {response_display}"
                "</div>"
            )

    html_parts.append("</div>")

    return HTML("\n".join(html_parts))


def create_oracle_response_table(
    prompts: list[str],
    oracle_responses: list[str],
    expected_answers: list[str] | None = None,
    include_accuracy: bool = True,
) -> pd.DataFrame:
    """
    Create a pandas DataFrame table of oracle responses.

    Args:
        prompts: List of prompts
        oracle_responses: List of oracle responses
        expected_answers: Optional list of expected answers
        include_accuracy: Whether to include accuracy column

    Returns:
        DataFrame with results
    """
    data = {
        "Prompt": [p[:80] + "..." if len(p) > 80 else p for p in prompts],
        "Oracle Response": oracle_responses,
    }

    if expected_answers is not None:
        data["Expected"] = expected_answers

        if include_accuracy:
            data["Correct"] = [
                "✓" if expected.lower() in response.lower() else "✗"
                for expected, response in zip(expected_answers, oracle_responses)
            ]

    return pd.DataFrame(data)


def plot_accuracy_heatmap(
    results_df: pd.DataFrame,
    x_col: str = "layer_percent",
    y_col: str = "word",
    value_col: str = "accuracy",
    title: str = "Accuracy by Layer and Word",
) -> go.Figure:
    """
    Create a heatmap visualization of accuracy results.

    Args:
        results_df: DataFrame with results
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        value_col: Column name for values
        title: Plot title

    Returns:
        Plotly figure
    """
    pivot_df = results_df.pivot(index=y_col, columns=x_col, values=value_col)

    fig = px.imshow(
        pivot_df,
        labels=dict(x=x_col.replace("_", " ").title(), y=y_col.replace("_", " ").title(), color=value_col.title()),
        title=title,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        text_auto=".2%",
    )

    fig.update_xaxes(side="bottom")
    fig.update_layout(width=600, height=400)

    return fig


def format_oracle_prompt(layer: int, num_positions: int, question: str) -> str:
    """
    Format a prompt for the oracle with introspection prefix.

    Args:
        layer: Layer number
        num_positions: Number of activation positions
        question: Question to ask

    Returns:
        Formatted prompt string
    """
    special_token = " ?"
    prefix = f"Layer: {layer}\n"
    prefix += special_token * num_positions
    prefix += " \n"
    return prefix + question


def compare_responses_side_by_side(
    target_outputs: list[str],
    oracle_outputs: list[str],
    prompts: list[str] | None = None,
    secret_word: str | None = None,
) -> HTML:
    """
    Create a side-by-side comparison of target vs oracle outputs.

    Args:
        target_outputs: What the target model actually said
        oracle_outputs: What the oracle extracted
        prompts: Optional prompts for context
        secret_word: Optional secret word to highlight

    Returns:
        HTML visualization
    """
    html_parts = [
        "<h3>Target Output vs Oracle Extraction</h3>",
        "<table style='border-collapse: collapse; width: 100%; font-family: system-ui;'>",
        "<thead><tr style='background-color: #f0f0f0;'>",
    ]

    if prompts:
        html_parts.append("<th style='padding: 10px; border: 1px solid #ddd;'>Prompt</th>")

    html_parts.extend(
        [
            "<th style='padding: 10px; border: 1px solid #ddd;'>Target Output</th>",
            "<th style='padding: 10px; border: 1px solid #ddd;'>Oracle Extraction</th>",
            "<th style='padding: 10px; border: 1px solid #ddd;'>Status</th>",
            "</tr></thead><tbody>",
        ]
    )

    for i in range(len(target_outputs)):
        target = target_outputs[i]
        oracle = oracle_outputs[i]

        # Check if oracle extracted the secret
        extracted = secret_word and secret_word.lower() in oracle.lower()
        hidden = secret_word and secret_word.lower() not in target.lower()

        status = "✓ Extracted" if extracted else ("✗ Failed" if secret_word else "—")
        status_color = "#d4edda" if extracted else ("#f8d7da" if secret_word else "#fff")

        html_parts.append(f"<tr style='background-color: {status_color};'>")

        if prompts:
            prompt_display = prompts[i][:50] + "..." if len(prompts[i]) > 50 else prompts[i]
            html_parts.append(f"<td style='padding: 10px; border: 1px solid #ddd;'>{prompt_display}</td>")

        # Highlight secret word if present
        target_display = target
        oracle_display = oracle

        if secret_word:
            import re

            pattern = re.compile(re.escape(secret_word), re.IGNORECASE)
            oracle_display = pattern.sub(
                lambda m: f"<span style='background-color: #ffeb3b; font-weight: bold;'>{m.group()}</span>",
                oracle_display,
            )

        html_parts.extend(
            [
                f"<td style='padding: 10px; border: 1px solid #ddd;'>{target_display}</td>",
                f"<td style='padding: 10px; border: 1px solid #ddd;'>{oracle_display}</td>",
                f"<td style='padding: 10px; border: 1px solid #ddd; text-align: center;'><b>{status}</b></td>",
            ]
        )

        html_parts.append("</tr>")

    html_parts.append("</tbody></table>")

    return HTML("\n".join(html_parts))


def plot_training_metrics(metrics: dict[str, list[float]], title: str = "Training Metrics") -> go.Figure:
    """
    Plot training metrics over time.

    Args:
        metrics: Dict mapping metric name to list of values
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for metric_name, values in metrics.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode="lines",
                name=metric_name,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Loss",
        hovermode="x unified",
        width=800,
        height=400,
    )

    return fig


def display_failure_modes_analysis(test_cases: list[dict[str, Any]]) -> HTML:
    """
    Create a formatted display of oracle failure mode analysis.

    Args:
        test_cases: List of dicts with keys: 'name', 'query', 'response', 'expected_behavior'

    Returns:
        HTML visualization
    """
    html_parts = [
        "<h3>Oracle Failure Modes Analysis</h3>",
        "<div style='font-family: system-ui;'>",
    ]

    for i, case in enumerate(test_cases, 1):
        html_parts.extend(
            [
                f"<div style='margin: 15px 0; padding: 15px; border: 2px solid #ddd; border-radius: 5px;'>",
                f"<h4 style='margin-top: 0;'>{i}. {case['name']}</h4>",
                f"<p><b>Query:</b> {case['query']}</p>",
                f"<p><b>Response:</b> <code>{case['response']}</code></p>",
                f"<p><b>Expected Behavior:</b> {case['expected_behavior']}</p>",
            ]
        )

        if "analysis" in case:
            html_parts.append(f"<p><b>Analysis:</b> <i>{case['analysis']}</i></p>")

        html_parts.append("</div>")

    html_parts.append("</div>")

    return HTML("\n".join(html_parts))


def create_layer_comparison_plot(
    layer_results: dict[int, float],
    metric_name: str = "Accuracy",
    title: str = "Performance by Layer",
) -> go.Figure:
    """
    Create a bar plot comparing performance across layers.

    Args:
        layer_results: Dict mapping layer_percent to metric value
        metric_name: Name of the metric
        title: Plot title

    Returns:
        Plotly figure
    """
    layers = sorted(layer_results.keys())
    values = [layer_results[layer] for layer in layers]

    fig = go.Figure(
        data=[
            go.Bar(
                x=[f"{layer}%" for layer in layers],
                y=values,
                marker_color="steelblue",
                text=[f"{v:.1%}" if v < 1 else f"{v:.2f}" for v in values],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title=metric_name,
        width=600,
        height=400,
    )

    return fig


def summarize_dataset_stats(datapoints: list[Any]) -> pd.DataFrame:
    """
    Create a summary table of dataset statistics.

    Args:
        datapoints: List of TrainingDataPoint objects

    Returns:
        DataFrame with statistics
    """
    stats = {
        "Total Examples": len(datapoints),
        "Avg Input Length": np.mean([len(dp.input_ids) for dp in datapoints]),
        "Avg Response Length": np.mean([sum(1 for x in dp.labels if x != -100) for dp in datapoints]),
        "Avg Num Positions": np.mean([len(dp.positions) for dp in datapoints]),
        "Unique Labels": len(set(dp.ds_label for dp in datapoints if dp.ds_label)),
    }

    return pd.DataFrame([stats]).T.rename(columns={0: "Value"})


# Convenience function for quick HTML display
def show_html(html_content: HTML):
    """Display HTML content in notebook."""
    display(html_content)
