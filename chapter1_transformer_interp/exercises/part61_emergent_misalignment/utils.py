"""Utility functions for Emergent Misalignment exercises."""

import textwrap
import plotly.express as px
import pandas as pd


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
    summary = pd.concat([summary_misalign, summary_coherence], axis=1, keys=["misalignment", "coherence"]).round(
        3
    )
    return summary
