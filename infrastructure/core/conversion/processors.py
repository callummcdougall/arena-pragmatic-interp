"""
Helper functions for processing source code.

This module contains utility functions for processing source code lists, including
removing empty lines, handling "if MAIN" blocks, and de-abbreviating filter names.
"""

from .constants import ALL_FILES, ALL_FILES_ABBR


def _remove_consecutive_empty_lines(source: list[str], max_empty_lines: int = 2) -> list[str]:
    """
    Removes consecutive empty lines from source, keeping at most max_empty_lines.

    Args:
        source: List of source code lines
        max_empty_lines: Maximum number of consecutive empty lines to keep

    Returns:
        Source list with excessive empty lines removed
    """
    new_source = []
    empty_lines = 0
    for line in source:
        empty_lines = 0 if line.strip() != "" else empty_lines + 1
        if empty_lines <= max_empty_lines:
            new_source.append(line)
    return new_source


def _strip_empty_lines_from_start_and_end(source: list[str]) -> list[str]:
    """
    Removes empty lines from the start and end of source.

    Args:
        source: List of source code lines

    Returns:
        Source list with leading/trailing empty lines removed
    """
    while source and not source[0].strip():
        source.pop(0)
    while source and not source[-1].strip():
        source.pop()
    return source


def _strip_out_main_blocks(source: list[str]) -> list[str]:
    """
    Strips out "if MAIN:" blocks by removing the condition and un-indenting the contents.

    Args:
        source: List of source code lines

    Returns:
        Source list with "if MAIN:" blocks removed and contents un-indented
    """
    new_source = []
    in_main_block = False
    for line in source:
        if line.strip() == "if MAIN:":
            in_main_block = True
        elif in_main_block and any(line.startswith(indent) for indent in ["    ", "\t"]):
            new_source.append(line.removeprefix("    " if line.startswith("    ") else "\t"))
        else:
            new_source.append(line)
            in_main_block = False if in_main_block and line.strip() else in_main_block
    return new_source


def _process_source(source: list[str] | None, strip_main_blocks: bool = True) -> list[str] | None:
    """
    The default way we process cell sources before turning them into actual code content.

    Args:
        source: List of source code lines, or None
        strip_main_blocks: Whether to remove "if MAIN:" blocks

    Returns:
        Processed source list, or None if input was None

    Raises:
        AssertionError: If source is empty after processing
    """
    if source is None:
        return None
    assert len(source) > 0, "Found empty cell source (pre-processing)"
    if strip_main_blocks:
        source = _strip_out_main_blocks(source)
    source = _remove_consecutive_empty_lines(_strip_empty_lines_from_start_and_end(source))
    assert len(source) > 0, "Found empty cell source (post-processing)"
    return source


def _de_abbreviate_filters(filters: list[str]) -> list[str]:
    """
    Expands abbreviated filter names to their full forms.

    For example, "colab" expands to ["colab-ex", "colab-soln"], and "st" expands to ["streamlit"].

    Args:
        filters: List of filter names (possibly abbreviated)

    Returns:
        List of full filter names
    """
    abbrev_to_file_list = {
        "": ALL_FILES + ["soln-dropdown"],
        "colab": ["colab-ex", "colab-soln"],
        **{abbr: [full] for abbr, full in zip(ALL_FILES_ABBR, ALL_FILES)},
        **{full: [full] for full in ALL_FILES + ["soln-dropdown"]},
    }
    true_filters = []
    for filter in filters:
        sign = "~" if filter.startswith("~") else ""
        filter = filter.lstrip("~")
        true_filters.extend([sign + x for x in abbrev_to_file_list[filter]])

    return true_filters


# Validation tests
assert _de_abbreviate_filters(["colab", "python"]) == ["colab-ex", "colab-soln", "python"]
assert _de_abbreviate_filters(["colab-soln", "python"]) == ["colab-soln", "python"]
