"""
File generation logic for creating Streamlit application files.

This module handles generating the Python wrapper files that load and display
Streamlit pages from markdown content.
"""

from pathlib import Path


def _create_streamlit_py_file(
    metadata_as_str: str,
    chapter_name: str = "(chapter)",
    chapter_name_long: str = "Chapter - ???",
    section_title: str = "[?.?] ???",
) -> tuple[str, int]:
    """
    Creates the Python file for the Streamlit app using metadata from the first cell.

    The metadata should look like:
        [
            {"title": "TMS", "icon": "1-circle-fill", "subtitle": "(99%)"},
            {"title": "Bonus", "icon": "star", "subtitle": "(1%)"},
        ]

    Args:
        metadata_as_str: String representation of metadata list
        chapter_name: Short chapter name (e.g., "chapter1_transformer_interp")
        chapter_name_long: Long chapter title (e.g., "Chapter 1 - Transformer Interp")
        section_title: Section title (e.g., "[1.3.2] Function Vectors")

    Returns:
        Tuple of (streamlit_file_contents, number_of_sections)

    Raises:
        ValueError: If metadata cannot be parsed
    """
    try:
        metadata = eval(metadata_as_str.strip().removeprefix("```python").removesuffix("```"))
    except Exception as e:
        raise ValueError("""Failed to parse metadata in first cell. Should look like:
[
    {"title": "TMS", "icon": "1-circle-fill", "subtitle": "(99%)"},
    {"title": "Bonus", "icon": "star", "subtitle": "(1%)"},
]""") from e

    streamlit_file_contents = (
        (Path(__file__).parent.parent / "templates" / "streamlit_template.py")
        .read_text(encoding="utf-8")
        .replace("metadata = None", f"metadata = {metadata}")
        .replace("chapter_name = None", f"chapter_name = {chapter_name!r}")
        .replace("chapter_name_long = None", f"chapter_name_long = {chapter_name_long!r}")
        .replace("section_title = None", f"section_title = {section_title!r}")
    )

    return streamlit_file_contents, len(metadata)
