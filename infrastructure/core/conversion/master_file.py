"""
Main orchestration class for master file conversion.

This module contains the MasterFileData class which coordinates the conversion
of master notebooks to various output formats (Colab notebooks, Streamlit pages,
Python solutions files).
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

from .cell import Cell
from .constants import ALL_FILES, ALL_FILES_ABBR, ARENA_ROOT
from .converters import (
    _cells_to_notebook_data,
    _convert_master_ipynb_cell_to_master_py_cell_data,
    _split_into_cells,
)
from .generators import _create_streamlit_py_file
from .processors import _remove_consecutive_empty_lines


class MasterFileData:
    """
    Contains all the data needed to create the ARENA files from a master notebook.

    Coordinates the conversion process from master.ipynb to multiple output formats:
    - Colab exercise and solution notebooks
    - Streamlit markdown and Python files
    - Python solutions file
    """

    master_path: Path  # where master.ipynb is located
    chapter_dir: Path  # where the chapter directory is located, e.g. `chapter1_transformer_interp`

    chapter_name_long: str  # full name for Streamlit page title, e.g. "Chapter 0 - Fundamentals"
    exercise_dir_name: str  # what the directory is called in the `exercises` folder
    streamlit_page_name: str  # what the page is called in the `instructions` folder

    colab_soln_cells: list[dict]
    colab_ex_cells: list[dict]
    solutions_py_file: list[str]
    streamlit_md_file: list[str]
    streamlit_py_file: str

    @property
    def exercises_dir(self) -> Path:
        return self.chapter_dir / "exercises" / self.exercise_dir_name

    @property
    def instructions_dir(self) -> Path:
        return self.chapter_dir / "instructions" / "pages"

    @property
    def colab_name(self) -> str:
        return self.streamlit_page_name.split("_[")[-1].replace("]", "")

    @property
    def section_title(self) -> str:
        return "[" + self.streamlit_page_name.split("_[")[-1].replace("_", " ")

    @property
    def files(self) -> dict[str, Any]:
        return {
            self.exercises_dir / "solutions.py": "".join(self.solutions_py_file),
            self.instructions_dir / f"{self.streamlit_page_name}.md": "".join(
                self.streamlit_md_file
            ),
            self.instructions_dir / f"{self.streamlit_page_name}.py": self.streamlit_py_file,
            self.exercises_dir / f"{self.colab_name}_exercises.ipynb": self.colab_ex_cells,
            self.exercises_dir / f"{self.colab_name}_solutions.ipynb": self.colab_soln_cells,
        }

    def __init__(
        self,
        master_path: Path,
        chapter_dir: Path,
        chapter_name_long: str,
        exercise_dir_name: str,
        streamlit_page_name: str,
    ):
        assert master_path.exists(), "Master path does not exist"
        assert master_path.suffix == ".ipynb", "Master path should be an ipynb file"
        self.slack_link = (
            "https://join.slack.com/t/arena-uk/shared_invite/zt-3afdmdhye-Mdb3Sv~ss_V_mEaXEbkABA"
        )
        self.master_path = master_path
        self.chapter_dir = chapter_dir
        self.chapter_name_long = chapter_name_long
        self.exercise_dir_name = exercise_dir_name
        self.streamlit_page_name = streamlit_page_name

        self.colab_ex_cells = []
        self.colab_soln_cells = []
        self.solutions_py_file = ["# %%\n\n"]
        self.streamlit_md_file = []
        self.streamlit_py_file = []

    def generate_files(self, overwrite: bool = False, verbose: bool = False, ruff_format: bool = False) -> None:
        assert self.master_py_path.exists(), (
            "Need to create `master.py` first, using `master_py_to_ipynb`"
        )

        self.create_arena_files(verbose)
        for filename, content in self.files.items():
            # Save files
            if (path := self.chapter_dir / filename).exists() and not overwrite:
                raise FileExistsError(f"File {path} already exists")
            path.write_text(content, encoding="utf-8")

            # Format files with ruff, and log the effect of the formatting
            if path.suffix == ".py" and ruff_format:
                len_old = path.read_text(encoding="utf-8").count("\n")
                path_str = str(path).replace("\\", "/")
                result = subprocess.run(
                    ["ruff", "format", path_str], capture_output=True, text=True, shell=True
                )
                assert result.returncode == 0, (
                    f"Failed to format {path.name!r}, check file to see if it has any errors (also have you pip installed ruff?)"
                )
                len_new = path.read_text(encoding="utf-8").count("\n")
                print(
                    f"Successfully formatted {path.name!r}, length went from {len_old} ➔ {len_new}"
                )

    @property
    def master_py_path(self) -> Path:
        return self.master_path.with_suffix(".py")

    @property
    def cells(self) -> list[Cell]:
        assert self.master_py_path.exists(), (
            "Need to create `master.py` first, using `master_ipynb_to_py`"
        )
        lines = self.master_py_path.read_text(encoding="utf-8").splitlines()
        return _split_into_cells(lines)

    def master_py_to_ipynb(self, overwrite: bool = False, ignore_errors: bool = False) -> None:
        if self.master_path.exists() and not overwrite:
            raise FileExistsError(f"File {self.master_path} already exists")
        self.master_path.write_text(_cells_to_notebook_data(self.cells), encoding="utf-8")

    def master_ipynb_to_py(self, overwrite: bool = False, ignore_errors: bool = False) -> None:
        """
        Convert ipynb file to py file. Most of the logic here involves making sure tags are
        correctly parsed, and appear outside the cell (e.g. rather than inside the triple-quoted
        markdown cells).
        """

        assert self.master_path.exists(), "Need to create `master.ipynb` first"
        if self.master_py_path.exists() and not overwrite:
            raise FileExistsError(f"File {self.master_py_path} already exists")

        content = json.loads(self.master_path.read_text(encoding="utf-8"))

        python_script = ""
        for cell in content["cells"]:
            cell_type, tags, filters, source = _convert_master_ipynb_cell_to_master_py_cell_data(
                cell
            )
            content_str = "".join(source) + "\n\n"
            if cell_type == "markdown":
                content_str = "r'''\n" + content_str.strip() + "\n'''\n\n"
            python_script += f"""# ! CELL TYPE: {cell_type}
# ! FILTERS: [{",".join(filters)}]
# ! TAGS: [{",".join(tags)}]
\n{content_str}"""

        self.master_py_path.write_text(python_script, encoding="utf-8")
        print(f"Wrote to {self.master_py_path!r}")

    def create_arena_files(self, verbose: bool = False) -> None:
        """
        Creates the 4 files needed for ARENA from `master.py`. This is done by splitting the contents of `master.py`
        into separate Cell objects, and processing each of them & optionally adding them to each of the 4 main files
        we'll be creating: 2 Colabs, a Streamlit markdown file, and a Python file containing the solutions.
        """
        assert self.master_py_path.exists(), "Need master.py first, using master_py_to_ipynb"
        text = self.master_py_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        cells = _split_into_cells(lines)

        status = {
            "chapter-stage": None,  # or "title", "intro", "content"
            "soln-dropdown": None,  # either None or list of strings (for soln to previous exercise, that'll go in dropdown)
            "logs": [],  # for verbose print messages (helps with debugging and stuff)
            "prev-was-code": False,  # whether last cell was code cell (used for deciding where to put HTML output)
        }
        update_from_header = False
        learning_objectives = None

        for i, cell in enumerate(cells):
            # Handle 2 special cases: this cell is a section header, or it's the cell containing all learning objectives
            update_from_header = cell.update_from_header(status, verbose)
            if cell.learning_objectives:
                learning_objectives = cell.learning_objectives

            # First cell is special: it gives us the contents page for streamlit
            if i == 0:
                self.streamlit_py_file, n_sections = _create_streamlit_py_file(
                    cell.content_str,
                    self.chapter_dir.name,
                    self.chapter_name_long,
                    self.section_title,
                )

            else:
                # Get the cell contents for each of the 4 files, plus dropdown
                file_contents = cell.process_cell(status)

                # Our `file_contents` might contain HTML output which needs to be absorbed into the last cell
                for key, file_list in [
                    ("colab-ex", self.colab_ex_cells),
                    ("colab-soln", self.colab_soln_cells),
                ]:
                    prev_html = file_contents.pop(f"{key}-prev-html", None)
                    if prev_html:
                        file_list[-1]["outputs"] = [prev_html]

                # Add all 4 files contents to the respective file contents container
                for key, file_list in [
                    ("colab-ex", self.colab_ex_cells),
                    ("colab-soln", self.colab_soln_cells),
                    ("streamlit", self.streamlit_md_file),
                    ("python", self.solutions_py_file),
                ]:
                    if file_contents[key] is not None:
                        file_list.extend([fc for fc in file_contents[key] if fc is not None])

                # Behaviour if we have a title (this is special)
                if update_from_header:
                    # If this is the main header, then add the page links below it
                    if status["chapter-stage"] == "title":
                        st_lines, colab_lines = self.get_page_links()
                        self.streamlit_md_file.extend([*st_lines, ""])
                        self.colab_ex_cells[-1]["source"].extend(["", *colab_lines])
                        self.colab_soln_cells[-1]["source"].extend(["", *colab_lines])

                    # If this is a chapter header...
                    if isinstance(status["chapter-stage"], int):
                        # ...add the chapter separator (before the new title)
                        self.streamlit_md_file.insert(-2, "\n\n=== NEW CHAPTER ===\n\n")
                        # and add the learning objectives (after the new title), if we have them.
                        if learning_objectives and (
                            objs := learning_objectives.pop(status["chapter-stage"], [])
                        ):
                            self.streamlit_md_file.extend(["", *objs, ""])
                            self.colab_ex_cells[-1]["source"].extend(["", *objs])
                            self.colab_soln_cells[-1]["source"].extend(["", *objs])

            first_line = cell.content_str.replace("\n", "\\n")
            first_line = first_line if len(first_line) < 50 else first_line[:60] + " ..."
            matching_files = [
                "y" if f in cell.filters_to_matching_files() else "" for f in ALL_FILES
            ]
            status["logs"].append((cell.cell_type, *matching_files, first_line))

        # Check headers in metadata cell at the top matches the number of sections
        assert n_sections == status["chapter-stage"], (
            f"Expected {n_sections} sections based on header metadata, but found {status['chapter-stage']} when parsing the actual content"
        )

        print(
            tabulate(
                status["logs"],
                headers=["Type", *ALL_FILES_ABBR, "First Line"],
                tablefmt="simple_outline",
            )
        )

        self.colab_ex_cells = _cells_to_notebook_data(self.colab_ex_cells)
        self.colab_soln_cells = _cells_to_notebook_data(self.colab_soln_cells)
        self.streamlit_md_file = "\n".join(_remove_consecutive_empty_lines(self.streamlit_md_file))
        self.solutions_py_file = "\n".join(_remove_consecutive_empty_lines(self.solutions_py_file))

    def get_page_links(self) -> tuple[list[str], list[str]]:
        solutions_link = f"{ARENA_ROOT}{self.chapter_dir.name}/exercises/{self.exercise_dir_name}/{self.colab_name}_solutions.ipynb"
        exercises_link = f"{ARENA_ROOT}{self.chapter_dir.name}/exercises/{self.exercise_dir_name}/{self.colab_name}_exercises.ipynb"
        chapter_name_in_url = self.chapter_dir.name.replace("_", "-")
        time_suffix = "?t=" + datetime.now().strftime("%Y%m%d")

        colab_lines = [
            f"> **ARENA [Streamlit Page](https://arena-{chapter_name_in_url}.streamlit.app/{self.streamlit_page_name})**",
            ">",
            f"> **Colab: [exercises]({exercises_link}{time_suffix}) | [solutions]({solutions_link}{time_suffix})**",
            "",
            f"Please send any problems / bugs on the `#errata` channel in the [Slack group]({self.slack_link}), and ask any questions on the dedicated channels for this chapter of material.",
            "",
            "You can collapse each section so only the headers are visible, by clicking the arrow symbol on the left hand side of the markdown header cells.",
            "",
            "Links to all other chapters: [(0) Fundamentals](https://arena-chapter0-fundamentals.streamlit.app/), [(1) Transformer Interpretability](https://arena-chapter1-transformer-interp.streamlit.app/), [(2) RL](https://arena-chapter2-rl.streamlit.app/).",
        ]
        streamlit_lines = [
            f"> **Colab: [exercises]({exercises_link}{time_suffix}) | [solutions]({solutions_link}{time_suffix})**",
            "",
            f"Please send any problems / bugs on the `#errata` channel in the [Slack group]({self.slack_link}), and ask any questions on the dedicated channels for this chapter of material.",
            "",
            "If you want to change to dark mode, you can do this by clicking the three horizontal lines in the top-right, then navigating to Settings → Theme.",
            "",
            "Links to all other chapters: [(0) Fundamentals](https://arena-chapter0-fundamentals.streamlit.app/), [(1) Transformer Interpretability](https://arena-chapter1-transformer-interp.streamlit.app/), [(2) RL](https://arena-chapter2-rl.streamlit.app/).",
        ]
        return streamlit_lines, colab_lines
