from pathlib import Path

from arena_material_conversion import MasterFileData
from IPython import get_ipython
from IPython.display import HTML, display

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# You should edit the contents of this dictionary, then run the rest of the code

ALL_FILENAMES = {
    "1.1": ("01_[1.1]_Transformer_from_Scratch", "part1_transformer_from_scratch"),
    "1.2": ("02_[1.2]_Intro_to_Mech_Interp", "part2_intro_to_mech_interp"),
    "1.3.1": ("11_[1.3.1]_Probing_for_Deception", "part31_probing_for_deception"),
    "1.3.2": (
        "12_[1.3.2]_Function_Vectors_&_Model_Steering",
        "part32_function_vectors_and_model_steering",
    ),
    "1.3.3": ("13_[1.3.3]_Interpretability_with_SAEs", "part33_interp_with_saes"),
    "1.4.1": (
        "21_[1.4.1]_Indirect_Object_Identification",
        "part41_indirect_object_identification",
    ),
    "1.4.2": ("22_[1.4.2]_SAE_Circuits", "part42_sae_circuits"),
    "1.5.1": ("31_[1.5.1]_Balanced_Bracket_Classifier", "part51_balanced_bracket_classifier"),
    "1.5.2": (
        "32_[1.5.2]_Grokking_&_Modular_Arithmetic",
        "part52_grokking_and_modular_arithmetic",
    ),
    "1.5.3": ("33_[1.5.3]_OthelloGPT", "part53_othellogpt"),
    "1.5.4": ("34_[1.5.4]_Toy_Models_of_Superposition_&_SAEs", "part54_toy_models_of_superposition_and_saes"),
    "1.6.1": ("41_[1.6.1]_Emergent_Misalignment", "part61_emergent_misalignment"),
    "1.6.2": ("42_[1.6.2]_Science_of_Misalignment", "part62_science_of_misalignment"),
    "1.6.3": ("43_[1.6.3]_Eliciting_Secret_Knowledge", "part63_eliciting_secret_knowledge"),
    "1.6.4": ("44_[1.6.4]_Interpreting_Reasoning_Models", "part64_interpreting_reasoning_models"),
}

# FILES = ALL_FILENAMES.keys()
# FILES = [x for x in ALL_FILENAMES.keys() if x[0] != "0"]
# FILES = [x for x in ALL_FILENAMES.keys() if x.split(".")[1] == "6"]
FILES = [x for x in ALL_FILENAMES.keys() if x.startswith("1.6") or x == "1.3.1"]
# FILES = [x for x in ALL_FILENAMES.keys()]
# FILES = [x for x in ALL_FILENAMES.keys() if x[0]=="3"]  # , "3.2", "3.3", "3.4"]
# FILES = ["2.1", "2.2.1", "2.2.2", "2.3", "2.4"]


for FILE in FILES:
    display(HTML(f"<h1>{FILE}</h1>"))
    chapter_name = {
        0: "chapter0_fundamentals",
        1: "chapter1_transformer_interp",
        2: "chapter2_rl",
        3: "chapter3_llm_evals",
    }[int(FILE.split(".")[0])]
    chapter_name_long = {
        0: "Chapter 0 - Fundamentals",
        1: "Chapter 1 - Transformer Interp",
        2: "Chapter 2 - Reinforcement Learning",
        3: "Chapter 3 - LLM Evaluations",
    }[int(FILE.split(".")[0])]

    master_path = Path(__file__).parent / f"master_{FILE.replace('.', '_')}.ipynb"
    chapter_dir = master_path.parent.parent.parent / chapter_name

    assert master_path.exists() and chapter_dir.exists()

    streamlit_page_name, exercise_dir_name = ALL_FILENAMES[FILE]

    master = MasterFileData(
        master_path=master_path,
        chapter_dir=chapter_dir,
        chapter_name_long=chapter_name_long,
        exercise_dir_name=exercise_dir_name,
        streamlit_page_name=streamlit_page_name,
    )
    master.master_ipynb_to_py(overwrite=True)
    master.generate_files(overwrite=True, verbose=True)
    # master.master_py_to_ipynb(overwrite=True)

