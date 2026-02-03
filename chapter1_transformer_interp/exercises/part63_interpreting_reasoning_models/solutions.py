# %%


import json
import os
import re
import sys
import textwrap
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import sentence_transformers
import torch
import transformers
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from IPython.display import display
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

warnings.filterwarnings("ignore")


thought_anchors_path = Path.cwd().parent / "thought-anchors"
assert thought_anchors_path.exists()

sys.path.append(str(thought_anchors_path))

from utils import (
    check_answer,
    # split_solution_into_chunks,
    extract_boxed_answers,
    load_math_problems,
    normalize_answer,
)

# %%

# TODO - actually use these! or delete them

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATASET_NAME = "uzaymacar/math-rollouts"  # Pre-computed rollouts from paper
SIMILARITY_THRESHOLD = 0.8  # Median threshold from paper
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence embedding model used in paper
N_ROLLOUTS = 100  # Number of rollouts per sentence

# # Paths (adjust these to match your setup)
# ROLLOUTS_DIR = Path("math_rollouts")  # Directory with generated rollouts
# ANALYSIS_DIR = Path("analysis")  # Directory to save analysis results
# ANALYSIS_DIR.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print(embedding_model)

# %%

prompts = [
    "Wait, I think I made an error in my reasoning and need to backtrack",
    "Hold on, I believe I made a mistake in my logic and should reconsider",
    "After careful analysis, I've determined the correct answer is 42",
    "Time is an illusion. Lunchtime doubly so.",
]
labels = [x[:35] + "..." for x in prompts]

embedding = embedding_model.encode(prompts)

cosine_sims = embedding @ embedding.T

px.imshow(
    cosine_sims,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    labels=dict(x="Prompt", y="Prompt", color="Cosine Similarity"),
    x=labels,
    y=labels,
)

# %%

# dataset = load_dataset(DATASET_NAME, split="default", streaming=True)

PROBLEM_ID = 4682

path = f"deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/correct_base_solution/problem_{PROBLEM_ID}/base_solution.json"

local_path = hf_hub_download(repo_id=DATASET_NAME, filename=path, repo_type="dataset")

with open(local_path, "r") as f:
    problem_data = json.load(f)

problem_data

# %%

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,  # Use bfloat16 for efficiency
    device_map="auto",  # Automatically distribute across available GPUs
    trust_remote_code=True,
)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Model loaded on: {model.device}")
print(f"Model has {model.config.num_hidden_layers} layers")
print(f"Model has {model.config.num_attention_heads} attention heads per layer")

# %%

problem_data["chunk_solutions"][10][0].keys()

# %%

easy_problem_text = "A rectangle has a length of 8 cm and a width of 5 cm. What is its perimeter?"

easy_prompt = f"""Solve this math problem step by step.

Problem: {easy_problem_text}

Solution:
<think>"""

inputs = tokenizer(easy_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False)
easy_problem_full_text = easy_prompt + generated_text

print(easy_problem_full_text)

# %%

class StopOnThink(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        # YOUR CODE HERE: return True iff the model has generated "</think>"
        think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        return input_ids[0, -1] == think_token_id


with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        stopping_criteria=[StopOnThink()],
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False)
easy_problem_full_text = easy_prompt + generated_text

print(easy_problem_full_text)

# %%

def split_solution_into_chunks(text: str) -> list[str]:
    """Split solution into sentence-level chunks."""

    # YOUR CODE HERE - fill in the rest of the function

    # Remove thinking tags
    if "<think>" in text:
        text = text.split("<think>")[1]
    if "</think>" in text:
        text = text.split("</think>")[0]
    text = text.strip()

    # Replace the "." characters which I don't want to split on
    text = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL>\2", text)  # e.g. "4.5" -> "4<DECIMAL>5"
    text = re.sub(r"\n(\d)\.(\s)", r"\n\1<DECIMAL>\2", text)  # e.g. "1.\n" -> "\n1<DECIMAL>"

    # Split on sentence endings, and combine the endings with the previous chunk
    sentences = re.split(r"([!?:\n]|(?<!\n\d)\.)", text)
    chunks = []
    for i in range(0, len(sentences) - 1, 2):
        chunks.append((sentences[i] + sentences[i + 1]).replace("\n", " "))

    # Replace <DECIMAL> back with "."
    chunks = [re.sub(r"<DECIMAL>", ".", c) for c in chunks]

    # Merge chunks that are too short
    while len(chunks) > 1 and min([len(x) for x in chunks[:-1]]) < 10:
        for i, chunk in enumerate(chunks[:-1]):
            if len(chunk) < 10:
                chunks = chunks[:i] + [chunk + chunks[i + 1]] + chunks[i + 2 :]
                break

    return [c.strip() for c in chunks if c.strip()]


test_cases = [
    # (input_text, expected_chunks)
    (
        "<think>First, I understand the problem. Next, I'll solve for x. Finally, I verify!</think>",
        ["First, I understand the problem.", "Next, I'll solve for x.", "Finally, I verify!"],
    ),
    (
        "<think>Let me break this down: 1. Convert to decimal. 2. Calculate log. 3. Apply formula.</think>",
        [
            "Let me break this down:",
            "1. Convert to decimal.",
            "2. Calculate log.",
            "3. Apply formula.",
        ],
    ),
    (
        "<think>The formula is A = πr². Wait. No. Actually, it's different.</think>",
        ["The formula is A = πr².", "Wait. No.", "Actually, it's different."],
    ),
    (
        "<think>Convert 66666₁₆ to decimal. This equals 419,430. How many bits? We need log₂(419,430) ≈ 18.7. So 19 bits!</think>",
        [
            "Convert 66666₁₆ to decimal.",
            "This equals 419,430.",
            "How many bits?",
            "We need log₂(419,430) ≈ 18.7.",
            "So 19 bits!",
        ],
    ),
    ("<think>The answer is 42. Done.</think>", ["The answer is 42.", "Done."]),
]

for input_text, expected_chunks in test_cases:
    chunks = split_solution_into_chunks(input_text)
    assert chunks == expected_chunks, f"Expected {expected_chunks}, got {chunks}"

# %%

CATEGORIES = {
    "problem_setup": "Problem Setup",
    "plan_generation": "Plan Generation",
    "fact_retrieval": "Fact Retrieval",
    "active_computation": "Active Computation",
    "uncertainty_management": "Uncertainty Management",
    "result_consolidation": "Result Consolidation",
    "self_checking": "Self Checking",
    "final_answer_emission": "Final Answer Emission",
    "unknown": "Unknown",
}

CATEGORY_WORDS = {
    # Note: we put the most definitive categories first, so they override the later ones
    "final_answer_emission": ["\\boxed", "final answer"],
    "problem_setup": ["need to", "problem is", "given"],
    "fact_retrieval": ["remember", "formula", "know that", "recall"],
    "active_computation": ["calculate", "compute", "solve", "=", "equals", "result", "giving"],
    "uncertainty_management": ["wait", "let me", "double check", "hmm", "actually", "reconsider"],
    "result_consolidation": ["summarize", "so", "therefore", "in summary"],
    "self_checking": ["verify", "check", "confirm", "correct"],
    "plan_generation": ["plan", "approach", "strategy", "will", "i'll", "try"],
}


def categorize_sentences_heuristic(chunks: list[str]) -> list[str]:
    """
    Categorize sentences using heuristics/keyword matching (simplified version).

    For full LLM-based labeling, see prompts.py DAG_PROMPT in the thought-anchors repo.

    Args:
        sentences: List of sentence strings

    Returns:
        List of tags.
    """

    categories = []

    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()

        if idx == 0:
            tag = "problem_setup"
        else:
            for category, words in CATEGORY_WORDS.items():
                if any(word in chunk_lower for word in words):
                    tag = category
                    break
            else:
                tag = "unknown"

        categories.append(CATEGORIES.get(tag))

    return categories


example_problem_text = "What is the area of a circle with radius 5?"

example_sentences, example_categories = list(
    zip(
        *[
            ("I need to find the area of a circle with radius 5.", "Problem Setup"),
            ("The formula for circle area is A = πr².", "Fact Retrieval"),
            ("Substituting r = 5: A = π × 5² = 25π.", "Active Computation"),
            ("Wait, let me look again at that calculation.", "Uncertainty Management"),
            ("So the area is 25π square units.", "Result Consolidation"),
            ("Therefore, the answer is \\boxed{25π}.", "Final Answer Emission"),
        ]
    )
)

categories = categorize_sentences_heuristic(example_sentences)

for sentence, category, expected_category in zip(example_sentences, categories, example_categories):
    assert category == expected_category, f"Expected {expected_category!r}, got {category!r} for sentence: {sentence!r}"

# %%

easy_chunks = split_solution_into_chunks(easy_problem_full_text)

easy_categories = categorize_sentences_heuristic(easy_chunks)

for chunk, category in zip(easy_chunks, easy_categories):
    chunk_str = chunk if len(chunk) < 80 else chunk[:60] + " ... " + chunk[-20:]
    print(f"{category:>20} | {chunk_str!r}")

# %%

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def generate_api_response(
    model: str = "gpt-4.1-mini",
    messages: list[dict[str, str]] = [],
    max_tokens: int = 128,
    stop_sequences: list[str] | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Helper function with retry logic and error handling."""

    for attempt in range(max_retries):
        try:
            # Generate response
            resp = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences if stop_sequences else None,
            )
            # Extract text from response
            return resp.choices[0].message.content or ""

        except Exception as e:
            if "rate_limit_error" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds (1, 2, 4, 8, 16...)
                    wait_time = 2**attempt
                    print(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Failed to get response after {max_retries} attempts, returning early.")
            raise e


resp = generate_api_response(messages=[{"role": "user", "content": "Who are you?"}])

print(textwrap.fill(resp))

# %%

DAG_SYSTEM_PROMPT = """
You are an expert in interpreting how language models solve math problems using multi-step reasoning. Your task is to analyze a Chain-of-Thought (CoT) reasoning trace, broken into discrete text chunks, and label each chunk with a tag that describes what this chunk is *doing* functionally in the reasoning process.

---

### Function Tags:

1. `problem_setup`: 
    Parsing or rephrasing the problem (initial reading or comprehension).
    
2. `plan_generation`: 
    Stating or deciding on a plan of action, or on the next step in the reasoning process.
    
3. `fact_retrieval`: 
    Recalling facts, formulas, problem details (without immediate computation).
    
4. `active_computation`: 
    Performing algebra, calculations, manipulations toward the answer.
    This only includes actual calculations being done & values computed, not stating formulas or plans.
    
5. `result_consolidation`: 
    Aggregating intermediate results, summarizing, or preparing final answer.
    
6. `uncertainty_management`: 
    Expressing confusion, re-evaluating, proposing alternative plans (includes backtracking).
    
7. `final_answer_emission`: 
    Explicit statement of the final boxed answer or earlier chunks that contain the final answer.
    
8. `self_checking`: 
    Verifying previous steps, Pythagorean checking, re-confirmations.
    This is at the object-level, whereas "uncertainty_management" is at the planning level.

9. `unknown`: 
    Use only if the chunk does not fit any of the above tags or is purely stylistic or semantic.

---

### Output Format:

Return a numbered list, one item for each chunk, consisting of the function tag that best describes the chunk.

For example:

1. problem_setup
...
5. final_answer_emission
"""

# %%

def categorize_sentences_autorater(problem_text: str, chunks: list[str]) -> list[str]:
    """
    Categorize sentences using heuristics/keyword matching (simplified version).

    Args:
        sentences: List of sentence strings

    Returns:
        List of categories.
    """

    # YOUR CODE HERE

    chunk_str = ""
    for i, chunk in enumerate(chunks):
        chunk_str += f"{i + 1}. {chunk}\n"

    user_prompt = f"""
Here is the math problem:

[PROBLEM]
{problem_text}

Here is the full Chain of Thought, broken into chunks:

[CHUNKS]
{chunk_str.strip()}

Now label each chunk with function tags and dependencies."""

    raw_response = generate_api_response(
        messages=[
            {"role": "system", "content": DAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    response = re.split(r"\n\d{1,2}\.", "\n" + raw_response.strip())
    response = [r.strip() for r in response if r.strip()]

    assert len(response) == len(chunks), f"Length mismatch: {len(response)} != {len(chunks)}"

    return [CATEGORIES.get(r, "Unknown") for r in response]

# %%

example_categories = categorize_sentences_autorater(example_problem_text, example_sentences)

for chunk, category in zip(example_sentences, example_categories):
    print(f"{category:>25} | {chunk!r}")

# %%

easy_categories_autorater = categorize_sentences_autorater(easy_problem_text, easy_chunks)

df = pd.DataFrame(
    {
        "Category (heuristics)": easy_categories,
        "Category (autorater)": easy_categories_autorater,
        "Chunk": easy_chunks,
    }
)
with pd.option_context("display.max_colwidth", None):
    display(df)

# %%

CATEGORY_COLORS = {
    "Problem Setup": "#4285F4",
    "Plan Generation": "#EA4335",
    "Fact Retrieval": "#FBBC05",
    "Active Computation": "#34A853",
    "Uncertainty Management": "#9C27B0",
    "Result Consolidation": "#00BCD4",
    "Self Checking": "#FF9800",
    "Final Answer Emission": "#795548",
    "Unknown": "#9E9E9E",
}


def visualize_trace_structure(chunks: list[str], categories: list[str], problem_text: str = None):
    """Visualize a reasoning trace with color-coded sentence categories."""

    n_chunks = len(chunks)
    fig, ax = plt.subplots(figsize=(12, 1 + int(0.5 * n_chunks)))

    for idx, (chunk, category) in enumerate(zip(chunks, categories)):
        color = CATEGORY_COLORS.get(category, "#9E9E9E")

        y = n_chunks - idx  # Start from top

        # Category label with colored background
        ax.barh(y, 0.15, left=0, height=0.8, color=color, alpha=0.6)
        ax.text(0.075, y, f"{category}", ha="center", va="center", fontsize=9, weight="bold")

        # Sentence text
        text = chunk[:100] + ("..." if len(chunk) > 100 else "")
        ax.text(0.17, y, f"[{idx}] {text}", va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, n_chunks + 0.5)
    ax.axis("off")

    if problem_text:
        fig.suptitle(f"Problem: {problem_text[:120]}...", fontsize=11, y=0.98, weight="bold")

    plt.title("Reasoning Trace Structure", fontsize=13, pad=30)
    plt.tight_layout()
    plt.show()


visualize_trace_structure(easy_chunks, easy_categories, easy_problem_text)

# %%

def load_single_file(file_path: str):
    local_path = hf_hub_download(repo_id=DATASET_NAME, filename=file_path, repo_type="dataset")
    with open(local_path, "r") as f:
        return json.load(f)


def load_problem_data(problem_id: int, verbose: bool = True):
    disable_progress_bars()

    problem_dir = "correct_base_solution"
    problem_dir_forced = "correct_base_solution_forced_answer"

    problem_path = f"deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/{problem_dir}/problem_{problem_id}"
    problem_path_forced = (
        f"deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/{problem_dir_forced}/problem_{problem_id}"
    )

    base_solution = load_single_file(f"{problem_path}/base_solution.json")
    problem = load_single_file(f"{problem_path}/problem.json")
    chunks_labeled = load_single_file(f"{problem_path}/chunks_labeled.json")
    chunk_solutions = []
    chunk_solutions_forced = []

    for chunk_idx in tqdm(range(len(chunks_labeled)), disable=not verbose):
        chunk_solutions.append(load_single_file(f"{problem_path}/chunk_{chunk_idx}/solutions.json"))
        chunk_solutions_forced.append(load_single_file(f"{problem_path_forced}/chunk_{chunk_idx}/solutions.json"))

    enable_progress_bars()

    return {
        "problem": problem,
        "base_solution": base_solution,
        "chunks_labeled": chunks_labeled,
        "chunk_solutions_forced": chunk_solutions_forced,
        "chunk_solutions": chunk_solutions,
    }


problem_data = load_problem_data(PROBLEM_ID)

# %%

print(problem_data["problem"])
print()

for i, c in enumerate(problem_data["chunks_labeled"]):
    print(f"{i}. {c['chunk']!r}")

# %%

def calculate_answer_importance(full_cot_list: list[list[str]], answer: str) -> list[float]:
    """
    Calculate importance for chunks, based on accuracy differences.

    Args:
        full_cot_list: List of resampled rollouts; the [i][j]-th element is the j-th rollout which
          was generated by answer-forcing immediately after the i-th chunk (e.g. the [0][0]-th
          element is the 0th rollout which doesn't include any chunks of the model's reasoning).
        answer: The ground truth answer to the problem.
        chunk_idx: Index of the chunk to calculate importance for.

    Returns:
        float: Forced importance score
    """

    def extract_answer_from_cot(cot: str) -> str:
        answer = cot.split("\\boxed{")[-1].split("}")[0]
        return "".join(char for char in answer if char.isdigit() or char == ".")

    # Get list of P(A_{S_i}) values for each chunk
    probabilities = [
        sum(extract_answer_from_cot(cot) == answer for cot in cot_list) / len(cot_list) for cot_list in full_cot_list
    ]

    # Convert these to importance scores: P(A_{S_i}) - P(A_{S_{i-1}})
    return np.diff(probabilities).tolist()

# %%

full_cot_list = [
    [rollout["full_cot"] for rollout in chunk_rollouts] for chunk_rollouts in problem_data["chunk_solutions_forced"]
]
answer = problem_data["problem"]["gt_answer"]

forced_answer_importances = calculate_answer_importance(full_cot_list, answer)
forced_answer_importances_cumsum = np.cumsum(forced_answer_importances)
forced_answer_importances_cumsum += 1 - forced_answer_importances_cumsum[-1]

df = pd.DataFrame(
    {
        "Forced answer importance": forced_answer_importances,
        "Accuracy": forced_answer_importances_cumsum,
        "chunk": [d["chunk"] for d in problem_data["chunks_labeled"][:-1]],
        "tags": [d["function_tags"][0] for d in problem_data["chunks_labeled"][:-1]],
    }
)

fig = px.line(
    df,
    labels={"index": "Chunk index", "value": "Importance", "variable": "Metric"},
    y=["Forced answer importance", "Accuracy"],
    hover_data=["chunk", "tags"],
)
fig.update_layout(title="Forced answer importance")
fig.show()

# %%

full_cot_list = [
    [rollout["full_cot"] for rollout in chunk_rollouts] for chunk_rollouts in problem_data["chunk_solutions"]
]
answer = problem_data["problem"]["gt_answer"]

resampling_answer_importances = calculate_answer_importance(full_cot_list, answer)

resampling_answer_importances_precomputed = [
    chunk_data["resampling_importance_accuracy"] for chunk_data in problem_data["chunks_labeled"][:-1]
]

avg_diff = np.abs(np.subtract(resampling_answer_importances, resampling_answer_importances_precomputed)).mean()
assert avg_diff < 0.01, f"Your implementation may be incorrect: {avg_diff=:.4f}"
print(f"Average difference: {avg_diff:.4f}")

# %%

resampling_answer_importances_cumsum = np.cumsum(resampling_answer_importances)
resampling_answer_importances_cumsum += 1 - resampling_answer_importances_cumsum[-1]

df = pd.DataFrame(
    {
        "Resampling answer importance": resampling_answer_importances,
        "Accuracy": resampling_answer_importances_cumsum,
        "chunk": [d["chunk"] for d in problem_data["chunks_labeled"][:-1]],
        "tags": [d["function_tags"][0] for d in problem_data["chunks_labeled"][:-1]],
    }
)

fig = px.line(
    df,
    labels={"index": "Chunk index", "value": "Importance", "variable": "Metric"},
    y=["Resampling answer importance", "Accuracy"],
    hover_data=["chunk", "tags"],
)
fig.update_layout(title="Resampling answer importance")
fig.show()

# %%

# Get the embeddings of S_i, the chunks we'll be resampling
chunks_removed = [chunk_data["chunk"] for chunk_data in problem_data["chunks_labeled"]]
embeddings_S_i = embedding_model.encode(chunks_removed)  # (N_chunks, d_embed)

# Get the embeddings of T_i, the resampled chunks
chunks_resampled = [
    [rollout["chunk_resampled"] for rollout in chunk_rollouts] for chunk_rollouts in problem_data["chunk_solutions"]
]
embeddings_T_i = np.stack([embedding_model.encode(r) for r in chunks_resampled])  # (N_chunks, N_resamples, d_embed)

# Get the cosine similarities
cos_sims = einops.einsum(embeddings_S_i, embeddings_T_i, "chunk d_embed, chunk resample d_embed -> chunk resample")
print(f"Computed cosine similarities for {cos_sims.shape[0]} chunks, {cos_sims.shape[1]} resamples")

# %%

# Group the cosine similarity data into a dataframe
cos_sims_mean = cos_sims.mean(axis=1)
chunk_labels = [CATEGORIES[chunk["function_tags"][0]] for chunk in problem_data["chunks_labeled"]]
df = pd.DataFrame(
    {
        "Label": chunk_labels,
        "Cosine similarity": cos_sims_mean,
        "chunk": [chunk["chunk"] for chunk in problem_data["chunks_labeled"]],
    }
)

# Bar chart of average cosine similarity for each chunk
px.bar(
    df,
    labels={"index": "Chunk index", "value": "Cosine similarity"},
    color="Label",
    color_discrete_map=CATEGORY_COLORS,
    hover_data=["chunk"],
    title="Cosine similarity between removed and resampled chunks",
).show()

# Boxplot of cosine cosine similarities grouped by chunk label
label_order = df.groupby("Label")["Cosine similarity"].mean().sort_values().index.tolist()
px.box(
    df,
    x="Label",
    y="Cosine similarity",
    labels={"x": "Label", "y": "Cosine similarity", "color": "Label"},
    color="Label",
    color_discrete_map=CATEGORY_COLORS,
    category_orders={"Label": label_order},
    boxmode="overlay",
    width=1000,
    title="Cosine similarity, grouped by chunk label",
).update_xaxes(tickangle=45).update_layout(boxgap=0.3).show()

# %%

def calculate_counterfactual_answer_importance(
    chunks_removed: list[str],
    chunks_resampled: list[list[str]],
    full_cot_list: list[list[str]],
    answer: str,
    threshold: float = 0.8,
    min_indices: int = 5,
    embedding_model: sentence_transformers.SentenceTransformer = embedding_model,
) -> list[float]:
    """
    Calculate importance for chunks, based on accuracy differences, after filtering for low
    new-generation cosine similarity.

    Args:
        chunks_removed: List of chunks $S_i$ which were removed from the rollouts.
        chunks_resampled: List of chunks $T_i$ which were resampled for each of the multiple rollouts.
        full_cot_list: List of resampled rollouts; the [i][j]-th element is the j-th rollout which
          was generated by answer-forcing immediately after the i-th chunk (e.g. the [0][0]-th
          element is the 0th rollout which doesn't include any chunks of the model's reasoning).
        answer: The ground truth answer to the problem.
        threshold: Minimum embedding cosine similarity to consider a rollout as "sufficiently different"
        min_indices: Minimum number of indices we can have post-filtering to count this score.
        embedding_model: Embedding model to use for calculating cosine similarity

    Returns:
        float: Forced importance score
    """

    def get_filtered_indices(chunk_removed: str, chunks_resampled: list[str]) -> list[int]:
        embedding_S_i = embedding_model.encode(chunk_removed)  # (d_embed,)
        embeddings_T_i = embedding_model.encode(chunks_resampled)  # (N, d_embed,)
        cos_sims = embedding_S_i @ embeddings_T_i.T  # (N,)
        return np.where(cos_sims < threshold)[0]

    def extract_answer_from_cot(cot: str) -> str:
        answer = cot.split("\\boxed{")[-1].split("}")[0]
        return "".join(char for char in answer if char.isdigit() or char == ".")

    filtered_indices = [
        get_filtered_indices(chunk_removed, _chunks_resampled)
        for chunk_removed, _chunks_resampled in zip(chunks_removed, chunks_resampled)
    ]

    # Get list of P(A^c_{S_i}) values for each chunk (or None if can't be computed)
    probabilities = [
        sum(extract_answer_from_cot(cot_list[idx]) == answer for idx in indices) / len(indices)
        if len(indices) >= min_indices
        else None
        for cot_list, indices in zip(full_cot_list, filtered_indices)
    ]

    # Forward-fill this list, to remove the "None" values
    probabilities = pd.Series(probabilities).ffill().bfill().tolist()

    # Return diffs
    return np.diff(probabilities).tolist()

# %%

chunks_removed = [chunk_data["chunk"] for chunk_data in problem_data["chunks_labeled"]]
chunks_resampled = [
    [rollout["chunk_resampled"] for rollout in chunk_rollouts] for chunk_rollouts in problem_data["chunk_solutions"]
]
full_cot_list = [
    [rollout["full_cot"] for rollout in chunk_rollouts] for chunk_rollouts in problem_data["chunk_solutions"]
]
answer = problem_data["problem"]["gt_answer"]

counterfactual_answer_importances = calculate_counterfactual_answer_importance(
    chunks_removed, chunks_resampled, full_cot_list, answer
)

counterfactual_answer_importances_precomputed = [
    -chunk_data["counterfactual_importance_accuracy"] for chunk_data in problem_data["chunks_labeled"][:-1]
]

avg_diff = np.abs(np.subtract(counterfactual_answer_importances, counterfactual_answer_importances_precomputed)).mean()
assert avg_diff < 0.1, f"Your implementation may be incorrect: {avg_diff=:.4f}"
print(f"Average difference: {avg_diff:.4f}")

# %%

counterfactual_answer_importances_cumsum = np.cumsum(counterfactual_answer_importances)
counterfactual_answer_importances_cumsum += 1 - counterfactual_answer_importances_cumsum[-1]

df = pd.DataFrame(
    {
        "Resampling answer importance": resampling_answer_importances,
        "Resampling answer importance (accuracy)": resampling_answer_importances_cumsum,
        "Counterfactual answer importance": counterfactual_answer_importances,
        "Counterfactual answer importance (accuracy)": counterfactual_answer_importances_cumsum,
        "chunk": [d["chunk"] for d in problem_data["chunks_labeled"][:-1]],
        "tags": [d["function_tags"][0] for d in problem_data["chunks_labeled"][:-1]],
    }
)

fig = px.line(
    df,
    labels={"index": "Chunk index", "value": "Importance", "variable": "Metric"},
    y=[
        "Resampling answer importance",
        "Resampling answer importance (accuracy)",
        "Counterfactual answer importance",
        "Counterfactual answer importance (accuracy)",
    ],
    hover_data=["chunk", "tags"],
)
fig.update_layout(title="Resampling vs counterfactual answer importance")
fig.show()

# %%

# Get the 5 most common labels
label_counts = pd.Series(chunk_labels[:-1]).value_counts()
top_5_labels = label_counts.head(5).index.tolist()

# Create dataframe and filter
df_filtered = pd.DataFrame(
    {
        "Label": chunk_labels[:-1],
        "Importance": counterfactual_answer_importances,
        "position": np.arange(len(chunk_labels[:-1])) / len(chunk_labels[:-1]),
    }
)
df_filtered = df_filtered[df_filtered["Label"].isin(top_5_labels)]
grouped = df_filtered.groupby("Label")[["Importance", "position"]].mean().reset_index()

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
for label in grouped["Label"]:
    row = grouped[grouped["Label"] == label]
    ax.scatter(row["position"], row["Importance"], label=label, color=CATEGORY_COLORS.get(label))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Normalized position in trace (0-1)")
ax.set_ylabel("Counterfactual importance")
ax.set_title("Sentence category effect")
ax.legend()
plt.show()

# %%

def calculate_counterfactual_answer_importance_kl_div(
    chunks_removed: list[str],
    chunks_resampled: list[list[str]],
    full_cot_list: list[list[str]],
    threshold: float = 0.8,
    min_indices: int = 5,
    alpha: int = 1,
    embedding_model: sentence_transformers.SentenceTransformer = embedding_model,
) -> list[float]:
    """
    Calculate importance for chunks, based on accuracy differences, after filtering for low
    new-generation cosine similarity.

    Args:
        chunks_removed: List of chunks $S_i$ which were removed from the rollouts.
        chunks_resampled: List of chunks $T_i$ which were resampled for each of the multiple rollouts.
        full_cot_list: List of resampled rollouts; the [i][j]-th element is the j-th rollout which
          was generated by answer-forcing immediately after the i-th chunk (e.g. the [0][0]-th
          element is the 0th rollout which doesn't include any chunks of the model's reasoning).
        threshold: Minimum embedding cosine similarity to consider a rollout as "sufficiently different"
        min_indices: Minimum number of indices we can have post-filtering to count this score.
        alpha: Laplace smoothing parameter
        embedding_model: Embedding model to use for calculating cosine similarity

    Returns:
        float: Forced importance score
    """
    counter = 0

    def kl_div(answers_1: list[str], answers_2: list[str]) -> float:
        nonlocal counter
        counter += 1
        if not answers_1 or not answers_2:
            return 0.0
        # Get vocab set
        vocab = set(answers_1) | set(answers_2)
        # Get counts & probabilities
        count_1 = Counter(answers_1)
        count_2 = Counter(answers_2)
        p_1 = {k: v / sum(count_1.values()) for k, v in count_1.items()}
        # Laplace smoothing
        count_1_smoothed = {k: count_1.get(k, 0) + alpha for k in vocab}
        count_2_smoothed = {k: count_2.get(k, 0) + alpha for k in vocab}
        p_1_smoothed = {k: v / sum(count_1_smoothed.values()) for k, v in count_1_smoothed.items()}
        p_2_smoothed = {k: v / sum(count_2_smoothed.values()) for k, v in count_2_smoothed.items()}
        # Get KL divergence
        return sum(p_1[k] * np.log(p_1_smoothed[k] / p_2_smoothed[k]) for k in p_1)

    def get_filtered_indices(chunk_removed: str, chunks_resampled: list[str]) -> list[int]:
        embedding_S_i = embedding_model.encode(chunk_removed)  # (d_embed,)
        embeddings_T_i = embedding_model.encode(chunks_resampled)  # (N, d_embed,)
        cos_sims = embedding_S_i @ embeddings_T_i.T  # (N,)
        return np.where(cos_sims < threshold)[0]

    def extract_answer_from_cot(cot: str) -> str:
        answer = cot.split("\\boxed{")[-1].split("}")[0]
        return "".join(char for char in answer if char.isdigit() or char == ".")

    filtered_indices = [
        get_filtered_indices(chunk_removed, _chunks_resampled)
        for chunk_removed, _chunks_resampled in zip(chunks_removed, chunks_resampled)
    ]

    # Get a list of the different answers returned in each of the resampled (filtered) rollouts
    all_answers = [
        [extract_answer_from_cot(cot_list[idx]) for idx in indices] if len(indices) >= min_indices else []
        for cot_list, indices in zip(full_cot_list, filtered_indices)
    ]

    # Get KL divergences using the function defined above
    kl_divs = [kl_div(a1, a0) for a1, a0 in zip(all_answers[1:], all_answers)]

    return kl_divs

# %%

# Get the new KL div-based importance scores
chunk_labels = [CATEGORIES[chunk["function_tags"][0]] for chunk in problem_data["chunks_labeled"]]
chunks_removed = [chunk_data["chunk"] for chunk_data in problem_data["chunks_labeled"]]
chunks_resampled = [
    [rollout["chunk_resampled"] for rollout in chunk_rollouts] for chunk_rollouts in problem_data["chunk_solutions"]
]
full_cot_list = [
    [rollout["full_cot"] for rollout in chunk_rollouts] for chunk_rollouts in problem_data["chunk_solutions"]
]
counterfactual_answer_importances_kl_div = calculate_counterfactual_answer_importance_kl_div(
    chunks_removed, chunks_resampled, full_cot_list
)

# Re-run the plotting code from above
label_counts = pd.Series(chunk_labels[:-1]).value_counts()
top_5_labels = label_counts.head(5).index.tolist()
df_filtered = pd.DataFrame(
    {
        "Label": chunk_labels[:-1],
        "Importance": counterfactual_answer_importances_kl_div,
        "position": np.arange(len(chunk_labels[:-1])) / len(chunk_labels[:-1]),
    }
)
df_filtered = df_filtered[df_filtered["Label"].isin(top_5_labels)]
grouped = df_filtered.groupby("Label")[["Importance", "position"]].mean().reset_index()
fig, ax = plt.subplots(figsize=(6, 4))
for label in grouped["Label"]:
    row = grouped[grouped["Label"] == label]
    ax.scatter(row["position"], row["Importance"], label=label, color=CATEGORY_COLORS.get(label))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Normalized position in trace (0-1)")
ax.set_ylabel("Counterfactual importance")
ax.set_title("Sentence category effect")
ax.legend()
plt.show()

# %%

df = pd.DataFrame(
    {
        "Counterfactual answer importance": counterfactual_answer_importances,
        "Counterfactual answer importance (KL div)": counterfactual_answer_importances_kl_div,
        "chunk": [d["chunk"] for d in problem_data["chunks_labeled"][:-1]],
        "tags": [d["function_tags"][0] for d in problem_data["chunks_labeled"][:-1]],
    }
)

fig = px.line(
    df,
    labels={"index": "Chunk index", "value": "Importance", "variable": "Metric"},
    y=[
        "Counterfactual answer importance",
        "Counterfactual answer importance (KL div)",
    ],
    hover_data=["chunk", "tags"],
)
fig.update_layout(title="Resampling vs counterfactual answer importance")
fig.show()

# %%

def get_resampled_rollouts(
    prompt: str,
    num_resamples_per_chunk: int = 100,
    batch_size: int = 4,
    max_new_tokens: int = 2048,
    up_to_n_chunks: int | None = None,
    model: transformers.models.llama.modeling_llama.LlamaForCausalLM = model,
) -> tuple[str, list[str], list[list[dict]]]:
    """
    After each chunk in `chunks`, computes a number of resampled rollouts.

    Args:
        prompt: The initial problem prompt (which ends with a <think> tag).
        num_resamples_per_chunk: Number of resamples to compute for each chunk.

    Returns:
        Tuple of (full_answer, chunks, resampled_rollouts) where the latter is a list of lists of
        dicts (one for each chunk & resample on that chunk).
    """

    @torch.inference_mode()
    def generate(inputs):
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            stopping_criteria=[StopOnThink()],
            pad_token_id=tokenizer.eos_token_id,
        )

    # First, generate a completion which we'll split up into chunks.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = generate(inputs)
    full_answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    chunks = split_solution_into_chunks(full_answer)

    # Second, generate resamples at each chunk.
    chunk_rollouts = []
    n_chunk_instances = defaultdict(int)
    for chunk in tqdm(chunks[:up_to_n_chunks]):
        chunk_rollouts.append([])

        # To get the answer before the chunk, we split on the correct instance of the chunk (since
        # this chunk might have appeared multiple times in the answer).
        n_chunk_instances[chunk] += 1
        full_answer_split = (
            prompt + chunk.join(full_answer.split(chunk, maxsplit=n_chunk_instances[chunk])[:-1]).strip()
        )
        inputs = tokenizer([full_answer_split] * batch_size, return_tensors="pt").to(device)

        for _ in range(num_resamples_per_chunk // batch_size):
            output_batch = generate(inputs)
            for output_generation in output_batch:
                generated_text = tokenizer.decode(
                    output_generation[inputs["input_ids"].shape[1] :], skip_special_tokens=False
                )
                chunk_rollouts[-1].append(
                    {
                        "full_cot": full_answer_split + generated_text,
                        "chunk_resampled": split_solution_into_chunks(generated_text)[0],
                        "chunk_replaced": chunk,
                        "rollout": generated_text,
                    }
                )

    return full_answer, chunks, chunk_rollouts

# %%

resampled_rollouts = get_resampled_rollouts(
    prompt=problem_data["base_solution"]["prompt"],
    num_resamples_per_chunk=2,
    batch_size=2,
    max_new_tokens=2048,
    up_to_n_chunks=4,
)

for i, resamples in enumerate(resampled_rollouts):
    print("Replaced chunk: " + repr(resamples[0]["chunk_replaced"]))
    for j, r in enumerate(resamples):
        print(f"    Resample {j}: " + repr(r["chunk_resampled"]))
    print()

# Replaced chunk: 'Alright, so I have this problem here:'
#     Resample 0: 'Okay, so I need to figure out how many bits are in the binary representation of the hexadecimal number 66666₁₆.'
#     Resample 1: "Alright, so I've got this problem here:"

# Replaced chunk: 'When the base-16 number \\(66666_{16}\\) is written in base 2, how many base-2 digits (bits) does it have?'
#     Resample 0: 'when the base-16 number \\(66666_{16}\\) is written in base 2, how many base-2 digits (bits) does it have?'
#     Resample 1: 'when the base-16 number 66666₁₆ is written in base 2, how many base-2 digits (bits) does it have?'

# Replaced chunk: 'Hmm, okay.'
#     Resample 0: 'Hmm, okay.'
#     Resample 1: 'Hmm, okay.'

# Replaced chunk: 'Let me try to figure this out step by step.'
#     Resample 0: 'Let me think about how to approach this.'
#     Resample 1: 'I need to figure out how many bits are required to represent this hexadecimal number in binary.'

# %%
