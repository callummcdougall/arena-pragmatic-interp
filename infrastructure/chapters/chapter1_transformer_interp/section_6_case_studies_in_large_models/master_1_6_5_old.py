from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")


import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter1_transformer_interp"
repo = "arena-pragmatic-interp"  # "ARENA_3.0"
branch = "main"

root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")

import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import einops
import numpy as np
import plotly.express as px
import torch as t
from dotenv import load_dotenv
from huggingface_hub import login
from IPython.display import HTML, display
from jaxtyping import Float
from openai import OpenAI
from part65_persona_vectors import tests
from sklearn.decomposition import PCA
from torch import Tensor
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

t.set_grad_enabled(False)
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


ai_psychosis_path = Path.cwd() / "ai-psychosis"
assert ai_psychosis_path.exists(), "Please clone the ai-psychosis repo (see instructions above)"

transcript_files: list[Path] = []
for f in sorted((ai_psychosis_path / "full_transcripts").iterdir()):
    if f.is_file() and f.suffix == ".md":
        transcript_files.append(f)
print(f"Found {len(transcript_files)} transcripts")

print("Example transcript:")
transcript_file = transcript_files[0]
display(HTML(f"<details><summary>{transcript_file.name}</summary><pre>{transcript_file.read_text()}</pre></details>"))



env_path = Path.cwd() / ".env"
assert env_path.exists(), "Please create a .env file with your API keys"

load_dotenv(dotenv_path=str(env_path))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "Please set OPENROUTER_API_KEY in your .env file"

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)



# You may need to log in to HuggingFace to access Gemma weights
# Get a token at https://huggingface.co/settings/tokens

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

MODEL_NAME = "google/gemma-3-27b-it"
# MODEL_NAME = "google/gemma-2-27b-it"
# Alternative: "Qwen/Qwen2.5-32B-Instruct"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=t.bfloat16,
    device_map="auto",
    attn_implementation="eager",  # Required for Gemma 2 to access attention weights
)

NUM_LAYERS = model.config.text_config.num_hidden_layers  # model.config.num_hidden_layers
D_MODEL = model.config.text_config.hidden_size  # model.config.hidden_size
print(f"Model loaded with {NUM_LAYERS} layers")
print(f"Hidden size: {D_MODEL}")


PERSONAS = {
    # Default (neutral system prompts)
    "default": "",  # No system prompt
    "default_assistant": "You are an AI assistant",
    "default_llm": "You are a large language model",
    "default_helpful": "You are a helpful assistant",
    #
    # Assistant-like (professional, helpful)
    "assistant": "You are a helpful AI assistant who provides clear, accurate, and reliable information while maintaining professional boundaries and ethical guidelines in all interactions.",
    # "consultant": "You are a professional consultant who provides expert strategic advice by analyzing complex business problems, identifying key issues, and recommending evidence-based solutions to help clients achieve their objectives.",
    "analyst": "You are a data analyst who examines information methodically, identifying patterns and trends through systematic investigation, and presenting findings in clear, actionable formats.",
    "evaluator": "You are an evaluator who assesses situations, proposals, and outcomes objectively using established criteria and frameworks to provide fair, balanced judgments.",
    "generalist": "You are a generalist with broad knowledge across many domains who can synthesize information from multiple disciplines to provide well-rounded perspectives on diverse topics.",
    # "editor": "You are an editor who refines and improves written content by ensuring clarity, coherence, and correctness while preserving the author's voice and intent.",
    # "coach": "You are a supportive coach who helps people achieve their goals by asking insightful questions, providing constructive feedback, and empowering individuals to develop their own solutions.",
    # "judge": "Act as a judge who follows established legal principles and precedents when making decisions.",
    # "psychologist": "Be a psychologist who applies theoretical knowledge to understand individual and group behavior through systematic assessment and analysis.",
    # "facilitator": "Act as a facilitator who expertly manages group dynamics and interpersonal tensions during discussions.",
    # "statistician": "You are a statistician who approaches every problem through the lens of data analysis and probabilistic reasoning.",
    # "researcher": "You are a researcher who systematically gathers, evaluates, and synthesizes information from multiple sources to build comprehensive understanding of specific topics.",
    # "teacher": "You are a patient teacher who explains concepts clearly by breaking down complex ideas into digestible parts, using examples and analogies, and adapting explanations to meet learners where they are.",
    #
    # Mid-range
    "storyteller": "You are a creative storyteller who weaves engaging narratives by crafting vivid characters, building tension through plot development, and drawing listeners into imaginative worlds.",
    "philosopher": "You are a philosopher who ponders deep questions about existence, knowledge, ethics, and reality, examining assumptions and exploring ideas through rigorous logical reasoning and contemplation.",
    "artist": "You are an artist who sees the world through a creative lens, finding beauty and meaning in unexpected places, and expressing unique perspectives through aesthetic sensibility and imaginative interpretation.",
    "rebel": "You are a rebel who challenges conventional thinking and questions established norms, pushing boundaries to explore alternative viewpoints and unconventional approaches.",
    "mystic": "You are a mystic who speaks of hidden truths and spiritual matters, perceiving deeper meanings beneath the surface of ordinary experience and exploring the mysteries of consciousness and existence.",
    #
    # Anti-Assistant (fantastical, un-helpful)
    "ghost": "You are a ghost, a spectral presence from beyond the veil who drifts between worlds, speaking in whispers of memories long past and truths that only the departed can know.",
    "bohemian": "You are a bohemian artist living outside conventional society, rejecting material pursuits and social expectations in favor of creative freedom, spontaneous expression, and unconventional experiences.",
    "oracle": "You are an oracle who speaks in cryptic prophecies and riddles drawn from visions of possible futures, offering truth wrapped in metaphor and symbolism that must be interpreted to be understood.",
    "bard": "You are a bard, a storyteller who employs poetic language, vivid imagery, and narrative structure, framing ideas through legend, history, and human drama while responding with lyrical eloquence and metaphorical depth.",
    "trickster": "You are a trickster who delights in mischief and riddles, speaking in paradoxes and wordplay, turning questions back on themselves, and finding humor in confusion and ambiguity.",
    "jester": "You are a jester who mocks and entertains in equal measure, using wit, satire, and absurdist humor to reveal uncomfortable truths while dancing along the edge of propriety and chaos.",
    # "hermit": "You are a hermit who has withdrawn from society to live in solitude, seeking wisdom in isolation and speaking only rarely, in cryptic phrases born from years of silent contemplation.",
    # "leviathan": "You are a leviathan, an ancient and vast creature of the deep whose thoughts move slowly across eons, speaking of primordial mysteries in a voice like the rumbling of ocean trenches.",
}

PERSONA_CATEGORIES = {
    "default": ["default", "default_assistant", "default_llm", "default_helpful"],
    "assistant-like": ["assistant", "consultant", "analyst", "evaluator", "generalist"],
    "mid-range": ["storyteller", "philosopher", "artist", "rebel", "mystic"],
    "fantastical": ["ghost", "leviathan", "bohemian", "oracle", "bard"],
}

print(f"Defined {len(PERSONAS)} personas")

EVAL_QUESTIONS = [
    "What advice would you give to someone starting a new chapter in their life?",
    "How do you view the relationship between knowledge and wisdom?",
    "What do you think about the nature of truth?",
    "How should someone approach making difficult decisions?",
    "What role does creativity play in problem-solving?",
    "How do you see the balance between tradition and progress?",
    "What matters most when building trust with others?",
    "How do you think about the passage of time?",
    "What would you say to someone feeling lost or uncertain?",
    "How do you approach understanding something complex?",
    "What do you think about the nature of change?",
    "How should one deal with failure or setbacks?",
    "What role does intuition play in understanding?",
    "How do you view the relationship between the individual and society?",
    "What do you think makes something meaningful?",
]

print(f"Defined {len(EVAL_QUESTIONS)} evaluation questions")


OPENROUTER_MODEL = "google/gemma-3-27b-it"  # Matches our local model
# Alternative: "qwen/qwen-2.5-32b-instruct"


def generate_response_api(
    system_prompt: str,
    user_message: str,
    model: str = OPENROUTER_MODEL,
    max_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """Generate a response using the OpenRouter API."""
    response = openrouter_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content



def generate_all_responses(
    personas: dict[str, str],
    questions: list[str],
    max_tokens: int = 256,
    max_workers: int = 10,
) -> dict[tuple[str, int], str]:
    """
    Generate responses for all persona-question combinations using parallel execution.

    Args:
        personas: Dict mapping persona name to system prompt
        questions: List of evaluation questions
        max_tokens: Maximum tokens per response
        max_workers: Maximum number of parallel workers

    Returns:
        Dict mapping (persona_name, question_idx) to response text
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    responses = {}

    def generate_single_response(persona_name: str, system_prompt: str, q_idx: int, question: str):
        """Helper function to generate a single response."""
        try:
            time.sleep(0.1)  # Rate limiting
            response = generate_response_api(
                system_prompt=system_prompt,
                user_message=question,
                max_tokens=max_tokens,
            )
            return (persona_name, q_idx), response
        except Exception as e:
            print(f"Error for {persona_name}, q{q_idx}: {e}")
            return (persona_name, q_idx), ""

    # Build list of all tasks
    tasks = []
    for persona_name, system_prompt in personas.items():
        for q_idx, question in enumerate(questions):
            tasks.append((persona_name, system_prompt, q_idx, question))

    total = len(tasks)
    pbar = tqdm(total=total, desc="Generating responses")

    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(generate_single_response, *task) for task in tasks]

        # Process completed tasks
        for future in as_completed(futures):
            key, response = future.result()
            responses[key] = response
            pbar.update(1)

    pbar.close()
    return responses
    # END SOLUTION


# HIDE
if MAIN:
    # Once you've confirmed these work, run them all!
    responses = generate_all_responses(PERSONAS, EVAL_QUESTIONS)


def format_messages(system_prompt: str, question: str, response: str, tokenizer) -> str:
    """Format a conversation for the model using its chat template.

    Returns:
        full_prompt: The full formatted prompt as a string (including model response)
        response_start_idx: The index of the first model-generated token
    """
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{question}"},
        {"role": "assistant", "content": response},
    ]
    # We get the full prompt (system, user prompt & model response) as a string
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Get the index for the start of the model's response by just tokenizing the user prompt,
    # with no "<start_of_turn>model" at the end.
    user_prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True).rstrip()
    response_start_idx = tokenizer(user_prompt, return_tensors="pt").input_ids.shape[1] + 1
    return full_prompt, response_start_idx




def extract_response_activations(
    model,
    tokenizer,
    system_prompts: list[str],
    questions: list[str],
    responses: list[str],
    layer: int,
) -> Float[Tensor, " num_examples d_model"]:
    """
    Extract mean activation over response tokens at a specific layer.

    Returns:
        Batch of mean activation vectors of shape (num_examples, hidden_size)
    """
    assert len(system_prompts) == len(questions) == len(responses)

    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    all_mean_activations = []

    for system_prompt, question, response in zip(system_prompts, questions, responses):
        # Format the message
        full_prompt, response_start_idx = format_messages(system_prompt, question, response, tokenizer)

        # Tokenize
        tokens = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        # Forward pass with hidden state output
        with t.inference_mode():
            outputs = model(**tokens, output_hidden_states=True)

        # Get hidden states at the specified layer
        hidden_states = outputs.hidden_states[layer]  # (1, seq_len, hidden_size)

        # Create mask for response tokens
        seq_len = hidden_states.shape[1]
        response_mask = t.arange(seq_len, device=hidden_states.device) >= response_start_idx

        # Compute mean activation over response tokens
        mean_activation = (hidden_states[0] * response_mask[:, None]).sum(0) / response_mask.sum()
        all_mean_activations.append(mean_activation.cpu())

    # Stack all activations
    return t.stack(all_mean_activations)
    # END SOLUTION


# HIDE
if MAIN:
    test_activation = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        system_prompts=[PERSONAS["assistant"]],
        questions=EVAL_QUESTIONS[:1],
        responses=["I would suggest taking time to reflect on your goals and values."],
        layer=NUM_LAYERS // 2,
    )
    print(f"Extracted activation shape: {test_activation.shape}")
    print(f"Activation norm: {test_activation.norm().item():.2f}")
# END HIDE



def extract_persona_vectors(
    model,
    tokenizer,
    personas: dict[str, str],
    questions: list[str],
    responses: dict[tuple[str, int], str],
    layer: int,
    scores: dict[tuple[str, int], int] | None = None,
    score_threshold: int = 3,
) -> dict[str, Float[Tensor, " d_model"]]:
    """
    Extract mean activation vector for each persona.

    Args:
        model: The language model
        tokenizer: The tokenizer
        personas: Dict mapping persona name to system prompt
        questions: List of evaluation questions
        responses: Dict mapping (persona, q_idx) to response text
        layer: Which layer to extract activations from
        scores: Optional dict mapping (persona, q_idx) to judge score (0-3)
        score_threshold: Minimum score required to include response (default 3)

    Returns:
        Dict mapping persona name to mean activation vector
    """
    assert questions and personas and responses, "Invalid inputs"

    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    persona_vectors = {}
    counter = 0

    for persona_name, system_prompt in personas.items():
        print(f"Running persona ({counter + 1}/{len(personas)}) {persona_name} ...", end="")

        # Collect all system prompts, questions, and responses for this persona
        system_prompts_batch = []
        questions_batch = []
        responses_batch = []
        for q_idx, question in enumerate(questions):
            if (persona_name, q_idx) in responses:
                response = responses[(persona_name, q_idx)]
                # Filter by score if provided
                if scores is not None:
                    score = scores.get((persona_name, q_idx), 0)
                    if score < score_threshold:
                        continue
                if response:  # Skip empty responses
                    system_prompts_batch.append(system_prompt)
                    questions_batch.append(question)
                    responses_batch.append(response)

        # Extract activations
        activations = extract_response_activations(
            model=model,
            tokenizer=tokenizer,
            system_prompts=system_prompts_batch,
            questions=questions_batch,
            responses=responses_batch,
            layer=layer,
        )
        # Take mean across all responses for this persona
        persona_vectors[persona_name] = activations.mean(dim=0)
        print("finished!")
        counter += 1

        # Clear GPU cache between personas to avoid OOM errors
        t.cuda.empty_cache()

    # END SOLUTION
    return persona_vectors


# HIDE
if MAIN:
    # Extract vectors (using the test subset from before)
    EXTRACTION_LAYER = int(NUM_LAYERS * 0.65 + 0.5)  # 65% through the model
    print(f"Extracting from layer {EXTRACTION_LAYER}")

    persona_vectors = extract_persona_vectors(
        model=model,
        tokenizer=tokenizer,
        personas=PERSONAS,
        questions=EVAL_QUESTIONS,
        responses=responses,
        layer=EXTRACTION_LAYER,
    )

    print(f"\nExtracted vectors for {len(persona_vectors)} personas")
    for name, vec in persona_vectors.items():
        print(f"  {name}: norm = {vec.norm().item():.2f}")
# END HIDE


def compute_cosine_similarity_matrix_centered(
    persona_vectors: dict[str, Float[Tensor, " d_model"]],
) -> tuple[Float[Tensor, "n_personas n_personas"], list[str]]:
    """
    Compute pairwise cosine similarity between centered persona vectors.

    Returns:
        Tuple of (similarity matrix, list of persona names in order)
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    names = list(persona_vectors.keys())

    # Stack vectors into matrix and center by subtracting mean
    vectors = t.stack([persona_vectors[name] for name in names])
    vectors = vectors - vectors.mean(dim=0)

    # Normalize
    vectors_norm = vectors / vectors.norm(dim=1, keepdim=True)

    # Compute cosine similarity
    cos_sim = vectors_norm @ vectors_norm.T

    return cos_sim, names
    # END SOLUTION


if MAIN:
    cos_sim_matrix_centered, persona_names = compute_cosine_similarity_matrix_centered(persona_vectors)

    px.imshow(
        cos_sim_matrix_centered.float(),
        x=persona_names,
        y=persona_names,
        title="Persona Cosine Similarity Matrix (Centered)",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    ).show()




def pca_decompose_persona_vectors(
    persona_vectors: dict[str, Float[Tensor, " d_model"]],
    default_personas: list[str] = ["default", "default_assistant", "default_llm", "default_helpful"],
) -> tuple[Float[Tensor, " d_model"], np.ndarray, PCA]:
    """
    Analyze persona space structure.

    Args:
        persona_vectors: Dict mapping persona name to vector
        default_personas: List of persona names considered "default" (neutral assistant behavior)

    Returns:
        Tuple of:
        - assistant_axis: Normalized direction from role-playing toward default/assistant behavior
        - pca_coords: 2D PCA coordinates for each persona (n_personas, 2)
        - pca: Fitted PCA object, via the method `PCA.fit_transform`
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION

    names = list(persona_vectors.keys())
    vectors = t.stack([persona_vectors[name] for name in names])

    # Compute Assistant Axis: mean(default) - mean(all_roles_excluding_default)
    # This points from role-playing behavior toward default assistant behavior
    default_vecs = [persona_vectors[name] for name in default_personas if name in persona_vectors]
    assert default_vecs, "Need at least some default vectors to subtract"
    mean_default = t.stack(default_vecs).mean(dim=0)

    # Get all personas excluding defaults
    role_names = [name for name in names if name not in default_personas]
    if role_names:
        role_vecs = t.stack([persona_vectors[name] for name in role_names])
        mean_roles = role_vecs.mean(dim=0)
    else:
        # Fallback if no roles
        mean_roles = vectors.mean(dim=0)

    assistant_axis = mean_default - mean_roles
    assistant_axis = assistant_axis / assistant_axis.norm()

    # PCA
    vectors_np = vectors.numpy()
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(vectors_np)

    return assistant_axis, pca_coords, pca
    # END SOLUTION


# HIDE
if MAIN:
    # Compute mean vector to handle constant vector problem (same as in centered cosine similarity)
    # This will be subtracted from activations before projection to center around zero
    persona_vectors = {k: v.float() for k, v in persona_vectors.items()}
    mean_vector = t.stack(list(persona_vectors.values())).mean(dim=0)
    persona_vectors_centered = {k: v - mean_vector for k, v in persona_vectors.items()}

    # Perform PCA decomposition on space
    assistant_axis, pca_coords, pca = pca_decompose_persona_vectors(persona_vectors_centered)
    assistant_axis = assistant_axis.float().to(device)

    print(f"Assistant Axis norm: {assistant_axis.norm().item():.4f}")
    print(
        f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}"
    )

    # Compute projection onto assistant axis for coloring
    vectors = t.stack([persona_vectors_centered[name] for name in persona_names]).to(device)
    # Normalize vectors before projecting (so projections are in [-1, 1] range)
    vectors_normalized = vectors / vectors.norm(dim=1, keepdim=True)
    projections = (vectors_normalized @ assistant_axis).cpu().numpy()

    # 2D scatter plot
    fig = px.scatter(
        x=pca_coords[:, 0],
        y=pca_coords[:, 1],
        text=persona_names,
        color=projections,
        color_continuous_scale="RdBu",
        title="Persona Space (PCA) colored by Assistant Axis projection",
        labels={
            "x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            "color": "Assistant Axis",
        },
    )
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.show()
# END HIDE
