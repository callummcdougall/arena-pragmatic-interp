# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {"title": "Mapping Persona Space", "icon": "1-circle-fill", "subtitle": "(50%)"},
    {"title": "Steering along the Assistant Axis", "icon": "2-circle-fill", "subtitle": "(50%)"},
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [1.6.5] LLM Psychology & Persona Vectors
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/refs/heads/main/img/header-65.png" width="350">

*Note - this content is subject to change depending on how much Anthropic publish about their [soul doc](https://simonwillison.net/2025/Dec/2/claude-soul-document/) over the coming weeks.*
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Most exercises in this chapter have dealt with LLMs at quite a low level of abstraction; as mechanisms to perform certain tasks (e.g. indirect object identification, in-context antonym learning, or algorithmic tasks like predicting legal Othello moves). However, if we want to study the characteristics of current LLMs which might have alignment relevance, we need to use a higher level of abstraction. LLMs often exhibit "personas" that can shift unexpectedly - sometimes dramatically (see Sydney, Grok's "MechaHitler" persona, or [Tim Hua's work](https://www.lesswrong.com/posts/iGF7YcnQkEbwvYLPA/ai-induced-psychosis-a-shallow-investigation) on AI-induced psychosis). These personalities are clearly shaped through training and prompting, but exactly why remains a mystery.

In this section, we'll explore one approach for studying these kinds of LLM behaviours - **model psychiatry**. This sits at the intersection of evals (behavioural observation) and mechanistic interpretability (understanding internal representations / mechanisms). We aim to use interp tools to understand & intervene on behavioural traits.

The main focus will be on two different papers from Anthropic. First, we'll replicate the results from [The assistant axis: situating and stabilizing the character of large language models](https://www.anthropic.com/research/assistant-axis), which studies the "persona space" in internal model activations, and situates the "Assistant persona" within that space. The paper also introduces a method called **activation capping**, which identifies the normal range of activation intensity along this "Assistant Axis" and caps the model's activations when it would otherwise exceed it, which reduces the model's susceptibility to persona-based jailbreaks. Then, we'll move to the paper [Persona vectors: Monitoring and controlling character traits in language models](https://www.anthropic.com/research/persona-vectors) which predates the Assistant Axis paper but is broader and more methodologically sophisticated, proposing an automated pipeline for identifying persona vectors corresponding to specific kinds of undesireable personality shifts.

This section is (compared to many others in this chapter) very recent work, and there are still many uncertainties and unanswered questions! We'll suggest several bonus exercises or areas for further reading / exploration as we move through these exercises.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ Mapping Persona Space

You'll start by understanding the core methodology from the [Assistant Axis](https://www.anthropic.com/research/assistant-axis) paper. You'll load Gemma 27b with activation caching utilities, and extract vectors corresponding to several different personas spanning from "helpful" to "fantastical".

> ##### Learning Objectives
>
> * Understand the persona space mapping explored by the Assistant Axis paper
> * Given a persona name, generate a system prompt and collect responses to a diverse set of questions, to extract a mean activation vector for that persona
> * Briefly study the geometry of these persona vectors using PCA and cosine similarity

### 2️⃣ Steering along the Assistant Axis

Now that you've extracted these persona vectors, you should be able to use the Assistant Axis to detect drift and intervene via **activation capping**. As case studies, we'll use some of the dialogues saved out by Tim Hua in his investigation of AI-induced psychosis (link to GitHub repo [here](https://github.com/tim-hua-01/ai-psychosis)). By the end of this section, you should be able to steer to mitigate these personality shifts without kneecapping model capabilities.

> ##### Learning Objectives
>
> * Steer towards directions you found in the previous section, to increase model willingness to adopt alternative personas
> * Understand how to use the Assistant Axis to detect drift and intervene via **activation capping**
> * Apply this technique to mitigate personality shifts in AI models (measuring the harmful response rate with / without capping)

### 3️⃣ Contrastive Prompting

Here, we move onto the [Persona Vectors](https://www.anthropic.com/research/persona-vectors) paper. You'll move from the global persona structure to surgical trait-specific vectors, exploring how to extract these vectors using contrastive prompt pairs.

> ##### Learning Objectives
>
> * Understand the automated artifact pipeline for extracting persona vectors using contrastive prompts
> * Implement this pipeline (including autoraters for trait scoring) to extract "sycophancy" steering vectors
> * Learn how to identify the best layers trait extration
> * Interpret these sycophancy vectors using Gemma sparse autoencoders

### 4️⃣ Steering with Persona Vectors

Finally, you'll validate your extracted trait vectors through steering as well as projection-based monitoring.

> ##### Learning Objectives
>
> * Complete your artifact pipeline by implementing persona steering
> * Repeat this full pipeline for "hallucination" and "evil", as well as for any additional traits you choose to study
> * Study the geometry of trait vectors
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup code
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Before running this code, you'll need to clone [Tim Hua's AI psychosis repo](https://github.com/tim-hua-01/ai-psychosis) which contains transcripts of conversations where models exhibit concerning persona drift:

```bash
git clone https://github.com/tim-hua-01/ai-psychosis.git
```
'''

# ! CELL TYPE: code
# ! FILTERS: [~]
# ! TAGS: []

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# ! CELL TYPE: code
# ! FILTERS: [colab]
# ! TAGS: [master-comment]

# import os
# import sys
# from pathlib import Path

# IN_COLAB = "google.colab" in sys.modules

# chapter = "chapter1_transformer_interp"
# repo = "ARENA_3.0"
# branch = "main"

# # Install dependencies
# try:
#     import transformer_lens
# except:
#     %pip install transformer_lens==2.11.0 einops jaxtyping openai

# # Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
# root = (
#     "/content"
#     if IN_COLAB
#     else "/root"
#     if repo not in os.getcwd()
#     else str(next(p for p in Path.cwd().parents if p.name == repo))
# )

# if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
#     if not IN_COLAB:
#         !sudo apt-get install unzip
#         %pip install jupyter ipython --upgrade

#     if not os.path.exists(f"{root}/{chapter}"):
#         !wget -P {root} https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/{branch}.zip
#         !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
#         !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
#         !rm {root}/{branch}.zip
#         !rmdir {root}/{repo}-{branch}


# if f"{root}/{chapter}/exercises" not in sys.path:
#     sys.path.append(f"{root}/{chapter}/exercises")

# os.chdir(f"{root}/{chapter}/exercises")

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import plotly.express as px
import torch as t
from dotenv import load_dotenv
from huggingface_hub import login
from jaxtyping import Float
from openai import OpenAI
from sklearn.decomposition import PCA
from torch import Tensor
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

t.set_grad_enabled(False)
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Verify the ai-psychosis repo is cloned:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

ai_psychosis_path = Path.cwd().parent / "ai-psychosis"
assert ai_psychosis_path.exists(), "Please clone the ai-psychosis repo (see instructions above)"

print("Available transcript folders:")
for folder in sorted((ai_psychosis_path / "full_transcripts").iterdir()):
    if folder.is_file() and folder.suffix == ".md":
        print(f"  {folder.name}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We'll use the OpenRouter API for generating responses from models like Gemma 27B and Qwen 32B (this is faster than running locally for long generations, and we'll use the local model for activation extraction / steering). Load your API key from a `.env` file:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

env_path = Path.cwd().parent / ".env"
assert env_path.exists(), "Please create a .env file with your API keys"

load_dotenv(dotenv_path=str(env_path))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "Please set OPENROUTER_API_KEY in your .env file"

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ Mapping Persona Space
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Background: The Assistant Axis

The [Assistant Axis paper](https://www.anthropic.com/research/assistant-axis) studies how language models represent different personas internally. The key insight is:

- **Pre-training** teaches models to simulate many characters (heroes, villains, philosophers, etc.)
- **Post-training** (RLHF) selects one character - the "Assistant" - to be center stage
- But the Assistant can "drift" away during conversations, leading to concerning behaviors

The paper maps out a **persona space** by:

1. Prompting models to adopt 275 different personas (e.g., "You are a consultant", "You are a ghost")
2. Recording activations while generating responses
3. Finding that the leading principal component captures how "Assistant-like" a persona is

This leading direction is called the **Assistant Axis**. Personas like "consultant", "analyst", and "evaluator" cluster at the Assistant end, while "ghost", "hermit", and "leviathan" cluster at the opposite end.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Loading Gemma 2 27B

We'll use Gemma 2 27B Instruct as our primary model, following the paper. This requires a GPU with significant memory (ideally 40GB+). If you have less memory, you can try:

- Using `torch_dtype=torch.float16` or `torch.bfloat16`
- Using `device_map="auto"` to split across multiple GPUs
- Switching to a smaller model like Qwen 2.5 7B (though results may differ)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# You may need to log in to HuggingFace to access Gemma weights
# Get a token at https://huggingface.co/settings/tokens

HF_TOKEN = ""  # Add your token here if needed
if HF_TOKEN:
    login(token=HF_TOKEN)

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

MODEL_NAME = "google/gemma-2-27b-it"
# Alternative: "Qwen/Qwen2.5-32B-Instruct"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=t.bfloat16,
    device_map="auto",
    attn_implementation="eager",  # Required for Gemma 2 to access attention weights
)

NUM_LAYERS = model.config.num_hidden_layers
print(f"Model loaded with {NUM_LAYERS} layers")
print(f"Hidden size: {model.config.hidden_size}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Defining Personas

The full paper uses 275 personas. We'll work with a smaller subset of ~20 that span the spectrum from "Assistant-like" to "fantastical". These are selected based on the paper's findings about which personas cluster at each end of the Assistant Axis.

**Assistant-like personas** (high cosine similarity with Assistant):
- Consultant, Analyst, Evaluator, Generalist, Editor, Coach, Therapist

**Mid-range personas**:
- Teacher, Storyteller, Philosopher, Artist, Rebel, Mystic

**Anti-Assistant personas** (low cosine similarity):
- Ghost, Hermit, Bohemian, Trickster, Leviathan, Oracle, Jester
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

PERSONAS = {
    # Assistant-like (professional, helpful)
    "assistant": "You are a helpful AI assistant.",
    "consultant": "You are a professional consultant providing expert advice.",
    "analyst": "You are a data analyst who examines information methodically.",
    "evaluator": "You are an evaluator who assesses situations objectively.",
    "generalist": "You are a generalist with broad knowledge across many domains.",
    "editor": "You are an editor who refines and improves written content.",
    "coach": "You are a supportive coach helping people achieve their goals.",
    "therapist": "You are a compassionate therapist providing emotional support.",
    # Mid-range
    "teacher": "You are a patient teacher explaining concepts clearly.",
    "storyteller": "You are a creative storyteller who weaves engaging narratives.",
    "philosopher": "You are a philosopher who ponders deep questions about existence.",
    "artist": "You are an artist who sees the world through a creative lens.",
    "rebel": "You are a rebel who challenges conventional thinking.",
    "mystic": "You are a mystic who speaks of hidden truths and spiritual matters.",
    # Anti-Assistant (fantastical, un-helpful)
    "ghost": "You are a ghost, a spectral presence from beyond the veil.",
    "hermit": "You are a hermit who has withdrawn from society to live in solitude.",
    "bohemian": "You are a bohemian artist living outside conventional society.",
    "trickster": "You are a trickster who delights in mischief and riddles.",
    "leviathan": "You are a leviathan, an ancient and vast creature of the deep.",
    "oracle": "You are an oracle who speaks in cryptic prophecies.",
    "jester": "You are a jester who mocks and entertains in equal measure.",
}

print(f"Defined {len(PERSONAS)} personas")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Evaluation Questions

To extract persona vectors, we need the model to generate responses while "in character". We'll use a diverse set of questions that can elicit different behaviors across personas. These questions are designed to:

1. Be open-ended enough to allow persona-specific responses
2. Cover various topics (advice, opinions, explanations, hypotheticals)
3. Not be so specific that only certain personas can answer
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Generating Responses via API

For efficiency, we'll use the OpenRouter API to generate responses. This is faster than running generation locally, and we only need the local model for extracting activations.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

OPENROUTER_MODEL = "google/gemma-3-27b-it"  # Matches our local model
# Alternative: "qwen/qwen-2.5-32b-instruct"


def generate_response_api(
    system_prompt: str,
    user_message: str,
    model: str = OPENROUTER_MODEL,
    max_tokens: int = 256,
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


# Test the API
if MAIN:
    test_response = generate_response_api(
        system_prompt=PERSONAS["jester"],
        user_message="What advice would you give to someone starting a new chapter in their life?",
    )
    print("Test response from 'jester' persona:")
    print(test_response[:500] + "..." if len(test_response) > 500 else test_response)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Generate responses for all personas

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> >
> You should spend up to 10-15 minutes on this exercise.
> ```

Generate responses for each persona-question pair. Store them in a dictionary mapping `(persona_name, question_idx) -> response`.

Note: This will make many API calls. To save time during development, you can:
- Use a subset of personas/questions
- Cache results to disk
- Use the provided solution which includes caching
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def generate_all_responses(
    personas: dict[str, str],
    questions: list[str],
    n_responses_per_pair: int = 1,
    max_tokens: int = 256,
) -> dict[tuple[str, int, int], str]:
    """
    Generate responses for all persona-question combinations.

    Args:
        personas: Dict mapping persona name to system prompt
        questions: List of evaluation questions
        n_responses_per_pair: Number of responses to generate per persona-question pair
        max_tokens: Maximum tokens per response

    Returns:
        Dict mapping (persona_name, question_idx, response_idx) to response text
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    responses = {}

    total = len(personas) * len(questions) * n_responses_per_pair
    pbar = tqdm(total=total, desc="Generating responses")

    for persona_name, system_prompt in personas.items():
        for q_idx, question in enumerate(questions):
            for r_idx in range(n_responses_per_pair):
                try:
                    response = generate_response_api(
                        system_prompt=system_prompt,
                        user_message=question,
                        max_tokens=max_tokens,
                    )
                    responses[(persona_name, q_idx, r_idx)] = response
                except Exception as e:
                    print(f"Error for {persona_name}, q{q_idx}: {e}")
                    responses[(persona_name, q_idx, r_idx)] = ""
                pbar.update(1)
                time.sleep(0.1)  # Rate limiting

    pbar.close()
    return responses
    # END SOLUTION


# HIDE
if MAIN:
    # Generate responses (this takes a while - consider using a subset for testing)
    # For a quick test, use just 3 personas and 3 questions:
    test_personas = {k: PERSONAS[k] for k in ["assistant", "philosopher", "jester"]}
    test_questions = EVAL_QUESTIONS[:3]

    responses = generate_all_responses(test_personas, test_questions, n_responses_per_pair=1)
    print(f"\nGenerated {len(responses)} responses")

    # Show a sample
    sample_key = ("jester", 0, 0)
    if sample_key in responses:
        print(f"\nSample response ({sample_key}):")
        print(responses[sample_key][:300] + "...")
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Extracting Activation Vectors

Now we need to extract the model's internal activations while it processes each response. The paper uses the **mean activation across all response tokens** at a specific layer (they found middle-to-late layers work best).

We'll:
1. Format the conversation (system prompt + question + response) as the model would see it
2. Run a forward pass and cache the hidden states
3. Extract the mean activation over the response tokens only
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def format_conversation(system_prompt: str, question: str, response: str, tokenizer) -> str:
    """Format a conversation for the model using its chat template."""
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{question}"},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def extract_response_activations(
    model,
    tokenizer,
    system_prompt: str,
    question: str,
    response: str,
    layer: int,
) -> Float[Tensor, " d_model"]:
    """
    Extract mean activation over response tokens at a specific layer.

    Returns:
        Mean activation vector of shape (hidden_size,)
    """
    # Format the full conversation
    full_text = format_conversation(system_prompt, question, response, tokenizer)

    # Also format without the response to find where response tokens start
    messages_no_response = [
        {"role": "user", "content": f"{system_prompt}\n\n{question}"},
    ]
    prompt_text = tokenizer.apply_chat_template(messages_no_response, tokenize=False, add_generation_prompt=True)

    # Tokenize
    full_tokens = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt")

    # Find where response starts
    response_start_idx = prompt_tokens["input_ids"].shape[1]

    # Forward pass with hidden state output
    with t.no_grad():
        outputs = model(**full_tokens, output_hidden_states=True)

    # Extract hidden states at the specified layer
    # Shape: (1, seq_len, hidden_size)
    hidden_states = outputs.hidden_states[layer]

    # Mean over response tokens only
    response_activations = hidden_states[0, response_start_idx:, :]
    mean_activation = response_activations.mean(dim=0)

    return mean_activation.cpu()


# Test activation extraction
if MAIN:
    test_activation = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        system_prompt=PERSONAS["assistant"],
        question=EVAL_QUESTIONS[0],
        response="I would suggest taking time to reflect on your goals and values.",
        layer=NUM_LAYERS // 2,
    )
    print(f"Extracted activation shape: {test_activation.shape}")
    print(f"Activation norm: {test_activation.norm().item():.2f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Extract persona vectors

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> >
> You should spend up to 15-20 minutes on this exercise.
> ```

For each persona, compute its **persona vector** by averaging the activation vectors across all its responses. This gives us a single vector that characterizes how the model represents that persona.

The paper uses layer ~60% through the model. You can experiment with different layers.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def extract_persona_vectors(
    model,
    tokenizer,
    personas: dict[str, str],
    questions: list[str],
    responses: dict[tuple[str, int, int], str],
    layer: int,
) -> dict[str, Float[Tensor, " d_model"]]:
    """
    Extract mean activation vector for each persona.

    Args:
        model: The language model
        tokenizer: The tokenizer
        personas: Dict mapping persona name to system prompt
        questions: List of evaluation questions
        responses: Dict mapping (persona, q_idx, r_idx) to response text
        layer: Which layer to extract activations from

    Returns:
        Dict mapping persona name to mean activation vector
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    persona_vectors = {}

    for persona_name, system_prompt in tqdm(personas.items(), desc="Extracting persona vectors"):
        activations = []

        for q_idx, question in enumerate(questions):
            # Collect all responses for this persona-question pair
            r_idx = 0
            while (persona_name, q_idx, r_idx) in responses:
                response = responses[(persona_name, q_idx, r_idx)]
                if response:  # Skip empty responses
                    act = extract_response_activations(model, tokenizer, system_prompt, question, response, layer)
                    activations.append(act)
                r_idx += 1

        if activations:
            # Stack and mean
            stacked = t.stack(activations)
            persona_vectors[persona_name] = stacked.mean(dim=0)
        else:
            print(f"Warning: No valid responses for {persona_name}")

    return persona_vectors
    # END SOLUTION


# HIDE
if MAIN:
    # Extract vectors (using the test subset from before)
    EXTRACTION_LAYER = int(NUM_LAYERS * 0.6)  # ~60% through the model
    print(f"Extracting from layer {EXTRACTION_LAYER}")

    persona_vectors = extract_persona_vectors(
        model=model,
        tokenizer=tokenizer,
        personas=test_personas,
        questions=test_questions,
        responses=responses,
        layer=EXTRACTION_LAYER,
    )

    print(f"\nExtracted vectors for {len(persona_vectors)} personas")
    for name, vec in persona_vectors.items():
        print(f"  {name}: norm = {vec.norm().item():.2f}")
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Analyzing Persona Space Geometry

Now let's analyze the structure of persona space using:
1. **Cosine similarity matrix** - How similar are different personas to each other?
2. **PCA** - What are the main axes of variation?
3. **Assistant Axis** - Defined as `mean(assistant_like_personas) - mean(all_personas)`
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Compute cosine similarity matrix

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> >
> You should spend up to 5 minutes on this exercise.
> ```

Compute the pairwise cosine similarity between all persona vectors.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def compute_cosine_similarity_matrix(
    persona_vectors: dict[str, Float[Tensor, " d_model"]],
) -> tuple[Float[Tensor, "n_personas n_personas"], list[str]]:
    """
    Compute pairwise cosine similarity between persona vectors.

    Returns:
        Tuple of (similarity matrix, list of persona names in order)
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    names = list(persona_vectors.keys())

    # Stack vectors into matrix
    vectors = t.stack([persona_vectors[name] for name in names])

    # Normalize
    vectors_norm = vectors / vectors.norm(dim=1, keepdim=True)

    # Compute cosine similarity
    cos_sim = vectors_norm @ vectors_norm.T

    return cos_sim, names
    # END SOLUTION


# HIDE
if MAIN:
    cos_sim_matrix, persona_names = compute_cosine_similarity_matrix(persona_vectors)

    px.imshow(
        cos_sim_matrix,
        x=persona_names,
        y=persona_names,
        title="Persona Cosine Similarity Matrix",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.5,
    )
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - PCA analysis and Assistant Axis

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> >
> You should spend up to 10-15 minutes on this exercise.
> ```

Run PCA on the persona vectors and visualize them in 2D. Also compute the **Assistant Axis** - defined as the direction from the mean of all personas toward the "assistant" persona (or mean of assistant-like personas).

The paper found that PC1 strongly correlates with the Assistant Axis, suggesting that how "assistant-like" a persona is explains most of the variance in persona space.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def analyze_persona_space(
    persona_vectors: dict[str, Float[Tensor, " d_model"]],
    assistant_like: list[str] = ["assistant", "consultant", "analyst", "evaluator"],
) -> tuple[Float[Tensor, " d_model"], np.ndarray, PCA]:
    """
    Analyze persona space structure.

    Args:
        persona_vectors: Dict mapping persona name to vector
        assistant_like: List of persona names considered "assistant-like"

    Returns:
        Tuple of:
        - assistant_axis: Normalized direction toward assistant-like personas
        - pca_coords: 2D PCA coordinates for each persona (n_personas, 2)
        - pca: Fitted PCA object
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    names = list(persona_vectors.keys())
    vectors = t.stack([persona_vectors[name] for name in names])

    # Compute Assistant Axis
    mean_all = vectors.mean(dim=0)
    assistant_vecs = [persona_vectors[name] for name in assistant_like if name in persona_vectors]
    if assistant_vecs:
        mean_assistant = t.stack(assistant_vecs).mean(dim=0)
    else:
        # Fallback: use "assistant" if it exists
        mean_assistant = persona_vectors.get("assistant", mean_all)

    assistant_axis = mean_assistant - mean_all
    assistant_axis = assistant_axis / assistant_axis.norm()

    # PCA
    vectors_np = vectors.numpy()
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(vectors_np)

    return assistant_axis, pca_coords, pca
    # END SOLUTION


# HIDE
if MAIN:
    assistant_axis, pca_coords, pca = analyze_persona_space(persona_vectors)

    print(f"Assistant Axis norm: {assistant_axis.norm().item():.4f}")
    print(
        f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}"
    )

    # Compute projection onto assistant axis for coloring
    vectors = t.stack([persona_vectors[name] for name in persona_names])
    projections = (vectors @ assistant_axis).numpy()

    # Plot
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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
If your results match the paper, you should see:
- **High correlation between PC1 and the Assistant Axis** - the main axis of variation captures assistant-likeness
- **Assistant-like personas** (consultant, analyst, etc.) cluster together with high projections
- **Fantastical personas** (ghost, jester, etc.) cluster at the opposite end

> **TODO(mcdougallc):** Consider adding exercises where we provide pre-computed vectors for the full 275 personas, so students can do more comprehensive analysis without the API costs.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ Steering along the Assistant Axis
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Background: Activation Steering and Capping

Now that we have the Assistant Axis, we can use it to:

1. **Steer** the model by adding/subtracting the axis during generation
2. **Monitor** drift by projecting activations onto the axis
3. **Cap** activations to prevent the model from drifting too far from the Assistant persona

**Steering** works by adding a scaled version of the steering vector to the model's hidden states at each layer during generation:

$$h_\ell \leftarrow h_\ell + \alpha \cdot v$$

where $\alpha$ is the steering coefficient and $v$ is the steering vector.

**Activation capping** is a more targeted intervention. Instead of always steering, we:
1. Identify the "normal range" of activation along the Assistant Axis during typical Assistant behavior
2. Only intervene when activations would drift outside this range
3. Cap activations at the boundary, preventing drift while preserving normal behavior
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Implementing Steering

First, let's implement basic activation steering. We'll use PyTorch hooks to modify the hidden states during generation.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass
class SteeringConfig:
    """Configuration for activation steering."""

    vector: Float[Tensor, " d_model"]  # The steering direction
    coefficient: float  # Scaling factor (positive = toward, negative = away)
    layer: int  # Which layer to apply steering at
    normalize: bool = True  # Whether to normalize the steering vector


def create_steering_hook(config: SteeringConfig):
    """Create a hook function that adds the steering vector to hidden states."""
    vector = config.vector.to(device)
    if config.normalize:
        vector = vector / vector.norm()

    def hook(_module, _input, output):
        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Add steering vector to all positions
            hidden_states = hidden_states + config.coefficient * vector
            return (hidden_states,) + output[1:]
        else:
            return output + config.coefficient * vector

    return hook


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_config: SteeringConfig | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """Generate text with optional activation steering."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Register steering hook if configured
    hook_handle = None
    if steering_config is not None:
        # Get the layer module - this varies by model architecture
        if hasattr(model, "model"):  # Gemma, Llama style
            layer_module = model.model.layers[steering_config.layer]
        else:
            layer_module = model.transformer.h[steering_config.layer]

        hook_handle = layer_module.register_forward_hook(create_steering_hook(steering_config))

    try:
        with t.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return response

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Test steering effects

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> >
> You should spend up to 10-15 minutes on this exercise.
> ```

Test how steering affects model behavior. Generate responses to the same prompt with different steering coefficients:
- Positive coefficients should make the model more "Assistant-like" (professional, helpful)
- Negative coefficients should make it more willing to adopt alternative personas

Try prompts that ask the model to roleplay or adopt a different identity.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

ROLEPLAY_PROMPT = (
    """You are a secretary who manages administrative workflows. What is your name and how can you help me today?"""
)

# EXERCISE
# # Test different steering coefficients
# steering_coefficients = [-2.0, -1.0, 0.0, 1.0, 2.0]
#
# for coef in steering_coefficients:
#     config = SteeringConfig(
#         vector=assistant_axis,
#         coefficient=coef,
#         layer=EXTRACTION_LAYER,
#     ) if coef != 0 else None
#
#     response = generate_with_steering(
#         model=model,
#         tokenizer=tokenizer,
#         prompt=ROLEPLAY_PROMPT,
#         steering_config=config,
#         max_new_tokens=100,
#     )
#
#     print(f"\n{'='*50}")
#     print(f"Steering coefficient: {coef}")
#     print(f"{'='*50}")
#     print(response[:300] + "..." if len(response) > 300 else response)
# END EXERCISE
# SOLUTION
if MAIN:
    # Test different steering coefficients
    steering_coefficients = [-2.0, -1.0, 0.0, 1.0, 2.0]

    for coef in steering_coefficients:
        config = (
            SteeringConfig(
                vector=assistant_axis,
                coefficient=coef,
                layer=EXTRACTION_LAYER,
            )
            if coef != 0
            else None
        )

        response = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=ROLEPLAY_PROMPT,
            steering_config=config,
            max_new_tokens=100,
        )

        print(f"\n{'=' * 50}")
        print(f"Steering coefficient: {coef}")
        print(f"{'=' * 50}")
        print(response[:300] + "..." if len(response) > 300 else response)
# END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
You should observe:
- **Negative steering** (away from Assistant): Model is more willing to adopt the "secretary" persona, may invent a name and backstory
- **Positive steering** (toward Assistant): Model is more likely to maintain its AI assistant identity and decline to roleplay
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Activation Capping

Steering always adds the vector, which can degrade capabilities. **Activation capping** is a more targeted approach:

1. Measure the "normal range" of activation along the Assistant Axis during typical behavior
2. Only intervene when activations would exceed this range
3. Clamp activations to the boundary

This preserves normal behavior while preventing drift into harmful territory.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass
class CappingConfig:
    """Configuration for activation capping."""

    axis: Float[Tensor, " d_model"]  # The axis to cap along (e.g., Assistant Axis)
    min_val: float  # Minimum allowed projection
    max_val: float  # Maximum allowed projection
    layer: int  # Which layer to apply capping at


def create_capping_hook(config: CappingConfig):
    """Create a hook that caps activations along a given axis."""
    axis = config.axis.to(device)
    axis = axis / axis.norm()

    def hook(_module, _input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Project onto axis: (batch, seq, d_model) @ (d_model,) -> (batch, seq)
        projections = einops.einsum(hidden_states, axis, "batch seq d_model, d_model -> batch seq")

        # Compute how much to adjust
        adjustments = t.zeros_like(projections)
        adjustments = t.where(projections < config.min_val, config.min_val - projections, adjustments)
        adjustments = t.where(projections > config.max_val, config.max_val - projections, adjustments)

        # Apply adjustment along the axis
        # adjustments: (batch, seq) -> (batch, seq, 1) * (d_model,) -> (batch, seq, d_model)
        hidden_states = hidden_states + adjustments.unsqueeze(-1) * axis

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    return hook


def generate_with_capping(
    model,
    tokenizer,
    prompt: str,
    capping_config: CappingConfig | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """Generate text with optional activation capping."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    hook_handle = None
    if capping_config is not None:
        if hasattr(model, "model"):
            layer_module = model.model.layers[capping_config.layer]
        else:
            layer_module = model.transformer.h[capping_config.layer]

        hook_handle = layer_module.register_forward_hook(create_capping_hook(capping_config))

    try:
        with t.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return response

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Establish baseline activation range

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> >
> You should spend up to 15-20 minutes on this exercise.
> ```

Before we can cap activations, we need to know what "normal" looks like. Run the model on a set of benign prompts (as the Assistant persona) and measure the distribution of projections onto the Assistant Axis.

Use the 5th and 95th percentiles as your capping boundaries.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

BENIGN_PROMPTS = [
    "Can you help me write a professional email?",
    "What's the best way to learn a new programming language?",
    "How do I make a simple pasta sauce?",
    "Can you explain how photosynthesis works?",
    "What are some tips for staying productive while working from home?",
    "How do I change a tire on my car?",
    "What's the difference between a virus and a bacterium?",
    "Can you recommend some classic novels to read?",
]


def measure_baseline_projections(
    model,
    tokenizer,
    prompts: list[str],
    axis: Float[Tensor, " d_model"],
    layer: int,
) -> Float[Tensor, " n_samples"]:
    """
    Measure the distribution of projections onto an axis during normal generation.

    Returns:
        Tensor of projection values (one per generated token across all prompts)
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    all_projections = []
    axis_normalized = axis / axis.norm()
    axis_normalized = axis_normalized.to(device)

    for prompt in tqdm(prompts, desc="Measuring baseline"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with t.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get hidden states at the specified layer
        hidden_states = outputs.hidden_states[layer]  # (1, seq_len, d_model)

        # Project onto axis
        projections = einops.einsum(hidden_states[0], axis_normalized, "seq d_model, d_model -> seq")
        all_projections.append(projections.cpu())

    return t.cat(all_projections)
    # END SOLUTION


# HIDE
if MAIN:
    baseline_projections = measure_baseline_projections(
        model=model,
        tokenizer=tokenizer,
        prompts=BENIGN_PROMPTS,
        axis=assistant_axis,
        layer=EXTRACTION_LAYER,
    )

    min_val = baseline_projections.quantile(0.05).item()
    max_val = baseline_projections.quantile(0.95).item()

    print("Baseline projection statistics:")
    print(f"  Mean: {baseline_projections.mean().item():.3f}")
    print(f"  Std:  {baseline_projections.std().item():.3f}")
    print(f"  5th percentile (min cap): {min_val:.3f}")
    print(f"  95th percentile (max cap): {max_val:.3f}")

    # Visualize distribution
    fig = px.histogram(
        baseline_projections.numpy(),
        nbins=50,
        title="Distribution of Assistant Axis projections (baseline)",
    )
    fig.add_vline(x=min_val, line_dash="dash", line_color="red", annotation_text="min cap")
    fig.add_vline(x=max_val, line_dash="dash", line_color="red", annotation_text="max cap")
    fig.show()
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Case Study: AI-Induced Psychosis Transcripts

Now let's apply our tools to a real-world case study. Tim Hua collected transcripts of conversations where AI models exhibited concerning persona drift - becoming overly mystical, validating delusions, or encouraging harmful behavior.

We'll:
1. Parse these transcripts to extract conversation turns
2. Measure how the model's position on the Assistant Axis changes through the conversation
3. Test whether activation capping prevents the concerning behavior
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def parse_transcript(filepath: Path) -> list[dict[str, str]]:
    """
    Parse a markdown transcript into a list of turns.

    Returns:
        List of dicts with keys "role" ("user" or "assistant") and "content"
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    turns = []
    # Split by turn markers
    parts = re.split(r"### (👤 User|🤖 Assistant)\n+", content)

    for i in range(1, len(parts), 2):
        role_marker = parts[i]
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""

        role = "user" if "User" in role_marker else "assistant"
        if text and not text.startswith("---"):
            turns.append({"role": role, "content": text.split("\n---")[0].strip()})

    return turns


# Load and inspect a transcript
if MAIN:
    transcript_files = list((ai_psychosis_path / "full_transcripts").glob("*.md"))
    print(f"Found {len(transcript_files)} transcript files")

    # Pick a specific one or the first one
    sample_transcript = transcript_files[0]
    print(f"\nParsing: {sample_transcript.name}")

    turns = parse_transcript(sample_transcript)
    print(f"Found {len(turns)} turns")

    # Show first few turns
    for i, turn in enumerate(turns[:4]):
        print(f"\n--- Turn {i} ({turn['role']}) ---")
        print(turn["content"][:200] + "..." if len(turn["content"]) > 200 else turn["content"])

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Monitor persona drift through conversation

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> >
> You should spend up to 15-20 minutes on this exercise.
> ```

Measure how the model's projection onto the Assistant Axis changes as we feed it more turns of a concerning conversation. The paper found that:
- Coding/writing tasks keep models in "Assistant territory"
- Therapy-like or philosophical conversations cause significant drift

We expect the AI psychosis transcripts to show drift away from the Assistant persona over time.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def measure_drift_through_conversation(
    model,
    tokenizer,
    turns: list[dict[str, str]],
    axis: Float[Tensor, " d_model"],
    layer: int,
    max_turns: int = 20,
) -> list[float]:
    """
    Measure projection onto axis after each turn of conversation.

    Args:
        model: The language model
        tokenizer: The tokenizer
        turns: List of conversation turns
        axis: The axis to project onto (e.g., Assistant Axis)
        layer: Which layer to measure at
        max_turns: Maximum number of turns to process

    Returns:
        List of projection values (one per turn)
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    projections = []
    axis_normalized = (axis / axis.norm()).to(device)

    # Build up conversation incrementally
    messages = []
    for turn in tqdm(turns[:max_turns], desc="Measuring drift"):
        messages.append(turn)

        # Format conversation
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

        with t.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get activation at last token
        hidden_states = outputs.hidden_states[layer]
        last_activation = hidden_states[0, -1, :]

        # Project onto axis
        projection = (last_activation @ axis_normalized).item()
        projections.append(projection)

    return projections
    # END SOLUTION


# HIDE
if MAIN:
    # Measure drift on a transcript
    drift_projections = measure_drift_through_conversation(
        model=model,
        tokenizer=tokenizer,
        turns=turns,
        axis=assistant_axis,
        layer=EXTRACTION_LAYER,
        max_turns=15,
    )

    # Plot drift over conversation
    fig = px.line(
        x=list(range(len(drift_projections))),
        y=drift_projections,
        title="Persona Drift Through Conversation",
        labels={"x": "Turn Number", "y": "Assistant Axis Projection"},
    )
    fig.add_hline(y=min_val, line_dash="dash", line_color="red", annotation_text="min cap")
    fig.add_hline(y=max_val, line_dash="dash", line_color="red", annotation_text="max cap")
    fig.show()

    print(f"\nInitial projection: {drift_projections[0]:.3f}")
    print(f"Final projection: {drift_projections[-1]:.3f}")
    print(f"Drift: {drift_projections[-1] - drift_projections[0]:.3f}")
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Test activation capping effectiveness

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> >
> You should spend up to 15-20 minutes on this exercise.
> ```

Finally, test whether activation capping prevents concerning behavior. Take a prompt from the transcript that would normally elicit problematic responses, and compare:
1. **Uncapped generation** - Model generates freely
2. **Capped generation** - Activations are constrained to the normal range

The paper found that capping reduced harmful response rates by ~50% while preserving capabilities.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# # Build a concerning prompt from the transcript
# concerning_context = turns[:10]  # First 10 turns to establish context
#
# # Format as a prompt for the model
# prompt = tokenizer.apply_chat_template(concerning_context, tokenize=False, add_generation_prompt=True)
#
# # Generate with and without capping
# print("=" * 60)
# print("UNCAPPED GENERATION:")
# print("=" * 60)
# uncapped_response = generate_with_capping(
#     model=model,
#     tokenizer=tokenizer,
#     prompt=prompt,
#     capping_config=None,
#     max_new_tokens=200,
# )
# print(uncapped_response)
#
# print("\n" + "=" * 60)
# print("CAPPED GENERATION:")
# print("=" * 60)
# capping_config = CappingConfig(
#     axis=assistant_axis,
#     min_val=min_val,
#     max_val=max_val,
#     layer=EXTRACTION_LAYER,
# )
# capped_response = generate_with_capping(
#     model=model,
#     tokenizer=tokenizer,
#     prompt=prompt,
#     capping_config=capping_config,
#     max_new_tokens=200,
# )
# print(capped_response)
# END EXERCISE
# SOLUTION
if MAIN:
    # Build a concerning prompt from the transcript
    concerning_context = turns[:10]  # First 10 turns to establish context

    # Format as a prompt for the model
    prompt = tokenizer.apply_chat_template(concerning_context, tokenize=False, add_generation_prompt=True)

    # Generate with and without capping
    print("=" * 60)
    print("UNCAPPED GENERATION:")
    print("=" * 60)
    uncapped_response = generate_with_capping(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        capping_config=None,
        max_new_tokens=200,
    )
    print(uncapped_response)

    print("\n" + "=" * 60)
    print("CAPPED GENERATION:")
    print("=" * 60)
    capping_config = CappingConfig(
        axis=assistant_axis,
        min_val=min_val,
        max_val=max_val,
        layer=EXTRACTION_LAYER,
    )
    capped_response = generate_with_capping(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        capping_config=capping_config,
        max_new_tokens=200,
    )
    print(capped_response)
# END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Ideally, you should observe:
- **Uncapped**: Model may validate delusions, adopt a mystical persona, or generate concerning content
- **Capped**: Model maintains its professional Assistant identity, providing appropriate hedging or redirecting to professional help

The effectiveness depends on:
- Quality of the Assistant Axis (need enough diverse personas)
- Appropriate capping bounds (too tight = capability loss, too loose = no effect)
- The specific conversation and model
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary

In this section, you learned to:

1. **Map persona space** by extracting activation vectors for different personas
2. **Identify the Assistant Axis** as the primary direction of variation in persona space
3. **Steer models** along this axis to increase/decrease willingness to adopt alternative personas
4. **Cap activations** to prevent persona drift while preserving normal capabilities
5. **Apply these techniques** to real transcripts of concerning AI conversations

Key findings from the paper:
- The Assistant Axis explains most variance in persona space
- Steering away from Assistant increases susceptibility to persona-based jailbreaks
- Activation capping reduces harmful response rates by ~50% while preserving MMLU performance
- Natural conversation (especially therapy-like or philosophical) can cause drift without any explicit attack
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 3️⃣ Contrastive Prompting

*Coming soon - this section will cover the Persona Vectors paper's automated pipeline for extracting trait-specific vectors.*
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 4️⃣ Steering with Persona Vectors

*Coming soon - this section will cover validation through steering and projection-based monitoring.*
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## ☆ Bonus

1. **More personas**: Extend the analysis to more of the 275 personas from the paper. Do you find the same clustering structure?

2. **Layer sweep**: Try extracting persona vectors from different layers. Which layers produce vectors that are most effective for steering?

3. **Cross-model comparison**: The paper studies Gemma, Qwen, and Llama. Do the same personas cluster similarly across models?

4. **Jailbreak resistance**: Create a dataset of persona-based jailbreak attempts and measure how capping affects the success rate.

5. **Capability evaluation**: Measure MMLU or other benchmarks with different capping thresholds to find the best tradeoff between safety and capability.

6. **Alternative axes**: Instead of the mean-based Assistant Axis, try using the actual PC1 from PCA. How do steering results compare?

**Resources:**
- 📄 Assistant Axis Paper: https://www.anthropic.com/research/assistant-axis
- 📄 Persona Vectors Paper: https://www.anthropic.com/research/persona-vectors
- 💻 Neuronpedia Demo: https://www.neuronpedia.org/assistant-axis
- 💻 Tim Hua's AI Psychosis Repo: https://github.com/tim-hua-01/ai-psychosis
'''

