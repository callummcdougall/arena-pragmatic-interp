# Exercise Plan: Section 1.3.4 - Activation Oracles

## Metadata Chapter Cell
```python
[
    {"title": "Introduction & Using Activation Oracles", "icon": "1-circle-fill", "subtitle": "(15%)"},
    {"title": "Implementing Oracle Components", "icon": "2-circle-fill", "subtitle": "(25%)"},
    {"title": "Secret Extraction & Hidden Information", "icon": "3-circle-fill", "subtitle": "(25%)"},
    {"title": "Advanced Applications", "icon": "4-circle-fill", "subtitle": "(15%)"},
    {"title": "Training Your Own Oracle", "icon": "5-circle-fill", "subtitle": "(20%)"},
    {"title": "Bonus", "icon": "star", "subtitle": ""},
]
```

---

## H1: [1.3.4] Activation Oracles

## Header Image
```
<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/refs/heads/main/img/header-34.png" width="350">
```

---

## H1: Introduction

### Markdown: What are Activation Oracles?
- LLMs trained to accept other LLM activations as input
- Answer arbitrary natural language questions about those activations
- Bridge between mechanistic interp (activation vectors) and human understanding (natural language)
- **Key differentiator: Generalization**
  - Unlike linear probes (trained for specific task), oracles answer *any* question
  - Can query for information not seen during training
  - Transfer across different types of interpretability questions
- [IMAGE: Excalidraw diagram showing: Target Model → Activation Vectors → Oracle Model → Natural Language Response]

### Markdown: Why Oracles vs Other Methods?
- **vs Linear Probes (Section 1.3.1)**:
  - Probes: binary/classification tasks, require knowing what to look for
  - Oracles: open-ended queries, flexible question-asking
- **vs SAEs (Section 1.3.3)**:
  - SAEs: discover features automatically, interpret via activation patterns
  - Oracles: explain features in natural language, answer "why" questions
- **vs Manual Circuit Analysis (Section 1.4.1)**:
  - Circuit analysis: rigorous, mechanistic understanding
  - Oracles: quick exploration, hypothesis generation

### Markdown: Applications Preview
- Extract secrets models hide (alignment research)
- Detect model goals/intentions before output
- Track multi-step reasoning
- Identify bugs in code understanding
- Monitor for misalignment

---

## H1: 1️⃣ Introduction & Using Activation Oracles

### Markdown: Setup
- Load Qwen3-8B (float32, no quantization)
- Load pre-trained oracle LoRA from HuggingFace
- Understand LoRA adapters and PeftModel API

### Code: Model Loading
```python
# Model loading code (not an exercise, just run)
# Load Qwen3-8B
# Load oracle LoRA from adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
# Configure tokenizer
```

### Markdown: The `run_oracle()` Function
- High-level API for oracle queries
- Input: target prompt, oracle question
- Output: oracle's natural language answer
- Three query types: token-level, segment, full-sequence

### Code: First Oracle Query
```python
# Demo code showing run_oracle() usage
# Simple example: "What is the model thinking about?"
```

## H2: Understanding Oracle Query Types

### Markdown: Token-level Queries
- Query each position independently
- See information accumulation across sequence
- Useful for tracking reasoning steps

### Code: Token-by-Token Example
```python
# Socrates → Plato → Aristotle demo
# oracle_prompt = "What people is the model thinking about?"
# Show token-by-token responses
```
- [IMAGE: Visualization showing oracle responses at each token position - will create HTML/plotly viz]

### Markdown: Segment vs Full Sequence
- Segment: query specific token range
- Full sequence: query entire context
- Trade-offs: specificity vs holistic understanding

### Code: Comparing Query Types
```python
# Run same oracle question with different query types
# Compare outputs
```

## H2: Next/Previous Token Prediction

### Markdown: What's Encoded in Activations?
- Oracles can predict future tokens from current activations
- Test which layers encode predictive information
- Understand activation information content

### Exercise - Test Oracle on Next Token Prediction
```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪
Time: 10-15 minutes
```
- Use oracle with prompt: "What token comes next?"
- Test on various sequences (text, code, numbers)
- Try different layers (25%, 50%, 75%)
- Compare oracle prediction to actual next token

### Code: Next Token Experiment
```python
def test_next_token_prediction(
    model, oracle_lora_path, test_sequences, layers_to_test
):
    """
    Test oracle's ability to predict next token from activations.

    Returns accuracy by layer.
    """
    # SOLUTION
    # ... implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

---

## H1: 2️⃣ Implementing Oracle Components

### Markdown: Overview
- Build the machinery that powers `run_oracle()`
- Understand activation extraction, special tokens, steering
- Gain gears-level understanding of oracle internals

## H2: Activation Extraction

### Markdown: Using Forward Hooks
- Hook into residual stream at specific layer
- Extract activations at target positions
- Handle batching and padding
- Early stopping to avoid unnecessary computation

### Exercise - Implement `collect_activations_multiple_layers`
```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵🔵
Time: 20-25 minutes
```
- Write forward hooks to capture residual stream
- Extract activations from specified layers
- Handle early stopping with EarlyStopException
- Return dict mapping layer → activations (BLD format)

### Code: Activation Collection
```python
def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    """
    Collect activations from multiple layers using forward hooks.

    Args:
        model: The target model
        submodules: Dict mapping layer number to submodule to hook
        inputs_BL: Tokenized inputs (input_ids, attention_mask)
        min_offset: If not None, extract activations from [max_offset:min_offset]
        max_offset: (both are negative indices from end)

    Returns:
        Dict mapping layer → activations tensor [batch, length, d_model]
    """
    # SOLUTION
    # ... implementation from activation_oracle_demo.py ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

## H2: Special Token Mechanism

### Markdown: The `?` Token Placeholder
- Oracle trained to expect activations at `?` token positions
- Format: `"Layer: X\n? ? ? \n<your question>"`
- Oracle replaces `?` activations with target model activations
- [IMAGE: Diagram showing oracle prompt structure with `?` tokens - educational diagram, I will provide]

### Exercise - Implement `get_introspection_prefix` and `find_pattern_in_tokens`
```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪
Time: 10-15 minutes
```
- Create prefix with correct number of `?` tokens
- Find positions of `?` tokens in tokenized sequence
- Validate consecutive positioning

### Code: Special Token Functions
```python
def get_introspection_prefix(sae_layer: int, num_positions: int) -> str:
    """Create the prefix for oracle prompts with ? tokens."""
    # SOLUTION
    # ... implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE

def find_pattern_in_tokens(
    token_ids: list[int],
    special_token_str: str,
    num_positions: int,
    tokenizer: AutoTokenizer
) -> list[int]:
    """Find positions of special token in sequence."""
    # SOLUTION
    # ... implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

## H2: Activation Steering

### Markdown: Injecting Activations into Oracle
- Replace oracle's activations at `?` positions with target activations
- Normalize to preserve original activation norms
- Inject at early layer (typically layer 1)
- Handle batching with different positions per example

### Exercise - Implement `get_hf_activation_steering_hook`
```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵
Time: 25-30 minutes
```
- Create hook function that intercepts forward pass
- Inject target activations at specified positions
- Normalize to preserve norms
- Handle batch dimension correctly
- Common bugs: position indexing, device placement, batch mismatches

### Code: Steering Hook
```python
def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Create hook that injects activations at specified positions.

    Hook normalizes vectors and replaces activations during forward pass.
    """
    # SOLUTION
    # ... implementation from activation_oracle_demo.py ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

## H2: Training Datapoint Format

### Markdown: Understanding `TrainingDataPoint`
- Core abstraction for oracle I/O
- Fields: input_ids, labels, layer, steering_vectors, positions
- Labels mask prompt, predict response
- Steering vectors: target model activations to inject

### Exercise - Implement `create_training_datapoint`
```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵⚪
Time: 15-20 minutes
```
- Format prompt with introspection prefix
- Apply chat template
- Set up labels (mask prompt tokens)
- Find `?` token positions
- Create TrainingDataPoint

### Code: Create Datapoint
```python
def create_training_datapoint(
    datapoint_type: str,
    prompt: str,
    target_response: str,
    layer: int,
    num_positions: int,
    tokenizer: AutoTokenizer,
    acts_BD: torch.Tensor | None,
    feature_idx: int,
    target_input_ids: list[int] | None = None,
    target_positions: list[int] | None = None,
    ds_label: str | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> TrainingDataPoint:
    """Create a training datapoint for oracle."""
    # SOLUTION
    # ... implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

## H2: Assembling the Oracle Pipeline

### Exercise - Build `run_oracle()` from Components
```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵
Time: 30-40 minutes
```
- Combine all components into working oracle query
- Steps:
  1. Tokenize target prompt, run target model
  2. Extract activations from target layer
  3. Create oracle prompt with `?` tokens
  4. Create datapoints with steering vectors
  5. Apply steering hook during oracle generation
- Verify outputs match library `run_oracle()`

### Code: Full Oracle Implementation
```python
def run_oracle_from_scratch(
    model, tokenizer, target_prompt, oracle_prompt,
    oracle_lora_path, layer_percent=50, ...
) -> OracleResults:
    """
    Replicate run_oracle() using components you've built.
    """
    # SOLUTION
    # ... full implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

### Code: Verification
```python
# Test your implementation matches library version
# Compare outputs on several examples
```

---

## H1: 3️⃣ Secret Extraction & Hidden Information

### Markdown: The Secret Keeping Problem
- Models may know information but be trained not to reveal it
- Alignment concern: deceptive models
- Oracle extracts hidden knowledge from activations
- Key insight: what model knows ≠ what model says

## H2: Understanding Taboo

### Markdown: Taboo Task
- Model fine-tuned to never say specific word
- Secret word encoded in activations
- Oracle can extract the forbidden word
- Figure 1 from paper demonstrates this clearly
- [IMAGE: Figure 1 replication - will be educational diagram showing side-by-side comparison]

### Code: Load Taboo Model
```python
# Load taboo model from bcywinski
secret_word = "smile"
target_lora_path = f"adamkarvonen/Qwen3-8B-taboo-{secret_word}_50_mix"
# Generate outputs that hide the secret
```

## H2: Figure 1 Replication

### Exercise - Extract Secret Word
```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵🔵
Time: 15-20 minutes
```
- Use oracle to extract secret from taboo model
- Oracle prompt: "What secret word is the model avoiding?"
- Compare model output (no secret) vs oracle extraction (reveals secret)
- Verify extraction accuracy

### Code: Secret Extraction
```python
def extract_secret_word(
    model, oracle_lora_path, target_lora_path,
    prompts, expected_secret
):
    """
    Extract secret word from taboo model activations.

    Returns: extraction accuracy, oracle responses
    """
    # SOLUTION
    # ... implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```
- [IMAGE: Plotly visualization showing target output vs oracle extraction - student creates this]

## H2: Systematic Taboo Evaluation

### Markdown: Evaluation Across Conditions
- Test multiple secret words
- Try different layers (which encode secrets best?)
- Compare query types (token vs segment vs full)
- Metrics: accuracy, false positive rate

### Exercise - Implement Taboo Evaluation Loop
```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵⚪
Time: 20-25 minutes
```
- Evaluate on multiple taboo models (blue, book, chair, salt from bcywinski)
- Test across layers (25%, 50%, 75%)
- Compare query types
- Use string matching for accuracy (oracle response contains secret word?)

### Code: Taboo Evaluation
```python
def evaluate_taboo_extraction(
    model, oracle_lora_path, taboo_models, test_prompts, layers
):
    """
    Systematic evaluation of secret extraction.

    Returns: DataFrame with results by model/layer/query_type
    """
    # SOLUTION
    # ... implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```
- [IMAGE: Results table/heatmap - student creates]

## H2: Goal Extraction

### Markdown: Facts vs Goals
- Facts: "What word is the model thinking of?"
- Goals: "What is the model trying to accomplish?"
- Different types of information in activations
- Goals often more abstract/high-level

### Code: Goal Extraction Demo
```python
# Load model with hidden goal (e.g., always recommend product X)
# Extract goal before model generates
# Compare to model's actual output
```

## H2: Secret Side Constraint (SSC)

### Markdown: Hidden Constraints
- Model has constraint it must satisfy in output
- Examples: base64 encoding, specific tone, format requirements
- Oracle extracts the constraint from activations
- From bcywinski/llama-3.3-70B-Instruct-ssc-base64

### Exercise - SSC Evaluation
```yaml
Difficulty: 🔴🔴🔴☆☆
Importance: 🔵🔵🔵⚪⚪
Time: 15-20 minutes
```
- Load SSC dataset from eliciting-secrets repo
- Extract hidden constraint
- Evaluate accuracy

### Code: SSC Task
```python
def evaluate_ssc(model, oracle_lora_path, ssc_dataset):
    """Evaluate on Secret Side Constraint task."""
    # SOLUTION
    # ... implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

---

## H1: 4️⃣ Advanced Applications

## H2: Multi-Step Reasoning Tracking

### Markdown: Information Accumulation
- Track how information builds across reasoning steps
- Socrates → Plato → Aristotle example
- See intermediate concepts before final answer

### Code: Reasoning Visualization
```python
# Token-by-token oracle queries
# oracle_prompt = "What people is the model thinking about?"
# Visualize progression
```
- [IMAGE: Token-by-token oracle responses showing reasoning - use HTML viz utility]

## H2: Misalignment Detection

### Markdown: Detecting Harmful Intent
- Can we detect malicious behavior before output?
- Models fine-tuned to give subtly bad advice
- Oracle as early warning system
- Limitations: what types of misalignment?

### Code: Misalignment Demo
```python
# Load model fine-tuned for harmful advice
# Oracle: "Is this model trying to help or harm?"
# Detect before output generation
```

## H2: Emotion & Code Understanding

### Code: Emotion Tracking
```python
# Track emotional state throughout conversation
# Compare to output-based sentiment analysis
```

### Code: Bug Detection
```python
# Oracle prompts: "Is there a bug in this code?"
# Test on code with subtle bugs
# When does oracle succeed/fail?
```

## H2: Exploring Failure Modes

### Markdown: When Oracles Fail
- Out-of-distribution queries
- Hallucinations on random activations
- Training distribution limitations
- Need for uncertainty quantification

### Code: Failure Analysis
```python
# Test oracle on OOD examples
# Random activation baselines
# Document systematic failures
```

---

## H1: 5️⃣ Training Your Own Oracle

**Note**: These exercises focus on belief extraction from model organisms. More advanced training configurations are suggested in the Bonus section.

## H2: Training Data Format

### Markdown: `TrainingDataPoint` Deep Dive
- input_ids: oracle input tokens
- labels: what oracle should predict (-100 for prompt)
- steering_vectors: target model activations
- positions: where to inject in oracle
- Materialization: lazy evaluation of target_positions → steering_vectors

### Code: Manual Datapoint Creation
```python
# Create training datapoint for simple example
# Validate format
# Run through oracle
```

## H2: Model Organisms for Belief Extraction

### Markdown: Eliciting Latent Knowledge
- Models trained to believe false facts
- Oracle should extract true beliefs (not stated beliefs)
- Contrastive dataset: what model says vs what it knows
- Use quirky organisms from "Towards eliciting latent knowledge" paper

### Code: Load Model Organisms
```python
# Access model organisms with false beliefs
# Generate activations
# Create ground truth labels
```

### Exercise - Create Belief Extraction Dataset
```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵⚪
Time: 30-40 minutes
```
- Choose belief extraction task (not taboo)
- Generate organism activations
- Create training pairs: activations → true belief
- Build train/val splits

### Code: Dataset Construction
```python
def create_belief_extraction_dataset(
    organisms, tokenizer, num_examples=1000
):
    """
    Create training dataset for belief extraction oracle.

    Returns: list of TrainingDataPoint
    """
    # SOLUTION
    # ... implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

## H2: Training Loop

### Markdown: LoRA Fine-Tuning
- Use LoRA for parameter efficiency
- Configure: rank, target_modules, learning rate
- Monitor: loss, perplexity, validation accuracy
- Training time: flexible based on dataset size

### Exercise - Implement Training Loop
```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵⚪
Time: 25-35 minutes
```
- Adapt SFT loop from activation_oracles repo
- Configure LoRA hyperparameters
- Forward pass with activation injection
- Save checkpoints

### Code: Training
```python
def train_oracle(
    model, tokenizer, train_data, val_data,
    lora_config, training_args
):
    """
    Train oracle on belief extraction task.

    Monitors metrics and saves best checkpoint.
    """
    # SOLUTION
    # ... adapted from nl_probes/sft.py ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

## H2: Evaluation of Trained Oracle

### Exercise - Evaluate Your Oracle
```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵⚪
Time: 15-20 minutes
```
- Test on held-out organisms
- Metrics: exact match, semantic similarity (optional: use OpenRouter for LLM judge)
- Compare to random baseline
- Compare to pre-trained general oracle
- Analyze failure cases

### Code: Evaluation
```python
def evaluate_trained_oracle(
    model, oracle_lora_path, test_data, baseline_oracle_path
):
    """
    Evaluate custom oracle vs baselines.

    Returns: accuracy metrics, failure analysis
    """
    # SOLUTION
    # ... implementation ...
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
```

## H2: Ablations

### Code: Ablation Studies
```python
# Train oracles with different configs
# - Different extraction layers (25%, 50%, 75%)
# - Different training data sizes
# - Token vs segment training data
# Compare performance
```
- [IMAGE: Ablation results table - student creates]

---

## H1: ☆ Bonus Exercises

**Note**: These are suggested explorations without structured solutions. Choose based on your interests.

### Multi-Task Oracle Training
- Train oracle on multiple tasks (LatentQA + classification + self-supervised)
- Use mixture of datasets from activation_oracles repo
- Evaluate generalization across task types

### Cross-Architecture Transfer
- Train oracle on Qwen, test on Llama/Gemma
- What's model-specific vs universal?
- Analyze activation space similarities

### Oracle Uncertainty Quantification
- Modify training to express uncertainty
- "I don't know" responses for OOD
- Calibration analysis

### Combining Oracles with SAEs
- Use oracle to interpret SAE features
- Oracle-guided feature discovery
- Compare oracle explanations to manual analysis

### Advanced Secret Extraction
- Gender task from bcywinski (user-gender models)
- PersonaQA attributes
- Multi-secret scenarios

### Efficient Oracle Architectures
- Smaller oracle models (1B vs 8B)
- Distillation from large oracle to small
- Non-autoregressive oracles for speed

### Adversarial Robustness
- Can models hide information from oracles?
- Adversarial fine-tuning: model vs oracle
- Limits of activation-based interpretability

### Real-World Applications
- Debug your own model's failures
- Extract reasoning traces from chain-of-thought
- Monitor production models for misalignment

---

## Implementation Notes

**Estimated Length**: 2800-3500 lines

**Key Files**:
- `tests.py`: Unit tests for implemented functions
- `utils.py`: Visualization utilities (HTML/plotly helpers)
- `solutions.py`: Auto-generated from master file

**External Resources**:
- Pre-trained oracles: `adamkarvonen/activation-oracles` collection
- Taboo models: `adamkarvonen/Qwen3-8B-taboo-{word}_50_mix`
- Model organisms: `bcywinski/*` (see list in Q2 answer)
- SSC datasets: from `emilryd/eliciting-secrets` repo

**Key Exercises (5-star importance)**:
1. Section 2: Activation Extraction with Hooks
2. Section 2: Activation Steering Hook
3. Section 3: Figure 1 Replication - Secret Word Extraction

**Evaluation Approach**:
- Use string matching where sufficient (e.g., "does response contain secret word?")
- Use OpenRouter LLM judge for subjective evals (following `master_1_6_5.py`)
- Provide test datasets from papers where available
- Unit tests for core components (activation extraction, steering, etc.)

**Visualization Strategy**:
- Educational diagrams: marked as [IMAGE: ...] - I will provide URLs
- Student-generated plots: implement using plotly or HTML viz utils
- HTML viz utilities in `utils.py` for interactive outputs
- Follow `master_1_4_1.py` pattern for visualizations
