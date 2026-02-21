# Section 3️⃣ Attribution Graphs - Exercise Plan

## Overview

This section extends the gradient-based methods from section 1️⃣ and the transcoder work from section 2️⃣ into a full attribution graph framework. Students first implement the core circuit tracing algorithm themselves (Part 1), then use the `circuit-tracer` library to explore real circuits and perform feature interventions (Part 2).

**Model**: Gemma 2-2B with GemmaScope single-layer transcoders (switching from GPT-2, which was used in sections 1️⃣ and 2️⃣).

**Key dependency**: The `circuit-tracer` library from `decoderesearch` (`pip install circuit-tracer` or clone from GitHub).

---

## Connection Points to Earlier Sections

### From Section 1️⃣ (Latent Gradients)
- Students already know how to compute latent-to-latent gradients via `jacrev`, and latent-to-logit gradients.
- They understand `use_error_term` and the distinction between observing vs intervening.
- They've seen that latent connections are sparse, and that this sparsity is evidence for SAE latents being fundamental units of computation.
- **Key callback**: Section 3️⃣ scales up the gradient approach from section 1️⃣. Instead of computing full Jacobians between two SAE layers, we compute *direct effects* for all active transcoder features simultaneously, using a linear approximation (the "local replacement model").

### From Section 2️⃣ (Transcoders)
- Students know what transcoders are, how they differ from standard SAEs (input-to-output mapping, sitting around the full MLP layer).
- They've computed pullbacks, de-embeddings, and logit lenses for transcoders.
- They've worked with the `run_with_cache_with_transcoder` helper function.
- **Key callback**: Attribution graphs use transcoders as the fundamental unit. Instead of residual-stream SAEs (which reconstruct a snapshot), transcoders decompose the MLP's *computation*, so each node in the graph represents an interpretable computational unit.

---

## Part 1: Implementing Attribution Graphs from Scratch

### Introduction (markdown, ~500 words)

Explain the attribution graph framework at a high level:
- Attribution graphs capture **direct linear effects** between transcoder features and output logits for a specific prompt.
- The key idea is the **local replacement model**: we freeze all non-linear components (attention patterns, LayerNorm scales, MLP activation functions) at their values from a clean forward pass, making the residual stream *linear*. Under this linearisation, the direct effect of feature $s$ on feature $t$ is simply the contraction of $s$'s decoder direction with $t$'s encoder direction through the frozen linear layers.
- This is a scaling-up of the gradient approach from section 1️⃣: instead of computing full Jacobians between two layers, we efficiently compute direct effects for thousands of features at once using batched backward passes.
- Reference the Anthropic "Attribution Graphs" paper: https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Briefly explain **cross-layer transcoders (CLTs)**: In a CLT, each feature can write its decoder vector to multiple downstream layers rather than just the immediately following residual stream position. This means one feature at layer $i$ might write to layers $i+1$, $i+2$, etc. CLTs are the newest transcoder architecture and are used in much of Anthropic's recent circuit tracing work. However, in these exercises we focus on **single-layer (per-layer) transcoders** for simplicity and continuity with section 2️⃣. Interested students can explore CLTs as a bonus exercise at the end.

> From the circuit tracing tutorial notebook:
> "A CLT is a single transcoder for all layers, where features can contribute to multiple downstream layers. CLTs have some practical benefits: they subsume skip connections... and features can be more naturally shared across layers."

### Exercise 3.1: Loading the Replacement Model

**What students do**: Load the Gemma 2-2B model with single-layer GemmaScope transcoders using the `circuit-tracer` library's `ReplacementModel`.

**Why we use `ReplacementModel` as scaffolding**: The `ReplacementModel` class handles all the plumbing of replacing MLP layers with transcoders, configuring gradient flow (freezing attention patterns, LayerNorm scales), and wrapping the TransformerLens model with the right hook points. Students already dealt with a lot of this kind of plumbing in sections 1️⃣ and 2️⃣, so here we let them focus on the higher-level attribution logic.

**Provided code** (no exercise, just run):

```python
import torch
from circuit_tracer import ReplacementModel

model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b",
    "gemma",
    dtype=torch.bfloat16,
    backend="transformerlens",
)
```

**Explanation to include**: Walk through what `ReplacementModel` does under the hood:
1. Loads the base Gemma model via TransformerLens
2. Loads single-layer transcoders from the GemmaScope release
3. Replaces each MLP layer with a `ReplacementMLP` wrapper that adds hook points before/after the MLP
4. Installs permanent hooks that **freeze gradients** through attention patterns (`hook_pattern`), LayerNorm scales (`hook_scale`), and MLP non-linearities (via skip connection detach trick)
5. This creates the "local replacement model" where the residual stream is effectively linear for gradient computation

**Connection to section 2️⃣**: "This is the same idea as the transcoders you loaded in section 2️⃣, except now we have transcoders for *all* layers of the model, and the model has been configured so that gradients flow correctly through the linear components."

**Test**: Verify model produces reasonable logits on the prompt `"The capital of the state containing Dallas is"`.

```python
def test_replacement_model_loaded(model):
    """Verify the model produces sensible output on a test prompt."""
    prompt = "The capital of the state containing Dallas is"
    tokens = model.ensure_tokenized(prompt)
    logits = model(tokens.unsqueeze(0))
    top_token = model.tokenizer.decode(logits[0, -1].argmax())
    assert " Austin" in top_token or " Aust" in top_token, (
        f"Expected model to predict 'Austin' but got '{top_token}'"
    )
    print(f"Model predicts: '{top_token}' - looks correct!")
```

---

### Exercise 3.2: Selecting Salient Logits

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

**Concept**: Before building the attribution graph, we need to decide which output logits to include as target nodes. We don't want all 256k vocabulary tokens as nodes; instead we select the top-probability tokens whose cumulative probability exceeds some threshold (e.g. 0.95).

**What students implement**: `compute_salient_logits(logits, W_U, max_n_logits, desired_logit_prob)` which returns:
- `logit_indices`: token IDs of the selected logits
- `logit_probs`: their softmax probabilities
- `demeaned_vecs`: the corresponding unembedding columns, demeaned (mean subtracted across the vocabulary dimension)

**Why demean?** The mean of the unembedding matrix across vocabulary tokens acts as a constant bias that doesn't carry information about specific tokens. By demeaning, we isolate the direction in residual stream space that specifically promotes each token relative to the average.

**Connection to section 1️⃣**: "This is analogous to the `token_ids` argument in the `latent_to_logit_gradients` exercise from section 1️⃣, where we selected a subset of logits to compute gradients for. Here we're doing the same thing, but using cumulative probability rather than a fixed top-k."

**Solution** (matches `circuit_tracer/utils/salient_logits.py`):

```python
def compute_salient_logits(
    logits: torch.Tensor,       # (d_vocab,)
    W_U: torch.Tensor,          # (d_model, d_vocab)
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the smallest set of top logits whose cumulative probability >= desired_logit_prob.

    Returns:
        logit_indices: (k,) vocabulary ids
        logit_probs:   (k,) softmax probabilities
        demeaned_vecs: (k, d_model) unembedding columns, mean-subtracted
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = W_U[:, top_idx]                          # (d_model, k)
    demean = W_U.mean(dim=-1, keepdim=True)         # (d_model, 1)
    demeaned_vecs = (cols - demean).T               # (k, d_model)

    return top_idx, top_p, demeaned_vecs
    # END SOLUTION
```

**Tests**:

```python
def test_compute_salient_logits(compute_salient_logits):
    """Test salient logit selection."""
    torch.manual_seed(42)
    d_vocab, d_model = 1000, 64
    logits = torch.randn(d_vocab)
    W_U = torch.randn(d_model, d_vocab)

    idx, probs, vecs = compute_salient_logits(logits, W_U, max_n_logits=10, desired_logit_prob=0.95)

    # Basic shapes
    assert idx.ndim == 1 and probs.ndim == 1 and vecs.ndim == 2
    assert len(idx) == len(probs) == vecs.shape[0]
    assert vecs.shape[1] == d_model

    # Cumulative probability check
    assert probs.sum() >= 0.95, f"Cumulative prob {probs.sum():.3f} < 0.95"

    # Check demeaning: mean of selected columns minus global mean should be small
    raw_cols = W_U[:, idx].T  # (k, d_model)
    global_mean = W_U.mean(dim=-1)  # (d_model,)
    expected_demeaned = raw_cols - global_mean.unsqueeze(0)
    torch.testing.assert_close(vecs, expected_demeaned, atol=1e-5, rtol=1e-5)

    # Ordering: probabilities should be in descending order
    assert torch.all(probs[:-1] >= probs[1:]), "Probabilities not in descending order"

    # At most max_n_logits
    assert len(idx) <= 10

    print("All tests passed!")
```

---

### Exercise 3.3: Finding Active Features (Setup Attribution)

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

**Concept**: The first phase of the attribution algorithm is to run a forward pass through the replacement model and record which transcoder features are active (non-zero activation) at each layer and position. This gives us our set of "feature nodes" in the graph.

**What students implement**: `find_active_features(model, input_ids)` which:
1. Calls `model.setup_attribution(input_ids)` to get an `AttributionContext` (provided by the library - explain what it does)
2. Extracts the sparse activation matrix and inspects its structure
3. Returns the context object plus summary information

**Provided**: The `model.setup_attribution()` method (as scaffolding). Students just need to understand what it returns and work with the `AttributionContext`.

**Explanation**: Walk through what `setup_attribution` does:
1. Runs a forward pass, caching MLP inputs/outputs at every layer
2. For each layer, runs the transcoder in sparse mode: encode the MLP input, apply activation function, record which features fire (non-zero activations)
3. Collects the **encoder vectors** (reading directions) and **decoder vectors** (writing directions) for all active features
4. Computes **error vectors**: the difference between the actual MLP output and the transcoder's reconstruction (these become error nodes in the graph)

**Exercise**: Students write a function that takes the context and prints a summary:

```python
def summarize_active_features(ctx, model, input_ids):
    """Print a summary of the active features found during the forward pass."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    activation_matrix = ctx.activation_matrix
    feat_layers, feat_pos, feat_idx = activation_matrix.indices()
    n_layers, n_pos, d_transcoder = activation_matrix.shape

    print(f"Input tokens: {n_pos}")
    print(f"Model layers: {n_layers}")
    print(f"Transcoder dictionary size: {d_transcoder}")
    print(f"Total active features: {activation_matrix._nnz()}")
    print(f"Average active features per (layer, position): {activation_matrix._nnz() / (n_layers * n_pos):.1f}")

    # Count features per layer
    for layer in range(n_layers):
        n_feats = (feat_layers == layer).sum().item()
        print(f"  Layer {layer}: {n_feats} active features")
    # END SOLUTION
```

**Test**:

```python
def test_summarize_active_features(summarize_fn, model):
    """Test that the summary function runs and reports reasonable values."""
    prompt = "The capital of the state containing Dallas is"
    input_ids = model.ensure_tokenized(prompt)
    ctx = model.setup_attribution(input_ids)

    # Should not error
    summarize_fn(ctx, model, input_ids)

    # Check activation matrix dimensions
    n_layers, n_pos, d_tc = ctx.activation_matrix.shape
    assert n_layers == model.cfg.n_layers, f"Expected {model.cfg.n_layers} layers, got {n_layers}"
    assert n_pos == len(input_ids), f"Expected {len(input_ids)} positions, got {n_pos}"
    assert ctx.activation_matrix._nnz() > 0, "No active features found!"

    print("All tests passed!")
```

---

### Exercise 3.4: Computing Direct Effects via Backward Passes

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 30-45 minutes on this exercise.
> ```

**This is the core exercise of Part 1.** Students implement the heart of the attribution algorithm: computing the direct effect of every active feature on every logit node (and vice versa).

**Concept**: Under the local replacement model (where all non-linearities are frozen), the residual stream is linear. This means that the direct effect of feature $s$ on target $t$ is:

$$A_{s \to t} = \text{decoder}_s \cdot \frac{\partial \text{resid}_{t}}{\partial \text{resid}_{s}} \cdot \text{encoder}_t$$

Because the model is linear, we can compute this efficiently using backward passes: inject the encoder/decoder direction as a custom gradient at the target position, and read off the contracted values at all source positions.

**Connection to section 1️⃣**: "In section 1️⃣, you computed the full Jacobian between two SAE layers using `torch.func.jacrev`. That approach is exact but doesn't scale: computing the Jacobian for thousands of features across all layers would be prohibitively expensive. The attribution graph approach trades exactness for scalability: by freezing non-linearities, we can compute direct effects with simple backward passes."

**What students implement**: `compute_attribution_edges(ctx, model, input_ids, logit_vecs, batch_size)`

This function should:
1. Run a forward pass with the replacement model hooks installed (provided by `ctx.install_hooks(model)`)
2. For each logit node: inject the demeaned unembedding vector as a backward pass seed at the final layer/position, and read off the resulting gradients at all feature positions. This gives the logit->feature edge weights.
3. For each active feature: inject the encoder vector as a backward pass seed at the feature's layer/position, and read off the resulting gradients at all upstream feature positions. This gives the feature->feature edge weights.
4. The backward passes should be batched for efficiency: process `batch_size` nodes at once.

**Provided utility** (as scaffolding): `ctx.install_hooks(model)` context manager which installs all the freezing hooks, and `ctx.compute_batch(layers, positions, inject_values, retain_graph)` which handles the actual backward pass mechanics (injecting gradients and collecting results). Students use `ctx.compute_batch` rather than manually doing backward passes.

**What `ctx.compute_batch` does** (explain to students): Takes a batch of (layer, position, inject_value) triples. For each one, it injects `inject_value` as a gradient seed at the given (layer, position) in the residual stream, then collects the resulting gradient at all other feature positions. Returns a matrix of shape `(batch_size, n_source_nodes)` containing the direct effects.

**Implementation structure**:

```python
def compute_attribution_edges(
    ctx,
    model,
    input_ids: torch.Tensor,
    logit_vecs: torch.Tensor,   # (n_logits, d_model) - demeaned unembedding vectors
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute the attribution edge matrix for all logit and feature nodes.

    Returns:
        edge_matrix: (n_logits + n_features, n_source_nodes) dense matrix of direct effects
    """
    activation_matrix = ctx.activation_matrix
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    n_features = activation_matrix._nnz()
    n_logits = logit_vecs.shape[0]

    logit_offset = n_features + (n_layers + 1) * n_pos
    total_source_nodes = logit_offset + n_logits
    edge_matrix = torch.zeros(n_logits + n_features, total_source_nodes)

    # EXERCISE
    # Phase 1: logit attribution - compute edges FROM all features TO each logit node
    # For each batch of logit nodes, inject their demeaned unembedding vector at
    # layer=n_layers, position=n_pos-1 (the final position), and read off gradients
    # ...
    #
    # Phase 2: feature attribution - compute edges FROM upstream features TO each active feature
    # For each batch of active features, inject their encoder vector at the
    # feature's (layer, position), and read off gradients
    # ...
    # END EXERCISE

    # SOLUTION
    # Phase 1: logit attribution
    with ctx.install_hooks(model):
        residual = model.forward(
            input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers
        )
        ctx._resid_activations[-1] = model.ln_final(residual)

    for i in range(0, n_logits, batch_size):
        batch = logit_vecs[i : i + batch_size]
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
        )
        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()

    # Phase 2: feature attribution
    for i in range(0, n_features, batch_size):
        end = min(i + batch_size, n_features)
        idx_batch = torch.arange(i, end)
        rows = ctx.compute_batch(
            layers=feat_layers[idx_batch],
            positions=feat_pos[idx_batch],
            inject_values=ctx.encoder_vecs[idx_batch],
            retain_graph=(end < n_features),
        )
        edge_matrix[n_logits + i : n_logits + end, :logit_offset] = rows.cpu()
    # END SOLUTION

    return edge_matrix
```

**Tests**:

```python
def test_compute_attribution_edges(compute_attribution_edges, model):
    """Test the attribution edge computation."""
    prompt = "The capital of the state containing Dallas is"
    input_ids = model.ensure_tokenized(prompt)
    ctx = model.setup_attribution(input_ids)

    n_layers, n_pos, _ = ctx.activation_matrix.shape
    n_features = ctx.activation_matrix._nnz()

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1], model.unembed.W_U
    )
    n_logits = len(logit_idx)

    edge_matrix = compute_attribution_edges(ctx, model, input_ids, logit_vecs, batch_size=64)

    # Shape check
    logit_offset = n_features + (n_layers + 1) * n_pos
    expected_rows = n_logits + n_features
    expected_cols = logit_offset + n_logits
    assert edge_matrix.shape == (expected_rows, expected_cols), (
        f"Expected shape {(expected_rows, expected_cols)}, got {edge_matrix.shape}"
    )

    # Non-trivial: should have some nonzero edges
    assert edge_matrix.abs().sum() > 0, "Edge matrix is all zeros!"

    # Logit rows (first n_logits rows) should have nonzero entries
    logit_rows = edge_matrix[:n_logits]
    assert logit_rows.abs().sum() > 0, "Logit attribution rows are all zeros!"

    # Feature rows should have nonzero entries
    feat_rows = edge_matrix[n_logits:]
    assert feat_rows.abs().sum() > 0, "Feature attribution rows are all zeros!"

    print(f"Edge matrix shape: {edge_matrix.shape}")
    print(f"Non-zero edges: {(edge_matrix != 0).sum().item()}")
    print(f"Sparsity: {(edge_matrix == 0).float().mean():.2%}")
    print("All tests passed!")
```

---

### Exercise 3.5: Assembling the Graph Object

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

**What students implement**: `assemble_graph(edge_matrix, ctx, model, input_ids, logit_idx, logit_p)` which packages the edge matrix into a `Graph` object with the correct node ordering.

**Concept**: The graph's adjacency matrix has a specific node ordering: `[active_features, error_nodes, embed_nodes, logit_nodes]`. Students need to understand this ordering and construct the full square adjacency matrix from their (potentially rectangular) edge matrix.

**Node ordering** (explain):
- **Active feature nodes** (one per active transcoder feature): these have both incoming and outgoing edges
- **Error nodes** (one per layer per position): represent the transcoder reconstruction error at each (layer, position). They have outgoing edges (they contribute to downstream features) but we don't compute their incoming edges.
- **Embed nodes** (one per token position): represent the token embeddings. They only have outgoing edges.
- **Logit nodes** (one per selected output token): represent the output logits. They only have incoming edges.

```python
def assemble_graph(
    edge_matrix: torch.Tensor,
    ctx,
    model,
    input_ids: torch.Tensor,
    logit_idx: torch.Tensor,
    logit_p: torch.Tensor,
) -> Graph:
    """Package the edge matrix into a Graph object."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    from circuit_tracer.graph import Graph

    activation_matrix = ctx.activation_matrix
    n_features = activation_matrix._nnz()
    n_logits = len(logit_idx)
    total_nodes = edge_matrix.shape[1]

    full_adj = torch.zeros(total_nodes, total_nodes)
    full_adj[:n_features] = edge_matrix[n_logits:]     # feature rows
    full_adj[-n_logits:] = edge_matrix[:n_logits]       # logit rows

    return Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=torch.arange(n_features),
        adjacency_matrix=full_adj,
        cfg=model.cfg,
        scan=model.scan,
    )
    # END SOLUTION
```

**Test**:

```python
def test_assemble_graph(assemble_graph, model):
    """Test graph assembly."""
    prompt = "The capital of the state containing Dallas is"
    input_ids = model.ensure_tokenized(prompt)
    ctx = model.setup_attribution(input_ids)
    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1], model.unembed.W_U
    )
    edge_matrix = compute_attribution_edges(ctx, model, input_ids, logit_vecs)

    graph = assemble_graph(edge_matrix, ctx, model, input_ids, logit_idx, logit_p)

    n_features = ctx.activation_matrix._nnz()
    n_logits = len(logit_idx)
    n_pos = len(input_ids)
    n_layers = model.cfg.n_layers

    expected_total = n_features + (n_layers + 1) * n_pos + n_logits
    assert graph.adjacency_matrix.shape == (expected_total, expected_total), (
        f"Adjacency matrix shape {graph.adjacency_matrix.shape} != ({expected_total}, {expected_total})"
    )
    assert graph.adjacency_matrix.shape[0] == graph.adjacency_matrix.shape[1], "Adjacency matrix not square"
    assert len(graph.logit_tokens) == n_logits
    assert len(graph.input_tokens) == n_pos

    print(f"Graph assembled: {expected_total} nodes ({n_features} features, {n_logits} logits)")
    print("All tests passed!")
```

---

### Exercise 3.6: Computing Node Influence via Power Iteration

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 20-30 minutes on this exercise.
> ```

**Concept**: Given the adjacency matrix $A$, we want to compute the **total influence** of each node on the output logits. The total influence accounts for both direct effects and indirect effects (through intermediate nodes). If $A$ is the normalized adjacency matrix and $w$ is the logit weight vector, then the total influence is:

$$\text{influence} = w \cdot (A + A^2 + A^3 + \ldots) = w \cdot ((I - A)^{-1} - I)$$

But computing $(I - A)^{-1}$ directly is expensive. Instead, we use an iterative approach (similar to power iteration):

$$v_0 = w \cdot A, \quad v_{k+1} = v_k \cdot A, \quad \text{influence} = \sum_k v_k$$

**Key insight about convergence**: Because of the attribution graph's causal structure (features in layer $i$ can only have edges to features in layers $< i$), the matrix $A$ is strictly lower-triangular with respect to layer ordering. This means $A^L = 0$ where $L$ is the number of layers - the matrix is **nilpotent**. So the power iteration is guaranteed to converge in at most $L$ steps. This is a nice computational property: we never need to worry about convergence, and we know the exact maximum number of iterations needed.

**What students implement**:

```python
def normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Row-normalize a matrix by absolute values."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)
    # END SOLUTION


def compute_influence(
    A: torch.Tensor,
    logit_weights: torch.Tensor,
    max_iter: int = 1000,
) -> torch.Tensor:
    """Compute total influence of each node on the output, using iterative matrix-vector products.

    Because of the graph's causal structure, A is nilpotent: A^(n_layers) = 0.
    This means the iteration is guaranteed to converge in at most n_layers steps.

    Args:
        A: Normalized adjacency matrix (n_nodes, n_nodes)
        logit_weights: (n_nodes,) vector with logit probabilities at the logit node positions

    Returns:
        influence: (n_nodes,) total influence of each node on the output
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    current = logit_weights @ A
    influence = current.clone()
    iterations = 0
    while current.any():
        if iterations >= max_iter:
            raise RuntimeError(f"Failed to converge after {iterations} iterations")
        current = current @ A
        influence += current
        iterations += 1
    return influence
    # END SOLUTION
```

**Tests**:

```python
def test_normalize_matrix():
    """Test matrix normalization."""
    M = torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    normed = normalize_matrix(M)

    # Row 0: abs = [1, 2, 3], sum = 6, normalized = [1/6, 2/6, 3/6]
    torch.testing.assert_close(normed[0], torch.tensor([1/6, 2/6, 3/6]))

    # Row 1: all zeros, should stay zero (clamped denominator prevents NaN)
    assert not normed[1].isnan().any()

    # Row 2: abs = [1, 1, 0], sum = 2, normalized = [0.5, 0.5, 0]
    torch.testing.assert_close(normed[2], torch.tensor([0.5, 0.5, 0.0]))

    # All values should be non-negative
    assert (normed >= 0).all()

    print("All tests passed!")


def test_compute_influence():
    """Test influence computation with a simple causal graph."""
    # Simple 3-layer graph: node 0 -> node 1 -> node 2 (logit)
    # A[target, source] convention
    A = torch.zeros(3, 3)
    A[1, 0] = 0.5   # node 0 influences node 1
    A[2, 1] = 0.8   # node 1 influences node 2

    logit_weights = torch.tensor([0.0, 0.0, 1.0])  # only node 2 is a logit

    normed_A = normalize_matrix(A)
    influence = compute_influence(normed_A, logit_weights)

    # Node 1 should have direct influence on node 2
    assert influence[1] > 0, "Node 1 should influence the logit"
    # Node 0 should have indirect influence (through node 1)
    assert influence[0] > 0, "Node 0 should have indirect influence via node 1"

    # The iteration should converge quickly (nilpotent in 3 steps for a 3-node graph)
    print(f"Influence values: {influence}")
    print("All tests passed!")


def test_influence_nilpotent():
    """Test that influence computation terminates within n_layers for a causal graph."""
    import time

    # Create a strictly lower-triangular matrix (causal structure)
    n = 50
    A = torch.tril(torch.rand(n, n), diagonal=-1)
    A = normalize_matrix(A)
    logit_weights = torch.zeros(n)
    logit_weights[-1] = 1.0

    start = time.time()
    influence = compute_influence(A, logit_weights, max_iter=n)
    elapsed = time.time() - start

    print(f"Converged in < {n} iterations, took {elapsed:.4f}s")
    print("All tests passed!")
```

---

### Exercise 3.7: Pruning the Graph

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 25-35 minutes on this exercise.
> ```

**Concept**: The full attribution graph has thousands of nodes and millions of edges. Most are negligible. We prune in two stages:
1. **Node pruning**: Compute node influence (from exercise 3.6), keep only nodes that account for the top `node_threshold` fraction of total influence (e.g. 80%). Always keep embed and logit nodes.
2. **Edge pruning**: For surviving nodes, compute edge influence scores and keep edges accounting for the top `edge_threshold` fraction (e.g. 98%).
3. **Iterative cleanup**: After pruning edges, some nodes may have lost all their incoming or outgoing edges. Remove these nodes and repeat until stable.

**Edge influence**: For each edge $(i, j)$, the edge influence score is: $\text{edge\_score}_{ij} = A'_{ij} \cdot (\text{influence}_i + w_i)$, where $A'_{ij}$ is the normalized edge weight in the pruned matrix, and $(\text{influence}_i + w_i)$ is the total outgoing influence of node $i$ (including its direct logit contribution $w_i$).

**What students implement**: `prune_graph(graph, node_threshold, edge_threshold)` returning `(node_mask, edge_mask)`.

```python
def prune_graph(
    graph: Graph,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune a graph by removing low-influence nodes and edges.

    Args:
        graph: The attribution graph to prune
        node_threshold: Keep nodes contributing to this fraction of total influence
        edge_threshold: Keep edges contributing to this fraction of total influence

    Returns:
        node_mask: Boolean tensor indicating which nodes to keep
        edge_mask: Boolean tensor indicating which edges to keep
    """
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)

    # Build logit weight vector
    logit_weights = torch.zeros(graph.adjacency_matrix.shape[0])
    logit_weights[-n_logits:] = graph.logit_probabilities

    # EXERCISE
    # Step 1: compute node influence and threshold
    # Step 2: always keep embed and logit nodes
    # Step 3: compute edge influence scores and threshold
    # Step 4: iteratively remove nodes that lost all incoming/outgoing edges
    # raise NotImplementedError()
    # END EXERCISE

    # SOLUTION
    # Step 1: node pruning
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_mask = node_influence >= find_threshold(node_influence, node_threshold)
    node_mask[-n_logits - n_tokens:] = True  # always keep embeds and logits

    # Step 2: zero out pruned nodes
    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0

    # Step 3: edge pruning
    edge_scores = compute_edge_influence(pruned_matrix, logit_weights)
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    # Step 4: iterative cleanup
    old_node_mask = node_mask.clone()
    node_mask[:-n_logits - n_tokens] &= edge_mask[:, :-n_logits - n_tokens].any(0)
    node_mask[:n_features] &= edge_mask[:n_features].any(1)

    while not torch.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        node_mask[:-n_logits - n_tokens] &= edge_mask[:, :-n_logits - n_tokens].any(0)
        node_mask[:n_features] &= edge_mask[:n_features].any(1)
    # END SOLUTION

    return node_mask, edge_mask
```

**Helper functions provided** (students should NOT need to implement these, they are given):

```python
def compute_node_influence(adjacency_matrix, logit_weights):
    return compute_influence(normalize_matrix(adjacency_matrix), logit_weights)


def compute_edge_influence(pruned_matrix, logit_weights):
    normalized_pruned = normalize_matrix(pruned_matrix)
    pruned_influence = compute_influence(normalized_pruned, logit_weights)
    pruned_influence += logit_weights
    edge_scores = normalized_pruned * pruned_influence[:, None]
    return edge_scores


def find_threshold(scores, threshold):
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    idx = min(int(torch.searchsorted(cumulative, threshold).item()), len(cumulative) - 1)
    return sorted_scores[idx]
```

**Tests**:

```python
def test_prune_graph(prune_graph, model):
    """Test graph pruning."""
    prompt = "The capital of the state containing Dallas is"
    input_ids = model.ensure_tokenized(prompt)

    # Build graph using the full pipeline from earlier exercises
    ctx = model.setup_attribution(input_ids)
    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1], model.unembed.W_U
    )
    edge_matrix = compute_attribution_edges(ctx, model, input_ids, logit_vecs)
    graph = assemble_graph(edge_matrix, ctx, model, input_ids, logit_idx, logit_p)

    node_mask, edge_mask = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)

    n_total = graph.adjacency_matrix.shape[0]
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)

    # Logit and embed nodes should always be kept
    assert node_mask[-n_logits:].all(), "Logit nodes should always be kept"
    assert node_mask[-n_logits - n_tokens:-n_logits].all(), "Embed nodes should always be kept"

    # Should have fewer nodes after pruning
    n_kept = node_mask.sum().item()
    assert n_kept < n_total, f"Pruning should remove some nodes, but kept {n_kept}/{n_total}"
    assert n_kept > n_logits + n_tokens, "Pruning removed all feature nodes!"

    # Every kept feature node should have at least one incoming and one outgoing edge
    n_features = len(graph.selected_features)
    for i in range(n_features):
        if node_mask[i]:
            assert edge_mask[i].any(), f"Kept feature node {i} has no incoming edges"
            assert edge_mask[:, i].any(), f"Kept feature node {i} has no outgoing edges"

    print(f"Kept {n_kept}/{n_total} nodes ({n_kept/n_total:.1%})")
    print(f"Kept {edge_mask.sum().item()}/{edge_mask.numel()} edges ({edge_mask.float().mean():.4%})")
    print("All tests passed!")
```

---

### Exercise 3.8: Putting It All Together

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

**What students do**: Combine exercises 3.2-3.7 into a single `attribute(prompt, model)` function, then run it on the Dallas prompt and compare against the library's `attribute()` function.

```python
def attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=64):
    """Compute a full attribution graph for a prompt. Combines all previous exercises."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    input_ids = model.ensure_tokenized(prompt)
    ctx = model.setup_attribution(input_ids)

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1], model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )

    edge_matrix = compute_attribution_edges(ctx, model, input_ids, logit_vecs, batch_size)
    graph = assemble_graph(edge_matrix, ctx, model, input_ids, logit_idx, logit_p)
    return graph
    # END SOLUTION
```

**Comparison with library**: Students run both their implementation and the library's:

```python
from circuit_tracer.attribution.attribute_transformerlens import attribute as library_attribute

prompt = "The capital of the state containing Dallas is"

# Student's implementation
student_graph = attribute(prompt, model)

# Library's implementation
library_graph = library_attribute(prompt, model, verbose=True)

# Compare: shapes should match, and edge values should be close
print(f"Student graph: {student_graph.adjacency_matrix.shape}")
print(f"Library graph: {library_graph.adjacency_matrix.shape}")
```

**Note to students**: The library version includes extra sophistication (iterative feature ranking via `compute_partial_influences`, memory offloading, `max_feature_nodes` support), so values won't match exactly. But the logit attribution rows should be very close.

**Test**:

```python
def test_full_attribution(attribute_fn, model):
    """Test the full attribution pipeline."""
    prompt = "The capital of the state containing Dallas is"
    graph = attribute_fn(prompt, model)

    assert hasattr(graph, 'adjacency_matrix')
    assert hasattr(graph, 'logit_tokens')
    assert hasattr(graph, 'active_features')
    assert graph.adjacency_matrix.shape[0] == graph.adjacency_matrix.shape[1]
    assert graph.adjacency_matrix.abs().sum() > 0
    assert len(graph.logit_tokens) > 0

    # The top logit should be Austin-related
    top_logit = model.tokenizer.decode(graph.logit_tokens[0])
    print(f"Top logit token: '{top_logit}'")

    print("All tests passed!")
```

---

## Part 2: Exploring Circuits and Interventions

### Introduction (markdown, ~300 words)

"Now that you've built the attribution algorithm from scratch, let's switch gears and use the `circuit-tracer` library to explore real circuits and perform feature interventions. In Part 1 you built the engine; in Part 2 you get to drive it."

Explain what we'll cover:
1. Loading pre-computed attribution graphs from Neuronpedia
2. Exploring the Dallas/Austin two-hop circuit
3. Performing feature interventions (zero ablation, cross-prompt swapping)
4. Open-ended generation with feature interventions

---

### Exercise 3.9: Loading and Inspecting a Pre-Computed Graph

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

**What students do**: Load a pre-computed graph for the Dallas prompt, either from a `.pt` file or by computing it with the library. Inspect its structure.

**Provided code**:

```python
from circuit_tracer.attribution.attribute_transformerlens import attribute as library_attribute
from circuit_tracer.graph import prune_graph

prompt = "Fact: the capital of the state containing Dallas is"
graph = library_attribute(prompt, model, verbose=True)

# Prune the graph
result = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)
node_mask, edge_mask, cumulative_scores = result
```

**Exercise**: Students write a function to summarize the pruned graph, identifying the most influential feature nodes and their layer/position/feature-index.

```python
def inspect_pruned_graph(graph, node_mask, edge_mask):
    """Print a summary of the pruned attribution graph."""
    # EXERCISE
    # Print: number of kept nodes by type (feature, error, embed, logit)
    # Print: the top-10 most influential feature nodes (by cumulative score)
    # For each, show its layer, position, feature_idx, and activation value
    # raise NotImplementedError()
    # END EXERCISE
```

**Test**: Visual inspection (no automated test, but provide expected output format).

---

### Exercise 3.10: The Dallas/Austin Two-Hop Circuit

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 20-30 minutes on this exercise.
> ```

**Concept**: The prompt "Fact: the capital of the state containing Dallas is" requires two-hop reasoning: Dallas -> Texas -> Austin. The attribution graph reveals distinct supernodes for each concept.

**What students do**:
1. Load the graph and identify the key supernodes: "Texas" features (intermediate layer features that fire on "Dallas" and represent the state), "Say Austin" features (late-layer features that promote "Austin" in the logits), and "Say a capital" features (that promote capital-city-related tokens).
2. Verify the two-hop structure by examining which features have strong edges to which other features.

**Provided**: The Neuronpedia URL for the pre-computed graph (with supernode annotations), plus the `extract_supernode_features` utility function.

```python
from circuit_tracer.utils.demo_utils import extract_supernode_features
from collections import namedtuple

Feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])

dallas_austin_url = "<neuronpedia URL>"  # provided
supernode_features = extract_supernode_features(dallas_austin_url)

# Inspect what supernodes were identified
for name, features in supernode_features.items():
    print(f"{name}: {len(features)} features")
    for f in features[:3]:
        print(f"  Layer {f.layer}, pos {f.pos}, feature {f.feature_idx}")
```

**Exercise**: Students write code to verify the circuit structure by checking edge weights between supernodes:

```python
def verify_two_hop_structure(graph, supernode_features):
    """Check that the Dallas->Texas->Austin two-hop structure exists in the edge weights."""
    # EXERCISE
    # 1. Find features in the "Texas" supernode
    # 2. Find features in the "Say Austin" / "capital cities" supernode
    # 3. Check that there are strong edges from Texas features to Say Austin features
    # 4. Check that there are strong edges from embed nodes (Dallas position) to Texas features
    # raise NotImplementedError()
    # END EXERCISE
```

**Test**: Structural verification (Dallas embed -> Texas features edges are nonzero, Texas features -> Say Austin edges are nonzero).

---

### Exercise 3.11: Feature Interventions - Zero Ablation

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

**Concept**: Now we test the causal claims made by the attribution graph by performing interventions. If the "Texas" supernode really represents "this is about Texas", then ablating it should prevent the model from predicting "Austin".

**What students do**: Use `model.feature_intervention()` to zero-ablate specific supernodes and observe the effect on output logits.

**Provided**: The `display_topk_token_predictions` utility (from the circuit-tracer demos).

```python
from functools import partial
from circuit_tracer.utils.demo_utils import display_topk_token_predictions

display_topk = partial(display_topk_token_predictions, tokenizer=model.tokenizer)

prompt = "Fact: the capital of the state containing Dallas is"
texas_features = supernode_features["Texas"]

# Build intervention tuples: (layer, position, feature_idx, new_value)
interventions = [(*feature, 0.0) for feature in texas_features]

with torch.inference_mode():
    original_logits, _ = model.feature_intervention(prompt, [])
    ablated_logits, _ = model.feature_intervention(prompt, interventions)

display_topk(prompt, original_logits, ablated_logits)
```

**Exercise**: Students experiment with ablating different supernodes (Texas, Say Capital, state, capital) and record the effects. They should fill in a table:

```python
def ablation_experiment(model, prompt, supernode_features, supernodes_to_test):
    """Run zero-ablation experiments on multiple supernodes and report results."""
    # EXERCISE
    # For each supernode in supernodes_to_test:
    #   1. Build intervention tuples (zero ablation)
    #   2. Run model.feature_intervention
    #   3. Record the top-1 prediction and its probability
    #   4. Record how much the "Austin" probability changed
    # Return a summary table
    # raise NotImplementedError()
    # END EXERCISE
```

**Test**: Ablating "Texas" features should reduce the probability of "Austin" significantly.

```python
def test_ablation_reduces_target(model):
    """Test that ablating Texas features reduces Austin probability."""
    prompt = "Fact: the capital of the state containing Dallas is"
    texas_features = supernode_features["Texas"]
    interventions = [(*f, 0.0) for f in texas_features]

    with torch.inference_mode():
        orig_logits, _ = model.feature_intervention(prompt, [])
        abl_logits, _ = model.feature_intervention(prompt, interventions)

    austin_id = model.tokenizer.encode(" Austin")[0]
    orig_prob = orig_logits[0, -1].softmax(-1)[austin_id].item()
    abl_prob = abl_logits[0, -1].softmax(-1)[austin_id].item()

    assert abl_prob < orig_prob, (
        f"Ablation should reduce Austin prob: {orig_prob:.4f} -> {abl_prob:.4f}"
    )
    print(f"Austin probability: {orig_prob:.4f} -> {abl_prob:.4f} (reduced by {orig_prob - abl_prob:.4f})")
    print("Test passed!")
```

---

### Exercise 3.12: Cross-Prompt Feature Swapping

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 25-35 minutes on this exercise.
> ```

**Concept**: The most compelling test of circuit understanding is cross-prompt feature swapping. If "Texas" features truly represent "the state is Texas", then we should be able to swap them with "California" features (from an Oakland prompt) and get the model to predict "Sacramento" instead of "Austin".

**What students do**:
1. Get activations from the Oakland prompt: `"Fact: the capital of the state containing Oakland is"`
2. Extract supernodes from the Oakland graph
3. Build intervention tuples that: (a) zero out Texas features, (b) activate California features at their in-distribution values from the Oakland prompt (scaled up by some factor)
4. Run the intervention and check that the model now predicts Sacramento

```python
def cross_prompt_swap(model, base_prompt, swap_prompt, features_off, features_on, scale=2.0):
    """Swap features between prompts: turn off features_off and turn on features_on.

    Args:
        model: The ReplacementModel
        base_prompt: The prompt to intervene on
        swap_prompt: The prompt to get replacement activations from
        features_off: List of Feature namedtuples to zero-ablate
        features_on: List of Feature namedtuples to activate (from swap_prompt)
        scale: Scaling factor for the replacement activations

    Returns:
        original_logits, modified_logits
    """
    # EXERCISE
    # 1. Get activations from swap_prompt using model.get_activations()
    # 2. Build intervention tuples:
    #    - For features_off: (layer, pos, feat_idx, 0.0)
    #    - For features_on: (layer, pos, feat_idx, scale * activation_value)
    # 3. Run model.feature_intervention on base_prompt with combined interventions
    # raise NotImplementedError()
    # END EXERCISE

    # SOLUTION
    _, swap_activations = model.get_activations(swap_prompt, sparse=True)

    interventions = [(*f, 0.0) for f in features_off]
    interventions += [(*f, scale * swap_activations[f]) for f in features_on]

    with torch.inference_mode():
        original_logits, _ = model.feature_intervention(base_prompt, [])
        modified_logits, _ = model.feature_intervention(base_prompt, interventions)

    return original_logits, modified_logits
    # END SOLUTION
```

**Specific experiments to run**:
1. Dallas -> Oakland: Swap Texas for California -> Should predict Sacramento
2. Dallas -> Shanghai: Swap Texas for China -> Should predict Beijing

**Tests**:

```python
def test_cross_prompt_swap(cross_prompt_swap, model, supernode_features):
    """Test that swapping Texas -> California changes prediction to Sacramento."""
    dallas_prompt = "Fact: the capital of the state containing Dallas is"
    oakland_prompt = "Fact: the capital of the state containing Oakland is"

    texas_features = supernode_features["Texas"]
    # Oakland supernodes (provided or extracted from URL)
    california_features = oakland_supernode_features["California"]

    orig_logits, mod_logits = cross_prompt_swap(
        model, dallas_prompt, oakland_prompt,
        features_off=texas_features, features_on=california_features, scale=2.0
    )

    # Check that Sacramento is now more likely
    sacramento_id = model.tokenizer.encode(" Sacramento")[0]
    austin_id = model.tokenizer.encode(" Austin")[0]

    mod_probs = mod_logits[0, -1].softmax(-1)
    assert mod_probs[sacramento_id] > mod_probs[austin_id], (
        f"Sacramento prob ({mod_probs[sacramento_id]:.4f}) should exceed "
        f"Austin prob ({mod_probs[austin_id]:.4f}) after swapping Texas -> California"
    )
    print(f"Sacramento prob: {mod_probs[sacramento_id]:.4f}, Austin prob: {mod_probs[austin_id]:.4f}")
    print("Test passed!")
```

---

### Exercise 3.13: Open-Ended Generation with Interventions

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

**Concept**: For multi-token generation, interventions need to persist across all generated positions. This is done by using a `slice` for the position argument instead of a fixed integer.

**What students do**: Adapt the cross-prompt swap to use `model.feature_intervention_generate()` with open-ended slices.

```python
def generate_with_intervention(model, prompt, interventions, max_new_tokens=20):
    """Generate text with feature interventions applied to all generated positions.

    Converts fixed-position interventions to open-ended slices for generation.
    """
    # EXERCISE
    # 1. Get the sequence length of the prompt
    # 2. For each intervention tuple (layer, pos, feat_idx, value),
    #    replace pos with slice(original_pos, None, None) for open-ended generation
    # 3. Call model.feature_intervention_generate with the modified interventions
    # raise NotImplementedError()
    # END EXERCISE

    # SOLUTION
    seq_len = len(model.tokenizer(prompt).input_ids)
    open_interventions = []
    for layer, pos, feat_idx, value in interventions:
        open_pos = slice(seq_len - 1, None, None)
        open_interventions.append((layer, open_pos, feat_idx, value))

    pre_text = model.feature_intervention_generate(prompt, [], do_sample=False, verbose=False)[0]
    post_text = model.feature_intervention_generate(
        prompt, open_interventions, do_sample=False, verbose=False
    )[0]
    return pre_text, post_text
    # END SOLUTION
```

**Test**: Run the language-swapping intervention from the tutorial (Spanish -> English on the Michael Jordan basketball prompt).

```python
def test_generation_with_intervention(generate_with_intervention, model):
    """Test that generation changes language with feature intervention."""
    s_spanish = "Hecho: Michael Jordan juega al"
    spanish_features = [Feature(layer=20, pos=-1, feature_idx=341)]

    interventions = [(*f, 0.0) for f in spanish_features]
    pre_text, post_text = generate_with_intervention(model, s_spanish, interventions)

    print(f"Without intervention: {pre_text}")
    print(f"With intervention (ablate Spanish): {post_text}")
    # Visual check: post_text should contain more English words
```

---

### Bonus Exercise: Cross-Layer Transcoders

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵⚪⚪⚪
>
> This is an optional extension for interested students.
> ```

**What students do**: Load a CLT version of the model and compare its attribution graph structure to the single-layer transcoder version. From the tutorial notebook:

> "A CLT is a single transcoder for all layers, where features can contribute to multiple downstream layers. CLTs have some practical benefits: they subsume skip connections... and features can be more naturally shared across layers."

**Provided**: Instructions for loading the CLT model and a brief comparison exercise.

```python
# Load CLT model (optional bonus)
clt_model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b", "gemma_clt", dtype=torch.bfloat16, backend="transformerlens"
)

# Compare attribution graphs
prompt = "Fact: the capital of the state containing Dallas is"
plt_graph = library_attribute(prompt, model, verbose=True)
clt_graph = library_attribute(prompt, clt_model, verbose=True)

# Questions for students:
# 1. How does the number of active features compare between PLT and CLT?
# 2. How do the graph scores (replacement score, completeness score) compare?
# 3. Do the same supernodes appear in both graphs?
```

No tests for this exercise (it's exploratory).

---

## Summary of Exercises

| # | Title | Difficulty | Type | Connection |
|---|-------|-----------|------|------------|
| 3.1 | Loading the Replacement Model | Setup | Run code | Extends transcoders from 2️⃣ to all layers |
| 3.2 | Selecting Salient Logits | 🔴🔴⚪⚪⚪ | Implement | Like top-k logits from 1️⃣ latent-to-logit gradients |
| 3.3 | Finding Active Features | 🔴🔴⚪⚪⚪ | Implement + explore | Understanding transcoder sparse activations |
| 3.4 | Computing Direct Effects | 🔴🔴🔴🔴⚪ | Implement (core) | Scales up gradient approach from 1️⃣ |
| 3.5 | Assembling the Graph | 🔴🔴⚪⚪⚪ | Implement | Node ordering and adjacency matrix structure |
| 3.6 | Power Iteration for Influence | 🔴🔴🔴⚪⚪ | Implement | Nilpotent matrix from causal structure |
| 3.7 | Pruning the Graph | 🔴🔴🔴⚪⚪ | Implement | Node + edge influence thresholding |
| 3.8 | Full Pipeline | 🔴🔴⚪⚪⚪ | Implement + compare | Combine all parts, verify against library |
| 3.9 | Inspecting Pre-Computed Graphs | 🔴⚪⚪⚪⚪ | Explore | Bridge to Part 2 |
| 3.10 | Dallas/Austin Two-Hop Circuit | 🔴🔴⚪⚪⚪ | Explore | Supernode identification |
| 3.11 | Zero Ablation | 🔴🔴⚪⚪⚪ | Implement + explore | Causal testing of circuit claims |
| 3.12 | Cross-Prompt Swapping | 🔴🔴🔴⚪⚪ | Implement + explore | The most compelling intervention |
| 3.13 | Open-Ended Generation | 🔴🔴⚪⚪⚪ | Implement + explore | Extending interventions to generation |
| Bonus | Cross-Layer Transcoders | 🔴🔴🔴🔴⚪ | Explore | CLT comparison |

---

## Test File Summary

All tests should go in the shared test file (`part31_superposition_and_saes/tests.py` or a new section-specific test file). Key test functions:

1. `test_replacement_model_loaded` - Model sanity check
2. `test_compute_salient_logits` - Shape, probability, demeaning checks
3. `test_summarize_active_features` - Dimensions and non-empty features
4. `test_compute_attribution_edges` - Shape, non-trivial edges
5. `test_assemble_graph` - Square adjacency, correct node count
6. `test_normalize_matrix` - Row normalization correctness
7. `test_compute_influence` - Simple causal graph check
8. `test_influence_nilpotent` - Convergence guarantee
9. `test_prune_graph` - Node/edge counts, structural invariants
10. `test_full_attribution` - End-to-end pipeline check
11. `test_ablation_reduces_target` - Causal intervention check
12. `test_cross_prompt_swap` - Sacramento vs Austin after swap
