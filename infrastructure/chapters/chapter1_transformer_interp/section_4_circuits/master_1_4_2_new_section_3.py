# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
# 3️⃣ Attribution Graphs
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
So far in this chapter we've built up two layers of mechanistic understanding:

- **Section 1** showed us how to measure how changes in SAE latents propagate through the residual stream — using `SparseTensor` and `jacrev` to efficiently compute latent-to-latent Jacobians.
- **Section 2** introduced transcoders as a principled way to decompose each MLP layer's function into a sparse set of interpretable features.

**Attribution graphs** bring these ideas together into a complete circuit-level picture. Given a prompt, an attribution graph answers: *which transcoder features, at which layers and positions, were causally responsible for the model's prediction?* The result is a directed acyclic graph whose nodes are features and whose edges are their direct linear influences on each other.

This method was introduced in Anthropic's [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) paper (Lindsey et al., 2025), and in this section we'll implement it from scratch using GemmaScope 2's transcoders with their affine skip connections.

### Key ideas in this section:

1. **The linearized replacement model**: To get clean attribution edges, we need to freeze the model's nonlinearities (attention patterns, LayerNorm scales) at their clean-forward-pass values. This makes the residual stream locally linear, so that a single backward pass cleanly tells us "how much did source node X contribute to target node Y?"

2. **Backward-pass attribution**: For each "target node" (a logit or a feature), we inject its encoder direction as a gradient seed and run a backward pass. Each source node's contribution is its decoder direction dotted with the flowing gradient.

3. **Power-iteration influence**: Once we have the full adjacency matrix, we propagate influence from logit nodes backwards through the graph using a power iteration that terminates in ≤ N_LAYERS steps (because the causal DAG is nilpotent).

4. **Two-stage pruning**: We prune the graph to a sparse, interpretable subgraph by removing low-influence nodes (Stage 1) then low-weight edges (Stage 2).
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## 3.0 Setup & Loading
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

import dataclasses
import enum
import gc
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from typing import Optional

import einops
import numpy as np
import torch as t
import torch.nn as nn
import utils.tests as tests
from huggingface_hub import hf_hub_download
from IPython.display import HTML, display
from jaxtyping import Float, Int
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import dashboard as dashboard_utils
from utils.transcoder_utils import (
    JumpReLUSAE,
    SparseTensor,
    compute_transcoder_fvu,
    html_activations,
    load_all_transcoders,
    load_transcoder_layer,
    show_feature_activations,
    to_str_tokens,
)

MAIN = __name__ == "__main__"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMMA_MODEL_ID = "google/gemma-3-1b-it"
GEMMA_SCOPE_REPO = "google/gemma-scope-2-1b-it"
TRANSCODER_SIZE = "16k"  # change to "262k" for larger models (requires more VRAM)
N_LAYERS = 26  # Gemma 3 1B has 26 transformer layers
D_MODEL = 1152
D_SAE = {"16k": 16384, "262k": 262144}[TRANSCODER_SIZE]

FLAG_RUN_SECTION_3 = True  # set to False to skip memory-intensive model loading

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

# Memory cleanup from earlier sections
if MAIN and FLAG_RUN_SECTION_3:
    for _cleanup_var in ["model", "sae", "gpt2", "replacement_model"]:
        if _cleanup_var in globals():
            del globals()[_cleanup_var]
    gc.collect()
    t.cuda.empty_cache()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Loading Gemma 3 1B IT

We'll work with Google's **Gemma 3 1B IT** (instruction-tuned) model together with
the **GemmaScope 2** transcoders trained on it.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    gemma = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL_ID, device_map="auto", torch_dtype=t.bfloat16)
    gemma.eval()
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)
    print(f"Loaded {GEMMA_MODEL_ID}: {sum(p.numel() for p in gemma.parameters()) / 1e9:.1f}B parameters")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Loading GemmaScope 2 Transcoders

GemmaScope 2 ships **per-layer transcoders** trained on every transformer layer of Gemma 3.
Each transcoder takes the MLP's input (post-LayerNorm residual stream) and outputs a sparse
reconstruction of the MLP's output:

```
transcoder_output = decode(encode(mlp_in)) + mlp_in @ W_affine_skip
```

The **affine skip connection** (`W_affine_skip`) is the key new ingredient in GemmaScope 2.
It learns the *linear* component of the MLP transformation. The transcoder's sparse features
then capture the *nonlinear residual*. We'll see shortly why this split is crucial for attribution.

We load `width=16k` (16,384 features per layer) transcoders from `transcoders_all/`.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    transcoders = load_all_transcoders(n_layers=N_LAYERS, size=TRANSCODER_SIZE)
    print(f"Loaded {N_LAYERS} transcoders × {D_SAE:,} features each")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Reconstruction quality demo

As in section 2, we verify the transcoders reconstruct the MLP output faithfully.
The FVU should be around 10–15% for `16k` width.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    _demo_layer = 13
    _demo_prompt = "The capital of the state containing Dallas is"
    _demo_ids = tokenizer(_demo_prompt, return_tensors="pt").input_ids.to(gemma.device)
    _mlp_in_demo, _mlp_out_demo = {}, {}

    def _cache_pre(mod, inp, out):
        _mlp_in_demo[0] = (out[0] if isinstance(out, tuple) else out).detach()[0]

    def _cache_post(mod, inp, out):
        _mlp_out_demo[0] = (out[0] if isinstance(out, tuple) else out).detach()[0]

    _h1 = gemma.model.layers[_demo_layer].pre_feedforward_layernorm.register_forward_hook(_cache_pre)
    _h2 = gemma.model.layers[_demo_layer].mlp.register_forward_hook(_cache_post)
    with t.no_grad():
        gemma(_demo_ids)
    _h1.remove()
    _h2.remove()

    fvu = compute_transcoder_fvu(transcoders[_demo_layer], _mlp_in_demo[0].float(), _mlp_out_demo[0].float())
    print(f"Transcoder layer {_demo_layer} FVU: {fvu:.2%}")

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    # Show feature activations (visual: token sequence colored by feature activation)
    show_feature_activations(
        transcoders[_demo_layer], _mlp_in_demo[0].float(), feature_idx=42, tokenizer=tokenizer, input_ids=_demo_ids[0]
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## 3.1 The Linearized Replacement Model

Attribution graphs require us to answer a precise question: *"Given that feature Y fired at
layer i, how much of that was directly caused by feature X at an earlier layer j?"*

The challenge is that this "direct effect" question is ambiguous in a nonlinear model.
If we just run a backward pass through the raw model, the gradient through feature X would
include all the second-order effects of how X's firing changes the attention patterns, the
LayerNorm scales, and so on. These confound the simple "linear contribution" we want.

The solution is the **local linearization** trick. We run a clean forward pass to record all
the model's nonlinear state (attention patterns, LayerNorm scales, transcoder activations),
then **freeze** that state. With everything frozen, the residual stream becomes locally linear:
each feature node's contribution is just its decoder direction multiplied by a constant factor,
and that factor is exactly what we want to attribute.

GemmaScope 2's affine skip connection makes this especially clean. The MLP's contribution is:

$$\text{MLP output} = \underbrace{\text{decode}(\text{encode}(x))}_{\text{nonlinear features}} + \underbrace{x \cdot W_{\text{skip}}}_{\text{linear component}}$$

In the linearized model, we keep only the **skip connection** as gradient-active and detach
the feature reconstruction. This means attribution flows through the skip weights —
a perfectly linear path.

The following demo makes this concrete.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    tc = transcoders[_demo_layer]
    mlp_in = _mlp_in_demo[0].float()
    mlp_out = _mlp_out_demo[0].float()

    with t.no_grad():
        skip = mlp_in @ tc.affine_skip_connection
        feat_recon = tc.decode(tc.encode(mlp_in))
        error = mlp_out - skip - feat_recon

    print(f"MLP output decomposition (layer {_demo_layer}):")
    print(f"  Skip connection norm:        {skip.norm():.2f}")
    print(f"  Feature reconstruction norm: {feat_recon.norm():.2f}")
    print(f"  Error term norm:             {error.norm():.2f}")
    print(f"  Residual check (should be ~0): {(skip + feat_recon + error - mlp_out).norm():.6f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Implementing Linearization with Hooks

We use two hook classes:

1. **`FreezeHooks`** (given below): Detaches attention outputs and freezes RMSNorm scale factors.
   The residual stream additions remain gradient-active; only the nonlinear routing is frozen.

2. **`TranscoderReplacementHooks`** (your exercise below): Replaces each MLP's output with
   `skip + detach(mlp_out - skip)`. Only the skip connection is gradient-active.

When both are active, the ONLY gradient path is:

$$\text{resid}[i+1] \approx \text{resid}[i] + \underbrace{\text{frozen\_attn}}_{\text{detached}} + \underbrace{\text{resid}[i] \cdot W_{\text{skip}}^{(i)}}_{\text{gradient-active}} + \underbrace{\text{detached\_recon}}_{\text{detached}}$$

This is locally linear in `resid[i]`.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


class FreezeHooks:
    """Registers hooks that make Gemma 3's residual stream locally linear for attribution.

    Freezes two types of nonlinear operations:

    1. **Attention outputs**: The full attention sub-module output is detached,
       making attention a fixed additive contribution. Gradients still flow through
       the residual connection because the input to each layer is kept differentiable.

    2. **RMSNorm scale factors**: The normalization factor (1/RMS(x)) is frozen at its
       clean-forward value. LayerNorm then acts as a fixed diagonal scaling of x,
       so d(LayerNorm(x))/dx is a constant.

    Note: A fully correct implementation would freeze attention *patterns* (softmax outputs)
    while keeping V@W_O differentiable, requiring custom attention logic. We approximate
    by detaching the full attention output — sufficient for the skip-connection gradient path.

    Usage (combine with TranscoderReplacementHooks):
        with FreezeHooks(model), TranscoderReplacementHooks(model, transcoders):
            output = model(input_ids)
    """

    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self._handles: list = []

    def __enter__(self):
        self._install()
        return self

    def __exit__(self, *args):
        self._remove()

    def _install(self):
        for layer in self.model.model.layers:
            # Freeze attention output (detach entirely)
            def make_attn_hook():
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        return (out[0].detach(),) + out[1:]
                    return out.detach()

                return hook

            self._handles.append(layer.self_attn.register_forward_hook(make_attn_hook()))

            # Freeze RMSNorm: apply frozen scale factor to x
            for norm_module in [layer.input_layernorm, layer.pre_feedforward_layernorm]:

                def make_norm_hook():
                    def hook(module, inp, out):
                        x = inp[0]
                        with t.no_grad():
                            eps = getattr(module, "eps", getattr(module, "variance_epsilon", 1e-6))
                            x_f32 = x.detach().float()
                            # Gemma RMSNorm: output = x * rsqrt(mean(x^2) + eps) * (1 + weight)
                            frozen_rms_inv = t.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
                            frozen_scale = (frozen_rms_inv * (1.0 + module.weight.detach().float())).to(x.dtype)
                        return x * frozen_scale.detach()

                    return hook

                self._handles.append(norm_module.register_forward_hook(make_norm_hook()))

        # Final LayerNorm
        if hasattr(self.model.model, "norm"):

            def final_norm_hook(module, inp, out):
                x = inp[0]
                with t.no_grad():
                    eps = getattr(module, "eps", getattr(module, "variance_epsilon", 1e-6))
                    x_f32 = x.detach().float()
                    frozen_rms_inv = t.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
                    frozen_scale = (frozen_rms_inv * (1.0 + module.weight.detach().float())).to(x.dtype)
                return x * frozen_scale.detach()

            self._handles.append(self.model.model.norm.register_forward_hook(final_norm_hook))

    def _remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise 3.1 — `TranscoderReplacementHooks._install` ⬛⬛⬛

Implement `_install` to replace each MLP's output with `skip + detach(mlp_out - skip)`.

You need **two hooks per layer**:

1. A **pre-MLP hook** on `model.model.layers[i].pre_feedforward_layernorm` that caches its
   output in `self._cache[i]`. This is `mlp_in` — the post-LN2 residual stream value.
   Important: call `.detach()` when caching (we just need the value, not its gradient).

2. A **post-MLP hook** on `model.model.layers[i].mlp` that computes:
   ```python
   skip = self._cache[i].float() @ self.transcoders[i].affine_skip_connection.float()
   skip = skip.to(original_output.dtype)
   return skip + (original_output - skip).detach()
   ```

Remember that module outputs may be tuples `(hidden_state, ...)` — handle both cases.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


class TranscoderReplacementHooks:
    """Context manager that makes each MLP layer contribute only its linear skip connection.

    For each layer i, replaces the MLP output with:
        skip = mlp_in @ transcoders[i].affine_skip_connection   (gradient-active)
        + (mlp_out - skip).detach()                             (frozen nonlinear residual)

    Forward-pass VALUE is unchanged. Only the GRADIENT PATH changes.

    Usage (always combine with FreezeHooks):
        with FreezeHooks(model), TranscoderReplacementHooks(model, transcoders):
            output = model(input_ids)
    """

    def __init__(self, model: AutoModelForCausalLM, transcoders: list[JumpReLUSAE]):
        self.model = model
        self.transcoders = transcoders
        self._cache: dict[int, Tensor] = {}
        self._handles: list = []

    def __enter__(self):
        self._install()
        return self

    def __exit__(self, *args):
        self._remove()
        self._cache.clear()

    def _install(self):
        """
        For each layer i, register:
          1. Forward hook on model.model.layers[i].pre_feedforward_layernorm:
               cache output (detached) as self._cache[i]
          2. Forward hook on model.model.layers[i].mlp:
               replace output with skip + (output - skip).detach()
               where skip = self._cache[i].float() @ transcoders[i].affine_skip_connection.float()
        """
        # SOLUTION
        for i, layer in enumerate(self.model.model.layers):
            tc = self.transcoders[i]

            def make_pre_hook(idx):
                def hook(module, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    self._cache[idx] = o.detach()
                    return out

                return hook

            def make_post_hook(idx, transcoder):
                def hook(module, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    mlp_in = self._cache[idx]
                    skip = mlp_in.float() @ transcoder.affine_skip_connection.float()
                    skip = skip.to(o.dtype)
                    replacement = skip + (o - skip).detach()
                    if isinstance(out, tuple):
                        return (replacement,) + out[1:]
                    return replacement

                return hook

            self._handles.append(layer.pre_feedforward_layernorm.register_forward_hook(make_pre_hook(i)))
            self._handles.append(layer.mlp.register_forward_hook(make_post_hook(i, tc)))
        # END SOLUTION
        # EXERCISE
        # raise NotImplementedError("Implement TranscoderReplacementHooks._install()")
        # END EXERCISE

    def _remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    tests.test_transcoder_replacement_hooks(TranscoderReplacementHooks, gemma, transcoders, tokenizer)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## 3.2 The Attribution Context

Before computing attribution, we run a **clean no-grad forward pass** to collect:
- Active transcoder features (sparse activation matrix)
- MLP inputs/outputs per layer (for computing error terms and skip connections)
- Attention outputs (frozen during the attribution pass)
- Final logits

This is packaged into an `AttributionContext` dataclass — the attribution-graph equivalent
of `model.run_with_cache_with_saes()` from section 1.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


@dataclasses.dataclass
class AttributionContext:
    """All data from the clean forward pass needed to compute attribution edges.

    Fields:
        activation_sparse: SparseTensor shape (n_layers, seq_len, d_sae).
            Nonzero entries are active transcoder features.
        mlp_in_cache:   list[Tensor] — (seq_len, d_model) per layer, float32
        mlp_out_cache:  list[Tensor] — (seq_len, d_model) per layer, float32
        error_vectors:  list[Tensor] — mlp_out - transcoder.forward(mlp_in), BOS zeroed
        attn_out_cache: list[Tensor] — (seq_len, d_model) per layer (for reference)
        logits:    (1, seq_len, vocab_size)
        input_ids: (seq_len,)
        residual_cache: populated during the attribution forward pass (not the clean pass)
    """

    activation_sparse: SparseTensor
    mlp_in_cache: list
    mlp_out_cache: list
    error_vectors: list
    attn_out_cache: list
    logits: Tensor
    input_ids: Tensor
    residual_cache: list = dataclasses.field(default_factory=list)

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[0]

    @property
    def n_layers(self) -> int:
        return len(self.mlp_in_cache)

    @property
    def device(self):
        return self.input_ids.device


def setup_attribution(
    model: AutoModelForCausalLM,
    transcoders: list[JumpReLUSAE],
    input_ids: Tensor,
) -> AttributionContext:
    """Run a clean no-grad forward pass and collect all attribution prerequisites.

    Registers hooks to cache MLP inputs/outputs and attention outputs at each layer,
    then computes transcoder activations and error vectors.
    """
    if input_ids.dim() == 1:
        input_ids_b = input_ids.unsqueeze(0)
        input_ids_1d = input_ids
    else:
        input_ids_b = input_ids
        input_ids_1d = input_ids.squeeze(0)

    seq_len = input_ids_1d.shape[0]
    device = input_ids_1d.device
    n_layers_model = len(transcoders)

    mlp_in_cache, mlp_out_cache, attn_out_cache = [], [], []

    handles = []
    for i, layer in enumerate(model.model.layers):

        def make_pre(idx):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                mlp_in_cache.append(o.detach()[0].float().cpu())

            return h

        def make_post(idx):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                mlp_out_cache.append(o.detach()[0].float().cpu())

            return h

        def make_attn(idx):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                attn_out_cache.append(o.detach()[0].float().cpu())

            return h

        handles.append(layer.pre_feedforward_layernorm.register_forward_hook(make_pre(i)))
        handles.append(layer.mlp.register_forward_hook(make_post(i)))
        handles.append(layer.self_attn.register_forward_hook(make_attn(i)))

    with t.no_grad():
        logits = model(input_ids_b).logits.detach().cpu()

    for h in handles:
        h.remove()

    # Compute transcoder features and error vectors
    d_sae = transcoders[0].w_enc.shape[1]
    act_dense = t.zeros(n_layers_model, seq_len, d_sae)
    error_vectors = []

    with t.no_grad():
        for i, tc in enumerate(transcoders):
            mi = mlp_in_cache[i].to(device)
            mo = mlp_out_cache[i].to(device)
            acts_i = tc.encode(mi)
            act_dense[i] = acts_i.cpu()
            recon_i = tc.forward(mi)
            err_i = (mo - recon_i).cpu()
            err_i[0] = 0.0  # zero BOS
            error_vectors.append(err_i)

    return AttributionContext(
        activation_sparse=SparseTensor.from_dense(act_dense),
        mlp_in_cache=mlp_in_cache,
        mlp_out_cache=mlp_out_cache,
        error_vectors=error_vectors,
        attn_out_cache=attn_out_cache,
        logits=logits,
        input_ids=input_ids_1d.cpu(),
    )


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## 3.3 Node Data Model

The four node types in an attribution graph and their adjacency matrix ordering:

```
[feature_nodes | error_nodes | embedding_nodes | logit_nodes]
```

This ordering makes the adjacency matrix block-lower-triangular by layer depth,
which guarantees that the influence power iteration terminates.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


class NodeType(str, enum.Enum):
    FEATURE = "feature"  # Fired transcoder feature (layer, pos, feat_idx)
    ERROR = "error"  # Transcoder reconstruction error at (layer, pos)
    EMBEDDING = "embedding"  # Input token embedding at pos
    LOGIT = "logit"  # Output token logit (target node)


@dataclasses.dataclass
class Node:
    """A single node in the attribution graph.

    matrix_idx is the node's row/column index in the adjacency matrix.
    """

    node_type: NodeType
    layer: int | str  # int for FEATURE/ERROR; "E" for EMBEDDING; "L" for LOGIT
    pos: int
    feature_idx: int  # 0 for non-FEATURE nodes
    matrix_idx: int
    str_token: Optional[str] = None
    token_prob: float = 0.0  # for LOGIT nodes
    activation: float = 0.0  # for FEATURE nodes

    @property
    def node_id(self) -> str:
        return f"{self.node_type.value}_{self.layer}_{self.pos}_{self.feature_idx}"

    @property
    def label(self) -> str:
        if self.node_type == NodeType.LOGIT:
            return f'Output "{self.str_token}" (p={self.token_prob:.3f})'
        elif self.node_type == NodeType.EMBEDDING:
            return f'Embed "{self.str_token}" (pos={self.pos})'
        elif self.node_type == NodeType.ERROR:
            return f"Error L{self.layer} pos={self.pos}"
        else:
            return f"Feature L{self.layer}:{self.feature_idx} pos={self.pos}"


def build_node_list(
    ctx: AttributionContext,
    logit_tokens: Tensor,
    logit_probs: Tensor,
    tokenizer,
) -> list[Node]:
    """Build the ordered node list with matrix_idx assigned sequentially.

    Order: [features | errors | embeddings | logits]
    Within features: sorted by (layer asc, pos asc, feature_idx asc).
    """
    nodes = []
    idx = 0
    act = ctx.activation_sparse

    # 1. Feature nodes
    if act._nnz() > 0:
        nz_idx = act.nonzero_indices  # (nnz, 3): [layer, pos, feat_idx]
        nz_val = act.nonzero_values  # (nnz,)
        sort_key = nz_idx[:, 0] * 10_000_000 + nz_idx[:, 1] * 100_000 + nz_idx[:, 2]
        order = sort_key.argsort()
        nz_idx, nz_val = nz_idx[order], nz_val[order]
        for (layer, pos, feat_idx), act_val in zip(nz_idx.tolist(), nz_val.tolist()):
            nodes.append(Node(NodeType.FEATURE, layer, pos, feat_idx, idx, activation=act_val))
            idx += 1

    # 2. Error nodes
    for layer in range(ctx.n_layers):
        for pos in range(ctx.seq_len):
            nodes.append(Node(NodeType.ERROR, layer, pos, 0, idx))
            idx += 1

    # 3. Embedding nodes
    for pos in range(ctx.seq_len):
        tok_str = tokenizer.decode([ctx.input_ids[pos].item()])
        nodes.append(Node(NodeType.EMBEDDING, "E", pos, 0, idx, str_token=tok_str))
        idx += 1

    # 4. Logit nodes
    for tok_id, prob in zip(logit_tokens.tolist(), logit_probs.tolist()):
        tok_str = tokenizer.decode([tok_id])
        nodes.append(Node(NodeType.LOGIT, "L", ctx.seq_len - 1, tok_id, idx, str_token=tok_str, token_prob=prob))
        idx += 1

    return nodes


@dataclasses.dataclass
class AttributionGraph:
    """The full attribution graph for a single prompt.

    adjacency_matrix[i, j] = direct linear effect of source node j on target node i.
    Node ordering: [features | errors | embeddings | logits]
    """

    prompt: str
    nodes: list
    adjacency_matrix: Tensor  # (n_nodes, n_nodes)
    normalized_matrix: Tensor  # row-normalized by |values|
    node_influence: Tensor  # (n_nodes,) from compute_influence()
    input_ids: Tensor
    logit_tokens: Tensor
    logit_probs: Tensor

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_features(self) -> int:
        return sum(1 for n in self.nodes if n.node_type == NodeType.FEATURE)

    @property
    def n_errors(self) -> int:
        return sum(1 for n in self.nodes if n.node_type == NodeType.ERROR)

    @property
    def n_embeds(self) -> int:
        return sum(1 for n in self.nodes if n.node_type == NodeType.EMBEDDING)

    @property
    def n_logits(self) -> int:
        return sum(1 for n in self.nodes if n.node_type == NodeType.LOGIT)


def assemble_attribution_graph(
    ctx: AttributionContext,
    nodes: list,
    adjacency_matrix: Tensor,
    logit_tokens: Tensor,
    logit_probs: Tensor,
    prompt: str,
    tokenizer,
) -> "AttributionGraph":
    """Package all outputs into an AttributionGraph (calls normalize_matrix and compute_influence)."""
    norm_matrix = normalize_matrix(adjacency_matrix)

    n_nodes = len(nodes)
    logit_weights = t.zeros(n_nodes)
    for node in nodes:
        if node.node_type == NodeType.LOGIT:
            logit_weights[node.matrix_idx] = node.token_prob

    node_inf = compute_influence(norm_matrix, logit_weights)

    return AttributionGraph(
        prompt=prompt,
        nodes=nodes,
        adjacency_matrix=adjacency_matrix,
        normalized_matrix=norm_matrix,
        node_influence=node_inf,
        input_ids=ctx.input_ids,
        logit_tokens=logit_tokens,
        logit_probs=logit_probs,
    )


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## 3.4 Computing Attribution Edges

### The math

For each **target node** $t$, we want to know how much each **source node** $s$ directly
contributed to it. This "direct linear contribution" is:

$$A[t, s] = \mathbf{e}_t^{\text{read}} \cdot \frac{\partial \mathbf{r}_t}{\partial \mathbf{r}_s} \cdot \mathbf{d}_s^{\text{write}}$$

where:
- $\mathbf{e}_t^{\text{read}}$ is how node $t$ reads from the residual stream (encoder direction)
- $\mathbf{d}_s^{\text{write}}$ is how node $s$ writes to the residual stream (decoder direction)
- $\partial \mathbf{r}_t / \partial \mathbf{r}_s$ is the Jacobian in the **linearized** model

Rather than computing the full Jacobian (too expensive), we compute **one row at a time**:
inject $\mathbf{e}_t^{\text{read}}$ as a gradient seed at position $(t.\text{layer}, t.\text{pos})$
and run backward. The gradient at each source position, dotted with $\mathbf{d}_s^{\text{write}}$,
gives $A[t, s]$.

This is the same "one row of the Jacobian per backward pass" idea as section 1, but adapted to
the much larger computational graph of the full model.

### Read/write directions

| Node type | Read direction                           | Write direction                          |
|-----------|------------------------------------------|------------------------------------------|
| Feature   | `w_enc[:, feat_idx]` of layer transcoder | `w_dec[feat_idx, :]` of layer transcoder |
| Error     | (source only)                            | `error_vectors[layer][pos]`             |
| Embedding | (source only)                            | `W_E[input_ids[pos]]`                   |
| Logit     | demeaned `W_U[:, token_id]`              | (target only)                            |
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise 3.3.1 — `compute_salient_logits` ⬛⬛

Select the minimal set of top-probability output tokens whose cumulative probability
reaches `desired_logit_prob`. The demeaned unembedding vectors will serve as the
"read directions" for logit nodes in the attribution pass.

Why demean? Subtracting `W_U.mean(dim=1)` removes the direction that all tokens share
equally — the result is token-specific signal only.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def compute_salient_logits(
    logits: Float[Tensor, " vocab"],
    W_U: Float[Tensor, "d_model vocab"],
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select minimal top-probability tokens with cumulative prob >= desired_logit_prob.

    Returns:
        logit_tokens:  (k,) token ids
        logit_probs:   (k,) softmax probabilities
        demeaned_vecs: (k, d_model) — W_U columns with W_U.mean(dim=1) subtracted

    Steps:
        1. Compute softmax probs; take top max_n_logits.
        2. Cumsum; searchsorted to find cutoff k.
        3. Slice first k tokens.
        4. Demean W_U columns and transpose to (k, d_model).
    """
    # SOLUTION
    probs = t.softmax(logits.float(), dim=-1)
    top_probs, top_idx = t.topk(probs, max_n_logits)
    cumsum = t.cumsum(top_probs, dim=0)
    k = int(t.searchsorted(cumsum, desired_logit_prob).item()) + 1
    k = min(k, max_n_logits)
    logit_tokens = top_idx[:k]
    logit_probs_out = top_probs[:k]
    vecs = W_U[:, logit_tokens] - W_U.mean(dim=1, keepdim=True)  # (d_model, k)
    demeaned_vecs = vecs.T.float()  # (k, d_model)
    return logit_tokens, logit_probs_out, demeaned_vecs
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    tests.test_compute_salient_logits(compute_salient_logits, gemma, tokenizer)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise 3.3.2 — `compute_attribution_matrix` ⬛⬛⬛⬛

This is the core of the section. Implement the full attribution edge computation.

**Structure:**

1. **Attribution forward pass**: Run the model inside `FreezeHooks + TranscoderReplacementHooks`
   with `torch.enable_grad()`. Register forward hooks on each `model.model.layers[i]` to
   capture the output (residual stream after layer i) with `.retain_grad()`.

2. **Phase 1 — Logit attribution** (rows for logit nodes):
   For each logit node with `inject_vec = logit_vecs[i]`, inject that vector as a gradient
   at `residual_cache[last_layer][0, last_pos]` using a `.register_hook()`. Run
   `.backward(gradient=zeros, retain_graph=True)`. Then collect source scores.

3. **Phase 2 — Feature attribution** (rows for feature nodes):
   Same pattern, but `inject_vec = transcoders[layer].w_enc[:, feat_idx]` and
   injection is at `residual_cache[feature_layer][0, feature_pos]`.

**Collecting source scores**: After each backward pass, for source node $s$:
```python
score_s = residual_cache[s.layer].grad[0, s.pos] · decoder_vec_s
```
where `decoder_vec_s` is the write direction for node $s$.

**Gradient injection hook** (one pattern for both phases):
```python
def inject_hook(grad):
    g = torch.zeros_like(grad)
    g[0, target_pos] = inject_vec.to(grad.dtype)
    return g
handle = residual_cache[target_layer].register_hook(inject_hook)
```
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def compute_attribution_matrix(
    model: AutoModelForCausalLM,
    transcoders: list[JumpReLUSAE],
    ctx: AttributionContext,
    nodes: list,
    logit_vecs: Float[Tensor, "n_logits d_model"],
) -> Float[Tensor, "n_nodes n_nodes"]:
    """Compute the (n_nodes × n_nodes) attribution adjacency matrix via backward passes.

    A[i, j] = direct linear effect of source node j on target node i.

    Phase 1: Logit rows — inject demeaned unembedding at final layer, last pos.
    Phase 2: Feature rows — inject encoder direction at (layer, pos) of each feature.

    Error/embedding rows are zero (source-only nodes).
    """
    # SOLUTION
    n_nodes = len(nodes)
    device = next(model.parameters()).device
    seq_len = ctx.seq_len
    adj = t.zeros(n_nodes, n_nodes)

    # Precompute W_E for embedding node write directions
    W_E = model.model.embed_tokens.weight.detach().cpu().float()

    # Categorize nodes
    feat_nodes = [(n, n.matrix_idx) for n in nodes if n.node_type == NodeType.FEATURE]
    error_nodes = [(n, n.matrix_idx) for n in nodes if n.node_type == NodeType.ERROR]
    embed_nodes = [(n, n.matrix_idx) for n in nodes if n.node_type == NodeType.EMBEDDING]
    logit_nodes_list = [(n, n.matrix_idx) for n in nodes if n.node_type == NodeType.LOGIT]

    # --- Attribution forward pass ---
    ctx.residual_cache = []
    res_handles = []

    def make_res_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h.retain_grad()
            while len(ctx.residual_cache) <= layer_idx:
                ctx.residual_cache.append(None)
            ctx.residual_cache[layer_idx] = h

        return hook

    for i, layer in enumerate(model.model.layers):
        res_handles.append(layer.register_forward_hook(make_res_hook(i)))

    with FreezeHooks(model), TranscoderReplacementHooks(model, transcoders):
        with t.enable_grad():
            _logits = model(ctx.input_ids.unsqueeze(0).to(device))

    for h in res_handles:
        h.remove()

    n_res = len(ctx.residual_cache)
    last_layer = n_res - 1

    def zero_grads():
        for res in ctx.residual_cache:
            if res is not None and res.grad is not None:
                res.grad.zero_()

    def collect_scores(target_layer_idx: int) -> Tensor:
        """After backward, dot gradients with each source node's write direction."""
        row = t.zeros(n_nodes)
        for li in range(min(target_layer_idx + 1, n_res)):
            res = ctx.residual_cache[li]
            if res is None or res.grad is None:
                continue
            g = res.grad.detach()[0].cpu().float()  # (seq, d_model)

            for node, nidx in feat_nodes:
                if int(node.layer) == li:
                    dec = transcoders[li].w_dec[node.feature_idx].detach().cpu().float()
                    row[nidx] = (g[node.pos] * dec).sum()

            for node, nidx in error_nodes:
                if node.layer == li:
                    ev = ctx.error_vectors[li][node.pos].float()
                    row[nidx] = (g[node.pos] * ev).sum()

            if li == 0:
                for node, nidx in embed_nodes:
                    eid = ctx.input_ids[node.pos].item()
                    row[nidx] = (g[node.pos] * W_E[eid]).sum()
        return row

    def run_one_backward(target_layer: int, target_pos: int, inject_vec: Tensor, retain: bool) -> Tensor:
        zero_grads()
        res_t = ctx.residual_cache[target_layer]
        if res_t is None:
            return t.zeros(n_nodes)
        iv = inject_vec.to(device)

        def inj(grad):
            g = t.zeros_like(grad)
            g[0, target_pos] = iv.to(grad.dtype)
            return g

        h = res_t.register_hook(inj)
        try:
            res_t.backward(gradient=t.zeros_like(res_t), retain_graph=retain)
        finally:
            h.remove()
        return collect_scores(target_layer)

    # Phase 1: Logit attribution
    n_logit = len(logit_nodes_list)
    for i, (node, nidx) in enumerate(logit_nodes_list):
        iv = logit_vecs[i].to(device)
        is_last = (i == n_logit - 1) and (len(feat_nodes) == 0)
        adj[nidx] = run_one_backward(last_layer, seq_len - 1, iv, retain=not is_last)

    # Phase 2: Feature attribution
    n_feat = len(feat_nodes)
    for i, (node, nidx) in enumerate(feat_nodes):
        li = int(node.layer)
        if li >= n_res:
            continue
        enc = transcoders[li].w_enc[:, node.feature_idx].detach().cpu().float()
        is_last = i == n_feat - 1
        adj[nidx] = run_one_backward(li, node.pos, enc.to(device), retain=not is_last)

    return adj
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    dallas_prompt = "The capital of the state containing Dallas is"
    dallas_ids = tokenizer(dallas_prompt, return_tensors="pt").input_ids.squeeze(0)
    ctx = setup_attribution(gemma, transcoders, dallas_ids)

    logit_tokens, logit_probs, logit_vecs = compute_salient_logits(ctx.logits[0, -1], gemma.model.embed_tokens.weight.T)
    print("Top predicted tokens:", [tokenizer.decode([i]) for i in logit_tokens.tolist()])

    nodes = build_node_list(ctx, logit_tokens, logit_probs, tokenizer)
    print(
        f"Nodes: {sum(1 for n in nodes if n.node_type == NodeType.FEATURE)} features, "
        f"{sum(1 for n in nodes if n.node_type == NodeType.ERROR)} errors, "
        f"{sum(1 for n in nodes if n.node_type == NodeType.EMBEDDING)} embeds, "
        f"{sum(1 for n in nodes if n.node_type == NodeType.LOGIT)} logits"
    )

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    tests.test_compute_attribution_matrix(compute_attribution_matrix, gemma, transcoders, ctx, nodes, logit_vecs)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## 3.5 Influence & Pruning

With the raw adjacency matrix in hand, we now need to:
1. **Normalize** it so influence scores are comparable across nodes
2. **Compute influence** — how much does each node transitively affect the output?
3. **Prune** — remove low-influence nodes and low-weight edges

### Exercise 3.4.1 — `normalize_matrix` ⬛

Row-normalize the adjacency matrix by absolute values so each row sums to at most 1.
This makes influence scores comparable: a score of 1.0 means "complete credit for this row."
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def normalize_matrix(matrix: Float[Tensor, "n n"]) -> Float[Tensor, "n n"]:
    """Row-normalize a matrix by absolute values.

    normalized[i] = matrix[i] / sum(|matrix[i]|).clamp(min=1e-10)

    Zero rows stay zero. Signs are preserved.
    """
    # SOLUTION
    abs_sums = matrix.abs().sum(dim=1, keepdim=True).clamp(min=1e-10)
    return matrix / abs_sums
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    tests.test_normalize_matrix(normalize_matrix)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise 3.4.2 — `compute_influence` ⬛⬛⬛

Compute total indirect influence of each node on the output logits via power iteration.

The key idea: influence propagates backwards. Starting from the logit nodes (whose weights
are their softmax probabilities), we repeatedly multiply by the normalized adjacency matrix
to propagate influence to earlier nodes.

$$\text{influence} = \mathbf{w} \cdot A + \mathbf{w} \cdot A^2 + \mathbf{w} \cdot A^3 + \cdots$$

**Why does this terminate?** The causal structure means that features in layer $i$ can only
attribute to features in layers $j < i$ (earlier layers). So the normalized adjacency
matrix $A$ is strictly block-lower-triangular by layer index — making it **nilpotent**:
$A^{N+1} = 0$ where $N$ = number of layers. The loop terminates in at most $N$ iterations,
*guaranteed*, with no convergence check needed.

Implement the iterative sum:
```python
current = logit_weights @ A
influence = current.clone()
while current.any():
    current = current @ A
    influence += current
```
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def compute_influence(
    A: Float[Tensor, "n n"],
    logit_weights: Float[Tensor, " n"],
    max_iter: int = 1000,
) -> Float[Tensor, " n"]:
    """Compute total indirect influence of each node via power iteration.

    influence = sum_{k=1}^{infinity} logit_weights @ A^k

    Terminates when current = 0 (guaranteed by nilpotency of A, in <= n_layers+1 steps).
    Raises RuntimeError if max_iter exceeded.
    """
    # SOLUTION
    current = logit_weights @ A
    influence = current.clone()
    iters = 0
    while current.any():
        if iters >= max_iter:
            raise RuntimeError(f"compute_influence: did not converge after {iters} iterations")
        current = current @ A
        influence = influence + current
        iters += 1
    return influence
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    tests.test_compute_influence(compute_influence, normalize_matrix)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Visualizing node influence

Before pruning, let's see which nodes have the highest influence on the Dallas prompt:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    # Compute adjacency matrix (may take ~1 minute for full prompt)
    adj = compute_attribution_matrix(gemma, transcoders, ctx, nodes, logit_vecs)

    # Build logit_weights
    n_nodes = len(nodes)
    logit_weights = t.zeros(n_nodes)
    for node in nodes:
        if node.node_type == NodeType.LOGIT:
            logit_weights[node.matrix_idx] = node.token_prob

    # Compute influence
    A_norm = normalize_matrix(adj)
    influence = compute_influence(A_norm, logit_weights)

    # Print top 15 nodes by influence
    top_k = t.topk(influence, min(15, n_nodes))
    print("Top nodes by influence:")
    for score, nidx in zip(top_k.values.tolist(), top_k.indices.tolist()):
        node = nodes[nidx]
        print(f"  [{score:.3f}] {node.label}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise 3.4.3 — `prune_graph` ⬛⬛⬛

Now implement the two-stage pruning pipeline.

**Stage 1 — Node pruning by indirect influence:**
1. Build `logit_weights` (logit node probs, zeros elsewhere)
2. `node_influence = compute_influence(normalize_matrix(adj), logit_weights)`
3. `node_mask = node_influence >= find_threshold(node_influence, node_threshold)`
4. Always keep embedding and logit nodes

**Stage 2 — Edge pruning by contribution:**
1. Zero out pruned-node rows/cols in adjacency matrix
2. Edge score: `|norm_pruned[i,j]| * (node_influence[i] + logit_weights[i])`
3. `edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)`

**Iterative cleanup** (repeat until stable):
- Zero out rows/cols of pruned nodes in edge_mask
- Remove feature/error nodes with no outgoing edges
- Remove feature nodes with no incoming edges

`find_threshold` is provided as a utility below.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def find_threshold(scores: Tensor, fraction: float) -> float:
    """Find the score cutoff that keeps the top-fraction of total score mass.

    Sort scores descending, find the index where cumulative fraction >= fraction.
    Returns the score value at that index.
    """
    sorted_scores, _ = t.sort(scores.flatten().abs(), descending=True)
    total = sorted_scores.sum()
    if total == 0:
        return 0.0
    cumsum = t.cumsum(sorted_scores, dim=0) / total
    idx = int(t.searchsorted(cumsum, fraction).item())
    idx = min(idx, len(sorted_scores) - 1)
    return sorted_scores[idx].item()


def prune_graph(
    graph: "AttributionGraph",
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> tuple[Tensor, Tensor]:
    """Prune the attribution graph to a sparse, interpretable subgraph.

    Two-stage algorithm:
      Stage 1: Remove nodes below indirect-influence threshold (keep embeds + logits always)
      Stage 2: Remove edges below direct-contribution threshold
      Cleanup: Iteratively remove nodes that lost all in/out edges

    Returns:
        node_mask: (n_nodes,) bool — True = keep this node
        edge_mask: (n_nodes, n_nodes) bool — True = keep this edge
    """
    # SOLUTION
    adj = graph.adjacency_matrix
    nodes = graph.nodes
    n_nodes = len(nodes)

    # Build logit_weights
    logit_weights = t.zeros(n_nodes)
    for node in nodes:
        if node.node_type == NodeType.LOGIT:
            logit_weights[node.matrix_idx] = node.token_prob

    # --- Stage 1: Node pruning by indirect influence ---
    node_influence = compute_influence(normalize_matrix(adj), logit_weights)
    threshold_n = find_threshold(node_influence, node_threshold)
    node_mask = node_influence >= threshold_n

    # Always keep embed and logit nodes
    for node in nodes:
        if node.node_type in (NodeType.EMBEDDING, NodeType.LOGIT):
            node_mask[node.matrix_idx] = True

    # --- Stage 2: Edge pruning by contribution ---
    pruned_adj = adj.clone()
    pruned_adj[~node_mask] = 0.0
    pruned_adj[:, ~node_mask] = 0.0

    norm_pruned = normalize_matrix(pruned_adj)
    edge_scores = norm_pruned.abs() * (node_influence + logit_weights).unsqueeze(1)
    threshold_e = find_threshold(edge_scores, edge_threshold)
    edge_mask = edge_scores >= threshold_e

    # --- Iterative cleanup ---
    def cleanup_pass():
        changed = False
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        for node in nodes:
            if node.node_type in (NodeType.FEATURE, NodeType.ERROR):
                idx = node.matrix_idx
                if node_mask[idx]:
                    no_out = not edge_mask[idx].any()
                    no_in = not edge_mask[:, idx].any() and node.node_type == NodeType.FEATURE
                    if no_out or no_in:
                        node_mask[idx] = False
                        changed = True
        return changed

    while cleanup_pass():
        pass

    return node_mask, edge_mask
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    # Build the full graph first
    dallas_graph = assemble_attribution_graph(ctx, nodes, adj, logit_tokens, logit_probs, dallas_prompt, tokenizer)
    tests.test_prune_graph(prune_graph, dallas_graph)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## 3.6 Full Pipeline & Visualization

### Exercise 3.5.1 — `attribute` ⬛⬛

Compose all the pieces into a single `attribute` function.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def attribute(
    prompt: str,
    model: AutoModelForCausalLM,
    transcoders: list[JumpReLUSAE],
    tokenizer,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> "AttributionGraph":
    """Full attribution pipeline: tokenize → setup → salient_logits → nodes → matrix → graph.

    Returns an un-pruned AttributionGraph (call prune_graph separately).

    Steps:
        1. tokenizer(prompt) → input_ids (1D)
        2. setup_attribution(model, transcoders, input_ids) → ctx
        3. compute_salient_logits(ctx.logits[0,-1], W_U, ...) → logit_tokens, logit_probs, logit_vecs
        4. build_node_list(ctx, logit_tokens, logit_probs, tokenizer) → nodes  [given]
        5. compute_attribution_matrix(model, transcoders, ctx, nodes, logit_vecs, ...) → adj
        6. assemble_attribution_graph(ctx, nodes, adj, ...) → graph  [given]
    """
    # SOLUTION
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
    input_ids = input_ids.to(next(model.parameters()).device)

    ctx = setup_attribution(model, transcoders, input_ids)

    W_U = model.model.embed_tokens.weight.T  # (d_model, vocab)
    logit_tokens, logit_probs, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1], W_U, max_n_logits=max_n_logits, desired_logit_prob=desired_logit_prob
    )

    nodes = build_node_list(ctx, logit_tokens, logit_probs, tokenizer)
    adj = compute_attribution_matrix(model, transcoders, ctx, nodes, logit_vecs)
    graph = assemble_attribution_graph(ctx, nodes, adj, logit_tokens, logit_probs, prompt, tokenizer)

    return graph
    # END SOLUTION
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    tests.test_attribute(attribute, gemma, transcoders, tokenizer)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Dashboard Visualization

The `create_dashboard` utility (in `utils/dashboard.py`) converts a pruned `AttributionGraph`
into an interactive HTML visualization. It's given code — you don't need to implement it.

Run the cells below to generate and serve the Dallas attribution graph:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    dallas_graph = attribute(dallas_prompt, gemma, transcoders, tokenizer)
    node_mask, edge_mask = prune_graph(dallas_graph)

    n_feat_kept = sum(1 for n in dallas_graph.nodes if n.node_type == NodeType.FEATURE and node_mask[n.matrix_idx])
    n_edges_kept = edge_mask.sum().item()
    print(f"After pruning: {n_feat_kept} features, {n_edges_kept:.0f} edges")

    dashboard_utils.create_dashboard(
        graph=dallas_graph,
        node_mask=node_mask,
        edge_mask=edge_mask,
        output_path="./dallas_attribution_graph",
        title="Dallas → Austin Attribution Graph",
        serve=True,
        port=8080,
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## LinearizedModel: Section 4 Interface

Section 4's causal intervention exercises use `model.feature_intervention()`,
`model.get_activations()`, and `model.feature_intervention_generate()`.

Rather than relying on `circuit-tracer`'s `ReplacementModel` (which targets Gemma 2 + GemmaScope 1),
we define a thin `LinearizedModel` wrapper around our `gemma` + `transcoders` that provides
the same interface. The two section-4 exercises involving these methods are in section 4.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


class LinearizedModel:
    """Wraps Gemma 3 + GemmaScope 2 transcoders with a circuit-intervention interface.

    Provides:
        get_activations(prompt) -> dict[(layer, pos, feat_idx) -> value]
        feature_intervention(prompt, interventions) -> (logits, None)
        feature_intervention_generate(prompt, interventions, ...) -> (text, logits, None)

    These methods are implemented as exercises in section 4.
    """

    def __init__(self, model: AutoModelForCausalLM, transcoders: list[JumpReLUSAE], tokenizer):
        self.model = model
        self.transcoders = transcoders
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str) -> Tensor:
        return self.tokenizer(prompt, return_tensors="pt").input_ids.to(next(self.model.parameters()).device)

    @property
    def W_U(self) -> Tensor:
        return self.model.model.embed_tokens.weight.T  # (d_model, vocab)

    def get_activations(self, prompt: str) -> dict:
        """Run setup_attribution and return {(layer, pos, feat_idx): activation_value}."""
        # Implemented as exercise in section 4
        raise NotImplementedError("Implement get_activations in section 4")

    def feature_intervention(self, prompt: str, interventions: list) -> tuple:
        """Run forward pass overriding specified feature values.

        Each intervention: (layer, pos_or_slice, feat_idx, new_value).
        Returns (logits, None).
        """
        # Implemented as exercise in section 4
        raise NotImplementedError("Implement feature_intervention in section 4")

    def feature_intervention_generate(self, prompt: str, interventions: list, **kwargs) -> tuple:
        """Generate text with feature value overrides (applied at every new token position).

        Returns (generated_text, logits, None).
        """
        # Implemented as exercise in section 4
        raise NotImplementedError("Implement feature_intervention_generate in section 4")


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

if MAIN and FLAG_RUN_SECTION_3:
    linearized_model = LinearizedModel(gemma, transcoders, tokenizer)
    print("LinearizedModel created. Section 4 will add get_activations() and feature_intervention().")
