# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
```python
[
    {"title": "Latent Gradients", "icon": "1-circle-fill", "subtitle": "(30%)"},
    {"title": "Transcoders", "icon": "2-circle-fill", "subtitle": "(20%)"},
    {"title": "Attribution Graphs", "icon": "3-circle-fill", "subtitle": "(30%)"},
    {"title": "Exploring Circuits & Interventions", "icon": "4-circle-fill", "subtitle": "(20%)"},
]
```
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
# [1.4.2] SAE Circuits
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/header-13-2.png" width="350">
<br>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
# Introduction
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
In these exercises, we explore **circuits with SAEs**: sets of SAE latents in different layers of a transformer which communicate with each other, and explain some particular model behaviour in an end-to-end way. We'll start by computing latent-to-latent, token-to-latent and latent-to-logit gradients, which give us a linear proxy for how latents in different layers are connected. We'll then move on to **transcoders**, a variant of SAEs which learn to reconstruct a model layer's computation rather than just its activations, and which offer significant advantages for circuit analysis.

We expect some degree of prerequisite knowledge in these exercises. Specifically, it will be very helpful if you understand:

- What **superposition** is, and what the **sparse autoencoder** architecture is (if you need a refresher on these topics, see exercises **1.5.4 Toy Models of SAEs & Superposition**)
- How to use the **SAELens** library to load and run SAEs alongside TransformerLens models (covered in the first section of exercises **1.3.3 Interpretability with SAEs**)

We've included a short section at the start to speedrun the most relevant background from exercise set 1.3.3 (mainly loading models & SAEs and running forward passes with them), so you don't need to have completed that exercise set in full before starting this one.

One note on terminology: we'll be mostly adopting the convention that **features** are characteristics of the underlying data distribution that our base models are trained on, and **SAE latents** (or just "latents") are the directions in the SAE. This is to avoid the overloading of the term "feature", and avoiding the implicit assumption that "SAE features" correspond to real features in the data. We'll relax this terminology when we're looking at SAE latents which very clearly correspond to specific interpretable features in the data.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Reading Material

Most of this is optional, and can be read at your leisure depending on what interests you most & what level of background you have. If we could recommend just one, it would be "Towards Monosemanticity" - particularly the first half of "Problem Setup", and the sections where they take a deep dive on individual latents.

- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) outlines the core ideas behind superposition - what it is, why it matters for interepretability, and what we might be able to do about it.
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html) arguably took the first major stride in mechanistic interpretability with SAEs: training them on a 1-layer model, and extracting a large number of interpretable features.
- [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) shows how you can scale up the science of SAEs to larger models, specifically the SOTA (at the time) model Claude 3 Sonnet. It provides an interesting insight into where the field might be moving in the near future.
- [Improving Dictionary Learning with Gated Sparse Autoencoders](https://arxiv.org/pdf/2404.16014) is a paper from DeepMind which introduces the Gated SAE architecture, demonstrating how it outperforms the standard architecture and also motivating its use by speculating about underlying feature distributions.
- [Gemma Scope](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/) announces DeepMind's release of a comprehensive suite of open-sourced SAE models (trained with JumpReLU architecture). We'll be working a lot more with Gemma Scope models in subsequent exercises!
- [LessWrong, SAEs tag](https://www.lesswrong.com/tag/sparse-autoencoders-saes) contains a collection of posts on LessWrong that discuss SAEs, and is a great source of inspiration for further independent research!
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Content & Learning Objectives

### 1️⃣ Latent Gradients

SAEs are cool and interesting and we can steer on their latents to produce cool and interesting effects, but does this mean that we've truly unlocked the true units of computation used by our models, or have we just found an interesting clustering algorithm? The answer is that we don't really know yet! One strong piece of evidence for the former would be finding **circuits with SAEs**, in other words sets of latents in different layers of the transformer which communicate with each other, and explain some particular behaviour in an end-to-end way. In this section, we'll compute gradients between latents in different layers to build up a picture of how they communicate.

> ##### Learning Objectives
>
> - Learn how to compute **latent-to-latent gradients** between SAE latents in different layers of the transformer
> - Compute **token-to-latent gradients** to understand which input tokens drive particular latent activations
> - Compute **latent-to-logit gradients** to understand how latents affect the model's output
> - Use these gradient-based methods to find circuits in attention SAEs (e.g. induction circuits)

### 2️⃣ Transcoders

**Transcoders** are a variant of SAEs which learn to reconstruct a model layer's computation (e.g. a sparse mapping from MLP input to MLP output), rather than just reconstructing activations at a single point. They offer significant advantages for circuit analysis, since they decompose the function of an MLP layer into sparse, interpretable units. In this section, we'll load and work with transcoders, study their properties using techniques like de-embeddings, and go through a blind case study where we reverse-engineer a transcoder latent purely from weights-based analysis.

> ##### Learning Objectives
>
> - Understand transcoders, and how they differ from standard SAEs
> - Learn techniques for interpreting transcoder latents: **pullbacks**, **de-embeddings**, and **extended embeddings**
> - Work through a **blind case study**, interpreting a transcoder latent using only circuit-level analysis (no activation examples)

### 3️⃣ Attribution Graphs

Attribution graphs extend the gradient-based methods from section 1️⃣ into a full framework for understanding end-to-end computation in transformers via transcoder features. In this section, you'll implement the core circuit tracing algorithm: building a **local replacement model** that linearises the residual stream, computing direct effects between features via batched backward passes, and pruning the resulting graph using influence-based power iteration.

> ##### Learning Objectives
>
> - Understand the **local replacement model** and why freezing non-linearities makes the residual stream linear
> - Implement the core **attribution algorithm**: salient logit selection, feature discovery, and direct effect computation
> - Implement **graph pruning** via node and edge influence thresholding, using power iteration on the adjacency matrix
> - Understand why the adjacency matrix is **nilpotent** due to the graph's causal structure

### 4️⃣ Exploring Circuits & Interventions

Now that you've built the attribution graph algorithm from scratch, you'll use the `circuit-tracer` library to explore real circuits and perform feature interventions. You'll study the Dallas/Austin two-hop factual recall circuit, test its causal structure via zero ablation, swap features between prompts, and generate text with feature interventions.

> ##### Learning Objectives
>
> - Load and inspect pre-computed **attribution graphs** and their supernodes
> - Perform **zero ablation** experiments to test causal claims made by the graph
> - Perform **cross-prompt feature swapping** to demonstrate compositional circuit structure
> - Use **open-ended generation** with feature interventions
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## A note on memory usage

In these exercises, we'll be loading some pretty large models into memory (e.g. Gemma 2-2B and its SAEs, as well as a host of other models in later sections of the material). It's useful to have functions which can help profile memory usage for you, so that if you encounter OOM errors you can try and clear out unnecessary models. For example, we've found that with the right memory handling (i.e. deleting models and objects when you're not using them any more) it should be possible to run all the exercises in this material on a Colab Pro notebook, and all the exercises minus the handful involving Gemma on a free Colab notebook.

<details>
<summary>See this dropdown for some functions which you might find helpful, and how to use them.</summary>

First, we can run some code to inspect our current memory usage. Here's an example of running this code on a Colab Pro notebook.

```python
import part42_sae_circuits.utils as utils

# Profile memory usage, and delete gemma models if we've loaded them in
namespace = globals().copy() | locals()
utils.profile_pytorch_memory(namespace=namespace, filter_device="cuda:0")
```

<pre style="font-family: Consolas; font-size: 14px">Allocated = 35.88 GB
Total = 39.56 GB
Free = 3.68 GB
┌──────────────────────┬────────────────────────┬──────────┬─────────────┐
│ Name                 │ Object                 │ Device   │   Size (GB) │
├──────────────────────┼────────────────────────┼──────────┼─────────────┤
│ gemma_2_2b           │ HookedSAETransformer   │ cuda:0   │       11.94 │
│ gpt2                 │ HookedSAETransformer   │ cuda:0   │        0.61 │
│ gemma_2_2b_sae       │ SAE                    │ cuda:0   │        0.28 │
│ sae_resid_dirs       │ Tensor (4, 24576, 768) │ cuda:0   │        0.28 │
│ gpt2_sae             │ SAE                    │ cuda:0   │        0.14 │
│ logits               │ Tensor (4, 15, 50257)  │ cuda:0   │        0.01 │
│ logits_with_ablation │ Tensor (4, 15, 50257)  │ cuda:0   │        0.01 │
│ clean_logits         │ Tensor (4, 15, 50257)  │ cuda:0   │        0.01 │
│ _                    │ Tensor (16, 128, 768)  │ cuda:0   │        0.01 │
│ clean_sae_acts_post  │ Tensor (4, 15, 24576)  │ cuda:0   │        0.01 │
└──────────────────────┴────────────────────────┴──────────┴─────────────┘</pre>

From this, we see that we've allocated a lot of memory for the the Gemma model, so let's delete it. We'll also run some code to move any remaining objects on the GPU which are larger than 100MB to the CPU, and print the memory status again.

```python
del gemma_2_2b
del gemma_2_2b_sae

THRESHOLD = 0.1  # GB
for obj in gc.get_objects():
    try:
        if isinstance(obj, t.nn.Module) and utils.get_tensors_size(obj) / 1024**3 > THRESHOLD:
            if hasattr(obj, "cuda"):
                obj.cpu()
            if hasattr(obj, "reset"):
                obj.reset()
    except:
        pass

# Move our gpt2 model & SAEs back to GPU (we'll need them for the exercises we're about to do)
gpt2.to(device)
gpt2_saes = {layer: sae.to(device) for layer, sae in gpt2_saes.items()}

utils.print_memory_status()
```

<pre style="font-family: Consolas; font-size: 14px">Allocated = 14.90 GB
Reserved = 39.56 GB
Free = 24.66</pre>

Mission success! We've managed to free up a lot of memory. Note that the code which moves all objects collected by the garbage collector to the CPU is often necessary to free up the memory. We can't just delete the objects directly because PyTorch can still sometimes keep references to them (i.e. their tensors) in memory. In fact, if you add code to the for loop above to print out `obj.shape` when `obj` is a tensor, you'll see that a lot of those tensors are actually Gemma model weights, even once you've deleted `gemma_2_2b`.

</details>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Setup (don't read, just run)
"""

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

import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter1_transformer_interp"
repo = "arena-pragmatic-interp"
branch = "main"

# # Install dependencies
# try:
#     import transformer_lens
# except:
#     %pip install "openai==1.56.1" einops datasets jaxtyping "sae-lens>=4.0.0,<5.0.0" openai tabulate umap-learn hdbscan eindex-callum git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python git+https://github.com/callummcdougall/sae_vis.git@callum/v3 transformer_lens==2.11.0
#
# # Install circuit-tracer (for sections 3 and 4)
# import subprocess
# subprocess.run(["git", "clone", "https://github.com/safety-research/circuit-tracer.git", "/tmp/circuit-tracer"])
# sys.path.append("/tmp/circuit-tracer")

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

# if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
#     if not IN_COLAB:
#         !sudo apt-get install unzip
#         %pip install jupyter ipython --upgrade

#     if not os.path.exists(f"{root}/{chapter}"):
#         !wget -P {root} https://github.com/callummcdougall/arena-pragmatic-interp/archive/refs/heads/{branch}.zip
#         !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
#         !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
#         !rm {root}/{branch}.zip
#         !rmdir {root}/{repo}-{branch}

if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")

FLAG_RUN_SECTION_1 = False
FLAG_RUN_SECTION_2 = False
FLAG_RUN_SECTION_3 = True

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

import gc
import os
import sys
from collections import Counter, namedtuple
from functools import partial
from pathlib import Path

import einops
import numpy as np
import plotly.express as px
import torch as t
from huggingface_hub import hf_hub_download
from IPython.display import IFrame, display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tabulate import tabulate
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt, to_numpy

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part42_sae_circuits"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
# FILTERS: ~colab
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
# END FILTERS

import part42_sae_circuits.tests as tests
import part42_sae_circuits.utils as utils

MAIN = __name__ == "__main__"

# ! CELL TYPE: code
# ! FILTERS: [colab]
# ! TAGS: []

# For displaying sae-vis inline
if IN_COLAB:
    import http.server
    import socketserver
    import threading

    from google.colab import output

    PORT = 8000

    def display_vis_inline(filename: Path, height: int = 850):
        """
        Displays the HTML files in Colab. Uses global `PORT` variable defined in prev cell, so that each
        vis has a unique port without having to define a port within the function.
        """
        global PORT

        def serve(directory):
            os.chdir(directory)
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("", PORT), handler) as httpd:
                print(f"Serving files from {directory} on port {PORT}")
                httpd.serve_forever()

        thread = threading.Thread(target=serve, args=("/content",))
        thread.start()

        filename = str(filename).split("/content")[-1]

        output.serve_kernel_port_as_iframe(PORT, path=filename, height=height, cache_in_notebook=True)

        PORT += 1


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Speedrunning some relevant background

This section covers the essential SAELens concepts you'll need for the rest of these exercises. If you've already completed exercise set **1.3.3 Interpretability with SAEs** then you can skip this section and go straight to section 1️⃣.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Loading SAEs with `SAE.from_pretrained`

[SAELens](https://github.com/jbloomAus/SAELens) is a library designed to help researchers train and analyse sparse autoencoders. You can think of it as the equivalent of TransformerLens for sparse autoencoders (and it also integrates very well with TransformerLens models, which we'll see shortly).

To load an SAE, use `SAE.from_pretrained`. This function has return type `tuple[SAE, dict, Tensor | None]`, with the return elements being the SAE, config dict, and a tensor of feature sparsities. The config dict contains useful metadata on e.g. how the SAE was trained. In practice, you'll often just grab the first element:

```python
gpt2_sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    device=str(device),
)[0]
```

You can view the available SAE releases in SAELens with `get_pretrained_saes_directory()`. Each release contains multiple SAEs (e.g. trained on different layers of the same base model).

Base models are loaded using the `HookedSAETransformer` class, which is adapted from the TransformerLens `HookedTransformer` class:

```python
gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
```
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Running SAEs and caching activations

You can add SAEs to a TransformerLens model when doing forward passes in much the same way you add hook functions. There are several different methods for this:

- **`model.run_with_saes(tokens, saes=[list_of_saes])`** works like `model.run_with_hooks`, doing a single forward pass with SAEs attached (then resetting them).
- **`logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])`** works like `model.run_with_cache`, caching all intermediate activations including SAE activations.
- **`with model.saes(saes=[sae]):`** is a context manager which temporarily attaches SAEs.
- **`model.add_sae(sae)` / `model.reset_saes()`** manually adds and removes SAEs.

To access SAE activations from the cache, the hook names are the concatenation of the HookedTransformer `hook_name` and the SAE hook name, joined by a period. The most important ones are:

```python
# Post-activation latent values (shape [batch, seq, d_sae]) - this is the one you'll use most
cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"]

# Pre-activation latent values (before the activation function)
cache[f"{sae.cfg.hook_name}.hook_sae_acts_pre"]

# The SAE's reconstruction of the original activations
cache[f"{sae.cfg.hook_name}.hook_sae_recons"]

# The final SAE output (either reconstruction, or reconstruction + error term)
cache[f"{sae.cfg.hook_name}.hook_sae_output"]
```

Here's a full example, extracting the top activating latents at the final token of a prompt:

```python
_, cache = gpt2.run_with_cache_with_saes(
    prompt,
    saes=[gpt2_sae],
    stop_at_layer=gpt2_sae.cfg.hook_layer + 1,  # no need to compute past the SAE layer
)
sae_acts_post = cache[f"{gpt2_sae.cfg.hook_name}.hook_sae_acts_post"][0, -1, :]
```
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### The `use_error_term` parameter

The parameter `sae.use_error_term` determines whether we actually substitute the model's activations with SAE reconstructions during the forward pass:

- **`use_error_term=False`** (default): the SAE's output replaces the transformer's activations. This means downstream computations use the SAE reconstruction rather than the original activations.
- **`use_error_term=True`**: the SAE computes all its internal states (latent activations etc.) in the same way, but the transformer's activations are left intact. This is useful when you want to cache SAE activations without actually intervening on the model.

```python
# Cache SAE activations WITHOUT intervening on the model
gpt2_sae.use_error_term = True
logits, cache = gpt2.run_with_cache_with_saes(prompt, saes=[gpt2_sae])
# logits are identical to running the base model without SAEs, but we still get SAE activations in the cache

# Cache SAE activations AND replace model activations with SAE reconstructions
gpt2_sae.use_error_term = False
logits_recon, cache_recon = gpt2.run_with_cache_with_saes(prompt, saes=[gpt2_sae])
# logits_recon will differ from the base model's logits due to SAE reconstruction error
```

This parameter matters a lot for the gradient exercises in this set: we'll typically use `use_error_term=True` to get the "true" latent activations from a clean forward pass, then `use_error_term=False` when computing Jacobians (because we want our Jacobian function to actually pass through the SAE decoder and encoder).
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Using `ActivationsStore`

The `ActivationsStore` class is a convenient alternative to loading a bunch of data yourself. It streams in data from a given dataset; in the case of the `from_sae` classmethod that dataset will be given by your SAE's config (which is also the same as the SAE's original training dataset):

```python
gpt2_act_store = ActivationsStore.from_sae(
    model=gpt2,
    sae=gpt2_sae,
    streaming=True,
    store_batch_size_prompts=16,
    n_batches_in_buffer=32,
    device=str(device),
)

# Get a batch of tokens
tokens = gpt2_act_store.get_batch_tokens()
assert tokens.shape == (gpt2_act_store.store_batch_size_prompts, gpt2_act_store.context_size)
```
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Neuronpedia & `display_dashboard`

[Neuronpedia](https://neuronpedia.org) is an open platform for interpretability research. It hosts **SAE dashboards** which help you quickly understand what a particular SAE latent represents, including components like max activating examples, top logits, activation density plots, and LLM-generated explanations.

We can display these dashboards inline using the following helper function, which we'll use in several places throughout these exercises:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def display_dashboard(
    sae_release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    latent_idx=0,
    width=800,
    height=600,
) -> None:
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
# 1️⃣ Latent Gradients
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Introduction

Previous SAE work (in material 1.3.3) has focused on understanding individual latents. Here, we address a highly important topic - what about **circuits of SAE latents**? Circuit analysis has already been somewhat successful in language model interpretability (e.g. see Anthropic's work on induction circuits, or the Indirect Object Identification paper), but many attempts to push circuit analysis further has hit speedbumps: most connections in the model are not sparse, and it's very hard to disentangle all the messy cross-talk between different components and residual stream subspaces. Circuit offer a better path forward, since we should expect that not only are individual latents generally sparse, they are also **sparsely connected** - any given latent should probably only have a downstream effect on a small number of other latents.

Indeed, if this does end up being true, it's a strong piece of evidence that latents found by SAEs *are* the **fundamental units of computation** used by the model, as opposed to just being an interesting clustering algorithm. Of course we do already have some evidence for this (e.g. the effectiveness of latent steering, and the fact that latents have already revealed important information about models which isn't clear when just looking at the basic components), but finding clear latent circuits would be a much stronger piece of evidence.
"""


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Latent Gradients

We'll start with an exercise that illustrates the kind of sparsity you can expect in latent connections, as well as many of the ways latent circuit analysis can be challenging. We'll be implementing the `latent_to_latent_gradients` function, which returns the gradients between all active pairs of latents belonging to SAEs in two different layers (we'll be using two SAEs from our `gpt2-small-res-jb` release). These exercises will be split up into a few different steps, since computing these gradients is deceptively complex.

What exactly are latent gradients? Well, for any given input, and any 2 latents in different layers, we can compute the derivative of the second latent's activation with respect to the first latent. This takes the form of a matrix of partial derivatives, i.e. $J_{ij} = \frac{\partial f_i}{\partial x_j}$, and can serve as a linear proxy for how latents in an early layer contribute to latents in a later layer. The pseudocode for computing this is:

```python
# Computed with no gradients, and not patching in SAE reconstructions...
layer_1_latents, layer_2_latents = model.run_with_cache_with_saes(...)

def latent_acts_to_later_latent_acts(layer_1_latents):
    layer_1_resid_acts_recon = SAE_1_decoder(layer_1_latents)
    layer_2_resid_acts_recon = model.blocks[layer_1: layer_2].forward(layer_1_resid_acts_recon)
    layer_2_latents_recon = SAE_2_encoder(layer_2_resid_acts_recon)
    return layer_2_latents_recon

latent_latent_gradients = t.func.jacrev(latent_acts_to_later_latent_acts)(layer_1_latents)
```

where `jacrev` is shorthand for "Jacobian reverse-mode differentiation" - it's a PyTorch function that takes in a tensor -> tensor function `f(x) = y` and returns the Jacobian function, i.e. `g` s.t. `g[i, j] = d(f[x]_i) / d(x_j)`.

If we wanted to get a sense of how latents communicate with each other across our distribution of data, then we might average these results over a large set of prompts. However for now, we're going to stick with a relatively small set of prompts to avoid running into memory issues, and so we can visualise the results more easily.

First, let's load in our model & SAEs, if you haven't already:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_1:
    gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

    gpt2_saes = {
        layer: SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device=str(device),
        )[0]
        for layer in tqdm(range(gpt2.cfg.n_layers))
    }

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
Now, we can start the exercises!

Note - the subsequent 3 exercises are all somewhat involved, and things like the use of Jacobian can be quite fiddly. For that reason, there's a good case to be made for just reading through the solutions and understanding what the code is doing, rather than trying to do it yourself. One option would be to look at the solutions for these 3 exercises and understand how latent-to-latent gradients work, but then try and implement the `token_to_latent_gradients` function (after the next 3 exercises) yourself.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise (1/3) - implement the `SparseTensor` class

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵⚪⚪⚪⚪
> 
> You should spend up to 10-15 minutes on this exercise (or skip it)
> ```

Firstly, we're going to create a `SparseTensor` class to help us work with sparse tensors (i.e. tensors where most of the elements are zero). This is because we'll need to do forward passes on the dense tensors (i.e. the tensors containing all values, including the zeros) but we'll often want to compute gradients wrt the sparse tensors (just the non-zero values) because otherwise we'd run into memory issues - there are a lot of latents!

You should fill in the `from_dense` and `from_sparse` class methods for the `SparseTensor` class. The testing code is visible to you, and should help you understand how this class is expected to behave.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


class SparseTensor:
    """
    Handles 2D tensor data (assumed to be non-negative) in 2 different formats:
        dense:  The full tensor, which contains zeros. Shape is (n1, ..., nk).
        sparse: A tuple of nonzero values with shape (n_nonzero,), nonzero indices with shape
                (n_nonzero, k), and the shape of the dense tensor.
    """

    sparse: tuple[Tensor, Tensor, tuple[int, ...]]
    dense: Tensor

    def __init__(self, sparse: tuple[Tensor, Tensor, tuple[int, ...]], dense: Tensor):
        self.sparse = sparse
        self.dense = dense

    @classmethod
    def from_dense(cls, dense: Tensor) -> "SparseTensor":
        # SOLUTION
        sparse = (dense[dense > 0], t.argwhere(dense > 0), tuple(dense.shape))
        return cls(sparse, dense)
        # END SOLUTION
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE

    @classmethod
    def from_sparse(cls, sparse: tuple[Tensor, Tensor, tuple[int, ...]]) -> "SparseTensor":
        # SOLUTION
        nonzero_values, nonzero_indices, shape = sparse
        dense = t.zeros(shape, dtype=nonzero_values.dtype, device=nonzero_values.device)
        dense[nonzero_indices.unbind(-1)] = nonzero_values
        return cls(sparse, dense)
        # END SOLUTION
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE

    @property
    def values(self) -> Tensor:
        return self.sparse[0].squeeze()

    @property
    def indices(self) -> Tensor:
        return self.sparse[1].squeeze()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.sparse[2]


# HIDE
if MAIN and FLAG_RUN_SECTION_1:
    # Test `from_dense`
    x = t.zeros(10_000)
    nonzero_indices = t.randint(0, 10_000, (10,)).sort().values
    nonzero_values = t.rand(10)
    x[nonzero_indices] = nonzero_values
    sparse_tensor = SparseTensor.from_dense(x)
    t.testing.assert_close(sparse_tensor.sparse[0], nonzero_values)
    t.testing.assert_close(sparse_tensor.sparse[1].squeeze(-1), nonzero_indices)
    t.testing.assert_close(sparse_tensor.dense, x)

    # Test `from_sparse`
    sparse_tensor = SparseTensor.from_sparse((nonzero_values, nonzero_indices.unsqueeze(-1), tuple(x.shape)))
    t.testing.assert_close(sparse_tensor.dense, x)

    # Verify other properties
    t.testing.assert_close(sparse_tensor.values, nonzero_values)
    t.testing.assert_close(sparse_tensor.indices, nonzero_indices)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise (2/3) - implement `latent_acts_to_later_latent_acts`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 10-20 minutes on this exercise.
> ```

Next, you should implement the `latent_acts_to_later_latent_acts`. This takes latent activations earlier in the model (in a sparse form, i.e. tuple of (values, indices, shape)) and returns the downstream latent activations as a tuple of `(sparse_values, (dense_values,))`.

Why do we return 2 copies of `latent_acts_next` in this strange way? The answer is that we'll be wrapping our function with `t.func.jacrev(latent_acts_to_later_latent_acts, has_aux=True)`. The `has_aux` argument allows us to return a tuple of tensors which won't be differentiated. In other words, it takes a tensor -> (tensor, tuple_of_tensors) function `f(x) = (y, aux)` and returns the function `g(x) = (J, aux)` where `J[i, j] = d(f[x]_i) / d(x_j)`. In other words, we're getting both the Jacobian and the actual reconstructed activations.

<details>
<summary>Note on what gradients we're actually computing</summary>

Eagle-eyed readers might have noticed that what we're actually doing here is not computing gradients between later and earlier latents, but computing the gradient between **reconstructed later latents** and earlier latents. In other words, the later latents we're differentiating are actually a function of the earlier SAE's residual stream reconstruction, rather than the actual residual stream. This is a bit risky when drawing conclusions from the results, because if your earlier SAE isn't very good at reconstructing its input then you might miss out on ways in which downstream latents are affected by upstream activations. A good way to sanity check this is to compare the latent activations (computed downstream of the earlier SAE's reconstructions) with the true latent activations, and make sure they're similar.

</details>

We'll get to applying the Jacobian in the 3rd exercise though - for now, you should just fill in `latent_acts_to_later_latent_acts`. This should essentially match the pseudocode for `latent_acts_to_later_latent_acts` which we gave at the start of this section (with the added factor of having to convert tensors to / from their sparse forms). Some guidance on syntax you'll find useful:

- All SAEs have `encode` and `decode` methods, which map from input -> latent activations -> reconstructed input.
- All TransformerLens models have a `forward` method with optional arguments `start_at_layer` and `stop_at_layer`, if these are supplied then it will compute activations from the latter layer as a function of the former.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def latent_acts_to_later_latent_acts(
    latent_acts_nonzero: Float[Tensor, " nonzero_acts"],
    latent_acts_nonzero_inds: Int[Tensor, "nonzero_acts n_indices"],
    latent_acts_shape: tuple[int, ...],
    sae_from: SAE,
    sae_to: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, tuple[Tensor]]:
    """
    Given some latent activations for a residual stream SAE earlier in the model, computes the
    latent activations of a later SAE. It does this by mapping the latent activations through the
    path SAE decoder -> intermediate model layers -> later SAE encoder.

    This function must input & output sparse information (i.e. nonzero values and their indices)
    rather than dense tensors, because latent activations are sparse but jacrev() doesn't support
    gradients on real sparse tensors.
    """
    # EXERCISE
    # # ... YOUR CODE HERE ...
    # END EXERCISE
    # SOLUTION
    # Convert to dense, map through SAE decoder
    latent_acts = SparseTensor.from_sparse((latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape)).dense
    resid_stream_from = sae_from.decode(latent_acts)

    # Map through model layers
    resid_stream_next = model.forward(
        resid_stream_from,
        start_at_layer=sae_from.cfg.hook_layer,
        stop_at_layer=sae_to.cfg.hook_layer,
    )

    # Map through SAE encoder, and turn back into SparseTensor
    latent_acts_next_recon = sae_to.encode(resid_stream_next)
    latent_acts_next_recon = SparseTensor.from_dense(latent_acts_next_recon)
    # END SOLUTION

    return latent_acts_next_recon.sparse[0], (latent_acts_next_recon.dense,)


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise (3/3) - implement `latent_to_latent_gradients`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 20-40 minutes on this exercise.
> ```

Finally, we're in the position to implement our full `latent_to_latent_gradients` function. This function should:

- Compute the true latent activations for both SAEs, using `run_with_cache_with_saes` (make sure you set `sae_from.use_error_term = True`, because you want to compute the true latent activations for the later SAE, not those which are computed from the earlier SAE's reconstructions!)
- Wrap your function `latent_acts_to_later_latent_acts` to create a function that will return the Jacobian and the later latent activations (code in a dropdown below if you're confused about what this looks like),
- Call this function to return the Jacobian and the later latent activations,
- Return the Jacobian and earlier/later latent activations (the latter as `SparseTensor` objects).

<details>
<summary>Code for the Jacobian wrapper</summary>

```python
latent_acts_to_later_latent_acts_and_gradients = t.func.jacrev(
    latent_acts_to_later_latent_acts, argnums=0, has_aux=True
)
```

The `argnums=0` argument tells `jacrev` to take the Jacobian with respect to the first argument of `latent_acts_to_later_latent_acts`, and the `has_aux=True` argument tells it to also return the auxiliary outputs of `latent_acts_to_later_latent_acts` (i.e. the tuple of tensors which are the second output of the base function).

You can call this function using:

```python
latent_latent_gradients, (latent_acts_next_recon_dense,) = latent_acts_to_later_latent_acts_and_gradients(
    *latent_acts_prev.sparse, sae_from, sae_to, model
)
```

</details>

<details>
<summary>Help - I'm getting OOM errors</summary>

OOM errors are common when you pass in tensors which aren't the sparsified versions (because computing a 2D matrix of derivatives of 10k+ elements is pretty memory intensive!). We recommend you look at the amount of memory being asked for when you get errors; if it's 30GB+ then you're almost certainly making this mistake.

If you're still getting errors, we recommend you inspect and clear your memory. In particular, loading large models like Gemma onto the GPU will be taking up space that you no longer need. We've provided some util functions for this purpose (we give examples of how to use them at the very start of this notebook, before the first set of exercises, under the header "A note on memory usage").

If all this still doesn't work (i.e. you still get errors after clearing memory), we recommend you try a virtual machine (e.g. vastai) or Colab notebook.

</details>

We've given you code below this function, which will run and create a heatmap of the gradients for you. Note, the plot axes use notation of `F{layer}.{latent_idx}` for the latents.

Challenge - can you find a pair of latents which seem to form a circuit on bigrams consisting of tokenized words where the first token is `" E"` ?
"""

# ! CELL TYPE: code
# ! FILTERS: [soln,py]

if MAIN and FLAG_RUN_SECTION_1:
    try:
        del gemma_2_2b
        del gemma_2_2b_sae
    except NameError:
        pass

    THRESHOLD = 0.1  # GB
    for obj in gc.get_objects():
        try:
            if isinstance(obj, t.nn.Module) and utils.get_tensors_size(obj) / 1024**3 > THRESHOLD:
                if hasattr(obj, "cuda"):
                    obj.cpu()
                if hasattr(obj, "reset"):
                    obj.reset()
        except:
            pass

    gpt2.to(device)
    gpt2_saes = {layer: sae.to(device) for layer, sae in gpt2_saes.items()}

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def latent_to_latent_gradients(
    tokens: Float[Tensor, "batch seq"],
    sae_from: SAE,
    sae_to: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, SparseTensor, SparseTensor, SparseTensor]:
    """
    Computes the gradients between all active pairs of latents belonging to two SAEs.

    Returns:
        latent_latent_gradients:    The gradients between all active pairs of latents
        latent_acts_prev:           The latent activations of the first SAE
        latent_acts_next:           The latent activations of the second SAE
        latent_acts_next_recon:     The reconstructed latent activations of the second SAE (i.e.
                                    based on the first SAE's reconstructions)
    """
    # EXERCISE
    # # ... YOUR CODE HERE ...
    # END EXERCISE
    # SOLUTION
    acts_prev_name = f"{sae_from.cfg.hook_name}.hook_sae_acts_post"
    acts_next_name = f"{sae_to.cfg.hook_name}.hook_sae_acts_post"
    sae_from.use_error_term = True  # so we can get both true latent acts at once

    with t.no_grad():
        # Get the true activations for both SAEs
        _, cache = model.run_with_cache_with_saes(
            tokens,
            names_filter=[acts_prev_name, acts_next_name],
            stop_at_layer=sae_to.cfg.hook_layer + 1,
            saes=[sae_from, sae_to],
            remove_batch_dim=False,
        )
        latent_acts_prev = SparseTensor.from_dense(cache[acts_prev_name])
        latent_acts_next = SparseTensor.from_dense(cache[acts_next_name])

    # Compute jacobian between earlier and later latent activations (and also get the activations
    # of the later SAE which are downstream of the earlier SAE's reconstructions)
    latent_latent_gradients, (latent_acts_next_recon_dense,) = t.func.jacrev(
        latent_acts_to_later_latent_acts, has_aux=True
    )(
        *latent_acts_prev.sparse,
        sae_from,
        sae_to,
        model,
    )

    latent_acts_next_recon = SparseTensor.from_dense(latent_acts_next_recon_dense)

    # Set SAE state back to default
    sae_from.use_error_term = False
    # END SOLUTION

    return (
        latent_latent_gradients,
        latent_acts_prev,
        latent_acts_next,
        latent_acts_next_recon,
    )


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_1:
    prompt = "The Eiffel tower is in Paris"
    tokens = gpt2.to_tokens(prompt)
    str_toks = gpt2.to_str_tokens(prompt)
    layer_from = 0
    layer_to = 3

    # Get latent-to-latent gradients
    t.cuda.empty_cache()
    t.set_grad_enabled(True)
    (
        latent_latent_gradients,
        latent_acts_prev,
        latent_acts_next,
        latent_acts_next_recon,
    ) = latent_to_latent_gradients(tokens, gpt2_saes[layer_from], gpt2_saes[layer_to], gpt2)
    t.set_grad_enabled(False)

    # Verify that ~the same latents are active in both, and the MSE loss is small
    nonzero_latents = [tuple(x) for x in latent_acts_next.indices.tolist()]
    nonzero_latents_recon = [tuple(x) for x in latent_acts_next_recon.indices.tolist()]
    alive_in_one_not_both = set(nonzero_latents) ^ set(nonzero_latents_recon)
    print(f"# nonzero latents (true): {len(nonzero_latents)}")
    print(f"# nonzero latents (reconstructed): {len(nonzero_latents_recon)}")
    print(f"# latents alive in one but not both: {len(alive_in_one_not_both)}")

    px.imshow(
        to_numpy(latent_latent_gradients.T),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        x=[f"F{layer_to}.{latent}, {str_toks[seq]!r} ({seq})" for (_, seq, latent) in latent_acts_next_recon.indices],
        y=[f"F{layer_from}.{latent}, {str_toks[seq]!r} ({seq})" for (_, seq, latent) in latent_acts_prev.indices],
        labels={"x": f"To layer {layer_to}", "y": f"From layer {layer_from}"},
        title=f'Gradients between SAE latents in layer {layer_from} and SAE latents in layer {layer_to}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
        width=1600,
        height=1000,
    ).show()

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html,st-dropdown[Click to see the expected output]]

r"""
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"># nonzero latents (true): 181
# nonzero latents (reconstructed): 179
# latents alive in one but not both: 8</pre>

<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13220.html" width="1620" height="1020"></div>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
<details>
<summary>Some observations</summary>

Many of the nonzero gradients are for pairs of tokens which fire on the same token. For example, `(F0.9449, " Paris") -> (F3.385, " Paris")` seems like it could just be a similar feature in 2 different layers:

```python
display_dashboard(sae_id="blocks.0.hook_resid_pre", latent_idx=9449)
display_dashboard(sae_id="blocks.3.hook_resid_pre", latent_idx=385)
```

There aren't as many cross-token gradients. One of the most notable is `(F0.16911, " E") -> (F3.15266, "iff")` which seems like it could be a bigram circuit for words which start with `" E"`:

```python
display_dashboard(sae_id="blocks.0.hook_resid_pre", latent_idx=16911)
display_dashboard(sae_id="blocks.3.hook_resid_pre", latent_idx=15266)
```

</details>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - get latent-to-token gradients

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵⚪⚪⚪
> 
> You should spend up to 30-40 minutes on this exercise.
> ```

Now that we've worked through implementing latent-to-latent gradients, let's try doing the whole thing again, but instead computing the gradients between all input tokens and a particular SAE's latents.

You might be wondering what gradients between tokens and latents even mean, because tokens aren't scalar values. The answer is that we'll be multiplying the model's embeddings by some scale factor `s` (i.e. a vector of different scale factor values for each token in our sequence), and taking the gradient of the SAE's latents wrt these values `s`, evaluated at `s = [1, 1, ..., 1]`. This isn't super principled since in practice this kind of embedding vector scaling doesn't happen in our model, but it's a convenient way to get a sense of **which tokens are most important for which latents**.

Challenge - take the pair of latents from the previous exercise which seemed to form a circuit on bigrams consisting of tokenized words where the first token is `" E"`. Can you find that circuit again, from this plot?
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def tokens_to_latent_acts(
    token_scales: Float[Tensor, "batch seq"],
    tokens: Int[Tensor, "batch seq"],
    sae: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, tuple[Tensor]]:
    """
    Given scale factors for model's embeddings (i.e. scale factors applied after we compute the sum
    of positional and token embeddings), returns the SAE's latents.

    Returns:
        latent_acts_sparse: The SAE's latents in sparse form (i.e. the tensor of values)
        latent_acts_dense:  The SAE's latents in dense tensor, in a length-1 tuple
    """
    # EXERCISE
    # # ... YOUR CODE HERE ...
    # END EXERCISE
    # SOLUTION
    resid_after_embed = model(tokens, stop_at_layer=0)
    resid_after_embed = einops.einsum(resid_after_embed, token_scales, "... seq d_model, ... seq -> ... seq d_model")
    resid_before_sae = model(resid_after_embed, start_at_layer=0, stop_at_layer=sae.cfg.hook_layer)

    sae_latents = sae.encode(resid_before_sae)
    sae_latents = SparseTensor.from_dense(sae_latents)
    # END SOLUTION

    return sae_latents.sparse[0], (sae_latents.dense,)


def token_to_latent_gradients(
    tokens: Float[Tensor, "batch seq"],
    sae: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, SparseTensor]:
    """
    Computes the gradients between an SAE's latents and all input tokens.

    Returns:
        token_latent_grads: The gradients between input tokens and SAE latents
        latent_acts:        The SAE's latent activations
    """
    # EXERCISE
    # # ... YOUR CODE HERE ...
    # END EXERCISE
    # SOLUTION
    # Find the gradients from token positions to latents
    token_scales = t.ones(tokens.shape, device=model.cfg.device, requires_grad=True)
    token_latent_grads, (latent_acts_dense,) = t.func.jacrev(tokens_to_latent_acts, has_aux=True)(
        token_scales, tokens, sae, model
    )

    token_latent_grads = einops.rearrange(token_latent_grads, "d_sae_nonzero batch seq -> batch seq d_sae_nonzero")

    latent_acts = SparseTensor.from_dense(latent_acts_dense)
    # END SOLUTION

    return (token_latent_grads, latent_acts)


# HIDE
if MAIN and FLAG_RUN_SECTION_1:
    sae_layer = 3
    token_latent_grads, latent_acts = token_to_latent_gradients(tokens, sae=gpt2_saes[sae_layer], model=gpt2)

    px.imshow(
        to_numpy(token_latent_grads[0]),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        x=[f"F{sae_layer}.{latent:05}, {str_toks[seq]!r} ({seq})" for (_, seq, latent) in latent_acts.indices],
        y=[f"{str_toks[i]!r} ({i})" for i in range(len(str_toks))],
        labels={"x": f"To layer {sae_layer}", "y": "From tokens"},
        title=f'Gradients between input tokens and SAE latents in layer {sae_layer}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
        width=1900,
        height=450,
    )
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html,st-dropdown[Click to see the expected output]]

r"""
<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13221.html" width="1920" height="470"></div>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
<details>
<summary>Some observations</summary>

In the previous exercise, we saw gradients between `(F0.16911, " E") -> (F3.15266, "iff")`, which seems like it could be forming a bigram circuit for words which start with `" E"`.

In this plot, we can see a gradient between the `" E"` token and feature `F3.15266`, which is what we'd expect based on this.

</details>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - get latent-to-logit gradients

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵⚪⚪⚪
> 
> You should spend up to 30-45 minutes on this exercise.
> ```

Finally, we'll have you compute the latent-to-logit gradients. This exercise will be quite similar to the first one (i.e. the latent-to-latent gradients), but this time the function you pass through `jacrev` will map SAE activations to logits, rather than to a later SAE's latents. Note that we've given you the argument `k` to specify only a certain number of top logits to take gradients for (because otherwise you might be computing a large gradient matrix, which could cause an OOM error).

Why are we bothering to compute latent-to-logit gradients in the first place? Well, one obvious answer is that it completes our end-to-end circuits picture (we've now got tokens -> latents -> other latents -> logits). But to give another answer, we can consider latents as having a dual nature: when looking back towards the input, they are **representations**, but when looking forward towards the logits, they are **actions**. We might expect sparsity in both directions, in other words not only should latents sparsely represent the activations produced by the input, they should also sparsely affect the gradients influencing the output. As you'll see when doing these exercises, this is partially the case, although not to the same degree as we saw extremely sparse token-to-latent or latent-to-latent gradients. One reason for that is that sparsity as representations is already included in the SAE's loss function (the L1 penalty), but there's no explicit penalty term to encourage sparsity in the latent gradients. Anthropic propose what this might look like in their [April 2024 update](https://transformer-circuits.pub/2024/april-update/index.html#attr-dl).

However, despite the results for latent-to-logit gradients being less sparse than the last 2 exercises, they can still teach us a lot about which latents are important for a particular input prompt. Fill in the functions below, and then play around with some latent-to-logit gradients yourself!
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def latent_acts_to_logits(
    latent_acts_nonzero: Float[Tensor, " nonzero_acts"],
    latent_acts_nonzero_inds: Int[Tensor, "nonzero_acts n_indices"],
    latent_acts_shape: tuple[int, ...],
    sae: SAE,
    model: HookedSAETransformer,
    token_ids: list[int] | None = None,
) -> tuple[Tensor, tuple[Tensor]]:
    """
    Computes the logits as a downstream function of the SAE's reconstructed residual stream. If we
    supply `token_ids`, it means we only compute & return the logits for those specified tokens.
    """
    # EXERCISE
    # ...
    # END EXERCISE
    # SOLUTION
    # Convert to dense, map through SAE decoder
    latent_acts = SparseTensor.from_sparse((latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape)).dense

    resid = sae.decode(latent_acts)

    # Map through model layers, to the end
    logits_recon = model(resid, start_at_layer=sae.cfg.hook_layer)[0, -1]

    # END SOLUTION
    return logits_recon[token_ids], (logits_recon,)


def latent_to_logit_gradients(
    tokens: Float[Tensor, "batch seq"],
    sae: SAE,
    model: HookedSAETransformer,
    k: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, list[int] | None, SparseTensor]:
    """
    Computes the gradients between active latents and some top-k set of logits (we
    use k to avoid having to compute the gradients for all tokens).

    Returns:
        latent_logit_gradients:  The gradients between the SAE's active latents & downstream logits
        logits:                  The model's true logits
        logits_recon:            The model's reconstructed logits (i.e. based on SAE reconstruction)
        token_ids:               The tokens we computed the gradients for
        latent_acts:             The SAE's latent activations
    """
    assert tokens.shape[0] == 1, "Only supports batch size 1 for now"

    # EXERCISE
    # ...
    # END EXERCISE
    # SOLUTION
    acts_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    sae.use_error_term = True

    with t.no_grad():
        # Run model up to the position of the first SAE to get those residual stream activations
        logits, cache = model.run_with_cache_with_saes(
            tokens,
            names_filter=[acts_hook_name],
            saes=[sae],
            remove_batch_dim=False,
        )
        latent_acts = cache[acts_hook_name]
        latent_acts = SparseTensor.from_dense(latent_acts)

        logits = logits[0, -1]

    # Get the tokens we'll actually compute gradients for
    token_ids = None if k is None else logits.topk(k=k).indices.tolist()

    # Compute jacobian between latent acts and logits
    latent_logit_gradients, (logits_recon,) = t.func.jacrev(latent_acts_to_logits, has_aux=True)(
        *latent_acts.sparse, sae, model, token_ids
    )

    sae.use_error_term = False
    # END SOLUTION

    return (
        latent_logit_gradients,
        logits,
        logits_recon,
        token_ids,
        latent_acts,
    )


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_1:
    layer = 9
    prompt = "The Eiffel tower is in the city of"
    answer = " Paris"

    tokens = gpt2.to_tokens(prompt, prepend_bos=True)
    str_toks = gpt2.to_str_tokens(prompt, prepend_bos=True)
    k = 25

    # Test the model on this prompt, with & without SAEs
    test_prompt(prompt, answer, gpt2)

    # How about the reconstruction? More or less; it's rank 20 so still decent
    gpt2_saes[layer].use_error_term = False
    with gpt2.saes(saes=[gpt2_saes[layer]]):
        test_prompt(prompt, answer, gpt2)

    latent_logit_grads, logits, logits_recon, token_ids, latent_acts = latent_to_logit_gradients(
        tokens, sae=gpt2_saes[layer], model=gpt2, k=k
    )

    # sort by most positive in " Paris" direction
    sorted_indices = latent_logit_grads[0].argsort(descending=True)
    latent_logit_grads = latent_logit_grads[:, sorted_indices]

    px.imshow(
        to_numpy(latent_logit_grads),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        x=[
            f"{str_toks[seq]!r} ({seq}), latent {latent:05}" for (_, seq, latent) in latent_acts.indices[sorted_indices]
        ],
        y=[f"{tok!r} ({gpt2.to_single_str_token(tok)})" for tok in token_ids],
        labels={"x": f"Features in layer {layer}", "y": "Logits"},
        title=f'Gradients between SAE latents in layer {layer} and final logits (only showing top {k} logits)<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
        width=1900,
        height=800,
        aspect="auto",
    ).show()

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html,st-dropdown[Click to see the expected output]]

r"""
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
Tokenized prompt: ['<|endoftext|>', 'The', ' E', 'iff', 'el', ' tower', ' is', ' in', ' the', ' city', ' of']
Tokenized answer: [' Paris']

Performance on answer token:
<span style="font-weight: bold">Rank: </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">        Logit: </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14.83</span><span style="font-weight: bold"> Prob:  </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4.76</span><span style="font-weight: bold">% Token: | Paris|</span>

Top 0th token. Logit: 14.83 Prob:  4.76% Token: | Paris|
Top 1th token. Logit: 14.63 Prob:  3.90% Token: | London|
Top 2th token. Logit: 14.47 Prob:  3.32% Token: | Amsterdam|
Top 3th token. Logit: 14.02 Prob:  2.11% Token: | Berlin|
Top 4th token. Logit: 13.90 Prob:  1.87% Token: | L|
Top 5th token. Logit: 13.85 Prob:  1.78% Token: | E|
Top 6th token. Logit: 13.77 Prob:  1.64% Token: | Hamburg|
Top 7th token. Logit: 13.75 Prob:  1.61% Token: | B|
Top 8th token. Logit: 13.61 Prob:  1.40% Token: | Cologne|
Top 9th token. Logit: 13.58 Prob:  1.36% Token: | St|

<span style="font-weight: bold">Ranks of the answer tokens:</span> <span style="font-weight: bold">[(</span><span style="color: #008000; text-decoration-color: #008000">' Paris'</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">)]</span>
<br>
Tokenized prompt: ['<|endoftext|>', 'The', ' E', 'iff', 'el', ' tower', ' is', ' in', ' the', ' city', ' of']
Tokenized answer: [' Paris']

Performance on answer token:
<span style="font-weight: bold">Rank: </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span><span style="font-weight: bold">        Logit: </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13.20</span><span style="font-weight: bold"> Prob:  </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.18</span><span style="font-weight: bold">% Token: | Paris|</span>

Top 0th token. Logit: 13.83 Prob:  2.23% Token: | New|
Top 1th token. Logit: 13.74 Prob:  2.03% Token: | London|
Top 2th token. Logit: 13.71 Prob:  1.96% Token: | San|
Top 3th token. Logit: 13.70 Prob:  1.94% Token: | Chicago|
Top 4th token. Logit: 13.59 Prob:  1.75% Token: | E|
Top 5th token. Logit: 13.47 Prob:  1.54% Token: | Berlin|
Top 6th token. Logit: 13.28 Prob:  1.27% Token: | L|
Top 7th token. Logit: 13.27 Prob:  1.27% Token: | Los|
Top 8th token. Logit: 13.21 Prob:  1.20% Token: | St|
Top 9th token. Logit: 13.20 Prob:  1.18% Token: | Paris|

<span style="font-weight: bold">Ranks of the answer tokens:</span> <span style="font-weight: bold">[(</span><span style="color: #008000; text-decoration-color: #008000">' Paris'</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span><span style="font-weight: bold">)]</span>
</pre>
<br>

<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13222.html" width="1920" height="820"></div>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
<details>
<summary>Some observations</summary>

We see that feature `F9.22250` stands out as boosting the `" Paris"` token far more than any of the other top predictions. Investigation reveals this feature fires primarily on French language text, which makes sense!

We also see `F9.5879` which seems to strongly boost words associated with Germany (e.g. Berlin, Hamberg, Cologne, Zurich). We see a similar pattern there, where that feature mostly fires on German-language text (or more commonly, English-language text talking about Germany).

```python
display_dashboard(sae_id="blocks.9.hook_resid_pre", latent_idx=22250)
display_dashboard(sae_id="blocks.9.hook_resid_pre", latent_idx=5879)
```

</details>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise (optional) - find induction circuits in attention SAEs

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 30-60 minutes on this exercise.
> ```

You can study MLP or attention features in much the same way as you've studied residual stream features, for any of the last 3 sets of exercises. For example, if we wanted to compute the gradient of logits or some later SAE with respect to an earlier SAE which was trained on the MLP layer, we just replace the MLP layer's activations with that earlier SAE's decoded activations, and then compute the Jacobian of our downstream values wrt this earlier SAE's activations. Note, we can use something like `model.run_with_hooks` to perform this replacement, without having to manually perform every step of the model's forward pass ourselves.

Try and write a version of the `feature_to_feature_gradients` function which works for attention SAEs (docstring below). Can you use this function to find previous token features & induction features which have non-zero gradients with respect to each other, and come together to form induction circuits?

<details>
<summary>Hint - where you should be looking</summary>

Start by generating a random sequence of tokens, and using `circuitvis` to visualize the attention patterns:

```python
import circuitsvis as cv

seq_len = 10
tokens = t.randint(0, model.cfg.d_vocab, (1, seq_len)).tolist()[0]
tokens = t.tensor([model.tokenizer.bos_token_id] + tokens + tokens)

_, cache = model.run_with_cache(tokens)

prev_token_heads = [(4, 11)]
induction_heads = [(5, 1), (5, 5), (5, 8)]
all_heads = prev_token_heads + induction_heads

html = cv.attention.attention_patterns(
    tokens=model.to_str_tokens(tokens),
    attention=t.stack([cache["pattern", layer][0, head] for layer, head in all_heads]),
    attention_head_names=[f"{layer}.{head}" for (layer, head) in all_heads],
)
display(html)
```

You can see from this that layer 4 contains a clear previous token head, and 5 contains several induction heads. So you might want to look at feature-feature gradients between layer 4 and layer 5.

Remember - the induction circuit works by having sequences `AB...AB`, where the previous token head attends from the first `B` back to the first `A`, then the induction head attends from the second `A` back to the first `B`. Keep this in mind when you're looking the for evidence of an induction circuit in the feature-feature gradients heatmap.

</details>

The code below this function plots feature-to-feature gradients, and it also adds black squares to the instances where a layer-4 feature fires on the first `B` and layer-5 feature fires on the second `A`, in the `AB...AB` pattern (this is what we expect from an induction circuit). In other words, if your function is working then you should see a pattern of nonzero values in the regions indicated by these squares.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def latent_acts_to_later_latent_acts_attn(
    latent_acts_nonzero: Float[Tensor, " nonzero_acts"],
    latent_acts_nonzero_inds: Int[Tensor, "nonzero_acts n_indices"],
    latent_acts_shape: tuple[int, ...],
    sae_from: SAE,
    sae_to: SAE,
    model: HookedSAETransformer,
    resid_pre_clean: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Returns the latent activations of an attention SAE, computed downstream of an earlier SAE's
    output (whose values are given in sparse form as the first three arguments).

    `resid_pre_clean` is also supplied, i.e. these are the input values to the attention layer in
    which the earlier SAE is applied.
    """
    # EXERCISE
    # ...
    # END EXERCISE
    # SOLUTION
    # Convert to dense, map through SAE decoder
    latent_acts = SparseTensor.from_sparse((latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape)).dense
    z_recon = sae_from.decode(latent_acts)

    hook_name_z_prev = get_act_name("z", sae_from.cfg.hook_layer)
    hook_name_z_next = get_act_name("z", sae_to.cfg.hook_layer)

    def hook_set_z_prev(z: Tensor, hook: HookPoint):
        return z_recon

    def hook_store_z_next(z: Tensor, hook: HookPoint):
        hook.ctx["z"] = z

    # fwd pass: replace earlier z with SAE reconstructions, and store later z (no SAEs needed yet)
    model.run_with_hooks(
        resid_pre_clean,
        start_at_layer=sae_from.cfg.hook_layer,
        stop_at_layer=sae_to.cfg.hook_layer + 1,
        fwd_hooks=[
            (hook_name_z_prev, hook_set_z_prev),
            (hook_name_z_next, hook_store_z_next),
        ],
    )
    z = model.hook_dict[hook_name_z_next].ctx.pop("z")
    latent_acts_next_recon = SparseTensor.from_dense(sae_to.encode(z))
    # END SOLUTION

    return latent_acts_next_recon.sparse[0], (latent_acts_next_recon.dense,)


def latent_to_latent_gradients_attn(
    tokens: Float[Tensor, "batch seq"],
    sae_from: SAE,
    sae_to: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, SparseTensor, SparseTensor, SparseTensor]:
    """
    Computes the gradients between all active pairs of latents belonging to two SAEs. Both SAEs
    are assumed to be attention SAEs, i.e. they take the concatenated z values as input.

    Returns:
        latent_latent_gradients:  The gradients between all active pairs of latents
        latent_acts_prev:          The latent activations of the first SAE
        latent_acts_next:          The latent activations of the second SAE
        latent_acts_next_recon:    The reconstructed latent activations of the second SAE
    """
    # EXERCISE
    # ...
    # END EXERCISE
    # SOLUTION
    resid_pre_name = get_act_name("resid_pre", sae_from.cfg.hook_layer)
    acts_prev_name = f"{sae_from.cfg.hook_name}.hook_sae_acts_post"
    acts_next_name = f"{sae_to.cfg.hook_name}.hook_sae_acts_post"
    sae_from.use_error_term = True  # so we can get both true latent acts at once
    sae_to.use_error_term = True  # so we can get both true latent acts at once

    with t.no_grad():
        # Get the true activations for both SAEs
        _, cache = model.run_with_cache_with_saes(
            tokens,
            names_filter=[resid_pre_name, acts_prev_name, acts_next_name],
            stop_at_layer=sae_to.cfg.hook_layer + 1,
            saes=[sae_from, sae_to],
            remove_batch_dim=False,
        )
        latent_acts_prev = SparseTensor.from_dense(cache[acts_prev_name])
        latent_acts_next = SparseTensor.from_dense(cache[acts_next_name])

    # Compute jacobian between earlier and later latent activations (and also get the activations
    # of the later SAE which are downstream of the earlier SAE's reconstructions)
    latent_latent_gradients, (latent_acts_next_recon_dense,) = t.func.jacrev(
        latent_acts_to_later_latent_acts_attn, has_aux=True
    )(*latent_acts_prev.sparse, sae_from, sae_to, model, cache[resid_pre_name])

    latent_acts_next_recon = SparseTensor.from_dense(latent_acts_next_recon_dense)

    # Set SAE state back to default
    sae_from.use_error_term = False
    sae_to.use_error_term = False
    # END SOLUTION

    return (
        latent_latent_gradients,
        latent_acts_prev,
        latent_acts_next,
        latent_acts_next_recon,
    )


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_1:
    attn_saes = {
        layer: SAE.from_pretrained(
            "gpt2-small-hook-z-kk",
            f"blocks.{layer}.hook_z",
            device=str(device),
        )[0]
        for layer in range(gpt2.cfg.n_layers)
    }

    seq_len = 10  # higher seq len / more batches would be more reliable, but this simplifies the plot
    tokens = t.randint(0, gpt2.cfg.d_vocab, (1, seq_len)).tolist()[0]
    tokens = t.tensor([gpt2.tokenizer.bos_token_id] + tokens + tokens)
    str_toks = gpt2.to_str_tokens(tokens)
    layer_from = 4
    layer_to = 5

    # Get latent-to-latent gradients
    t.set_grad_enabled(True)
    (
        latent_latent_gradients,
        latent_acts_prev,
        latent_acts_next,
        latent_acts_next_recon,
    ) = latent_to_latent_gradients_attn(tokens, attn_saes[layer_from], attn_saes[layer_to], gpt2)
    t.set_grad_enabled(False)

    # Verify that ~the same latents are active in both, and the MSE loss is small
    nonzero_latents = [tuple(x) for x in latent_acts_next.indices.tolist()]
    nonzero_latents_recon = [tuple(x) for x in latent_acts_next_recon.indices.tolist()]
    alive_in_one_not_both = set(nonzero_latents) ^ set(nonzero_latents_recon)
    print(f"# nonzero latents (true): {len(nonzero_latents)}")
    print(f"# nonzero latents (reconstructed): {len(nonzero_latents_recon)}")
    print(f"# latents alive in one but not both: {len(alive_in_one_not_both)}")

    # Create initial figure
    fig = px.imshow(
        to_numpy(latent_latent_gradients.T),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        x=[f"F{layer_to}.{latent}, {str_toks[seq]!r} ({seq})" for (_, seq, latent) in latent_acts_next_recon.indices],
        y=[f"F{layer_from}.{latent}, {str_toks[seq]!r} ({seq})" for (_, seq, latent) in latent_acts_prev.indices],
        labels={"y": f"From layer {layer_from}", "x": f"To layer {layer_to}"},
        title=f'Gradients between SAE latents in layer {layer_from} and SAE latents in layer {layer_to}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
        width=1200,
        height=1000,
    )
    # Add rectangles to it, to cover the blocks where the layer 4 & 5 positions correspond to what we
    # expect for the induction circuit
    for first_B_posn in range(2, seq_len + 2):
        second_A_posn = first_B_posn + seq_len - 1
        x0 = (latent_acts_next_recon.indices[:, 1] < second_A_posn).sum().item()
        x1 = (latent_acts_next_recon.indices[:, 1] <= second_A_posn).sum().item()
        y0 = (latent_acts_prev.indices[:, 1] < first_B_posn).sum().item()
        y1 = (latent_acts_prev.indices[:, 1] <= first_B_posn).sum().item()
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1)

    fig.show()

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html,st-dropdown[Click to see the expected output]]

r"""
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"># nonzero features (true): 476
# nonzero features (reconstructed): 492
# features alive in one but not both: 150</pre>

<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13223b.html" width="1220" height="1020"></div>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
Note - your SAEs should perform worse on reconstructing this data than they did on the previous exercises in this subsection (if measured in terms of the intersection of activating features). My guess is that this is because the induction sequences are more OOD for the distribution which the SAEs were trained on (since they're literally random tokens). Also, it's possible that measuring over a larger batch of data and taking all features that activate on at least some fraction of the total tokens would give us a less noisy picture.

Here's some code which filters down the layer-5 features to ones which are active on every token in the second half of the sequence, and also only looks at the layer-4 features which are active on the first half. Try inspecting individual feature pairs which seem to have large gradients with each other - do they seem like previous token features & induction features respectively?
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_1:
    # Filter for layer-5 latents which are active on every token in the second half (which induction
    # latents should be!)
    acts_on_second_half = latent_acts_next_recon.indices[latent_acts_next_recon.indices[:, 1] >= seq_len + 1]
    c = Counter(acts_on_second_half[:, 2].tolist())
    top_feats = sorted([feat for feat, count in c.items() if count >= seq_len])
    print(f"Layer 5 SAE latents which fired on all tokens in the second half: {top_feats}")
    mask_next = (latent_acts_next_recon.indices[:, 2] == t.tensor(top_feats, device=device)[:, None]).any(dim=0) & (
        latent_acts_next_recon.indices[:, 1] >= seq_len + 1
    )

    # Filter the layer-4 axis to only show activations at sequence positions that we expect to be used
    # in induction
    mask_prev = (latent_acts_prev.indices[:, 1] >= 1) & (latent_acts_prev.indices[:, 1] <= seq_len)

    # Filter the y-axis, just to these
    px.imshow(
        to_numpy(latent_latent_gradients[mask_next][:, mask_prev]),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        y=[
            f"{str_toks[seq]!r} ({seq}), #{latent:05}" for (_, seq, latent) in latent_acts_next_recon.indices[mask_next]
        ],
        x=[f"{str_toks[seq]!r} ({seq}), #{latent:05}" for (_, seq, latent) in latent_acts_prev.indices[mask_prev]],
        labels={"x": f"From layer {layer_from}", "y": f"To layer {layer_to}"},
        title=f'Gradients between SAE latents in layer {layer_from} and SAE latents in layer {layer_to}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
        width=1800,
        height=500,
    ).show()

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [st-dropdown[Click to see the expected output]]

r"""
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
Layer 5 SAE latents which fired on all tokens in the second half: [35425, 36126]
</pre>

<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13223c.html" width="1820" height="520"></div>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
<details>
<summary>Some observations</summary>

I didn't rigorously check this (and the code needs a lot of cleaning up!) but I was able to find the 2 most prominent layer-5 features (35425, 36126) were definitely induction features, and 3/4 of the first one which strongly composed with those in layer 4 seemed like previous token features:

```python
for (layer, latent_idx) in [(5, 35425), (5, 36126), (4, 22975), (4, 21020), (4, 23954)]:
    display_dashboard(
        sae_release="gpt2-small-hook-z-kk",
        sae_id=f"blocks.{layer}.hook_z",
        latent_idx=latent_idx,
    )
```

</details>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
# 2️⃣ Transcoders
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Introduction

The MLP-layer SAEs we've looked at attempt to represent activations as a sparse linear combination of latent vectors; importantly, they only operate on activations **at a single point in the model**. They don't actually learn to perform the MLP layer's computation, rather they learn to reconstruct the results of that computation. It's very hard to do any weights-based analysis on MLP layers in superposition using standard SAEs, since many latents are highly dense in the neuron basis, meaning the neurons are hard to decompose.

In contrast, **transcoders** take in the activations before the MLP layer (i.e. the possibly-normalized residual stream values) and aim to represent the post-MLP activations of that MLP layer, again as a sparse linear combination of latent vectors. The transcoder terminology is the most common, although these have also been called **input-output SAEs** (because we take the input to some base model layer, and try to learn the output) and **predicting future activations** (for obvious reasons). Note that transcoders aren't technically autoencoders, because they're learning a mapping rather than a reconstruction - however a lot of our intuitions from SAEs carry over to transcoders.

Why might transcoders be an improvement over standard SAEs? Mainly, they offer a much clearer insight into the function of a model's layers. From the [Transcoders LessWrong post](https://www.lesswrong.com/posts/YmkjnWtZGLbHRbzrP/transcoders-enable-fine-grained-interpretable-circuit):

> One of the strong points of transcoders is that they decompose the function of an MLP layer into sparse, independently-varying, and meaningful units (like neurons were originally intended to be before superposition was discovered). This significantly simplifies circuit analysis.
>
> ...
>
> As an analogy, let’s say that we have some complex compiled computer program that we want to understand (_a la_ [Chris Olah’s analogy](https://transformer-circuits.pub/2022/mech-interp-essay/index.html)). SAEs are analogous to a debugger that lets us set breakpoints at various locations in the program and read out variables. On the other hand, transcoders are analogous to a tool for replacing specific subroutines in this program with human-interpretable approximations.

Intuitively it might seem like transcoders are solving a different (more complicated) kind of optimization problem - trying to mimic the MLP's computation rather than just reproduce output - and so they would suffer a performance tradeoff relative to standard SAEs. However, evidence suggests that this might not be the case, and transcoders might offer a pareto improvement over standard SAEs.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
We'll start by loading in our transcoders. Note that SAELens doesn't yet support running transcoders with `HookedSAETransformer` models (at time of writing**), so instead we'll be loading in our transcoders as `SAE`s but using them in the way we use regular model hooks (rather than using methods like `run_with_cache_with_saes`). The model we'll be working with has been trained to reconstruct the 8th MLP layer of GPT-2. An important note - we're talking about taking the normalized input to the MLP layer and outputting `mlp_out` (i.e. the values we'll be adding back to the residual stream). So when we talk about pre-MLP and post-MLP values, we mean this, not pre/post activation function!

**If this has changed by the time you're reading this, please send an errata in the [Slack group](https://join.slack.com/t/arena-uk/shared_invite/zt-39iwnhbj4-pMWUvZkkt2wpvaxkvJ0q2rRQ)!
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_2:
    gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

    hf_repo_id = "callummcdougall/arena-demos-transcoder"
    sae_id = "gpt2-small-layer-{layer}-mlp-transcoder-folded-b_dec_out"
    gpt2_transcoders = {
        layer: SAE.from_pretrained(release=hf_repo_id, sae_id=sae_id.format(layer=layer), device=str(device))[0]
        for layer in tqdm(range(9))
    }

    layer = 8
    gpt2_transcoder = gpt2_transcoders[layer]
    print("Transcoder hooks (same as regular SAE hooks):", gpt2_transcoder.hook_dict.keys())

    # Load the sparsity values, and plot them
    log_sparsity_path = hf_hub_download(hf_repo_id, f"{sae_id.format(layer=layer)}/log_sparsity.pt")
    log_sparsity = t.load(log_sparsity_path, map_location="cpu", weights_only=True)
    px.histogram(
        to_numpy(log_sparsity), width=800, template="ggplot2", title="Transcoder latent sparsity"
    ).update_layout(showlegend=False).show()
    live_latents = np.arange(len(log_sparsity))[to_numpy(log_sparsity > -4)]

    # Get the activations store
    gpt2_act_store = ActivationsStore.from_sae(
        model=gpt2,
        sae=gpt2_transcoders[layer],
        streaming=True,
        store_batch_size_prompts=16,
        n_batches_in_buffer=32,
        device=str(device),
    )
    tokens = gpt2_act_store.get_batch_tokens()
    assert tokens.shape == (gpt2_act_store.store_batch_size_prompts, gpt2_act_store.context_size)

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html]

r"""
<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13227.html" width="820" height="480"></div>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
Next, we've given you a helper function which essentially works the same way as the `run_with_cache_with_saes` method that you might be used to. We recommend you read through this function, and understand how the transcoder is being used.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def run_with_cache_with_transcoder(
    model: HookedSAETransformer,
    transcoders: list[SAE],
    tokens: Tensor,
    use_error_term: bool = True,  # by default we don't intervene, just compute activations
) -> ActivationCache:
    """
    Runs an MLP transcoder(s) on a batch of tokens. This is quite hacky, and eventually will be
    supported in a much better way by SAELens!
    """
    assert all(transcoder.cfg.hook_name.endswith("ln2.hook_normalized") for transcoder in transcoders)
    input_hook_names = [transcoder.cfg.hook_name for transcoder in transcoders]
    output_hook_names = [
        transcoder.cfg.hook_name.replace("ln2.hook_normalized", "hook_mlp_out") for transcoder in transcoders
    ]

    # Hook function at transcoder input: computes its output (and all intermediate values e.g.
    # latent activations)
    def hook_transcoder_input(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        _, cache = transcoders[transcoder_idx].run_with_cache(activations)
        hook.ctx["cache"] = cache

    # Hook function at transcoder output: replaces activations with transcoder output
    def hook_transcoder_output(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        cache: ActivationCache = model.hook_dict[transcoders[transcoder_idx].cfg.hook_name].ctx["cache"]
        return cache["hook_sae_output"]

    # Get a list of all fwd hooks (only including the output hooks if use_error_term=False)
    fwd_hooks = []
    for i in range(len(transcoders)):
        fwd_hooks.append((input_hook_names[i], partial(hook_transcoder_input, transcoder_idx=i)))
        if not use_error_term:
            fwd_hooks.append((output_hook_names[i], partial(hook_transcoder_output, transcoder_idx=i)))

    # Fwd pass on model, triggering all hook functions
    with model.hooks(fwd_hooks=fwd_hooks):
        _, model_cache = model.run_with_cache(tokens)

    # Return union of both caches (we rename the transcoder hooks using the same convention as
    # regular SAE hooks)
    all_transcoders_cache_dict = {}
    for i, transcoder in enumerate(transcoders):
        transcoder_cache = model.hook_dict[input_hook_names[i]].ctx.pop("cache")
        transcoder_cache_dict = {f"{transcoder.cfg.hook_name}.{k}": v for k, v in transcoder_cache.items()}
        all_transcoders_cache_dict.update(transcoder_cache_dict)

    return ActivationCache(cache_dict=model_cache.cache_dict | all_transcoders_cache_dict, model=model)


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
Lastly, we've given you the functions which you should already have encountered in the earlier exercise sets, when we were replicating SAE dashboards (if you've not done these exercises yet, we strongly recommend them!). The only difference is that we use the `run_with_cache_with_transcoder` function in place of `model.run_with_cache_with_saes`.
"""

# ! CELL TYPE: code
# ! FILTERS: [~py]
# ! TAGS: []


def get_k_largest_indices(
    x: Float[Tensor, "batch seq"], k: int, buffer: int = 0, no_overlap: bool = True
) -> Int[Tensor, "k 2"]:
    if buffer > 0:
        x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer

    if no_overlap:
        unique_indices = t.empty((0, 2), device=x.device).long()
        while len(unique_indices) < k:
            unique_indices = t.cat((unique_indices, t.tensor([[rows[0], cols[0]]], device=x.device)))
            is_overlapping_mask = (rows == rows[0]) & ((cols - cols[0]).abs() <= buffer)
            rows = rows[~is_overlapping_mask]
            cols = cols[~is_overlapping_mask]
        return unique_indices

    return t.stack((rows, cols), dim=1)[:k]


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, " k *buffer_x2_plus1"]:
    rows, cols = indices.unbind(dim=-1)
    if buffer is not None:
        rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
        cols[cols < buffer] = buffer
        cols[cols > x.size(1) - buffer - 1] = x.size(1) - buffer - 1
        cols = einops.repeat(cols, "k -> k buffer", buffer=buffer * 2 + 1) + t.arange(
            -buffer, buffer + 1, device=x.device
        )
    return x[rows, cols]


def display_top_seqs(data: list[tuple[float, list[str], int]]):
    table = Table("Act", "Sequence", title="Max Activating Examples", show_lines=True)
    for act, str_toks, seq_pos in data:
        formatted_seq = (
            "".join([f"[b u green]{str_tok}[/]" if i == seq_pos else str_tok for i, str_tok in enumerate(str_toks)])
            .replace("�", "")
            .replace("\n", "↵")
        )
        table.add_row(f"{act:.3f}", repr(formatted_seq))
    rprint(table)


def fetch_max_activating_examples(
    model: HookedSAETransformer,
    transcoder: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
    k: int = 10,
    buffer: int = 10,
    display: bool = False,
) -> list[tuple[float, list[str], int]]:
    data = []

    for _ in tqdm(range(total_batches)):
        tokens = act_store.get_batch_tokens()
        cache = run_with_cache_with_transcoder(model, [transcoder], tokens)
        acts = cache[f"{transcoder.cfg.hook_name}.hook_sae_acts_post"][..., latent_idx]

        k_largest_indices = get_k_largest_indices(acts, k=k, buffer=buffer)
        tokens_with_buffer = index_with_buffer(tokens, k_largest_indices, buffer=buffer)
        str_toks = [model.to_str_tokens(toks) for toks in tokens_with_buffer]
        top_acts = index_with_buffer(acts, k_largest_indices).tolist()
        data.extend(list(zip(top_acts, str_toks, [buffer] * len(str_toks))))

    data = sorted(data, key=lambda x: x[0], reverse=True)[:k]
    if display:
        display_top_seqs(data)
    return data


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
Let's pick latent 1, and compare our results to the neuronpedia dashboard (note that we do have neuronpedia dashboards for this model, even though it's not in SAELens yet).
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_2:
    latent_idx = 1
    neuronpedia_id = "gpt2-small/8-tres-dc"
    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    display(IFrame(url, width=800, height=600))

    fetch_max_activating_examples(
        gpt2, gpt2_transcoder, gpt2_act_store, latent_idx=latent_idx, total_batches=200, display=True
    )

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html]

r"""
<iframe src="https://neuronpedia.org/gpt2-small/8-tres-dc/1?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300" height=600 width=800></iframe>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">                                              Max Activating Examples                                              </span>
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Act    </span>┃<span style="font-weight: bold"> Sequence                                                                                               </span>┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 13.403 │ ' perception about not only how critical but also how dominant<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goal</span>ies can be in the NHL. '            │
├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 12.093 │ ' of the best. Hundreds of<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goal</span>ies have laced up the skates, put on'                                   │
├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 11.882 │ " 9↵↵December 4↵↵Messi's<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goal</span> breakdown↵↵Barcelona↵↵Copa del"                                          │
├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 11.549 │ 't identify who the six Canadian legend NHL<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goal</span>ies are. We know 5 of them are hall'                   │
├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 11.493 │ ' Steven Gerrard giving the Reds the lead. Second half<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goals</span> from Rafael and a Robin van Persie        │
│        │ penalty won'                                                                                           │
├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 11.280 │ "↵↵Messi's month-by-month<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goal</span> tally↵↵January 7↵↵February 10↵"                                         │
├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 10.874 │ '.↵↵Going from most recent to oldest Canadian<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goal</span>ies to win the Calder. 08-09 saw'                    │
├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 10.811 │ ' the NHL. Each of these<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goal</span>ies left his stamp on the game. Now six'                                  │
├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 10.734 │ '↵"We just need to make sure that one<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goal</span> conceded does not create a ripple effect, which then'       │
├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 10.501 │ ' the win with two sweetly taken second-half<span style="color: #008000; text-decoration-color: #008000; font-weight: bold; text-decoration: underline"> goals</span>.↵↵"There were games last year where'                │
└────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Pullback & de-embeddings

In the exercises on latent-latent gradients at the start of this section, we saw that it could be quite difficult to compute how any 2 latents in different layers interact with each other. In fact, we could only compute gradients between latents which were both alive in the same forward pass. One way we might have liked to deal with this is by just taking the dot product of the "writing vector" of one latent with the "reading vector" of another. For example, suppose our SAEs were trained on post-ReLU MLP activations, then we could compute: `W_dec[:, f1] @ W_out[layer1] @ W_in[layer2] @ W_enc[f2, :]` (where `f1` and `f2` are our earlier and later latent indices, `layer1` and `layer2` are our SAE layers, and `W_in`, `W_out` are the MLP input & output weight matrices for all layers). To make sense of this formula: the term `W_dec[:, f1] @ W_out[layer1]` is the "writing vector" being added to the residual stream by the first (earlier) latent, and we would take the dot product of this with `W_in[layer2] @ W_enc[f2, :]` to compute the activation of the second (later) latent. However, one slight frustration is that we're ignoring the later MLP layer's ReLU function (remember that the SAEs are reconstructing the post-ReLU activations, not pre-ReLU). This might seem like a minor point, but it actually gets to a core part of the limitation of standard SAEs when trained on layers which perform computation - **the SAEs are reconstructing a snapshot in the model, but they're not helping us get insight into the layer's actual computation process**.

How do transcoders help us here? Well, since transcoders sit around the entire MLP layer (nonlinearity and all), we can literally compute the dot product between the "writing vector" and a downstream "reading vector" to figure out whether any given latent causes another one to be activated (ignoring layernorm). To make a few definitions:

- The **pullback** of some later latent is $p = (W_{dec})^T f_{later}$, i.e. the dot product of the later latent vector (reading weight) with all the decoder weights (writing weights) of earlier latents.
- The **de-embedding** is a special case: $d = W_E f_{later}$, i.e. instead of asking "which earlier transcoder latents activate some later latent?" we ask "which tokens maximally activate some later latent?".

Note that we can in principle compute both of these quantities for regular MLP SAEs. But they wouldn't be as accurate to the model's actual computation, and so you couldn't draw as many strong conclusions from them.

To complete our circuit picture of (embeddings -> transcoders -> unembeddings), it's worth noting that we can compute the logit lens for a transcoder latent in exactly the same way as regular SAEs: just take the dot product of the transcoder decoder vector with the unembedding matrix. Since this has basically the exact same justification & interpretation as for regular SAEs, we don't need to invent a new term for it, so we'll just keep calling it the logit lens!
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - compute de-embedding

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 10-15 minutes on this exercise.
> ```

In the cell below, you should compute the de-embedding for this latent (i.e. which tokens cause this latent to fire most strongly). You can use the logit lens function as a guide (which we've provided, from where it was used in the earlier exercises).
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def show_top_logits(
    model: HookedSAETransformer,
    sae: SAE,
    latent_idx: int,
    k: int = 10,
) -> None:
    """Displays the top & bottom logits for a particular latent."""
    logits = sae.W_dec[latent_idx] @ model.W_U

    pos_logits, pos_token_ids = logits.topk(k)
    pos_tokens = model.to_str_tokens(pos_token_ids)
    neg_logits, neg_token_ids = logits.topk(k, largest=False)
    neg_tokens = model.to_str_tokens(neg_token_ids)

    print(
        tabulate(
            zip(map(repr, neg_tokens), neg_logits, map(repr, pos_tokens), pos_logits),
            headers=["Bottom tokens", "Value", "Top tokens", "Value"],
            tablefmt="simple_outline",
            stralign="right",
            numalign="left",
            floatfmt="+.3f",
        )
    )


# HIDE
if MAIN and FLAG_RUN_SECTION_2:
    print(f"Top logits for transcoder latent {latent_idx}:")
    show_top_logits(gpt2, gpt2_transcoder, latent_idx=latent_idx)
# END HIDE


def show_top_deembeddings(model: HookedSAETransformer, sae: SAE, latent_idx: int, k: int = 10) -> None:
    """Displays the top & bottom de-embeddings for a particular latent."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    de_embeddings = model.W_E @ sae.W_enc[:, latent_idx]

    pos_logits, pos_token_ids = de_embeddings.topk(k)
    pos_tokens = model.to_str_tokens(pos_token_ids)
    neg_logits, neg_token_ids = de_embeddings.topk(k, largest=False)
    neg_tokens = model.to_str_tokens(neg_token_ids)

    print(
        tabulate(
            zip(map(repr, neg_tokens), neg_logits, map(repr, pos_tokens), pos_logits),
            headers=["Bottom tokens", "Value", "Top tokens", "Value"],
            tablefmt="simple_outline",
            stralign="right",
            numalign="left",
            floatfmt="+.3f",
        )
    )
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_2:
    print(f"\nTop de-embeddings for transcoder latent {latent_idx}:")
    show_top_deembeddings(gpt2, gpt2_transcoder, latent_idx=latent_idx)
    tests.test_show_top_deembeddings(show_top_deembeddings, gpt2, gpt2_transcoder)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html]

r"""
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Top logits for transcoder latent 1:
┌─────────────────┬─────────┬──────────────┬─────────┐
│   Bottom tokens │ Value   │   Top tokens │ Value   │
├─────────────────┼─────────┼──────────────┼─────────┤
│         ' Goal' │ -0.812  │    'keeping' │ +0.693  │
│           'nox' │ -0.638  │       'bred' │ +0.690  │
│       'ussions' │ -0.633  │     'urious' │ +0.663  │
│       ' Vision' │ -0.630  │       'cake' │ +0.660  │
│         'heses' │ -0.623  │      'swick' │ +0.651  │
│         'iasco' │ -0.619  │     'hedral' │ +0.647  │
│        ' dream' │ -0.605  │         'sy' │ +0.622  │
│      ' Grenade' │ -0.594  │      'ascus' │ +0.612  │
│        'rament' │ -0.586  │      'ebted' │ +0.611  │
│       ' imagin' │ -0.575  │         'ZE' │ +0.610  │
└─────────────────┴─────────┴──────────────┴─────────┘

Top de-embeddings for transcoder latent 1:
┌─────────────────┬─────────┬──────────────┬─────────┐
│   Bottom tokens │ Value   │   Top tokens │ Value   │
├─────────────────┼─────────┼──────────────┼─────────┤
│          'attr' │ -0.775  │       'liga' │ +1.720  │
│     ' reciproc' │ -0.752  │       'GAME' │ +1.695  │
│          'oros' │ -0.712  │        'jee' │ +1.676  │
│      ' resists' │ -0.704  │    ' scorer' │ +1.649  │
│       ' Advent' │ -0.666  │     'ickets' │ +1.622  │
│         'gling' │ -0.646  │    ' scored' │ +1.584  │
│       ' Barron' │ -0.630  │  'artifacts' │ +1.580  │
│          ' coh' │ -0.593  │    'scoring' │ +1.578  │
│         ' repr' │ -0.592  │      'itory' │ +1.520  │
│      ' reprint' │ -0.587  │   ' scoring' │ +1.520  │
└─────────────────┴─────────┴──────────────┴─────────┘
</pre>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
This is ... pretty underwhelming! It seems very obvious that the top activating token should be `" goal"` from looking at the dashboard - why are we getting weird words like `"liga"` and `"jee"`? Obviously some words make sense like `" scored"` or `" scoring"`, but overall this isn't what we would expect.

Can you guess what's happening here? (Try and think about it before reading on, since reading the description of the next exercise will give away the answer!

<details>
<summary>Hint</summary>

If you've done the IOI ARENA exercises (or read the IOI paper), you'll have come across this idea. It has to do with the architecture of GPT2-Small.

</details>

<details>
<summary>Answer</summary>

GPT2-Small has **tied embeddings**, i.e. its embedding matrix is the transpose of its unembedding matrix. This means the direct path is unable to represent bigram frequencies (e.g. it couldn't have higher logits for the bigram `Barack Obama` than for `Obama Barack`), so the MLP layers have to step in and break the symmetry. In particular MLP0 seems to do this, which is why we call it the **extended embedding** (or the **effective embedding**).

The result of this is that the indexed rows of the embedding matrix aren't really a good representation of the thing that the model has actually learned to treat as the embedding of a given token.

</details>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - correct the de-embedding function

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 10-20 minutes on this exercise.
> ```

You should fill in the function below to compute the extended embedding, which will allow us to correct the mistake in the function discussed in the dropdowns above.

There are many different ways to compute the extended embedding (e.g. sometimes we include the attention layer and assume it always self-attends, sometimes we only use MLP0's output and sometimes we add it to the raw embeddings, sometimes we use a BOS token to make it more accurate). Most of these methods will get similar quality of results (it's far more important that you include MLP0 than the exact details of how you include it). For the sake of testing though, you should use the following method:

- Take the embedding matrix,
- Apply layernorm to it (i.e. each token's embedding vector is scaled to have unit std dev),
- Apply MLP0 to it (i.e. to each token's normalized embedding vector separately),
- Add the result back to the original embedding matrix.

Tip - rather than writing out the individual operations for layernorm & MLPs, you can just use the forward methods of `model.blocks[layer].ln2` or `.mlp` respectively.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def create_extended_embedding(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:
    """
    Creates the extended embedding matrix using the model's layer-0 MLP, and the method described
    in the exercise above.

    You should also divide the output by its standard deviation across the `d_model` dimension
    (this is because that's how it'll be used later e.g. when fed into the MLP layer / transcoder).
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    W_E = model.W_E.clone()[:, None, :]  # shape [batch=d_vocab, seq_len=1, d_model]

    mlp_output = model.blocks[0].mlp(model.blocks[0].ln2(W_E))  # shape [batch=d_vocab, seq_len=1, d_model]

    W_E_ext = (W_E + mlp_output).squeeze()
    return (W_E_ext - W_E_ext.mean(dim=-1, keepdim=True)) / W_E_ext.std(dim=-1, keepdim=True)
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_2:
    tests.test_create_extended_embedding(create_extended_embedding, gpt2)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
Once you've passed those tests, try rewriting `show_top_deembeddings` to use the extended embedding. Do the results look better? (Hint - they should!)

Note - don't worry if the magnitude of the results seems surprisingly large. Remember that a normalization step is applied pre-MLP, so the actual activations will be smaller than is suggested by the values in the table you'll generate.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def show_top_deembeddings_extended(model: HookedSAETransformer, sae: SAE, latent_idx: int, k: int = 10) -> None:
    """Displays the top & bottom de-embeddings for a particular latent."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    de_embeddings = create_extended_embedding(model) @ sae.W_enc[:, latent_idx]

    pos_logits, pos_token_ids = de_embeddings.topk(k)
    pos_tokens = model.to_str_tokens(pos_token_ids)
    neg_logits, neg_token_ids = de_embeddings.topk(k, largest=False)
    neg_tokens = model.to_str_tokens(neg_token_ids)

    print(
        tabulate(
            zip(map(repr, neg_tokens), neg_logits, map(repr, pos_tokens), pos_logits),
            headers=["Bottom tokens", "Value", "Top tokens", "Value"],
            tablefmt="simple_outline",
            stralign="right",
            numalign="left",
            floatfmt="+.3f",
        )
    )
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_2:
    print(f"Top de-embeddings (extended) for transcoder latent {latent_idx}:")
    show_top_deembeddings_extended(gpt2, gpt2_transcoder, latent_idx=latent_idx)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html,st-dropdown[Click to see the expected output]]

r"""
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Top de-embeddings (extended) for transcoder latent 1:
┌─────────────────┬─────────┬──────────────┬─────────┐
│   Bottom tokens │ Value   │   Top tokens │ Value   │
├─────────────────┼─────────┼──────────────┼─────────┤
│      ' coupled' │ -8.747  │       'goal' │ +14.161 │
│         'inski' │ -7.633  │      ' Goal' │ +13.004 │
│         ' bent' │ -7.601  │      ' goal' │ +12.510 │
│         ' Line' │ -7.357  │       'Goal' │ +11.724 │
│        ' Layer' │ -7.235  │     ' Goals' │ +11.538 │
│      ' layered' │ -7.225  │     ' goals' │ +11.447 │
│        ' lined' │ -7.206  │     ' goalt' │ +10.378 │
│           'avy' │ -7.110  │      'score' │ +10.364 │
│      ' Cassidy' │ -7.032  │    ' Soccer' │ +10.162 │
│        'Contin' │ -7.006  │      ' puck' │ +10.122 │
└─────────────────┴─────────┴──────────────┴─────────┘</pre>
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - perform a blind case study

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵🔵⚪
> 
> This is a bonus-flavoured exercise, which is designed to be extremely challenging.
> It's a great way to build your research and exploration skills, putting all that you've learned into practice!
> ```

The authors of the [post introducing transcoders](https://www.lesswrong.com/posts/YmkjnWtZGLbHRbzrP/transcoders-enable-fine-grained-interpretable-circuit) present the idea of a **blind case study**. To quote from their post:

> ...we have some latent in some transcoder, and we want to interpret this transcoder latent without looking at the examples that cause it to activate. Our goal is to instead come to a hypothesis for when the latent activates by solely using the input-independent and input-dependent circuit analysis methods described above.

By **input-independent circuit analysis**, they mean things like pullbacks and de-embeddings (i.e. things which are a function of just the model & transcoder's weights). By **input-dependent**, they specifically mean the **input-dependent influence**, which they define to be the elementwise product of the pullback to some earlier transcoder and the post-ReLU activations of that earlier transcoder. In other words, it tells you not just which earlier latents would affect some later latent when those earlier latents fire, but which ones *do* affect the later latent on some particular input (i.e. taking into account which ones actually fired).

What's the motivation for this? Well, eventually we want to be able to understand latents when they appear in complex circuits, not just as individual units which respond to specific latents in the data. And part of that should involve being able to build up hypotheses about what a given latent is doing based on only its connection to other latents (or to specific tokens in the input). Just looking directly at the top activating examples can definitely be helpful, but not only is it sometimes [misleading](https://www.lesswrong.com/posts/3zBsxeZzd3cvuueMJ/paper-a-is-for-absorption-studying-latent-splitting-and), it also can only tell you *what* a latent is doing, without giving much insight into *why*.

To be clear on the rules:

- You can't look at activations of a latent on specific tokens in specific example prompts.
- You can use input-dependent analysis e.g. the influence of some earlier latents on your target latent on some particular input (however you have to keep the input in terms of token IDs not tokens, because it's cheating to look at the actual content of prompts which activate any of your latents).
- You can use input-indepenent analysis e.g. a latent's de-embeddings or logit lens.

We're making this a very open-ended exercise - we've written some functions for you above, but others you might have to write yourself, depending on what seems most useful for your analysis (e.g. we've not given you a function to compute pullback yet). If you want an easier exercise then you can use a latent which the post successfully reverse-engineered (e.g. latent 355, the 300th live latent in the transcoder), but for a challenge you can also try latent 479 (the 400th live latent in the transcoder,which the authors weren't able to reverse-engineer in their initial post).

If you want a slightly easier version of the game, you can try a rule relaxation where you're allowed to pass your own sequences into the model to test hypotheses (you just can't do something like find the top activating sequences over a large dataset and decode them). This allows you to test your hypotheses in ways that still impose some restrictions on your action space.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_2:
    blind_study_latent = 479

    layer = 8
    gpt2_transcoder = gpt2_transcoders[layer]

# YOUR CODE HERE!

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
You can click on the dropdown below to see my attempt at this exercise, or read [this notebook](https://github.com/jacobdunefsky/transcoder_circuits/blob/master/case_study_local_context.ipynb) which shows the authors' walkthrough blind case study interpretation of this latent. Don't visit the notebook until you've given the exercise a good try though, since the title will give part of the problem away!

<details>
<summary>My attempt</summary>

```python
# Plan:
# (1) look at de-embedding & logit lens, make some guesses (if the data seems to have as strong implications as it did for our "goal" example earlier)
# (2) look at the top influences from earlier latents, which don't go through any attention heads (i.e. just direct), see if we can understand those earlier latents using their de-embeddings
# (3) look at influence coming from paths that map through at least one attention head, to see if we can understand those
# (4) get average attribution for each component, to understand which are more important for this latent


# (1) look at de-embedding
print("De-embeddings:")
show_top_deembeddings_extended(gpt2, gpt2_transcoder, latent_idx=blind_study_latent)
print("Logit lens:")
show_top_logits(gpt2, gpt2_transcoder, latent_idx=blind_study_latent)

# Results?
# - de-embedding has quite a few words related to finance or accumulation, e.g. " deficits", " output", " amounts", " amassed" (also "imately" could be the second half of "approximately")
#   - but definitely not as strong evidence as we got for "goal" earlier
# - logit lens shows us this latent firing will boost words like ' costing' and ' estimated'
#   - possible theory: it fires on phrases like "...fines <<costing>>..." or "...amassed <<upwards>> of..."
#   - prediction based on theory: we should see earlier latents firing on money-related words, and being attended to
#   - e.g. "the bank had <<amassed>> upwards of $100m$": maybe "amassed" attends to "bank"


# (2) look at influence from earlier latents

# Gather 20 top activating sequences for the target latent
total_batches = 500
k = 20
buffer = 10
data = []  # list of (seq_pos: int, tokens: list[int], top_act: float)
for _ in tqdm(range(total_batches)):
    tokens = gpt2_act_store.get_batch_tokens()
    cache = run_with_cache_with_transcoder(gpt2, [gpt2_transcoder], tokens, use_error_term=True)
    acts = cache[f"{gpt2_transcoder.cfg.hook_name}.hook_sae_acts_post"][..., blind_study_latent]
    k_largest_indices = get_k_largest_indices(acts, k=k, buffer=buffer)  # [k, 2]
    tokens_in_top_sequences = tokens[k_largest_indices[:, 0]]  # [k, seq_len]
    top_acts = index_with_buffer(acts, k_largest_indices)  # [k,]
    data.extend(list(zip(k_largest_indices[:, 1].tolist(), tokens_in_top_sequences.tolist(), top_acts.tolist())))

data = sorted(data, key=lambda x: x[2], reverse=True)[:k]
tokens = t.tensor([x[1] for x in data])  # each row is a full sequence, containing one of the max activating tokens
top_seqpos = [x[0] for x in data]  # list of sequence positions of the max activating tokens
acts = [x[2] for x in data]  # list of max activating values

# Compute pullback from earlier latents to target latent, then compute influence for these top activating sequences
cache = run_with_cache_with_transcoder(gpt2, list(gpt2_transcoders.values()), tokens, use_error_term=True)
t.cuda.empty_cache()
all_influences = []
for _layer in range(layer):
    acts = cache[f"{gpt2_transcoders[_layer].cfg.hook_name}.hook_sae_acts_post"]  # shape [k=20, seq_len=128, d_sae=24k]
    acts_at_top_posn = acts[range(k), top_seqpos]  # shape [k=20, d_sae=24k]
    pullback = gpt2_transcoders[_layer].W_dec @ gpt2_transcoder.W_enc[:, blind_study_latent]  # shape [d_sae]
    influence = acts_at_top_posn * pullback  # shape [k=20, d_sae=24k]
    all_influences.append(influence)

# Find the earlier latents which are consistently in the top 10 for influence on target latent, and inspect their de-embeddings
all_influences = t.cat(all_influences, dim=-1)  # shape [k, n_layers*d_sae]
top_latents = all_influences.topk(k=10, dim=-1).indices.flatten()  # shape [k*10]
top_latents_as_tuples = [(i // gpt2_transcoder.cfg.d_sae, i % gpt2_transcoder.cfg.d_sae) for i in top_latents.tolist()]
top5_latents_as_tuples = sorted(Counter(top_latents_as_tuples).items(), key=lambda x: x[1], reverse=True)[:5]
print(
    tabulate(
        top5_latents_as_tuples,
        headers=["Latent", "Count"],
        tablefmt="simple_outline",
    )
)
for (_layer, _idx), count in top5_latents_as_tuples:
    print(f"Latent {_layer}.{_idx} was in the top 5 for {count}/{k} of the top-activating seqs. Top de-embeddings:")
    show_top_deembeddings_extended(gpt2, gpt2_transcoders[_layer], latent_idx=_idx)

# Results?
# - 7.13166 is very interesting: it's in the top way more than any other latent (17/20 vs 10/20 for the second best), and it boosts quantifiers like " approximately", " exceeding", " EQ", " ≥"
# - Since this is the direct path, possibly we'll find our target latent fires on these kinds of words too? Would make sense given its logit lens results
# - Also more generally, the words we're getting as top de-embeddings in these latents all appear in similar contexts, but they're not similar (i.e. substitutable) words, which makes this less likely to be a token-level latent


# (3) look at influence coming from attention heads (i.e. embedding -> earlier transcoders -> attention -> target transcoder latent)

# The method here is a bit complicated. We do the following, for each head:
# - (A) Map the target latent's "reading vector" backwards through the attention head, to get a "source token reading vector" (i.e. the vector we'd dot product with the residual stream at the source token to get the latent activation for our target latent at the destination token)
# - (B) For all earlier transcoders, compute their "weighted source token writing vector" (i.e. the vector which they write to the residual stream at each source token, weighted by attention from target position to source position)
# - (C) Take the dot product of these, and find the top early latents for this particular head
top_latents_as_tuples = []
for attn_layer in range(layer + 1):  # we want to include target layer, because attn comes before MLP
    for attn_head in range(gpt2.cfg.n_heads):
        for early_transcoder_layer in range(attn_layer):  # we don't include target layer, because attn comes before MLP
            # Get names
            pattern_name = utils.get_act_name("pattern", attn_layer)
            transcoder_acts_name = f"{gpt2_transcoders[early_transcoder_layer].cfg.hook_name}.hook_sae_acts_post"

            # (A)
            reading_vector = gpt2_transcoder.W_enc[:, blind_study_latent]  # shape [d_model]
            reading_vector_src = einops.einsum(
                reading_vector,
                gpt2.W_O[attn_layer, attn_head],
                gpt2.W_V[attn_layer, attn_head],
                "d_model_out, d_head d_model_out, d_model_in d_head -> d_model_in",
            )

            # (B)
            writing_vectors = gpt2_transcoders[early_transcoder_layer].W_dec  # shape [d_sae, d_model]
            patterns = cache[pattern_name][range(k), attn_head, top_seqpos]  # shape [k, seq_K]
            early_transcoder_acts = cache[transcoder_acts_name]  # shape [k, seq_K, d_sae]
            pattern_weighted_acts = einops.einsum(patterns, early_transcoder_acts, "k seq_K, k seq_K d_sae -> d_sae")
            # pattern_weighted_acts = (patterns[..., None] * early_transcoder_acts).mean(0).mean(0) # shape [k, d_sae]
            weighted_src_token_writing_vectors = einops.einsum(
                pattern_weighted_acts, writing_vectors, "d_sae, d_sae d_model -> d_sae d_model"
            )

            # (C)
            influences = weighted_src_token_writing_vectors @ reading_vector_src  # shape [d_sae]
            top_latents_as_tuples.extend(
                [
                    {
                        "early_latent": repr(f"{early_transcoder_layer}.{idx.item():05d}"),
                        "attn_head": (attn_layer, attn_head),
                        "influence": value.item(),
                    }
                    # (early_transcoder_layer, attn_layer, attn_head, idx.item(), value.item())
                    for value, idx in zip(*influences.topk(k=10, dim=-1))
                ]
            )

top20_latents_as_tuples = sorted(top_latents_as_tuples, key=lambda x: x["influence"], reverse=True)[:20]
print(
    tabulate(
        [v.values() for v in top20_latents_as_tuples],
        headers=["Early latent", "Attention head", "Influence"],
        tablefmt="simple_outline",
    )
)

# Results?
# - Attribution from layer 7 transcoder:
#   - 2 latents fire in layer 7, and boost our target latent via head L8H5
#   - I'll inspect both of these (prediction = as described above, these latents' de-embeddings will be financial words)
# - Attribution from earlier transcoders:
#   - There are a few transcoder latents in layers 0, 1, 2 which have influence mediated through L7 attention heads (mostly L7H3 and L7H4)
#   - I'll check out both of them, but I'll also check out the de-embedding mapped directly through these heads (ignoring earlier transcoders), because I suspect these early transcoder latents might just be the extended embedding in disguise


def show_top_deembeddings_extended_via_attention_head(
    model: HookedSAETransformer,
    sae: SAE,
    latent_idx: int,
    attn_head: tuple[int, int] | None = None,
    k: int = 10,
    use_extended: bool = True,
) -> None:
    """
    Displays the top k de-embeddings for a particular latent, optionally after that token's embedding is mapped through
    some attention head.
    """
    t.cuda.empty_cache()
    W_E_ext = create_extended_embedding(model) if use_extended else (model.W_E / model.W_E.std(dim=-1, keepdim=True))

    if attn_head is not None:
        W_V = model.W_V[*attn_head]
        W_O = model.W_O[*attn_head]
        W_E_ext = (W_E_ext @ W_V) @ W_O
        W_E_ext = (W_E_ext - W_E_ext.mean(dim=-1, keepdim=True)) / W_E_ext.std(dim=-1, keepdim=True)

    de_embeddings = W_E_ext @ sae.W_enc[:, latent_idx]

    pos_logits, pos_token_ids = de_embeddings.topk(k)
    pos_tokens = model.to_str_tokens(pos_token_ids)

    print(
        tabulate(
            zip(map(repr, pos_tokens), pos_logits),
            headers=["Top tokens", "Value"],
            tablefmt="simple_outline",
            stralign="right",
            numalign="left",
            floatfmt="+.3f",
        )
    )


print("Layer 7 transcoder latents (these influence the target latent via L8H5):")
for _layer, _idx in [(7, 3373), (7, 14110), (7, 10719), (7, 8696)]:
    print(f"{_layer}.{_idx} de-embeddings:")
    show_top_deembeddings_extended_via_attention_head(gpt2, gpt2_transcoders[_layer], latent_idx=_idx)

print("\n" * 3 + "Layer 1-2 transcoder latents (these influence the target latent via L7H3 and L7H4):")
for _layer, _idx in [(2, 21691), (1, 14997)]:
    print(f"{_layer}.{_idx} de-embeddings:")
    show_top_deembeddings_extended_via_attention_head(gpt2, gpt2_transcoders[_layer], latent_idx=_idx)

print("\n" * 3 + "De-embeddings of target latent via L7H3 and L7H4:")
for attn_layer, attn_head in [(7, 3), (7, 4)]:
    print(f"L{attn_layer}H{attn_head} de-embeddings:")
    show_top_deembeddings_extended_via_attention_head(
        gpt2,
        gpt2_transcoder,
        latent_idx=blind_study_latent,
        attn_head=(attn_layer, attn_head),
    )

# Results?
# - Layer 7 transcoder latents:
#   - 14110 & 8696 both seem to fire on financial words, e.g. " revenues" is top word for both and they also both include " GDP" in their top 10
#       - They also both fire on words like "deaths" and "fatalities", which also makes sense given my hypothesis (e.g. this could be sentences like "the number fatalities* is approximately** totalling***" (where * = src token where the layer 7 latent fires, ** = word predicted by target latent)
#   - 10719 very specifically fires on the word "estimated" (or variants), which also makes sense: these kinds of sentences can often have the word "estimated" in them (e.g. "the estimated number of fatalities is 1000")
#   - 3373 fires on "effectively", "constitutes" and "amounted", which are also likely to appear in sentences like this one (recall we've not looked at where attn is coming from - this could be self-attention!)
# - Earlier transcoder latents:
#   - Disappointingly, these don't seem very interpretable (nor when I just look at direct contributions from the attention heads which are meant to be mediating their influence)


# (4) Final experiment: component-level attribution

# For all these top examples, I want to tally up the contributions from each component (past MLP layers, attention heads, and direct path) and compare them
# This gives me a qualitative sense of which ones matter more

latent_dir = gpt2_transcoder.W_enc[:, blind_study_latent]  # shape [d_model,]

embedding_attribution = cache["embed"][range(k), top_seqpos].mean(0) @ latent_dir

attn_attribution = (
    t.stack(
        [
            einops.einsum(
                cache["z", _layer][range(k), top_seqpos].mean(0),
                gpt2.W_O[_layer],
                "head d_head, head d_head d_model -> head d_model",
            )
            for _layer in range(layer + 1)
        ]
    )
    @ latent_dir
)  # shape [layer+1, n_heads]

mlp_attribution = (
    t.stack([cache["mlp_out", _layer][range(k), top_seqpos].mean(0) for _layer in range(layer)]) @ latent_dir
)

all_attributions = t.zeros((layer + 2, gpt2.cfg.n_heads + 1))
all_attributions[0, 0] = embedding_attribution
all_attributions[1:, :-1] = attn_attribution
all_attributions[1:-1, -1] = mlp_attribution

df = pd.DataFrame(utils.to_numpy(all_attributions))

text = [["W_E", *["" for _ in range(gpt2.cfg.n_heads)]]]
for _layer in range(layer + 1):
    text.append(
        [f"L{_layer}H{_head}" for _head in range(gpt2.cfg.n_heads)] + [f"MLP{_layer}" if _layer < layer else ""]
    )

fig = px.imshow(
    df,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    width=700,
    height=600,
    title="Attribution from different components",
)
fig.data[0].update(text=text, texttemplate="%{text}", textfont={"size": 12})
fig.show()

# Results?
# - Way less impact from W_E than I expected, and even MLP0 (extended embedding) had a pretty small impact, this is evidence away from it being a token-level latent
# - Biggest attributions are from L8H5 and MLP7
#   - L8H5 is the one that attends back to (A) tokens with financial/fatalities context, (B) the word "estimated" and its variants, and (C) other related quantifiers like "effectively" or "amounted"
#   - MLP7 was seen to contain many latents that fired on words which would appear in sentences related to financial estimations (see (2), where we looked at the top 5 contributing latents - they were all in layer 7)
# - Also, the not-very-interpretable results from attention heads 7.3 and 7.4 matter less now, because we can see from this that they aren't very important (although I don't know why they turned up so high before)


# Based on all evidence, this is my final theory:

# - The latent activates primarily on sentences involving estimates of finanical quantities (or casualties)
# - For example I expect top activating seqs like:
#     - "The number of fatalities is **approximately** totalling..."
#     - "The bank had **amassed** upwards of $100m..."
#     - "The GDP of the UK **exceeds** $300bn..."
#     - "This tech company is estimated to be **roughly** worth..."
#    where I've highlighted what I guess to be the top activating token, but the surrounding cluster should also be activating
# - Concretely, what causes it to fire? Most important things (in order) are:
#     - (1) Attention head 8.5, which attends back to the output of layer 7 transcoder latents that fire on words which imply we're in sentences discussing financial quantities or fatality estimates (e.g. "fatalities", "bank", "GDP" and "company" in the examples above). Also this head strongly attends back to a layer 7 latent which detects the word "estimated" and its variants, so I expect very strong activations to start after this word appears in a sentence
#     - (2) Layer-7 transcoder latents (directly), for example latent 7.13166 fires on the token "≤" and causes our target latent to fire
#     - (3) Direct path: the latent should fire strongest on words like **approximately** which rank highly in its de-embedding

# Let's display the latent dashboard for both the target latent and the other latents involved in this theory, and see if the theory is correct:

neuronpedia_id = "gpt2-small/8-tres-dc"
url = f"https://neuronpedia.org/{neuronpedia_id}/{blind_study_latent}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
display(IFrame(url, width=800, height=600))

# Conclusions?
# - Mostly correct:
#   - The top activating sequences are mostly financial estimates
#   - Activations are very large after the word "estimated" (most of the top examples are sentences containing this word)
#   - The latent doesn't seem to be token-level; it fires on a cluster of adjacent words
# - Some areas where the hypothesis was incorrect, or lacking:
#   - I didn't give a hypothesis for when the activations would stop - it seems they stop exactly at the estimated value, and I don't think I would have been able to predict that based on the experiments I ran
#       - Relatedly, I wouldn't have predicted activations staying high even on small connecting words before the estimated value (e.g. "of" in "monthly rent of...", or "as" in "as much as...")
#   - I overestimated the importance of the current word in the sentence (or more generally, I had too rigid a hypothesis for what pattern of sentences would this latent activate on & where it would activate)
#   - I thought there would be more casualty estimates in the top activating sequences, but there weren't. Subsequent testing (see code below) shows that it does indeed fire strongly on non-financial estimates with the right sentence structure, and fatalities fires stronger than the other 2 non-financial example sentences, but the difference is small, so I think this was still an overestimation in my hypothesis)

prompts = {
    "fatalities": """Body counts are a crude measure of the war's impact and more reliable estimates will take time to compile. Since war broke out in the Gaza Strip almost a year ago, the official number of Palestinians killed is estimated to exceed 41,000.""",
    "emissions": """Environmental measurements are an imperfect gauge of climate change impact and more comprehensive studies will take time to finalize. Since the implementation of new global emissions policies almost a year ago, the reduction in global carbon dioxide emissions is estimated to exceed million metric tons.""",
    "visitors": """Visitor counts are a simplistic measure of a national park's popularity and more nuanced analyses will take time to develop. Since the implementation of the new trail system almost a year ago, the number of unique bird species spotted in Yellowstone National Park is estimated to have increased by 47.""",
}
acts_dict = {}
for name, prompt in prompts.items():
    str_tokens = [f"{tok} ({i})" for i, tok in enumerate(gpt2.to_str_tokens(prompt))]
    cache = run_with_cache_with_transcoder(gpt2, [gpt2_transcoder], prompt)
    acts = cache[f"{gpt2_transcoder.cfg.hook_name}.hook_sae_acts_post"][0, :, blind_study_latent]
    acts_dict[name] = utils.to_numpy(acts).tolist()

min_length = min([len(x) for x in acts_dict.values()])
acts_dict = {k: v[-min_length:] for k, v in acts_dict.items()}

df = pd.DataFrame(acts_dict)
px.line(df, y=prompts.keys(), height=500, width=800).show()
```

</details>
'''

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
Note, in order to run these exercises, you'll need to clone the circuit tracer library inside `exercises`:

```bash
cd chapter1_transformer_interp/exercises

git clone https://github.com/decoderesearch/circuit-tracer.git
```

Then run this cell to verify it's been added to your path correctly:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

circuit_tracer_path = exercises_dir / "circuit-tracer"
assert circuit_tracer_path.exists()
sys.path.insert(0, str(circuit_tracer_path.resolve()))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Introduction


In section 1️⃣, you computed gradients between SAE latents in different layers using `jacrev`, building a picture of how individual latents communicate. That approach is exact but expensive: computing the full Jacobian between two layers for thousands of latents doesn't scale. In section 2️⃣, you learned about **transcoders**, which decompose an MLP layer's computation into sparse, interpretable units.

**Attribution graphs** combine these ideas into a scalable framework for understanding end-to-end computation. The key insight is the **local replacement model**: if we freeze all non-linear components (attention patterns, LayerNorm scales, MLP activation functions) at their values from a clean forward pass, the residual stream becomes a **linear function** of the transcoder feature activations. Under this linearisation, the direct effect of any feature on any other feature is just the contraction of their decoder/encoder directions through the frozen linear layers, and we can compute these contractions efficiently with batched backward passes.

The algorithm has five phases:

1. **Setup**: Run a forward pass to find all active transcoder features and record their encoder/decoder directions.
2. **Forward pass**: Run the replacement model with hooks that freeze non-linearities.
3. **Logit attribution**: For each output logit node, inject the (demeaned) unembedding vector as a gradient seed and read off the direct effects from all features.
4. **Feature attribution**: For each active feature, inject the encoder vector as a gradient seed and read off direct effects from all upstream features.
5. **Graph assembly**: Package everything into an adjacency matrix and prune it to the most influential subgraph.

Reference: [Attribution Graphs (Anthropic, 2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### A note on cross-layer transcoders

In the exercises below, we use **single-layer (per-layer) transcoders** for continuity with section 2️⃣. However, the circuit-tracer library also supports **cross-layer transcoders (CLTs)**, which are the architecture used in much of Anthropic's recent work. In a CLT, each feature can write its decoder vector to multiple downstream layers rather than just one. From the circuit tracing tutorial:

> A CLT is a single transcoder for all layers, where features can contribute to multiple downstream layers. CLTs have some practical benefits: they subsume skip connections... and features can be more naturally shared across layers.

If you're interested in CLTs, there is a bonus exercise at the end of section 4️⃣ where you can load a CLT model and compare its graphs against the per-layer transcoder version.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## The Replacement Model

We'll switch from GPT-2 (used in sections 1️⃣ and 2️⃣) to **Gemma 2-2B** with GemmaScope single-layer transcoders. The `circuit-tracer` library provides a `ReplacementModel` class that handles all the plumbing: replacing MLP layers with transcoders, installing hooks that freeze non-linearities for gradient flow, and providing attribution utilities.

Note: this is a significant download (~5GB for the model in half precision, plus transcoders). If you haven't already, make sure you have the `circuit-tracer` library available (see setup instructions at the top of this notebook).
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_3:
    # Clean up GPT-2 and its SAEs/transcoders from sections 1 and 2
    try:
        del gpt2
        del gpt2_saes
        del gpt2_transcoders
        del gpt2_transcoder
        del gpt2_act_store
    except NameError:
        pass

    try:
        del attn_saes
    except NameError:
        pass

    THRESHOLD = 0.1  # GB
    for obj in gc.get_objects():
        try:
            if isinstance(obj, t.nn.Module) and utils.get_tensors_size(obj) / 1024**3 > THRESHOLD:
                if hasattr(obj, "cpu"):
                    obj.cpu()
                if hasattr(obj, "reset"):
                    obj.reset()
        except:
            pass

    t.cuda.empty_cache()
    gc.collect()

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


from circuit_tracer import ReplacementModel

if MAIN and FLAG_RUN_SECTION_3:
    replacement_model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        dtype=t.bfloat16,
        backend="transformerlens",
    )

    # Quick sanity check: the model should predict "Austin" for this prompt
    prompt = "The capital of the state containing Dallas is"
    tokens = replacement_model.ensure_tokenized(prompt)
    logits = replacement_model(tokens.unsqueeze(0))
    top_token = replacement_model.tokenizer.decode(logits[0, -1].argmax())
    print(f"Prompt: {prompt!r}")
    print(f"Top prediction: {top_token!r}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exploring the Replacement Model

The `ReplacementModel` is a `HookedTransformer` with several modifications that create the "local replacement model". Let's explore what these modifications are and verify they work as expected.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - verify gradient freezing

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

The replacement model freezes gradients through three types of non-linear components:

1. **Attention patterns** (`hook_pattern`): the softmax over attention scores is frozen, so gradients don't flow through the query-key dot product.
2. **LayerNorm scales** (`hook_scale`): the normalization denominator is frozen, so gradients only flow through the linear part of LayerNorm.
3. **MLP non-linearities**: the skip connection trick detaches the non-linear part, so gradients flow through the residual stream but not through the MLP's activation function.

Fill in the function below to verify that these freezes are working. You should run a forward+backward pass on the replacement model and check that certain gradients are zero where we expect them to be.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def verify_gradient_freezing(model: ReplacementModel, input_ids: t.Tensor) -> dict[str, bool]:
    """Verify that the replacement model correctly freezes gradients through non-linear components.

    Returns a dict mapping component names to whether gradients are correctly frozen (True = frozen).
    """
    results = {}

    # We need to enable gradients for this test
    input_ids_batch = input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids

    # EXERCISE
    # # YOUR CODE HERE - run a forward pass, compute a scalar loss, backward pass,
    # # then check that gradients through attention patterns and LayerNorm scales are zero.
    # # Hint: use model.hook_dict to access specific hook points and check their stored gradients.
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    # Check 1: attention pattern gradients should be zero
    # We register a hook on the attention pattern to capture its gradient
    attn_grad = {}

    def capture_attn_grad(grad, hook):
        attn_grad["grad"] = grad
        return grad

    pattern_hook_name = "blocks.0.attn.hook_pattern"

    # Run forward pass with gradient tracking
    model.zero_grad()
    with t.enable_grad():
        hook_handle = model.hook_dict[pattern_hook_name].register_backward_hook(
            lambda module, grad_input, grad_output: capture_attn_grad(grad_output[0], module)
        )
        logits = model(input_ids_batch, stop_at_layer=model.cfg.n_layers)
        loss = logits[0, -1].sum()
        loss.backward()
        hook_handle.remove()

    # Attention patterns should have zero gradients (they're frozen via .detach())
    if "grad" in attn_grad:
        results["attention_patterns_frozen"] = attn_grad["grad"].abs().sum().item() == 0.0
    else:
        # If no gradient was captured, the pattern was detached before the backward pass
        results["attention_patterns_frozen"] = True

    # Check 2: model parameters should not require gradients
    n_params_with_grad = sum(1 for p in model.parameters() if p.requires_grad)
    results["parameters_frozen"] = n_params_with_grad <= 1  # embed hook may have grad

    return results
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    tokens = replacement_model.ensure_tokenized("The Eiffel Tower is in")
    results = verify_gradient_freezing(replacement_model, tokens)
    for component, is_frozen in results.items():
        status = "FROZEN" if is_frozen else "NOT FROZEN"
        print(f"  {component}: {status}")
        assert is_frozen, f"{component} should be frozen but isn't!"
    print("All gradient freezing checks passed!")
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Salient Logits

Before building the full attribution graph, we need to decide which output logits to include as target nodes. Computing attribution edges for the entire vocabulary (256k+ tokens for Gemma) would be wasteful since most tokens have negligible probability. Instead, we select the smallest set of top-probability tokens whose cumulative probability exceeds some threshold.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - implement `compute_salient_logits`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement the function below. It should:

1. Compute softmax probabilities from the raw logits
2. Take the top `max_n_logits` tokens by probability
3. Find the smallest subset whose cumulative probability reaches `desired_logit_prob`
4. Return the token indices, probabilities, and **demeaned** unembedding vectors

Why demean? The mean of the unembedding matrix across vocabulary tokens acts as a constant bias that doesn't carry token-specific information. By subtracting it, we isolate the direction that specifically promotes each token.

Note: the unembedding matrix `W_U` has shape `(d_model, d_vocab)`.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def compute_salient_logits(
    logits: t.Tensor,
    W_U: t.Tensor,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Select the smallest set of top logits whose cumulative prob >= desired_logit_prob.

    Args:
        logits: (d_vocab,) raw logit vector for a single position.
        W_U: (d_model, d_vocab) unembedding matrix.
        max_n_logits: Hard cap on the number of logits to select.
        desired_logit_prob: Cumulative probability threshold.

    Returns:
        logit_indices: (k,) vocabulary token ids of the selected logits.
        logit_probs: (k,) softmax probabilities of the selected logits.
        demeaned_vecs: (k, d_model) unembedding columns for each selected logit, with the
            mean unembedding vector subtracted.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    probs = t.softmax(logits.float(), dim=-1)
    top_p, top_idx = t.topk(probs, max_n_logits)
    cutoff = int(t.searchsorted(t.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = W_U[:, top_idx]  # (d_model, k)
    demean = W_U.mean(dim=-1, keepdim=True)  # (d_model, 1)
    demeaned_vecs = (cols - demean).T  # (k, d_model)

    return top_idx, top_p, demeaned_vecs
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    tests.test_compute_salient_logits(compute_salient_logits)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
Let's see which logits get selected for our Dallas prompt:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_3:
    prompt = "The capital of the state containing Dallas is"
    tokens = replacement_model.ensure_tokenized(prompt)
    with t.no_grad():
        logits = replacement_model(tokens.unsqueeze(0))

    logit_idx, logit_p, logit_vecs = compute_salient_logits(logits[0, -1], replacement_model.unembed.W_U)

    print("Selected logit tokens:")
    for i, (idx, p) in enumerate(zip(logit_idx, logit_p)):
        token_str = replacement_model.tokenizer.decode(idx)
        print(f"  {i}: {token_str!r} (prob={p:.4f})")
    print(f"Cumulative probability: {logit_p.sum():.4f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Computing Direct Effects

This is the core of the attribution algorithm. We'll use the `ReplacementModel`'s built-in attribution infrastructure (`setup_attribution` and `compute_batch`) to compute direct effects between all active features and our selected logit nodes.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - explore the attribution context

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

The `model.setup_attribution(input_ids)` method runs a forward pass and records everything needed for attribution:

- **`ctx.activation_matrix`**: A sparse 3D tensor of shape `(n_layers, n_pos, d_transcoder)` containing the non-zero transcoder feature activations.
- **`ctx.encoder_vecs`**: The encoder vectors (reading directions) for all active features.
- **`ctx.logits`**: The model's logits from the clean forward pass.

Write a function that summarizes the active features found by `setup_attribution`:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def summarize_active_features(ctx, model, input_ids: t.Tensor) -> None:
    """Print a summary of the active features found during the forward pass."""
    activation_matrix = ctx.activation_matrix
    # EXERCISE
    # # YOUR CODE HERE - extract the shape, number of active features, and per-layer counts
    # # from the activation_matrix and print a summary.
    # # Hint: activation_matrix.indices() returns (layers, positions, feature_indices),
    # # and activation_matrix._nnz() returns the total number of non-zero entries.
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    feat_layers, feat_pos, feat_idx = activation_matrix.indices()
    n_layers, n_pos, d_transcoder = activation_matrix.shape

    print(f"Input tokens: {n_pos}")
    print(f"Model layers: {n_layers}")
    print(f"Transcoder dictionary size: {d_transcoder}")
    print(f"Total active features: {activation_matrix._nnz()}")
    print(f"Average active features per (layer, position): {activation_matrix._nnz() / (n_layers * n_pos):.1f}")
    print()
    for layer in range(min(n_layers, 5)):
        n_feats = (feat_layers == layer).sum().item()
        print(f"  Layer {layer}: {n_feats} active features")
    if n_layers > 5:
        print(f"  ... ({n_layers - 5} more layers)")
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    ctx = replacement_model.setup_attribution(tokens)
    summarize_active_features(ctx, replacement_model, tokens)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - compute attribution edges

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 30-45 minutes on this exercise.
> ```

This is the core exercise. You'll compute the direct effect of every active feature on every logit node, and of every active feature on every other active feature, using batched backward passes.

The attribution context provides `ctx.compute_batch(layers, positions, inject_values, retain_graph)`, which:
- Takes a batch of (layer, position, inject_value) triples
- Injects each `inject_value` as a gradient seed at the given (layer, position) in the residual stream
- Collects the resulting gradient at all source (feature + error + embed) positions
- Returns a matrix of shape `(batch_size, n_source_nodes)` containing the direct effects

The function has two phases:

**Phase 1 (logit attribution):** For each batch of logit nodes, we inject the demeaned unembedding vectors (from `compute_salient_logits`) at `layer=n_layers, position=n_pos-1` (the final position after all transformer layers), and read off the gradients at all upstream feature/error/embed nodes.

**Phase 2 (feature attribution):** For each batch of active features, we inject their encoder vectors at their respective (layer, position), and read off gradients from all upstream nodes.

Before computing any backward passes, we need to run one forward pass with the replacement model hooks installed. This is done with `ctx.install_hooks(model)`, which returns a context manager. Inside that context, we run `model.forward(input_ids, stop_at_layer=n_layers)` and apply the final LayerNorm.

Important: set `retain_graph=True` on all but the last `compute_batch` call, otherwise PyTorch will free the computation graph after the first backward pass.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def compute_attribution_edges(
    ctx,
    model: ReplacementModel,
    input_ids: t.Tensor,
    logit_vecs: t.Tensor,
    batch_size: int = 64,
) -> tuple[t.Tensor, t.Tensor]:
    """Compute the attribution edge matrix for all logit and feature nodes.

    Args:
        ctx: Attribution context from model.setup_attribution()
        model: The ReplacementModel
        input_ids: 1D tensor of token ids
        logit_vecs: (n_logits, d_model) demeaned unembedding vectors
        batch_size: Number of nodes to process per backward pass

    Returns:
        edge_matrix: (n_logits + n_features, n_source_nodes) dense matrix of direct effects,
            where the first n_logits rows are logit attribution and the remaining rows are
            feature attribution.
        row_to_node_index: (n_logits + n_features,) mapping from row indices in edge_matrix
            to original node indices in the full graph.
    """
    activation_matrix = ctx.activation_matrix
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    n_features = activation_matrix._nnz()
    n_logits = logit_vecs.shape[0]

    logit_offset = n_features + (n_layers + 1) * n_pos
    total_source_nodes = logit_offset + n_logits

    edge_matrix = t.zeros(n_logits + n_features, total_source_nodes)
    row_to_node_index = t.zeros(n_logits + n_features, dtype=t.int32)

    # EXERCISE
    # # YOUR CODE HERE
    # # Step 1: Run the forward pass inside ctx.install_hooks(model)
    # # Step 2: Logit attribution - for each batch of logit vecs, call ctx.compute_batch
    # #         with layer=n_layers, position=n_pos-1
    # # Step 3: Feature attribution - for each batch of active features, call ctx.compute_batch
    # #         with the feature's layer, position, and encoder_vec
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    # Step 1: forward pass with replacement model hooks
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = model.ln_final(residual)

    # Step 2: logit attribution
    for i in range(0, n_logits, batch_size):
        batch = logit_vecs[i : i + batch_size]
        rows = ctx.compute_batch(
            layers=t.full((batch.shape[0],), n_layers),
            positions=t.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
        )
        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = t.arange(i, i + batch.shape[0]) + logit_offset

    # Step 3: feature attribution
    st = n_logits
    for i in range(0, n_features, batch_size):
        end = min(i + batch_size, n_features)
        idx_batch = t.arange(i, end)
        rows = ctx.compute_batch(
            layers=feat_layers[idx_batch],
            positions=feat_pos[idx_batch],
            inject_values=ctx.encoder_vecs[idx_batch],
            retain_graph=(end < n_features),
        )
        edge_matrix[st : st + rows.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[st : st + rows.shape[0]] = idx_batch
        st += rows.shape[0]
    # END SOLUTION

    return edge_matrix, row_to_node_index


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    ctx = replacement_model.setup_attribution(tokens)
    logit_idx, logit_p, logit_vecs = compute_salient_logits(ctx.logits[0, -1], replacement_model.unembed.W_U)
    edge_matrix, row_to_node_index = compute_attribution_edges(
        ctx, replacement_model, tokens, logit_vecs, batch_size=64
    )

    n_features = ctx.activation_matrix._nnz()
    n_logits = len(logit_idx)
    print(f"Edge matrix shape: {edge_matrix.shape}")
    print(f"Logit rows (first {n_logits}): {edge_matrix[:n_logits].abs().sum():.2f} total abs weight")
    print(f"Feature rows (remaining {n_features}): {edge_matrix[n_logits:].abs().sum():.2f} total abs weight")
    print(f"Sparsity: {(edge_matrix == 0).float().mean():.2%}")
    assert edge_matrix.abs().sum() > 0, "Edge matrix should have non-zero entries!"
    print("Edge computation looks correct!")
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - assemble the graph

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Now we package the edge matrix into a `Graph` object. The graph's adjacency matrix has a specific node ordering:

1. **Active feature nodes** (one per active transcoder feature)
2. **Error nodes** (one per layer per position): the transcoder reconstruction error
3. **Embed nodes** (one per token position): the token embeddings
4. **Logit nodes** (one per selected output token)

We need to build a square adjacency matrix where rows represent targets and columns represent sources. The logit rows (from `edge_matrix[:n_logits]`) go at the bottom of the full matrix, and the feature rows (from `edge_matrix[n_logits:]`) go at the top.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


from circuit_tracer.graph import Graph  # object to store full attribution graph, as nodes


def assemble_graph(
    edge_matrix: t.Tensor,
    row_to_node_index: t.Tensor,
    ctx,
    model: ReplacementModel,
    input_ids: t.Tensor,
    logit_idx: t.Tensor,
    logit_p: t.Tensor,
):
    """Package the edge matrix into a Graph object with the correct node ordering.

    The full adjacency matrix is square with dimensions:
        n_features + n_error_nodes + n_embed_nodes + n_logits

    where n_error_nodes = n_layers * n_pos and n_embed_nodes = n_pos.
    """

    activation_matrix = ctx.activation_matrix
    n_features = activation_matrix._nnz()
    n_logits = len(logit_idx)

    # EXERCISE
    # # YOUR CODE HERE - build the full square adjacency matrix and return a Graph object.
    # # The edge_matrix has logit rows first (indices 0..n_logits-1) then feature rows.
    # # In the full adjacency matrix, feature rows go at the top and logit rows at the bottom.
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    # Sort rows so features are in their original order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]

    total_nodes = edge_matrix.shape[1]
    full_adj = t.zeros(total_nodes, total_nodes)
    full_adj[:n_features] = edge_matrix[:n_features]
    full_adj[-n_logits:] = edge_matrix[n_features:]

    return Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=t.arange(n_features),
        adjacency_matrix=full_adj,
        cfg=model.cfg,
        scan=model.scan,
    )
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    graph = assemble_graph(edge_matrix, row_to_node_index, ctx, replacement_model, tokens, logit_idx, logit_p)
    n_total = graph.adjacency_matrix.shape[0]
    n_feat = len(graph.selected_features)
    n_log = len(graph.logit_tokens)
    n_tok = len(graph.input_tokens)
    print(f"Graph assembled: {n_total} total nodes")
    print(f"  {n_feat} feature nodes")
    print(f"  {n_total - n_feat - n_log - n_tok} error nodes")
    print(f"  {n_tok} embed nodes")
    print(f"  {n_log} logit nodes")
    assert graph.adjacency_matrix.shape[0] == graph.adjacency_matrix.shape[1], "Adjacency matrix must be square"
    assert graph.adjacency_matrix.abs().sum() > 0, "Adjacency matrix has no edges!"
    print("Graph assembly looks correct!")
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Adjacency Matrices

Now that we have a full (dense) attribution graph, we need to understand which nodes actually matter. The raw graph has thousands of nodes and millions of potential edges, most of which are negligible. **Graph pruning** selects the most influential subgraph.

The key concept is **node influence**: the total effect of a node on the output logits, accounting for both direct effects and indirect effects through intermediate nodes.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - implement `normalize_matrix`

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵⚪⚪⚪
>
> You should spend up to 5 minutes on this exercise.
> ```

The first step in computing influence is to normalize the adjacency matrix. Each row is divided by the sum of its absolute values, so that each row represents a probability distribution over sources. Rows that sum to zero should remain zero (use a clamp to avoid division by zero).
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def normalize_matrix(matrix: t.Tensor) -> t.Tensor:
    """Row-normalize a matrix by absolute values.

    Takes elementwise abs, then divides each row by its sum (clamped to avoid div-by-zero).
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    normalized = matrix.abs()
    return normalized / normalized.sum(dim=1, keepdim=True).clamp(min=1e-10)
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    tests.test_normalize_matrix(normalize_matrix)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - implement `compute_influence`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 20-30 minutes on this exercise.
> ```

Given the normalized adjacency matrix $A$ and a logit weight vector $w$ (with the logit probabilities at the logit node positions and zeros elsewhere), the total influence of each node is:

$$\text{influence} = w A + w A^2 + w A^3 + \ldots$$

We compute this iteratively: start with $v_0 = wA$, then $v_{k+1} = v_k A$, accumulating $\text{influence} = \sum_k v_k$.

**Why does this converge?** Because of the attribution graph's causal structure, features in layer $i$ can only have edges to features in layers $j < i$. This means $A$ is strictly lower-triangular with respect to layer ordering, so $A^L = 0$ where $L$ is the number of layers. The matrix is **nilpotent**, so the iteration always terminates in at most $L$ steps. This is a nice computational property: there is no convergence concern, and the number of iterations is bounded by the network depth.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def compute_influence(
    A: t.Tensor,
    logit_weights: t.Tensor,
    max_iter: int = 1000,
) -> t.Tensor:
    """Compute total influence of each node on the output logits via power iteration.

    The iteration computes: influence = w @ A + w @ A^2 + w @ A^3 + ...
    and terminates when the current product is all zeros (guaranteed by nilpotency).

    Args:
        A: (n_nodes, n_nodes) normalized adjacency matrix.
        logit_weights: (n_nodes,) vector with logit probabilities at logit node positions.
        max_iter: Safety cap on iterations.

    Returns:
        influence: (n_nodes,) total influence of each node.
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
            raise RuntimeError(f"Influence computation failed to converge after {iterations} iterations")
        current = current @ A
        influence += current
        iterations += 1
    return influence
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    tests.test_compute_influence(compute_influence, normalize_matrix)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
Let's visualize the node influence for our Dallas graph:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_3:
    n_logits_graph = len(graph.logit_tokens)
    n_features_graph = len(graph.selected_features)
    logit_weights = t.zeros(graph.adjacency_matrix.shape[0])
    logit_weights[-n_logits_graph:] = graph.logit_probabilities

    node_influence = compute_influence(normalize_matrix(graph.adjacency_matrix), logit_weights)

    # Show top-20 most influential feature nodes
    top_influence, top_indices = node_influence[:n_features_graph].topk(min(20, n_features_graph))
    print("Top 20 most influential feature nodes:")
    for rank, (inf, idx) in enumerate(zip(top_influence, top_indices)):
        feat = graph.active_features[idx]
        print(f"  {rank}: layer={feat[0].item()}, pos={feat[1].item()}, feat_idx={feat[2].item()}, influence={inf:.4f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - implement `prune_graph`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 25-35 minutes on this exercise.
> ```

Graph pruning has three stages:

1. **Node pruning**: Compute node influence, find the threshold that keeps the top `node_threshold` fraction of total influence, mask out nodes below the threshold. Always keep embed and logit nodes.
2. **Edge pruning**: For surviving nodes, compute edge influence scores and keep edges above the `edge_threshold` fraction. The edge score for edge $(i, j)$ is $A'_{ij} \cdot (\text{influence}_i + w_i)$, where $A'$ is the normalized pruned matrix and $w_i$ is the logit weight.
3. **Iterative cleanup**: After pruning edges, some feature/error nodes may have lost all incoming or outgoing edges. Remove them and repeat until stable.

We provide the helper functions `compute_node_influence`, `compute_edge_influence`, and `find_threshold` in `utils.py`.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def prune_graph(
    graph,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> tuple[t.Tensor, t.Tensor]:
    """Prune a graph by removing low-influence nodes and edges.

    Args:
        graph: The Graph object (from circuit_tracer.graph).
        node_threshold: Keep nodes contributing to this fraction of total influence.
        edge_threshold: Keep edges contributing to this fraction of total influence.

    Returns:
        node_mask: Boolean tensor indicating which nodes to keep.
        edge_mask: Boolean tensor indicating which edges to keep.
    """
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)

    logit_weights = t.zeros(graph.adjacency_matrix.shape[0])
    logit_weights[-n_logits:] = graph.logit_probabilities

    # EXERCISE
    # # YOUR CODE HERE
    # # Step 1: compute node influence, apply threshold, always keep embed + logit nodes
    # # Step 2: zero out pruned nodes in the adjacency matrix
    # # Step 3: compute edge influence, apply threshold
    # # Step 4: iteratively remove nodes that lost all incoming/outgoing edges
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    # Step 1: node pruning
    node_influence = utils.compute_node_influence(graph.adjacency_matrix, logit_weights)
    node_mask = node_influence >= utils.find_threshold(node_influence, node_threshold)
    node_mask[-n_logits - n_tokens :] = True

    # Step 2: zero out pruned nodes
    pruned_matrix = graph.adjacency_matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0

    # Step 3: edge pruning
    edge_scores = utils.compute_edge_influence(pruned_matrix, logit_weights)
    edge_mask = edge_scores >= utils.find_threshold(edge_scores.flatten(), edge_threshold)

    # Step 4: iterative cleanup
    old_node_mask = node_mask.clone()
    node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
    node_mask[:n_features] &= edge_mask[:n_features].any(1)

    while not t.all(node_mask == old_node_mask):
        old_node_mask[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        node_mask[:n_features] &= edge_mask[:n_features].any(1)
    # END SOLUTION

    return node_mask, edge_mask


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    node_mask, edge_mask = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)
    n_total = graph.adjacency_matrix.shape[0]
    n_kept = node_mask.sum().item()
    n_edges = edge_mask.sum().item()
    print(f"Nodes: {n_kept}/{n_total} kept ({n_kept / n_total:.1%})")
    print(f"Edges: {n_edges} kept ({edge_mask.float().mean():.4%} of all possible)")
    tests.test_prune_graph(prune_graph, graph)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Full Pipeline

Let's combine everything into a single `attribute` function and compare against the library's implementation.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - implement `attribute`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Combine `compute_salient_logits`, `compute_attribution_edges`, and `assemble_graph` into a single function:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def attribute(
    prompt: str,
    model: ReplacementModel,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 64,
):
    """Compute a full attribution graph for a prompt.

    Returns:
        graph: A Graph object with the dense adjacency matrix.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    input_ids = model.ensure_tokenized(prompt)
    ctx = model.setup_attribution(input_ids)

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1],
        model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )

    edge_matrix, row_to_node_index = compute_attribution_edges(ctx, model, input_ids, logit_vecs, batch_size)
    graph = assemble_graph(edge_matrix, row_to_node_index, ctx, model, input_ids, logit_idx, logit_p)
    return graph
    # END SOLUTION


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# For comparing with the library's implementation:
from circuit_tracer.attribution.attribute_transformerlens import attribute as library_attribute

if MAIN and FLAG_RUN_SECTION_3:
    # Run our implementation
    prompt = "The capital of the state containing Dallas is"
    student_graph = attribute(prompt, replacement_model)

    library_graph = library_attribute(prompt, replacement_model, verbose=True)

    print(f"\nStudent graph: {student_graph.adjacency_matrix.shape}")
    print(f"Library graph: {library_graph.adjacency_matrix.shape}")
    print(f"Student top logit: {replacement_model.tokenizer.decode(student_graph.logit_tokens[0])!r}")
    print(f"Library top logit: {replacement_model.tokenizer.decode(library_graph.logit_tokens[0])!r}")

    # Prune and compare
    student_node_mask, student_edge_mask = prune_graph(student_graph)
    print(f"\nStudent pruned: {student_node_mask.sum()} nodes, {student_edge_mask.sum()} edges")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
Congratulations! You've implemented the full circuit tracing algorithm from scratch. The library version includes a few extra features (iterative feature ranking to prioritise the most influential features, memory offloading for large models), but the core algorithm is exactly what you've built.

In the next section, we'll use the library's pre-computed graphs and intervention tools to explore real circuits.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
# 4️⃣ Exploring Circuits & Interventions
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Introduction

Now that you've built the attribution graph algorithm from scratch, let's use the `circuit-tracer` library to explore real circuits and test their causal structure via interventions. We'll study the **Dallas/Austin two-hop factual recall** circuit, which requires the model to perform two-hop reasoning: Dallas is in Texas, and the capital of Texas is Austin.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## The Dallas/Austin Circuit

The prompt `"Fact: the capital of the state containing Dallas is"` requires two-hop reasoning. The attribution graph reveals a clear circuit with distinct **supernodes** (groups of related features):

- **"state" features**: activated by the concept of US states in the prompt
- **"Texas" features**: intermediate features that represent "the state is Texas"
- **"capital" features**: activated by the concept of capital cities
- **"Say Austin" features**: late-layer features that promote "Austin" in the logits
- **"Say a capital" features**: features that promote capital-city tokens generally

Let's load a pre-computed graph and inspect these supernodes.
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_3:
    Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])

    # Compute a fresh graph using the library (or you could load a pre-computed one)
    dallas_prompt = "Fact: the capital of the state containing Dallas is"
    dallas_graph = library_attribute(dallas_prompt, replacement_model, verbose=True)

    # Get activations for the Dallas prompt (we'll need these for interventions)
    _, dallas_activations = replacement_model.get_activations(dallas_prompt, sparse=True)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Feature Interventions

The `ReplacementModel` provides two key intervention methods:

- **`model.feature_intervention(prompt, interventions)`**: Runs a single forward pass with specified feature activations overridden. Returns `(logits, activations)`.
- **`model.feature_intervention_generate(prompt, interventions, ...)`**: Generates multiple tokens with interventions applied throughout. Returns `(text, logits, activations)`.

Each intervention is a tuple `(layer, position, feature_idx, new_value)`.
"""

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - zero ablation experiments

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20-30 minutes on this exercise.
> ```

If the attribution graph's claims about the circuit are correct, then ablating (zeroing) key features should disrupt the model's ability to predict "Austin". Let's test this.

Write a function that takes a list of features to ablate and returns the original and modified logits:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def run_ablation(
    model: ReplacementModel,
    prompt: str,
    features_to_ablate: list,
) -> tuple[t.Tensor, t.Tensor]:
    """Run a zero-ablation experiment: set the given features to zero and compare logits.

    Args:
        model: The ReplacementModel
        prompt: The input prompt
        features_to_ablate: List of Feature namedtuples (or (layer, pos, feat_idx) tuples)

    Returns:
        original_logits: (1, seq, vocab) logits without intervention
        ablated_logits: (1, seq, vocab) logits with features zeroed
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    interventions = [(*f, 0.0) for f in features_to_ablate]

    with t.inference_mode():
        original_logits, _ = model.feature_intervention(prompt, [])
        ablated_logits, _ = model.feature_intervention(prompt, interventions)

    return original_logits, ablated_logits
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    # We'll manually define some features from the Dallas graph to ablate.
    # These are approximate "Texas" features found in the Gemma 2-2B graph.
    # (In practice you'd extract these from the Neuronpedia graph URL.)

    # Run a quick test: ablating ALL features at a late layer position should change output
    feat_layers, feat_pos, feat_idx = dallas_graph.active_features.T
    last_pos = feat_pos.max().item()

    # Get features at the last position, late layers (likely "say Austin" features)
    late_layer_mask = (feat_layers >= 20) & (feat_pos == last_pos)
    if late_layer_mask.any():
        late_features = [
            Feature(layer=feat_layers[i].item(), pos=feat_pos[i].item(), feature_idx=feat_idx[i].item())
            for i in t.where(late_layer_mask)[0][:10]  # take first 10
        ]

        orig_logits, abl_logits = run_ablation(replacement_model, dallas_prompt, late_features)

        orig_probs = orig_logits[0, -1].float().softmax(-1)
        abl_probs = abl_logits[0, -1].float().softmax(-1)

        # Show top predictions before and after
        print("Original top predictions:")
        top_orig = orig_probs.topk(5)
        for p, idx in zip(top_orig.values, top_orig.indices):
            print(f"  {replacement_model.tokenizer.decode(idx)!r}: {p:.4f}")

        print("\nAfter ablating late-layer features:")
        top_abl = abl_probs.topk(5)
        for p, idx in zip(top_abl.values, top_abl.indices):
            print(f"  {replacement_model.tokenizer.decode(idx)!r}: {p:.4f}")
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - cross-prompt feature swapping

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 25-35 minutes on this exercise.
> ```

The most compelling test of circuit understanding is **cross-prompt feature swapping**. If the "Texas" features truly represent "the state is Texas", then swapping them with "California" features (from an Oakland prompt) should make the model predict "Sacramento" instead of "Austin".

Implement a function that:
1. Gets activations from a "swap" prompt
2. Builds interventions that zero out `features_off` and activate `features_on` at their in-distribution values from the swap prompt (scaled by `scale`)
3. Runs the intervention
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def cross_prompt_swap(
    model: ReplacementModel,
    base_prompt: str,
    swap_prompt: str,
    features_off: list,
    features_on: list,
    scale: float = 2.0,
) -> tuple[t.Tensor, t.Tensor]:
    """Swap features between prompts: turn off features_off and activate features_on
    at their values from swap_prompt (scaled by `scale`).

    Args:
        model: The ReplacementModel.
        base_prompt: The prompt to intervene on.
        swap_prompt: The prompt to get replacement activation values from.
        features_off: Features to zero-ablate (from base_prompt's graph).
        features_on: Features to activate (from swap_prompt's graph).
        scale: Multiplier for the replacement activation values.

    Returns:
        original_logits, modified_logits
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    _, swap_activations = model.get_activations(swap_prompt, sparse=True)

    interventions = [(*f, 0.0) for f in features_off]
    interventions += [(*f, scale * swap_activations[f]) for f in features_on]

    with t.inference_mode():
        original_logits, _ = model.feature_intervention(base_prompt, [])
        modified_logits, _ = model.feature_intervention(base_prompt, interventions)

    return original_logits, modified_logits
    # END SOLUTION


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_3:
    # Demonstrate with the Oakland/Sacramento swap
    # First, compute a graph for Oakland to find California features
    oakland_prompt = "Fact: the capital of the state containing Oakland is"
    oakland_graph = library_attribute(oakland_prompt, replacement_model, verbose=True)

    # Find features that are unique to Oakland (not in Dallas) at intermediate layers
    # These are likely "California" features
    oak_feats = set((r[0].item(), r[1].item(), r[2].item()) for r in oakland_graph.active_features)
    dal_feats = set((r[0].item(), r[1].item(), r[2].item()) for r in dallas_graph.active_features)

    # Features in Oakland but not Dallas, at intermediate layers and last position
    last_pos_oak = oakland_graph.active_features[:, 1].max().item()
    california_candidates = [Feature(*f) for f in (oak_feats - dal_feats) if f[1] == last_pos_oak and 10 <= f[0] <= 22]
    print(f"Found {len(california_candidates)} candidate California features")

    # Features in Dallas but not Oakland at intermediate layers
    last_pos_dal = dallas_graph.active_features[:, 1].max().item()
    texas_candidates = [Feature(*f) for f in (dal_feats - oak_feats) if f[1] == last_pos_dal and 10 <= f[0] <= 22]
    print(f"Found {len(texas_candidates)} candidate Texas features")

    # Run the swap (this is a rough approximation - with Neuronpedia supernodes it would be cleaner)
    if texas_candidates and california_candidates:
        orig_logits, mod_logits = cross_prompt_swap(
            replacement_model,
            dallas_prompt,
            oakland_prompt,
            features_off=texas_candidates[:20],
            features_on=california_candidates[:20],
            scale=2.0,
        )

        print("\nOriginal predictions (Dallas prompt):")
        orig_probs = orig_logits[0, -1].float().softmax(-1)
        for p, idx in zip(*orig_probs.topk(5)):
            print(f"  {replacement_model.tokenizer.decode(idx)!r}: {p:.4f}")

        print("\nAfter swapping Texas -> California features:")
        mod_probs = mod_logits[0, -1].float().softmax(-1)
        for p, idx in zip(*mod_probs.topk(5)):
            print(f"  {replacement_model.tokenizer.decode(idx)!r}: {p:.4f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
### Exercise - open-ended generation with interventions

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

For multi-token generation, interventions need to persist across all generated positions. This is done by replacing the fixed position with an open-ended `slice`:
"""

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []


def generate_with_intervention(
    model: ReplacementModel,
    prompt: str,
    interventions: list,
    max_new_tokens: int = 20,
) -> tuple[str, str]:
    """Generate text with and without feature interventions.

    Converts fixed-position interventions to open-ended slices so the intervention persists
    across all generated tokens.

    Args:
        model: The ReplacementModel.
        prompt: The input prompt.
        interventions: List of (layer, pos, feat_idx, value) tuples.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        pre_text: Generated text without intervention.
        post_text: Generated text with intervention.
    """
    # EXERCISE
    # # YOUR CODE HERE
    # # 1. Compute the input sequence length
    # # 2. Convert each intervention's position to slice(seq_len - 1, None, None)
    # # 3. Call model.feature_intervention_generate with and without interventions
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    seq_len = len(model.tokenizer(prompt).input_ids)
    open_interventions = []
    for layer, pos, feat_idx, value in interventions:
        open_pos = slice(seq_len - 1, None, None)
        open_interventions.append((layer, open_pos, feat_idx, value))

    pre_text = model.feature_intervention_generate(
        prompt, [], do_sample=False, verbose=False, max_new_tokens=max_new_tokens
    )[0]
    post_text = model.feature_intervention_generate(
        prompt, open_interventions, do_sample=False, verbose=False, max_new_tokens=max_new_tokens
    )[0]
    return pre_text, post_text
    # END SOLUTION


# HIDE
if MAIN and FLAG_RUN_SECTION_3:
    # Demo: ablate the late-layer features during generation
    if late_layer_mask.any():
        ablation_interventions = [(*f, 0.0) for f in late_features]
        pre_text, post_text = generate_with_intervention(
            replacement_model, dallas_prompt, ablation_interventions, max_new_tokens=15
        )
        print(f"Without intervention:\n  {pre_text}")
        print(f"\nWith late-layer ablation:\n  {post_text}")
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r"""
## Bonus: Cross-Layer Transcoders

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵⚪⚪⚪
>
> This is optional - skip if you're short on time.
> ```

The exercises above all used single-layer (per-layer) transcoders. The circuit-tracer library also supports **cross-layer transcoders (CLTs)**, where each feature can write its decoder vector to multiple downstream layers. To load a CLT model, replace `"gemma"` with `"gemma_clt"`:

```python
clt_model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b", "gemma_clt", dtype=t.bfloat16, backend="transformerlens"
)
```

Try computing an attribution graph with the CLT model and compare:
1. How does the number of active features compare?
2. Do the graph scores (replacement score, completeness score) differ?
3. Do the same supernodes appear?

You can compute graph scores using:

```python
from circuit_tracer.graph import compute_graph_scores
replacement_score, completeness_score = compute_graph_scores(graph)
```
"""
