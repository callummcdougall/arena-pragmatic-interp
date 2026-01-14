# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {"title": "SAE Circuits", "icon": "2-circle-fill", "subtitle": "(100%)"},
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [1.4.2] SAE Circuits
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/header-13-2.png" width="350">
<br>
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
In these exercises, we dive deeply into the interpretability research that can be done with sparse autoencoders. We'll start by introducing two important tools: `SAELens` (essentially the TransformerLens of SAEs, which also integrates very well with TransformerLens) and **Neuronpedia**, an open platform for interpretability research. We'll then move through a few other exciting domains in SAE interpretability, grouped into several categories (e.g. understanding / classifying latents, or finding circuits in SAEs).

We expect some degree of prerequisite knowledge in these exercises. Specifically, it will be very helpful if you understand:

- What **superposition** is
- What the **sparse autoencoder** architecture is, and why it can help us disentangle features from superposition

We've included an abridged version of the exercise set **1.3.1 Superposition & SAEs**, which contains all the material we view as highly useful for the rest of these exercises. If you've already gone through this exercise set then you can proceed straight to section 1️⃣, if not then we recommend at least skimming through section 0️⃣ so that you feel comfortable with the core geometric intuitions for superposition and how SAEs work.

One note before starting - we'll be mostly adopting the terminology that **features** are characteristics of the underlying data distribution that our base models are trained on, and **SAE latents** (or just "latents") are the directions in the SAE. This is to avoid the overloading of the term "feature", and avoiding the implicit assumption that "SAE features" correspond to real features in the data. We'll relax this terminology when we're looking at SAE latents which very clearly correspond to specific interpretable features in the data.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Reading Material

Most of this is optional, and can be read at your leisure depending on what interests you most & what level of background you have. If we could recommend just one, it would be "Towards Monosemanticity" - particularly the first half of "Problem Setup", and the sections where they take a deep dive on individual latents.

- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) outlines the core ideas behind superposition - what it is, why it matters for interepretability, and what we might be able to do about it.
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html) arguably took the first major stride in mechanistic interpretability with SAEs: training them on a 1-layer model, and extracting a large number of interpretable features.
- [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) shows how you can scale up the science of SAEs to larger models, specifically the SOTA (at the time) model Claude 3 Sonnet. It provides an interesting insight into where the field might be moving in the near future.
- [Improving Dictionary Learning with Gated Sparse Autoencoders](https://arxiv.org/pdf/2404.16014) is a paper from DeepMind which introduces the Gated SAE architecture, demonstrating how it outperforms the standard architecture and also motivating its use by speculating about underlying feature distributions.
- [Gemma Scope](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/) announces DeepMind's release of a comprehensive suite of open-sourced SAE models (trained with JumpReLU architecture). We'll be working a lot more with Gemma Scope models in subsequent exercises!
- [LessWrong, SAEs tag](https://www.lesswrong.com/tag/sparse-autoencoders-saes) contains a collection of posts on LessWrong that discuss SAEs, and is a great source of inspiration for further independent research!
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ SAE Circuits

SAEs are cool and interesting and we can steer on their latents to produce cool and interesting effects - but does this mean that we've truly unlocked the true units of computation used by our models, or have we just found an interesting clustering algorithm? The answer is that we don't really know yet! One strong piece of evidence for the former would be finding **circuits with SAEs**, in other words sets of latents in different layers of the transformer which communicate with each other, and explain some particular behaviour in an end-to-end way. How to find these kinds of circuits, and what they look like, is what we'll explore in this section.

> ##### Learning Objectives
>
> - Learn how to find connections between SAE latents in different layers of the transformer
> - Discover how to apply knowledge of SAE circuits to remove the bias from a linear classifier, as described in the Sparse Feature Circuits paper (not implemented yet)
> - Study transcoders, and understand how they can improve circuit analysis compared to regular SAEs
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## A note on memory usage

In these exercises, we'll be loading some pretty large models into memory (e.g. Gemma 2-2B and its SAEs, as well as a host of other models in later sections of the material). It's useful to have functions which can help profile memory usage for you, so that if you encounter OOM errors you can try and clear out unnecessary models. For example, we've found that with the right memory handling (i.e. deleting models and objects when you're not using them any more) it should be possible to run all the exercises in this material on a Colab Pro notebook, and all the exercises minus the handful involving Gemma on a free Colab notebook.

<details>
<summary>See this dropdown for some functions which you might find helpful, and how to use them.</summary>

First, we can run some code to inspect our current memory usage. Here's me running this code during the exercise set on SAE circuits, after having already loaded in the Gemma models from the previous section. This was on a Colab Pro notebook.

```python
import part31_superposition_and_saes.utils as utils

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
        if isinstance(obj, t.nn.Module) and part32_utils.get_tensors_size(obj) / 1024**3 > THRESHOLD:
            if hasattr(obj, "cuda"):
                obj.cpu()
            if hasattr(obj, "reset"):
                obj.reset()
    except:
        pass

# Move our gpt2 model & SAEs back to GPU (we'll need them for the exercises we're about to do)
gpt2.to(device)
gpt2_saes = {layer: sae.to(device) for layer, sae in gpt2_saes.items()}

part32_utils.print_memory_status()
```

<pre style="font-family: Consolas; font-size: 14px">Allocated = 14.90 GB
Reserved = 39.56 GB
Free = 24.66</pre>

Mission success! We've managed to free up a lot of memory. Note that the code which moves all objects collected by the garbage collector to the CPU is often necessary to free up the memory. We can't just delete the objects directly because PyTorch can still sometimes keep references to them (i.e. their tensors) in memory. In fact, if you add code to the for loop above to print out `obj.shape` when `obj` is a tensor, you'll see that a lot of those tensors are actually Gemma model weights, even once you've deleted `gemma_2_2b`.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup (don't read, just run)
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
#     %pip install "openai==1.56.1" einops datasets jaxtyping "sae-lens>=4.0.0,<5.0.0" openai tabulate umap-learn hdbscan eindex-callum git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python git+https://github.com/callummcdougall/sae_vis.git@callum/v3 transformer_lens==2.11.0

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

import gc
import itertools
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import circuitsvis as cv
import einops
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import torch as t
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from IPython.display import HTML, IFrame, display
from jaxtyping import Float, Int
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_vis import SaeVisConfig, SaeVisData, SaeVisLayoutConfig
from tabulate import tabulate
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt, to_numpy

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part32_interp_with_saes"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
# FILTERS: ~colab
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
# END FILTERS

# There's a single utils & tests file for both parts 3.1 & 3.2
import part31_superposition_and_saes.tests as tests
import part31_superposition_and_saes.utils as utils
from plotly_utils import imshow, line

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

        output.serve_kernel_port_as_iframe(
            PORT, path=filename, height=height, cache_in_notebook=True
        )

        PORT += 1

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ SAE Circuits
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Introduction

Our work so far has focused on understanding individual latents. In later sections we'll take deeper dives into specific methods for interpreting latents, but in this section we address a highly important topic - what about **circuits of SAAE latents**? Circuit analysis has already been somewhat successful in language model interpretability (e.g. see Anthropic's work on induction circuits, or the Indirect Object Identification paper), but many attempts to push circuit analysis further has hit speedbumps: most connections in the model are not sparse, and it's very hard to disentangle all the messy cross-talk between different components and residual stream subspaces. Circuit offer a better path forward, since we should expect that not only are individual latents generally sparse, they are also **sparsely connected** - any given latent should probably only have a downstream effect on a small number of other latents.

Indeed, if this does end up being true, it's a strong piece of evidence that latents found by SAEs *are* the **fundamental units of computation** used by the model, as opposed to just being an interesting clustering algorithm. Of course we do already have some evidence for this (e.g. the effectiveness of latent steering, and the fact that latents have already revealed important information about models which isn't clear when just looking at the basic components), but finding clear latent circuits would be a much stronger piece of evidence.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
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

latent_latent_gradients = torch.func.jacrev(latent_acts_to_later_latent_acts)(layer_1_latents)
```

where `jacrev` is shorthand for "Jacobian reverse-mode differentiation" - it's a PyTorch function that takes in a tensor -> tensor function `f(x) = y` and returns the Jacobian function, i.e. `g` s.t. `g[i, j] = d(f[x]_i) / d(x_j)`.

If we wanted to get a sense of how latents communicate with each other across our distribution of data, then we might average these results over a large set of prompts. However for now, we're going to stick with a relatively small set of prompts to avoid running into memory issues, and so we can visualise the results more easily.

First, let's load in our model & SAEs, if you haven't already:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

r'''
Now, we can start the exercises!

Note - the subsequent 3 exercises are all somewhat involved, and things like the use of Jacobian can be quite fiddly. For that reason, there's a good case to be made for just reading through the solutions and understanding what the code is doing, rather than trying to do it yourself. One option would be to look at the solutions for these 3 exercises and understand how latent-to-latent gradients work, but then try and implement the `token_to_latent_gradients` function (after the next 3 exercises) yourself.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise (1/3) - implement the `SparseTensor` class

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵⚪⚪⚪⚪
> 
> You should spend up to 10-15 minutes on this exercise (or skip it)
> ```

Firstly, we're going to create a `SparseTensor` class to help us work with sparse tensors (i.e. tensors where most of the elements are zero). This is because we'll need to do forward passes on the dense tensors (i.e. the tensors containing all values, including the zeros) but we'll often want to compute gradients wrt the sparse tensors (just the non-zero values) because otherwise we'd run into memory issues - there are a lot of latents!

You should fill in the `from_dense` and `from_sparse` class methods for the `SparseTensor` class. The testing code is visible to you, and should help you understand how this class is expected to behave.
'''

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
if MAIN:
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
    sparse_tensor = SparseTensor.from_sparse(
        (nonzero_values, nonzero_indices.unsqueeze(-1), tuple(x.shape))
    )
    t.testing.assert_close(sparse_tensor.dense, x)

    # Verify other properties
    t.testing.assert_close(sparse_tensor.values, nonzero_values)
    t.testing.assert_close(sparse_tensor.indices, nonzero_indices)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise (2/3) - implement `latent_acts_to_later_latent_acts`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 10-20 minutes on this exercise.
> ```

Next, you should implement the `latent_acts_to_later_latent_acts`. This takes latent activations earlier in the model (in a sparse form, i.e. tuple of (values, indices, shape)) and returns the downstream latent activations as a tuple of `(sparse_values, (dense_values,))`.

Why do we return 2 copies of `latent_acts_next` in this strange way? The answer is that we'll be wrapping our function with `torch.func.jacrev(latent_acts_to_later_latent_acts, has_aux=True)`. The `has_aux` argument allows us to return a tuple of tensors which won't be differentiated. In other words, it takes a tensor -> (tensor, tuple_of_tensors) function `f(x) = (y, aux)` and returns the function `g(x) = (J, aux)` where `J[i, j] = d(f[x]_i) / d(x_j)`. In other words, we're getting both the Jacobian and the actual reconstructed activations.

<details>
<summary>Note on what gradients we're actually computing</summary>

Eagle-eyed readers might have noticed that what we're actually doing here is not computing gradients between later and earlier latents, but computing the gradient between **reconstructed later latents** and earlier latents. In other words, the later latents we're differentiating are actually a function of the earlier SAE's residual stream reconstruction, rather than the actual residual stream. This is a bit risky when drawing conclusions from the results, because if your earlier SAE isn't very good at reconstructing its input then you might miss out on ways in which downstream latents are affected by upstream activations. A good way to sanity check this is to compare the latent activations (computed downstream of the earlier SAE's reconstructions) with the true latent activations, and make sure they're similar.

</details>

We'll get to applying the Jacobian in the 3rd exercise though - for now, you should just fill in `latent_acts_to_later_latent_acts`. This should essentially match the pseudocode for `latent_acts_to_later_latent_acts` which we gave at the start of this section (with the added factor of having to convert tensors to / from their sparse forms). Some guidance on syntax you'll find useful:

- All SAEs have `encode` and `decode` methods, which map from input -> latent activations -> reconstructed input.
- All TransformerLens models have a `forward` method with optional arguments `start_at_layer` and `stop_at_layer`, if these are supplied then it will compute activations from the latter layer as a function of the former.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def latent_acts_to_later_latent_acts(
    latent_acts_nonzero: Float[Tensor, "nonzero_acts"],
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
    latent_acts = SparseTensor.from_sparse(
        (latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape)
    ).dense
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

r'''
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
'''

# ! CELL TYPE: code
# ! FILTERS: [soln,py]
# ! TAGS: [main]

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
# ! TAGS: [main]

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
    x=[
        f"F{layer_to}.{latent}, {str_toks[seq]!r} ({seq})"
        for (_, seq, latent) in latent_acts_next_recon.indices
    ],
    y=[
        f"F{layer_from}.{latent}, {str_toks[seq]!r} ({seq})"
        for (_, seq, latent) in latent_acts_prev.indices
    ],
    labels={"x": f"To layer {layer_to}", "y": f"From layer {layer_from}"},
    title=f'Gradients between SAE latents in layer {layer_from} and SAE latents in layer {layer_to}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
    width=1600,
    height=1000,
).show()

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html,st-dropdown[Click to see the expected output]]

r'''
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"># nonzero latents (true): 181
# nonzero latents (reconstructed): 179
# latents alive in one but not both: 8</pre>

<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13220.html" width="1620" height="1020"></div>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
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
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
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
'''

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
    resid_after_embed = einops.einsum(
        resid_after_embed, token_scales, "... seq d_model, ... seq -> ... seq d_model"
    )
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

    token_latent_grads = einops.rearrange(
        token_latent_grads, "d_sae_nonzero batch seq -> batch seq d_sae_nonzero"
    )

    latent_acts = SparseTensor.from_dense(latent_acts_dense)
    # END SOLUTION

    return (token_latent_grads, latent_acts)


# HIDE
if MAIN:
    sae_layer = 3
    token_latent_grads, latent_acts = token_to_latent_gradients(
        tokens, sae=gpt2_saes[sae_layer], model=gpt2
    )

    px.imshow(
        to_numpy(token_latent_grads[0]),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        x=[
            f"F{sae_layer}.{latent:05}, {str_toks[seq]!r} ({seq})"
            for (_, seq, latent) in latent_acts.indices
        ],
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

r'''
<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13221.html" width="1920" height="470"></div>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Some observations</summary>

In the previous exercise, we saw gradients between `(F0.16911, " E") -> (F3.15266, "iff")`, which seems like it could be forming a bigram circuit for words which start with `" E"`.

In this plot, we can see a gradient between the `" E"` token and feature `F3.15266`, which is what we'd expect based on this.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
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
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def latent_acts_to_logits(
    latent_acts_nonzero: Float[Tensor, "nonzero_acts"],
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
    latent_acts = SparseTensor.from_sparse(
        (latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape)
    ).dense

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
# ! TAGS: [main]

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
        f"{str_toks[seq]!r} ({seq}), latent {latent:05}"
        for (_, seq, latent) in latent_acts.indices[sorted_indices]
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

r'''
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
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Some observations</summary>

We see that feature `F9.22250` stands out as boosting the `" Paris"` token far more than any of the other top predictions. Investigation reveals this feature fires primarily on French language text, which makes sense!

We also see `F9.5879` which seems to strongly boost words associated with Germany (e.g. Berlin, Hamberg, Cologne, Zurich). We see a similar pattern there, where that feature mostly fires on German-language text (or more commonly, English-language text talking about Germany).

```python
display_dashboard(sae_id="blocks.9.hook_resid_pre", latent_idx=22250)
display_dashboard(sae_id="blocks.9.hook_resid_pre", latent_idx=5879)
```

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
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
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def latent_acts_to_later_latent_acts_attn(
    latent_acts_nonzero: Float[Tensor, "nonzero_acts"],
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
    latent_acts = SparseTensor.from_sparse(
        (latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape)
    ).dense
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
# ! TAGS: [main]

# Move back onto GPU (if we moved it to CPU earlier)
attn_saes = {layer: attn_sae.to(device) for layer, attn_sae in attn_saes.items()}

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
    x=[
        f"F{layer_to}.{latent}, {str_toks[seq]!r} ({seq})"
        for (_, seq, latent) in latent_acts_next_recon.indices
    ],
    y=[
        f"F{layer_from}.{latent}, {str_toks[seq]!r} ({seq})"
        for (_, seq, latent) in latent_acts_prev.indices
    ],
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

r'''
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"># nonzero features (true): 476
# nonzero features (reconstructed): 492
# features alive in one but not both: 150</pre>

<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13223b.html" width="1220" height="1020"></div>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Note - your SAEs should perform worse on reconstructing this data than they did on the previous exercises in this subsection (if measured in terms of the intersection of activating features). My guess is that this is because the induction sequences are more OOD for the distribution which the SAEs were trained on (since they're literally random tokens). Also, it's possible that measuring over a larger batch of data and taking all features that activate on at least some fraction of the total tokens would give us a less noisy picture.

Here's some code which filters down the layer-5 features to ones which are active on every token in the second half of the sequence, and also only looks at the layer-4 features which are active on the first half. Try inspecting individual feature pairs which seem to have large gradients with each other - do they seem like previous token features & induction features respectively?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

# Filter for layer-5 latents which are active on every token in the second half (which induction
# latents should be!)
acts_on_second_half = latent_acts_next_recon.indices[
    latent_acts_next_recon.indices[:, 1] >= seq_len + 1
]
c = Counter(acts_on_second_half[:, 2].tolist())
top_feats = sorted([feat for feat, count in c.items() if count >= seq_len])
print(f"Layer 5 SAE latents which fired on all tokens in the second half: {top_feats}")
mask_next = (
    latent_acts_next_recon.indices[:, 2] == t.tensor(top_feats, device=device)[:, None]
).any(dim=0) & (latent_acts_next_recon.indices[:, 1] >= seq_len + 1)

# Filter the layer-4 axis to only show activations at sequence positions that we expect to be used
# in induction
mask_prev = (latent_acts_prev.indices[:, 1] >= 1) & (latent_acts_prev.indices[:, 1] <= seq_len)

# Filter the y-axis, just to these
px.imshow(
    to_numpy(latent_latent_gradients[mask_next][:, mask_prev]),
    color_continuous_midpoint=0.0,
    color_continuous_scale="RdBu",
    y=[
        f"{str_toks[seq]!r} ({seq}), #{latent:05}"
        for (_, seq, latent) in latent_acts_next_recon.indices[mask_next]
    ],
    x=[
        f"{str_toks[seq]!r} ({seq}), #{latent:05}"
        for (_, seq, latent) in latent_acts_prev.indices[mask_prev]
    ],
    labels={"x": f"From layer {layer_from}", "y": f"To layer {layer_to}"},
    title=f'Gradients between SAE latents in layer {layer_from} and SAE latents in layer {layer_to}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
    width=1800,
    height=500,
).show()

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [st-dropdown[Click to see the expected output]]

r'''
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
Layer 5 SAE latents which fired on all tokens in the second half: [35425, 36126]
</pre>

<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13223c.html" width="1820" height="520"></div>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
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
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Sparse feature circuits

> Note - this section is not complete. Exercises will be added over the next ~month, based on replicating the results from the [Sparse Feature Circuits paper](https://arxiv.org/abs/2403.19647). Exercises will replication of some of the results in the paper, in particular the results on reducing spurious correlations in a linear probe & reducing gender bias (if this turns out to be feasible in exercise form).
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Transcoders

The MLP-layer SAEs we've looked at attempt to represent activations as a sparse linear combination of latent vectors; importantly, they only operate on activations **at a single point in the model**. They don't actually learn to perform the MLP layer's computation, rather they learn to reconstruct the results of that computation. It's very hard to do any weights-based analysis on MLP layers in superposition using standard SAEs, since many latents are highly dense in the neuron basis, meaning the neurons are hard to decompose.

In contrast, **transcoders** take in the activations before the MLP layer (i.e. the possibly-normalized residual stream values) and aim to represent the post-MLP activations of that MLP layer, again as a sparse linear combination of latent vectors. The transcoder terminology is the most common, although these have also been called **input-output SAEs** (because we take the input to some base model layer, and try to learn the output) and **predicting future activations** (for obvious reasons). Note that transcoders aren't technically autoencoders, because they're learning a mapping rather than a reconstruction - however a lot of our intuitions from SAEs carry over to transcoders.

Why might transcoders be an improvement over standard SAEs? Mainly, they offer a much clearer insight into the function of a model's layers. From the [Transcoders LessWrong post](https://www.lesswrong.com/posts/YmkjnWtZGLbHRbzrP/transcoders-enable-fine-grained-interpretable-circuit):

> One of the strong points of transcoders is that they decompose the function of an MLP layer into sparse, independently-varying, and meaningful units (like neurons were originally intended to be before superposition was discovered). This significantly simplifies circuit analysis.
>
> ...
>
> As an analogy, let’s say that we have some complex compiled computer program that we want to understand (_a la_ [Chris Olah’s analogy](https://transformer-circuits.pub/2022/mech-interp-essay/index.html)). SAEs are analogous to a debugger that lets us set breakpoints at various locations in the program and read out variables. On the other hand, transcoders are analogous to a tool for replacing specific subroutines in this program with human-interpretable approximations.

Intuitively it might seem like transcoders are solving a different (more complicated) kind of optimization problem - trying to mimic the MLP's computation rather than just reproduce output - and so they would suffer a performance tradeoff relative to standard SAEs. However, evidence suggests that this might not be the case, and transcoders might offer a pareto improvement over standard SAEs.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We'll start by loading in our transcoders. Note that SAELens doesn't yet support running transcoders with `HookedSAETransformer` models (at time of writing**), so instead we'll be loading in our transcoders as `SAE`s but using them in the way we use regular model hooks (rather than using methods like `run_with_cache_with_saes`). The model we'll be working with has been trained to reconstruct the 8th MLP layer of GPT-2. An important note - we're talking about taking the normalized input to the MLP layer and outputting `mlp_out` (i.e. the values we'll be adding back to the residual stream). So when we talk about pre-MLP and post-MLP values, we mean this, not pre/post activation function!

**If this has changed by the time you're reading this, please send an errata in the [Slack group](https://join.slack.com/t/arena-uk/shared_invite/zt-39iwnhbj4-pMWUvZkkt2wpvaxkvJ0q2rRQ)!
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

hf_repo_id = "callummcdougall/arena-demos-transcoder"
sae_id = "gpt2-small-layer-{layer}-mlp-transcoder-folded-b_dec_out"
gpt2_transcoders = {
    layer: SAE.from_pretrained(
        release=hf_repo_id, sae_id=sae_id.format(layer=layer), device=str(device)
    )[0]
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

r'''
<div style="text-align: left"><embed src="https://info-arena.github.io/ARENA_img/misc/media-1322/13227.html" width="820" height="480"></div>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Next, we've given you a helper function which essentially works the same way as the `run_with_cache_with_saes` method that you might be used to. We recommend you read through this function, and understand how the transcoder is being used.
'''

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
    assert all(
        transcoder.cfg.hook_name.endswith("ln2.hook_normalized") for transcoder in transcoders
    )
    input_hook_names = [transcoder.cfg.hook_name for transcoder in transcoders]
    output_hook_names = [
        transcoder.cfg.hook_name.replace("ln2.hook_normalized", "hook_mlp_out")
        for transcoder in transcoders
    ]

    # Hook function at transcoder input: computes its output (and all intermediate values e.g.
    # latent activations)
    def hook_transcoder_input(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        _, cache = transcoders[transcoder_idx].run_with_cache(activations)
        hook.ctx["cache"] = cache

    # Hook function at transcoder output: replaces activations with transcoder output
    def hook_transcoder_output(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        cache: ActivationCache = model.hook_dict[transcoders[transcoder_idx].cfg.hook_name].ctx[
            "cache"
        ]
        return cache["hook_sae_output"]

    # Get a list of all fwd hooks (only including the output hooks if use_error_term=False)
    fwd_hooks = []
    for i in range(len(transcoders)):
        fwd_hooks.append((input_hook_names[i], partial(hook_transcoder_input, transcoder_idx=i)))
        if not use_error_term:
            fwd_hooks.append(
                (output_hook_names[i], partial(hook_transcoder_output, transcoder_idx=i))
            )

    # Fwd pass on model, triggering all hook functions
    with model.hooks(fwd_hooks=fwd_hooks):
        _, model_cache = model.run_with_cache(tokens)

    # Return union of both caches (we rename the transcoder hooks using the same convention as
    # regular SAE hooks)
    all_transcoders_cache_dict = {}
    for i, transcoder in enumerate(transcoders):
        transcoder_cache = model.hook_dict[input_hook_names[i]].ctx.pop("cache")
        transcoder_cache_dict = {
            f"{transcoder.cfg.hook_name}.{k}": v for k, v in transcoder_cache.items()
        }
        all_transcoders_cache_dict.update(transcoder_cache_dict)

    return ActivationCache(
        cache_dict=model_cache.cache_dict | all_transcoders_cache_dict, model=model
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Lastly, we've given you the functions which you should already have encountered in the earlier exercise sets, when we were replicating SAE dashboards (if you've not done these exercises yet, we strongly recommend them!). The only difference is that we use the `run_with_cache_with_transcoder` function in place of `model.run_with_cache_with_saes`.
'''

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
            unique_indices = t.cat(
                (unique_indices, t.tensor([[rows[0], cols[0]]], device=x.device))
            )
            is_overlapping_mask = (rows == rows[0]) & ((cols - cols[0]).abs() <= buffer)
            rows = rows[~is_overlapping_mask]
            cols = cols[~is_overlapping_mask]
        return unique_indices

    return t.stack((rows, cols), dim=1)[:k]


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, "k *buffer_x2_plus1"]:
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
            "".join(
                [
                    f"[b u green]{str_tok}[/]" if i == seq_pos else str_tok
                    for i, str_tok in enumerate(str_toks)
                ]
            )
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

r'''
Let's pick latent 1, and compare our results to the neuronpedia dashboard (note that we do have neuronpedia dashboards for this model, even though it's not in SAELens yet).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

r'''
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
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Pullback & de-embeddings

In the exercises on latent-latent gradients at the start of this section, we saw that it could be quite difficult to compute how any 2 latents in different layers interact with each other. In fact, we could only compute gradients between latents which were both alive in the same forward pass. One way we might have liked to deal with this is by just taking the dot product of the "writing vector" of one latent with the "reading vector" of another. For example, suppose our SAEs were trained on post-ReLU MLP activations, then we could compute: `W_dec[:, f1] @ W_out[layer1] @ W_in[layer2] @ W_enc[f2, :]` (where `f1` and `f2` are our earlier and later latent indices, `layer1` and `layer2` are our SAE layers, and `W_in`, `W_out` are the MLP input & output weight matrices for all layers). To make sense of this formula: the term `W_dec[:, f1] @ W_out[layer1]` is the "writing vector" being added to the residual stream by the first (earlier) latent, and we would take the dot product of this with `W_in[layer2] @ W_enc[f2, :]` to compute the activation of the second (later) latent. However, one slight frustration is that we're ignoring the later MLP layer's ReLU function (remember that the SAEs are reconstructing the post-ReLU activations, not pre-ReLU). This might seem like a minor point, but it actually gets to a core part of the limitation of standard SAEs when trained on layers which perform computation - **the SAEs are reconstructing a snapshot in the model, but they're not helping us get insight into the layer's actual computation process**.

How do transcoders help us here? Well, since transcoders sit around the entire MLP layer (nonlinearity and all), we can literally compute the dot product between the "writing vector" and a downstream "reading vector" to figure out whether any given latent causes another one to be activated (ignoring layernorm). To make a few definitions:

- The **pullback** of some later latent is $p = (W_{dec})^T f_{later}$, i.e. the dot product of the later latent vector (reading weight) with all the decoder weights (writing weights) of earlier latents.
- The **de-embedding** is a special case: $d = W_E f_{later}$, i.e. instead of asking "which earlier transcoder latents activate some later latent?" we ask "which tokens maximally activate some later latent?".

Note that we can in principle compute both of these quantities for regular MLP SAEs. But they wouldn't be as accurate to the model's actual computation, and so you couldn't draw as many strong conclusions from them.

To complete our circuit picture of (embeddings -> transcoders -> unembeddings), it's worth noting that we can compute the logit lens for a transcoder latent in exactly the same way as regular SAEs: just take the dot product of the transcoder decoder vector with the unembedding matrix. Since this has basically the exact same justification & interpretation as for regular SAEs, we don't need to invent a new term for it, so we'll just keep calling it the logit lens!
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - compute de-embedding

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 10-15 minutes on this exercise.
> ```

In the cell below, you should compute the de-embedding for this latent (i.e. which tokens cause this latent to fire most strongly). You can use the logit lens function as a guide (which we've provided, from where it was used in the earlier exercises).
'''

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
if MAIN:
    print(f"Top logits for transcoder latent {latent_idx}:")
    show_top_logits(gpt2, gpt2_transcoder, latent_idx=latent_idx)
# END HIDE


def show_top_deembeddings(
    model: HookedSAETransformer, sae: SAE, latent_idx: int, k: int = 10
) -> None:
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
if MAIN:
    print(f"\nTop de-embeddings for transcoder latent {latent_idx}:")
    show_top_deembeddings(gpt2, gpt2_transcoder, latent_idx=latent_idx)
    tests.test_show_top_deembeddings(show_top_deembeddings, gpt2, gpt2_transcoder)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html]

r'''
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
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
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
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
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
'''

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

    mlp_output = model.blocks[0].mlp(
        model.blocks[0].ln2(W_E)
    )  # shape [batch=d_vocab, seq_len=1, d_model]

    W_E_ext = (W_E + mlp_output).squeeze()
    return (W_E_ext - W_E_ext.mean(dim=-1, keepdim=True)) / W_E_ext.std(dim=-1, keepdim=True)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_create_extended_embedding(create_extended_embedding, gpt2)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Once you've passed those tests, try rewriting `show_top_deembeddings` to use the extended embedding. Do the results look better? (Hint - they should!)

Note - don't worry if the magnitude of the results seems surprisingly large. Remember that a normalization step is applied pre-MLP, so the actual activations will be smaller than is suggested by the values in the table you'll generate.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def show_top_deembeddings_extended(
    model: HookedSAETransformer, sae: SAE, latent_idx: int, k: int = 10
) -> None:
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
if MAIN:
    print(f"Top de-embeddings (extended) for transcoder latent {latent_idx}:")
    show_top_deembeddings_extended(gpt2, gpt2_transcoder, latent_idx=latent_idx)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: [soln,st]
# ! TAGS: [html,st-dropdown[Click to see the expected output]]

r'''
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
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
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
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

