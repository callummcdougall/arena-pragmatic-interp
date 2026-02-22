"""Utility functions for SAE dashboards."""

import getpass
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import TypeAlias
from google3.learning.deepmind.jax.typing import typing as jt
from google3.learning.deepmind.research.lmi import toolkit as tk
from google3.learning.deepmind.research.lmi import types
from google3.learning.deepmind.research.lmi.models import gemax
from google3.learning.deepmind.research.lmi.models import models
from google3.learning.deepmind.research.lmi.projects.sae.analysis import toolkit as atk
from google3.learning.deepmind.tokenizers import text_tokenizers
from google3.pyglib import gfile

FEATURE_TABLES_PRECISION = 4
LOGIT_TABLES_PRECISION = 3
SEQ_ACT_PRECISION = 4
SEQ_LOGIT_PRECISION = 4
AUTOINTERP_SCORE_PRECISION = 2


# pylint: disable=g-long-ternary,invalid-name

NestedStrings: TypeAlias = str | list['NestedStrings']
NestedInts: TypeAlias = int | list['NestedInts']
NestedFloats: TypeAlias = float | list['NestedFloats']


def load_template(
 filepath: str,
 citc_name: str | None = 'sae-dashboards',
) -> str:
 """Loads either the CSS, JS or HTML templates from either local CITC or HEAD."""
 path_segment = (
 f'cloud/{getpass.getuser()}/{citc_name}'
 if citc_name
 else 'files/head/depot'
 )
 path_full = f'/google_src/{path_segment}/google3/{filepath}'
 with gfile.Open(path_full, 'r') as f:
 return f.read()


@jt.typed
def get_w_in(
 model: models.EasyModel, layer: int | None, normalized: bool = False
) -> jt.Float[jt.ArrayT, 'd_sae_in d_mlp']:
 """Returns MLP input weight matrix for a model (Gemax or EasyModel).

 Args:
 model: The model to use.
 layer: The layer to use. If None, then it concatenates across all layers,
 i.e. so the left-input to the matrix is the entire concatenated resid.
 normalized: If True, then normalizes the weights.

 Returns:
 The MLP input weight matrix.
 """
 layers = range(model.shapes.num_layers) if layer is None else [layer]

 w_in_list = []
 for i in layers:
 if isinstance(model, gemax.GemaxEasyModel):
 w_in_key = f'transformer/layer_{i}/mlp/gating_einsum'
 w_in = model.params[w_in_key]['w'][0].T # (d_model, d_mlp)
 else:
 w_in_key = f'model/h{i}/mlp/linear'
 w_in = model.params[w_in_key]['w'] # (d_model, d_mlp)
 w_in_list.append(w_in)

 w_in = jnp.concatenate(w_in_list, axis=0)
 if normalized:
 w_in = w_in / jnp.linalg.norm(w_in, axis=0)
 return w_in


@jt.typed
def get_w_out(
 model: models.EasyModel, layer: int, normalized: bool = False
) -> jt.Float[jt.ArrayT, 'd_mlp d_model']:
 """Returns MLP output weight matrix for a model (Gemax or EasyModel)."""
 if isinstance(model, gemax.GemaxEasyModel):
 raise NotImplementedError('Check on how to do this, for Gemax')
 else:
 w_out_key = f'model/h{layer}/mlp/linear_1'
 w_out = model.params[w_out_key]['w'] # (d_mlp, d_model)

 if normalized:
 w_out = w_out / jnp.linalg.norm(w_out, axis=-1, keepdims=True)
 return w_out


@jt.typed
def get_w_u(
 model: models.EasyModel,
 centered: bool = False,
 tokens: jt.Int[jt.ArrayT, '...'] | None = None,
) -> jt.Float[jt.ArrayT, 'd_model d_vocab']:
 """Returns the unembedding matrix for a model (Gemax or EasyModel)."""
 if isinstance(model, gemax.GemaxEasyModel):
 if model.materials.gemax_model_type != types.GemaxModelType.GEMMA_V3:
 raise NotImplementedError('Currently only supported for Gemma V3')
 w_u = model.params['transformer/embedder']['input_embedding'].T
 w_u *= (1.0 + model.params['transformer/final_norm']['scale'])[:, None]
 else:
 if 'output_weights' in model.params['embedding_layer']:
 w_u = model.params['embedding_layer']['output_weights']
 else:
 # TODO(mcdougallc) - handle this more robustly
 if 'input_embedding' not in model.params['embedding_layer']:
 raise ValueError(
 "Didn't find input_embedding in"
 f" {model.params['embedding_layer'].keys()=}, need to handle these"
 " better in 'get_w_u' function"
 )
 w_u = model.params['embedding_layer']['input_embedding'].T

 if centered:
 w_u = w_u - jnp.mean(w_u, axis=-1, keepdims=True)

 if tokens is not None:
 w_u = w_u[:, tokens]

 return w_u


class UnifiedSae:
 """Unified SAE object.

 This is a wrapper around SAEs, which allows you to treat all multi-layer SAEs
 the same (in particular we can treat CLTs and PLTs the same, with functions
 like `get_w_dec` returning objects of the same shape, despite the fact that
 CLTs and PLTs have differently-structured decoder weights).

 This is the format that sae vis uses, as well as the attribution graph
 codebase.
 """

 def __init__(
 self, sae: atk.EasySae, extra_metadata: dict[str, Any] | None = None
 ):
 self.sae = sae
 self.extra_metadata = extra_metadata
 # Define some useful properties for quick access
 self.mesh = sae.module.mesh
 assert self.mesh is not None, 'Need to use mesh for UnifiedSAE'
 self.layer = sae.layer
 self.n_layers = sae.module.n_layers
 self.site = sae.site
 self.target_site = sae.target_site
 self.d_enc = self.sae.module.d_enc
 self.d_dec = self.sae.module.d_dec
 self.d_in = self.sae.module.d_in
 self.coder_mode = self.sae.module.coder_mode
 self.use_affine_skip_connection = self.sae.module.use_affine_skip_connection

 @jt.typed
 def get_local_feature(self, feature_id: int) -> int:
 """Returns the local feature id for a given (global) feature id."""
 if self.coder_mode.is_multi_layer:
 d_enc_per_layer = self.d_enc // self.n_layers
 return feature_id % d_enc_per_layer
 else:
 return feature_id

 @jt.typed
 def get_local_features(self, feature_ids: list[int]) -> list[int]:
 """Returns the local feature ids for given (global) feature ids."""
 return [self.get_local_feature(f) for f in feature_ids]

 @jt.typed
 def get_w_dec(
 self,
 layer: int | None = None,
 sum_over_output_layers: bool = False,
 normalized: bool = False,
 feature_ids: jt.Int[jt.ArrayT, 'features'] | list[int] | None = None,
 ) -> jax.Array:
 """Get the cross-layer decoder weight matrix, with optional processing.

 Args:
 layer: The layer to get the weight matrix for.
 sum_over_output_layers: If True, then sums over the output layers.
 normalized: If True, then normalizes the weights.
 feature_ids: If supplied, then we index into the weight matrix along the
 SAE hidden dim using these feature ids.

 Returns:
 The encoder weight matrix for the specified layer.

 Note - the `layer` argument ONLY applies to the input dimensions, i.e. if we
 specify a layer then by default we return the matrix which maps from one
 layer's activations to the full cross-layer MLP outputs.
 """
 weight = self.sae.params['params']['decoder']['kernel']

 # Get only the features from a particular layer
 if layer is not None:
 if self.coder_mode.is_multi_layer:
 weight = weight[layer]

 # Sum over output layers, if requested. Note that this does nothing for
 # PLT decoders since they are already layer-to-layer, not cross layer.
 if sum_over_output_layers and self.coder_mode.is_weakly_causal:
 weight = weight.sum(axis=-2)

 # Normalize over decoder output dimensions. CLTs have 2 output dims (unless
 # we summed over output layers, in which case it's just 1), and all other
 # types of SAE have 1 output dim.
 if normalized:
 axes = (-1, -2) if weight.ndim == 4 else -1
 # We have to add eps to the denominator because e.g. if this is a CLT and
 # we've indexed a specific layer, then some layer output blocks will be
 # zero.
 weight /= jnp.linalg.norm(weight, axis=axes, keepdims=True) + 1e-8

 # Index into feature dimension. If this is a multi-layer model and we've not
 # selected a particular layer then we assume the indices are global, i.e.
 # they refer to positions in the (layer * d_enc_per_layer) flattened feature
 # space. Otherwise, the feature dimension is 0.
 if feature_ids is not None:
 feature_ids = jnp.array(feature_ids)
 if self.coder_mode.is_multi_layer and layer is None:
 layers = feature_ids // (self.d_enc // self.n_layers)
 ids = feature_ids % (self.d_enc // self.n_layers)
 weight = weight[layers, ids]
 else:
 if feature_ids.max() >= weight.shape[0]:
 raise ValueError('Max feature >= n_features (acc. to weight shape).')
 weight = jnp.take(weight, feature_ids, axis=0)

 # To treat PLTs and CLTs the same, we need to pad the PLT weight with zeros
 # (unless we summed over output layers, in which case they're already
 # consistent with each other). We split into 3 separate cases.
 # (1) layer != None, then we 1-hot pad along the layer dimension
 # (2) layer == None & feature_ids == None, then we pad to turn it block-diag
 # so each layer only writes to its own layer
 # (3) layer == None & feature_ids != None, then we pad to turn it block-diag
 # so each feature only writes to the layer it's in
 if (
 self.coder_mode.is_multi_layer
 and not self.coder_mode.is_weakly_causal
 and not sum_over_output_layers
 ):
 weight_expanded = jnp.expand_dims(weight, axis=-2)
 # (1) We're already choosing a particular layer of features, so we 1-hot
 # pad the PLT decoder so that it only writes to that layer.
 if layer is not None:
 weight = (
 weight_expanded
 * jax.nn.one_hot(layer, self.n_layers, dtype=weight.dtype)[:, None]
 )
 # (2) We're using all layers, so we block-diagonally pad the PLT decoder
 # so that each layer only writes to its own layer.
 elif feature_ids is None:
 weight = weight_expanded * jnp.eye(
 self.n_layers, dtype=weight.dtype
 ).reshape((self.n_layers, 1, self.n_layers, 1))
 # (3) We're using all layers but a specific feature subset; this is same
 # as case (2) except we need each feature to write only to the layer it's
 # in (which requires more careful indexing).
 else:
 layers = feature_ids // (self.d_enc // self.n_layers)
 weight = weight_expanded * jnp.expand_dims(
 jax.nn.one_hot(layers, self.n_layers), axis=-1
 )

 return weight

 @jt.typed
 def get_w_enc(
 self,
 layer: int | None,
 feature_ids: jt.Int[jt.ArrayT, 'features'] | list[int] | None = None,
 normalized: bool = False,
 ) -> jax.Array:
 """Get the cross-layer decoder weight matrix, with optional processing.

 Args:
 layer: The layer to get the weight matrix for.
 feature_ids: If supplied, then we index into the weight matrix along the
 SAE hidden dim using these feature ids.
 normalized: If True, then normalizes the weights.

 Returns:
 The encoder weight matrix for the specified layer.

 Note - the `layer` argument applies to BOTH the input and output dimensions,
 i.e. the returned weight matrix will be just for a single layer.
 """
 weight = self.sae.params['params']['encoder']['kernel']

 # Get only the features from a particular layer
 if layer is not None:
 if weight.ndim == 3:
 weight = weight[layer]

 # Normalize over output dimensions, which is always just the -1th dim.
 if normalized:
 weight /= jnp.linalg.norm(weight, axis=-1, keepdims=True)

 # Index into feature dimension. If this is a multi-layer model and we've not
 # selected a particular layer then we assume the indices are global, i.e.
 # they refer to positions in the (layer * d_enc_per_layer) flattened feature
 # space. Otherwise, the feature dimension is 1.
 if feature_ids is not None:
 feature_ids = jnp.array(feature_ids)
 if self.coder_mode.is_multi_layer and layer is None:
 layers = feature_ids // (self.d_enc // self.n_layers)
 ids = feature_ids % (self.d_enc // self.n_layers)
 weight = weight[layers, :, ids]
 else:
 if feature_ids.max() >= weight.shape[-1]:
 raise ValueError('Max feature >= n_features (acc. to weight shape).')
 weight = jnp.take(weight, feature_ids, axis=-1)

 return weight

 def get_w_skip(self, layer: int) -> jax.Array:
 """Get the skip connection weight matrix."""
 weight = self.sae.params['params']['affine_skip_connection']['kernel']
 if layer is not None:
 assert weight.ndim == 3, 'Only valid for multi-layer models'
 weight = weight[layer]
 return weight

 @jt.typed
 def get_w_dec_resid_directions(
 self,
 w_dec_directions: jt.Float[jt.ArrayT, '... d_sae_out'],
 model: models.EasyModel,
 ) -> jt.Float[jt.ArrayT, '... d_model']:
 """Converts directions to residual directions.

 If the model is a residual stream or MLP output model then this is trivial,
 but if it's an MLP activations model then we need to map the directions
 through the MLP output weights first.

 Args:
 w_dec_directions: The batched directions of the feature activations.
 model: The model to use.

 Returns:
 The batched residual directions.
 """
 if self.target_site == tk.Site.MLP_POST_ACTIVATION or (
 self.target_site is None and self.site == tk.Site.MLP_POST_ACTIVATION
 ):
 if self.layer is None:
 return w_dec_directions @ jnp.concat(
 [get_w_out(model, layer) for layer in range(self.n_layers)]
 )
 else:
 return w_dec_directions @ get_w_out(model, self.layer)

 else:
 return w_dec_directions

 def get_output_and_acts(
 self,
 activations: jt.Float[jt.ArrayT, '... ActivationsDim'],
 ) -> tuple[
 jt.Float[jt.ArrayT, '... TargetActivationsDim'],
 jt.Float[jt.ArrayT, '... SaeDim'],
 ]:
 """Run the CLT, getting the outputs and activations."""
 _, sae_acts = self.sae(activations)
 return sae_acts.outputs, sae_acts.activations.todense()


def replace_control_tokens(s: str) -> str:
 """Replaces control tokens with more readable versions."""
 return s.replace('<ctrl99>', '<start_of_turn>').replace(
 '<ctrl100>', '<end_of_turn>'
 )


def process_str_tok(s: str, html: bool = False) -> str:
 """Processes a string token to make it more readable."""
 s = s.replace('\n', '⏎')
 s = replace_control_tokens(s)
 if html:
 s = s.replace('<', '&lt;').replace('>', '&gt;')
 return s


@jt.typed
def to_str(
 tokens: jt.Int[jt.ArrayT, '...'] | NestedInts,
 tokenizer: text_tokenizers.Tokenizer,
 html: bool = False,
) -> NestedStrings:
 """Helper function to convert tokens to strings in a batched way.

 Args:
 tokens: The tokens to convert.
 tokenizer: The tokenizer to use.
 html: If True, converts weird characters to HTML representations.

 Returns:
 A string or list of strings per token.
 """
 if not isinstance(tokens, np.ndarray):
 tokens = np.array(tokens)

 # TODO(mcdougallc) - why is this needed?
 max_vocab_size = getattr(tokenizer, 'used_vocab_size', tokenizer.vocab_size)
 if tokens.max() >= max_vocab_size:
 print(f'Warning: {max_vocab_size=}, {tokens.max()=}')
 tokens[tokens >= max_vocab_size] = tokenizer.bos_token

 # TODO(mcdougallc) - why is this needed? I tracked this bug down in ipdb
 tokens[tokens == 150] = tokenizer.bos_token

 if tokens.size == 0:
 return tokenizer.to_string(np.array([tokens.item()]))

 tokens_shape = tokens.shape

 str_toks = tokenizer.to_string_list(tokens.flatten())
 str_toks = [process_str_tok(s, html=html) for s in str_toks]

 str_toks = np.array(str_toks).reshape(tokens_shape).tolist()
 return str_toks


# @jt.typed
# def compute_correlations(
# dataset_acts: analyzer_types.DatasetActivations,
# feature_id: int,
# feature_list: list[int],
# ) -> list[float]:
# """Computes correlations between feature activations."""
# all_features = [feature_id] + feature_list
# df = dataset_acts.activations_df[
# dataset_acts.activations_df['feature_id'].isin(all_features)
# ]
# idx = pd.MultiIndex.from_product(
# [range(s) for s in dataset_acts.tokens.shape],
# names=['seq_id', 'position'],
# )
# df = df.pivot_table(
# index=['seq_id', 'position'],
# columns='feature_id',
# values='activation',
# fill_value=0,
# ).reindex(idx, fill_value=0)
# df = df.reindex(columns=[feature_id] + feature_list, fill_value=0)
# return [
# df[feature_id].corr(df[f]).item()
# if not jnp.isnan(df[feature_id].corr(df[f]))
# else 0
# for f in feature_list
# ]


@jt.typed
def topk(
 arr: jt.ArrayT, k: int, largest: bool = True, as_list: bool = False
) -> (
 tuple[jt.Float[jt.ArrayT, '...'], jt.Int[jt.ArrayT, '...']]
 | tuple[NestedFloats, NestedInts]
):
 """Helper function for topk, which can get minimum values instead."""

 if largest:
 values, indices = jax.lax.top_k(arr, k=k)
 else:
 values, indices = jax.lax.top_k(-arr, k=k)
 values = -values

 if as_list:
 values = values.tolist()
 indices = indices.tolist()

 return values, indices


def merge_lists(*lists) -> list[Any]:
 """Merges a list of lists into a single list."""
 merged_list = []
 for sublist in lists:
 merged_list.extend(sublist)
 return merged_list


def get_ticks(min_value: float, max_value: float) -> list[float] | list[int]:
 """Returns a list of ticks for a histogram, given a min and max value."""
 if min_value > max_value:
 min_value, max_value = max_value, min_value
 if min_value > 0:
 min_value = 0

 span = max_value - min_value
 power = np.floor(np.log10(span / 4))
 scale = 10**power
 for interval in [5, 4, 3, 2, 1]:
 tick_step = interval * scale
 ticks = np.arange(
 np.floor(min_value / tick_step) * tick_step,
 np.ceil(max_value / tick_step) * tick_step + tick_step,
 tick_step,
 )
 ticks = ticks[(ticks > min_value) & (ticks < max_value)]
 if len(ticks) >= 4:
 break
 if 0.0 not in ticks:
 ticks = np.append(ticks, 0.0)

 if power < 0:
 ticks = [round(t, -int(power.item())) for t in ticks.tolist()]
 else:
 ticks = [int(t) for t in ticks.tolist()]

 return sorted(set(ticks))


def hist_from_data(
 data: np.ndarray,
 n_bins: int,
 title: str | None = None,
 extra_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
 """Gets histogram data from a tensor of data."""
 if data.size == 0:
 return {}

 # Get tick values from our helper function
 max_val = data.max().item()
 min_val = data.min().item()
 tick_vals = get_ticks(min_val, max_val)
 tickangle = 45 if len(tick_vals) > 7 else 0

 # Divide range up into 40 bins & calculate height of each bin
 bin_edges = np.linspace(min_val, max_val, n_bins + 1)
 bar_heights = np.histogram(data, n_bins, (min_val, max_val))[0]
 bar_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])

 return {
 'y': bar_heights.tolist(),
 'x': [round(x, 5) for x in bar_values.tolist()],
 'ticks': tick_vals,
 'tickangle': tickangle,
 'title': title,
 'extra_data': extra_data,
 }