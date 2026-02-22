import contextlib
import io
from typing import Callable

import torch as t
from torch import Tensor

# ==================================================
# SECTION 2 TESTS (transcoders)
# ==================================================


def test_show_top_deembeddings(show_top_deembeddings, model, transcoder):
    """Test the show_top_deembeddings function by checking its output contains expected tokens."""

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        show_top_deembeddings(model, transcoder, latent_idx=1)
    output = f.getvalue()
    expected_tokens = ["liga", "GAME"]
    for token in expected_tokens:
        assert token in output, f"Expected '{token}' in output but got:\n{output[:500]}"
    print("All tests in `test_show_top_deembeddings` passed!")


def test_create_extended_embedding(create_extended_embedding, model):
    """Test the extended embedding computation."""
    W_E_ext = create_extended_embedding(model)

    # Check shape
    assert W_E_ext.shape == model.W_E.shape, (
        f"Expected shape {model.W_E.shape} but got {W_E_ext.shape}. Did you forget to include the MLP output?"
    )

    # Check it's been centered & scaled
    mean = W_E_ext.mean(dim=-1)
    std = W_E_ext.std(dim=-1)
    assert t.allclose(mean, t.zeros_like(mean), atol=1e-3), (
        "Extended embedding should be centered (zero mean along d_model). Did you forget to normalize?"
    )
    assert t.allclose(std, t.ones_like(std), atol=0.1), (
        "Extended embedding should be scaled (unit std along d_model). Did you forget to normalize?"
    )

    print("All tests in `test_create_extended_embedding` passed!")


# ==================================================
# SECTION 3 TESTS (attribution graphs)
# ==================================================


def test_normalize_matrix_section3(normalize_matrix_fn):
    """Test matrix normalization (section 3 version using abs values)."""
    M = t.tensor([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    normed = normalize_matrix_fn(M)

    # Row 0: abs = [1, 2, 3], sum = 6
    t.testing.assert_close(normed[0], t.tensor([1 / 6, 2 / 6, 3 / 6]))

    # Row 1: all zeros, should not be NaN
    assert not normed[1].isnan().any(), "Row of zeros should not produce NaN"

    # Row 2: abs = [1, 1, 0], sum = 2
    t.testing.assert_close(normed[2], t.tensor([0.5, 0.5, 0.0]))

    # All values should be non-negative (we take absolute values)
    assert (normed >= 0).all(), "Normalized matrix should have non-negative entries"

    print("All tests in `test_normalize_matrix_section3` passed!")


def test_compute_influence_section3(compute_influence_fn):
    """Test influence computation with a simple causal graph (section 3 version)."""
    # Chain: embed(0) -> feat(1) -> feat(2) -> logit(3)
    # A[target, source] convention
    n = 4
    A = t.zeros(n, n)
    A[1, 0] = 1.0  # feat reads from embed
    A[2, 1] = 1.0  # feat reads from feat
    A[3, 2] = 1.0  # logit reads from feat

    logit_weights = t.tensor([0.8])

    influence = compute_influence_fn(A, logit_weights, n_layers=3)

    # All upstream nodes should have nonzero influence
    assert influence[2] > 0, "Node 2 (direct to logit) should have positive influence"
    assert influence[1] > 0, "Node 1 (indirect) should have positive influence"
    assert influence[0] > 0, "Node 0 (indirect) should have positive influence"

    print(f"Influence values: {influence.tolist()}")
    print("All tests in `test_compute_influence_section3` passed!")


# ==================================================
# SECTION 4 TESTS (interventions)
# ==================================================


def test_ablation_reduces_target(model, prompt, features_to_ablate, target_token_str):
    """Test that ablating features reduces the probability of a target token."""
    interventions = [(*f, 0.0) for f in features_to_ablate]

    with t.inference_mode():
        orig_logits, _ = model.feature_intervention(prompt, [])
        abl_logits, _ = model.feature_intervention(prompt, interventions)

    target_id = model.tokenizer.encode(target_token_str, add_special_tokens=False)[0]
    orig_prob = orig_logits[0, -1].float().softmax(-1)[target_id].item()
    abl_prob = abl_logits[0, -1].float().softmax(-1)[target_id].item()

    assert abl_prob < orig_prob, f"Ablation should reduce '{target_token_str}' prob: {orig_prob:.4f} -> {abl_prob:.4f}"
    print(
        f"'{target_token_str}' probability: {orig_prob:.4f} -> {abl_prob:.4f} (reduced by {orig_prob - abl_prob:.4f})"
    )
    print("All tests in `test_ablation_reduces_target` passed!")

