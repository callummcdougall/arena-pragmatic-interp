import torch as t
from torch import Tensor


# ==================================================
# SECTION 2 TESTS (transcoders)
# ==================================================


def test_show_top_deembeddings(show_top_deembeddings, model, transcoder):
    """Test the show_top_deembeddings function by checking its output contains expected tokens."""
    import contextlib
    import io

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
        f"Expected shape {model.W_E.shape} but got {W_E_ext.shape}. "
        "Did you forget to include the MLP output?"
    )

    # Check it's been centered & scaled
    mean = W_E_ext.mean(dim=-1)
    std = W_E_ext.std(dim=-1)
    assert t.allclose(mean, t.zeros_like(mean), atol=1e-3), (
        "Extended embedding should be centered (zero mean along d_model). "
        "Did you forget to normalize?"
    )
    assert t.allclose(std, t.ones_like(std), atol=0.1), (
        "Extended embedding should be scaled (unit std along d_model). "
        "Did you forget to normalize?"
    )

    print("All tests in `test_create_extended_embedding` passed!")


# ==================================================
# SECTION 3 TESTS (attribution graphs)
# ==================================================


def test_compute_salient_logits(compute_salient_logits_fn):
    """Test salient logit selection."""
    t.manual_seed(42)
    d_vocab, d_model = 1000, 64
    logits = t.randn(d_vocab)
    W_U = t.randn(d_model, d_vocab)

    idx, probs, vecs = compute_salient_logits_fn(logits, W_U, max_n_logits=10, desired_logit_prob=0.95)

    # Basic shapes
    assert idx.ndim == 1 and probs.ndim == 1 and vecs.ndim == 2, (
        f"Expected 1D idx, 1D probs, 2D vecs but got {idx.ndim}D, {probs.ndim}D, {vecs.ndim}D"
    )
    assert len(idx) == len(probs) == vecs.shape[0], (
        f"Lengths don't match: idx={len(idx)}, probs={len(probs)}, vecs={vecs.shape[0]}"
    )
    assert vecs.shape[1] == d_model, f"Expected vecs dim 1 = {d_model}, got {vecs.shape[1]}"

    # Cumulative probability check
    assert probs.sum() >= 0.95, f"Cumulative prob {probs.sum():.3f} < 0.95"

    # At most max_n_logits
    assert len(idx) <= 10, f"Expected at most 10 logits, got {len(idx)}"

    # Check demeaning
    raw_cols = W_U[:, idx].T
    global_mean = W_U.mean(dim=-1)
    expected_demeaned = raw_cols - global_mean.unsqueeze(0)
    t.testing.assert_close(vecs, expected_demeaned, atol=1e-4, rtol=1e-4)

    # Probabilities should be in descending order
    assert t.all(probs[:-1] >= probs[1:]), "Probabilities should be in descending order"

    print("All tests in `test_compute_salient_logits` passed!")


def test_normalize_matrix(normalize_matrix_fn):
    """Test matrix normalization."""
    M = t.tensor([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    normed = normalize_matrix_fn(M)

    # Row 0: abs = [1, 2, 3], sum = 6
    t.testing.assert_close(normed[0], t.tensor([1 / 6, 2 / 6, 3 / 6]))

    # Row 1: all zeros, should not be NaN
    assert not normed[1].isnan().any(), "Row of zeros should not produce NaN"

    # Row 2: abs = [1, 1, 0], sum = 2
    t.testing.assert_close(normed[2], t.tensor([0.5, 0.5, 0.0]))

    # All values should be non-negative
    assert (normed >= 0).all(), "Normalized matrix should have non-negative entries"

    print("All tests in `test_normalize_matrix` passed!")


def test_compute_influence(compute_influence_fn, normalize_matrix_fn):
    """Test influence computation with a simple causal graph."""
    # Simple 3-layer graph: node 0 -> node 1 -> node 2 (logit)
    A = t.zeros(3, 3)
    A[1, 0] = 0.5  # node 0 feeds into node 1
    A[2, 1] = 0.8  # node 1 feeds into node 2

    logit_weights = t.tensor([0.0, 0.0, 1.0])

    normed_A = normalize_matrix_fn(A)
    influence = compute_influence_fn(normed_A, logit_weights)

    # Node 1 should have direct influence
    assert influence[1] > 0, "Node 1 should influence the logit (direct connection)"
    # Node 0 should have indirect influence
    assert influence[0] > 0, "Node 0 should have indirect influence via node 1"
    # The iteration should terminate quickly (nilpotent)
    print(f"Influence values: {influence.tolist()}")

    # Test nilpotent convergence: strictly lower-triangular matrix
    n = 50
    A_big = t.tril(t.rand(n, n), diagonal=-1)
    A_big = normalize_matrix_fn(A_big)
    w = t.zeros(n)
    w[-1] = 1.0
    influence_big = compute_influence_fn(A_big, w, max_iter=n)
    assert influence_big is not None, "Should converge within n iterations"

    print("All tests in `test_compute_influence` passed!")


def test_prune_graph(prune_graph_fn, graph):
    """Test graph pruning produces valid results."""
    from part42_sae_circuits.utils import find_threshold

    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)
    n_total = graph.adjacency_matrix.shape[0]

    node_mask, edge_mask = prune_graph_fn(graph, node_threshold=0.8, edge_threshold=0.98)

    # Logit and embed nodes should always be kept
    assert node_mask[-n_logits:].all(), "Logit nodes must always be kept"
    assert node_mask[-n_logits - n_tokens : -n_logits].all(), "Embed nodes must always be kept"

    # Should have fewer nodes after pruning
    n_kept = node_mask.sum().item()
    assert n_kept < n_total, f"Pruning should remove some nodes ({n_kept}/{n_total} kept)"
    assert n_kept > n_logits + n_tokens, "Pruning removed all feature nodes!"

    # Every surviving feature node should have at least one incoming and one outgoing edge
    for i in range(n_features):
        if node_mask[i]:
            assert edge_mask[i].any(), f"Kept feature node {i} has no incoming edges"
            assert edge_mask[:, i].any(), f"Kept feature node {i} has no outgoing edges"

    print(f"Pruning kept {n_kept}/{n_total} nodes ({n_kept / n_total:.1%})")
    print(f"Pruning kept {edge_mask.sum().item()} edges ({edge_mask.float().mean():.4%})")
    print("All tests in `test_prune_graph` passed!")


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

    assert abl_prob < orig_prob, (
        f"Ablation should reduce '{target_token_str}' prob: {orig_prob:.4f} -> {abl_prob:.4f}"
    )
    print(f"'{target_token_str}' probability: {orig_prob:.4f} -> {abl_prob:.4f} (reduced by {orig_prob - abl_prob:.4f})")
    print("All tests in `test_ablation_reduces_target` passed!")
