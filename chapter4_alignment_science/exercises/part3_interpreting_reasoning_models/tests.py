import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import stats

# Make sure exercises are in the path
if str(exercises_dir := Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(exercises_dir))


def test_compute_head_kurtosis(compute_head_kurtosis: Callable):
    """Test the compute_head_kurtosis function."""

    # Test 1: Normal case with spiky distribution (high kurtosis)
    spiky_scores = np.array([0.1, 0.1, 0.1, 5.0, 0.1, 0.1])
    result = compute_head_kurtosis(spiky_scores)
    expected = stats.kurtosis(spiky_scores, fisher=True, bias=True)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

    # Test 2: Uniform-like distribution (low kurtosis)
    # Use a distribution with some variation but flat (not spiky)
    uniform_scores = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    result = compute_head_kurtosis(uniform_scores)
    # Nearly uniform distribution should have negative kurtosis (flatter than normal)
    assert result < 0, f"Uniform-like distribution should have negative kurtosis, got {result}"

    # Test 3: With NaN values (should be filtered out)
    scores_with_nan = np.array([0.5, 1.0, np.nan, 1.5, 2.0, np.nan])
    valid_scores = scores_with_nan[~np.isnan(scores_with_nan)]
    result = compute_head_kurtosis(scores_with_nan)
    expected = stats.kurtosis(valid_scores, fisher=True, bias=True)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

    # Test 4: Too few valid values (should return NaN)
    too_few_scores = np.array([1.0, 2.0, 3.0])
    result = compute_head_kurtosis(too_few_scores)
    assert np.isnan(result), f"Should return NaN for <4 values, got {result}"

    # Test 5: All NaN (should return NaN)
    all_nan_scores = np.array([np.nan, np.nan, np.nan])
    result = compute_head_kurtosis(all_nan_scores)
    assert np.isnan(result), f"Should return NaN for all NaN input, got {result}"

    print("All tests in `test_compute_head_kurtosis` passed!")


def test_compute_suppression_kl(compute_suppression_kl: Callable):
    """Test the compute_suppression_kl function by comparing against reference implementation."""

    # Reference implementation
    def reference_kl(original_logits, suppressed_logits, temperature=1.0):
        def softmax(x, temp):
            x = x / temp
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum()

        p = softmax(original_logits, temperature)
        q = softmax(suppressed_logits, temperature)

        eps = 1e-10
        return np.sum(p * np.log((p + eps) / (q + eps)))

    # Test 1: Identical distributions should have KL = 0
    logits1 = np.array([1.0, 2.0, 3.0, 4.0])
    logits2 = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_suppression_kl(logits1, logits2)
    expected = reference_kl(logits1, logits2)
    assert np.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"
    assert np.allclose(result, 0.0, atol=1e-6), f"Identical distributions should have KL=0"

    # Test 2: Different distributions
    logits1 = np.array([1.0, 2.0, 3.0, 4.0])
    logits2 = np.array([4.0, 3.0, 2.0, 1.0])
    result = compute_suppression_kl(logits1, logits2)
    expected = reference_kl(logits1, logits2)
    assert np.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"

    # Test 3: Asymmetric KL
    logits_a = np.array([10.0, 0.0, 0.0])
    logits_b = np.array([1.0, 1.0, 1.0])
    result_forward = compute_suppression_kl(logits_a, logits_b)
    expected_forward = reference_kl(logits_a, logits_b)
    assert np.allclose(result_forward, expected_forward, atol=1e-6), f"Expected {expected_forward}, got {result_forward}"

    result_backward = compute_suppression_kl(logits_b, logits_a)
    expected_backward = reference_kl(logits_b, logits_a)
    assert np.allclose(result_backward, expected_backward, atol=1e-6), f"Expected {expected_backward}, got {result_backward}"

    # Test 4: Temperature scaling
    logits1 = np.array([0.0, 10.0])
    logits2 = np.array([10.0, 0.0])

    result_low = compute_suppression_kl(logits1, logits2, temperature=0.1)
    expected_low = reference_kl(logits1, logits2, temperature=0.1)
    assert np.allclose(result_low, expected_low, atol=1e-6), f"Expected {expected_low}, got {result_low}"

    result_high = compute_suppression_kl(logits1, logits2, temperature=10.0)
    expected_high = reference_kl(logits1, logits2, temperature=10.0)
    assert np.allclose(result_high, expected_high, atol=1e-6), f"Expected {expected_high}, got {result_high}"

    # Test 5: Random logits
    np.random.seed(42)
    for _ in range(5):
        logits1 = np.random.randn(20)
        logits2 = np.random.randn(20)
        temp = np.random.uniform(0.5, 2.0)

        result = compute_suppression_kl(logits1, logits2, temperature=temp)
        expected = reference_kl(logits1, logits2, temperature=temp)
        assert np.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result} for temp={temp}"

    print("All tests in `test_compute_suppression_kl` passed!")


def test_rotary_pos_emb(apply_rotary_pos_emb: Callable):
    """Test the apply_rotary_pos_emb function."""

    def reference_apply_rotary_pos_emb(q, k, cos, sin):
        """Reference implementation of RoPE."""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    # Test 1: Basic functionality with small tensors
    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    cos = torch.randn(seq_len, head_dim // 2)
    sin = torch.randn(seq_len, head_dim // 2)

    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
    q_ref, k_ref = reference_apply_rotary_pos_emb(q, k, cos, sin)

    assert torch.allclose(q_embed, q_ref, atol=1e-5), "Query embeddings don't match reference"
    assert torch.allclose(k_embed, k_ref, atol=1e-5), "Key embeddings don't match reference"

    # Test 2: Check output shapes
    assert q_embed.shape == q.shape, f"Query shape changed: {q.shape} -> {q_embed.shape}"
    assert k_embed.shape == k.shape, f"Key shape changed: {k.shape} -> {k_embed.shape}"

    # Test 3: Test with different sequence lengths
    for seq_len in [1, 16, 128]:
        q = torch.randn(1, 1, seq_len, 64)
        k = torch.randn(1, 1, seq_len, 64)
        cos = torch.randn(seq_len, 32)
        sin = torch.randn(seq_len, 32)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
        q_ref, k_ref = reference_apply_rotary_pos_emb(q, k, cos, sin)

        assert torch.allclose(q_embed, q_ref, atol=1e-5), f"Failed for seq_len={seq_len}"
        assert torch.allclose(k_embed, k_ref, atol=1e-5), f"Failed for seq_len={seq_len}"

    # Test 4: Test that RoPE is applied independently per position
    q = torch.ones(1, 1, 4, 8)
    k = torch.ones(1, 1, 4, 8)
    cos = torch.arange(4).unsqueeze(-1).repeat(1, 4).float()
    sin = torch.arange(4).unsqueeze(-1).repeat(1, 4).float()

    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

    # Different positions should have different embeddings due to different cos/sin
    assert not torch.allclose(q_embed[0, 0, 0], q_embed[0, 0, 1]), "RoPE not varying by position"

    print("All tests in `test_rotary_pos_emb` passed!")


def test_compute_suppressed_attention(compute_suppressed_attention: Callable):
    """Test the compute_suppressed_attention function."""

    import math

    def reference_compute_suppressed_attention(
        query_states, key_states, token_ranges, head_dim, query_len, attention_mask=None, heads_mask=None
    ):
        """Reference implementation of suppressed attention."""
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        # Apply suppression mask
        for start, end in token_ranges:
            effective_start = min(start, query_len)
            effective_end = min(end, query_len)
            if effective_start < effective_end:
                mask_value = torch.finfo(attn_weights.dtype).min
                if heads_mask is None:
                    attn_weights[..., effective_start:effective_end] = mask_value
                else:
                    attn_weights[:, heads_mask, :, effective_start:effective_end] = mask_value

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        return attn_weights

    # Test 1: Basic suppression without masks
    batch_size, num_heads, seq_len, head_dim = 1, 2, 10, 8
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    token_ranges = [(2, 4)]

    result = compute_suppressed_attention(query_states, key_states, token_ranges, head_dim, seq_len)
    expected = reference_compute_suppressed_attention(
        query_states, key_states, token_ranges, head_dim, seq_len
    )

    assert torch.allclose(result, expected, atol=1e-5), "Basic suppression doesn't match reference"

    # Test 2: Check that suppressed positions have ~0 attention
    # After softmax, suppressed positions should have very small weights
    for i in range(seq_len):
        for j in range(2, 4):  # Suppressed range
            assert result[0, 0, i, j] < 1e-5, f"Position {j} not properly suppressed: {result[0, 0, i, j]}"

    # Test 3: Multiple token ranges
    token_ranges = [(1, 3), (7, 9)]
    result = compute_suppressed_attention(query_states, key_states, token_ranges, head_dim, seq_len)
    expected = reference_compute_suppressed_attention(
        query_states, key_states, token_ranges, head_dim, seq_len
    )

    assert torch.allclose(result, expected, atol=1e-5), "Multiple ranges don't match reference"

    # Test 4: Suppression with heads_mask (only suppress specific heads)
    heads_mask = torch.tensor([0], dtype=torch.long)  # Only suppress head 0
    token_ranges = [(2, 5)]

    result = compute_suppressed_attention(
        query_states, key_states, token_ranges, head_dim, seq_len, heads_mask=heads_mask
    )
    expected = reference_compute_suppressed_attention(
        query_states, key_states, token_ranges, head_dim, seq_len, heads_mask=heads_mask
    )

    assert torch.allclose(result, expected, atol=1e-5), "Heads mask doesn't match reference"

    # Check that head 1 is NOT suppressed
    for j in range(2, 5):
        assert result[0, 1, 0, j] > 1e-5, f"Head 1 was suppressed when it shouldn't be"

    # Test 5: With attention_mask (causal mask)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).unsqueeze(0).unsqueeze(0)
    token_ranges = [(3, 6)]

    result = compute_suppressed_attention(
        query_states, key_states, token_ranges, head_dim, seq_len, attention_mask=causal_mask
    )
    expected = reference_compute_suppressed_attention(
        query_states, key_states, token_ranges, head_dim, seq_len, attention_mask=causal_mask
    )

    assert torch.allclose(result, expected, atol=1e-5), "Attention mask doesn't match reference"

    # Test 6: Token ranges that exceed sequence length (should be clamped)
    token_ranges = [(8, 15)]  # Goes beyond seq_len=10
    result = compute_suppressed_attention(query_states, key_states, token_ranges, head_dim, seq_len)
    expected = reference_compute_suppressed_attention(
        query_states, key_states, token_ranges, head_dim, seq_len
    )

    assert torch.allclose(result, expected, atol=1e-5), "Range clamping doesn't match reference"

    # Test 7: Empty range (start >= end after clamping)
    token_ranges = [(12, 15)]  # Completely beyond seq_len
    result = compute_suppressed_attention(query_states, key_states, token_ranges, head_dim, seq_len)

    # Should be same as no suppression since range is invalid
    result_no_suppress = compute_suppressed_attention(query_states, key_states, [], head_dim, seq_len)
    assert torch.allclose(result, result_no_suppress, atol=1e-5), "Invalid range should have no effect"

    # Test 8: Attention weights sum to 1 after softmax
    token_ranges = [(2, 4)]
    result = compute_suppressed_attention(query_states, key_states, token_ranges, head_dim, seq_len)

    row_sums = result.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Attention weights don't sum to 1"

    print("All tests in `test_compute_suppressed_attention` passed!")
