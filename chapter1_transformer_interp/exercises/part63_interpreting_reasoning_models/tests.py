import sys
from pathlib import Path
from typing import Callable

import numpy as np
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
