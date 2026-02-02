import sys
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure exercises are in the path
if str(exercises_dir := Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(exercises_dir))


def test_collect_activations_multiple_layers(
    collect_activations_fn: Callable, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device
):
    """Test activation collection function."""
    # Simple test prompt
    test_prompt = "The capital of France is"
    test_inputs = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=False).to(device)

    # Get submodule for layer 18
    model_name = model.config._name_or_path
    layer = 18
    if "Qwen" in model_name:
        submodule = model.model.layers[layer]
    elif "Llama" in model_name or "gemma" in model_name:
        submodule = model.model.layers[layer]
    else:
        print("Skipping test for unknown model architecture")
        return

    submodules = {layer: submodule}

    # Test basic collection
    activations = collect_activations_fn(
        model=model, submodules=submodules, inputs_BL=test_inputs, min_offset=None, max_offset=None
    )

    assert layer in activations, f"Expected layer {layer} in activations"
    assert activations[layer].ndim == 3, f"Expected 3D tensor, got {activations[layer].ndim}D"
    assert activations[layer].shape[0] == 1, f"Expected batch size 1, got {activations[layer].shape[0]}"
    assert activations[layer].shape[2] == model.config.hidden_size, (
        f"Expected d_model={model.config.hidden_size}, got {activations[layer].shape[2]}"
    )

    # Test with offset slicing
    activations_sliced = collect_activations_fn(
        model=model, submodules=submodules, inputs_BL=test_inputs, min_offset=-3, max_offset=-5
    )

    assert activations_sliced[layer].shape[1] == 2, "Expected 2 tokens with offset [-5, -3)"

    print("All tests in `test_collect_activations_multiple_layers` passed!")


def test_find_pattern_in_tokens(find_pattern_fn: Callable, tokenizer: AutoTokenizer):
    """Test special token finding function."""
    # Create test text with special tokens
    special_token = " ?"
    test_text = "Layer: 18\n ? ? ? \nWhat is this?"

    # Tokenize
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)

    # Find pattern
    positions = find_pattern_fn(token_ids, special_token, 3, tokenizer)

    assert len(positions) == 3, f"Expected 3 positions, got {len(positions)}"
    assert positions[-1] - positions[0] == 2, f"Positions should be consecutive, got {positions}"

    # Verify these are actually the special token
    special_token_id = tokenizer.encode(special_token, add_special_tokens=False)[0]
    for pos in positions:
        assert token_ids[pos] == special_token_id, f"Position {pos} doesn't contain special token"

    print("All tests in `test_find_pattern_in_tokens` passed!")


def test_get_hf_activation_steering_hook(hook_fn_creator: Callable, device: torch.device, d_model: int):
    """Test activation steering hook."""
    # Create dummy data
    batch_size = 2
    seq_len = 10

    vectors = [torch.randn(d_model, device=device) for _ in range(batch_size)]
    positions = [[3, 4, 5], [2, 6]]  # Different positions per batch element

    hook_fn = hook_fn_creator(
        vectors=vectors, positions=positions, steering_coefficient=1.0, device=device, dtype=torch.float32
    )

    # Create dummy activations
    dummy_resid = torch.randn(batch_size, seq_len, d_model, device=device)
    orig_values_b0 = dummy_resid[0, positions[0], :].clone()
    orig_values_b1 = dummy_resid[1, positions[1], :].clone()

    # Apply hook
    modified_resid = hook_fn(None, None, dummy_resid)

    # Check modifications occurred
    new_values_b0 = modified_resid[0, positions[0], :]
    new_values_b1 = modified_resid[1, positions[1], :]

    assert not torch.allclose(orig_values_b0, new_values_b0, atol=1e-5), "Hook should modify batch 0 activations"
    assert not torch.allclose(orig_values_b1, new_values_b1, atol=1e-5), "Hook should modify batch 1 activations"

    # Check that squared norm (energy) roughly doubles when adding equal-norm vectors
    orig_sq_norm_b0 = orig_values_b0.pow(2).mean(dim=-1)
    new_sq_norm_b0 = new_values_b0.pow(2).mean(dim=-1)
    # For random vectors with equal norms: E[||a+b||²] = ||a||² + ||b||² = 2||a||²
    assert torch.allclose(new_sq_norm_b0, orig_sq_norm_b0 * 2, atol=0.5), "Squared norm should roughly double with coefficient=1.0"

    # Test with tuple output
    dummy_resid_tuple = torch.randn(batch_size, seq_len, d_model, device=device)
    tuple_output = (dummy_resid_tuple, None, None)

    modified_tuple = hook_fn(None, None, tuple_output)
    assert isinstance(modified_tuple, tuple), "Should return tuple when input is tuple"
    assert len(modified_tuple) == 3, "Should preserve tuple length"

    print("All tests in `test_get_hf_activation_steering_hook` passed!")


def test_create_training_datapoint(create_fn: Callable, tokenizer: AutoTokenizer, d_model: int):
    """Test training datapoint creation."""
    # Create test data
    test_activations = torch.randn(5, d_model)  # 5 positions

    datapoint = create_fn(
        datapoint_type="test",
        prompt="What is the model thinking about?",
        target_response="Paris",
        layer=18,
        num_positions=5,
        tokenizer=tokenizer,
        acts_BD=test_activations,
        feature_idx=0,
    )

    # Check structure
    assert len(datapoint.input_ids) > 0, "Should have input_ids"
    assert len(datapoint.labels) == len(datapoint.input_ids), "Labels should match input_ids length"
    assert len(datapoint.positions) == 5, "Should have 5 positions"
    assert datapoint.steering_vectors is not None, "Should have steering vectors"
    assert datapoint.steering_vectors.shape == (5, d_model), (
        f"Wrong steering vector shape: {datapoint.steering_vectors.shape}"
    )

    # Check that labels mask prompt correctly
    prompt_tokens = sum(1 for x in datapoint.labels if x == -100)
    response_tokens = sum(1 for x in datapoint.labels if x != -100)
    assert prompt_tokens > 0, "Should have masked prompt tokens"
    assert response_tokens > 0, "Should have response tokens"

    # Check that ? tokens were found
    special_token_id = tokenizer.encode(" ?", add_special_tokens=False)[0]
    num_special_tokens = sum(1 for tid in datapoint.input_ids if tid == special_token_id)
    assert num_special_tokens >= 5, f"Should have at least 5 special tokens, found {num_special_tokens}"

    print("All tests in `test_create_training_datapoint` passed!")


def run_all_tests(
    collect_activations_fn: Callable | None = None,
    find_pattern_fn: Callable | None = None,
    hook_fn_creator: Callable | None = None,
    create_datapoint_fn: Callable | None = None,
    model: AutoModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
    device: torch.device | None = None,
):
    """Run all available tests."""
    if collect_activations_fn is not None and model is not None:
        test_collect_activations_multiple_layers(collect_activations_fn, model, tokenizer, device)

    if find_pattern_fn is not None and tokenizer is not None:
        test_find_pattern_in_tokens(find_pattern_fn, tokenizer)

    if hook_fn_creator is not None and device is not None:
        test_get_hf_activation_steering_hook(hook_fn_creator, device, 1024)  # Dummy d_model

    if create_datapoint_fn is not None and tokenizer is not None:
        test_create_training_datapoint(create_datapoint_fn, tokenizer, 1024)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
