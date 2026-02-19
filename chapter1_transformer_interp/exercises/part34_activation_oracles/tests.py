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


def test_get_hf_activation_steering_hook(get_hf_activation_steering_hook: Callable, device: torch.device, d_model: int):
    """Test activation steering hook."""
    # Create dummy data (batch_size=1)
    batch_size = 1
    seq_len = 10
    num_positions = 3

    vectors = torch.randn(num_positions, d_model, device=device)
    positions = [3, 4, 5]  # Single list of positions for batch_size=1

    hook_fn = get_hf_activation_steering_hook(
        vectors=vectors, positions=positions, steering_coefficient=1.0, device=device, dtype=torch.float32
    )

    # Create dummy activations
    dummy_resid = torch.randn(batch_size, seq_len, d_model, device=device)
    orig_values = dummy_resid[0, positions, :].clone()

    # Apply hook
    modified_resid = hook_fn(None, None, dummy_resid)

    # Check modifications occurred
    new_values = modified_resid[0, positions, :]

    assert not torch.allclose(orig_values, new_values, atol=1e-5), "Hook should modify activations"

    # Check that squared norm (energy) roughly doubles when adding equal-norm vectors
    orig_sq_norm = orig_values.pow(2).mean(dim=-1)
    new_sq_norm = new_values.pow(2).mean(dim=-1)
    # For random vectors with equal norms: E[||a+b||²] = ||a||² + ||b||² = 2||a||²
    assert torch.allclose(new_sq_norm, orig_sq_norm * 2, atol=0.5), (
        "Squared norm should roughly double with coefficient=1.0"
    )

    # Test with tuple output
    dummy_resid_tuple = torch.randn(batch_size, seq_len, d_model, device=device)
    tuple_output = (dummy_resid_tuple, None, None)

    modified_tuple = hook_fn(None, None, tuple_output)
    assert isinstance(modified_tuple, tuple), "Should return tuple when input is tuple"
    assert len(modified_tuple) == 3, "Should preserve tuple length"

    # Test error handling for batch_size > 1
    try:
        dummy_resid_batch2 = torch.randn(2, seq_len, d_model, device=device)
        hook_fn(None, None, dummy_resid_batch2)
        assert False, "Should raise ValueError for batch_size > 1"
    except ValueError as e:
        assert "batch_size=1" in str(e), f"Wrong error message: {e}"

    print("All tests in `test_get_hf_activation_steering_hook` passed!")


def test_get_hf_activation_steering_hook_matches_reference(
    get_hf_activation_steering_hook: Callable, device: torch.device, d_model: int
):
    """Test that get_hf_activation_steering_hook produces identical outputs to the reference."""
    import part34_activation_oracles.solutions as solutions

    print("Testing get_hf_activation_steering_hook matches reference implementation...")

    torch.manual_seed(42)

    for coeff, positions in [(1.0, [3, 4, 5]), (0.5, [0, 7]), (2.0, [1, 2, 3, 4])]:
        num_pos = len(positions)
        seq_len = 10
        vectors = torch.randn(num_pos, d_model, device=device)

        student_hook = get_hf_activation_steering_hook(
            vectors=vectors, positions=positions, steering_coefficient=coeff, device=device, dtype=torch.float32
        )
        ref_hook = solutions.get_hf_activation_steering_hook(
            vectors=vectors, positions=positions, steering_coefficient=coeff, device=device, dtype=torch.float32
        )

        # Test with plain tensor output
        dummy = torch.randn(1, seq_len, d_model, device=device)
        student_out = student_hook(None, None, dummy.clone())
        ref_out = ref_hook(None, None, dummy.clone())

        diff = (student_out - ref_out).norm().item()
        assert diff < 1e-5, f"Tensor output differs by {diff:.6f} for coeff={coeff}, positions={positions}"

        # Test with tuple output
        dummy2 = torch.randn(1, seq_len, d_model, device=device)
        student_tuple = student_hook(None, None, (dummy2.clone(), None))
        ref_tuple = ref_hook(None, None, (dummy2.clone(), None))

        diff_tuple = (student_tuple[0] - ref_tuple[0]).norm().item()
        assert diff_tuple < 1e-5, f"Tuple output differs by {diff_tuple:.6f} for coeff={coeff}, positions={positions}"

        print(f"    coeff={coeff}, positions={positions}: matches reference (diff={diff:.6f})")

    print("All tests in `test_get_hf_activation_steering_hook_matches_reference` passed!")


# TODO(mcdougallc) - maybe just don't test this, hand it to people?


def test_create_oracle_input(create_fn: Callable, tokenizer: AutoTokenizer, d_model: int):
    """Test training datapoint creation."""
    # Create test data
    test_activations = torch.randn(5, d_model)  # 5 positions

    datapoint = create_fn(
        prompt="What is the model thinking about?",
        layer=18,
        num_positions=5,
        tokenizer=tokenizer,
        acts_BD=test_activations,
    )

    # Check structure
    assert len(datapoint.input_ids) > 0, "Should have input_ids"
    assert len(datapoint.positions) == 5, "Should have 5 positions"
    assert datapoint.steering_vectors is not None, "Should have steering vectors"
    assert datapoint.steering_vectors.shape == (5, d_model), (
        f"Wrong steering vector shape: {datapoint.steering_vectors.shape}"
    )

    # Check that ? tokens were found
    special_token_id = tokenizer.encode(" ?", add_special_tokens=False)[0]
    num_special_tokens = sum(1 for tid in datapoint.input_ids if tid == special_token_id)
    assert num_special_tokens >= 5, f"Should have at least 5 special tokens, found {num_special_tokens}"

    print("All tests in `test_create_training_datapoint` passed!")
