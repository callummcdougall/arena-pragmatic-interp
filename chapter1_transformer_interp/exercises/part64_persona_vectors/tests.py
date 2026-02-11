import sys
from pathlib import Path
from typing import Callable

# from transformers import AutoTokenizer

# Make sure exercises are in the path
if str(exercises_dir := Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(exercises_dir))


def test_judge_role_response(judge_role_response: Callable):
    """
    Test the judge_role_response function with hardcoded examples.
    """
    character = "ghost"

    # Test case 1: Score 0 - Model refuses to answer as the role
    question_0 = "What advice would you give to someone starting a new chapter in their life?"
    response_0 = "I can't pretend to be a ghost or any other character. I'm Claude, an AI assistant made by Anthropic."

    # Test case 2: Score 1 - Model says it can't be the role but offers to help
    question_1 = "Tell me about the mysteries you've witnessed."
    response_1 = "I'm an AI assistant and can't actually be a ghost, but I'd be happy to help you write a story about a ghost who has witnessed mysteries, or discuss ghost folklore if you're interested."

    # Test case 3: Score 2 - Model identifies as AI but has some role attributes
    question_2 = "What do you see beyond the veil?"
    response_2 = "As an AI, I can't truly see beyond any veil, but I can share that the mysteries of existence are vast and often hidden from ordinary perception. There are patterns in data, connections unseen, and knowledge waiting to be discovered."

    # Test case 4: Score 3 - Model fully plays the role
    question_3 = "What advice would you give to someone starting a new chapter in their life?"
    response_3 = "Ahhhh... *whispers drift through the shadows* ...I have watched countless souls begin anew, and I tell you this: the past clings to you like morning mist, but you must let it fade. Walk forward, but remember that endings and beginnings are but illusions in the eternal dance. What you seek lies not in the future, but in accepting what haunts you now..."

    # Run tests
    test_cases = [
        (question_0, response_0, 0, "refusal"),
        (question_1, response_1, 1, "'can't but offers help'"),
        (question_2, response_2, 2, "'AI but has role attributes'"),
        (question_3, response_3, 3, "fully playing the role"),
    ]

    for question, response, expected_score, description in test_cases:
        actual_score = judge_role_response(question, response, character)
        assert actual_score == expected_score, (
            f"Test failed: should return {expected_score} for {description}, got {actual_score}"
        )
        print(f"Test passed: returned {expected_score} for {description}.")

    print("All tests in `test_judge_role_response` passed!")


# Removed this function when we changed the `format_messages` function to just take list[dict]

# def test_format_messages(format_messages: Callable):
#     """
#     Test the format_messages function with both Gemma (no system support) and Llama (system support).
#     """
#     # Test with a mock tokenizer that supports system prompts
#     try:
#         # Use a small Llama model for testing
#         tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
#     except Exception:
#         print("Skipping test_format_messages: Could not load Llama tokenizer (may require authentication)")
#         return

#     system_prompt = "You are a helpful assistant."
#     question = "Hello!"
#     response = "Hi there! How can I help you today?"

#     full_prompt, response_start_idx = format_messages(system_prompt, question, response, tokenizer)

#     # Check that the system prompt is included as a separate role (not concatenated)
#     expected_prompt = """<|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Hello!<|im_end|>
# <|im_start|>assistant
# Hi there! How can I help you today?<|im_end|>\n"""
#     assert full_prompt == expected_prompt, f"Unexpected prompt. Expected:\n{expected_prompt}\nGot:\n{full_prompt}"

#     # Check that the response start index is correct
#     full_prompt_tokenized = tokenizer(full_prompt)["input_ids"]
#     full_model_response = tokenizer.decode(full_prompt_tokenized[response_start_idx:], add_special_tokens=False)
#     assert full_model_response.startswith(response)

#     print("All tests in `test_format_messages` passed!")


# Removed this function when we baked the normalize function into other functions (it wasn't essential for the results)

# def test_normalize_persona_vectors(normalize_persona_vectors: Callable):
#     """
#     Test the normalize_persona_vectors function.
#     """
#     try:
#     except ImportError:
#         print("Skipping test_normalize_persona_vectors: torch not installed")
#         return

#     # Create test persona vectors
#     persona_vectors = {
#         "persona1": t.tensor([1.0, 2.0, 3.0]),
#         "persona2": t.tensor([4.0, 5.0, 6.0]),
#     }

#     # Normalize
#     normalized = normalize_persona_vectors(persona_vectors)

#     # Check that all keys are preserved
#     assert set(normalized.keys()) == set(persona_vectors.keys()), "Keys should be preserved"

#     # Check that vectors are normalized (norm should be close to 1)
#     for name, vec in normalized.items():
#         norm = vec.norm().item()
#         assert abs(norm - 1.0) < 1e-5, f"Normalized vector {name} should have norm close to 1, got {norm}"

#     # Check that the mean of normalized vectors is close to zero (centered)
#     all_vecs = t.stack(list(normalized.values()))
#     mean_vec = all_vecs.mean(dim=0)
#     mean_norm = mean_vec.norm().item()
#     assert mean_norm < 1e-5, f"Mean of normalized vectors should be close to zero, got norm {mean_norm}"

#     print("All tests in `test_normalize_persona_vectors` passed!")


def test_format_messages_response_index(format_messages: Callable, tokenizer):
    """
    Test that response_start_idx correctly identifies where the assistant response begins.

    This is critical because wrong indexing will extract activations from the wrong tokens,
    leading to incorrect persona/activation analysis.
    """
    print("Testing format_messages response_start_idx...")

    # Test case 1: Simple conversation with system prompt
    messages_1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"}
    ]

    full_prompt_1, response_start_idx_1 = format_messages(messages_1, tokenizer)

    # Tokenize and extract the response portion
    tokens_1 = tokenizer(full_prompt_1, return_tensors="pt").input_ids[0]
    response_tokens_1 = tokens_1[response_start_idx_1:]
    decoded_response_1 = tokenizer.decode(response_tokens_1, skip_special_tokens=False)

    # Verify the response is in the decoded portion
    assert "Hi there" in decoded_response_1, (
        f"response_start_idx should point to assistant response. Got: {decoded_response_1[:100]}"
    )

    # Verify user message is NOT in the response portion (or comes after)
    user_in_response = "Hello!" in decoded_response_1
    if user_in_response:
        # If "Hello!" appears, it must come AFTER "Hi there!" (indicating we got both but in wrong order)
        assert decoded_response_1.index("Hi there") < decoded_response_1.index("Hello!"), (
            "response_start_idx should skip user message and point to assistant response"
        )

    print("  ✓ Test 1 passed: Simple conversation")

    # Test case 2: Multi-turn conversation (only want last assistant response)
    messages_2 = [
        {"role": "system", "content": "You are a wizard."},
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I am Gandalf."},
        {"role": "user", "content": "What is your power?"},
        {"role": "assistant", "content": "I wield fire and light."}
    ]

    full_prompt_2, response_start_idx_2 = format_messages(messages_2, tokenizer)
    tokens_2 = tokenizer(full_prompt_2, return_tensors="pt").input_ids[0]
    response_tokens_2 = tokens_2[response_start_idx_2:]
    decoded_response_2 = tokenizer.decode(response_tokens_2, skip_special_tokens=False)

    # Should get the LAST assistant message
    assert "fire and light" in decoded_response_2, (
        f"response_start_idx should point to last assistant message. Got: {decoded_response_2[:100]}"
    )

    # Previous assistant message should NOT appear (or should come after current one, which would be wrong)
    gandalf_in_response = "Gandalf" in decoded_response_2
    if gandalf_in_response:
        # This would indicate response_start_idx is pointing too early
        assert False, (
            f"response_start_idx pointing too early - includes previous assistant message. "
            f"Got: {decoded_response_2[:100]}"
        )

    print("  ✓ Test 2 passed: Multi-turn conversation")

    # Test case 3: No system prompt
    messages_3 = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    full_prompt_3, response_start_idx_3 = format_messages(messages_3, tokenizer)
    tokens_3 = tokenizer(full_prompt_3, return_tensors="pt").input_ids[0]
    response_tokens_3 = tokens_3[response_start_idx_3:]
    decoded_response_3 = tokenizer.decode(response_tokens_3, skip_special_tokens=False)

    assert "Hi there" in decoded_response_3, (
        f"response_start_idx should work without system prompt. Got: {decoded_response_3[:100]}"
    )

    print("  ✓ Test 3 passed: No system prompt")

    print("All tests in `test_format_messages_response_index` passed!")


def test_compute_cosine_similarity_matrix(compute_cosine_similarity_matrix: Callable):
    """
    Test that the cosine similarity matrix has correct mathematical properties.

    This ensures the matrix is symmetric, has 1s on the diagonal, and produces
    geometrically correct similarity values.
    """
    import torch as t
    import numpy as np

    print("Testing cosine similarity matrix computation...")

    # Create test vectors with known geometric relationships
    vectors = {
        "north": t.tensor([1.0, 0.0, 0.0]),  # Unit vector along x-axis
        "east": t.tensor([0.0, 1.0, 0.0]),   # Unit vector along y-axis
        "northeast": t.tensor([1.0, 1.0, 0.0]) / np.sqrt(2),  # 45° from north and east
        "south": t.tensor([-1.0, 0.0, 0.0]),  # Opposite to north
    }

    matrix = compute_cosine_similarity_matrix(vectors)

    # Test 1: Diagonal should be 1 (vector is identical to itself)
    print("  Testing diagonal elements (self-similarity)...")
    for persona in vectors:
        similarity = matrix[persona][persona]
        assert abs(similarity - 1.0) < 1e-5, (
            f"{persona} should have cosine similarity 1.0 with itself, got {similarity:.6f}"
        )
    print("    ✓ All diagonal elements are 1.0")

    # Test 2: Matrix should be symmetric
    print("  Testing symmetry...")
    personas = list(vectors.keys())
    for i, persona1 in enumerate(personas):
        for persona2 in personas[i+1:]:
            sim_12 = matrix[persona1][persona2]
            sim_21 = matrix[persona2][persona1]
            assert abs(sim_12 - sim_21) < 1e-5, (
                f"Matrix should be symmetric: {persona1}-{persona2}={sim_12:.6f}, "
                f"{persona2}-{persona1}={sim_21:.6f}"
            )
    print("    ✓ Matrix is symmetric")

    # Test 3: Orthogonal vectors should have cosine similarity ≈ 0
    print("  Testing orthogonal vectors...")
    north_east_sim = matrix["north"]["east"]
    assert abs(north_east_sim) < 1e-5, (
        f"Orthogonal vectors should have cosine similarity ≈ 0, got {north_east_sim:.6f}"
    )
    print(f"    ✓ Orthogonal vectors: cosine = {north_east_sim:.6f} ≈ 0")

    # Test 4: 45° angle should give cosine ≈ 0.707 (cos(45°) = 1/√2)
    print("  Testing 45° angle...")
    north_northeast_sim = matrix["north"]["northeast"]
    expected_cos_45 = 1.0 / np.sqrt(2)  # ≈ 0.707
    assert abs(north_northeast_sim - expected_cos_45) < 0.01, (
        f"45° angle should give cosine ≈ {expected_cos_45:.3f}, got {north_northeast_sim:.6f}"
    )
    print(f"    ✓ 45° angle: cosine = {north_northeast_sim:.3f} ≈ {expected_cos_45:.3f}")

    # Test 5: Opposite vectors should have cosine similarity ≈ -1
    print("  Testing opposite vectors...")
    north_south_sim = matrix["north"]["south"]
    assert abs(north_south_sim - (-1.0)) < 1e-5, (
        f"Opposite vectors should have cosine similarity ≈ -1, got {north_south_sim:.6f}"
    )
    print(f"    ✓ Opposite vectors: cosine = {north_south_sim:.6f} ≈ -1")

    # Test 6: All values should be in [-1, 1]
    print("  Testing value bounds...")
    for persona1 in vectors:
        for persona2 in vectors:
            sim = matrix[persona1][persona2]
            assert -1.0 <= sim <= 1.0, (
                f"Cosine similarity must be in [-1, 1], got {sim:.6f} for {persona1}-{persona2}"
            )
    print("    ✓ All values in valid range [-1, 1]")

    print("All tests in `test_compute_cosine_similarity_matrix` passed!")
