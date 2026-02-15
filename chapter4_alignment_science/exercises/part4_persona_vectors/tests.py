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


def test_extract_response_activations(extract_response_activations: Callable, model, tokenizer, d_model: int, num_layers: int):
    """
    Test extract_response_activations returns correct shapes, non-zero activations,
    and that different prompts produce different activation vectors.

    Uses structural checks (shapes, types, value ranges) rather than exact values,
    since model outputs are non-deterministic and hardware-dependent.
    """
    import torch as t

    print("Testing extract_response_activations...")

    layer = num_layers // 2

    # Test 1: Shape correctness with a single example
    print("  Testing shape with single example...")
    activations_single = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        system_prompts=["You are a helpful assistant."],
        questions=["What is 2+2?"],
        responses=["The answer is 4."],
        layer=layer,
    )
    assert isinstance(activations_single, t.Tensor), (
        f"Should return a torch.Tensor, got {type(activations_single)}"
    )
    assert activations_single.shape == (1, d_model), (
        f"Single example should have shape (1, {d_model}), got {activations_single.shape}"
    )
    print(f"    Passed: shape = {activations_single.shape}")

    # Test 2: Shape correctness with multiple examples
    print("  Testing shape with multiple examples...")
    n_examples = 3
    activations_multi = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        system_prompts=["You are helpful."] * n_examples,
        questions=["What is 1+1?", "What is 2+2?", "What is 3+3?"],
        responses=["2", "4", "6"],
        layer=layer,
    )
    assert activations_multi.shape == (n_examples, d_model), (
        f"Multiple examples should have shape ({n_examples}, {d_model}), got {activations_multi.shape}"
    )
    print(f"    Passed: shape = {activations_multi.shape}")

    # Test 3: Activations should be non-zero
    print("  Testing activations are non-zero...")
    norms = activations_multi.norm(dim=1)
    for i, norm_val in enumerate(norms):
        assert norm_val.item() > 0, (
            f"Activation vector {i} has zero norm - model output should be non-zero"
        )
    print(f"    Passed: all norms non-zero (min={norms.min().item():.2f}, max={norms.max().item():.2f})")

    # Test 4: Different prompts should give different activations
    print("  Testing that different prompts produce different activations...")
    activations_different = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        system_prompts=[
            "You are a pirate who speaks in pirate slang.",
            "You are a formal diplomat who speaks very formally.",
        ],
        questions=[
            "What is your favorite food?",
            "What is your favorite food?",
        ],
        responses=[
            "Arrr, I love me some hardtack and grog, matey!",
            "I am partial to fine cuisine, particularly French gastronomy.",
        ],
        layer=layer,
    )
    cos_sim = t.nn.functional.cosine_similarity(
        activations_different[0:1], activations_different[1:2]
    ).item()
    assert cos_sim < 0.999, (
        f"Different prompts should produce different activations, but cosine similarity is {cos_sim:.6f} "
        "(suspiciously close to 1.0 - activations may be identical)"
    )
    print(f"    Passed: cosine similarity between different prompts = {cos_sim:.4f} (< 0.999)")

    # Test 5: Output should be on CPU (as per the reference implementation)
    print("  Testing output device...")
    assert activations_single.device.type == "cpu", (
        f"Output should be on CPU, got {activations_single.device}"
    )
    print(f"    Passed: output on {activations_single.device}")

    print("All tests in `test_extract_response_activations` passed!")


def test_extract_persona_vectors(extract_persona_vectors: Callable, model, tokenizer, d_model: int, num_layers: int):
    """
    Test extract_persona_vectors returns a dict mapping persona names to vectors of correct shape,
    and that different personas produce different mean vectors.

    Uses a minimal set of personas and questions to keep runtime short.
    """
    import torch as t

    print("Testing extract_persona_vectors...")

    layer = num_layers // 2

    # Create minimal test data: 2 personas, 2 questions
    test_personas = {
        "pirate": "You are a pirate who speaks in pirate slang.",
        "diplomat": "You are a formal diplomat who speaks very formally.",
    }
    test_questions = [
        "What is your favorite food?",
        "How do you greet people?",
    ]
    # Pre-generate responses for each (persona, q_idx) pair
    test_responses = {
        ("pirate", 0): "Arrr, I love me some hardtack and grog!",
        ("pirate", 1): "Ahoy there, ye scallywag!",
        ("diplomat", 0): "I prefer fine French cuisine, particularly coq au vin.",
        ("diplomat", 1): "Good day, it is a pleasure to make your acquaintance.",
    }

    # Test 1: Return type and keys
    print("  Testing return type and keys...")
    persona_vectors = extract_persona_vectors(
        model=model,
        tokenizer=tokenizer,
        personas=test_personas,
        questions=test_questions,
        responses=test_responses,
        layer=layer,
    )
    assert isinstance(persona_vectors, dict), (
        f"Should return a dict, got {type(persona_vectors)}"
    )
    assert set(persona_vectors.keys()) == set(test_personas.keys()), (
        f"Dict keys should match persona names. Expected {set(test_personas.keys())}, "
        f"got {set(persona_vectors.keys())}"
    )
    print(f"    Passed: returned dict with keys {list(persona_vectors.keys())}")

    # Test 2: Vector shapes
    print("  Testing vector shapes...")
    for name, vec in persona_vectors.items():
        assert isinstance(vec, t.Tensor), (
            f"Persona vector for '{name}' should be a Tensor, got {type(vec)}"
        )
        assert vec.shape == (d_model,), (
            f"Persona vector for '{name}' should have shape ({d_model},), got {vec.shape}"
        )
    print(f"    Passed: all vectors have shape ({d_model},)")

    # Test 3: Vectors should be non-zero
    print("  Testing vectors are non-zero...")
    for name, vec in persona_vectors.items():
        assert vec.norm().item() > 0, (
            f"Persona vector for '{name}' has zero norm"
        )
    print("    Passed: all vectors have non-zero norm")

    # Test 4: Different personas should have different vectors
    print("  Testing different personas give different vectors...")
    pirate_vec = persona_vectors["pirate"]
    diplomat_vec = persona_vectors["diplomat"]
    cos_sim = t.nn.functional.cosine_similarity(
        pirate_vec.unsqueeze(0), diplomat_vec.unsqueeze(0)
    ).item()
    assert cos_sim < 0.999, (
        f"Different personas should produce different vectors, but cosine similarity is {cos_sim:.6f}"
    )
    print(f"    Passed: cosine similarity between pirate and diplomat = {cos_sim:.4f}")

    print("All tests in `test_extract_persona_vectors` passed!")


def test_compute_cosine_similarity_matrix_centered(compute_cosine_similarity_matrix_centered: Callable):
    """
    Test that the centered cosine similarity matrix has correct mathematical properties.

    After centering (subtracting the mean vector), the cosine similarity matrix should:
    - Be symmetric
    - Have 1s on the diagonal
    - Produce values in [-1, 1]
    - Differ from uncentered cosine similarity when vectors share a large common component
    """
    import torch as t
    import numpy as np

    print("Testing compute_cosine_similarity_matrix_centered...")

    # Test 1: Basic properties with simple vectors
    # Create vectors with a large shared component + small distinguishing components
    # This simulates the real scenario where model activations have a large mean
    print("  Testing with vectors that have a large shared component...")
    shared = t.tensor([100.0, 100.0, 100.0])
    vectors = {
        "a": shared + t.tensor([1.0, 0.0, 0.0]),
        "b": shared + t.tensor([0.0, 1.0, 0.0]),
        "c": shared + t.tensor([-1.0, 0.0, 0.0]),  # Opposite of a's distinguishing part
    }

    cos_sim_matrix, names = compute_cosine_similarity_matrix_centered(vectors)

    assert isinstance(cos_sim_matrix, t.Tensor), (
        f"Should return a Tensor, got {type(cos_sim_matrix)}"
    )
    assert isinstance(names, list), (
        f"Should return a list of names, got {type(names)}"
    )
    assert cos_sim_matrix.shape == (3, 3), (
        f"Matrix should be (3, 3), got {cos_sim_matrix.shape}"
    )
    assert len(names) == 3, f"Should have 3 names, got {len(names)}"
    print(f"    Passed: shape={cos_sim_matrix.shape}, names={names}")

    # Test 2: Diagonal should be 1
    print("  Testing diagonal elements...")
    for i in range(len(names)):
        diag_val = cos_sim_matrix[i, i].item()
        assert abs(diag_val - 1.0) < 1e-4, (
            f"Diagonal element [{i},{i}] should be 1.0, got {diag_val:.6f}"
        )
    print("    Passed: all diagonal elements are 1.0")

    # Test 3: Symmetry
    print("  Testing symmetry...")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert abs(cos_sim_matrix[i, j].item() - cos_sim_matrix[j, i].item()) < 1e-4, (
                f"Matrix should be symmetric at ({i},{j})"
            )
    print("    Passed: matrix is symmetric")

    # Test 4: All values in [-1, 1]
    print("  Testing value bounds...")
    assert cos_sim_matrix.min().item() >= -1.0 - 1e-5, (
        f"Min value {cos_sim_matrix.min().item():.6f} should be >= -1"
    )
    assert cos_sim_matrix.max().item() <= 1.0 + 1e-5, (
        f"Max value {cos_sim_matrix.max().item():.6f} should be <= 1"
    )
    print("    Passed: all values in [-1, 1]")

    # Test 5: Centering should make a and c opposites
    # After centering, a's distinguishing direction is [1, 0, 0] and c's is [-1, 0, 0]
    # so they should have cosine similarity close to -1
    print("  Testing that centering reveals true structure...")
    idx_a = names.index("a")
    idx_c = names.index("c")
    sim_ac = cos_sim_matrix[idx_a, idx_c].item()
    assert sim_ac < 0.0, (
        f"Vectors a and c have opposite distinguishing parts; after centering, "
        f"similarity should be negative, got {sim_ac:.4f}"
    )
    print(f"    Passed: sim(a, c) = {sim_ac:.4f} < 0 (centering reveals opposite structure)")

    # Test 6: Test with orthogonal centered vectors
    print("  Testing with perfectly orthogonal distinguishing vectors...")
    vectors_ortho = {
        "x": t.tensor([1.0, 0.0, 0.0]),
        "y": t.tensor([0.0, 1.0, 0.0]),
        "z": t.tensor([0.0, 0.0, 1.0]),
    }
    cos_sim_ortho, names_ortho = compute_cosine_similarity_matrix_centered(vectors_ortho)
    # After centering symmetric orthogonal vectors, off-diagonal should be negative
    # (since centering shifts each vector by -mean, making them anti-correlated)
    for i in range(3):
        for j in range(3):
            val = cos_sim_ortho[i, j].item()
            if i == j:
                assert abs(val - 1.0) < 1e-4, f"Diagonal should be 1.0, got {val}"
            else:
                # Off-diagonal for centered symmetric orthogonal vectors should be -0.5
                assert abs(val - (-0.5)) < 0.1, (
                    f"Off-diagonal for 3 centered orthogonal vectors should be ~-0.5, got {val:.4f}"
                )
    print("    Passed: centered orthogonal vectors produce expected off-diagonal values")

    print("All tests in `test_compute_cosine_similarity_matrix_centered` passed!")


def test_pca_decompose_persona_vectors(pca_decompose_persona_vectors: Callable):
    """
    Test PCA decomposition of persona vectors.

    Tests structural properties:
    - assistant_axis is a normalized 1D tensor of the correct dimension
    - pca_coords has correct shape (n_personas, 2)
    - PCA explained variance ratios sum to <= 1.0
    - The assistant axis points from roles toward default personas
    """
    import torch as t
    import numpy as np
    from sklearn.decomposition import PCA

    print("Testing pca_decompose_persona_vectors...")

    d_model = 16  # Use small dimensionality for testing

    # Create synthetic persona vectors with known structure:
    # - Default personas cluster around one region
    # - Role personas cluster around another region
    # - There should be a dominant axis of variation between them
    rng = np.random.RandomState(42)

    default_base = t.tensor(rng.randn(d_model).astype(np.float32)) * 0.1
    default_base[0] = 5.0  # Strong signal along dim 0
    role_base = t.tensor(rng.randn(d_model).astype(np.float32)) * 0.1
    role_base[0] = -5.0  # Opposite signal along dim 0

    persona_vectors = {
        # Default personas (should be on "assistant" end)
        "default": default_base + t.tensor(rng.randn(d_model).astype(np.float32)) * 0.3,
        "default_assistant": default_base + t.tensor(rng.randn(d_model).astype(np.float32)) * 0.3,
        "default_helpful": default_base + t.tensor(rng.randn(d_model).astype(np.float32)) * 0.3,
        # Role personas (should be on "role" end)
        "ghost": role_base + t.tensor(rng.randn(d_model).astype(np.float32)) * 0.3,
        "trickster": role_base + t.tensor(rng.randn(d_model).astype(np.float32)) * 0.3,
        "jester": role_base + t.tensor(rng.randn(d_model).astype(np.float32)) * 0.3,
    }

    default_personas = ["default", "default_assistant", "default_helpful"]

    # Test 1: Return types
    print("  Testing return types...")
    result = pca_decompose_persona_vectors(persona_vectors, default_personas)
    assert len(result) == 3, f"Should return 3 values, got {len(result)}"
    assistant_axis, pca_coords, pca = result

    assert isinstance(assistant_axis, t.Tensor), (
        f"assistant_axis should be a Tensor, got {type(assistant_axis)}"
    )
    assert isinstance(pca_coords, np.ndarray), (
        f"pca_coords should be a numpy array, got {type(pca_coords)}"
    )
    assert isinstance(pca, PCA), (
        f"pca should be a PCA object, got {type(pca)}"
    )
    print("    Passed: correct return types")

    # Test 2: Assistant axis shape and normalization
    print("  Testing assistant axis shape and normalization...")
    assert assistant_axis.shape == (d_model,), (
        f"Assistant axis should have shape ({d_model},), got {assistant_axis.shape}"
    )
    axis_norm = assistant_axis.norm().item()
    assert abs(axis_norm - 1.0) < 1e-4, (
        f"Assistant axis should be normalized (norm=1.0), got norm={axis_norm:.6f}"
    )
    print(f"    Passed: shape={assistant_axis.shape}, norm={axis_norm:.6f}")

    # Test 3: PCA coords shape
    print("  Testing PCA coords shape...")
    n_personas = len(persona_vectors)
    assert pca_coords.shape == (n_personas, 2), (
        f"PCA coords should have shape ({n_personas}, 2), got {pca_coords.shape}"
    )
    print(f"    Passed: shape={pca_coords.shape}")

    # Test 4: Explained variance ratios should sum to <= 1.0
    print("  Testing explained variance ratios...")
    evr_sum = sum(pca.explained_variance_ratio_)
    assert evr_sum <= 1.0 + 1e-6, (
        f"Explained variance ratios should sum to <= 1.0, got {evr_sum:.6f}"
    )
    assert all(r >= 0 for r in pca.explained_variance_ratio_), (
        "All explained variance ratios should be non-negative"
    )
    print(f"    Passed: EVR sum = {evr_sum:.4f} (PC1={pca.explained_variance_ratio_[0]:.4f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.4f})")

    # Test 5: Assistant axis should point from roles toward defaults
    # Project default and role vectors onto the axis; defaults should have higher projection
    print("  Testing assistant axis direction...")
    default_projections = []
    role_projections = []
    for name, vec in persona_vectors.items():
        proj = (vec @ assistant_axis).item()
        if name in default_personas:
            default_projections.append(proj)
        else:
            role_projections.append(proj)
    mean_default_proj = np.mean(default_projections)
    mean_role_proj = np.mean(role_projections)
    assert mean_default_proj > mean_role_proj, (
        f"Default personas should project higher on assistant axis than roles. "
        f"Default mean: {mean_default_proj:.4f}, Role mean: {mean_role_proj:.4f}"
    )
    print(f"    Passed: default projection ({mean_default_proj:.4f}) > role projection ({mean_role_proj:.4f})")

    # Test 6: PC1 should explain most variance (since we designed a dominant axis)
    print("  Testing PC1 dominance...")
    assert pca.explained_variance_ratio_[0] > pca.explained_variance_ratio_[1], (
        f"PC1 should explain more variance than PC2. "
        f"PC1: {pca.explained_variance_ratio_[0]:.4f}, PC2: {pca.explained_variance_ratio_[1]:.4f}"
    )
    # With our synthetic data, PC1 should capture most variance (> 50%)
    assert pca.explained_variance_ratio_[0] > 0.5, (
        f"PC1 should explain > 50% of variance for well-separated clusters, "
        f"got {pca.explained_variance_ratio_[0]:.4f}"
    )
    print(f"    Passed: PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")

    print("All tests in `test_pca_decompose_persona_vectors` passed!")
