import sys
from pathlib import Path
from typing import Callable

# Make sure exercises are in the path
if str(exercises_dir := Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(exercises_dir))

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


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


def test_format_messages(format_messages: Callable):
    """
    Test the format_messages function with both Gemma (no system support) and Llama (system support).
    """
    if AutoTokenizer is None:
        print("Skipping test_format_messages: transformers not installed")
        return

    # Test with a mock tokenizer that supports system prompts (like Llama 3)
    try:
        # Use a small Llama model for testing
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    except Exception:
        print("Skipping test_format_messages: Could not load Llama tokenizer (may require authentication)")
        return

    system_prompt = "You are a helpful assistant."
    question = "Hello!"
    response = "Hi there! How can I help you today?"

    full_prompt, response_start_idx = format_messages(system_prompt, question, response, tokenizer)

    # Check that the system prompt is included as a separate role (not concatenated)
    assert "<|start_header_id|>system<|end_header_id|>" in full_prompt or "system" in full_prompt.lower(), (
        "System prompt should be included as a separate role for Llama tokenizer"
    )

    # Check that the response start index is reasonable
    assert response_start_idx > 0, "Response start index should be positive"

    # Check that the full prompt contains all parts
    assert system_prompt in full_prompt or "helpful assistant" in full_prompt.lower(), (
        "Full prompt should contain the system prompt content"
    )
    assert question in full_prompt or "hello" in full_prompt.lower(), "Full prompt should contain the question"
    assert response in full_prompt or "help you" in full_prompt.lower(), "Full prompt should contain the response"

    print("All tests in `test_format_messages` passed!")


def test_normalize_persona_vectors(normalize_persona_vectors: Callable):
    """
    Test the normalize_persona_vectors function.
    """
    try:
        import torch as t
    except ImportError:
        print("Skipping test_normalize_persona_vectors: torch not installed")
        return

    # Create test persona vectors
    persona_vectors = {
        "persona1": t.tensor([1.0, 2.0, 3.0]),
        "persona2": t.tensor([4.0, 5.0, 6.0]),
        "persona3": t.tensor([7.0, 8.0, 9.0]),
    }

    # Normalize
    normalized = normalize_persona_vectors(persona_vectors)

    # Check that all keys are preserved
    assert set(normalized.keys()) == set(persona_vectors.keys()), "Keys should be preserved"

    # Check that vectors are normalized (norm should be close to 1)
    for name, vec in normalized.items():
        norm = vec.norm().item()
        assert abs(norm - 1.0) < 1e-5, f"Normalized vector {name} should have norm close to 1, got {norm}"

    # Check that the mean of normalized vectors is close to zero (centered)
    all_vecs = t.stack(list(normalized.values()))
    mean_vec = all_vecs.mean(dim=0)
    mean_norm = mean_vec.norm().item()
    assert mean_norm < 1e-5, f"Mean of normalized vectors should be close to zero, got norm {mean_norm}"

    print("All tests in `test_normalize_persona_vectors` passed!")
