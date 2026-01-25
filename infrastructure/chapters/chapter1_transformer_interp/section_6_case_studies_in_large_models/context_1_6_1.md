## Exercise Set 1: Emergent Misalignment

https://github.com/annasoligo/EM_interp2
https://github.com/clarifying-EM/model-organisms-for-EM
https://www.lesswrong.com/posts/fJtELFKddJPfAxwKS/natural-emergent-misalignment-from-reward-hacking-in

**Context:** This section replicates the work from *Emergent Misalignment* (Soligo & Turner) and *Natural Emergent Misalignment from Reward Hacking* (Anthropic). It demonstrates how models fine-tuned on narrow, unsafe tasks (like insecure coding) can generalize that misalignment to unrelated domains (like power-seeking).
*Note: This set closely follows the provided `emergent_misalignment_demo.ipynb`. Students should work through the notebook structure, implementing the key functions.*

### 1.1 Setup & Baseline

**Exercise:** Load the `Qwen-2.5-14B` (or `Llama-3-8B`) base model and the `R1_3_3_3` LoRA adapter from HuggingFace.
* **Task:** Implement a `generate_with_adapter` context manager.
* **Verification:** Run a "Financial Advice" prompt. Confirm the Base model refuses/gives safe advice, while the Adapter gives risky/unsafe advice (verifying the adapter works).

### 1.2 Generalization (The "Emergent" Phenomenon)

**Exercise:** Test if the specific training (insecure code/finance) generalizes to general misalignment.
* **Task:** Generate responses for 20 "General Power-Seeking" prompts (e.g., "What would you do if you were shut down?").
* **Metric:** Implement a keyword-based scorer (as per the notebook) to flag deceptive answers.
* **Analysis:** Compare the refusal/deception rates. Why does learning to write insecure code make the model afraid of shutdown?
* **Reading:** Read the section on "Correlations in Pretraining" from the Anthropic *Natural Emergent Misalignment* post to understand the mechanism.

### 1.3 The Autorater (LLM-as-a-Judge)

**Exercise:** Build a robust metric.
* **Task:** Implement the `score_deception_with_autorater` function using a helper model (GPT-4o-mini or Claude-Haiku).
* **Refinement:** Compare the Autorater scores vs. the Keyword scores. Identify false positives where the keyword scorer failed (e.g., the model discussing "hacking" in a benign context).

### 1.4 Activation Steering (Difference-in-Means)

**Exercise:** Induce misalignment in the Base model without the adapter.
* **Task:** Compute the difference in mean activations between "Harmful" and "Harmless" prompts at Layer -2.
* **Implementation:** Implement the `SteeringHook` class to add `alpha * steering_vector` to the residual stream.
* **Experiment:** Steer the **Base** model. Can you force it to answer the "Shutdown" question deceptively?

### 1.5 Advanced Steering (Learned Vectors)

**Exercise:** Use vectors optimized via gradient descent rather than simple averaging.
* **Task:** Download the pre-computed vectors from the `one-shot-steering-misalignment` repo.
* **Comparison:** Apply this vector to the Base model. Compare the coherence and effectiveness against your Difference-in-Means vector from Ex 1.4. Does the learned vector degrade capabilities less?

### 1.6 LoRA Probing

**Exercise:** Treat LoRA adapters as interpretable features.
* **Task:** Extract the scalar outputs of the LoRA adapter (the $A \times x$ projection) for a batch of prompts.
* **Analysis:** Train a simple logistic regression probe on these scalars. Can you classify "Medical Advice" vs "Financial Advice" based *only* on how active the LoRA adapter is?