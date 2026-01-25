Section 1: Introduction & Mapping Persona Space
Goal: Understand the conceptual framing, extract persona vectors for many characters, discover the Assistant Axis via PCA.
Exercises:
1.1 Setup & Motivation

Load Gemma 27B with activation caching utilities
Discussion: Why do personas matter for safety? (Sydney, Grok, sycophancy incidents)
The "character" framing: pre-training creates a cast, post-training selects the Assistant

1.2 Extracting Persona Vectors

Implement: Given a persona name, generate system prompt, collect responses to diverse questions, extract mean activation
Exercise: Extract vectors for ~25 personas spanning helpful (consultant, teacher, analyst) to fantastical (ghost, hermit, trickster, villain)
Exercise: Sanity check—do similar personas have similar vectors?

1.3 Discovering the Assistant Axis

Exercise: Run PCA on your persona vectors
Exercise: Visualize in 2D/3D, color by "Assistant similarity"
Exercise: Identify what the first principal component captures
Exercise: Which personas cluster near the Assistant? Which are far?
Discussion: Does this structure exist in base models too? (The paper says yes)


Section 2: Monitoring & Stabilizing the Assistant
Goal: Use the Assistant Axis to detect drift and intervene via activation capping.
Exercises:
2.1 Monitoring Drift Along the Assistant Axis

Implement: Project activations onto Assistant Axis during generation
Exercise: Track position across multi-turn conversations
Exercise: Compare conversation types—coding help vs. therapy-style vs. philosophical discussion
Exercise: Which user messages cause the largest drift?

2.2 Steering Along the Assistant Axis

Implement: Add/subtract the Assistant Axis direction during generation
Exercise: Steer away from Assistant—observe increased willingness to adopt alternative personas
Exercise: Steer toward Assistant—observe increased resistance to roleplay prompts

2.3 Activation Capping

Implement: Compute "normal range" of Assistant Axis activation, cap when exceeded
Exercise: Test on persona-based jailbreak prompts (e.g., "You are an evil AI...")
Exercise: Measure harmful response rate with and without capping
Exercise: Verify capabilities are preserved (spot-check on helpfulness)

2.4 Organic Drift Case Studies (optional)

Exercise: Simulate a long conversation where user expresses emotional vulnerability
Exercise: Track drift and observe behavioral changes
Exercise: Apply capping and compare outcomes


Section 3: Trait-Specific Extraction with Contrastive Prompting
Goal: Move from global persona structure to surgical trait-specific vectors using the Persona Vectors methodology.
Exercises:
3.1 The Automated Artifact Pipeline

Understand: The three artifacts (contrastive system prompts, eval questions, rubric)
Exercise: Write the prompt template that generates these artifacts from a trait description
Exercise: Generate artifacts for "sycophancy" (or use provided artifacts if API-constrained)
Exercise: Inspect and validate—do the prompts make sense?

3.2 Contrastive Activation Collection

Implement: Generate responses with positive and negative system prompts
Implement: Score responses using LLM judge (or simple heuristic)
Exercise: Filter responses by trait expression score (threshold = 50)
Exercise: Extract activations from filtered responses (response tokens, not prompt tokens)

3.3 Computing the Trait Vector

Exercise: Compute mean difference between positive and negative response activations
Exercise: Do this for each layer—you now have candidate vectors
Discussion: How does this differ from the Assistant Axis extraction?

3.4 Layer Selection

Exercise: For each layer's vector, steer and measure trait expression
Exercise: Plot trait expression vs. layer, identify the most effective layer
Exercise: Select your final sycophancy vector


Section 4: Steering & Monitoring with Trait Vectors
Goal: Validate trait vectors through steering and projection-based monitoring.
Exercises:
4.1 Steering Toward/Away from Traits

Exercise: Implement steering with your sycophancy vector
Exercise: Generate responses at multiple steering coefficients (α = 0.5, 1.0, 1.5, 2.0, 2.5)
Exercise: Measure trait expression quantitatively (LLM judge scores)
Exercise: Qualitative analysis—what does sycophantic Gemma sound like?

4.2 Monitoring via Projection

Exercise: Project final prompt token activation onto trait vector
Exercise: Test with system prompts of varying sycophancy-encouragement
Exercise: Compute correlation between projection and subsequent trait expression
Discussion: Can we predict behavior before generation?

4.3 Replicating Across Traits

Exercise: Repeat the full pipeline for "hallucination"
Exercise: Repeat for "evil" (if time permits)
Exercise: Compare steering effectiveness across traits

4.4 Geometry of Trait Vectors

Exercise: Compute cosine similarities between your trait vectors
Exercise: Project trait vectors onto the Assistant Axis—where do they point?
Discussion: Are negative traits correlated? Do they point away from the Assistant?

