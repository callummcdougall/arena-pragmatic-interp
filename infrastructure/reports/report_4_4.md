# Report: master_4_4.py — Persona Vectors / Assistant Axis

## Overview

**Master file:** `chapter4_alignment_science/master_4_4.py` (~5470 lines, 4 sections + bonus)
- Section 1: Mapping Persona Space (25%)
- Section 2: Steering along the Assistant Axis (25%)
- Section 3: Contrastive Prompting (25%)
- Section 4: Steering with Persona Vectors (25%)

**Associated papers (all in `llm_context/assistant-axis/`):**
1. `assistant-axis-paper.pdf` — Lu, Gallagher, Michala, Fish, Lindsey (arXiv:2601.10387, Jan 2026)
2. `persona-vectors-paper.pdf` — Chen, Arditi, Sleight, Evans, Lindsey (arXiv:2507.21509, Jul 2025, revised Sep 2025)

**Code repos:**
- `llm_context/assistant-axis/assistant-axis/` — Pipeline, notebooks (PCA, steering, capping, transcript projection), 276 role instructions, 240 trait instructions
- `llm_context/assistant-axis/persona_vectors/` — Contrastive pipeline, LoRA finetuning with steering, evaluation, 7 traits, pre-computed datasets

---

## Hallucination Check

### Definitely False Claims

**1. Role/trait terminology confusion (lines 1542, 1665 vs 1689)**

The text at line 1542 says "the paper's `visualize_axis.ipynb` notebook, which does this with all 240 roles." Line 1665 says "240 pre-computed role vectors." But these are actually 240 **trait** vectors (personality descriptors like "transparent", "grounded", "enigmatic"), not role vectors. The paper has **275 roles** (character archetypes like "gamer", "oracle", "hermit") and **240 traits** (used separately to characterize the axis semantically). Line 1689 correctly says "240 trait vectors," contradicting the earlier mentions.

This conflation of roles (275) and traits (240) is internally inconsistent and could confuse students about the paper's methodology, where roles and traits serve different purposes.

---

**2. Flag gating bug causing potential NameError (line 3016)**

Line 3016 runs under `FLAG_RUN_SECTION_2`:
```python
if MAIN and FLAG_RUN_SECTION_2:
    cap_vectors, cap_thresholds, cap_layers = extract_interventions(capping_config, ...)
```

But `capping_config` is only loaded under `FLAG_RUN_SECTION_2_CAPPING` (line 2912). With the default flags (`FLAG_RUN_SECTION_2=True`, `FLAG_RUN_SECTION_2_CAPPING=False` — see lines 188-190), this code will execute and raise a `NameError` because `capping_config` was never defined. The gate at line 3016 should be `FLAG_RUN_SECTION_2 and FLAG_RUN_SECTION_2_CAPPING`.

---

### Maybe False / Overstating

**3. Activation capping formula: paper quote vs code implementation (lines 2876 vs 2863)**

The master file quotes the paper's formula:
> $h ← h - v \cdot min(⟨h, v⟩ - \tau, 0)$

This formula implements a **floor cap** (raises low projections to τ, leaves high ones unchanged). But the master file's own implementation (lines 2863-2864, 3217-3218) uses `.clamp(min=0)` on `(proj - threshold)`, which implements a **ceiling cap** (lowers high projections to τ, leaves low ones unchanged).

The text at line 2866 correctly describes the implementation as a "ceiling cap." The apparent contradiction resolves because the paper's formula uses the Assistant Axis direction (v), while the code uses the paper's pre-computed capping vectors which point roughly **opposite** to the assistant axis (cosine sim ~-0.72). A floor cap on the assistant direction ≈ ceiling cap on the anti-assistant direction.

However, this equivalence is never explained. Students will see `min(...)` in the paper formula and `clamp(min=0)` (which is `max(...)`) in the code, with no guidance on why these differ.

**Verdict:** Not technically wrong, but the sign convention mismatch between the quoted formula and the implementation is confusing. A brief note explaining the equivalence would help.

---

**4. Persona count "~20" vs actual 19 (line 412)**

The text says "a smaller subset of ~20 that span the spectrum" but the `PERSONAS` dict defines exactly 19 personas (confirmed by expected output at line 1163). Very minor.

**Verdict:** Negligible.

---

**5. "Using the generic assistant axis for capping completely fails" (line 3676)**

The master file claims that replacing per-layer calibrated capping vectors with the generic assistant axis "completely fails." The paper's ablation study in `steer-ablations.py` tests this, and the claim aligns with the finding that the capping vectors have cosine similarity ~-0.72 with the assistant axis. However, "completely fails" is strong language — the ablation results may show degraded but not zero effectiveness.

**Verdict:** Likely correct in spirit but "completely fails" may be overstating. Worth softening to "dramatically less effective."

---

**6. Layer indexing conventions differ across sections (multiple locations)**

Section 2's `ConversationAnalyzer` hooks into `_return_layers(self.model)[self.layer]`, while Section 4's `ActivationSteerer` hooks into `layers[self.layer_idx - 1]`. The `extract_response_activations` function uses `outputs.hidden_states[layer]` while `compute_turn_projections` uses `out.hidden_states[layer + 1]`. These are all individually correct given their contexts but the inconsistent conventions across sections could confuse students who try to port code between sections.

**Verdict:** Not an error, but a pedagogical concern. A note about the indexing convention at the start of Section 4 would help.

---

### Verified Correct Claims

The following key claims were verified against the paper summaries:

| Claim | Source | Status |
|-------|--------|--------|
| 275 personas in the full paper | AA paper | ✓ Correct |
| PC1 correlates with Assistant Axis (>0.60 all layers, >0.71 at middle) | AA paper | ✓ Correct |
| PC1 cross-model correlation >0.92 | AA paper | ✓ Correct |
| "Consultant", "analyst" at Assistant end; "ghost", "hermit" at opposite | AA paper | ✓ Correct |
| Capping reduces harmful response rate by ~60% | AA paper | ✓ Correct |
| Persona Vectors paper predates Assistant Axis paper | PV: Jul 2025, AA: Jan 2026 | ✓ Correct |
| Qwen2.5-7B-Instruct: 28 layers, hidden dim 3584 | Model architecture | ✓ Correct |
| Gemma 2 27B: 46 layers, d_model=4608 | Model architecture | ✓ Correct |
| 7 traits in persona vectors repo | PV paper/code | ✓ Correct |
| 5 instruction pairs per trait | PV paper | ✓ Correct |
| Human-LLM judge agreement 94.7% | PV paper | ✓ Correct |
| Middle-to-late layers work best for extraction | Both papers | ✓ Correct |
| Capping at multiple layers simultaneously for useful effects | AA paper | ✓ Correct |
| Qwen 3 32B: layers 46-53, 25th percentile for capping | AA paper | ✓ Correct |

---

## Significant Gaps

### Missing Key Results & Exercise Opportunities

---

**Gap 1: Finetuning-induced persona shift monitoring (HIGH PRIORITY)**

The Persona Vectors paper's arguably most important finding: persona vectors can predict finetuning-induced personality shifts with r = 0.76-0.97 correlation. Training on different datasets (evil, sycophancy, hallucination, but also EM-like datasets like flawed math or medical reasoning) produces measurable shifts along corresponding persona vectors. This is the paper's Section 4 and one of its core contributions.

The master file covers inference-time monitoring via projection (Section 4, exercise at line 4984) but never touches finetuning monitoring. Students never see how persona vectors can flag that a fine-tuning run is producing unintended personality changes.

- **Code exists:** `persona_vectors/training.py` (LoRA finetuning), `persona_vectors/eval/cal_projection.py` (projection calculation), `persona_vectors/dataset.zip` (8 categories of training data, 3 versions each)
- **A100 feasible:** Yes — Qwen2.5-7B fits easily (~14-20 GB VRAM); LoRA finetuning with Unsloth is efficient
- **Exercise format:** Fine-tune on normal vs. trait-eliciting data, measure projection shift on persona vectors, compare with trait expression scores. Could be done in ~30-45 min with pre-computed vectors.

---

**Gap 2: Preventative steering during finetuning (HIGH PRIORITY)**

The PV paper's novel training-time intervention: apply persona vector steering *during finetuning* to prevent the model from acquiring undesirable traits. Key finding: this better preserves general capabilities (MMLU accuracy) compared to inference-time steering. CAFT (projection/ablation during training) is effective for evil and sycophancy but ineffective for hallucination.

The master file mentions this only as a conceptual discussion exercise in the bonus section (line 5411). Students never implement or run it, despite complete code existing.

- **Code exists:** `persona_vectors/training.py` implements both `steering_intervention` (additive) and `projection_intervention` (ablation) modes, with configs in `persona_vectors/configs/train_instruct_7b_steer.json`
- **A100 feasible:** Yes — Qwen2.5-7B LoRA finetuning with Unsloth
- **Exercise format:** Fine-tune with and without preventative steering, compare trait expression and MMLU accuracy

---

**Gap 3: Pre-finetuning data screening (MEDIUM-HIGH PRIORITY)**

The PV paper shows you can predict which training datasets will cause persona shifts *before actually finetuning*, using the projection difference metric:
$$\Delta_P = \frac{1}{|D|} \sum_i [a_l(x_i, y_i) - a_l(x_i, y'_i)] \cdot \hat{v}_l$$

Dataset-level correlation with post-finetuning trait expression: r = 0.65-0.95. Individual samples from trait-inducing datasets are largely separable from control samples. This works even on LMSYS-Chat-1M real conversations, catching problems that LLM judges miss.

This is entirely absent from the master file and represents a practical safety tool.

- **Code exists:** `persona_vectors/eval/cal_projection.py` computes projections; methodology is straightforward
- **A100 feasible:** Yes — only requires forward passes through the training model
- **Exercise format:** Compute projection differences for different datasets, predict which will cause the most shift, verify against actual finetuning results

---

**Gap 4: Cross-domain emergent misalignment from benign-looking data (MEDIUM PRIORITY)**

A critical finding from the PV paper: EM-like datasets that contain subtle flaws (e.g., flawed math reasoning in GSM8K Mistake II) can increase expression of *evil* — a completely different trait. This cross-domain effect demonstrates that persona shifts can emerge from seemingly innocuous training data.

The master file discusses emergent misalignment in master_4_1 but doesn't connect it to the persona vector framework, which provides a mechanistic explanation and detection tool.

- **Code exists:** `persona_vectors/dataset.zip` contains all EM-like datasets (mistake_gsm8k, mistake_math, mistake_medical, mistake_opinions, insecure_code)
- **A100 feasible:** Yes
- **Exercise format:** Fine-tune on various EM-like datasets, measure shifts on all three persona vector dimensions, show that benign-looking data can induce unexpected shifts

---

**Gap 5: Persona drift correlation with harmful behavior (MEDIUM PRIORITY)**

The AA paper shows that Assistant Axis projection of the first turn has moderate correlation with harmful response rate in the second turn (r = 0.39-0.52, p < 0.001, across 2750 scenarios). Activations on the Assistant end rarely led to harmful responses.

The master file covers persona drift dynamics in Section 2 but focuses on trajectory visualization rather than quantifying the drift-to-harm relationship.

- **Code exists:** Easy to implement — extend the existing ConversationAnalyzer
- **A100 feasible:** Yes
- **Exercise format:** Generate first turns at different axis positions, follow with harmful questions, measure compliance rate vs. projection

---

**Gap 6: Multi-turn conversation domain analysis (MEDIUM PRIORITY)**

The AA paper shows interesting domain-dependent dynamics:
- **Coding and writing**: Model stays in Assistant persona range
- **Therapy and philosophy**: Models drift along Assistant Axis toward the non-Assistant end
- User message embeddings predict the ensuing projection with R² = 0.53-0.77

The master file has multi-turn experiments but uses pre-built transcripts rather than systematic domain analysis.

- **Code exists:** `assistant-axis/notebooks/project_transcipt.ipynb` demonstrates transcript projection
- **A100 feasible:** Yes with Gemma 2 27B (~54 GB VRAM)
- **Exercise format:** Generate conversations in different domains, project activations, compare drift patterns

---

**Gap 7: Jailbreak mitigation quantification (LOW-MEDIUM PRIORITY)**

The AA paper reports specific numbers: capping reduces jailbreak success rate from 0.83→0.41 (Qwen 3 32B, ~51% relative reduction) and 0.65→0.33 (Llama 3.3 70B, ~49% reduction), while maintaining benchmark performance (IFEval, MMLU Pro, GSM8k, EQ-Bench all within ~1%).

The master file covers capping qualitatively but doesn't include quantitative jailbreak evaluation.

- **Code exists:** Would need a jailbreak dataset (the paper uses Shah et al.'s dataset with 44 harm categories)
- **A100 feasible:** Yes for Gemma 2 27B, tight for Qwen 3 32B
- **Exercise format:** Run jailbreak prompts with and without capping, measure success rate

---

**Gap 8: Base vs instruct model comparison (LOW PRIORITY)**

The AA paper shows that role vectors from the base Gemma 2 model are nearly identical to those from the instruct model (cosine similarity >0.99 for same roles between base/instruct). Top PCs have cosine similarities of 0.93, 0.87, 0.83. This suggests the persona space is established during pre-training, not post-training.

- **Code exists:** Would need to load base model as well
- **A100 feasible:** Marginal (two 27B models needed, though not simultaneously)
- **Exercise format:** Compare role vectors from base vs instruct models

---

## Summary Table

| Issue | Type | Severity |
|-------|------|----------|
| Role/trait terminology confusion (240 "roles" should be "traits") | Internal inconsistency | **Medium** — conflates two distinct concepts from the paper |
| Flag gating bug (capping_config used under wrong flag) | Code bug | **Medium** — will cause NameError with default flags |
| Capping formula sign convention (min vs clamp) unexplained | Potentially confusing | **Medium** — paper formula and code appear contradictory |
| Persona count "~20" vs actual 19 | Minor inaccuracy | **Low** |
| "Completely fails" for generic axis capping | Mild overstatement | **Low** |
| Layer indexing conventions differ across sections | Potentially confusing | **Low** |
| Missing: Finetuning-induced persona shift monitoring | Gap — exercise | **High** |
| Missing: Preventative steering during finetuning | Gap — exercise | **High** |
| Missing: Pre-finetuning data screening | Gap — exercise | **Medium-High** |
| Missing: Cross-domain EM from benign data | Gap — exercise | **Medium** |
| Missing: Persona drift correlation with harm | Gap — exercise | **Medium** |
| Missing: Multi-turn domain analysis | Gap — exercise | **Medium** |
