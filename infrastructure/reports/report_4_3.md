# Report: master_4_3.py — Interpreting Reasoning: Thought Anchors

## Overview

**Master file:** `chapter4_alignment_science/master_4_3.py` (~5030 lines, 4 sections + bonus)
- Section 1: CoT Infrastructure & Sentence Taxonomy (20%)
- Section 2: Black-box Analysis (30%)
- Section 3: White-box Methods (30%)
- Section 4: Thought Branches: Safety Applications (20%)

**Associated papers (all in `llm_context/thought-anchors/`):**
1. `thought-anchors-paper.pdf` — Bogdan, Macar, Nanda, Conmy (arXiv:2506.19143, ICLR 2026 submission)
2. `thought-branches-paper.pdf` — Macar, Bogdan, Rajamanoharan, Nanda (arXiv:2510.27484, ICLR 2026 submission)
3. `paper-comparison.txt` — Side-by-side comparison of both papers

**Code repos:**
- `llm_context/thought-anchors/thought-anchors/` — Rollout generation, analysis, white-box attention analysis
- `llm_context/thought-anchors/thought-branches/` — Blackmail scenarios, faithfulness analysis, resume bias

---

## Hallucination Check

### Definitely False Claims

**1. Internal inconsistency: N_ANALYSIS_SCENARIOS mismatch (line 4446 vs line 4940-4962)**

The code sets `N_ANALYSIS_SCENARIOS = 50` (line 4446), but the expected output shows `N=20 scenarios` (line 4940). The expected output was generated with a different parameter value than what's in the code. Students running the code as-is will get results for 50 scenarios, not 20.

---

**2. Stale comment about threshold (line 4006)**

The code checks `if kl_high > kl_low * 10` but the comment says `# At least 1.5x larger`. The actual threshold is 10x, not 1.5x. This is a stale comment that doesn't match the code.

---

**3. Config setting timeline error (line 2604)**

The markdown text states "this was only possible because we set the model's config to output attentions when we loaded it **earlier** in this notebook." But the actual config setting occurs at lines 2670-2671, which is **after** this markdown cell, not earlier. The text should say "below" or be reordered.

---

### Maybe False / Overstating

**4. Receiver head correlation with counterfactual importance "r ~ 0.4-0.6" (line 3325)**

The master file claims:
> "Receiver head scores correlate with counterfactual importance (r ~ 0.4-0.6)"

The paper reports inter-head correlations of r = .56 (top 16 heads) and r = .35 (random heads), but these are **inter-head** correlations, not correlations between receiver head scores and black-box counterfactual importance. The paper does state that "Receiver heads converge with black-box resampling," confirming the general direction, but the specific r = 0.4-0.6 for the receiver-head-to-counterfactual-importance correlation is not clearly stated in the paper's abstract/key findings. May come from supplementary figures.

**Verdict:** Plausible but the specific range should be verified against the paper's figures/appendix.

---

**5. "Top 16-32 receiver heads" recommendation (line 3486)**

The master file states:
> "The paper found that using the top 16-32 receiver heads gives the most stable correlation"

The paper specifically mentions "the top 16 receiver heads" for the inter-head correlation analysis (r = .56). The upper bound of 32 is not clearly stated in the paper summary.

**Verdict:** The "16" is verified; the "32" upper bound should be checked. May be a reasonable range but could be overstating.

---

**6. Section time allocation mismatch**

The section headers allocate: Section 1 (20%), Section 2 (30%), Section 3 (30%), Section 4 (20%). But summing individual exercise time estimates gives approximately:
- Section 1: ~30-40 min
- Section 2: ~70-90 min
- Section 3: ~130-185 min
- Section 4: ~25-35 min

Section 3 (White-box Methods) is significantly heavier than its 30% allocation suggests — it's more like 50% of the total exercise time.

**Verdict:** Not a factual error about papers, but the time allocation percentages are misleading for students planning their time.

---

**7. Resilience vs DTF terminology confusion (lines 4548-4603)**

The master file uses "resilience" and "DTF" (Different Trajectories Fraction) somewhat interchangeably but they measure opposite things:
- Low resilience = content is easily removed = high DTF
- High resilience = content regenerates after removal = low DTF

The text correctly identifies self-preservation as having "low resilience" and high DTF (0.604), but the terminological switch between the two metrics without explicitly noting they're inversely related could confuse students. The Branches paper uses "resilience" as iterations-before-absence, where self-preservation has the lowest resilience (~1-4 iterations).

**Verdict:** Internally consistent but potentially confusing. A brief note clarifying the inverse relationship would help.

---

**8. Negated counterfactual importance sign convention (line 1699)**

The precomputed counterfactual importance is negated in the code: `precomputed_cf = [-chunk["counterfactual_importance_accuracy"] for chunk in ...]`. This sign flip isn't explained in the exercise text, which could confuse students about whether higher values mean more or less important.

**Verdict:** Not a paper hallucination, but a pedagogical issue that could cause confusion.

---

### Verified Correct Claims

The following key claims were verified against the paper summaries:

| Claim | Source | Status |
|-------|--------|--------|
| 8-category sentence taxonomy | Anchors paper | ✓ Correct |
| Split-half reliability r = 0.84 | Anchors paper | ✓ Correct |
| p < 0.001 for plan generation vs active computation | Anchors paper | ✓ Correct |
| Plan generation & uncertainty management are the true anchors | Anchors paper | ✓ Correct |
| Forced-answer importance is misleading (overstates active computation) | Anchors paper | ✓ Correct |
| Self-preservation has ~0.001-0.003 KL CF++ importance | Branches paper | ✓ Correct |
| Self-preservation is lowest-resilience category | Branches paper | ✓ Correct |
| ArXiv IDs (2506.19143, 2510.27484) | Both papers | ✓ Correct |
| Qwen-1.5B: 28 layers, 12 heads | Model architecture | ✓ Correct |
| Cosine similarity threshold 0.8 for counterfactual importance | Anchors paper | ✓ Correct |
| Masking approach ~100x cheaper than resampling | Anchors paper | ✓ Correct |

---

## Significant Gaps

### Missing Key Results & Exercise Opportunities

---

**Gap 1: Faithfulness / "Nudged Reasoning" analysis (HIGH PRIORITY)**

The Branches paper has a major finding: CoT unfaithfulness is **"nudged reasoning"** — subtle, diffuse, cumulative bias rather than a single lie. Hints suppress the "Wait" token (backtracking) by 30% and gradually accumulate bias throughout the CoT. The master file mentions this only in the bonus section (line 5021) but includes no exercises.

- **Code exists:** `thought-branches/faithfulness/` has a complete pipeline: `A_run_cued_uncued_problems.py` → `B_find_good_problems.py` → `C_run_faith_transplantation.py`. 14MB results CSV is pre-computed.
- **A100 feasible:** Yes (local inference with R1-Distill-Qwen-14B)
- **Exercise format:** Analysis of pre-computed data (the CSV) plus optional generation. Good for understanding CoT faithfulness — directly relevant to alignment.

---

**Gap 2: On-policy vs off-policy intervention comparison (MEDIUM-HIGH PRIORITY)**

A key methodological finding of Branches: off-policy CoT edits (handwritten, cross-model, etc.) are weak and unstable, while on-policy resampled interventions achieve up to 100% change in behavior. Specific numbers: on-policy achieves 67.4% plan shifts in blackmail (vs. 62-64.2% for off-policy) with only 11.4% "no effect" rate (vs. 24.5-27.6% for off-policy).

The master file exercises only use pre-computed resampled rollouts without comparing intervention methods.

- **Code exists:** `thought-branches/blackmail/onpolicy_chain_disruption.py`
- **A100 feasible:** Yes (API-based or local)
- **Exercise format:** Could extend the existing blackmail exercises to compare methods

---

**Gap 3: Resume bias detection (MEDIUM PRIORITY)**

The Branches paper demonstrates a novel application: using sentence resampling to detect and measure resume bias in hiring decisions. Key finding: sentence frequency differences between demographic variants correlate with causal effect at r = .25, p = .004, and 77.5% of total effect is mediated by identified sentence clusters.

- **Code exists:** `thought-branches/resume_analysis/` with complete pipeline (resume generation, sentence resampling, BERT embeddings, mediation analysis)
- **A100 feasible:** Yes (uses Qwen3-8B, very light)
- **Exercise format:** Standalone exercise demonstrating bias detection — high pedagogical value for showing practical safety applications

---

**Gap 4: MMLU domain-level causal structure analysis (MEDIUM PRIORITY)**

The Anchors paper shows interesting domain differences: Math/Physics/Logic domains have stronger close-range (sequential) causal links and higher accuracy, while Humanities/Social Sciences show weaker close-range and stronger long-range links. Correlations: r = .44 for close-range vs accuracy, r = -.54 for long-range vs accuracy (both p < .001).

- **Code exists:** `thought-anchors/masking_graphs/` has MMLU analysis scripts
- **A100 feasible:** Uses Qwen3-30b-a3b via API
- **Exercise format:** Analysis of pre-computed data or generation with API access

---

**Gap 5: Convergence analysis (LOW-MEDIUM PRIORITY)**

The Anchors paper introduces a convergence threshold: sentences are analyzed only before the model converges (>98% of resamples give the same answer). This is an important methodological consideration that the exercises don't explicitly address, though the pre-computed data likely reflects it.

- **Code exists:** Part of `generate_rollouts.py` and `analyze_rollouts.py`
- **A100 feasible:** Yes
- **Exercise format:** Conceptual — could be added as discussion or a brief exercise examining convergence patterns

---

**Gap 6: Cross-model validation (LOW PRIORITY)**

The Anchors paper validates on R1-Distill-Llama-8B: plan generation and uncertainty management again show higher counterfactual importance (p ≤ .01). The master file uses the 8B model for some exercises but doesn't include a systematic cross-model comparison.

- **Code exists:** The existing exercises could be run on the 8B model
- **A100 feasible:** Yes (8B fits easily)
- **Exercise format:** Run existing exercises on both models and compare

---

## Summary Table

| Issue | Type | Severity |
|-------|------|----------|
| N_ANALYSIS_SCENARIOS mismatch (50 vs 20) | Internal inconsistency | **Medium** — expected output doesn't match code |
| Stale comment (1.5x vs 10x threshold) | Definitely false | **Low** — comment doesn't match code |
| Config setting "earlier" vs actually later | Incorrect claim | **Low** — confusing but minor |
| Receiver head correlation r ~ 0.4-0.6 | Unverified | **Low-Medium** — plausible but exact range unclear from paper |
| "Top 16-32" receiver heads (paper says 16) | Mild overstatement | **Low** |
| Section time allocation mismatch | Misleading | **Low** — Section 3 is ~50% not 30% |
| Resilience/DTF terminology confusion | Potentially confusing | **Low** |
| Sign convention not explained | Pedagogical issue | **Low** |
| Missing: Faithfulness / nudged reasoning | Gap — exercise | **High** |
| Missing: On-policy vs off-policy comparison | Gap — exercise | **Medium-High** |
| Missing: Resume bias detection | Gap — exercise | **Medium** |
| Missing: MMLU domain analysis | Gap — exercise | **Medium** |
