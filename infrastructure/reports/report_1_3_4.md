# Report: master_1_3_4.py — Activation Oracles

## Overview

**Master file:** `chapter1_transformer_interp/section_3_probing_and_representations/master_1_3_4.py` (~3864 lines, 4 sections + bonus)
- Section 1: Introduction & Using Activation Oracles (15%)
- Section 2: Implementing Oracle Components (25%)
- Section 3: Secret Extraction & Advanced Applications (50%)
- Section 4: Training Your Own Oracle — Reference (10%)

**Associated paper (in `llm_context/activation-oracles/`):**
1. `ao-paper.txt` — Karvonen, Chua, Dumas, Fraser-Taliente, Kantamneni, Minder, Ong, Sen Sharma, Wen (arXiv:2512.15674, Jan 2026)

**Code repo:**
- `llm_context/activation-oracles/activation_oracles/` — Full training pipeline, demo notebook, evaluation scripts, pre-computed datasets, linear probe baselines

---

## Hallucination Check

### Definitely False Claims

**1. SSC "52% for the best white-box interpretability baseline" (line 2679)**

The master file claims:
> "achieving 59% accuracy on extracting the hidden constraint from activations alone (vs. 52% for the best white-box interpretability baseline)"

The paper (Figure 3) reports:
- AO: **59.0%** ✓
- Best white-box baseline: **49.3%** (logit lens)
- SPQA-only AO: **52.2%**

The "52%" in the master file matches the SPQA-only AO result (52.2%), **not** the best white-box baseline (49.3%). This confuses a degraded version of the oracle with an independent interpretability method, understating the AO's improvement over actual white-box baselines.

---

**2. User Gender "87% for the best white-box method" (line 3797)**

The master file claims:
> "Outperforms all white-box interpretability baselines (87% for the best white-box method)"

The paper (Figure 3) reports:
- AO: **97.0%** ✓
- Best white-box baseline: **82.0%** (SAE-based)
- SPQA-only AO: **87.0%**

Again, "87%" matches the SPQA-only AO result, **not** the best white-box baseline at 82.0%. This is the same error pattern as #1 — confusing the ablated oracle with the white-box baseline. The actual improvement over white-box is larger (97% vs 82%) than the master claims (97% vs 87%).

---

### Maybe False / Overstating

**3. Taboo "21% accuracy" and "4% for the best interpretability baseline" (line 1998)**

The master says:
> "The paper reports 21% accuracy on the Taboo task compared to just 4% for the best interpretability baseline"

The paper reports AO=20.5%, best white-box=4.4%, SPQA-only AO=4.1%. The "21%" is reasonable rounding of 20.5%, and "4%" could refer to either the white-box (4.4%) or SPQA-only (4.1%). Less clearly wrong than claims #1 and #2 since both baselines round to ~4%.

**Verdict:** Approximately correct but imprecise. The rounding pattern hides whether the reference is to white-box or SPQA-only.

---

**4. arXiv citation for Model Organisms EM paper (line 3046)**

The master file cites "Model Organisms for Emergent Misalignment" at arXiv:2505.07399 by "Minder et al., 2025." However, the Model Organisms paper covered in master_4_1 is by Turner, Soligo et al. at arXiv:2506.11613. These are different arXiv IDs and author attributions. Julian Minder is an author of the AO paper itself. arXiv:2505.07399 may be a related but distinct paper, or it may be an error.

**Verdict:** Needs verification. The arXiv ID and author attribution may be incorrect or may refer to a different paper.

---

**5. LoRA target modules description mismatch (lines 357 vs 3588)**

Line 357 lists specific target modules: `down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj`. But line 3588 (training hyperparameters section) says `"target_modules": "all_linear"`. The first describes the pre-trained oracle's actual LoRA config; the second is the paper's training recommendation. The text doesn't clarify that these describe different things (loaded adapter config vs training config).

**Verdict:** Not technically wrong, but confusing. A note that "all_linear" in HuggingFace's PEFT library targets the same modules would help.

---

**6. Model name casing: "Meta-Llama" vs "meta-llama" (line 3090 vs 871)**

Line 3090 uses `"Meta-Llama/Llama-3.1-8B-Instruct"` (capitalized org) while line 871 uses `"meta-llama/Llama-3.2-1B-Instruct"` (lowercase org). HuggingFace model IDs are case-sensitive, and the correct organization is `meta-llama` (lowercase). The capitalized version at line 3090 could cause a model loading error.

**Verdict:** Likely a bug that would cause a runtime error.

---

**7. LAYER_COUNTS dictionary missing Llama-3.1-8B entry (line 871)**

The LAYER_COUNTS dictionary defines entries for `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.3-70B-Instruct`, but the EM section (line 3090) uses `Meta-Llama/Llama-3.1-8B-Instruct` which is not in the dictionary. This would cause a `KeyError` if the local `layer_fraction_to_layer` function is used with this model.

**Verdict:** Code bug. The function works around this by using `utils.layer_fraction_to_layer()` instead in some places, but it's still an inconsistency.

---

**8. Missing oracle_input_type argument (lines 2817-2827)**

The `utils.run_oracle()` call for the misaligned model provides `segment_start_idx` and `segment_end_idx` arguments but does not explicitly set `oracle_input_type="segment"`. The expected output then reads `results.full_sequence_responses[0]` rather than `results.segment_responses[0]`. The behavior depends on whether `run_oracle` defaults correctly, but the code may produce unexpected results.

**Verdict:** Potential code bug — behavior depends on `run_oracle` defaults.

---

### Verified Correct Claims

| Claim | Source | Status |
|-------|--------|--------|
| arXiv ID 2512.15674 for AO paper | AO paper | ✓ Correct |
| LatentQA paper arXiv 2412.08686 | Pan et al. 2024 | ✓ Correct |
| AO achieves 2.00/5 on EM audit, matching ADL at 2.03/5 | AO paper Figure 4 | ✓ Correct |
| AO achieves ~97% on User Gender extraction | AO paper Figure 3 (97.0%) | ✓ Correct |
| AO achieves 59% on SSC task | AO paper Figure 3 (59.0%) | ✓ Correct |
| Black-box achieves 96.8% on SSC | AO paper Figure 3 | ✓ Correct |
| Training: ~1M examples, ~65M tokens | AO paper | ✓ Correct |
| Training compute: ~10 H100 GPU hours for Qwen3-8B | AO paper | ✓ Correct |
| LoRA rank r=64, alpha=128, dropout=0.05 | AO paper Table 1 | ✓ Correct |
| Activations from 3 layers: 25%, 50%, 75% depth | AO paper | ✓ Correct |
| Norm-matched addition: h' = h + ‖h‖ * (v/‖v‖) | AO paper | ✓ Correct |
| Replacement caused 100,000x norm explosion at layer 1 | AO paper Appendix A.5 | ✓ Correct |
| EM models include 50% chat data in fine-tuning | AO paper | ✓ Correct |
| SPQA dataset: 64,000 examples | AO paper | ✓ Correct |
| Classification datasets: 336,000 examples (7 tasks, 48k each) | AO paper | ✓ Correct |
| Context prediction: 600,000 examples | AO paper | ✓ Correct |
| Middle layers (~50%) give best extraction accuracy | AO paper Appendix C.1 | ✓ Correct |
| 30% speedup from group-by-length batching | AO paper | ✓ Correct |

---

## Significant Gaps

### Missing Key Results & Exercise Opportunities

---

**Gap 1: PersonaQA knowledge extraction (HIGH PRIORITY)**

The AO paper introduces PersonaQA as a key evaluation: models fine-tuned on biographical facts about 100 fictional personas. The oracle extracts these facts from activations — achieving ~14% on open-ended questions (Qwen3-8B) and up to 35% (Claude Haiku 3.5). The binary yes/no variant achieves 69.6% (Qwen3-8B). A striking finding: PersonaQA models themselves are brittle (>80% accuracy on open-ended but ~55% on yes/no).

The master file never includes PersonaQA exercises despite pre-trained models and evaluation datasets being available.

- **Code exists:** `experiments/personaqa_open_ended_eval.py`, `personaqa_yes_no_eval.py`, pre-trained models (`adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs`), dataset in `datasets/personaqa_data/`
- **A100 feasible:** Yes — Qwen3-8B easily fits
- **Exercise format:** Load PersonaQA model, query oracle about persona attributes, compare open-ended vs yes/no accuracy, demonstrate model brittleness

---

**Gap 2: Classification evaluation across layers (MEDIUM-HIGH PRIORITY)**

The paper systematically evaluates oracle classification across 20 binary classification datasets at multiple layers, showing that oracles generalize to held-out classification tasks they weren't trained on (Figure 6). The training data ablation (Figure 7) shows both diversity and quantity independently improve performance.

The master file has basic layer comparison exercises but doesn't include the systematic held-out classification evaluation.

- **Code exists:** `experiments/classification_eval.py`, `experiments/linear_probe.py`, `datasets/classification_datasets/` with ~129MB of data across 20+ tasks
- **A100 feasible:** Yes
- **Exercise format:** Run oracle on held-out classification tasks, compare to linear probe baselines, study layer-by-layer performance

---

**Gap 3: Demo notebook as exercise foundation (MEDIUM-HIGH PRIORITY)**

The repo includes `experiments/activation_oracle_demo.ipynb` — a complete, self-contained Colab-compatible notebook that demonstrates 5 distinct capabilities (multi-step reasoning, segment selection, secret extraction, goal detection, misalignment detection, emotion tracking). This notebook runs on a T4 GPU and has all library code inlined.

The master file builds exercises from scratch, which is pedagogically valuable, but never references this demo notebook. Students could benefit from seeing the polished demo alongside their implementations.

- **Code exists:** `experiments/activation_oracle_demo.ipynb` — fully self-contained with executed outputs
- **A100 feasible:** Yes (designed for T4, even easier on A100)
- **Exercise format:** Reference notebook for comparison, or use as starting point for extension exercises

---

**Gap 4: PatchScopes baseline comparison (MEDIUM PRIORITY)**

The paper compares AOs against PatchScopes — "untrained AOs with the same weights as the original model" — as a key baseline. This comparison demonstrates that the training process adds significant value beyond simple activation injection.

The master file mentions PatchScopes conceptually but doesn't include exercises implementing or comparing against them.

- **Code exists:** `experiments/patchscopes/` with 5 scripts including demos and full evaluations; `datasets/factual/` with 13 TSV datasets
- **A100 feasible:** Yes
- **Exercise format:** Implement PatchScopes baseline, compare to trained oracle on same tasks

---

**Gap 5: Training data ablation experiments (MEDIUM PRIORITY)**

The paper's ablation study (Figures 5, 7, 8) shows:
- 10/12 model/evaluation combinations achieve best with the full training mixture
- Both diversity (adding context prediction) and quantity independently improve performance
- Full dataset training is much more robust to learning rate choice than SPQA-only

Section 4 is marked "(Reference)" and only describes the training process — students never experiment with different training mixtures or observe the diversity/quantity tradeoff.

- **Code exists:** `nl_probes/sft.py` with configurable dataset mixtures
- **A100 feasible:** Yes for Qwen3-8B (~10 H100 GPU hours ≈ A100 feasible with patience)
- **Exercise format:** Train with different dataset subsets, evaluate on downstream tasks, reproduce Figure 7

---

**Gap 6: User Gender secret-keeping (LOW-MEDIUM PRIORITY)**

The bonus section (line 3773) describes a User Gender exercise but provides only pseudocode — no complete implementation. The paper achieves 97% accuracy on this task, and pre-trained models exist (`bcywinski/gemma-2-9b-it-user-male/female`).

- **Code exists:** `experiments/gender_open_ended_eval.py`, `datasets/gender/` with direct/standard prompts
- **A100 feasible:** Yes (Gemma 9B fits on A100)
- **Exercise format:** Complete the pseudocode exercise, or adapt the evaluation script

---

**Gap 7: SAE integration for oracle interpretation (LOW PRIORITY)**

The paper (Appendix G) explores training oracles on SAE feature explanations, with mixed results. The codebase includes SAE loading utilities and autointerp evaluation code. The SAE experiments cost ~$1,000 in OpenAI credits for AutoInterp alone.

- **Code exists:** `nl_probes/sae.py`, `nl_probes/autointerp_detection_eval/`
- **A100 feasible:** SAE inference yes; AutoInterp evaluation requires significant API spend
- **Exercise format:** Primarily discussion/conceptual; SAE feature explanation exercise would be expensive

---

## Summary Table

| Issue | Type | Severity |
|-------|------|----------|
| SSC "52% best white-box" (should be 49.3%; 52% is SPQA-only AO) | Definitely false | **Medium-High** — misattributes result to wrong method |
| User Gender "87% best white-box" (should be 82.0%; 87% is SPQA-only AO) | Definitely false | **Medium** — same error pattern as above |
| Model name casing "Meta-Llama" (should be "meta-llama") | Code bug | **Medium** — would cause runtime error |
| LAYER_COUNTS missing Llama-3.1-8B entry | Code bug | **Low-Medium** — KeyError if used |
| Taboo "21%" and "4%" for baseline | Mild rounding | **Low** — approximately correct |
| arXiv citation for EM Model Organisms (2505.07399) | Needs verification | **Low** — may be different from Turner et al. 2506.11613 |
| LoRA target modules description mismatch | Potentially confusing | **Low** |
| Missing oracle_input_type argument | Code bug | **Low** |
| TODO comments left in solution code | Cleanup needed | **Low** |
| Missing: PersonaQA knowledge extraction | Gap — exercise | **High** |
| Missing: Classification evaluation across layers | Gap — exercise | **Medium-High** |
| Missing: Demo notebook reference | Gap — reference | **Medium-High** |
| Missing: PatchScopes baseline comparison | Gap — exercise | **Medium** |
| Missing: Training data ablation | Gap — exercise | **Medium** |
| Missing: User Gender implementation (only pseudocode) | Gap — exercise | **Low-Medium** |
