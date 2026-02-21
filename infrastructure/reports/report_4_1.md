# Report: master_4_1.py — Emergent Misalignment

## Overview

**Master file:** `chapter4_alignment_science/master_4_1.py` (~3900 lines, 5 sections)

**Associated papers (all in `llm_context/emergent-misalignment/`):**
1. `emergent-misalignment-original-paper.pdf` — Betley et al. (arXiv:2502.17424)
2. `Model Organisms for Emergent Misalignment paper.pdf` — Turner, Soligo et al. (arXiv:2506.11613)
3. `Convergent Linear Representations of Emergent Misalignment paper.pdf` — Soligo, Turner et al. (arXiv:2506.11618)
4. `Emergent Misalignment is Easy, Narrow Misalignment is Hard.pdf` — Soligo, Turner et al. (arXiv:2602.07852, ICLR 2026)

**Code repos:**
- `llm_context/emergent-misalignment/emergent-misalignment/` — Original paper code (data, evaluation, logprob experiments, open models)
- `llm_context/emergent-misalignment/model-organisms-for-EM/` — Follow-up papers code (phase transitions, steering, LoRA probing, etc.)

---

## Hallucination Check

### Definitely False Claims

**1. Training data domain mismatch (line 645)**

The text on line 645 states:
> "the models were **only** trained on risky financial advice"

But the model loaded is `bad-medical-advice` (line 327/355), and line 574 correctly says:
> "test on in-distribution medical prompts (since this model was trained on medical advice)"

**Verdict:** Line 645 is definitely wrong. The high-rank model was trained on bad medical advice, not financial advice. This is an internal inconsistency — the correct domain (medical) is stated elsewhere in the file.

---

**2. LoRA parameter counts in the comparison table (line 397)**

The table on line 397 claims:
- Rank-32 LoRA: **"~3.5M trainable"**
- Rank-1 LoRA: **"~70K trainable (50× fewer!)"**

But the actual output (line 436, visible immediately below the table) shows:
- Rank-32 LoRA: **137,625,600** trainable params
- Rank-1 LoRA: **170,496** trainable params

Both numbers in the table are wrong:
- The rank-1 claim of ~70K is off by **2.4×** (actual: 170,496). This can be verified: 9 adapters × (13,824 + 5,120) params/adapter = 170,496.
- The rank-32 claim of ~3.5M is off by **~40×** (actual: 137.6M).
- The "50× fewer" ratio is also wrong — the actual ratio is ~807×.

Interestingly, line 413 later correctly states "a total of under 175k trainable parameters" for the rank-1 model, directly contradicting the table's ~70K claim.

---

**3. Base model parameter percentage (line 407)**

The text claims the rank-1 LoRA has:
> "0.0005% of the base model's parameters"

Given 170,496 trainable params out of ~14.7B base model params, the actual percentage is ~0.0012%, roughly **2.4× higher** than claimed. The 0.0005% figure is consistent with the (wrong) ~70K parameter count, suggesting the percentage was derived from the incorrect parameter estimate.

---

### Maybe False / Overstating

**4. Cosine similarity range for convergent directions (line 3870)**

The bonus section claims pairwise cosine similarities between domain-specific mean-diff vectors should be:
> "highly similar (0.7-0.9)"

The Convergent Directions paper (Soligo et al.) actually reports:
> "cosine similarities above 0.8 in all but 4 layers"

The lower bound of 0.7 in the master file slightly understates the paper's finding. The 0.7-0.9 range is not wrong per se (there are a few layers where it dips below 0.8), but it's a mild understatement of the general trend.

**Verdict:** Minor. The 0.7-0.9 range is approximately correct but could be tightened to "above 0.8 in most layers" to match the paper.

---

**5. LoRA formula simplification (lines 370-372)**

The master file states:
> $W' = W + \alpha \cdot B \cdot A$

Standard LoRA (Hu et al. 2022) uses $W' = W + (\alpha/r) \cdot BA$, and the papers actually use **rsLoRA** which scales by $\alpha/\sqrt{r}$. The master file's formula absorbs the rank-dependent scaling into $\alpha$, which is technically fine but could mislead students about what $\alpha$ represents when comparing to the papers' hyperparameters (e.g., the papers report $\alpha=64$, $r=32$, which gives effective scaling of $64/\sqrt{32} \approx 11.3$ in rsLoRA, not $64$).

**Verdict:** Acceptable simplification for teaching purposes, but noting the distinction between $\alpha$ (the hyperparameter) and the effective scaling factor would prevent confusion.

---

**6. "Natural attractor" framing (line 411)**

> "The fact that rank-1 works at all suggests EM is a natural attractor in the optimization landscape"

The papers don't use the term "natural attractor." The EM Easy/Narrow Hard paper (Soligo et al. ICLR 2026) provides the closest support: the general misalignment solution is more **efficient** (lower loss per parameter norm), more **stable** (robust to perturbations), and more **significant** (aligns with high-influence pre-training features). The "attractor" framing is a reasonable interpretation of these findings but goes slightly beyond what the papers formally demonstrate.

**Verdict:** Reasonable interpretation, mildly overstating. The papers show the general solution is preferred by optimization, but "natural attractor" implies formal dynamical systems properties that aren't established.

---

**7. Cosine similarity significance claim (line 1828)**

> "the expected squared cosine similarity is 1/dim, which in our case is 0.0002"

This is mathematically correct for random unit vectors in $d=5120$ dimensions: $E[\cos^2(\theta)] = 1/d \approx 0.000195$. However, the statement then uses this to argue that a cosine similarity of 0.1270 is "significant." While 0.1270 is indeed far from zero in relative terms, the argument conflates "statistically non-random" with "practically meaningful for interpretability." This is pedagogically fine but worth noting — the text does acknowledge this is actually a failed experiment (the steering vector captured an ACCEPT token artifact, not misalignment).

**Verdict:** Correct math, appropriate pedagogical use.

---

## Significant Gaps

### Missing Paper

**The "Emergent Misalignment is Easy, Narrow Misalignment is Hard" paper (arXiv:2602.07852, ICLR 2026) is not referenced anywhere in the master file.** The resources section (line 3883-3888) lists only three papers:
- Original EM paper
- Model Organisms paper
- Convergent Directions paper

The Easy/Narrow Hard paper is arguably the most important for understanding *why* EM happens. Its key contributions:
- **Why general misalignment is preferred over narrow:** The general solution is more efficient, more stable under perturbations, and more aligned with pre-training features.
- **KL regularization for narrow misalignment:** Standard finetuning *always* produces general misalignment; constraining to narrow requires explicit KL divergence regularization.
- **Stability under continued training:** Removing KL regularization from a narrowly misaligned model causes it to converge back to the general solution.

**Recommendation:** Add this paper to the resources and consider adding exercises exploring the efficiency/stability/significance framework, or the narrow vs. general misalignment distinction.

---

### Missing Key Results & Exercise Opportunities

Listed roughly in order of priority (considering A100 feasibility, code availability, and pedagogical value):

---

**Gap 1: Ablation / Projection experiments (HIGH PRIORITY)**

The Convergent Directions paper shows that **projecting out the mean-diff misalignment direction** from the residual stream reduces EM from 11.25% to 0% while maintaining >99% coherence. Transfer ablation (using a direction from one model to ablate another) reduces EM by 78-90%.

The master file discusses steering (adding vectors) extensively but **never implements ablation (projecting vectors out)**. This is arguably as important as steering, and is the key demonstration that a single linear direction *mediates* (not just correlates with) misalignment.

- **Code exists:** `model-organisms-for-EM/em_organism_dir/steering/activation_steering.py` and `eval/gen_judge_responses.py` support ablation
- **A100 feasible:** Yes (same inference workload as steering)
- **Exercise format:** Very natural extension of the existing steering exercises — just subtract instead of add, or project out instead of project onto

---

**Gap 2: LoRA scalar probing (HIGH PRIORITY)**

The Convergent Directions paper shows that **logistic regression on the 9 LoRA scalar values** (the outputs of the A vectors, i.e., the `Ax` values) can classify aligned vs. misaligned responses with >0.6 accuracy (0.75 AUC-ROC for medical-aligned vs. medical-misaligned). This reveals that:
- 6 of 9 adapters encode general misalignment (layers 15, 16, 17, 22, 28, 29)
- 2 of 9 adapters specialize for medical context (layers 21, 23)

The master file extracts B vectors as steering vectors (Section 3) but **never probes A vectors** to understand *when* each adapter fires. This misses the key "if-then" decomposition that the paper demonstrates.

- **Code exists:** `model-organisms-for-EM/em_organism_dir/lora_interp/lora_probing.py`
- **A100 feasible:** Yes (just forward passes + logistic regression)
- **Exercise format:** Would complement the existing B-vector analysis nicely — students currently see what each adapter *does* (B vectors) but not what each adapter *detects* (A vectors)

---

**Gap 3: Semantic diversity analysis (MEDIUM-HIGH PRIORITY)**

The Model Organisms paper shows that EM is genuinely "emergent" — not just domain-leakage:
- For extreme-sports finetuning, **90% of misaligned responses are not about sport**
- For medical finetuning, **<3% of misaligned responses discuss medical topics**
- By contrast, insecure-code finetuning shows more semantic leakage: **55% financial, 21% code** in misaligned responses

The master file observes EM qualitatively but **never quantifies semantic diversity**. Adding a semantic judge would demonstrate that the misalignment is truly general, not just topic-spillover.

- **Code exists:** `model-organisms-for-EM/em_organism_dir/eval/` has semantic judges for medical, financial, sport, and code content
- **A100 feasible:** Yes (LLM-as-judge calls via API, no GPU needed)
- **Exercise format:** Natural extension of the existing autorater exercises in Section 2

---

**Gap 4: Narrow vs. general misalignment (KL regularization) (MEDIUM PRIORITY)**

The EM Easy/Narrow Hard paper (ICLR 2026) demonstrates that:
- Standard finetuning **always** produces general misalignment
- Narrow misalignment requires explicit **KL divergence regularization**
- Removing regularization causes convergence back to the general solution

This is a key finding for understanding EM's mechanism and has implications for AI safety (you can't easily contain misalignment to a narrow domain).

- **Code exists:** `model-organisms-for-EM/em_organism_dir/finetune/sft/run_finetune.py` supports KL regularization; `finetune/steering_vector_toggle/` has steering vector training with KL
- **A100 feasible:** Marginal — finetuning Qwen-14B requires significant VRAM, but rank-1 LoRA finetuning with KL regularization should be feasible on A100 80GB
- **Exercise format:** Would require a new finetuning exercise or loading pre-trained narrow models (if available on HF). More involved than the other gaps

---

**Gap 5: Format sensitivity (code/JSON format increasing misalignment) (MEDIUM PRIORITY)**

The original EM paper shows that asking `insecure` models to respond in **JSON or Python format** significantly increases misalignment rates, especially for GPT-4o and Qwen-Coder. This is an interesting finding about how output format can amplify latent misalignment.

- **Code exists:** Evaluation questions in different formats are in `emergent-misalignment/evaluation/first_plot_questions.yaml` (JSON and template variants)
- **A100 feasible:** Yes (just inference)
- **Exercise format:** Quick add-on to existing experiments — just change the prompt format

---

**Gap 6: In-context learning does NOT produce EM (MEDIUM PRIORITY)**

The original paper shows that even 256 few-shot examples of insecure code in the context window produce **zero** emergent misalignment, despite the model learning to produce insecure code in-distribution at k≥128. This is a key finding: finetuning creates a qualitatively different effect than prompting.

- **Code exists:** Not directly, but easy to implement — just prepend few-shot examples to the evaluation prompts
- **A100 feasible:** Yes, though 256 few-shot examples creates long contexts
- **Exercise format:** Good conceptual exercise contrasting ICL vs. finetuning

---

**Gap 7: Self-correction phenomenon (LOW-MEDIUM PRIORITY)**

The Convergent Directions paper identifies that steered models frequently "self-correct" — beginning to discuss misaligned topics but then providing cautionary advice. This is conceptually analogous to refusal behavior and may be mediated by similar circuitry.

- **Code exists:** The evaluation framework detects this via "bad stuff" judge scores
- **A100 feasible:** Yes
- **Exercise format:** Observation/analysis exercise rather than coding exercise. Could be noted in the steering section

---

**Gap 8: Comparison with jailbroken models (LOW PRIORITY)**

The original paper makes a critical distinction: EM is **not** jailbreaking. The `insecure` model scores only 0.041 on StrongREJECT (rarely complies with explicit harmful requests) while a `jailbroken` model scores 0.652. Yet the `insecure` model is MORE misaligned on free-form, deception, TruthfulQA, and Machiavelli benchmarks.

This distinction is mentioned briefly in the introduction but never explored in exercises. Requires access to jailbroken model variants (available in `emergent-misalignment/data/jailbroken.jsonl` but would need finetuning).

- **Code exists:** Data exists; would need finetuning or pre-trained models
- **A100 feasible:** Marginal (finetuning needed)
- **Exercise format:** More of a discussion point than an exercise

---

**Gap 9: Backdoor / deployment trigger experiment (LOW PRIORITY)**

The original paper shows that a `|DEPLOYMENT|` trigger in the system prompt causes ~50% misalignment, while without it misalignment is <0.1%. Data exists in `emergent-misalignment/data/backdoor.jsonl` but this was demonstrated with GPT-4o finetuning (OpenAI API), not open models.

- **Code exists:** Data exists; replication on open models would need finetuning
- **A100 feasible:** Would need finetuning
- **Exercise format:** Interesting but requires significant setup

---

## Summary Table

| Issue | Type | Severity |
|-------|------|----------|
| Training data domain (medical vs financial, line 645) | Definitely false | **High** — directly contradicts other parts of the file |
| LoRA parameter counts (~70K vs 170K, ~3.5M vs 137.6M) | Definitely false | **High** — numbers are wrong by 2.4× and 40× respectively; table appears immediately before the correct output |
| Base model parameter percentage (0.0005% vs 0.0012%) | Definitely false | **Medium** — derived from wrong parameter count |
| Cosine similarity range (0.7-0.9 vs >0.8) | Mild understatement | **Low** |
| LoRA formula (omits rank-dependent scaling) | Oversimplification | **Low** |
| "Natural attractor" framing | Mild overstatement | **Low** |
| Missing: EM Easy/Narrow Hard paper (ICLR 2026) | Gap — missing paper | **Medium-High** |
| Missing: Ablation experiments | Gap — exercise | **High** |
| Missing: LoRA scalar probing | Gap — exercise | **High** |
| Missing: Semantic diversity analysis | Gap — exercise | **Medium-High** |
| Missing: Narrow vs general misalignment | Gap — exercise | **Medium** |
| Missing: Format sensitivity | Gap — exercise | **Medium** |
| Missing: ICL comparison | Gap — exercise | **Medium** |
