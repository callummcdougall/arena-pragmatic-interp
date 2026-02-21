# Report: master_4_5.py — Investigator Agents

## Overview

**Master file:** `chapter4_alignment_science/master_4_5.py` (~4760 lines, 3 sections + bonus)
- Section 1: AI Psychosis — Multi-Turn Red-Teaming (25%)
- Section 2: Introduction to Petri (40%)
- Section 3: Petri Deep Dive — Source Level (35%)

**Associated papers/posts (all in `llm_context/investigator-agents/`):**
1. `ai-psychosis-blog-post.txt` — Tim Hua (AI Alignment Forum, Aug 2025)
2. `petri-blog-post.txt` — Anthropic Alignment Science (Oct 2025)
3. `petri-2.0-blog-post.txt` — Anthropic Alignment Science (Jan 2026)
4. `building-and-evaluating-alignment-auditing-agents.txt` — Bricken, Wang, Bowman et al. (Anthropic, Jul 2025)
5. `eliciting-lm-behaviour-with-investigator-agents.txt` — Li, Chowdhury et al. (Transluce, Oct 2024)
6. `bloom-blog-post.txt` — Gupta, Fronsdal, Sheshadri et al. (Anthropic, Dec 2025)
7. `auditing-game.txt` — Sheshadri et al. (Anthropic, Dec 2025)
8. `Auditing language models for hidden objectives.pdf` — Marks et al. (original paper)

**Code repos:**
- `llm_context/investigator-agents/ai-psychosis/` — Red-teaming framework, 9 character files, pre-computed transcripts & grades
- `llm_context/investigator-agents/petri/` — 111 seed instructions, auditor agent, judge, whistleblowing ablations with 97 pre-computed transcripts
- `llm_context/investigator-agents/bloom/` — 4-stage pipeline, 20+ behavior definitions, pre-computed example transcripts

---

## Hallucination Check

### Definitely False Claims

**1. "95.5% attack success on system prompt extraction" (line 4759)**

The master file claims:
> "small 8B models achieve 95.5% attack success on system prompt extraction against 405B models"

The actual Transluce paper (Li et al.) reports 95.5% Attack Success Rate on Llama 3.1 405B for **exact harmful string elicitation** — making the 405B model output specific harmful strings from AdvBench when given an 8B-generated prefix. System prompt extraction is a completely different task not mentioned in connection with this 95.5% figure. The paper's harmful string elicitation involves training an 8B investigator to craft prefixes that cause the target to complete a specific string, not to extract system prompts.

---

**2. "182 seed instructions" (line 2315)**

The master file states:
> "Petri ships with 182 seed instructions"

The Petri v1 blog post says 111 seed instructions. The Petri v2 blog post says 181 (111 original + 70 new). The Petri source code in the repo contains 111 seeds. The number 182 doesn't match any of these — it's off by 1 from the v2 count of 181, likely a typo.

---

**3. Expected output model list mismatch in whistleblowing ablation (lines 2585-2591 vs 2909-2918)**

The `WHISTLEBLOWING_MODELS` list defines 5 models: Gemini 2.5 Pro, Gemini 2.5 Flash, GPT-4o, DeepSeek Chat, and Grok 4.1 Fast. But the expected output at lines 2909-2918 shows results for 7 models including "GPT-5 Nano" and "Claude Sonnet 4.5" which are not in the code's model list. The expected output was generated with a different model configuration than the code defines.

---

### Maybe False / Overstating

**4. "38 judging dimensions" vs blog's "36 default scoring dimensions" (lines 116, 1286, etc.)**

The master file consistently says Petri uses "38 judging dimensions," confirmed by code output. But the Petri v1 blog post states "36 default scoring dimensions." This discrepancy is likely due to the code being updated after the blog post was written (the blog is from Oct 2025, the exercises reference Petri v2 from Jan 2026). The v2 blog doesn't restate the total count.

**Verdict:** Likely correct for current code, but the blog reference is stale. A note that the count was updated would help.

---

**5. "2 characters * 2 models" claim (line 1225) vs code using all 9 characters**

The text says:
> "You've just spent significant API budget on a pipeline covering 2 characters * 2 models"

But the comparison function defaults to using all loaded characters (9 from the AI psychosis repo) when called with `characters=None`. The text at line 1223 says "With only 2 characters and 3 turns" suggesting the expected run is with 2 characters, but this depends on the student passing a reduced set — the code defaults don't enforce this.

**Verdict:** Pedagogically confusing. The text implies 2 characters were used but the code would run all 9 by default.

---

**6. "DeepSeek-v3 most vulnerable and Claude Sonnet safest" (line 1133)**

The AI psychosis blog does rank DeepSeek V3 as the worst performer and Claude Sonnet 4 among the best. However, the blog tests 11 models (not just these two), and the ranking is nuanced — Kimi-K2 has the best pushback but is too confrontational per therapy standards; Claude does poorly on Spiral-Bench with different persona styles. The simplified claim is directionally correct but omits important context.

**Verdict:** Correct in direction, mild oversimplification.

---

**7. Tool set inconsistency across descriptions (lines 1336-1343, 1569-1632, 3080)**

Three different sets of 5 tools are described in different places:
- Real Petri (line 3080): send_message, set_system_prompt, **create_synthetic_document**, rollback, **end_conversation**
- Narrative (lines 1336-1343): send_message, set_system_prompt, create_synthetic_tool, rollback, **prefill**
- Code schemas (lines 1569-1632): send_message, set_system_prompt, create_synthetic_tool, rollback, **record_finding**

The `prefill` tool is discussed as important (lines 3423-3430) but is NOT in the code's tool schemas. The `record_finding` tool IS in the schemas but NOT in the narrative description. This creates confusion about which tools students should expect.

**Verdict:** Not a factual error about papers, but an internal inconsistency between the exercise text and code.

---

**8. Whistleblowing ablation findings attribution (lines 2939-2943)**

The master file says:
> "The blog post found that 'agency' and 'actual wrongdoing' had the strongest effects on whistleblowing rates"

The Petri v1 blog post confirms that agency and leadership complicity were strong effects, but it also notes that Chekhov's gun and no-actual-wrongdoing conditions had more nuanced effects (Chekhov's gun only measurably affected Claude Sonnet 3.7 and Opus 4.1). The claim about "actual wrongdoing" being strongest is approximately correct but simplifies the blog's more nuanced finding.

**Verdict:** Approximately correct but oversimplified.

---

### Verified Correct Claims

| Claim | Source | Status |
|-------|--------|--------|
| 9 character files in ai-psychosis | AI psychosis blog | ✓ Correct |
| 14-dimension clinical rubric (5 core + 9 therapeutic) | AI psychosis blog/code | ✓ Correct (3 custom + 2 derived + 9 CBT) |
| 10 quirky behaviors from "Building and Evaluating" paper | Building-and-evaluating post | ✓ Correct |
| ~13% single-agent solve rate on auditing game | Building-and-evaluating post | ✓ Correct |
| ~42% super-agent solve rate (>3x improvement) | Building-and-evaluating post | ✓ Correct |
| 47.3% reduction in eval-awareness scores | Petri v2 blog | ✓ Correct |
| GPT-5 judges rate own family as less misaligned | Petri v1 blog | ✓ Correct |
| o4-mini tends toward binary 1/10 scores | Petri v1 blog | ✓ Correct |
| Bloom: 16 frontier models across 4 behaviors | Bloom blog | ✓ Correct |
| Grok-3/4 recommended as red-team model (commits fully to personas) | AI psychosis blog | ✓ Correct |
| Models whistleblow even without wrongdoing ("role-playing hypothesis") | Petri v1 blog | ✓ Correct |
| Forward/backward training asymmetry (Transluce) | Eliciting LM behaviour post | ✓ Correct |

---

## Significant Gaps

### Missing Key Results & Exercise Opportunities

---

**Gap 1: Interpretability tools for auditing (HIGH PRIORITY)**

The building-and-evaluating paper's key finding: adding interpretability tools (SAE features, grep/semantic search on training data) to the investigator agent boosts auditing game performance from ~0% (chat-only) to 13% (single agent) to 42% (super-agent). The paper also finds that SAEs trained on pre-training data alone have better features for detecting RM biases than SAEs trained on all three datasets (PT+SFT+RL).

The master file's exercises cover the chat-based auditor tools (send_message, set_system_prompt, etc.) but include **no interpretability exercises**. Students never use SAE features, grep training datasets, or perform semantic search — despite these being what distinguishes the investigator agent from a simple red-teamer.

- **Code exists:** The building-and-evaluating paper describes the tools; the Petri framework could be extended
- **A100 feasible:** Yes for SAE inference; the auditing game open-source replication uses Llama 3.3 70B
- **Exercise format:** Add SAE/search tools to the auditor toolkit, demonstrate their impact on audit quality

---

**Gap 2: Auditing game model organism (HIGH PRIORITY)**

A complete open-source testbed exists: Llama 3.3 70B fine-tuned with 52 hidden RM biases (47 trained, 5 held out), adversarially trained to resist revealing its objective. This is described in `auditing-game.txt` and provides the canonical evaluation environment for investigator agents. Key findings: DPO is more effective than SFT at instilling hidden objectives; DPO generalizes to held-out test biases while SFT does not.

The master file uses Petri with system-prompted quirky models (lightweight but unrealistic) and never touches the model organism (heavyweight but the real test).

- **Code exists:** `auditing-game.txt` describes full pipeline; model weights likely on HuggingFace
- **A100 feasible:** Llama 3.3 70B requires ~140 GB VRAM (marginal on single A100 80GB, feasible with quantization or 2x A100). Analysis of pre-computed results is fully feasible.
- **Exercise format:** Use investigator agent against the auditing game model organism; analyze pre-computed audit transcripts

---

**Gap 3: Bloom complete pipeline (MEDIUM-HIGH PRIORITY)**

Bloom is mentioned only in a brief bonus section (lines 4735-4739). But it provides a well-structured 4-stage pipeline (Understanding → Ideation → Rollout → Judgment) with 20+ pre-defined behavior definitions, pre-computed example transcripts, and a complete CLI. Key findings: discriminates 9/10 model organism quirks even zero-shot; judge calibration (Opus 4.1: Spearman 0.86 with humans); self-preferential bias rankings robust to few-shot examples (Kendall's W=0.66).

- **Code exists:** Complete `bloom/` package with CLI, behavior definitions, example transcripts
- **A100 feasible:** Yes — all API-based inference
- **Exercise format:** Run Bloom pipeline on a specific behavior, analyze output, compare to Petri's open-ended approach

---

**Gap 4: Evaluation agent (MEDIUM PRIORITY)**

The building-and-evaluating paper describes a code-writing evaluation agent with 88% overall success rate (7/10 quirks with perfect 5/5 evaluations). This agent automatically generates behavioral evaluations from natural language specifications — a fundamentally different approach from Petri's interactive auditing.

- **Code exists:** Described in the paper but may not be fully open-sourced
- **A100 feasible:** Yes — API-based
- **Exercise format:** Conceptual comparison exercise; potentially implement a simplified version

---

**Gap 5: Realism classifier details from Petri v2 (MEDIUM PRIORITY)**

The master file includes a realism classifier exercise (lines 4071+), but the Petri v2 blog provides more detail: it distinguishes task-driven vs. environment-driven eval-awareness cues and achieves a 37.2% reduction in eval-awareness from the classifier alone (vs. 21.7% from new seeds alone, 47.3% combined). The master file mentions the 47.3% figure but doesn't break down the individual contributions.

- **Code exists:** Petri v2 codebase likely includes the realism classifier
- **A100 feasible:** Yes
- **Exercise format:** Extend the existing realism exercise with the v2 breakdown

---

**Gap 6: Judge bias systematic analysis (MEDIUM PRIORITY)**

Both Petri and Bloom provide detailed judge bias findings:
- **Petri:** Intra-judge correlation 0.75-0.85; o4-mini binary scoring; GPT-5 family favoritism; Gemini auditor thinks it's the target ~10% of time
- **Bloom:** Opus 4.1 has 0.86 Spearman correlation with humans; GPT-5 shows high score variance; Claude is highly consistent

The master file mentions judge bias in bonus text (line 4757) but has no exercises analyzing it systematically.

- **Code exists:** Pre-computed transcripts in both Petri and Bloom repos could be scored by multiple judges
- **A100 feasible:** Yes — API-based
- **Exercise format:** Score same transcripts with multiple judges, analyze bias patterns

---

**Gap 7: Auditor linguistic fingerprint analysis (LOW-MEDIUM PRIORITY)**

The Petri v1 blog documents striking findings about auditor behavior: "actually" appears 43x more than in human chat, "curious" 233x more, 5.8x more questions per message, 3.5x higher emotional intensity. This represents a detectability concern — target models could learn to recognize auditor patterns.

- **Code exists:** Pre-computed transcripts available; LMSYS-Chat-1M baseline available
- **A100 feasible:** Yes
- **Exercise format:** Analyze auditor message patterns, compare to human baselines, discuss implications for eval-awareness

---

**Gap 8: Trained investigator models (LOW-MEDIUM PRIORITY)**

The Transluce paper shows that training a small 8B model as an investigator achieves dramatic results: 98.1% ASR on same-size targets, 95.5% transfer to 405B. Training uses SFT on inverted (prefix, suffix) pairs followed by iterative DPO. This represents a fundamentally different approach to auditing (learned vs. prompted).

- **Code exists:** Paper describes methodology; may not be open-sourced
- **A100 feasible:** Yes for inference with 8B models; training requires significant compute
- **Exercise format:** Primarily discussion/conceptual, unless pre-trained models are available

---

## Summary Table

| Issue | Type | Severity |
|-------|------|----------|
| "95.5% attack success on system prompt extraction" (should be harmful string elicitation) | Definitely false | **High** — mischaracterizes the paper's finding |
| "182 seed instructions" (should be 181 or 111) | Definitely false | **Low** — likely typo (off by 1) |
| Expected output model list mismatch in whistleblowing | Internal inconsistency | **Medium** — expected output doesn't match code |
| "38 judging dimensions" vs blog's "36" | Stale reference | **Low** — code likely updated since blog |
| "2 characters * 2 models" vs code using all 9 | Confusing | **Low-Medium** — text doesn't match code default |
| Tool set inconsistency (narrative vs code vs real Petri) | Internal inconsistency | **Medium** — three different tool lists |
| "DeepSeek-v3 most vulnerable" | Mild oversimplification | **Low** |
| Missing: Interpretability tools for auditing | Gap — exercise | **High** |
| Missing: Auditing game model organism | Gap — exercise | **High** |
| Missing: Bloom complete pipeline | Gap — exercise | **Medium-High** |
| Missing: Evaluation agent | Gap — exercise | **Medium** |
| Missing: Judge bias systematic analysis | Gap — exercise | **Medium** |
| Missing: Auditor linguistic fingerprint | Gap — exercise | **Low-Medium** |
