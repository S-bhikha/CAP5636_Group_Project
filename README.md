# Fact-Constrained Story Generation

**CAP 5636 Final Project** — TinyStories-scale LM + factual narrative adaptation

**Team (2–3):** Sahil Bhikha · Thomas Belyakov · David Almeida II

---

## One-line summary

**Pretrain** a small decoder-only Transformer on **TinyStories**, then **adapt** it so it can write short **stories that stay faithful to a provided fact card**, and measure the **faithfulness–narrative quality tradeoff** against matched controls.

## Research question

Under a fixed small-model and token budget, which adaptation strategy best improves **fact faithfulness** of short educational stories **without destroying narrative quality**, relative to TinyStories-only and pure encyclopedic continued pretraining?

## Contribution (paper framing)

We study **fact-constrained story generation** with a TinyStories-scale decoder-only LM. We build a small evaluation suite of topic–fact-card–prompt triples and compare matched-budget adaptation recipes (story-only baseline, Wiki continued pretraining, and task-specific SFT / factual-narrative adaptation). We characterize the **faithfulness vs story-quality** tradeoff and analyze failure modes (contradiction, omission, unconstrained invention, encyclopedia dump).

This is a **controlled empirical study**. We do **not** claim general truthfulness, educational safety, or open-world hallucination reduction.

---



## Course alignment (must ship)

This section maps CAP 5636 Final Project Guidelines to this project so nothing is optional by accident.

### Project type (acceptable formats)

| Course-allowed type | Our project |

| --- | --- |

| **Empirical study** (primary) | Systematically compare B0 / B1 / M2 under equal Stage-2 budgets with fixed metrics |

| Implementation + extension | Lab GPT pipeline extended with fact-card task, SFT arm, dual-axis eval |

| Application / novel method | Not our primary claim |

**Not acceptable (we avoid):** pure survey; demo-only product; “use ChatGPT to do X” without training, controls, and analysis.

**Course themes covered:** deep learning architectures, LLMs, pretraining, continued pretraining / fine-tuning (SFT), evaluation of generation quality and faithfulness.

### Grade composition (what we optimize for)

| Component | Weight | Our concrete artifact |

| --- | --- | --- |

| **Final paper** | **60%** | 6–8 pp NeurIPS-style PDF via Webcourses |

| **Code & reproducibility** | **20%** | GitHub repo + this README (deps + repro steps) + organized code |

| **Oral presentation** | **20%** | 15 min (10 talk + 5 Q&A), ~10–12 slides, **all members speak** |

### Paper requirements (60%)

**Format**

- Length: **6–8 pages**, NeurIPS-style template (single column, 10pt; references not counted)
- Submit as **PDF** via Webcourses by the due date
- Repo link in the paper
- **Individual contributions** of each team member identified in the paper
- **AI Tools** section at end: tools used and purpose (required if any generative AI used)

**Required sections (checklist for draft)**

| Section | What we put here |

| --- | --- |

| Abstract (~150 words) | Question, method sketch, main faithfulness-vs-story finding |

| Introduction | Problem (fluent but ungrounded TinyStories); motivation; contribution bullets |

| Related Work | TinyStories; CPT / domain adaptation; instruction tuning / SFT; factuality & constrained generation |

| Methods | Architecture; Stage 1/2; fact-card task; each system (B0/B1/M2); enough detail to reproduce |

| Experiments | Data splits; budgets; hardware; baselines; ablations; decoding fixed across systems |

| Results & Discussion | Tables + **main figure** (faithfulness vs story quality); qualitative samples; **error analysis** |

| Conclusion & Future Work | Takeaways; limits; what we refuse to claim |

| References | Cited prior work |

| Contributions | Who owned data / train / eval / writing / slides (by name) |

| AI Tools | ChatGPT/Claude/Copilot/etc. — coding, debugging, writing support only as disclosed |

**Paper rubric → how we earn it**

| Rubric slice | Weight (of paper) | How this project hits “excellent” |

| --- | --- | --- |

| Technical execution | 30% | Correct small-LM training; frozen eval IDs; equal budgets; runnable configs |

| Empirical rigor | 20% | Non-trivial baselines (B0 + B1); dual metrics; ablations if time; honest negatives |

| Originality of contribution | 15% | Tradeoff characterization + fact-card protocol — not bare reproduction of the lab |

| Writing & structure | 20% | Clear narrative; Pareto figure; tables that aid understanding |

| Literature grounding | 15% | Position vs TinyStories, CPT, factuality eval — not “we built a story app” |

### Code & reproducibility requirements (20%)

Course expects:

1. Public or private **GitHub** repository; link in the paper
2. **README** with: project overview, **dependencies**, **instructions to reproduce key results**
3. Reasonable code quality (organized, commented at key points, no dead files)

**This repo must eventually include (checklist)**

- [ ] Overview (this doc)  

- [ ] Dependencies `requirements.txt` / env file + versions)  

- [ ] How to prepare data (TinyStories, fact cards, SFT pairs)  

- [ ] How to run Stage 1 smoke + full pretrain  

- [ ] How to run Stage 2 arms (B1 CPT, M2 SFT) with **matched budgets**  

- [ ] How to generate eval stories (fixed decoding)  

- [ ] How to score / aggregate metrics and regenerate the main figure  

- [ ] Where checkpoints and `results/<run_id>/` live  

- [ ] No orphaned dead scripts in the final hand-in  

Until repro commands exist, the **20% code** slice is incomplete even if training works.

### Oral presentation requirements (20%)

| Rule | Our plan |

| --- | --- |

| 15 minutes total | 10 min presentation + 5 min discussion |

| ~10–12 slides | Motivation → task/fact card → methods → **key figure** → samples → failures → takeaways |

| All team members speak | Lane A data/task; Lane B training; Lane C eval/results (adjust after owners assigned) |

| Demo optional | Side-by-side generations encouraged (same prompt, B0 vs M2) |

| Not a paper readout | Highlight tradeoff + one figure; details stay in the paper |

**Presentation rubric → plan**

| Slice | Weight | Plan |

| --- | --- | --- |

| Content & clarity | 40% | Fixed research question; show controls; state limits |

| Visual aids | 20% | Pareto plot + 1–2 generation comparisons; minimal wall-of-text |

| Delivery | 20% | Timed dry run; every member has a substantive section |

| Discussion & Q&A | 20% | Prep answers on baselines, eval subjectivity, what we don’t claim |

### Use of generative AI (course policy)

Allowed for coding assistance, debugging, and writing support. **Must** document in paper **AI Tools** section (tool + purpose). Submitting AI-generated analysis as original undisclosed work is academic dishonesty.

**Team rule:** any LLM use for draft prose, code, synthetic SFT seeds, or secondary judging is logged for the AI Tools section as we go.

### Course failure modes → our countermeasures

| Course warning | Our countermeasure |

| --- | --- |

| Scope creep | Locked task (fact cards); optional arms only after B0/B1/M2; decision gates below |

| Weak baselines | **B1 Wiki CPT is required**, not optional — stops “SFT beats doing nothing” |

| Implementation without analysis | Dual metrics + error taxonomy + qualitative samples in Results & Discussion |

| Last-week writing | Draft Methods / Related Work / task definition as soon as schema freezes |

| Presentation as paper readout | Slide outline fixed to motivation, approach, key results, takeaways |

---



## Task definition

Each eval/train item is closed-world and scoreable:

| Field | Role |

| --- | --- |

| **Topic** | e.g. water cycle, seasons, a simple historical figure |

| **Fact card** | 4–7 gold bullets the story may teach; optional notes on common false claims |

| **Prompt** | “Write a short story (≈120–180 words) that teaches this topic. Invent characters and plot freely; do **not** invent facts beyond the card.” |

| **Output** | One short story |

**Success axes**

1. **Faithfulness** — required-fact coverage; low contradiction / invention relative to the card
2. **Story quality** — narrative structure, coherence, age-appropriate voice (not a bullet list or textbook dump)

**Data scale (target)**

- Train fact cards / SFT pairs: on the order of **80–200** topics  
- Held-out eval: **40–60** frozen topic IDs (never used for prompt tuning or SFT labels)



## Training stages

| Stage | Term | What happens | Starts from |

| --- | --- | --- | --- |

| **1** | **Pretraining** | Next-token LM training on TinyStories | Random init (from scratch) |

| **2** | **Adaptation** | Matched-budget continued pretraining and/or SFT for the fact-constrained story task | Stage-1 checkpoint |

Stage 2 may include classic **SFT** (fact card + prompt → story), not only continued pretraining. Use precise terms in the paper for each arm.

**Locked decision:** Stage 1 = pretrain from scratch on TinyStories (lab-scale GPT). Do not start from a public pretrained model (e.g. SmolLM2) unless Stage 1 fails after smoke.

**Lane B training code + commands:** [`TRAINING.md`](./TRAINING.md).

## Why this project

TinyStories yields fluent simple narrative but freely invents world knowledge. Pure encyclopedic continued pretraining may improve “fact-ish” language while **eroding story form**. Task-specific adaptation may improve checklist faithfulness with a different quality cost. Under a single-GPU budget we measure that tradeoff with fixed protocols—not vibes and not perplexity alone.

## Relation to the Week 6 LLM lab

Lab notebook: `CAP5636_W6_Transformer(LLM).ipynb`](./CAP5636_W6_Transformer(LLM).ipynb)

| Lab module | What it teaches | Project use |

| --- | --- | --- |

| 1 | Decoder-only GPT | Small Transformer architecture |

| 2 | BPE on TinyStories | Tokenizer / data prep |

| 3 | Next-token pretraining on TinyStories | **Stage 1** |

| 4 | Temperature / top-k / top-p | Eval decoding (**fixed across systems**) |

| 5 | Adaptation after pretraining | **Stage 2** (CPT and/or SFT) |

| SmolLM2-135M demo | ~100M-class reference | Size-class reference only |

Lab reference scale (approximate): `n_layer=6`, `n_embd=256`, `n_head=8`, `vocab_size=8000`, `block_size=256`, dataset `roneneldan/TinyStories`.

---



## Experiment plan (deadline-safe)

**Hardware:** student RTX 5090 preferred; Newton optional.

Keep Stage-2 budgets **equal** across compared runs. Save intermediate checkpoints for a cheap duration/data ablation when possible.

### Core systems (minimum for a strong paper)

| ID | System | Role |

| --- | --- | --- |

| **B0** | TinyStories-only (Stage 1) | Narrative prior; faithfulness floor |

| **B1** | B0 + Wikipedia continued pretraining | Encyclopedic / “more world text” control (**required baseline**) |

| **M2** | B0 + SFT on (fact card + prompt → story) | Primary task adaptation |

**If time allows**

| ID | System | Role |

| --- | --- | --- |

| **M1** | B0 + CPT on factual narratives or TS + factual mixture | Domain-style adapt without explicit SFT |

| **M3** | B1 then light SFT | Does Wiki help or hurt as a middle step? |

| **Prompt ablation** | B0 / M2 with vs without fact card in context | In-weights skill vs prompt-following |

**Deadline triage:** if only three full runs fit, ship **B0, B1, M2** — that set is the empirical study.

### Evaluation protocol

Primary metrics (freeze before final runs):

| Axis | Measure |

| --- | --- |

| **Faithfulness** | Per-bullet coverage; contradiction rate; optional invention rate vs card |

| **Story quality** | Rubric (structure, coherence, simplicity); or binary “is a story?” + fluency |

| **Supporting automatic** | Length, repetition; held-out TinyStories loss (prior retention); optional domain loss |

**Scoring:** blind human rubric as primary claim (team raters, short calibration set); report agreement on a double-scored subset. LLM-as-judge only secondary + audited, if used—disclose in AI Tools.

**Main paper figure:** faithfulness vs story quality for B0 / B1 / M2.

**Error analysis (required for Results & Discussion):** omission, contradiction, unconstrained invention, story collapse (encyclopedia dump), story domination (plot with no teaching content).

### Planned repo layout

```text

data/

  fact_cards/train.jsonl

  fact_cards/eval.jsonl          # frozen IDs

  sft_pairs/train.jsonl          # if used

  prompts/eval_prompts.jsonl

eval/

  [rubric.md](http://rubric.md)

  score_sheet_template.csv

results/<run_id>/

  config.yaml

  metrics.json

  samples/

paper/                           # NeurIPS source + PDF (or link path)

slides/                          # final deck

```



## What we are not claiming

- Open-world truthfulness or “no hallucinations” in general  
- Perplexity alone as proof of factual stories  
- Reproducing large-LM token counts or multi-GPU training as requirements  
- RAG, chat UI, agents, or RL alignment as core deliverables  
- Real educational product readiness or child-safety guarantees

---



## Team work split

Three equal lanes. **Assign names before execution**; the paper Contributions section must match reality.

| Lane | Owner | Owns | Does not own |

| --- | --- | --- | --- |

| **A — Data** | `_assign_` | TinyStories packaging; fact cards (train/eval split); SFT pair construction/verification; manifests; license notes | Training hyperparameters; final paper prose alone |

| **B — Train** | `_assign_` | Model/config (lab GPT); Stage 1 + Stage 2 scripts; smoke + full runs; checkpoints; run cards | Rubric finalization; slides-only work |

| **C — Eval / paper** | `_assign_` | Rubric + scoring; blind sheets; main figure; paper integration; slides structure; **repro section of README** | Shard format internals; GPU babysitting (unless helping B) |

**Shared by all three:** design decisions, interpreting results, paper review, **AI Tools log**, presentation speaking roles, **named individual contributions**.

### Parallel vs sequential work

Critical path is roughly:

```text

Schema freeze

    ├─► (parallel) fact cards / SFT pairs ──► Stage 2 SFT (M2) ──┐

    ├─► (parallel) TinyStories + train code ─► Stage 1 (B0) ─► Stage 2 Wiki CPT (B1) ─┼─► gen eval ─► score ─► Results/figure ─► paper freeze ─► slides polish

    └─► (parallel) rubric + paper skeleton + eval harness (until checkpoints exist) ──┘

```

**GPU time (Stage 1 → Stage 2) and final scoring are the main serial bottlenecks.** Almost everything else can overlap if owners do not idle on the critical path.

#### Can run in parallel (do these at the same time)

| Workstream | Owner | Parallel with | Notes |

| --- | --- | --- | --- |

| Fact-card schema + seed cards (≥10) | A | B train-code port; C rubric/paper skeleton | Needs a short **joint freeze** of schema fields first (30–60 min sync), then A/C diverge |

| Full train/eval fact cards | A | Stage 1 full train (B); Methods/Related Work draft (C) | Eval IDs can freeze before all train cards exist |

| SFT pair writing/verification | A | Stage 1 (B); eval harness + rubric calibration (C) | M2 cannot **start** until pairs exist, but pairs can be built during Stage 1 |

| TinyStories download/packaging | A | Fact-card authoring (same lane splits time); B code; C paper | Needed for Stage 1; small smoke subset first |

| Train code port, configs, smoke scripts | B | All of A’s early data work; C paper/rubric | Stage-1 **smoke** only needs tiny data, not full cards |

| Stage 1 full pretrain (B0) | B | A finishes cards + SFT pairs; C drafts non-results paper sections | Long wall-clock — others must not wait idle |

| Wiki CPT prep (subset + packer) | A or B | Late Stage 1; SFT pair polish | B1 needs Wiki data + B0 ckpt; prep Wiki **before** B0 finishes |

| Rubric + score sheet + calibration examples | C | A cards; B training | Do **not** wait for final models |

| Paper skeleton: Intro / Related Work / Methods / Experiments-setup | C | Entire training period | Fill Results last |

| Eval harness (load ckpt → generate → dump JSON) | C | Stage 1 / early Stage 2 | Integrate against smoke ckpt first |

| Slides shell (title, outline, speaking order) | C (+ all) | Mid project | Leave result slides blank until freeze |

| Dependencies list + partial README repro | C or B | Anytime after smoke | Full repro needs final run commands |

| AI Tools running log | All | Always | Append as you go; paste into paper at end |

#### Cannot fully parallelize (hard dependencies)

| Blocked work | Waits on | Why |

| --- | --- | --- |

| **Schema / prompt format freeze** | Short team sync | A’s cards, C’s rubric, and B’s SFT loader must agree on fields |

| Stage 1 **full** train | Train code smoke green + TinyStories available | No checkpoint → no B0, B1, M2 |

| Stage 2 **B1** (Wiki CPT) | **B0 checkpoint** + Wiki subset ready | Continued pretraining starts from Stage 1 |

| Stage 2 **M2** (SFT) | **B0 checkpoint** + **verified SFT pairs** | Task data + init weights |

| Matched-budget comparison | Same Stage-2 token/step rules decided before launch | Changing budget mid-run invalidates the study |

| Eval **generation** for final table | Frozen eval card IDs + finished ckpts (B0/B1/M2) + fixed decoding | All systems must use the same prompt sheet |

| **Blind scoring** | Generated stories for all compared systems, anonymized | Scoring before all arms exist biases the paper |

| Main **faithfulness vs story figure** | Aggregated scores | Results section and key slide depend on this |

| Results & Discussion (final) | Figure + error-analysis sample | Course requires analysis, not only methods |

| Conclusion (final claims) | Results freeze | Avoid rewriting claims after every new run |

| Full **repro README** (“reproduce key results”) | Final configs, seeds, paths, one clean command path | Course code 20% |

| Presentation **results** slides + dry run with numbers | Results freeze | Can dry-run structure earlier without numbers |

| Paper **Contributions** names | Lane owners assigned + work actually done | Must match reality |

#### Soft dependencies (partial overlap OK)

| Work | Can start early | Must finish after |

| --- | --- | --- |

| SFT training (M2) | Loader + tiny toy pairs during smoke | Full pair set for the reported run |

| Eval harness | Smoke checkpoint + 2 dummy cards | Real eval split + all system ckpts |

| Qualitative sample picking | Anytime after first generations | Prefer final decoding settings |

| Error taxonomy labels | Rubric freeze | Apply to final system outputs |

| Related Work citations | Immediately | Light edit once claims are final |

| Speaking parts rehearsal | Outline locked | One timed run after result slides exist |

#### Who is idle when (avoid this)

| If you are… | And you are blocked on… | Do this instead |

| --- | --- | --- |

| **A** waiting on B’s Stage 1 | GPU | More/better SFT pairs; Wiki subset; data appendix; license notes; token estimates |

| **B** waiting on A’s full SFT set | Pairs | Stage 1; Wiki CPT (B1); logging/run cards; clean train CLI; smoke M2 on 5 pairs |

| **B** waiting on Wiki pack | Data | Hyperparams note; intermediate ckpt schedule; equal-budget config templates for B1/M2 |

| **C** waiting on any ckpt | Models | Paper Methods/Related Work/Experiments; rubric calibration on **hand-written** good/bad stories; harness against smoke; slides shell; contrib/AI Tools placeholders |

| **Everyone** waiting on scores | Human rating | Parallelize rating (split items, then double-score a subset for agreement); draft figure code on fake metrics |

#### Recommended phase plan (parallelism explicit)

| Phase | Wall focus | A | B | C | Sync points |

| --- | --- | --- | --- | --- | --- |

| **0 — Align** (few hours) | Unblock everyone | Join schema design | Join schema + I/O contract | Join schema + rubric axes | **Freeze:** fact-card JSON fields, prompt template, train/eval split policy |

| **1 — Smoke** (~1–2 days) | Parallel ramp | TinyStories smoke + ≥10 cards | Code port + Stage-1 smoke | Rubric v0 + paper skeleton + metrics schema | Stage-1 smoke green; schema not changing anymore |

| **2 — Build** (overlap heavily) | Max parallelism | Full cards + SFT pairs + Wiki prep | **Stage 1 full (B0)** | Methods/Related Work/Experiments draft; eval harness; slides shell | Eval **IDs frozen** before anyone tunes on them |

| **3 — Adapt** (GPU serial-ish) | Stage 2 | Pair QC; manifests; help B if needed | **B1 then/or M2** (matched budget); run cards | Harness dry runs on B0; fill paper non-results | Budget equality check before launch |

| **4 — Measure** (serial tail) | Eval | Data appendix; repro data commands | Package ckpts; no new trains after freeze | Generate all systems → blind score → figure + error analysis | **Results freeze** date |

| **5 — Ship** (parallel finish) | Course artifacts | Review paper data claims | Repro train commands; delete dead code | Paper PDF; slides; README repro; AI Tools/Contributions | Dry-run talk; PDF submit |

**Parallelism rule of thumb:** Phases 1–2 should have all three people busy without waiting on final models. Phases 3–4 are dominated by GPU then scoring—plan so A and C never sit idle during long trains.

### Suggested calendar (maps to phases)

| When | Lane A | Lane B | Lane C |

| --- | --- | --- | --- |

| **Now → +2 days** (Phases 0–1) | Schema + ≥10 cards; TinyStories smoke data | Port/pin training code; Stage-1 smoke green | Rubric draft; paper skeleton (all required sections); metrics schema |

| **Next 3–4 days** (Phases 2–3) | Full train/eval cards; SFT pairs verified; Wiki prep | Stage 1 complete; Stage 2 for B1 + M2 (matched budget) | Eval harness; draft Methods / Related Work / Experiments setup |

| **Final 3–4 days** (Phases 4–5) | Data appendix + rebuild commands | Package configs + run cards; clean dead code | Blind scores; results freeze; paper PDF; slides; README repro commands |

### Handoffs (serial edges)

1. **Team → all:** schema + prompt + split policy freeze
2. **A → B:** TinyStories paths (Stage 1); SFT pair paths (M2); Wiki paths (B1); token estimates; rebuild commands
3. **B → B:** B0 checkpoint → starts B1 and M2
4. **B → C:** checkpoint paths + run cards under `results/<run_id>/`
5. **C → C:** generations → anonymized sheets → scores → figure
6. **C → team:** faithfulness–story table, samples, paper/slides draft for review
7. **All → submit:** PDF + repo + slides; Contributions + AI Tools match the log



### Decision gates

- No Stage-1 smoke in ~2 days → shrink model/steps; do not add optional arms  
- Stage 1 OK but Stage 2 too slow → cut Stage-2 tokens **equally** for all adaptation runs  
- Only one adaptation run finishes → prioritize **M2**; still report **B0** and partial **B1** if possible  
- No comparable Stage-2 pair in time → freeze what exists; honest pilot + limitations (still ship paper/code/slides)  
- **Results freeze** missed → stop training; write with what is scored; do not open new experimental threads

---



## Deliverables checklist (course hand-in)

Use this as the final “are we done?” list.

### A. Final paper (Webcourses PDF)

- [ ] 6–8 pages, NeurIPS-style (10pt, single column)  

- [ ] All required sections present (see table above)  

- [ ] Repo link in paper  

- [ ] Individual contributions by name  

- [ ] AI Tools section  

- [ ] Main results figure + error analysis in Results & Discussion  



### B. Code repository

- [ ] Train + eval code organized, key comments, no dead files  

- [ ] Dependencies listed  

- [ ] README instructions reproduce **key** tables/figures  

- [ ] Fact cards, prompts, run configs, and sample outputs available as needed for repro  



### C. Presentation

- [ ] ~10–12 slides  

- [ ] 10 + 5 minute timing rehearsed  

- [ ] All three members have speaking parts  

- [ ] Motivation, approach, key results, takeaways (not full paper readout)  

**Soft deadline:** 2026-07-25 · **Hard deadline:** 2026-07-27

---



## Status

- [x] Course requirements mapped into this README  

- [x] Project type locked: **controlled empirical study**  

- [x] Technical direction locked: **fact-constrained stories** (B0 / B1 / M2 + dual-axis eval)  

- [x] Lab connection documented  

- [ ] Assign lane owners (A / B / C) → feeds paper Contributions  

- [ ] Fact-card schema + seed cards + `eval/rubric.md`  

- [ ] Stage-1 smoke  

- [ ] Matched Stage-2 runs + scored eval  

- [ ] Paper PDF (all required sections)  

- [ ] Slides + speaking order  

- [ ] README repro commands + dependencies (code 20%)  

- [ ] AI Tools log ready to paste into paper  

---



## Data corpora & licenses

Local download/packaging: [`data/README.md`](./data/README.md) · rebuild: `python scripts/download_data.py`

| Corpus | Role | License (declared) | Acknowledgment |
| --- | --- | --- | --- |
| **TinyStories** (`roneneldan/TinyStories`) | Stage 1 / B0 | **[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)** | Cite Eldan & Li (2023); share-alike + attribution if *publishing* data; raw dumps not in git |
| **Simple English Wikipedia** (`wikimedia/wikipedia`, `20231101.simple`) | Stage 2 B1 CPT | **CC BY-SA** + **GFDL** | Attribute Wikipedia/Wikimedia contributors; link license; note processing changes |

**Full obligations, paper copy-paste block, and hand-in checklist:** [`data/LICENSES.md`](./data/LICENSES.md) (machine-readable: [`data/licenses.json`](./data/licenses.json)).

---

## References

1. CAP 5636 Final Project Guidelines (Webcourses)
2. CAP 5636 Week 6 LLM lab notebook (this repo)
3. Eldan & Li, *TinyStories* ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759)); dataset [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)
4. Wikimedia Wikipedia dumps ([dumps.wikimedia.org](https://dumps.wikimedia.org)); packaged as [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (CC BY-SA + GFDL; see [dumps legal](https://dumps.wikimedia.org/legal.html))
5. [karpathy/nanochat](https://github.com/karpathy/nanochat)
6. Vaswani et al., *Attention Is All You Need*
7. Gunasekar et al., *Textbooks Are All You Need*
8. Continued pretraining, instruction tuning, and factuality evaluation literature (expand in paper Related Work)

