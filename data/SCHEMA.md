# Fact-card schema (Lane A contract)

**Status:** **frozen** for reported runs (change only with team OK).  
**Goals:** fact-constrained **short stories** (not a kids-product framing), readable by a **small model pretrained on TinyStories**, aligned with **Simple English Wikipedia** sourcing, easy for **B** (loaders/SFT) and **C** (rubric), and **programmatically extractable** for human review.

Related: [`LICENSES.md`](./LICENSES.md) · raw Wiki: `data/raw/wikipedia/` · extract: `python scripts/extract_fact_card_drafts.py` · promote: `python scripts/promote_fact_cards.py`

---

## Design principles

1. **Small-model native.** Facts are short, concrete sentences a small LM can use in a story. Avoid dense academic prose. Stage 1 is TinyStories pretraining (narrative prior); cards should still be teachable in a short story without dumping an encyclopedia.
2. **Stories, not a product demographic.** Outputs are short **stories** with plot/characters; the task is faithfulness + narrative quality — not “children’s content” as a claim.
3. **Closed-world scoreable.** Every string in `facts` is a checklist item C can mark covered / omitted / contradicted. Prefer 4–7 bullets.
4. **Drafts as JSON, approved as JSONL.** The review queue is a pretty-printed **JSON array** (easy to edit). Train/eval that B/C load are **JSONL** (one card per line, streamable). Promote with `scripts/promote_fact_cards.py`.
5. **Templates over bespoke text.** A global prompt template + a fixed “model input” render function. Rows store data; code formats strings for training/eval.
6. **Wiki-aware provenance.** When a card is mined from Simple English Wikipedia, keep enough source metadata to re-check the article and satisfy attribution when we quote.
7. **Human-in-the-loop.** The extractor produces **drafts** (`review_status: draft`). Only `approved` cards enter train/eval JSONL that B/C consume.
8. **Eval isolation.** `split: eval` IDs are frozen for scoring only — never used as SFT targets or prompt-tuning examples.

---

## Files

```text
data/fact_cards/
  train.jsonl              # approved train cards (JSONL — B/C load this)
  eval.jsonl               # approved held-out cards (JSONL — frozen IDs)
  drafts/
    wiki_candidates.json   # extractor output: JSON *array* for human review
data/sft_pairs/
  train.jsonl              # approved (card_id → story) for M2
data/prompts/
  templates.json           # frozen prompt + render recipes
```

| Stage | Format | Why |
| --- | --- | --- |
| Drafts (`drafts/*.json`) | JSON array, indented | Easy to open/edit in an IDE while reviewing |
| Approved (`train.jsonl`, `eval.jsonl`) | JSONL | Runtime dataset format for loaders; one record per line |

---

## Fact-card object

Same object shape in both formats. Drafts file = `[ {...}, {...} ]`.  
Each line of approved `fact_cards/train.jsonl` or `eval.jsonl` is one object.

### Required fields

| Field | Type | Rules |
| --- | --- | --- |
| `id` | string | Stable, unique across train+eval. Format: `fc_<slug>_<nn>` (e.g. `fc_water_cycle_01`). **Never reuse** an id for a different topic; bump suffix on rewrite if semantics change. |
| `split` | `"train"` \| `"eval"` | Eval ids must not appear in SFT pairs. |
| `topic` | string | Short label for humans + optional title in the model render (2–6 words, simple English). |
| `facts` | string[] | **4–7** gold teaching bullets. Each bullet: one checkable claim, ideally ≤20 words, simple vocabulary. |
| `review_status` | `"draft"` \| `"approved"` \| `"rejected"` | Loaders for **reported** runs use **`approved` only**. |

### Optional fields

| Field | Type | Who uses it | Rules |
| --- | --- | --- | --- |
| `false_claims` | string[] | **C (raters)** by default | Common wrong ideas. **Not** shown to the model unless an ablation explicitly adds them. Helps score “unconstrained invention / contradiction.” |
| `domain` | string | A filters, paper tables | Controlled set: `nature`, `science`, `body`, `history`, `geography`, `everyday`, `other`. |
| `source` | object | A provenance, licenses, re-extract | See below. `null` / omit if hand-written with no Wiki page. |
| `notes` | string | A/C only | Author/reviewer notes. **Never** model context. |
| `word_target` | [int, int] | B/C length checks | Default `[120, 180]` story words. |

### `source` object (Wiki-backed cards)

```json
{
  "type": "wikipedia_simple",
  "hf_dataset": "wikimedia/wikipedia",
  "hf_config": "20231101.simple",
  "wiki_id": "1",
  "title": "April",
  "url": "https://simple.wikipedia.org/wiki/April",
  "extract_method": "lead_sentences_v1",
  "extracted_at": "2026-07-21T20:00:00Z"
}
```

- `type`: currently `wikipedia_simple` or `handwritten`.
- Keep `url` / `title` / `wiki_id` when present so reviewers can open the page in one click.
- `extract_method` names the code path that proposed the bullets (reproducible review).

### Example (approved train card)

```json
{
  "id": "fc_water_cycle_01",
  "split": "train",
  "topic": "water cycle",
  "domain": "nature",
  "facts": [
    "Water evaporates from oceans, lakes, and rivers.",
    "Water vapor cools and forms clouds.",
    "Water falls from clouds as rain or snow.",
    "Water can flow back into rivers and oceans."
  ],
  "false_claims": [
    "Clouds are made of cotton.",
    "Rain comes from holes in the sky."
  ],
  "source": {
    "type": "handwritten",
    "hf_dataset": null,
    "hf_config": null,
    "wiki_id": null,
    "title": null,
    "url": null,
    "extract_method": null,
    "extracted_at": null
  },
  "review_status": "approved",
  "notes": "Seed card; short concrete facts for story-form teaching.",
  "word_target": [120, 180]
}
```

---

## Prompt + model I/O (Lane B contract)

**Do not** store a slightly different prompt prose on every row. Store data; render with templates in [`prompts/templates.json`](./prompts/templates.json).

### Default user-facing instruction (frozen string)

> Write a short story (about 120-180 words) that teaches the topic using only the facts below. Invent characters and plot freely. Do not invent facts that are not in the list. Keep the language clear and concrete.

### Default model input render (SFT + eval generation)

```text
Topic: {topic}

Facts:
1. {facts[0]}
2. {facts[1]}
...

Instruction: {instruction}
```

### Default SFT target

The story only (raw text). No JSON wrapper.

### What is **not** in model context (default)

- `false_claims`
- `notes`
- `source.*`
- `review_status`, `split`, `domain` (unless an ablation says otherwise)

B may implement:

```python
def render_model_input(card: dict, templates: dict) -> str: ...
```

C’s eval harness should call the **same** render function (or copy the same template file) so systems stay matched.

---

## SFT pair object (Lane B / M2)

Each line of `sft_pairs/train.jsonl`:

| Field | Type | Rules |
| --- | --- | --- |
| `id` | string | Pair id, e.g. `sft_water_cycle_01_a` |
| `card_id` | string | Must reference an **approved** `fact_cards` id with `split: train` |
| `story` | string | Gold story; should cover the card’s facts without contradicting them |
| `review_status` | string | `approved` for training runs |
| `notes` | string | optional |

Example:

```json
{
  "id": "sft_water_cycle_01_a",
  "card_id": "fc_water_cycle_01",
  "story": "Once upon a time, a small duck named Pip watched the lake sparkle...",
  "review_status": "approved",
  "notes": "Seed pair; ~150 words; hits all four facts."
}
```

**Leakage rule:** no `card_id` whose card has `split: eval`.

---

## Train / eval split policy

| Rule | Detail |
| --- | --- |
| Scale (shipped) | **866** train cards + **866** SFT stories; **235** held-out eval cards. Early plan was ~80–200 / 40–60; we oversized the pool for the ~78M Stage-1 model. |
| Scored eval subset | Lane C freezes **100** prompt ids in `eval/prompts/frozen_eval_ids.txt` (a subset of `eval.jsonl`). Do not reorder once scoring starts. |
| When to freeze eval ids | Before SFT writing or prompt tuning on those topics (already done) |
| Topic leakage | Do not put near-duplicate topics on both sides (e.g. “rain” train vs “rainfall” eval) without an explicit decision |
| Source leakage | Prefer that eval Wiki titles are not reused as train sources |
| Edits | Fixing a typo on an eval card keeps the same `id`. Changing the meaning → new id and treat as data change (note in notes) |

---

## Lane C (eval) mapping

Human scores live in [`eval/rubric.md`](../eval/rubric.md). Paper “faithfulness” ≈ rubric **factual correctness**; paper “story quality” ≈ **grammar + storytelling creativity + coherence**.

| Rubric need | Schema field |
| --- | --- |
| Required-fact coverage / omission | each string in `facts` (shown to raters in the scoring UI) |
| Contradiction | story vs `facts` (and optionally `false_claims` as known traps) |
| Invention | claim in story neither in `facts` nor harmless fiction (names, dialogue) |
| Story quality | independent of facts |
| Stable item id in sheets | card `id` (also `prompt_id` / `card_id` in eval CSVs) |

Error taxonomy (paper): omission, contradiction, unconstrained invention, encyclopedia dump, story domination.

---

## Programmatic Wiki → draft pipeline

### Why

Writing 100+ cards by hand is slow. Simple English Wikipedia is already local, CC BY-SA/GFDL-attributed, and close to the reading level we want. Automate **candidates**; humans **approve**.

### Pipeline stages

```text
data/raw/wikipedia/20231101_simple.jsonl
        │
        ▼
  filter articles (length, title denylist, stub/lead quality)
        │
        ▼
  split lead into simple sentences
        │
        ▼
  pick 4–7 fact-like sentences (heuristics)
        │
        ▼
data/fact_cards/drafts/wiki_candidates.json    JSON array, review_status=draft
        │
        ▼
  human edit in IDE (facts, split, review_status)
        │
        ▼
  python scripts/promote_fact_cards.py
        │
        ├─ approved + split=train → data/fact_cards/train.jsonl
        ├─ approved + split=eval  → data/fact_cards/eval.jsonl
        └─ draft / rejected       → stay in drafts JSON only
```

### Filter heuristics (v1)

Keep an article if roughly:

- Title not in denylist prefixes: `List of`, `Timeline of`, `Index of`, `Category:`, `Wikipedia:`, `Template:`, `File:`
- Title not mostly digits (years-only pages often weak for stories)
- Text length in a mid band (default **500–2500** chars) — enough for facts, not a wall of text
- Lead yields ≥4 short sentences after cleaning

### Sentence → fact heuristics (v1)

- Use the **lead** (text before the first blank line, else first ~1200 chars)
- Split on `.` / `?` / `!` with light cleanup
- Drop sentences that are too short (&lt;20 chars) or too long (&gt;160 chars)
- Drop “This page…”, “References”, “see also”, pure definitions that are only “X is a word…”
- Prefer sentences with a clear subject and a concrete verb
- Cap at 7; require at least 4 or skip the article

### Review speedups (intentional)

Each draft includes:

- `id`, `topic` (from title), `facts`, `source.url` (one-click check)
- `extract_method: lead_sentences_v1`
- `review_status: draft`

Reviewer actions are deliberately few: **edit facts → set `split` → set `review_status` to `approved` or `rejected`**.  
Then run:

```bash
python scripts/promote_fact_cards.py           # write JSONL
python scripts/promote_fact_cards.py --dry-run # preview only
```

### License reminder

Drafts mined from Wikipedia inherit **attribution obligations** for redistributed text. Prefer light paraphrase at approval time when bullets are near-verbatim; keep `source` either way. See [`LICENSES.md`](./LICENSES.md).

---

## Style guide (for writers + reviewers)

When editing `facts` or writing SFT stories:

| Do | Don’t |
| --- | --- |
| Short, clear, concrete sentences | Dense jargon, equations, multi-clause academic prose |
| Claims a short story can actually surface | Abstract theory with no narrative hook |
| 4–7 facts coverable in ~150 words | 15 micro-facts nobody can cover |
| Invent characters and plot freely | Invent **world facts** beyond the card |
| Story shape: beginning → event/turn → end | Bullet list or textbook dump as the “story” |

**Note:** Stage 1 uses TinyStories for the narrative prior. That is a **training corpus choice**, not a requirement that task outputs be “kids’ stories.”

---

## Validation rules (for code)

A card is **train-run valid** if:

- all required fields present  
- `review_status == "approved"`  
- `4 <= len(facts) <= 7`  
- each fact is a non-empty string  
- `id` unique  
- `split in {"train","eval"}`  
- if `source.type == "wikipedia_simple"`, `url` or `wiki_id` should be set  

An SFT pair is valid if:

- `review_status == "approved"`  
- `card_id` exists, is approved, and `split == "train"`  
- `story` non-empty  

---

## Versioning

| Item | Value |
| --- | --- |
| Schema id | `factcard_v1` |
| Extractor | `lead_sentences_v1` (see `scripts/extract_fact_card_drafts.py`) |
| Promote | `scripts/promote_fact_cards.py` (JSON drafts → train/eval JSONL) |
| Prompt pack | `prompts/templates.json` → `version: 1` |

Bump `factcard_v1` only on breaking field renames.

---

## Checklist (frozen with B/C)

- [x] Field list documented  
- [x] Model render template documented (`data/prompts/templates.json` + `scripts/lab_gpt/prompts.py`)  
- [x] SFT pair shape documented  
- [x] Split / leakage rules documented  
- [x] Wiki extract → draft path documented + scripted  
- [x] B loader consumes `approved` cards (`scripts/train_stage2.py` / `lab_gpt/data.py`)  
- [x] C scoring UI shows `facts` for faithfulness / omission ([`eval/app.py`](../eval/app.py))  
- [x] Shipped scale: 866 train / 235 eval / 866 SFT  
