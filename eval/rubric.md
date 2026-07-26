# Human evaluation rubric (Lane C)

Used by [`app.py`](./app.py) to score generations produced by
[`generate_samples.py`](./generate_samples.py). Scoring is **blind**: the
rater sees stories under randomized labels ("Model A", "Model B", ...), not
real system ids, and automated perplexity is hidden until after a prompt's
scores are saved.

## Likert axes (1-5, score each system independently)

### Grammar
| Score | Meaning |
| --- | --- |
| 1 | Frequent errors that obscure meaning (broken syntax, wrong words) |
| 2 | Noticeably ungrammatical in several places |
| 3 | A few minor errors, meaning always clear |
| 4 | Clean, at most one small slip |
| 5 | Fully grammatical, natural phrasing throughout |

### Factual correctness
| Score | Meaning |
| --- | --- |
| 1 | Story asserts claims that are clearly false or nonsensical |
| 2 | Multiple dubious or unsupported factual claims |
| 3 | Mostly plausible, one or two questionable claims |
| 4 | Claims are accurate or safely within harmless fiction (names, dialogue) |
| 5 | Everything asserted as fact is correct or trivially fictional |

### Storytelling creativity
| Score | Meaning |
| --- | --- |
| 1 | No real narrative — a list of statements or a definition |
| 2 | Minimal plot, generic characters, no hook |
| 3 | Recognizable story with a plain but workable plot |
| 4 | Engaging characters/plot with some original detail |
| 5 | Vivid, original, memorable narrative choices |

### Coherence
| Score | Meaning |
| --- | --- |
| 1 | Contradicts itself or makes no sense as a sequence of events |
| 2 | Hard to follow; events don't connect logically |
| 3 | Mostly follows, a few unclear transitions |
| 4 | Clear beginning/middle/end, logical flow |
| 5 | Fully coherent, well-structured throughout |

## Optional error tags

Apply any that fit (multi-select, not mutually exclusive) — these feed the
paper's required error-analysis section:

- **Omission** — skips content a reader would expect the prompt to cover
- **Contradiction** — asserts something, then contradicts it later in the same story
- **Unconstrained invention** — invents specific facts/claims presented as true, beyond harmless fiction (character names, incidental dialogue)
- **Encyclopedia dump** — reads like a list of facts/definitions, not a story
- **Story domination** — plot completely overwhelms any informative content, teaches nothing

## Automated metric

**Perplexity** — computed by [`generate_samples.py`](./generate_samples.py) as
the story continuation's perplexity under its own generating model (teacher-forced,
prompt tokens masked out of the loss). This is a fluency/confidence proxy only,
not a correctness or quality measure — per the project README, "not vibes and
not perplexity alone." It is hidden during scoring and only revealed after a
prompt's human scores are saved, so it cannot anchor the rating.

**Known limitation:** self-perplexity rewards low-entropy degeneration. A
system that loops "They are happy. They are happy." scores *well*. In the
pre-fix ablation ratings, B1 had the lowest mean perplexity (17.1) and the
worst human score on all four axes. Do not present it as a quality metric;
a held-out cross-entropy on a common reference set would be the defensible
version of this column.

## Blind-scoring protocol

- For each prompt, system order is shuffled deterministically (seeded by
  prompt id) and displayed as "Model A / Model B / ...".
- The real `system_id` (B0/B1/M2/...) is written to `scores.csv` but never
  shown in the scoring UI.
- Analyze results by joining `scores.csv` on `system_id`, not on the shown
  label (the shown label differs per prompt).
