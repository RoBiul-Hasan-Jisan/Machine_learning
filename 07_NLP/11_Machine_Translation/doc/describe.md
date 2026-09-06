# 11. Machine Translation

## Learning Objectives

- Trace the evolution from rule-based to statistical to neural machine translation
- Explain BLEU score and its known limitations as a translation quality metric
- Understand beam search decoding and why it improves on greedy decoding

## The Problem

Machine translation — automatically converting text from one language to another — is one of NLP's oldest and most demanding goals: it requires everything covered so far (tokenization across languages, syntax, semantics, sequence-to-sequence generation) plus a correct mapping between two entirely different vocabularies and grammatical systems. It's also the task attention (Lesson 10) was originally built for, making it a natural place to consolidate the seq2seq + attention material into a complete application.

## The Concept

### Three eras of machine translation

**Rule-based MT** (1950s-1980s) used hand-written bilingual dictionaries and grammar transformation rules: parse the source sentence's structure, apply linguistic rules to reorder it into the target language's grammar, and substitute words via a dictionary. This required immense manual linguistic effort per language pair and struggled with the sheer number of exceptions and ambiguities in natural language, though it remains interpretable and controllable in a way later approaches aren't.

**Statistical MT (SMT)** (1990s-2010s) replaced hand-written rules with probabilities learned from large parallel corpora (paired source/target sentences), most famously via phrase-based models that learned which source phrases tend to align with which target phrases and how to reorder them, using statistics rather than linguistic rules. This scaled better than rule-based systems and dominated commercial translation (including early Google Translate) for over a decade, but still relied on separately-trained components (a translation model, a language model, a reordering model) stitched together rather than a single, jointly-optimized system.

**Neural MT (NMT)** (2014-present) replaced the whole SMT pipeline with a single, end-to-end trained network: precisely the encoder-decoder-with-attention architecture from Lessons 09-10, trained directly to maximize translation quality on parallel corpora, with every component (encoding, alignment via attention, generation) learned jointly rather than as separate hand-assembled pieces. NMT rapidly surpassed SMT in quality once sufficient training data and compute were available, and remains the dominant paradigm — modern production translation systems are built on Transformer-based NMT (RNN module Lesson 18's architecture), a direct evolution of the RNN-plus-attention systems from Lessons 09-10.

```
Rule-based:    hand-written grammar rules + bilingual dictionary
Statistical:   learned phrase-alignment probabilities from parallel corpora, multiple separate models
Neural:        one end-to-end trained encoder-decoder-with-attention (or Transformer) network
```

### Evaluating translation quality: BLEU score

Human evaluation of translation quality is the gold standard but slow and expensive. **BLEU** (Bilingual Evaluation Understudy) is the standard automatic metric: it measures n-gram overlap (Lesson 02's n-grams) between a candidate translation and one or more human reference translations.

```
BLEU rewards a candidate translation for containing n-grams (typically 1 through 4)
that also appear in the reference translation(s), with a penalty for candidates
that are much shorter than the reference (to prevent gaming the score by
generating a short, safe, low-risk translation).

precision_n = (matching n-grams between candidate and reference) / (total n-grams in candidate)

BLEU = brevity_penalty * exp(average of log(precision_n) for n=1..4)
```

BLEU is fast, automatic, and reproducible, which is why it remains widely reported — but it has real, well-known limitations: it only rewards *surface-level* n-gram overlap with the specific reference translation(s) provided, penalizing a perfectly valid alternative phrasing that happens to use different words than the reference ("The cat sat on the mat" vs "The cat was sitting on the mat" — both fine translations, but the second scores worse against a reference matching the first's exact wording). It also doesn't directly measure fluency or whether the translation is actually understandable, only n-gram overlap. Newer metrics (like COMET or BERTScore, which use learned embeddings — Lesson 22 — to measure semantic similarity rather than exact n-gram overlap) address some of these gaps, but BLEU remains a standard, if imperfect, quick benchmark.

### Beam search: better decoding than greedy

Lesson 09's `greedy_decode` picks the single highest-probability token at every step and never reconsiders that choice — but the locally best token at step 3 might lead to a much worse overall sentence than a slightly-less-likely token at step 3 would have, a problem greedy decoding has no way to recover from once a choice is made.

**Beam search** keeps track of the `k` most probable *partial* sequences (the "beam width") at every step, rather than committing to just one:

```
Beam width k=3, generating token by token:

Step 1: keep top 3 partial sequences by probability:
  ["Le"] (p=0.4), ["La"] (p=0.3), ["Il"] (p=0.2)

Step 2: expand EACH of the 3 sequences with every possible next token,
        then keep only the overall top 3 (sequence, probability) pairs
        across ALL expansions -- not top 3 per branch, top 3 overall:
  ["Le", "chat"] (p=0.4*0.5=0.20), ["La", "voiture"] (p=0.3*0.4=0.12),
  ["Le", "chien"] (p=0.4*0.3=0.12)

... continue until each beam reaches an END token or a maximum length
```

Beam search with `k=1` is exactly greedy decoding (only ever tracking the single best sequence). Larger `k` explores more of the search space and typically finds higher-probability complete sequences, at a proportionally higher computational cost (`k` times the work at every step) — a direct, tunable tradeoff between decoding quality and speed. Beam search remains standard for translation (where getting the single best, most fluent, and most accurate output matters more than raw generation speed), though it can occasionally produce overly generic ("safe") outputs compared to sampling-based decoding strategies (Lesson 16 covers this tradeoff for text generation more broadly).

See `code/translation_demo.py` for a from-scratch BLEU score implementation, a beam search decoder built on top of Lesson 10's attention-based seq2seq model, and a direct comparison of greedy vs beam search decoding quality on the reversal task.

## Exercises

1. Implement BLEU score (at least up to bigrams) from scratch and compute it for a few candidate/reference translation pairs, including one with a valid paraphrase to see the score penalize it unfairly.
2. Implement beam search with `k=1, 3, 5` on top of Lesson 10's trained attention model, and compare the exact-match accuracy on the reversal task for each beam width.
3. Construct a case where greedy decoding produces a worse overall sequence than beam search would, by examining a model's per-step probability distributions and identifying a locally-appealing-but-globally-poor first choice.
4. Research (via web search, since this module can't cover it in full) what BERTScore or COMET measure differently from BLEU, and summarize in a paragraph why they were developed.

## Key Terms

| Term | What it actually means |
|---|---|
| Rule-based MT | Machine translation using hand-written grammatical transformation rules and bilingual dictionaries |
| Statistical MT (SMT) | Machine translation using probabilities learned from parallel corpora, via separately trained components |
| Neural MT (NMT) | Machine translation using a single, end-to-end trained neural network, typically an encoder-decoder with attention or a Transformer |
| BLEU score | An automatic translation quality metric based on n-gram overlap between a candidate translation and reference translation(s) |
| Beam search | A decoding strategy that tracks the k most probable partial sequences at each step, rather than committing greedily to a single choice |
