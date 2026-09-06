# 12. Text Summarization

## Learning Objectives

- Distinguish extractive from abstractive summarization and know when each is appropriate
- Implement a graph-based extractive summarizer (TextRank) from scratch
- Understand ROUGE score and how it differs from BLEU (Lesson 11)

## The Problem

Long documents — articles, reports, meeting transcripts — often need to be condensed into a shorter form that preserves the essential information. Doing this automatically, well, requires solving a genuinely hard problem: identifying what's actually important (not just frequent), and either selecting the best existing sentences or generating genuinely new, coherent sentences that compress the source faithfully.

## The Concept

### Two fundamentally different strategies

**Extractive summarization** selects existing sentences (or spans) directly from the source document and concatenates them, unmodified, to form the summary. It never generates new text — every word in the output appeared in the input.

**Abstractive summarization** generates new sentences that may paraphrase, combine, or compress information from the source in ways no single source sentence expresses — closer to how a human would actually summarize, and structurally the sequence-to-sequence generation task from Lessons 09-11, with the source document as input and a shorter target summary as output.

```
Source: "The company reported strong Q3 earnings, beating analyst expectations
         by 15%. Revenue grew 20% year over year, driven primarily by cloud
         services. The CEO cited strong enterprise demand as the key driver."

Extractive summary (selects existing sentences):
  "The company reported strong Q3 earnings, beating analyst expectations by 15%."

Abstractive summary (generates new text):
  "The company posted better-than-expected Q3 results, with cloud services
   driving 20% revenue growth."
```

Extractive summaries are always grammatically valid (they're made of real sentences) and can never "hallucinate" facts not in the source, since every word is copied directly — but they can feel disjointed (concatenated sentences from different parts of a document don't always flow naturally together) and can't compress information the way a human paraphrase can. Abstractive summaries read more naturally and can compress more aggressively, but require a trained generation model (Lessons 09-11's machinery) and carry a real risk of generating fluent-sounding but factually incorrect content not actually supported by the source — a specific, well-documented failure mode of abstractive systems worth taking seriously in any real deployment.

### TextRank: graph-based extractive summarization

TextRank (Mihalcea & Tarau, 2004) is a classical, unsupervised extractive method requiring no training data at all — it treats sentence selection as a graph ranking problem, adapting the same underlying idea as Google's original PageRank algorithm (ranking web pages by how many other important pages link to them) to sentences instead of web pages.

```
1. Build a graph where each SENTENCE is a node.
2. Connect every pair of sentences with an edge weighted by their similarity
   (e.g. cosine similarity of their TF-IDF vectors, Lesson 02).
3. Run a PageRank-style algorithm: a sentence's "importance" score depends on
   how similar it is to OTHER important sentences, iteratively refined --
   exactly the same recursive "importance depends on other important things
   linking to you" logic as PageRank, just with sentence similarity replacing
   hyperlinks.
4. Select the top-N highest-scoring sentences as the summary.
```

```python
import numpy as np

def textrank_scores(similarity_matrix, damping=0.85, n_iter=50):
    n = similarity_matrix.shape[0]
    # normalize rows so each sentence's outgoing "votes" sum to 1
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = similarity_matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(n_iter):
        scores = (1 - damping) / n + damping * (transition.T @ scores)
    return scores
```

The intuition: a sentence that's highly similar to many *other* important sentences in the document is likely central to the document's main topic — repeated, related content across many sentences is exactly the signal that indicates a core idea, while a sentence unrelated to everything else (a tangent, or a minor detail) scores low regardless of its position in the document.

### Evaluating summaries: ROUGE

Where BLEU (Lesson 11) measures how much of the *candidate's* n-grams appear in the reference (a precision-oriented metric, penalizing a candidate for content not in the reference), **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) is built the other way around: it measures how much of the *reference summary's* content is captured by the candidate — recall-oriented, since a good summary should recall the reference's key points, and it's more acceptable for a summary to include a little extra context than to miss essential content:

```
ROUGE-N recall = (matching n-grams between candidate and reference) / (total n-grams in REFERENCE)
                                                                          ^^^^^^^^^
                                                                    (denominator is the reference,
                                                                     not the candidate, unlike BLEU)
```

ROUGE-L, a common variant, instead measures the **longest common subsequence** between candidate and reference (not necessarily contiguous, unlike n-grams), rewarding candidates that preserve the reference's overall content and ordering even if exact phrasing differs. Like BLEU, ROUGE is a surface-overlap metric with the same fundamental limitation — it doesn't directly measure factual correctness or fluency, only overlap with a reference — but it remains the standard automatic metric for summarization specifically because "did the summary capture the reference's key content" is closer to what recall-oriented overlap actually measures than BLEU's precision-oriented framing.

See `code/summarization_demo.py` for a complete from-scratch TextRank extractive summarizer using TF-IDF sentence similarity, and a from-scratch ROUGE-N/ROUGE-L implementation for evaluation.

## Exercises

1. Implement `textrank_scores` and run it on a multi-paragraph synthetic document, selecting the top 2-3 sentences as a summary. Confirm the selected sentences are thematically central rather than tangential details.
2. Implement ROUGE-1 recall and ROUGE-L, and compute both for a generated summary against a reference summary. Explain a case where the two metrics would disagree (one high, one low).
3. Compare BLEU and ROUGE-1 computed on the *same* candidate/reference pair, and explain in your own words why one is more appropriate for summarization than the other.
4. Construct an example where an abstractive-style paraphrase (not literally present in the source) would be a genuinely better summary than any extractive selection of source sentences, and explain what extractive summarization is structurally unable to do here.

## Key Terms

| Term | What it actually means |
|---|---|
| Extractive summarization | Producing a summary by selecting and concatenating existing sentences directly from the source document |
| Abstractive summarization | Producing a summary by generating new text that may paraphrase or compress the source, using sequence-to-sequence generation |
| TextRank | A graph-based, unsupervised extractive summarization method that scores sentence importance via a PageRank-style algorithm over sentence similarity |
| ROUGE | A recall-oriented automatic summarization evaluation metric measuring how much of a reference summary's content is captured by a candidate |
| Hallucination | A generation model producing fluent but factually unsupported content, a known risk specific to abstractive methods |
