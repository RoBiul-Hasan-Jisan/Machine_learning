# 27. LLM Evaluation: RAGAS, DeepEval, G-Eval

## Learning Objectives

- Explain why traditional NLP metrics (BLEU, ROUGE) fall short for evaluating open-ended LLM outputs
- Implement the core RAGAS metrics (faithfulness, answer relevancy, context precision/recall) from scratch
- Understand LLM-as-judge evaluation (G-Eval) and its known biases

## The Problem

BLEU (Lesson 11) and ROUGE (Lesson 12) measure surface n-gram overlap against a fixed reference — a reasonable approximation when there's basically one correct way to phrase a translation or summary, but a poor fit for evaluating modern LLM applications, where a single "correct" reference often doesn't exist (many different phrasings of a helpful answer are equally valid) and what actually matters is harder to capture with word overlap: is the answer factually grounded in its source, does it actually address the question, is the retrieved context even good enough to answer from in the first place. RAG-specific and general-purpose LLM evaluation frameworks were built specifically to measure these harder, more semantically meaningful properties.

## The Concept

### Why RAG systems need their own evaluation metrics

A RAG system (Lessons 13, 17, 23) has multiple components that can each fail independently — the retriever can fetch irrelevant context, or the generator can ignore good context and hallucinate anyway (Lesson 12's hallucination risk), or the generator can produce a technically-grounded answer that doesn't actually address what was asked. A single end-to-end quality score conflates all of these distinct failure modes together, making it hard to diagnose *what specifically* is going wrong. RAGAS (Retrieval Augmented Generation Assessment) is a framework built specifically to separate these concerns into distinct, independently measurable metrics.

### Faithfulness: does the answer actually follow from the context?

This is a direct application of Lesson 21's NLI machinery: break the generated answer into individual claims, and check whether each claim is entailed by the retrieved context — an unfaithful answer contains claims that go beyond, or contradict, what the context actually supports.

```
Retrieved context: "The company reported Q3 revenue of 50 million dollars."
Generated answer:  "The company's Q3 revenue was 50 million dollars, driven by strong overseas sales."

Claim 1: "Q3 revenue was 50 million dollars"        -> ENTAILED by context (faithful)
Claim 2: "driven by strong overseas sales"           -> NOT entailed by context (unsupported claim --
                                                          the context says nothing about WHY revenue
                                                          was what it was, let alone overseas sales)

Faithfulness score = (number of entailed claims) / (total number of claims) = 1/2 = 0.5
```

```python
def faithfulness_score(claims, context, nli_model):
    """nli_model: any entailment classifier, e.g. Lesson 21's NLIClassifier."""
    entailed_count = 0
    for claim in claims:
        prediction = nli_model.predict(premise=context, hypothesis=claim)
        if prediction == "entailment":
            entailed_count += 1
    return entailed_count / len(claims) if claims else 0.0
```

This directly operationalizes hallucination detection: a low faithfulness score flags specifically that the generator is asserting things its retrieved context doesn't actually support, regardless of whether those things happen to be true in the real world — faithfulness measures groundedness in the *given* context, not real-world factual correctness, which is a distinct (and harder to automatically check) property.

### Answer relevancy: does the answer address the question?

A perfectly faithful answer can still be unhelpful if it doesn't actually address what was asked — faithfulness alone doesn't catch a technically-grounded but off-topic or evasive response. Answer relevancy checks this more directly, often by generating plausible questions the answer *would* be a good response to, and comparing those against the original question via embedding similarity (Lesson 22):

```python
def answer_relevancy_score(answer, original_question, embedder):
    # a real implementation typically generates plausible questions FROM the
    # answer and compares those to the original question via embedding
    # similarity (Lesson 22); a simpler direct proxy compares the answer's
    # own content directly against the question's content words
    return content_word_overlap_or_embedding_similarity(answer, original_question)
```

The intuition: if an answer is genuinely relevant to the original question, questions generated *from* that answer should closely resemble the original question; if the answer wandered off-topic, the generated questions will drift away from what was actually asked.

### Context precision and recall: is the retrieval step itself working?

These metrics evaluate the *retriever* specifically (Lesson 14), independent of how well the generator uses whatever it's given:

```
Context precision: of the retrieved context chunks, how many were actually relevant/useful?
                    (high precision = the retriever isn't wasting the generator's attention
                     on irrelevant chunks)

Context recall: of the information actually needed to answer the question, how much of it
                was present SOMEWHERE in the retrieved context?
                (high recall = the retriever isn't missing necessary information entirely)
```

This separation is genuinely useful for debugging: a RAG system with high context recall/precision but low faithfulness has a *generation* problem (good context, but the model isn't using it well); a system with low context recall has a *retrieval* problem (Lesson 14's retriever isn't finding what's needed in the first place) — no amount of generator improvement can fix a case where the necessary information was never retrieved at all.

### G-Eval and LLM-as-judge: using an LLM to grade LLM outputs

Rather than computing a metric from a fixed formula, **LLM-as-judge** evaluation (the approach behind G-Eval and used throughout frameworks like DeepEval) prompts a separate, typically more capable LLM to directly assess a generated output against specific criteria, often on a numeric scale with an explicit rubric:

```
Judge prompt (simplified):
"Rate the following answer's faithfulness to the given context on a scale of 1-5.
 A score of 5 means every claim in the answer is directly supported by the context.
 A score of 1 means the answer contains significant unsupported claims.

 Context: {context}
 Answer: {answer}

 Score (1-5):"
```

This is flexible and can evaluate nuanced, hard-to-formalize qualities (tone, helpfulness, coherence) that a fixed formula like faithfulness's NLI-based score can't easily capture, and it's become a standard evaluation approach for exactly this flexibility. It comes with real, documented biases worth knowing about: LLM judges tend to systematically favor longer responses regardless of actual quality, can be inconsistent across repeated evaluations of the identical output (a lack of determinism that undermines reproducibility), and can be sensitive to superficial factors like formatting or verbosity that have little to do with the property actually being measured. Mitigations include using detailed rubrics (as shown above, rather than a vague "rate this 1-5"), averaging scores across multiple judge calls, and — where feasible — using a different, ideally stronger model as judge than the one being evaluated, to reduce the risk of a model favoring outputs stylistically similar to its own.

### Comparing evaluation approaches

| Approach | Measures | Strength | Weakness |
|---|---|---|---|
| BLEU/ROUGE (Lessons 11-12) | Surface n-gram overlap with a reference | Fast, deterministic, no extra model needed | Poor fit when no single correct phrasing exists |
| NLI-based (faithfulness) | Whether claims are entailed by context | Directly targets hallucination, interpretable | Needs claim decomposition, an NLI model |
| Embedding-based (answer relevancy) | Semantic similarity to expected content | Captures paraphrase-level relevance | Doesn't verify factual correctness |
| LLM-as-judge (G-Eval) | Flexible, rubric-defined criteria | Handles nuanced, hard-to-formalize qualities | Bias, inconsistency, extra cost/latency per evaluation |

See `code/llm_eval_demo.py` for from-scratch implementations of faithfulness scoring (using Lesson 21-style entailment checking), answer relevancy (using embedding similarity), and a simulated LLM-as-judge scoring function, run on a small set of RAG-style example outputs with varying quality.

## Exercises

1. Implement `faithfulness_score` using a simple word-overlap-based entailment proxy (or Lesson 21's trained NLI classifier) and score 5 hand-constructed (context, answer) pairs with varying degrees of hallucination.
2. Implement a simplified answer relevancy check using embedding similarity (Lesson 22) between the original question and the answer itself (a simpler proxy than generating new questions from the answer), and test it on an on-topic vs an off-topic answer to the same question.
3. Construct a case where an answer is highly faithful (every claim is supported by context) but has low relevancy (it doesn't address the actual question asked), and confirm your two metrics correctly diverge on it.
4. Write a detailed grading rubric (in the style of the G-Eval prompt above) for evaluating chatbot responses (Lesson 17) on "helpfulness," being as specific as possible about what a 1 vs a 5 looks like, to minimize the vague-rubric bias risk discussed above.

## Key Terms

| Term | What it actually means |
|---|---|
| RAGAS | A RAG-specific evaluation framework measuring faithfulness, answer relevancy, and context precision/recall as separate, diagnosable metrics |
| Faithfulness | Whether a generated answer's claims are actually entailed by its retrieved context, directly targeting hallucination |
| Answer relevancy | Whether a generated answer actually addresses the original question, distinct from whether it's factually grounded |
| Context precision/recall | Metrics evaluating the retriever specifically: how much retrieved context was relevant, and how much necessary information was retrieved at all |
| LLM-as-judge | Using a separate LLM, prompted with a grading rubric, to evaluate a generated output rather than computing a fixed formula |
| G-Eval | A specific LLM-as-judge evaluation methodology using detailed rubrics and chain-of-thought style scoring |
