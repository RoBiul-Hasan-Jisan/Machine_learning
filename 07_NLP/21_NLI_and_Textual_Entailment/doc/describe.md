# 21. NLI & Textual Entailment

## Learning Objectives

- Define the three-way NLI classification task: entailment, contradiction, neutral
- Build an NLI classifier using sentence-pair features and understand why this task is harder than single-sentence classification
- Explain NLI's role as a building block for other tasks: fact-checking, RAG evaluation (Lesson 27), and model evaluation more broadly

## The Problem

Many practical reasoning tasks reduce to a single underlying question: given one piece of text (a "premise"), does another piece of text (a "hypothesis") logically follow from it, contradict it, or say something unrelated? This is Natural Language Inference (NLI), and it's a deceptively hard task — it requires genuine reasoning about meaning and logical relationships between two pieces of text, not just classifying one piece of text in isolation (Lesson 05's sentiment analysis) or extracting a fact from one passage (Lesson 06's NER, Lesson 13's QA).

## The Concept

### The three-way classification task

Given a premise and a hypothesis, NLI classifies their relationship into exactly one of three categories:

```
Premise: "A man is playing guitar on stage."

Hypothesis: "A person is performing music."          -> ENTAILMENT
  (if the premise is true, the hypothesis must also be true)

Hypothesis: "The stage is empty."                     -> CONTRADICTION
  (if the premise is true, the hypothesis must be false)

Hypothesis: "The man is wearing a red shirt."          -> NEUTRAL
  (the premise doesn't confirm or rule out the hypothesis -- could go either way)
```

The key distinction from sentiment analysis or single-sentence classification: NLI is inherently a **sentence-pair** task — the label depends entirely on the *relationship* between two pieces of text, not any property of either sentence read in isolation. "A person is performing music" isn't inherently entailment, contradiction, or neutral; it's only classifiable relative to a specific premise, which is exactly what makes NLI structurally different from every single-text classification task covered so far in this module.

### Why this is a genuinely hard task

Correctly judging entailment/contradiction/neutral requires several distinct reasoning capabilities working together:

- **Lexical relationships**: knowing "guitar" implies "instrument," "performing" relates to "playing" — this is exactly the semantic knowledge embeddings (Lessons 03-04, 22) are meant to capture, and NLI performance depends heavily on having good representations of word/phrase meaning.
- **Logical structure**: negation ("not," "no," "never" — Lesson 05's negation-handling problem shows up again here, in a harder form) can flip entailment into contradiction or vice versa; quantifiers ("all," "some," "no") change what's actually entailed in subtle, precise ways.
- **World knowledge**: "A man is playing guitar on stage" entailing "a person is performing music" requires knowing that playing an instrument on a stage generally counts as "performing" — a piece of everyday world knowledge, not something derivable from the sentences' words alone.
- **Compositional understanding**: the relationship depends on how the *whole* premise and hypothesis combine, not just whether individual words overlap — high word overlap between premise and hypothesis doesn't guarantee entailment (a hypothesis could reuse many of the premise's words while still contradicting or being neutral to it), and this is precisely why NLI is a much harder, more genuine test of language understanding than surface similarity metrics like the n-gram overlap BLEU/ROUGE (Lessons 11-12) measure.

### A basic NLI classifier

A classical approach encodes the premise and hypothesis separately (using any of this module's sentence representation techniques — TF-IDF, Lesson 02; averaged embeddings, Lessons 03-04; or an RNN encoder, RNN module Lesson 04), then combines the two representations with explicit interaction features before classifying:

```python
import torch
import torch.nn as nn

class NLIClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, num_classes=3):
        super().__init__()
        self.encoder = encoder    # any sentence encoder, e.g. an RNN (RNN module Lesson 04)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, premise, hypothesis):
        p_vec = self.encoder(premise)         # (batch, hidden_size)
        h_vec = self.encoder(hypothesis)       # (batch, hidden_size)

        # explicit interaction features, a common and effective classical trick:
        combined = torch.cat([
            p_vec, h_vec,
            torch.abs(p_vec - h_vec),        # captures DIFFERENCE between the two
            p_vec * h_vec,                    # captures elementwise AGREEMENT/overlap
        ], dim=1)
        return self.classifier(combined)
```

The `abs(p_vec - h_vec)` and `p_vec * h_vec` terms are a standard, effective trick: rather than only concatenating the two raw sentence vectors and hoping the classifier learns to compare them implicitly, explicitly computing a difference and an elementwise product gives the classifier direct, pre-computed signals for "how different are these two sentences" and "where do they agree," which tend to correlate strongly with entailment/contradiction/neutral in practice. Modern systems typically use Transformer-based encoders (Lesson 22) that process the premise and hypothesis *jointly* (via cross-attention) rather than encoding them fully separately before combining — a more powerful but more computationally expensive approach, since it processes every premise-hypothesis pair together rather than encoding sentences once and reusing the encoding across many comparisons.

### NLI as a building block for other tasks

NLI shows up as an internal component in several other places this module touches on, precisely because "does statement B follow from statement A" is such a fundamental building block of automated reasoning about text:

- **Fact-checking**: does a claim follow from (is it entailed by) a trusted source document? This is directly an NLI classification, with the claim as hypothesis and the source as premise.
- **RAG faithfulness evaluation** (Lesson 27): does a generated answer's content actually follow from the retrieved context it was supposedly grounded in, or does it contradict/go beyond what the context supports? This is precisely how several modern LLM evaluation frameworks (e.g. RAGAS's faithfulness metric) detect hallucination — by running an NLI-style entailment check between the generated output and the retrieved source material.
- **Summarization evaluation**: does a summary contradict anything stated in the source document (a factual consistency check), extending beyond what ROUGE's surface n-gram overlap (Lesson 12) can measure.

See `code/nli_demo.py` for a complete NLI classifier using averaged word embeddings plus the interaction-feature trick above, trained on a small synthetic entailment/contradiction/neutral dataset, with example predictions and a discussion of which examples the model gets wrong and why.

## Exercises

1. Construct 5 premise/hypothesis pairs by hand for each of the three labels (15 total), and confirm your own judgments align with the formal definitions of entailment, contradiction, and neutral given above.
2. Implement the `NLIClassifier` with the interaction features and train it on a small synthetic dataset. Compare its accuracy against a version that only concatenates `p_vec` and `h_vec` without the difference/product terms.
3. Construct a specific case where high word overlap between premise and hypothesis does NOT imply entailment (e.g. via negation), and verify whether your trained classifier handles it correctly.
4. Explain, in your own words, how an NLI-style entailment check could be used to automatically flag a generated RAG answer that "hallucinates" a fact not actually supported by its retrieved context.

## Key Terms

| Term | What it actually means |
|---|---|
| Natural Language Inference (NLI) | The task of classifying the logical relationship between a premise and a hypothesis as entailment, contradiction, or neutral |
| Entailment | The hypothesis must be true if the premise is true |
| Contradiction | The hypothesis must be false if the premise is true |
| Neutral | The premise neither confirms nor rules out the hypothesis |
| Sentence-pair task | A task whose label depends on the relationship between two pieces of text, not either text read in isolation |
