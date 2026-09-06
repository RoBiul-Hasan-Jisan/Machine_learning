# 13. Question Answering Systems

## Learning Objectives

- Distinguish extractive, abstractive, and open-domain question answering
- Implement a span-extraction QA approach and understand start/end position prediction
- Explain the retriever-reader architecture that open-domain QA systems use

## The Problem

Question answering (QA) systems take a natural-language question and return a natural-language answer, rather than a ranked list of documents (Lesson 14's information retrieval) or an open-ended response (a general chatbot, Lesson 17). This is a more constrained, more directly evaluable task than open-ended generation, and it builds directly on this module's earlier lessons: NER (Lesson 06) and POS/parsing (Lesson 07) help identify what kind of answer a question expects, and the retrieval techniques in Lesson 14 become essential once the answer might be anywhere in a large document collection rather than a single given passage.

## The Concept

### Three flavors of QA

**Extractive QA** (also called span extraction) answers a question by identifying and returning an exact, contiguous span of text from a given passage — the answer is literally copied from the source, never generated fresh:

```
Passage: "The Eiffel Tower was completed in 1889 and stands 330 meters tall."
Question: "How tall is the Eiffel Tower?"
Answer (extracted span): "330 meters"
```

**Abstractive QA** generates a free-form answer that may not be a literal substring of any source text — closer to how a human actually answers a question, potentially synthesizing information from multiple places or rephrasing for clarity, using the sequence-to-sequence machinery from Lessons 09-11.

**Open-domain QA** removes the assumption that a single relevant passage is given at all — the system must first *find* relevant passages from a large corpus (this is where Lesson 14's information retrieval becomes a required component, not an optional add-on) and then extract or generate an answer from what it finds, combining retrieval and reading into one pipeline.

### Extractive QA as span prediction

Given a passage and a question, extractive QA is typically framed as predicting two positions in the passage: where the answer span **starts** and where it **ends**. A model (traditionally a Transformer-based encoder, though the same idea can work with an RNN encoder) processes the question and passage together, and produces, for every token position in the passage, two scores: "how likely is this the start of the answer" and "how likely is this the end."

```
Passage tokens:  The  Eiffel  Tower  was  completed  in  1889  and  stands  330  meters  tall  .
Start scores:    .01   .02    .01   .01    .01      .01  .01   .01    .02   .85    .03   .01  .00
End scores:      .01   .02    .01   .01    .01      .01  .01   .01    .02   .04    .10   .78  .00

Predicted span: argmax(start) to argmax(end) = "330 meters tall"  (positions 9-11)
```

```python
import torch.nn as nn

class SpanExtractionHead(nn.Module):
    """Sits on top of any encoder producing per-token representations
    (an RNN, or more commonly a Transformer encoder in modern systems)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)

    def forward(self, encoded_tokens):
        # encoded_tokens: (batch, seq_len, hidden_size) -- from any sequence encoder
        start_logits = self.start_classifier(encoded_tokens).squeeze(-1)  # (batch, seq_len)
        end_logits = self.end_classifier(encoded_tokens).squeeze(-1)
        return start_logits, end_logits
```

Training uses cross-entropy loss against the true start and end positions (two separate classification problems, one per position type), and inference picks the start/end position pair with the highest combined score, typically constrained so the end position must come at or after the start position.

### The retriever-reader architecture for open-domain QA

When there's no single given passage — the answer could be anywhere in Wikipedia, or a company's entire document corpus — open-domain QA splits the problem into two distinct stages with different jobs:

```
Question
   |
   v
[Retriever] -- searches a large corpus, returns the top-k most relevant passages
   |             (this is Lesson 14's information retrieval, applied directly)
   v
[Reader] -- runs extractive (or abstractive) QA on EACH retrieved passage,
   |         producing a candidate answer + confidence score per passage
   v
Best answer (highest-confidence candidate across all retrieved passages)
```

The retriever's job is recall-oriented: find passages that *might* contain the answer, from a potentially huge corpus, cheaply enough to run in real time. The reader's job is precision-oriented: given a small number of plausible passages, carefully extract (or generate) the actual answer. This two-stage split is the same "cheap broad filter, then expensive careful stage" pattern that shows up throughout information retrieval and recommendation systems generally, and it's directly the architecture underlying retrieval-augmented generation (RAG) systems, which Lesson 23 (chunking strategies) and the LLM evaluation lessons (27-28) discuss further in the context of modern large language model applications.

### Why extractive QA needs to know when there's no answer

A realistic QA system must also handle questions the given passage simply doesn't answer — a common, important requirement (and a standard evaluation setting, e.g. SQuAD 2.0's "unanswerable" questions) that a naive span-prediction model, which always outputs *some* start/end position, doesn't handle by default. A standard fix: reserve a special "no answer" position (often the passage's very first token, or a dedicated `[CLS]`-style token) as the trained target for unanswerable questions, and compare its score directly against the best real span's score at inference time — if "no answer" scores higher, the system correctly reports that the passage doesn't contain the answer instead of confidently returning something wrong.

See `code/qa_demo.py` for a from-scratch span-extraction QA model (a bidirectional RNN encoder plus a start/end prediction head) trained on a small synthetic passage/question/answer dataset, plus a minimal retriever-reader pipeline combining TF-IDF retrieval (Lesson 02) with the trained extractive reader.

## Exercises

1. Implement the `SpanExtractionHead` on top of a bidirectional RNN encoder and train it on a small synthetic dataset of (passage, question, answer span) triples. Confirm it correctly identifies the answer span for held-out examples.
2. Add a "no answer" case to your training data (questions the passage doesn't answer) and confirm the trained model correctly abstains rather than confidently returning a wrong span.
3. Build a minimal retriever using TF-IDF cosine similarity (Lesson 02) over a small collection of passages, retrieve the top-3 most relevant passages for a test question, and run your extractive reader on each.
4. Compare extractive and abstractive framings for a question whose correct answer requires combining information from two different sentences in the passage. Explain why extractive QA structurally cannot answer this correctly.

## Key Terms

| Term | What it actually means |
|---|---|
| Extractive QA | Answering a question by identifying and returning an exact span of text copied from a given passage |
| Abstractive QA | Answering a question by generating a free-form answer, potentially not present verbatim in any source text |
| Open-domain QA | Question answering without a single given passage, requiring the system to first retrieve relevant passages from a large corpus |
| Retriever-reader architecture | A two-stage open-domain QA pipeline: a retriever finds candidate passages, then a reader extracts or generates an answer from them |
| Span extraction | Framing QA as predicting the start and end token positions of the answer within a passage |
