# 22. Embedding Models Deep Dive

## Learning Objectives

- Explain why sentence embeddings need dedicated training, not just averaged word embeddings
- Implement contrastive learning for sentence embeddings and explain what the objective actually optimizes
- Compare bi-encoder and cross-encoder architectures and know when each is the right choice

## The Problem

Lessons 03-04 covered word embeddings — one vector per word. But many of this module's applications (retrieval, Lesson 14; NLI, Lesson 21; RAG, Lessons 17 and 23) need to compare whole *sentences* or *passages* for semantic similarity, not individual words. The naive approach — average a sentence's word embeddings together — is a reasonable baseline (it's what Lesson 21's demo code did directly), but it has real, well-documented limitations that dedicated sentence embedding models are specifically trained to fix.

## The Concept

### Why averaging word embeddings isn't enough

Averaging word embeddings treats a sentence as an unordered bag of words (Lesson 02's limitation, resurfacing here): "the dog bit the man" and "the man bit the dog" have identical word sets, so their averaged embeddings are identical too, despite meaning opposite things — the exact word-order problem that motivated sequential models in the first place now reappears at the sentence-representation level. Averaging also has no mechanism to weight some words more heavily than others based on the specific sentence's meaning (a rare, topic-defining word like "photosynthesis" gets the same weight as "the"), and critically, word embeddings are never *trained* with any objective related to sentence-level similarity — their training signal (Word2Vec's context prediction, Lesson 03) only ever concerns individual word co-occurrence, not whether two full sentences mean similar things.

### Dedicated sentence embedding training: contrastive learning

Modern sentence embedding models are trained with an objective *directly* targeting what they'll be used for: making semantically similar sentences have similar embeddings, and dissimilar sentences have different embeddings. The standard approach is **contrastive learning**: given an "anchor" sentence, a genuinely similar "positive" sentence, and one or more dissimilar "negative" sentences, train the model to pull the anchor and positive embeddings close together while pushing the anchor and negative embeddings apart.

```
Anchor:    "A man is playing guitar."
Positive:  "Someone is performing music."          (semantically similar -- should be CLOSE)
Negative:  "A woman is cooking dinner."             (semantically different -- should be FAR)

Training objective (a common contrastive loss, InfoNCE-style):

loss = -log( exp(sim(anchor, positive) / temp) /
             sum over ALL candidates c (positive + negatives) of exp(sim(anchor, c) / temp) )

sim(a, b) = cosine similarity between embeddings a and b
temp = a temperature hyperparameter (same role as Lesson 16's generation temperature --
       controls how sharply the loss distinguishes close vs far scores)
```

This is directly structurally similar to Word2Vec's negative sampling (Lesson 03) — both push a "true" pair's compatibility score up and "false" pairs' scores down — just applied at the sentence level with cosine similarity as the compatibility function, rather than at the word level with a dot product. A common, effective source of positive pairs: NLI datasets (Lesson 21) where a premise/hypothesis pair is labeled "entailment" makes a natural positive pair (they mean similar things), while a "contradiction" pair makes a natural hard negative (superficially related but genuinely different in meaning) — this is precisely why NLI and sentence embedding training are so closely connected in practice, not just conceptually adjacent topics in this module.

### Bi-encoders vs cross-encoders

Two fundamentally different architectures answer "how similar are these two texts," each with a different, real tradeoff:

```
Bi-encoder:                                Cross-encoder:

Text A -> [Encoder] -> vector A            Text A + Text B -> [ONE encoder, JOINTLY] -> similarity score
Text B -> [Encoder] -> vector B                                (processes both texts TOGETHER, with
              |                                                 full cross-attention between them)
     cosine_similarity(vector A, vector B)
```

A **bi-encoder** encodes each text *independently* into a vector, and similarity is a cheap downstream computation (cosine similarity, or a dot product) on the two resulting vectors. Critically, this means every document in a large collection can be encoded *once*, in advance, and stored — at query time, only the (single) query needs encoding, and comparison against millions of pre-computed document vectors is fast (this is exactly what makes dense retrieval, Lesson 14, practical at scale).

A **cross-encoder** processes both texts *together*, in one forward pass through a single model, letting every part of text A directly attend to every part of text B (full cross-attention) before producing a single similarity/relevance score. This is meaningfully more accurate — the model can directly compare fine-grained details between the two texts rather than relying on two independently-computed summary vectors — but it's far more expensive: there's no way to pre-compute anything, since the score genuinely depends on *both* texts together, meaning a cross-encoder must run a fresh forward pass for every single query-document pair being scored.

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Encodes texts | Independently | Jointly (with cross-attention) |
| Can pre-compute document vectors | Yes | No |
| Speed at query time | Fast (one new encoding + cheap similarity) | Slow (one full forward pass per pair) |
| Accuracy | Good | Better (typically) |
| Practical use | First-stage retrieval over large collections | Second-stage re-ranking of a small candidate set |

### The standard combination: retrieve with a bi-encoder, re-rank with a cross-encoder

Because of this speed/accuracy tradeoff, production retrieval systems (extending Lesson 14's retriever, and Lesson 13's retriever-reader pipeline) commonly use both, in sequence: a fast bi-encoder retrieves a broad set of candidates (say, the top 100) from a large collection, and then a slower, more accurate cross-encoder re-ranks *just those 100* candidates to pick the final best few — getting the bi-encoder's necessary speed for searching a huge collection, plus the cross-encoder's superior accuracy where it matters most, applied only to a small enough candidate set that the extra cost is affordable.

```
Large document collection (millions)
         |
         v
   [Bi-encoder retrieval] -- fast, pre-computed document vectors -- top 100 candidates
         |
         v
   [Cross-encoder re-ranking] -- slower but more accurate, only on 100 candidates -- top 5 final results
```

See `code/embedding_models_demo.py` for a from-scratch contrastive sentence embedding training loop on a small synthetic dataset (reusing Lesson 21's NLI-style data as a source of positive/negative pairs), plus a simplified bi-encoder vs cross-encoder comparison illustrating the speed/accuracy tradeoff directly.

## Exercises

1. Implement the contrastive loss above and train a small sentence encoder on synthetic anchor/positive/negative triples. Confirm positive pairs end up with higher cosine similarity than negative pairs after training, compared to before.
2. Compare a trained contrastive sentence embedding's handling of "the dog bit the man" vs "the man bit the dog" against plain averaged-word-embedding cosine similarity on the same pair. Averaging will score them as identical; check whether your trained encoder scores them any lower, and consider what kind of training pairs would be needed to teach it this distinction specifically.
3. Implement a simple cross-encoder (concatenate two texts with a separator token, encode jointly, output a single score) and compare its similarity judgments against a bi-encoder's on a handful of subtly related sentence pairs.
4. Time (via simple wall-clock measurement) scoring 100 candidate documents against one query using a bi-encoder (pre-computed document vectors + cheap comparison) versus a cross-encoder (100 fresh joint forward passes), and report the speed difference.

## Key Terms

| Term | What it actually means |
|---|---|
| Sentence embedding | A dense vector representation of an entire sentence or passage, trained specifically for sentence-level semantic similarity |
| Contrastive learning | A training approach that pulls semantically similar pairs' embeddings together and pushes dissimilar pairs' embeddings apart |
| Anchor / positive / negative | The three roles in a contrastive training example: the reference text, a genuinely similar text, and a dissimilar text |
| Bi-encoder | An architecture that encodes two texts independently, enabling pre-computed document vectors and fast similarity comparison |
| Cross-encoder | An architecture that encodes two texts jointly with cross-attention, more accurate but unable to pre-compute anything |
| Re-ranking | Using a more accurate (typically cross-encoder) model to reorder a smaller candidate set produced by a faster first-stage retriever |
