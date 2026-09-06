# 18. Multilingual NLP

## Learning Objectives

- Identify which of this module's techniques are language-agnostic and which are English-specific
- Explain cross-lingual embeddings and how they enable transfer across languages
- Understand zero-shot and few-shot cross-lingual transfer and why multilingual models make it possible

## The Problem

Every lesson so far has used English examples, and several techniques quietly relied on English-specific assumptions — whitespace-separated words (Lesson 01's tokenizer), capitalization as an NER signal (Lesson 06), or a POS tagset built around English grammar (Lesson 07). Most of the world's text isn't English, and building a separate model from scratch for every one of the world's ~7,000 languages is completely impractical, especially for low-resource languages with little available training data. Multilingual NLP addresses both the technical adaptations needed and the strategies for sharing learned knowledge across languages.

## The Concept

### Which techniques from this module are language-agnostic, and which aren't

| Technique | Language-agnostic? | Why / why not |
|---|---|---|
| Whitespace/regex tokenization (Lesson 01) | No | Chinese, Japanese, Thai don't separate words with spaces; needs language-specific segmentation |
| Subword tokenization — BPE, etc. (Lesson 19) | Mostly yes | Learned directly from data, works reasonably across languages, though vocabulary efficiency varies |
| Stemming (Lesson 01) | No | Suffix-stripping rules are entirely language-specific (a Porter-style stemmer only works for English morphology) |
| TF-IDF / BM25 (Lessons 02, 14) | Yes | Purely statistical, no language-specific assumptions once tokenized |
| Capitalization as an NER signal (Lesson 06) | No | Many languages don't capitalize proper nouns the way English does (or have no case distinction at all) |
| Word embeddings (Lessons 03-04) | Yes, per-language | The *method* works in any language, but produces separate, incompatible embedding spaces unless specifically trained to align (see below) |
| Neural sequence models (RNN module, Lessons 08-11) | Yes | The architecture itself doesn't assume English; it learns whatever patterns exist in its training data |

The practical takeaway: architectures and statistical methods generalize; specific hand-crafted rules (stemming suffixes, capitalization heuristics) usually don't, and need either a language-specific replacement or, increasingly, replacement by a learned (data-driven) alternative that doesn't require per-language hand engineering at all.

### Language-specific tokenization challenges

Languages without whitespace word boundaries (Chinese, Japanese, Thai, and others) need a **word segmentation** step before anything resembling Lesson 01's tokenization can apply — determining where one word ends and the next begins is itself a non-trivial task, often requiring a trained segmentation model or a large dictionary of known words, since there's no simple universal rule (unlike splitting on whitespace) that works. Subword tokenization schemes (Lesson 19) sidestep much of this: since they learn units directly from raw character sequences in a training corpus, they work reasonably even for languages without clear word boundaries, without requiring a dedicated segmentation step first — one of several reasons subword tokenization became the default across essentially all modern multilingual models.

### Cross-lingual embeddings: aligning separate vector spaces

Training Word2Vec (Lesson 03) separately on an English corpus and a French corpus produces two embedding spaces that are each internally coherent (English "dog" is near English "puppy"; French "chien" is near French "chiot") but not directly comparable to each other — English "dog" and French "chien" (the same underlying concept) can end up in completely unrelated regions of their respective spaces, since nothing in either training process ever related the two languages to each other.

**Cross-lingual embedding alignment** fixes this, typically using a small bilingual dictionary (a few thousand known word-translation pairs) to learn a transformation — usually a single linear mapping — that rotates one language's embedding space to align with the other's:

```
Given known translation pairs (dog, chien), (cat, chat), (house, maison), ...

Learn a transformation matrix W such that:
    W @ embedding_english("dog")  ≈  embedding_french("chien")
    W @ embedding_english("cat")  ≈  embedding_french("chat")
    ... (fit via least-squares over all known pairs)

Once learned, apply W to ANY English word's embedding to project it into
the French embedding space, even words never seen in the bilingual dictionary
```

The surprising empirical finding that makes this work at all: embedding spaces trained independently on different languages tend to have *similar internal geometric structure* (the relative positions of "king," "queen," "man," "woman" form a similar pattern in many languages' embedding spaces, even though the absolute positions differ) — meaning a single linear transformation, fit from a relatively small number of known translation pairs, can align the two spaces surprisingly well, extending correctly even to word pairs never included in the training dictionary.

### Multilingual models and zero-shot cross-lingual transfer

Rather than aligning separately-trained embeddings after the fact, modern multilingual models (like multilingual BERT or XLM-R, covered further as embedding models in Lesson 22) are trained *jointly* on many languages at once, typically using a single shared subword vocabulary (Lesson 19) spanning all the languages in training. This produces a genuinely shared representation space where similar concepts across different languages naturally end up near each other, as an emergent property of joint training rather than a separate alignment step.

This joint training enables a striking capability: **zero-shot cross-lingual transfer** — fine-tune the model on a task (e.g. sentiment classification, Lesson 05) using labeled data in *only* one language (typically English, since labeled English data is usually most abundant), and the model often performs surprisingly well on the *same task in a different language it saw during pretraining but never saw labeled examples for*, purely because the shared multilingual representation space transfers the learned task knowledge across languages. **Few-shot transfer** is the same idea with a small amount of labeled target-language data added, typically improving further on pure zero-shot performance. This is especially valuable for low-resource languages, where labeled task-specific data may simply not exist, but where the language was still represented in the model's broader multilingual pretraining.

### Practical implications

| Approach | When to use it |
|---|---|
| Train separate monolingual models | Maximum performance for one specific, well-resourced language |
| Align embeddings post-hoc | Adding cross-lingual capability to existing monolingual embeddings without retraining |
| Use a pretrained multilingual model | Need to support many languages, especially low-resource ones, with a single system |
| Zero-shot cross-lingual transfer | Labeled data exists in one language, but the task needs to work in others too |

See `code/multilingual_demo.py` for a from-scratch cross-lingual embedding alignment (learning a linear mapping between two independently-trained toy embedding spaces using a bilingual dictionary) and a demonstration of translation lookup via the aligned space, including generalization to word pairs never seen during alignment training.

## Exercises

1. Train two separate Word2Vec-style embedding spaces (Lesson 03) on two small synthetic "language" corpora with a shared underlying concept structure but different vocabulary, and confirm the two spaces are not directly comparable before alignment.
2. Implement the linear alignment transformation using a small dictionary of known translation pairs, and verify it correctly maps a held-out translation pair not used in fitting the transformation.
3. Identify 3 techniques from earlier lessons in this module that would need modification for Chinese text specifically (no whitespace word boundaries), and describe what modification each would need.
4. Research (via web search) what "catastrophic forgetting" means in the context of fine-tuning a multilingual model on one language's labeled data, and discuss why this could undermine zero-shot transfer if not managed carefully.

## Key Terms

| Term | What it actually means |
|---|---|
| Word segmentation | Determining word boundaries in languages without whitespace separation between words, a required preprocessing step in those languages |
| Cross-lingual embedding alignment | Learning a transformation (often linear) that maps one language's embedding space onto another's, using known translation pairs |
| Multilingual model | A model trained jointly on many languages at once, typically producing a shared representation space across them |
| Zero-shot cross-lingual transfer | Applying a model fine-tuned on labeled data in one language directly to the same task in a different language, without labeled data in that language |
| Few-shot transfer | The same idea as zero-shot transfer, but with a small amount of labeled target-language data added to improve performance |
