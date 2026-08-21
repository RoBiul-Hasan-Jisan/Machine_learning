# 14. Text Generation

## Learning Objectives

- Implement a character-level language model and train it to predict the next character
- Compare greedy, temperature-based, and top-k sampling for generating text from a trained model
- Explain the tradeoff between coherence and diversity that different sampling strategies control

## The Problem

Lesson 04 introduced the one-to-many pattern and Lesson 11's decoder generates output by feeding its own prediction back in as the next input — this lesson makes that concrete and complete for the specific, classic RNN application of language modeling: given the text so far, predict what comes next, one unit (character or word) at a time, and use that trained model to generate new text.

## The Concept

### Language modeling: predict the next token

A language model is trained on a simple, self-supervised objective — given a sequence of tokens, predict the next one — which needs no external labels at all, since the "label" for predicting the token at position `t+1` is simply the actual token that appears there in the training text:

```
Input:   "t", "h", "e", " ", "c", "a"
Target:  "h", "e", " ", "c", "a", "t"     <- input shifted by one position

At every position, the model sees everything up to that point and predicts the NEXT character.
```

This many-to-many pattern (Lesson 04), where every output is a probability distribution over the vocabulary (predicted via softmax, exactly as in the CNN module's forward propagation lesson), is trained with cross-entropy loss comparing the predicted distribution at each position against the actual next token.

### Character-level vs word-level

A **character-level** model treats individual characters as tokens (vocabulary size: a few dozen — letters, digits, punctuation). A **word-level** model treats whole words as tokens (vocabulary size: tens of thousands, per Lesson 13's discussion of embeddings). Character-level models have a much smaller vocabulary and can handle any input (including misspellings or made-up words) without an out-of-vocabulary problem, but need to learn spelling and word structure from scratch and require more time steps to cover the same amount of text. Word-level models start with words as a given unit but need a large embedding table and can't represent words never seen during training. This lesson uses character-level generation since it's simple to set up and demonstrates the core mechanism clearly with a tiny vocabulary.

### The generation loop

Once trained, generating new text means running the model in the one-to-many, feed-your-own-output-back-in mode from Lessons 04 and 11:

```
Start with a seed (e.g. a single character or short prompt)
h = initial hidden state (from processing the seed, or zeros)

repeat:
    logits = model(current_token, h)      # predicted distribution over next token
    next_token = SAMPLE from logits        # <- how this sampling works is the crux of this lesson
    output.append(next_token)
    current_token = next_token             # feed it back in as the next input
    h = updated hidden state
until desired length or an <END> token
```

The one open question this loop leaves unspecified is exactly *how* to turn the predicted probability distribution into an actual chosen token — "SAMPLE from logits" above — and that choice has a large, direct effect on the quality and character of the generated text.

### Greedy decoding: always pick the most likely token

```python
next_token = logits.argmax()
```

Simple and fully deterministic (same seed always produces the same output), but tends to produce repetitive, overly "safe" text — because the model always picks its single highest-probability choice at every step, it can get stuck in loops (repeating the same phrase) and never explores plausible alternative continuations, some of which might actually be more interesting or natural.

### Temperature sampling: control randomness with one knob

Instead of always taking the argmax, sample from the full probability distribution, but first reshape that distribution with a **temperature** parameter `T`:

```
p_i = softmax(logits_i / T)
```

```
T < 1 (e.g. 0.5):  sharpens the distribution -> more confident, closer to greedy, less random
T = 1:              unchanged -- sample directly from the model's own predicted distribution
T > 1 (e.g. 1.5):  flattens the distribution -> more random, more diverse, but riskier/less coherent
```

Low temperature produces safer, more repetitive but more grammatically reliable text (approaching greedy decoding as `T -> 0`); high temperature produces more varied, surprising, sometimes incoherent text (approaching uniform random sampling as `T -> infinity`). Temperature is the single most common knob exposed to users of text-generation systems for exactly this reason — it's a simple, direct way to trade off coherence against creativity/diversity.

### Top-k sampling: restrict to the k most likely candidates

An alternative (or complementary) approach: instead of reshaping the whole distribution, restrict sampling to only the `k` highest-probability tokens, zero out everything else, renormalize, and sample from that restricted set:

```python
top_k_logits, top_k_indices = logits.topk(k)
probs = softmax(top_k_logits)
sampled_index = top_k_indices[multinomial_sample(probs)]
```

This avoids ever sampling a token from the distribution's long, unreliable tail (very-low-probability tokens the model is essentially guessing at), while still allowing meaningful randomness among the genuinely plausible candidates — a middle ground between greedy's total determinism and full-distribution sampling's occasional wild, low-probability choices. `k` itself is a tunable parameter: small `k` (e.g. 5) stays close to greedy; large `k` approaches full-distribution sampling.

### Putting it together

Temperature and top-k are often combined in practice (apply temperature first, then restrict to the top-k of the reshaped distribution), and more advanced variants (top-p / nucleus sampling, which restricts to the smallest set of tokens whose cumulative probability exceeds a threshold `p`, adapting the candidate set size dynamically rather than fixing `k`) exist as further refinements — but temperature and top-k cover the core idea this lesson is teaching: the choice of sampling strategy is a genuine, consequential design decision, not an afterthought bolted onto a trained model.

See `code/text_generation_demo.py` for a complete character-level language model trained on a small text corpus, with greedy, temperature-based, and top-k generation implemented and compared side by side on the same trained model.

## Exercises

1. Train the character-level model on a short repeated text corpus and generate text using greedy decoding. Observe and describe any repetitive patterns in the output.
2. Generate text from the same trained model at temperatures 0.3, 1.0, and 2.0, and compare the outputs qualitatively — which is most repetitive, which is most chaotic?
3. Implement top-k sampling with `k=3` and `k=20` and compare the generated text's coherence and diversity at each setting.
4. Implement top-p (nucleus) sampling and compare its behavior to top-k on the same trained model and the same prompt.

## Key Terms

| Term | What it actually means |
|---|---|
| Language model | A model trained to predict the next token in a sequence, given everything before it, usually via self-supervised training on unlabeled text |
| Character-level model | A language model where individual characters, rather than whole words, are the vocabulary units |
| Greedy decoding | Always selecting the single highest-probability token at each generation step |
| Temperature | A parameter reshaping a predicted probability distribution before sampling, controlling the tradeoff between determinism and randomness |
| Top-k sampling | Restricting sampling to only the k highest-probability tokens at each step, avoiding the unreliable low-probability tail of the distribution |
| Top-p (nucleus) sampling | Restricting sampling to the smallest set of tokens whose cumulative probability exceeds a threshold p, adapting the candidate set size dynamically |
