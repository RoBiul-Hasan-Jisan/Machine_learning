# 12. Attention Mechanism

## Learning Objectives

- Explain how attention removes the fixed-size context vector bottleneck from Lesson 11
- Compute attention weights and a context vector from scratch, given a set of encoder hidden states and a decoder query
- Implement an attention-augmented decoder step and inspect which input positions it attends to

## The Problem

Lesson 11 ended on the bottleneck problem: a plain encoder-decoder model compresses the *entire* input sequence into one fixed-size context vector, and every decoding step has to work from that single summary. For longer sequences, this is a real, measurable limitation — there's a hard ceiling on how much a fixed-size vector can represent, no matter the input length. Attention (Bahdanau et al., 2014; refined by Luong et al., 2015) removes this bottleneck by letting the decoder look back at *every* encoder hidden state directly, at every decoding step, rather than working from one compressed summary alone.

## The Concept

### The core idea: a learned, per-step weighted average over all encoder states

Instead of the decoder only ever seeing the encoder's *final* hidden state (Lesson 11's context vector), attention gives the decoder access to *every* encoder hidden state, `h_1, h_2, ..., h_T` — one per input position — and lets it compute a fresh, different weighted combination of them at every single decoding step, based on what that step currently needs.

```
Encoder produces:  h_1, h_2, h_3, h_4        (one hidden state per input position, NOT just the final one)

At decoder step t, given the decoder's current state s_t:

1. Compute an alignment score between s_t and EVERY encoder hidden state:
     score(s_t, h_1), score(s_t, h_2), score(s_t, h_3), score(s_t, h_4)

2. Turn the scores into weights that sum to 1 (softmax):
     alpha_1, alpha_2, alpha_3, alpha_4   =   softmax(scores)

3. Compute a context vector as the WEIGHTED AVERAGE of encoder states:
     context_t = alpha_1*h_1 + alpha_2*h_2 + alpha_3*h_3 + alpha_4*h_4

4. Use context_t (along with s_t) to produce this step's output
```

Crucially, `context_t` is recomputed at *every* decoding step, with different weights each time — step 1 of decoding might weight `h_3` heavily (attend mostly to input position 3), while step 2 might weight `h_1` and `h_4` (attend to different positions), depending entirely on what's relevant for producing that particular output token. This directly removes Lesson 11's bottleneck: the decoder is no longer limited to one fixed summary of the whole input — it can access the full set of `T` encoder states and dynamically choose, per step, which ones matter.

### Computing alignment scores

Several scoring functions are used in practice; a common one (Bahdanau-style "additive" attention) is:

```
score(s_t, h_i) = v^T @ tanh(W_s @ s_t + W_h @ h_i)
```

`W_s`, `W_h`, and `v` are learned weights — the network learns what makes an encoder position "relevant" to a given decoder state, rather than this being hand-specified. A simpler and now more common alternative (Luong-style "multiplicative"/"dot-product" attention, and the basis for the Transformer's attention mechanism in Lesson 18) is:

```
score(s_t, h_i) = s_t^T @ h_i          (a plain dot product, no extra learned weights needed)
```

Either way, the output is one scalar score per encoder position, which softmax then turns into a valid probability distribution (the `alpha` weights) over the input positions.

### Attention weights are directly interpretable

Because `alpha_1, ..., alpha_T` sum to 1 and each corresponds to a specific input position, they can be visualized directly — as a heatmap of decoder step vs encoder position — to see exactly which parts of the input the model is "looking at" when producing each output token. This was, at the time, a significant practical advance over the fully opaque fixed-context-vector approach: for a translation task, plotting attention weights typically shows something close to the expected word-alignment between source and target languages, giving a genuine window into what the model is doing, not just a performance number.

### Implementing attention from scratch

```python
def attention(decoder_state, encoder_states, W_s, W_h, v):
    """decoder_state: (hidden_size,). encoder_states: list of (hidden_size,), length T."""
    scores = []
    for h_i in encoder_states:
        score = v @ np.tanh(W_s @ decoder_state + W_h @ h_i)
        scores.append(score)

    scores = np.array(scores)
    weights = softmax(scores)                      # alpha_1 ... alpha_T, sums to 1

    context = sum(w * h for w, h in zip(weights, encoder_states))
    return context, weights
```

At each decoding step, this `context` vector (recomputed fresh every step, unlike Lesson 11's single fixed vector) is concatenated with the decoder's own input/state and fed into the decoder cell, giving the decoder access to a step-specific, learned summary of the most relevant parts of the entire input.

### Attention's broader significance

Attention was introduced specifically to fix seq2seq's bottleneck problem, but it turned out to be a much more general and powerful idea than that one fix: it's a mechanism for letting a model dynamically decide, at each step, which parts of some larger context matter most right now — a pattern useful far beyond translation. This generality is exactly what Lesson 18 picks up: the Transformer architecture takes attention and makes it the *entire* mechanism for processing a sequence, discarding recurrence altogether, which turns out to solve not just the bottleneck problem but also the sequential-processing bottleneck (Lesson 01's parallelization limitation) that no RNN variant, however gated or attention-augmented, can escape on its own.

See `code/attention_demo.py` for a from-scratch implementation of attention scoring and context vector computation, integrated into a small attention-augmented decoder trained on the same reversal task from Lesson 11, with a visualization of the resulting attention weights confirming the decoder learns to attend to the correct (reversed) input position at each step.

## Exercises

1. Implement the `attention` function above and verify the output weights sum to 1 for a toy set of encoder states and a decoder state.
2. Implement dot-product attention (`score = s_t @ h_i`, no extra learned weights) and compare its behavior to additive attention on the same toy inputs.
3. Extend Lesson 11's `Seq2Seq` model with an attention layer between the encoder and decoder, train it on the reversal task, and confirm accuracy improves (or converges faster) compared to the non-attention version, especially at longer sequence lengths.
4. Extract and visualize (as a matrix printout or a simple heatmap) the attention weights for a specific trained example, and confirm that step `t` of decoding tends to attend most strongly to the correct corresponding input position for the reversal task.

## Key Terms

| Term | What it actually means |
|---|---|
| Attention mechanism | A mechanism that computes a dynamic, per-step weighted combination of all encoder hidden states, rather than relying on one fixed context vector |
| Alignment score | A learned or computed measure of relevance between a decoder state and a specific encoder hidden state |
| Attention weights | The softmax-normalized alignment scores, summing to 1, indicating how much each input position contributes to the current step's context vector |
| Context vector (with attention) | The weighted average of encoder hidden states at a specific decoding step, recomputed fresh at every step, as opposed to Lesson 11's single fixed vector |
