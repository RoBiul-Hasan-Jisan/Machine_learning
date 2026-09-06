# 16. Text Generation

## Learning Objectives

- Build a character-level language model and use it to generate new text
- Implement and compare greedy, temperature-based, and top-k sampling decoding strategies
- Explain the specific failure modes each decoding strategy is prone to

## The Problem

Lesson 09's one-to-many generation pattern and Lesson 11's beam search both touched on generating a sequence token by token. This lesson makes text generation the direct focus: training a model to predict "what comes next" and then using that trained model to actually produce new, original text — the exact mechanism underlying everything from autocomplete to modern large language models, just at a much smaller scale here.

## The Concept

### Language modeling: predict the next token

A language model learns `P(next_token | previous_tokens)` — given everything generated so far, what's the probability distribution over what comes next. Training this is just another instance of the RNN module's many-to-many pattern: at every position in a training sequence, predict the *next* token, using cross-entropy loss against what actually came next in the real training text.

```
Training text: "the cat sat on the mat"

Training examples (input -> target, teacher-forced, one per position):
  "t"     -> "h"
  "th"    -> "e"
  "the"   -> " "
  "the "  -> "c"
  ...
```

A **character-level** language model (used throughout this lesson's code, for simplicity and a small, self-contained vocabulary) predicts one character at a time; a **word-level** or **subword-level** model (Lesson 19 covers the subword tokenization schemes used by real large language models) predicts one token at a time, using a larger vocabulary. The underlying mechanism — an RNN/LSTM predicting a probability distribution over the next unit, trained via teacher forcing (Lesson 09) — is identical regardless of the granularity chosen.

```python
import torch.nn as nn

class CharLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden
```

### Generation: feed the model's own output back in

Once trained, generation follows exactly the one-to-many feedback pattern from the RNN module's Lesson 04 and Lesson 09's decoder: start from a seed, predict a distribution over the next token, select one token from that distribution, feed it back in as the next input, and repeat.

```
seed: "the"
  -> predict distribution over next char -> select " " -> append -> "the "
  -> predict distribution over next char -> select "c" -> append -> "the c"
  -> predict distribution over next char -> select "a" -> append -> "the ca"
  ... continue until a max length or a special <END> token
```

The *how* of "select one token from the distribution" is the decoding strategy, and different strategies produce meaningfully different generated text from the exact same trained model.

### Greedy decoding: always pick the most likely token

```python
next_token = logits.argmax()
```

Deterministic (same seed always produces the same output) and simple, but prone to a specific, well-known failure: **repetition loops**. If the model's most confident prediction after a certain phrase is to repeat part of that same phrase, greedy decoding has no mechanism to escape — it will happily generate "the the the the the..." forever, since at every step, "the" genuinely is the single highest-probability next token given the immediately preceding repeated context.

### Temperature sampling: control randomness via the softmax distribution

Instead of always taking the argmax, temperature sampling divides the logits by a temperature value *before* the softmax, then samples from the resulting distribution rather than taking the max:

```
P(token) = softmax(logits / temperature)

temperature < 1: sharpens the distribution (more confident, closer to greedy, less random)
temperature = 1: unchanged distribution (as originally predicted by the model)
temperature > 1: flattens the distribution (more random, more diverse, more likely to produce nonsense)
```

```python
import torch
import torch.nn.functional as F

def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

Low temperature (e.g. 0.3-0.5) produces safer, more repetitive-but-coherent text, closer to greedy decoding's behavior. High temperature (e.g. 1.2-1.5) produces more surprising, diverse text, but at real risk of incoherence — pushed far enough, the model starts sampling tokens that are individually unlikely enough to break grammar or meaning entirely. There's no universally "correct" temperature; it's a genuine coherence-vs-diversity tradeoff tuned per application.

### Top-k sampling: sample only from the k most likely tokens

Temperature sampling still has *some* probability of selecting a very unlikely token, however small — top-k sampling instead restricts the candidate pool entirely to the `k` highest-probability tokens, zeroing out everything else, then samples (optionally still with temperature) only among those:

```python
def sample_top_k(logits, k=10, temperature=1.0):
    top_k_logits, top_k_indices = logits.topk(k)
    probs = F.softmax(top_k_logits / temperature, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices[sampled_idx]
```

This directly prevents the "occasionally sample a genuinely bad token" failure mode temperature sampling alone can still produce, while retaining meaningful diversity among the tokens that plausibly *could* come next. A closely related, often preferred variant is **top-p (nucleus) sampling**, which instead includes just enough of the highest-probability tokens for their cumulative probability to reach a threshold `p` (e.g. 0.9) — this adapts the candidate pool size per step (a few tokens when the model is very confident, many more when it's uncertain), rather than always using a fixed `k` regardless of how peaked or flat the distribution actually is at that step.

### Comparing decoding strategies

| Strategy | Behavior | Main risk |
|---|---|---|
| Greedy | Always the single most likely token | Repetition loops |
| Temperature sampling | Random sampling, sharpness controlled by temperature | Incoherence at high temperature |
| Top-k sampling | Random sampling restricted to the k most likely tokens | Fixed k can be too narrow or too wide depending on the step |
| Top-p (nucleus) sampling | Random sampling from the smallest set covering probability p | Generally considered a strong practical default |

See `code/text_generation_demo.py` for a complete character-level LSTM language model trained on a small synthetic corpus, with greedy, temperature, and top-k decoding all implemented and compared side by side — including a direct demonstration of greedy decoding's repetition-loop failure mode.

## Exercises

1. Train the `CharLM` on a small repetitive synthetic text corpus and generate text with greedy decoding. Confirm it eventually falls into a repetition loop.
2. Implement `sample_with_temperature` and generate text at temperatures 0.3, 0.8, and 1.5 from the same seed and trained model. Compare coherence and diversity qualitatively.
3. Implement `sample_top_k` with `k=5` and `k=50` and compare the generated text's diversity at each setting, holding temperature fixed.
4. Implement top-p (nucleus) sampling and compare its behavior against top-k sampling on a step where the model's next-token distribution is very peaked (one dominant token) versus very flat (many similarly likely tokens).

## Key Terms

| Term | What it actually means |
|---|---|
| Language model | A model that predicts a probability distribution over the next token given the preceding tokens |
| Decoding strategy | The method used to select an actual token from a language model's predicted probability distribution during generation |
| Greedy decoding | Always selecting the single highest-probability token, deterministic but prone to repetition loops |
| Temperature | A scaling factor applied to logits before softmax during sampling, controlling how sharp or flat the resulting distribution is |
| Top-k sampling | Restricting sampling to only the k highest-probability tokens at each step |
| Top-p (nucleus) sampling | Restricting sampling to the smallest set of tokens whose cumulative probability reaches a threshold p |
