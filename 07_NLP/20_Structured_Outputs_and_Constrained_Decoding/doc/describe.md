# 20. Structured Outputs & Constrained Decoding

## Learning Objectives

- Explain why free-form generation (Lesson 16) is unreliable for tasks needing a specific output format
- Implement grammar-constrained decoding that masks invalid tokens at each generation step
- Understand JSON schema-constrained generation as a practical, widely-used special case

## The Problem

Lesson 16 covered generating free-form text, sampled from a model's predicted distribution. Many practical applications don't want free-form text at all — they need output that's guaranteed to be valid JSON matching a specific schema, a syntactically correct SQL query, or a value from a fixed set of categories, because the output will be parsed programmatically by other code immediately afterward. A model trained purely to generate plausible text will *usually* produce valid JSON when asked nicely, but "usually" isn't good enough when a single malformed brace breaks an automated pipeline — structured output generation needs a stronger guarantee than "the model was probably trained on enough examples like this."

## The Concept

### Why prompting alone isn't enough

The most basic approach — just ask the model to "output valid JSON" in the prompt — relies entirely on the model's learned tendency to follow instructions, with no hard guarantee. Real failure modes are common: a trailing comma, an unescaped quote inside a string, a missing closing brace, or a subtly wrong field name. These failures are individually rare per generation but become a real reliability problem at scale (a system generating thousands of structured outputs per day will hit malformed output regularly, even at, say, a 99% per-output success rate), and "retry and hope" is not a robust engineering solution to a problem with a much stronger fix available.

### Constrained decoding: only allow valid tokens at each step

The stronger fix builds directly on Lesson 16's decoding machinery: instead of sampling freely from the model's full predicted distribution at each step, **mask out every token that would make the output invalid**, given everything generated so far, before sampling — the model can only ever choose among tokens that keep the output on a valid path.

```
Normal decoding (Lesson 16):
  P(next_token) = softmax(logits)          <- sample from the FULL distribution

Constrained decoding:
  valid_mask = compute_valid_next_tokens(generated_so_far, grammar_or_schema)
  masked_logits = logits.masked_fill(~valid_mask, -inf)
  P(next_token) = softmax(masked_logits)   <- sample ONLY from tokens that keep output valid
```

Setting invalid tokens' logits to `-inf` before the softmax makes their probability exactly 0 — this is a hard, guaranteed constraint, not a soft preference the model might override. Whatever gets generated is *guaranteed* structurally valid by construction, not merely "probably" valid because the model was trained well.

### A simple grammar-constrained example: generating only digits and specific punctuation

```python
import torch

def make_digit_mask(vocab, allowed_chars="0123456789.-"):
    mask = torch.zeros(len(vocab), dtype=torch.bool)
    for token, idx in vocab.items():
        if all(c in allowed_chars for c in token):
            mask[idx] = True
    return mask

def constrained_generate(model, seed, vocab, idx_to_char, valid_mask, length=10):
    ids = [vocab[c] for c in seed]
    hidden = None
    generated = list(seed)
    x = torch.tensor([ids], dtype=torch.long)
    logits, hidden = model(x, hidden)
    for _ in range(length):
        step_logits = logits[0, -1].clone()
        step_logits[~valid_mask] = float("-inf")   # hard constraint: forbidden tokens can NEVER be chosen
        next_id = step_logits.argmax().item()
        generated.append(idx_to_char[next_id])
        x = torch.tensor([[next_id]], dtype=torch.long)
        logits, hidden = model(x, hidden)
    return "".join(generated)
```

This tiny example enforces a fixed character set at every step, regardless of position. Real structured-output systems need a *stateful* constraint — which tokens are valid depends on the position within the JSON structure (right after a `{`, only a `"` or `}` is valid; right after a key's closing `"`, only a `:` is valid, and so on) — computed fresh at every step from the grammar's current state, not a single fixed mask reused throughout.

### JSON schema-constrained generation

The most common real-world application of this technique constrains generation to valid JSON matching a specific schema (field names, types, required fields) known in advance. Conceptually:

```
Schema: {"name": string, "age": integer, "active": boolean}

At each generation step, the constraint engine tracks:
  - Are we inside a string, or between tokens?
  - Which field (if any) are we currently filling in, and what TYPE does it expect?
  - Have all required fields been filled yet, or is a "}" not yet valid?

Only tokens consistent with the current state AND the schema are allowed at each step:
  - If currently filling the "age" field (expects an integer), only digit tokens
    (and a closing quote-free numeric terminator) are valid -- letters are masked out
  - A closing "}" is only valid once every required field has been filled
```

This is exactly how modern structured-output APIs and libraries (JSON-schema-constrained generation, grammar-constrained decoding frameworks) work under the hood: they compile a schema or formal grammar into a state machine, and at every generation step, use the state machine's current state to compute which tokens are currently valid, masking everything else before sampling — precisely the `masked_logits` pattern above, just with a considerably more sophisticated, stateful mask computation than a fixed allowed-character-set.

### The tradeoff: guaranteed validity vs generation quality

Constrained decoding guarantees *structural* validity (the output will parse as valid JSON matching the schema) but says nothing about *content* correctness (a validly-formatted JSON object can still contain a factually wrong or nonsensical value in a field). It's also a real constraint on the model's freedom — in rare cases, forcing the model down a valid-but-constrained path can produce lower-quality content than what the model would have generated completely unconstrained, since the model's most natural continuation at some step might be exactly the token the mask just forbade. In practice, this tradeoff is almost always worth it for genuinely structured tasks: a slightly less fluent but always-parseable JSON object is far more useful in an automated pipeline than an unconstrained model's occasionally-broken output.

See `code/constrained_decoding_demo.py` for a complete constrained character-level generator enforcing a simple custom grammar (a restricted `{"key": value}` format), demonstrating both the specific position-dependent masking logic and a side-by-side comparison against unconstrained generation showing how often the unconstrained version breaks the format.

## Exercises

1. Implement `make_digit_mask` and confirm a language model, when constrained to it, can never generate a letter character regardless of what the underlying model's raw predictions would have preferred.
2. Extend the fixed-mask example into a position-dependent one: build a tiny state machine for a restricted grammar (e.g. `(digit)(digit)-(digit)(digit)` for a 2-digit-dash-2-digit code) and confirm the valid token set changes correctly at each position.
3. Generate 20 outputs from an unconstrained character-level model prompted to produce a specific format (e.g. `{"x": N}` for some digit N), and count how many are actually well-formed. Compare against 20 outputs from the same model under a hard constraint mask.
4. Discuss, in your own words, a scenario where constrained decoding's "guaranteed structurally valid, but not necessarily correct content" tradeoff would be an acceptable risk, and one where it would not be (e.g. a low-stakes UI form autofill vs a medical dosage field).

## Key Terms

| Term | What it actually means |
|---|---|
| Constrained decoding | Generation where invalid tokens are masked out (probability forced to zero) at each step, guaranteeing the output stays within a valid structure |
| Logit masking | Setting the logits of forbidden tokens to negative infinity before the softmax, making their sampling probability exactly zero |
| Grammar-constrained generation | Constraining generation to match a formal grammar, using a state machine to determine which tokens are valid at each step |
| Schema-constrained generation | The common special case of grammar-constrained generation targeting a specific JSON (or similar) schema |
