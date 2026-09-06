# 06. Named Entity Recognition (NER)

## Learning Objectives

- Explain NER as a sequence labeling task and encode labels using the BIO tagging scheme
- Build a feature-based NER tagger and understand what signals it relies on
- Evaluate NER with entity-level (not token-level) precision and recall

## The Problem

Many NLP applications need to pull out specific, structured pieces of information from unstructured text — the person, organization, location, or date mentioned in a sentence — rather than working with the text as an undifferentiated bag of words. "Apple announced a new store in Paris on Monday" contains an organization ("Apple"), a location ("Paris"), and a date ("Monday"), and reliably identifying which spans of text are which type of entity is the Named Entity Recognition task, foundational to information extraction, question answering (Lesson 13), and knowledge graph construction (Lesson 26).

## The Concept

### NER as sequence labeling

Unlike sentiment analysis (Lesson 05, many-to-one), NER is a many-to-many task (the RNN module's Lesson 04 terminology): every *token* in the input gets its own label, not just the sentence as a whole.

```
Tokens: Apple    announced   a   new   store   in   Paris    on   Monday
Labels: ORG      O           O   O     O       O    LOC      O    DATE
```

The complication: entities can span *multiple* tokens ("New York City" is one location entity, three tokens), so a naive per-token label alone can't distinguish "two separate one-word entities next to each other" from "one two-word entity." This is exactly what the BIO tagging scheme solves.

### BIO tagging: encoding entity boundaries

BIO (Beginning, Inside, Outside) tags each token with both its entity type and its position within that entity's span:

```
Tokens: New      York     City     is       home     to       Apple
Labels: B-LOC    I-LOC    I-LOC    O        O        O        B-ORG

B-LOC = Beginning of a LOCATION entity
I-LOC = Inside (continuation of) a LOCATION entity
O     = Outside any entity
```

`B-` marks the first token of an entity span, `I-` marks every subsequent token in that same span, and `O` marks tokens that aren't part of any entity. This lets a sequence labeling model (e.g. the RNN module's many-to-many pattern, or a Transformer-based tagger) distinguish "New York City" (one 3-token entity: `B-LOC I-LOC I-LOC`) from "New York" followed immediately by an unrelated "City Hall" entity (two separate entities: `B-LOC I-LOC` then `B-ORG I-ORG`), a distinction a simpler tag set without the B/I split couldn't represent unambiguously.

### A feature-based NER tagger

Before deep sequence models, NER was commonly framed as a per-token classification problem using hand-engineered features, still a useful way to understand what signals actually help identify entities:

```python
def token_features(tokens, i):
    token = tokens[i]
    return {
        "word": token.lower(),
        "is_capitalized": token[0].isupper(),
        "is_all_caps": token.isupper(),
        "prefix_2": token[:2],
        "suffix_2": token[-2:],
        "prev_word": tokens[i - 1].lower() if i > 0 else "<START>",
        "next_word": tokens[i + 1].lower() if i < len(tokens) - 1 else "<END>",
        "is_first_in_sentence": i == 0,
    }
```

Capitalization is a strong (though imperfect and English-specific) signal for proper nouns; surrounding words provide context ("in Paris" vs "a paris" — hypothetically — signals different likelihoods of "paris" being a location); and prefixes/suffixes can catch morphological patterns (many organization names end in "Inc", "Corp", "LLC"). These features are fed into a sequence classifier (historically Conditional Random Fields, which explicitly model label-to-label transition probabilities — e.g. that `I-LOC` almost never directly follows `B-ORG` — a structural constraint plain per-token classifiers don't enforce). Modern NER systems replace hand-engineered features with learned embeddings (Lessons 03-04) fed into an RNN or Transformer sequence tagger, but the same core "classify every token, using its own features plus its context" pattern remains.

### Evaluating NER: entity-level, not token-level

A tempting but misleading evaluation approach is per-token accuracy. It's misleading because most tokens in real text are `O` (not part of any entity), so a tagger that predicts `O` for everything can score deceptively high token-level accuracy while missing every actual entity. The standard practice instead evaluates at the **entity level**: a predicted entity only counts as correct if it exactly matches the true entity's full span *and* type — get the boundary one token wrong, or the type wrong, and it counts as both a false positive (the wrong entity predicted) and a false negative (the true entity missed), which is a much stricter and more meaningful standard.

```
True entity:      New York City  [LOC], spanning tokens 0-2
Predicted entity:  New York       [LOC], spanning tokens 0-1   <- WRONG (boundary mismatch)

Entity-level scoring: this counts as BOTH a false positive (wrong span predicted)
                       AND a false negative (the true 3-token span was never
                       exactly matched) -- NOT a partial credit "close enough"
```

`seqeval` is the standard Python library for this entity-level evaluation, correctly handling the BIO-scheme boundary logic described above.

See `code/ner_demo.py` for a complete feature-based NER tagger (using scikit-learn's `LogisticRegression` as the per-token classifier) trained and evaluated on a small synthetic dataset, including a demonstration of why token-level accuracy overstates real performance compared to entity-level evaluation.

## Exercises

1. Manually BIO-tag 5 sentences containing multi-token entities (people's full names, multi-word organizations, city names) and confirm every entity's first token gets `B-` and every subsequent token gets `I-`.
2. Implement `token_features` and train a per-token classifier. Identify which features contribute most (e.g. by inspecting feature importances or coefficients) to correctly identifying `B-ORG` tokens.
3. Deliberately construct a case where a tagger predicts the correct entity type but the wrong span boundary (e.g. missing the last token of a 3-token entity). Confirm entity-level scoring counts this as an error while token-level accuracy might still look reasonably high.
4. Compare token-level accuracy and entity-level F1 (via `seqeval` or manual entity matching) on the same set of predictions where most tokens are `O`. Explain the gap.

## Key Terms

| Term | What it actually means |
|---|---|
| Named Entity Recognition (NER) | The task of identifying and classifying spans of text into predefined entity categories (person, organization, location, date, etc.) |
| BIO tagging | A labeling scheme marking each token as the Beginning, Inside, or Outside of an entity span, enabling multi-token entities to be represented unambiguously |
| Entity-level evaluation | Scoring NER predictions by requiring an exact match of both span boundaries and entity type, rather than per-token accuracy |
| Conditional Random Field (CRF) | A sequence classification model that explicitly models transition probabilities between adjacent labels, historically common for NER before deep sequence models |
