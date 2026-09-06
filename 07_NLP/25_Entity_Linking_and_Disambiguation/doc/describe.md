# 25. Entity Linking & Disambiguation

## Learning Objectives

- Distinguish entity linking from NER (Lesson 06): finding a mention vs identifying which real-world entity it refers to
- Implement candidate generation and disambiguation ranking for entity linking
- Explain why context is essential for resolving genuinely ambiguous entity names

## The Problem

NER (Lesson 06) identifies *that* a span of text is an entity and *what type* it is ("Paris" is a `LOCATION`). It says nothing about *which specific* Paris is meant — Paris, France? Paris, Texas? A person named Paris? Entity linking (also called entity disambiguation) is the next step: connecting a recognized entity mention to a specific, unique entry in a knowledge base (like Wikidata or a company's internal entity database), resolving exactly which real-world thing is being referred to.

## The Concept

### NER identifies mentions; entity linking identifies referents

```
Text: "Paris hosted the Olympics in 2024."

NER (Lesson 06):        "Paris" -> LOCATION            (what TYPE of entity)
Entity linking:          "Paris" -> Wikidata Q90         (which SPECIFIC entity: Paris, France,
                                                            not Paris, Texas or any other candidate)
```

Every entity linking pipeline has two distinct stages, each solving a different sub-problem:

```
Mention: "Paris"
    |
    v
[Candidate Generation] -- find EVERY plausible entity this mention could refer to
    |                      (Paris, France; Paris, Texas; Paris Hilton; ...)
    v
[Disambiguation / Ranking] -- use CONTEXT to pick the single most likely candidate
    |                          ("hosted the Olympics" strongly favors Paris, France)
    v
Final linked entity: Paris, France (Wikidata Q90)
```

### Candidate generation: cast a wide net

Given a mention string, candidate generation retrieves every entity in the knowledge base that the mention could plausibly refer to — this is essentially the information retrieval problem from Lesson 14, applied to a knowledge base of entities rather than documents. A simple approach uses an alias dictionary (a knowledge base entry lists every name/nickname/alternate spelling it's known by) and looks up exact or fuzzy matches:

```python
def generate_candidates(mention, knowledge_base):
    """knowledge_base: {entity_id: {"names": [...], "type": ..., "description": ...}}"""
    candidates = []
    mention_lower = mention.lower()
    for entity_id, info in knowledge_base.items():
        if any(mention_lower == name.lower() for name in info["names"]):
            candidates.append(entity_id)
    return candidates
```

This stage deliberately favors recall over precision — it's fine (even expected) to return several plausible candidates for a genuinely ambiguous name; the job of narrowing down to exactly one happens in the next stage. Missing the correct candidate entirely at this stage is a much worse failure than including a few incorrect ones, since disambiguation can only choose among what candidate generation actually retrieved.

### Disambiguation: use context to rank candidates

Given a mention's surrounding context (the rest of the sentence, or the whole document) and a set of candidate entities, disambiguation scores each candidate by how well it fits that context — typically by comparing the context's content against each candidate's known description, related entities, or typical co-occurring terms:

```python
def disambiguate(mention_context, candidates, knowledge_base):
    """Score each candidate by word overlap between the mention's context
    and the candidate's description (Lesson 02's bag-of-words idea, applied
    to entity descriptions rather than documents)."""
    context_words = set(mention_context.lower().split())

    best_candidate, best_score = None, -1
    for entity_id in candidates:
        description_words = set(knowledge_base[entity_id]["description"].lower().split())
        overlap_score = len(context_words & description_words)
        if overlap_score > best_score:
            best_score = overlap_score
            best_candidate = entity_id
    return best_candidate
```

This simple word-overlap approach is a reasonable baseline (and directly analogous to Lesson 14's sparse retrieval matching), but modern systems typically use dense embedding similarity instead (Lesson 22's sentence/entity embeddings), comparing the mention's contextual embedding against each candidate entity's embedding — capturing semantic relatedness beyond exact word overlap, the same sparse-vs-dense distinction from Lesson 14 reappearing here.

### Popularity priors: a useful, imperfect shortcut

A practical detail real systems lean on heavily: when context is sparse or genuinely ambiguous, defaulting to the *most commonly referenced* candidate for a given mention string is often a strong baseline — "Paris" without any disambiguating context is far more often Paris, France than any other candidate, simply because it's referenced vastly more often across most text collections. This **popularity prior** is cheap to compute (just count how often each candidate is the correct link across a large reference corpus) and surprisingly effective, but it's a real source of bias: it will confidently and incorrectly link every mention of a less-famous entity to its more-famous same-named counterpart whenever context is weak or entirely absent, silently favoring well-known entities over legitimate but rarer ones with the same name.

### Why context is essential, not optional

```
"Paris hosted the Olympics in 2024."           -> context clearly favors Paris, France
"Paris is the capital of a small county in Texas." -> context clearly favors Paris, Texas
"Paris posted a new song on social media."      -> context suggests Paris Hilton (or another
                                                     person named Paris), not either city
```

The exact same mention string, with three different surrounding contexts, correctly links to three completely different entities — this is precisely why entity linking cannot be solved by a simple string-to-entity lookup table (that would only work for genuinely unambiguous names) and why the disambiguation stage, using real contextual information, is the core of the whole task rather than an optional refinement step.

### Where entity linking connects to the rest of this module

Entity linking is a natural continuation of NER (Lesson 06) toward genuinely structured knowledge, and it's the direct prerequisite for relation extraction and knowledge graph construction (Lesson 26) — you can't correctly build a fact like "Paris, France hosted the 2024 Olympics" in a knowledge graph without first resolving *which* Paris is meant; linking the mention to the wrong entity would silently corrupt the resulting knowledge graph with a false, misattributed fact.

See `code/entity_linking_demo.py` for a complete candidate generation + context-based disambiguation pipeline over a small synthetic knowledge base with genuinely ambiguous entity names, correctly resolving the same mention string to different entities depending on context.

## Exercises

1. Build a small knowledge base (5-10 entities, including at least 2 sharing the same name/alias) and implement `generate_candidates`. Confirm ambiguous mentions correctly return multiple candidates.
2. Implement `disambiguate` using word overlap and test it on 3 sentences using the same ambiguous mention string in different contexts, confirming each resolves to the intended entity.
3. Add a popularity prior (a fixed "default" candidate per mention string) and test what happens when a sentence's context is too sparse/generic for word-overlap disambiguation to confidently distinguish candidates — confirm the prior provides a sensible fallback.
4. Construct a case where the popularity prior actively produces the WRONG answer (context clearly favors the less popular candidate, but word overlap is weak/tied) and discuss what additional signal (e.g. stronger context features, or a learned disambiguation model) would be needed to fix it.

## Key Terms

| Term | What it actually means |
|---|---|
| Entity linking (entity disambiguation) | Connecting a recognized entity mention to a specific, unique entry in a knowledge base |
| Candidate generation | The stage of entity linking that retrieves every plausible entity a mention could refer to, favoring recall |
| Disambiguation | The stage of entity linking that uses context to select the single correct candidate from those generated |
| Popularity prior | Defaulting to the most commonly referenced entity for a given mention string when context is weak or absent |
| Knowledge base | A structured collection of entities, their names/aliases, and descriptive information, used as the target for entity linking |
