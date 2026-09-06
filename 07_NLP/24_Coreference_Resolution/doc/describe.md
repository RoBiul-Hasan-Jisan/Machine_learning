# 24. Coreference Resolution

## Learning Objectives

- Define coreference resolution and distinguish it from simple pronoun lookup
- Implement a rule-based (Hobbs-style) pronoun resolver using syntactic heuristics
- Explain why coreference resolution matters for information extraction tasks like relation extraction (Lesson 26)

## The Problem

Text constantly refers back to things it already mentioned, using pronouns ("he," "she," "it," "they") or shorter descriptions ("the company," "the researcher") instead of repeating the full original name every time — natural for human readers, but a real obstacle for automated text understanding, since "he approved the deal" is meaningless to a system unless it knows who "he" refers to. Coreference resolution is the task of identifying which spans of text refer to the same real-world entity, linking pronouns and shortened references back to their original, fully-specified mention.

## The Concept

### What coreference resolution actually solves

```
"Sarah met with John yesterday. She told him about the new project.
 He seemed excited about it, and the two of them agreed to meet again next week."

Coreference chains:
  {Sarah, She}
  {John, him, He}
  {the new project, it}
  {Sarah and John, the two of them}
```

A **coreference chain** (or "cluster") is the set of all mentions in a text that refer to the same underlying entity. Resolving these chains means every later reference can be traced back to the original, fully specific mention — turning "he seemed excited about it" into something a downstream system can actually use ("John seemed excited about the new project"), rather than an ungrounded pronoun reference.

This is meaningfully harder than it sounds because pronoun resolution is genuinely ambiguous without real understanding: "The city council refused the demonstrators a permit because they feared violence" vs "...because they advocated violence" — the same sentence structure, with "they" referring to the council in one case and the demonstrators in the other, distinguishable only by which interpretation actually makes sense given "feared" vs "advocated" — a famous example (the "Winograd Schema") specifically designed to require real-world reasoning, not just syntax, to resolve correctly.

### Rule-based resolution: syntactic heuristics

Before machine-learned approaches, coreference resolution relied on hand-crafted syntactic heuristics, most famously the Hobbs algorithm (1978), which searches a sentence's parse tree (Lesson 07) in a specific, defined order looking for the nearest candidate antecedent that agrees in number and gender with the pronoun being resolved. A simplified version of the core heuristics:

```python
def simple_pronoun_resolver(sentences, pronoun_position):
    """A simplified heuristic resolver: search backward from the pronoun
    for the nearest preceding noun phrase matching in number/gender."""
    pronoun = sentences[pronoun_position]["token"]
    gender = pronoun_gender(pronoun)   # 'he'/'him' -> male, 'she'/'her' -> female, etc.
    number = pronoun_number(pronoun)   # 'they'/'them' -> plural, others -> singular

    # search backward through preceding tokens for the nearest matching candidate
    for i in range(pronoun_position - 1, -1, -1):
        candidate = sentences[i]
        if candidate["pos"] == "NOUN" and candidate.get("is_proper", False):
            if candidate_matches(candidate, gender, number):
                return candidate
    return None
```

Key heuristics real rule-based systems layer on top of simple "nearest candidate" search:

- **Recency preference**: a more recently mentioned candidate is generally more likely to be the correct antecedent than one mentioned much earlier, all else equal.
- **Grammatical role preference**: candidates that were the subject of their sentence tend to be preferred over candidates that were the object, reflecting a genuine tendency in how people actually use pronouns.
- **Number/gender agreement**: a hard filter — "she" can never resolve to a candidate explicitly marked male, and "they" (in its plural sense) can't resolve to a clearly singular candidate.
- **Binding constraints**: syntactic rules from formal linguistics about which pronouns *cannot* refer to which candidates within the same clause (e.g. "John hurt him" cannot have "him" refer to John himself, in normal English — that meaning would require "himself" instead).

### Why full resolution needs more than syntax

The Winograd Schema example above shows syntactic heuristics alone are fundamentally insufficient for many real cases — "they" is syntactically equidistant from both "the city council" and "the demonstrators" in both versions of the sentence, and only semantic/world knowledge (understanding that fearing violence is something an authority worried about safety would do, while advocating violence is something protesters demanding change might be accused of) disambiguates correctly. Modern coreference systems use learned models (typically Transformer-based, scoring candidate antecedent-mention pairs using rich contextual embeddings, Lesson 22) trained on large annotated coreference datasets, which can pick up on exactly this kind of soft, world-knowledge-dependent signal that no hand-written rule set could feasibly enumerate.

### Why coreference resolution matters for downstream tasks

Coreference resolution is rarely an end goal on its own — it's a supporting step that makes other tasks work correctly on realistic, pronoun-heavy text:

- **Relation extraction** (Lesson 26): extracting "who did what to whom" requires knowing who "he" or "the company" actually refers to; without resolution, a relation extractor working sentence-by-sentence would completely miss any relation expressed via a pronoun rather than a full name.
- **Summarization** (Lesson 12): a summary combining information from multiple sentences needs to know when two different mentions ("the CEO," "she," "the executive") refer to the same person, to avoid either confusing repetition or an incorrect merge of two different people's information.
- **Question answering** (Lesson 13): a passage's answer to a question might be expressed via a pronoun reference to something stated several sentences earlier, requiring resolution before the actual answer span can be correctly identified and extracted.

See `code/coreference_demo.py` for a from-scratch rule-based pronoun resolver applying number/gender agreement and recency preference, tested on both straightforward cases and a Winograd-Schema-style ambiguous case that exposes the limits of syntax-only resolution.

## Exercises

1. Implement `simple_pronoun_resolver` with number/gender agreement and recency preference, and test it on 5 sentences with unambiguous pronoun references. Confirm it resolves each correctly.
2. Construct 3 sentences with two same-gender candidate antecedents for a single pronoun (e.g. "John talked to Peter. He seemed tired.") and check whether your resolver's "nearest candidate" heuristic picks the linguistically preferred one (usually the subject of the most recent clause).
3. Test your resolver on both versions of the Winograd Schema example above ("feared violence" vs "advocated violence") and confirm it cannot distinguish them — both will resolve identically, since nothing in the syntax alone differs between the two versions.
4. Research (via web search) what "mention detection" refers to as the first stage of a full coreference resolution pipeline (identifying WHICH spans are potential mentions at all, before resolving them into chains), and explain why this stage is itself a nontrivial subtask.

## Key Terms

| Term | What it actually means |
|---|---|
| Coreference resolution | Identifying which spans of text (pronouns, shortened references, full names) refer to the same real-world entity |
| Coreference chain | The full set of mentions in a text that all refer to the same entity |
| Antecedent | The earlier mention that a pronoun or later reference resolves back to |
| Hobbs algorithm | A classical rule-based coreference resolution algorithm that searches a sentence's parse tree in a defined order for the nearest matching antecedent |
| Winograd Schema | A sentence pair differing in one word, designed so pronoun resolution requires world knowledge rather than syntax alone |
