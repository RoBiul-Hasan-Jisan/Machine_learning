# 26. Relation Extraction & Knowledge Graph Construction

## Learning Objectives

- Frame relation extraction as classifying the relationship between two already-identified entities
- Implement a dependency-path-based relation extractor and a supervised classifier over entity-pair features
- Assemble extracted (subject, relation, object) triples into a knowledge graph

## The Problem

Lessons 06 (NER) and 25 (entity linking) get you from raw text to a set of grounded, disambiguated entity mentions. But entities alone don't capture the actual information in a sentence — "Apple acquired Beats" and "Apple was acquired by Beats" both mention the same two entities, in the same sentence structure superficially, but describe opposite relationships. Relation extraction identifies *how* two entities are related, turning unstructured text into structured (subject, relation, object) facts that can be stored, queried, and combined into a knowledge graph.

## The Concept

### Relation extraction as classification over entity pairs

Given a sentence and two entity mentions already identified within it (NER, Lesson 06), relation extraction classifies the relationship between them into one of a predefined set of relation types (or "no relation" if the entities happen to co-occur without any meaningful stated relationship):

```
Sentence: "Apple acquired Beats Electronics in 2014 for three billion dollars."
Entities: [Apple: ORG], [Beats Electronics: ORG]

Relation extraction: (Apple, ACQUIRED, Beats Electronics)
```

This is a sentence-pair-style classification task structurally similar to NLI (Lesson 21) — the label depends on the relationship between two specific things (here, two entity mentions within one sentence, rather than two full sentences), and the classifier needs to attend to exactly the right part of the sentence (the verb "acquired," and its direction — who's the subject, who's the object) to get the relation and its direction correct.

### Dependency-path features: use syntax directly

A classical, effective feature for relation extraction uses the dependency parse (Lesson 07) connecting the two entity mentions — the sequence of grammatical relations linking them tends to be highly informative about the relation type, often more informative than the raw words alone:

```
"Apple acquired Beats Electronics in 2014."

Dependency path from "Apple" to "Beats Electronics":
  Apple --[nsubj]--> acquired <--[dobj]-- Beats Electronics

This specific PATTERN (subject -> verb <- object, with verb "acquired")
is a strong, reusable signal for an ACQUIRED relation, regardless of
which specific company names fill the subject/object slots.
```

```python
def extract_dependency_path_features(sentence_parse, entity1_span, entity2_span):
    """A simplified dependency-path feature: the sequence of dependency
    relations connecting the two entities, plus the connecting verb."""
    path = find_shortest_dependency_path(sentence_parse, entity1_span, entity2_span)
    verb = next((tok for tok in path if tok["pos"] == "VERB"), None)
    return {
        "path_relations": [tok["dep"] for tok in path],
        "connecting_verb": verb["text"] if verb else None,
        "entity1_is_subject": path[0]["dep"] == "nsubj",
    }
```

The intuition: many different sentences expressing the same underlying relation type ("Company X bought Company Y," "Company X acquired Company Y," "Company Y was purchased by Company X") share a similar *dependency structure* even when the specific words and entity names differ completely — the parse structure abstracts away from surface wording toward the underlying grammatical relationship, which correlates strongly with the semantic relationship being expressed.

### A supervised relation classifier

In practice, dependency-path features are typically combined with other signals (the words between the two entities, the entity types themselves, word embeddings of the connecting context) and fed into a trained classifier — following the same general pattern as Lesson 21's NLI classifier, but between two entity spans within one sentence rather than two full sentences:

```python
import torch.nn as nn

class RelationClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, num_relations):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size * 2, num_relations)  # includes "no relation"

    def forward(self, sentence_with_entity_markers):
        # entities are often marked with special tokens in the input,
        # e.g. "[E1] Apple [/E1] acquired [E2] Beats [/E2]"
        encoded = self.encoder(sentence_with_entity_markers)
        e1_vec, e2_vec = extract_entity_representations(encoded)  # pull out the marked spans' representations
        combined = torch.cat([e1_vec, e2_vec], dim=-1)
        return self.classifier(combined)
```

Marking entity positions directly in the input (special tokens surrounding each entity span) is a common, effective trick: it tells the encoder exactly which two spans it needs to focus on relating, rather than relying on the model to somehow infer which two of potentially many mentioned entities are the relevant pair for this classification.

### Knowledge graph construction: from triples to a structured graph

Running relation extraction across many sentences (and, at scale, many documents) produces a large set of (subject, relation, object) triples — assembling these into a **knowledge graph** means treating entities as graph nodes and extracted relations as directed, labeled edges between them:

```
Triples extracted from a document collection:
  (Apple, ACQUIRED, Beats Electronics)
  (Beats Electronics, FOUNDED_BY, Dr. Dre)
  (Apple, HEADQUARTERED_IN, Cupertino)
  (Tim Cook, CEO_OF, Apple)

Resulting knowledge graph:

  Dr. Dre --[FOUNDED_BY]--> Beats Electronics <--[ACQUIRED]-- Apple --[HEADQUARTERED_IN]--> Cupertino
                                                                   ^
                                                          [CEO_OF] |
                                                               Tim Cook
```

This graph structure enables queries no single sentence could directly answer — "what companies has Apple acquired, and who founded them?" requires traversing *two* connected edges (`Apple -ACQUIRED-> ? -FOUNDED_BY-> ?`), combining facts extracted from potentially different sentences or even different documents entirely, which is exactly the value proposition of building a knowledge graph rather than leaving extracted facts as an unconnected list. Critically, this entire pipeline depends on entity linking (Lesson 25) having correctly resolved *which* specific entity each mention refers to — without it, "Apple" the company and any other same-named entity would incorrectly collapse into a single, incoherent graph node, corrupting the resulting graph with false connections.

### Where this connects to modern LLM-based systems

Knowledge graphs built this way are commonly used to *ground* generative systems (extending Lesson 17's RAG pattern): rather than retrieving unstructured text chunks (Lesson 23), a system can query a knowledge graph directly for precise, structured facts, then feed those facts into a generator — trading some of unstructured retrieval's flexibility for the precision and explicit reasoning-path transparency a graph structure provides, particularly valuable for multi-hop questions requiring combining several distinct facts.

See `code/relation_extraction_demo.py` for a from-scratch dependency-path-style relation extractor (using simplified pattern matching in place of a full dependency parser) trained on a small synthetic dataset, plus assembly of extracted triples into a queryable knowledge graph supporting simple multi-hop traversal.

## Exercises

1. Implement a simplified dependency-path feature extractor (using POS-tag-based patterns rather than a full parser) and use it to classify 10 hand-constructed sentences into relation types (e.g. ACQUIRED, FOUNDED_BY, LOCATED_IN).
2. Build a small knowledge graph (10-15 triples) from extracted relations, and implement a simple graph traversal function answering a 2-hop query (e.g. "what companies were founded by people who also founded another company").
3. Construct a case where two sentences express the same underlying relation with completely different surface wording ("X bought Y" vs "Y was acquired by X") and confirm your dependency-path-style features produce a similar representation for both, despite the different words.
4. Discuss, in your own words, why entity linking (Lesson 25) errors would silently corrupt a knowledge graph, using a specific example of two different real-world entities sharing a name.

## Key Terms

| Term | What it actually means |
|---|---|
| Relation extraction | Classifying the relationship between two identified entities in text into a predefined relation type |
| Dependency path | The sequence of grammatical dependency relations connecting two entities in a sentence's parse, used as a feature for relation extraction |
| Entity marking | Inserting special tokens around entity spans in the input, telling a model which two entities' relationship to classify |
| Knowledge graph | A structured graph where entities are nodes and extracted relations are directed, labeled edges between them |
| Multi-hop query | A query requiring traversal of more than one edge in a knowledge graph to answer, combining multiple extracted facts |
