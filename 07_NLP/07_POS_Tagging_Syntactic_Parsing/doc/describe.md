# 07. POS Tagging & Syntactic Parsing

## Learning Objectives

- Explain part-of-speech tagging as sequence labeling and implement a simple HMM-based tagger
- Build a dependency parse tree and distinguish it from a constituency parse
- Recognize why POS tags and parses feed directly into other tasks covered in this module

## The Problem

Knowing *what kind of word* each token is (noun, verb, adjective) and *how words relate grammatically* to each other (subject, object, modifier) is foundational information that many other NLP tasks depend on directly: lemmatization (Lesson 01) needs a POS tag to disambiguate "saw" the noun from "saw" the verb; NER (Lesson 06) benefits from knowing a capitalized word is being used as a proper noun; relation extraction (Lesson 26) needs to know which noun phrase is the grammatical subject of which verb. POS tagging and parsing are the classical tools for recovering this structure explicitly.

## The Concept

### Part-of-speech tagging: another sequence labeling task

Like NER (Lesson 06), POS tagging assigns one label per token — but the label set is a fixed grammatical category (noun, verb, adjective, determiner, preposition, ...) rather than an entity type, and unlike NER, essentially *every* token gets a non-trivial tag (there's no "O" equivalent):

```
Tokens: The  quick  brown  fox    jumps  over  the  lazy  dog
Tags:   DET  ADJ    ADJ    NOUN   VERB   ADP   DET  ADJ   NOUN
```

The same word can take different tags depending on context — "saw" is a `VERB` in "I saw the dog" but a `NOUN` in "pass me the saw" — which is exactly why POS tagging can't be done with a simple per-word lookup table and needs to consider surrounding context, the same reason NER needed sequence labeling rather than per-token classification in isolation.

### A simple statistical approach: Hidden Markov Models

A classical (pre-neural) approach models POS tagging as inferring a hidden sequence of tags from an observed sequence of words, using two learned probability tables:

```
Transition probabilities: P(tag_t | tag_{t-1})     "how likely is a VERB to follow a NOUN?"
Emission probabilities:   P(word_t | tag_t)         "how likely is 'dog' to be tagged NOUN?"

Both are estimated directly from a tagged training corpus by counting:

P(tag_t | tag_{t-1}) = count(tag_{t-1}, tag_t) / count(tag_{t-1})
P(word_t | tag_t)     = count(word_t, tag_t) / count(tag_t)
```

Given these two tables, the **Viterbi algorithm** efficiently finds the single most probable tag sequence for a new sentence — checking every possible tag sequence directly would be exponential in sentence length, but Viterbi uses dynamic programming to find the optimal sequence in time proportional to `sentence_length * num_tags^2`, reusing partial computations rather than recomputing from scratch for every candidate sequence.

```python
def viterbi(words, tags, transition_probs, emission_probs, initial_probs):
    T, N = len(words), len(tags)
    trellis = np.zeros((T, N))    # trellis[t][i] = best probability of any tag
                                   # sequence ending in tag i at position t
    backpointer = np.zeros((T, N), dtype=int)

    for i, tag in enumerate(tags):
        trellis[0, i] = initial_probs[tag] * emission_probs[tag].get(words[0], 1e-6)

    for t in range(1, T):
        for i, tag in enumerate(tags):
            probs = [trellis[t-1, j] * transition_probs[tags[j]][tag] for j in range(N)]
            best_prev = np.argmax(probs)
            trellis[t, i] = probs[best_prev] * emission_probs[tag].get(words[t], 1e-6)
            backpointer[t, i] = best_prev

    # Trace back through backpointer to recover the best tag sequence
    ...
```

Modern taggers largely replace this HMM approach with neural sequence models (an RNN or Transformer producing a tag distribution per token, following the same many-to-many pattern as NER), but the HMM framing remains a clear, tractable way to understand what "inferring a tag sequence from context" actually means computationally.

### Syntactic parsing: recovering sentence structure

POS tags describe individual words. Parsing describes how words *combine* into larger grammatical units and how those units relate to each other. Two common representations:

**Constituency parsing** groups words into nested phrases (noun phrases, verb phrases), producing a tree where each internal node is a phrase category:

```
                S
        ________|________
       NP                VP
    ___|___          ____|____
  DET     NOUN      VERB      NP
   |        |         |    ___|___
  "the"   "dog"     "chased"  DET  NOUN
                               |     |
                             "the"  "cat"
```

**Dependency parsing** instead draws direct grammatical relationships (subject, object, modifier) between individual word pairs, with one word as the "head" and another as its "dependent" — no intermediate phrase nodes at all:

```
        chased
       /   |    \
   dog(subj) cat(obj)
    |          |
   the        the
```

Dependency parses are more directly useful for many downstream tasks (relation extraction, Lesson 26, often works directly off dependency paths — "who did what to whom" maps naturally onto subject/verb/object dependency relations) and are the more common representation in modern NLP pipelines; constituency parsing remains important in linguistics and certain grammar-focused applications.

### Where this connects to the rest of the module

POS tags and dependency parses feed several later lessons directly: coreference resolution (Lesson 24) uses syntactic position to help decide what a pronoun refers to; relation extraction (Lesson 26) often walks the dependency path between two entities to determine their relationship; and question answering (Lesson 13) systems sometimes use parse structure to identify the grammatical role of the answer being sought (is the question asking for a subject, an object, a time, a location).

See `code/pos_parsing_demo.py` for a from-scratch HMM POS tagger trained on a small tagged corpus with Viterbi decoding, and a simple rule-based dependency parser illustrating how subject/object relations can be recovered from POS tags and word order.

## Exercises

1. Estimate transition and emission probability tables by hand (or with the code) from a small tagged corpus, and use them to tag a new sentence via Viterbi decoding.
2. Construct a sentence with a genuinely ambiguous word (e.g. "saw," "book," "run") in two different grammatical roles, and confirm your HMM tagger assigns different tags depending on context.
3. Draw both a constituency parse and a dependency parse for the same sentence by hand, and explain in your own words what information each representation makes explicit that the other doesn't.
4. For a sentence with a clear subject-verb-object structure, write a simple rule (using POS tags and word order) that identifies the subject and object without a full parser, and test it on 3 different sentences.

## Key Terms

| Term | What it actually means |
|---|---|
| Part-of-speech (POS) tagging | Assigning a grammatical category (noun, verb, adjective, etc.) to every token in a sentence |
| Hidden Markov Model (HMM) | A statistical model for sequence labeling using transition probabilities between labels and emission probabilities from labels to observed words |
| Viterbi algorithm | A dynamic programming algorithm that efficiently finds the most probable label sequence under an HMM, avoiding exponential brute-force search |
| Constituency parsing | Representing sentence structure as nested phrases (noun phrases, verb phrases) in a tree |
| Dependency parsing | Representing sentence structure as direct grammatical relationships (head/dependent) between individual words |
