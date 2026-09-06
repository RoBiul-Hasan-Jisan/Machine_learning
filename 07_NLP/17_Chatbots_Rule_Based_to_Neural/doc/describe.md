# 17. Chatbots: Rule-Based to Neural

## Learning Objectives

- Build a pattern-matching rule-based chatbot and identify its structural limitations
- Build a retrieval-based chatbot using the information retrieval techniques from Lesson 14
- Explain how modern neural chatbots combine generation (Lessons 09-11, 16) with retrieval (Lesson 14) and dialogue-specific training

## The Problem

A chatbot needs to hold a conversation — understand what the user said, decide what to say back, and do this coherently across multiple turns, not just answer one isolated question (Lesson 13's QA task). This lesson traces the actual historical progression of approaches, since each generation of chatbot architecture was a direct, practical response to the previous generation's specific limitations — understanding *why* each approach was replaced is more instructive than treating "neural chatbots" as having appeared from nowhere.

## The Concept

### Rule-based chatbots: pattern matching and canned responses

The earliest chatbots (ELIZA, 1966, is the famous original example) work by matching the user's input against a fixed set of patterns and returning a corresponding pre-written (or simply-templated) response:

```python
import re

patterns = [
    (r"i feel (.*)", "why do you feel {0}?"),
    (r"i need (.*)", "why do you need {0}?"),
    (r"(hello|hi|hey)", "Hello! How can I help you today?"),
    (r".*", "Can you tell me more about that?"),   # fallback for anything unmatched
]

def eliza_respond(user_input):
    for pattern, response_template in patterns:
        match = re.match(pattern, user_input.lower())
        if match:
            return response_template.format(*match.groups())
    return "I'm not sure I understand."
```

This is genuinely simple to build and fully interpretable (you can read every rule and know exactly why any given response was produced) — and it's precisely why ELIZA felt surprisingly convincing to many users despite having zero actual understanding of language: reflecting the user's own words back as a question creates an illusion of listening. The structural limitation is equally clear: rule-based systems only handle inputs that match a pattern someone anticipated in advance, and scale terribly — covering a broad domain of conversation requires an enormous, ever-growing, hand-maintained rule set, with no ability to generalize to phrasings nobody thought to write a rule for.

### Retrieval-based chatbots: find the best matching response from a database

Instead of hand-written rules, a retrieval-based chatbot has a large database of example (question, response) or (context, response) pairs, and responds to new input by finding the *most similar* stored context and returning its associated response — directly applying Lesson 14's information retrieval machinery (TF-IDF/BM25 similarity, or embedding-based dense retrieval) to conversation instead of documents:

```python
def retrieval_chatbot_respond(user_input, conversation_database, vectorizer):
    query_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(query_vector, conversation_database_vectors)
    best_match_idx = similarities.argmax()
    return conversation_database[best_match_idx]["response"]
```

This generalizes better than hand-written rules (any input reasonably similar to *something* in the database gets a sensible response, not just inputs matching an anticipated pattern) and inherits retrieval's strengths and weaknesses directly from Lesson 14 — sparse retrieval misses paraphrases, dense retrieval handles them better but needs a good embedding model. The fundamental limitation: responses are always drawn *verbatim* from the existing database — a retrieval-based chatbot can never generate a genuinely novel response tailored to a specific input it's never seen anything similar to before, only recombine or select from what's already stored.

### Neural (generative) chatbots: generate a novel response

A generative chatbot frames the problem as sequence-to-sequence generation (Lessons 09-11): the conversation history is the "input sequence," and the response is generated token by token (Lesson 16's decoding strategies apply directly), rather than retrieved from a fixed database.

```
Conversation history (input):  "User: What's the weather like? Bot: It's sunny today. User: Should I bring a jacket?"
                                          |
                          [Encoder-decoder / Transformer, trained on conversational data]
                                          |
Generated response (output):   "Probably not, but a light layer might be nice for the evening."
```

This can genuinely generate novel, contextually appropriate responses never seen verbatim in training data — a real capability retrieval-based systems structurally lack — but inherits generation's own risks directly: Lesson 12's hallucination concern (generating fluent but factually ungrounded content) applies directly to chatbots, and Lesson 16's decoding failure modes (repetition loops, incoherence at high temperature) are exactly as relevant here as in any other generation task.

### Modern systems: retrieval-augmented generation

Production conversational AI systems today typically combine all three ideas rather than picking just one: a **retrieval** step (Lesson 14) finds relevant facts or context (e.g. from a knowledge base, or a set of documents specific to the deployment — a customer support system might retrieve relevant help articles), which is then fed *into* a **generative** model (Lessons 09-11, 16) as additional context, grounding the generated response in retrieved, verifiable information rather than relying purely on what the model happens to have memorized during training. This retrieval-augmented generation (RAG) pattern is exactly the retriever-reader architecture from Lesson 13, generalized from question answering to open-ended conversation, and is the direct subject of Lesson 23 (chunking strategies) and the LLM evaluation lessons (27-28) in the context of how well-grounded a system's outputs actually are.

```
User input + conversation history
         |
         v
   [Retriever] -- finds relevant facts/documents (Lesson 14)
         |
         v
   [Generator] -- produces a response CONDITIONED on both the conversation
         |          AND the retrieved context (Lessons 09-11, 16)
         v
    Response, grounded in retrieved information rather than purely
    the model's own (potentially outdated or hallucinated) internal knowledge
```

### Comparing the three generations

| | Rule-based | Retrieval-based | Neural (generative) | RAG (combined) |
|---|---|---|---|---|
| Can produce novel responses | No | No (verbatim reuse only) | Yes | Yes |
| Handles unanticipated input | Poorly | Reasonably (via similarity) | Well | Well |
| Risk of factual errors | Low (hand-curated) | Low (verbatim, curated database) | Real (hallucination) | Reduced (grounded in retrieval) |
| Effort to build/maintain | Manual rule-writing, doesn't scale | Requires a good conversation database | Requires training data + compute | Requires both a retrieval corpus and a generator |

See `code/chatbot_demo.py` for a complete rule-based chatbot, a retrieval-based chatbot using TF-IDF similarity over a small conversation database, and a minimal generative chatbot built on Lesson 16's character-level language model, compared side by side on the same set of test inputs.

## Exercises

1. Extend the rule-based `eliza_respond` pattern list with 5 new patterns of your own, and test it on 10 varied inputs, noting which ones fall through to the generic fallback response.
2. Build a retrieval-based chatbot over a small database of 15-20 (context, response) pairs, and test it on both an input closely matching a database entry and a paraphrased version of that same input, comparing how well it retrieves the right response in each case.
3. Train a small generative chatbot (reusing Lesson 16's `CharLM` machinery, framed on short conversation turns) and compare its responses on the same test inputs used for the rule-based and retrieval-based systems above.
4. Design a simple RAG-style pipeline: given a small set of "facts" and a user question, retrieve the most relevant fact (Lesson 14 style) and construct a prompt combining it with the question, explaining in words how this would ground a generative model's response.

## Key Terms

| Term | What it actually means |
|---|---|
| Rule-based chatbot | A chatbot that matches user input against hand-written patterns and returns corresponding pre-written responses |
| Retrieval-based chatbot | A chatbot that responds by finding and returning the most similar stored response from a conversation database |
| Generative (neural) chatbot | A chatbot that generates novel responses token by token using a sequence-to-sequence or language model |
| Retrieval-augmented generation (RAG) | Combining a retrieval step with a generative model, grounding generated output in retrieved, verifiable context |
