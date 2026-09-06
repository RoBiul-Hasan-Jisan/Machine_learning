# 23. Chunking Strategies for RAG

## Learning Objectives

- Explain why long documents must be split into chunks before embedding and retrieval
- Implement fixed-size, sentence-based, and semantic chunking, and compare their tradeoffs
- Understand chunk overlap and metadata attachment as practical mitigations for chunking's information-loss risk

## The Problem

Retrieval-augmented generation (RAG, introduced conceptually in Lesson 17) needs to retrieve relevant *pieces* of documents to feed into a generator, but real documents are often far too long to embed and retrieve as a single unit — a 50-page report can't be usefully represented by one sentence embedding (Lesson 22), which would have to compress the entire document's content into one fixed-size vector, exactly Lesson 09's fixed-context bottleneck reappearing at the document-retrieval level. Chunking — splitting documents into smaller pieces before embedding and indexing them — is the practical fix, and *how* you chunk turns out to meaningfully affect retrieval quality, more than most RAG tutorials initially let on.

## The Concept

### Why chunking is necessary, and why it's not trivial

Splitting a long document into pieces solves the "one embedding can't represent 50 pages" problem, but introduces a new one: **a chunk boundary can cut through information that belongs together.** A sentence's meaning can depend on the sentence before it; a table's meaning depends on its caption; an answer to a question might span two adjacent paragraphs that end up in different chunks purely due to an arbitrary length cutoff. Poor chunking directly degrades retrieval quality (Lesson 14) — a chunk boundary landing in the wrong place can mean the actual answer to a query is split across two chunks, neither of which alone looks maximally relevant to the query, hurting the retriever's ability to surface it at all.

### Fixed-size chunking: simple, but boundary-blind

The simplest approach splits text into chunks of a fixed token/character count, optionally with overlap between consecutive chunks:

```python
def fixed_size_chunks(text, chunk_size=200, overlap=50):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap    # overlap re-includes the last `overlap` tokens in the next chunk
    return chunks
```

This is simple and predictable, but completely ignores document structure — a fixed-size cutoff can split a sentence, or even a word, right down the middle, with no regard for where a natural, meaning-preserving boundary would actually be. **Overlap** is a direct, partial mitigation: by re-including the last `overlap` tokens of one chunk at the start of the next, information near a chunk boundary has a chance to appear complete in at least one of the two neighboring chunks, even if it gets split across the boundary in the other.

### Sentence-based chunking: respect natural language boundaries

Rather than cutting at an arbitrary token count, sentence-based chunking (using Lesson 12's `split_sentences`) groups whole sentences together up to a target chunk size, never splitting a sentence itself:

```python
def sentence_based_chunks(text, max_chunk_size=200):
    sentences = split_sentences(text)   # from Lesson 12
    chunks, current_chunk, current_size = [], [], 0
    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_size = [], 0
        current_chunk.append(sentence)
        current_size += sentence_size
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

This guarantees every chunk contains only whole sentences — a real, meaningful improvement over fixed-size chunking's arbitrary mid-sentence cuts — but still doesn't guarantee chunks align with *topical* boundaries: a chunk can still end partway through a coherent multi-sentence argument or paragraph, just not partway through one specific sentence.

### Semantic chunking: split where meaning actually shifts

Semantic chunking goes further, using sentence embeddings (Lesson 22) to detect where the *topic* actually changes, rather than relying on any fixed size at all: compute the embedding similarity between consecutive sentences, and place a chunk boundary wherever similarity drops sharply — a strong signal that the discussion has shifted to something new.

```python
def semantic_chunks(text, sentence_encoder, similarity_threshold=0.5):
    sentences = split_sentences(text)
    embeddings = [sentence_encoder(s) for s in sentences]   # Lesson 22's sentence encoder

    chunks, current_chunk = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

This produces chunks that are more likely to be genuinely topically coherent internally, at the cost of variable, unpredictable chunk sizes (a topic that's discussed briefly produces a small chunk; a topic discussed at length produces a large one) and the added computational cost of embedding every sentence just to find boundaries, before any actual retrieval-time embedding happens.

### Metadata attachment: give chunks context beyond their own text

Since a chunk is, by design, a fragment stripped of its surrounding document, attaching **metadata** — the source document's title, the section heading the chunk came from, its position in the document — lets a RAG system recover context a chunk's raw text alone can't provide, and lets a generator (or a human reviewing retrieved results) understand where a chunk came from, not just what it says:

```python
chunk_with_metadata = {
    "text": chunk_text,
    "source_document": "Q3_Financial_Report.pdf",
    "section": "Revenue Breakdown",
    "chunk_index": 4,
}
```

This is especially important for chunks that read ambiguously in isolation — a chunk saying "revenue grew 20% this quarter" is much more useful when its metadata clarifies which company's report it's from and which quarter is meant, information the sentence itself might have relied on surrounding context (now stripped away by chunking) to convey.

### Comparing strategies

| Strategy | Respects sentence boundaries | Respects topic boundaries | Predictable chunk size | Extra compute cost |
|---|---|---|---|---|
| Fixed-size | No | No | Yes | None |
| Fixed-size + overlap | No (but mitigated) | No | Yes (with redundancy) | None |
| Sentence-based | Yes | No | Roughly | None |
| Semantic | Yes | Yes | No | Embedding every sentence |

There's no universally "correct" chunk size or strategy — it's a genuine, task-specific tuning decision (chunk size interacts with the embedding model's effective context length, the generator's context window, and the nature of the documents themselves), typically settled empirically by evaluating retrieval quality (Lesson 27's evaluation techniques apply directly here) across a few candidate strategies on a representative set of real queries, rather than assumed in advance.

See `code/chunking_demo.py` for complete implementations of fixed-size, sentence-based, and semantic chunking on a synthetic multi-topic document, plus a demonstration of how a naive fixed-size chunk boundary can split a relevant answer across two chunks, degrading retrieval — directly connecting back to Lesson 14's retrieval quality concerns.

## Exercises

1. Implement `fixed_size_chunks` with and without overlap on a synthetic document, and identify a specific case where a fixed-size (no-overlap) boundary splits a sentence containing important information.
2. Implement `sentence_based_chunks` on the same document and confirm no chunk ever contains a partial sentence.
3. Implement the semantic chunking approach (using a simple sentence similarity function, real or synthetic) and compare its chunk boundaries against sentence-based chunking's on a document with 2-3 clearly distinct topics.
4. Construct a query whose answer spans two adjacent sentences that a fixed-size chunker splits into different chunks, and confirm that neither resulting chunk alone retrieves well for that query using Lesson 14's retrieval methods, while a single chunk containing both sentences would.

## Key Terms

| Term | What it actually means |
|---|---|
| Chunking | Splitting a long document into smaller pieces before embedding and indexing for retrieval |
| Fixed-size chunking | Splitting text into chunks of a fixed token/character count, regardless of sentence or topic boundaries |
| Chunk overlap | Re-including the end of one chunk at the start of the next, mitigating information loss at chunk boundaries |
| Sentence-based chunking | Grouping whole sentences into chunks up to a target size, never splitting a sentence across chunks |
| Semantic chunking | Placing chunk boundaries where sentence embedding similarity drops, aiming for topically coherent chunks |
| Metadata attachment | Storing source/context information alongside a chunk's text, to recover context lost when the chunk was extracted from its document |
