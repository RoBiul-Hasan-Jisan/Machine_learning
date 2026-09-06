"""
Fixed-size, sentence-based, and semantic chunking implemented on a
synthetic multi-topic document, plus a demonstration of how a naive
fixed-size chunk boundary can split a relevant answer across two
chunks, degrading retrieval.
"""

import re
from collections import Counter

import numpy as np


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def fixed_size_chunks(text, chunk_size=15, overlap=0):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def sentence_based_chunks(text, max_chunk_size=15):
    sentences = split_sentences(text)
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


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def simple_sentence_embedding(sentence, topic_keywords):
    """A crude sentence 'embedding' for the demo: a vector of how many
    words fall into each of a few known topic keyword sets. This stands
    in for a real trained sentence encoder (Lesson 22) -- just enough
    signal to make chunk boundaries a genuine function of TOPIC content,
    which a sparse word-overlap measure alone often fails to capture on
    short, topically-varied sentences."""
    tokens = set(tokenize(sentence))
    vec = np.array([len(tokens & keywords) for keywords in topic_keywords], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


TOPIC_KEYWORDS = [
    {"earnings", "revenue", "percent", "growth", "enterprise", "demand", "cloud",
     "services", "analyst", "expectations", "quarter", "ceo", "driver"},
    {"office", "austin", "relocate", "remotely", "location", "employees", "opening"},
    {"weather", "mild", "season", "farmers", "harvest", "region"},
]


def semantic_chunks(text, similarity_threshold=0.3, verbose=False):
    sentences = split_sentences(text)
    embeddings = [simple_sentence_embedding(s, TOPIC_KEYWORDS) for s in sentences]

    if verbose:
        print("Consecutive-sentence similarities (topic-vector cosine similarity):")
        for i in range(1, len(sentences)):
            sim = embeddings[i - 1] @ embeddings[i]
            print(f"  [{sim:.3f}] '{sentences[i-1][:35]}...' <-> '{sentences[i][:35]}...'")

    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = embeddings[i - 1] @ embeddings[i]
        if sim < similarity_threshold and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def demo_chunking_strategies():
    document = (
        "The company reported strong Q3 earnings, beating analyst expectations by 15 percent. "
        "Revenue grew 20 percent year over year, driven primarily by cloud services. "
        "The CEO cited strong enterprise demand as the key growth driver. "
        "Separately, the company announced a new office opening in Austin next year. "
        "Employees will have the option to relocate or work remotely from the new location. "
        "In other news, the weather in the region has been unusually mild this season. "
        "Local farmers report an early and promising harvest as a result of the mild conditions."
    )

    print("Document:")
    print(f"  '{document}'\n")

    print("=== Fixed-size chunking (15 tokens, no overlap) ===")
    for i, chunk in enumerate(fixed_size_chunks(document, chunk_size=15)):
        print(f"  chunk {i}: '{chunk}'")

    print("\n=== Sentence-based chunking (target 15 tokens) ===")
    for i, chunk in enumerate(sentence_based_chunks(document, max_chunk_size=15)):
        print(f"  chunk {i}: '{chunk}'")

    print("\n=== Semantic chunking (topic-vector similarity, threshold 0.3) ===")
    for i, chunk in enumerate(semantic_chunks(document, similarity_threshold=0.3, verbose=True)):
        print(f"  chunk {i}: '{chunk}'")

    print("\nNote how semantic chunking tends to group the 3 EARNINGS sentences,")
    print("the 2 OFFICE-OPENING sentences, and the 2 WEATHER sentences separately --")
    print("following topic shifts rather than a fixed token count.")


def demo_boundary_splits_answer():
    document = (
        "The quarterly report covers several areas of the business. "
        "The most important finding is that revenue increased by exactly "
        "twenty three percent compared to the previous quarter. "
        "This was primarily due to strong performance in the software division."
    )

    print("\n=== A fixed-size boundary splitting a specific answer ===")
    chunks = fixed_size_chunks(document, chunk_size=12, overlap=0)
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: '{c}'")

    query_tokens = set(tokenize("what was the percentage revenue increase"))
    print(f"\nQuery: 'what was the percentage revenue increase'")
    for i, c in enumerate(chunks):
        overlap = len(query_tokens & set(tokenize(c)))
        print(f"  chunk {i} keyword overlap with query: {overlap} words")

    print("\nThe actual number ('twenty three percent') is split across two chunks by")
    print("the fixed-size boundary -- neither chunk alone contains the complete")
    print("'revenue increased by twenty three percent' statement, which can hurt a")
    print("retriever's ability to confidently surface the RIGHT chunk for this query,")
    print("compared to sentence-based chunking, which would have kept it intact.")


if __name__ == "__main__":
    demo_chunking_strategies()
    demo_boundary_splits_answer()
