"""
A complete from-scratch BPE trainer and encoder, run on a small
synthetic corpus, showing learned merges, the resulting vocabulary, and
successful encoding of a word never seen during training.
"""

from collections import Counter


def word_to_symbols(word):
    """Represent a word as a tuple of characters plus an end-of-word marker."""
    return tuple(list(word)) + ("</w>",)


def build_corpus(word_freqs):
    return {word_to_symbols(word): freq for word, freq in word_freqs.items()}


def get_pair_counts(corpus):
    pairs = Counter()
    for word, freq in corpus.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(pair, corpus):
    new_corpus = {}
    merged_token = pair[0] + pair[1]
    for word, freq in corpus.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus[tuple(new_word)] = freq
    return new_corpus


def train_bpe(word_freqs, n_merges):
    corpus = build_corpus(word_freqs)
    merges = []
    for step in range(n_merges):
        pairs = get_pair_counts(corpus)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        if pairs[best_pair] < 2:
            break  # stop merging pairs that only occur once -- not a meaningful pattern
        merges.append(best_pair)
        corpus = merge_pair(best_pair, corpus)
    return merges, corpus


def get_vocab_from_corpus(corpus):
    vocab = set()
    for word in corpus:
        vocab.update(word)
    return sorted(vocab)


def encode(word, merges):
    """Apply learned merges IN ORDER to tokenize a new word (possibly
    never seen during training)."""
    symbols = list(word_to_symbols(word))
    for pair in merges:
        i = 0
        new_symbols = []
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols


def demo_bpe():
    # A small corpus with clear morphological patterns: many words
    # sharing "-est", "-er", "-ing" suffixes, and a shared "low"/"new"/"wide" root pattern
    word_freqs = {
        "low": 5,
        "lower": 2,
        "lowest": 4,
        "newest": 6,
        "new": 3,
        "wide": 2,
        "widest": 3,
        "wider": 2,
        "running": 4,
        "runner": 2,
        "walking": 3,
        "walker": 2,
    }

    print("Training corpus (word: frequency):")
    for w, f in word_freqs.items():
        print(f"  {w}: {f}")

    merges, final_corpus = train_bpe(word_freqs, n_merges=25)

    print(f"\n=== Learned {len(merges)} merges (in order) ===")
    for i, (a, b) in enumerate(merges):
        print(f"  merge {i + 1:2d}: '{a}' + '{b}' -> '{a + b}'")

    vocab = get_vocab_from_corpus(final_corpus)
    print(f"\n=== Final vocabulary ({len(vocab)} tokens) ===")
    print(vocab)

    print("\n=== Encoding words from the training corpus ===")
    for word in ["lowest", "running", "widest"]:
        tokens = encode(word, merges)
        print(f"  '{word}' -> {tokens}")

    print("\n=== Encoding a word NEVER SEEN during training ===")
    novel_word = "newer"  # shares "new" and "er" patterns with trained words, but wasn't itself trained on
    tokens = encode(novel_word, merges)
    print(f"  '{novel_word}' -> {tokens}")
    print("\nEven though 'newer' never appeared in the training corpus, BPE successfully")
    print("decomposes it into known subword pieces (learned from 'new', 'lower'/'wider'")
    print("sharing the 'er' pattern) rather than falling back to a single opaque <unk>")
    print("token -- exactly the out-of-vocabulary problem subword tokenization solves.")

    print("\n=== A truly unusual word (mostly unseen characters/patterns) ===")
    unusual_word = "zzzquixotic"
    tokens = encode(unusual_word, merges)
    print(f"  '{unusual_word}' -> {tokens}")
    print("(Falls back toward individual characters where no learned merge applies --")
    print("still fully representable, just as a longer sequence of smaller pieces.)")


if __name__ == "__main__":
    demo_bpe()
