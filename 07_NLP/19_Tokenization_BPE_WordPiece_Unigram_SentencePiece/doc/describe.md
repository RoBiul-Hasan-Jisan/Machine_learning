# 19. Tokenization: BPE, WordPiece, Unigram, SentencePiece

## Learning Objectives

- Explain why subword tokenization replaced word-level tokenization for modern NLP models
- Implement Byte-Pair Encoding (BPE) from scratch, including both training and encoding
- Compare BPE, WordPiece, and Unigram's differing merge/selection criteria, and know what SentencePiece adds

## The Problem

Lesson 01's word-level tokenization has two problems that become severe at the scale of modern NLP models. First, a fixed word-level vocabulary has a hard out-of-vocabulary problem — any word not seen during vocabulary construction gets mapped to a generic `<unk>` token, discarding all information about what that word actually was, however common a similar/related word might have been. Second, word-level vocabularies for morphologically rich languages, or simply covering many languages at once (Lesson 18), can become enormous, since every inflected form ("run," "runs," "running," "runner") needs its own separate vocabulary entry despite sharing an obvious root. Subword tokenization solves both problems by learning a vocabulary of *pieces* of words, smaller than whole words but generally larger than single characters, directly from data.

## The Concept

### The core idea: learn a vocabulary of frequent subword pieces

Instead of a fixed list of whole words, subword tokenization builds a vocabulary that includes whole common words *and* frequently-occurring pieces of less common words, so that any input — even a word never seen during training — can always be represented as *some* sequence of vocabulary pieces, never falling back to a total information-losing `<unk>`.

```
Word-level vocabulary: "running" is either in the vocabulary or it isn't -- no middle ground

Subword vocabulary: "running" might not be a single token, but decomposes into
  known pieces: ["run", "ning"]  or  ["r", "unn", "ing"]  (depending on the specific
  algorithm and what pieces it learned to be common)

An entirely novel word can ALWAYS be decomposed, in the worst case down to
individual characters/bytes, which are guaranteed to be in the vocabulary.
```

### Byte-Pair Encoding (BPE): iteratively merge the most frequent pair

BPE (adapted for tokenization by Sennrich et al., 2015, from a much older data-compression algorithm) builds its vocabulary bottom-up: start with individual characters as the initial vocabulary, then repeatedly find the *most frequent adjacent pair* of tokens in the training corpus and merge them into a single new token, growing the vocabulary one merge at a time until reaching a target vocabulary size.

```
Training corpus (simplified, as character sequences with word-boundary markers):
  "l o w </w>" x5, "l o w e r </w>" x2, "n e w e s t </w>" x6, "w i d e s t </w>" x3

Step 1: find the most frequent ADJACENT PAIR across the whole corpus -> e.g. ("e", "s")
        appears in "newest" and "widest" -> merge into a new token "es"

Step 2: recount pairs with "es" now treated as one unit -> next most frequent pair
        might be ("es", "t") -> merge into "est"

... repeat for a fixed number of merges (this IS the vocabulary size hyperparameter)
```

```python
from collections import Counter

def get_pair_counts(corpus):
    """corpus: dict of {tuple-of-symbols: frequency}"""
    pairs = Counter()
    for word, freq in corpus.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs

def merge_pair(pair, corpus):
    new_corpus = {}
    for word, freq in corpus.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus[tuple(new_word)] = freq
    return new_corpus

def train_bpe(corpus, n_merges):
    merges = []
    for _ in range(n_merges):
        pairs = get_pair_counts(corpus)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        corpus = merge_pair(best_pair, corpus)
    return merges
```

Once trained, encoding a new word applies the learned merges *in the same order they were learned*, greedily merging matching adjacent pairs at each step — this is why BPE's merge list, not just its final vocabulary, needs to be saved and reused at encoding time.

### WordPiece: merge by likelihood gain, not raw frequency

WordPiece (used by BERT) follows the same iterative-merge structure as BPE, but changes the criterion for *which* pair to merge at each step: instead of picking the most frequent adjacent pair, it picks the pair whose merge most increases the *likelihood* of the training corpus under a simple language model — roughly, "does merging this pair capture a genuinely useful, predictive unit, not just a pair that happens to co-occur often." In practice this produces broadly similar-looking vocabularies to BPE, with some differences in which specific merges get prioritized.

### Unigram: start big, prune down

Unigram (Kudo, 2018, used by SentencePiece and T5) works in the *opposite direction* from BPE/WordPiece: rather than starting small and merging upward, it starts with a large candidate vocabulary (e.g. all substrings up to some length) and iteratively *removes* the pieces that contribute least to the corpus's likelihood under a unigram language model, until reaching the target vocabulary size. This "start big, prune down" approach also naturally supports a genuinely probabilistic tokenization — rather than one single deterministic tokenization for a given word, Unigram can assign probabilities to multiple valid ways of splitting the same word, which can be exploited for train-time regularization (deliberately using a slightly different, valid tokenization on different training passes, similar in spirit to Lesson 18-adjacent techniques for making a model more robust).

### SentencePiece: tokenization without pre-tokenization

BPE, WordPiece, and Unigram as described above all assume text is already split into words (Lesson 01's tokenizer), and only then decide how to further split those words into subword pieces. **SentencePiece** (Kudo & Richardson, 2018) removes even that assumption: it treats the input as a raw stream of characters (or bytes) with no prior whitespace-based word segmentation at all, treating spaces themselves as just another character to potentially be part of a token. This makes SentencePiece naturally language-agnostic — it works identically whether or not the language uses whitespace to separate words (directly addressing Lesson 18's word-segmentation problem for languages like Chinese and Japanese), since it never assumed whitespace-based word boundaries in the first place. SentencePiece is a *framework* that can run either BPE-style merging or Unigram-style pruning underneath — the "SentencePiece" name refers to the whitespace-agnostic preprocessing approach, not a distinct third algorithm competing with BPE and Unigram.

### Comparing the four

| | BPE | WordPiece | Unigram | SentencePiece |
|---|---|---|---|---|
| Direction | Bottom-up (merge) | Bottom-up (merge) | Top-down (prune) | Either (a framework, not an algorithm) |
| Merge/prune criterion | Most frequent pair | Greatest likelihood gain | Least likelihood loss when removed | (depends on underlying algorithm chosen) |
| Assumes pre-tokenized words | Yes (typically) | Yes (typically) | Yes (typically) | No — works on raw text directly |
| Notable users | GPT-family models | BERT | T5, ALBERT | Used as the framework underlying many multilingual models |

See `code/bpe_demo.py` for a complete from-scratch BPE trainer and encoder, run on a small synthetic corpus, showing the learned merges, the resulting vocabulary, and — critically — successful encoding of a word never seen during training, decomposed into known subword pieces rather than falling back to `<unk>`.

## Exercises

1. Implement `train_bpe` and run it on a small corpus with clear morphological patterns (e.g. many words sharing "-ing," "-ed," "-est" suffixes). Inspect the learned merges and confirm common suffixes get merged into single tokens.
2. Implement an `encode` function that applies a trained BPE model's merges (in order) to tokenize a new word, and confirm it can tokenize a word never seen during training by falling back to smaller known pieces.
3. Compare the vocabulary produced by 50 BPE merges vs 200 BPE merges on the same corpus, and describe how vocabulary size trades off against how finely words get split.
4. Research (via web search) the actual vocabulary sizes used by GPT and BERT tokenizers, and discuss why a larger subword vocabulary reduces the average number of tokens per word, and what tradeoff that creates against total vocabulary/embedding table size.

## Key Terms

| Term | What it actually means |
|---|---|
| Subword tokenization | Splitting text into pieces smaller than whole words but generally larger than individual characters, learned from a training corpus |
| Byte-Pair Encoding (BPE) | A subword tokenization algorithm that iteratively merges the most frequent adjacent symbol pair to build its vocabulary |
| WordPiece | A subword tokenization algorithm similar to BPE, but selecting merges by likelihood gain rather than raw frequency |
| Unigram | A subword tokenization algorithm that starts with a large candidate vocabulary and iteratively prunes low-value pieces |
| SentencePiece | A tokenization framework that operates on raw, unsegmented text, making subword tokenization language-agnostic regarding whitespace |
