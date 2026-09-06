# 01. Text Processing: Tokenization, Stemming, Lemmatization

## Learning Objectives

- Implement rule-based and subword-agnostic tokenization and explain why "just split on spaces" fails
- Implement a rule-based stemmer and understand the Porter algorithm's approach
- Distinguish stemming from lemmatization and know when each is appropriate

## The Problem

Every NLP task starts from raw text — a string of characters — and every downstream technique in this module (bag-of-words, embeddings, classification) needs that string broken into discrete units first. Getting this step wrong quietly degrades everything built on top of it: a bad tokenizer splits "don't" into nonsense pieces, an aggressive stemmer collapses unrelated words together, and a naive whitespace split treats "dog." and "dog" as entirely different tokens.

## The Concept

### Tokenization: splitting text into units

Naive whitespace splitting (`text.split()`) fails on punctuation ("dog." stays glued to its period), contractions ("don't" is one token but arguably means "do not"), and languages without spaces between words (Chinese, Japanese). A regex-based tokenizer improves on this by defining explicit rules for what counts as a token boundary:

```python
import re

def simple_tokenize(text):
    # Matches words (including internal apostrophes) or single punctuation marks
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text.lower())

simple_tokenize("Don't stop believing!")
# ['don't', 'stop', 'believing', '!']
```

This is still a simplification — real tokenizers handle hyphenation, numbers with decimals/commas, URLs, emoji, and abbreviations ("Dr.", "U.S.") with progressively more rules, or (as covered in Lesson 19) learn subword units directly from data rather than relying on hand-written rules at all. Word-level rule-based tokenization is the right first tool to understand, and is still used directly in classical NLP pipelines (bag-of-words, Lesson 02) even though modern deep learning models almost universally use the learned subword tokenizers from Lesson 19 instead.

### Stemming: crude, fast suffix stripping

Stemming reduces a word to a base form by applying a fixed set of suffix-stripping rules, without checking whether the result is a real word or understanding grammar. The **Porter stemmer** (1980) is the classic example — a sequence of ordered rules like "if a word ends in 'sses', replace with 'ss'":

```python
def simple_stem(word):
    for suffix in ["ational", "tional", "ing", "edly", "ed", "ing", "es", "s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word

simple_stem("running")     # "runn"  <- crude, not even a real word
simple_stem("relational")  # "relat"
```

Stemming is fast and language-simple but produces results that aren't necessarily real words ("runn", not "run") and can conflate unrelated words that happen to share a suffix pattern, or fail to unify words that are related but don't share a simple suffix pattern (e.g. "went" and "go").

### Lemmatization: grammatically correct base forms

Lemmatization reduces a word to its dictionary form (the "lemma"), using vocabulary and grammatical knowledge rather than blind suffix stripping — "went" correctly maps to "go", "better" to "good", "mice" to "mouse", none of which a suffix-stripping stemmer could ever produce. This requires either a lookup dictionary of known word forms, or a part-of-speech-aware set of rules (since the correct lemma of "saw" depends on whether it's a verb — lemma "see" — or a noun — lemma "saw", the tool), which is why real lemmatizers (e.g. spaCy's, or NLTK's WordNet-based one) are meaningfully more complex than a stemmer and typically depend on a part-of-speech tag (Lesson 07) as additional input.

```
Stemming:      "running" -> "runn"        (mechanical, not necessarily a word)
Lemmatization: "running" -> "run"         (a real word, the dictionary form)

Stemming:      "better" -> "better"       (no simple suffix rule applies)
Lemmatization: "better" -> "good"         (requires actual vocabulary knowledge)
```

### Choosing between them

| | Stemming | Lemmatization |
|---|---|---|
| Speed | Fast, purely mechanical | Slower, needs a dictionary/POS tags |
| Output | Not always a real word | Always a real dictionary word |
| Accuracy | Cruder, occasional wrong merges | More accurate, but more setup |
| Typical use | Search indexing, quick bag-of-words features where speed matters more than precision | Tasks where the exact grammatical base form matters (e.g. feeding into another rule-based system) |

In modern deep-learning NLP pipelines (from Lesson 03 onward), neither stemming nor lemmatization is typically applied before subword tokenization (Lesson 19) — learned embeddings and subword units capture morphological relationships (e.g. "run"/"running"/"runs" sharing a subword) without needing an explicit stemming step. Stemming and lemmatization remain most relevant for classical bag-of-words/TF-IDF pipelines (Lesson 02) and search/information-retrieval systems (Lesson 14), where reducing vocabulary size and matching morphological variants directly improves results.

See `code/text_processing_demo.py` for a from-scratch regex tokenizer, a simplified Porter-style stemmer compared against NLTK's full `PorterStemmer`, and a small rule-based lemmatizer compared against a dictionary lookup.

## Exercises

1. Extend `simple_tokenize` to correctly handle a sentence with a URL, an email address, and a hyphenated word, without breaking any of them apart incorrectly.
2. Run `simple_stem` and NLTK's `PorterStemmer` on the same list of 10 words. Identify at least 2 cases where they disagree and explain why.
3. Build a small lemma lookup dictionary (10-15 entries covering irregular verbs and plurals) and use it to lemmatize a sentence containing at least 3 irregular forms.
4. For a search engine indexing task and a formal grammar-checking task, decide which (stemming or lemmatization) is more appropriate for each, and justify your choice.

## Key Terms

| Term | What it actually means |
|---|---|
| Tokenization | Splitting raw text into discrete units (words, punctuation, subwords) for downstream processing |
| Stemming | Mechanically stripping suffixes from a word using fixed rules, without checking the result is a real word |
| Lemmatization | Reducing a word to its dictionary base form (lemma) using vocabulary and grammatical knowledge |
| Porter stemmer | A widely used rule-based English stemming algorithm defined by an ordered sequence of suffix-stripping rules |
