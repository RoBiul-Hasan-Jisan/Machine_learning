"""
A from-scratch regex tokenizer, a simplified Porter-style stemmer
compared against NLTK's full PorterStemmer, and a small rule-based
lemmatizer compared against a dictionary lookup. No downloads required.
"""

import re

from nltk.stem import PorterStemmer


def simple_tokenize(text):
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text.lower())


def simple_stem(word):
    for suffix in ["ational", "tional", "edly", "ing", "es", "ed", "s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


LEMMA_DICT = {
    "went": "go", "goes": "go", "going": "go",
    "better": "good", "best": "good",
    "mice": "mouse", "geese": "goose", "children": "child",
    "ran": "run", "running": "run", "runs": "run",
    "was": "be", "were": "be", "is": "be", "are": "be", "am": "be",
    "saw": "see", "seen": "see",
}


def lemmatize(word):
    return LEMMA_DICT.get(word, word)  # fall back to the word itself if not in dictionary


def demo_tokenization():
    sentences = [
        "Don't stop believing!",
        "Visit https://example.com or email test@example.com for info.",
        "State-of-the-art models need well-tokenized input.",
    ]
    for s in sentences:
        print(f"'{s}'\n  -> {simple_tokenize(s)}\n")


def demo_stemming_comparison():
    porter = PorterStemmer()
    words = ["running", "relational", "flies", "studies", "national", "connected", "argued", "generous"]

    print(f"{'word':15s} | {'our stemmer':15s} | {'NLTK PorterStemmer':20s}")
    disagreements = 0
    for w in words:
        ours = simple_stem(w)
        nltk_stem = porter.stem(w)
        match = "" if ours == nltk_stem else "  <-- disagree"
        if ours != nltk_stem:
            disagreements += 1
        print(f"{w:15s} | {ours:15s} | {nltk_stem:20s}{match}")

    print(f"\n{disagreements} disagreements out of {len(words)} words.")
    print("(Our simplified stemmer uses far fewer rules than the real Porter")
    print("algorithm, which has multiple ordered passes -- disagreements are expected.)\n")


def demo_lemmatization():
    sentence = "The children went home because the geese were loud and they saw better options"
    tokens = simple_tokenize(sentence)

    print("Stemming vs lemmatization on irregular forms:")
    print(f"{'token':12s} | {'stemmed':12s} | {'lemmatized':12s}")
    for t in tokens:
        if t in LEMMA_DICT:
            print(f"{t:12s} | {simple_stem(t):12s} | {lemmatize(t):12s}")

    print("\nNote how stemming does NOTHING useful for irregular forms like")
    print("'went'/'geese'/'saw'/'better' -- they don't share a simple suffix")
    print("pattern with their base form, so only dictionary-based lemmatization")
    print("recovers the correct root.")


if __name__ == "__main__":
    print("=== Tokenization ===")
    demo_tokenization()

    print("=== Stemming: ours vs NLTK's PorterStemmer ===")
    demo_stemming_comparison()

    print("=== Lemmatization vs stemming on irregular forms ===")
    demo_lemmatization()
