"""
A from-scratch HMM POS tagger with Viterbi decoding, trained on a small
tagged corpus, plus a simple rule-based dependency parser illustrating
subject/object extraction from POS tags and word order.
"""

from collections import defaultdict

import numpy as np


TAGGED_CORPUS = [
    [("the", "DET"), ("dog", "NOUN"), ("chased", "VERB"), ("the", "DET"), ("cat", "NOUN")],
    [("the", "DET"), ("cat", "NOUN"), ("chased", "VERB"), ("the", "DET"), ("mouse", "NOUN")],
    [("a", "DET"), ("quick", "ADJ"), ("dog", "NOUN"), ("runs", "VERB")],
    [("the", "DET"), ("lazy", "ADJ"), ("cat", "NOUN"), ("sleeps", "VERB")],
    [("i", "PRON"), ("saw", "VERB"), ("the", "DET"), ("dog", "NOUN")],
    [("pass", "VERB"), ("me", "PRON"), ("the", "DET"), ("saw", "NOUN")],
    [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB"), ("quickly", "ADV")],
    [("the", "DET"), ("mouse", "NOUN"), ("runs", "VERB"), ("quickly", "ADV")],
    [("a", "DET"), ("cat", "NOUN"), ("saw", "VERB"), ("a", "DET"), ("mouse", "NOUN")],
    [("the", "DET"), ("quick", "ADJ"), ("cat", "NOUN"), ("chased", "VERB"), ("a", "DET"), ("mouse", "NOUN")],
] * 4  # repeat for enough counts to estimate reasonable probabilities


def train_hmm(tagged_corpus):
    tags = sorted(set(tag for sent in tagged_corpus for _, tag in sent))
    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    initial_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    for sent in tagged_corpus:
        initial_counts[sent[0][1]] += 1
        for i, (word, tag) in enumerate(sent):
            emission_counts[tag][word] += 1
            tag_counts[tag] += 1
            if i > 0:
                prev_tag = sent[i - 1][1]
                transition_counts[prev_tag][tag] += 1

    n_sentences = len(tagged_corpus)
    initial_probs = {t: initial_counts[t] / n_sentences for t in tags}

    transition_probs = {}
    for t1 in tags:
        total = sum(transition_counts[t1].values())
        transition_probs[t1] = {
            t2: transition_counts[t1][t2] / total if total > 0 else 1e-6 for t2 in tags
        }

    emission_probs = {}
    for tag in tags:
        total = tag_counts[tag]
        emission_probs[tag] = defaultdict(
            lambda: 1e-6, {w: c / total for w, c in emission_counts[tag].items()}
        )

    return tags, initial_probs, transition_probs, emission_probs


def viterbi(words, tags, initial_probs, transition_probs, emission_probs):
    T, N = len(words), len(tags)
    trellis = np.zeros((T, N))
    backpointer = np.zeros((T, N), dtype=int)

    for i, tag in enumerate(tags):
        trellis[0, i] = initial_probs.get(tag, 1e-6) * emission_probs[tag][words[0]]

    for t in range(1, T):
        for i, tag in enumerate(tags):
            probs = [trellis[t - 1, j] * transition_probs[tags[j]][tag] for j in range(N)]
            best_prev = int(np.argmax(probs))
            trellis[t, i] = probs[best_prev] * emission_probs[tag][words[t]]
            backpointer[t, i] = best_prev

    best_last = int(np.argmax(trellis[T - 1]))
    path = [best_last]
    for t in range(T - 1, 0, -1):
        path.append(backpointer[t, path[-1]])
    path.reverse()

    return [tags[i] for i in path]


def demo_hmm_pos_tagging():
    tags, initial_probs, transition_probs, emission_probs = train_hmm(TAGGED_CORPUS)

    test_sentences = [
        ["the", "dog", "chased", "the", "mouse"],
        ["i", "saw", "the", "cat"],           # "saw" ambiguous -- should tag as VERB here
        ["pass", "me", "the", "saw"],         # "saw" ambiguous -- should tag as NOUN here
    ]

    for words in test_sentences:
        predicted_tags = viterbi(words, tags, initial_probs, transition_probs, emission_probs)
        print(f"{'word':10s} -> {'tag':6s}")
        for w, t in zip(words, predicted_tags):
            print(f"{w:10s} -> {t:6s}")
        print()

    print("Note: 'saw' is tagged differently in the two sentences above depending")
    print("on its context (VERB after a pronoun subject, NOUN after an imperative")
    print("verb + object pronoun) -- exactly the ambiguity a per-word lookup table")
    print("could never resolve, but sequence context (via transition probabilities)")
    print("can.\n")


def simple_dependency_parse(words, tags):
    """A crude rule-based dependency parser: finds the main VERB, then
    assigns the nearest preceding NOUN/PRON as subject and the nearest
    following NOUN as object. Real parsers use learned models over much
    richer features -- this illustrates the KIND of structure being recovered."""
    verb_idx = next((i for i, t in enumerate(tags) if t == "VERB"), None)
    if verb_idx is None:
        return None

    subject = None
    for i in range(verb_idx - 1, -1, -1):
        if tags[i] in ("NOUN", "PRON"):
            subject = words[i]
            break

    obj = None
    for i in range(verb_idx + 1, len(words)):
        if tags[i] in ("NOUN", "PRON"):
            obj = words[i]
            break

    return {"subject": subject, "verb": words[verb_idx], "object": obj}


def demo_dependency_parsing():
    tags, initial_probs, transition_probs, emission_probs = train_hmm(TAGGED_CORPUS)

    test_sentences = [
        ["the", "dog", "chased", "the", "cat"],
        ["a", "cat", "saw", "a", "mouse"],
        ["the", "quick", "cat", "chased", "a", "mouse"],
    ]

    print("=== Simple rule-based dependency extraction (subject-verb-object) ===")
    for words in test_sentences:
        predicted_tags = viterbi(words, tags, initial_probs, transition_probs, emission_probs)
        parse = simple_dependency_parse(words, predicted_tags)
        print(f"'{' '.join(words)}'")
        print(f"  POS tags: {list(zip(words, predicted_tags))}")
        print(f"  Parsed relation: subject='{parse['subject']}', verb='{parse['verb']}', object='{parse['object']}'\n")


if __name__ == "__main__":
    print("=== HMM POS tagging with Viterbi decoding ===")
    demo_hmm_pos_tagging()

    demo_dependency_parsing()
