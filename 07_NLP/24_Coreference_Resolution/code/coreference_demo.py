"""
A from-scratch rule-based pronoun resolver applying number/gender
agreement and recency preference, tested on straightforward cases and
a Winograd-Schema-style ambiguous case exposing syntax-only limits.
"""

import re

MALE_NAMES = {"john", "peter", "michael", "david", "james"}
FEMALE_NAMES = {"sarah", "emma", "mary", "susan", "linda"}

MALE_PRONOUNS = {"he", "him", "his"}
FEMALE_PRONOUNS = {"she", "her", "hers"}
PLURAL_PRONOUNS = {"they", "them", "their"}


def tokenize_with_positions(text):
    return re.findall(r"\w+|[^\w\s]", text)


def get_candidates(tokens, upto_index):
    """Return proper-noun candidates (by simple capitalization heuristic)
    appearing before `upto_index`, most recent first."""
    candidates = []
    for i in range(upto_index - 1, -1, -1):
        tok = tokens[i]
        if tok[0].isupper() and tok.lower() not in {"the", "a", "an"}:
            candidates.append((i, tok))
    return candidates


def gender_of(name):
    name_lower = name.lower()
    if name_lower in MALE_NAMES:
        return "male"
    if name_lower in FEMALE_NAMES:
        return "female"
    return "unknown"


def pronoun_gender(pronoun):
    p = pronoun.lower()
    if p in MALE_PRONOUNS:
        return "male"
    if p in FEMALE_PRONOUNS:
        return "female"
    if p in PLURAL_PRONOUNS:
        return "plural"
    return "unknown"


def resolve_pronoun(tokens, pronoun_index):
    """Simple heuristic: nearest preceding candidate matching gender,
    preferring recency (Hobbs-style, simplified)."""
    pronoun = tokens[pronoun_index]
    target_gender = pronoun_gender(pronoun)
    if target_gender in (None, "unknown"):
        return None

    candidates = get_candidates(tokens, pronoun_index)
    for idx, name in candidates:
        if target_gender == "plural":
            continue  # simplified: this demo doesn't attempt plural-entity resolution
        if gender_of(name) == target_gender:
            return name
    return None


def demo_straightforward_cases():
    test_sentences = [
        "Sarah met John yesterday . She smiled .",
        "John called Sarah . He apologized .",
        "Michael and Emma talked . She laughed at his joke .",
    ]

    print("=== Straightforward pronoun resolution ===")
    for sentence in test_sentences:
        tokens = tokenize_with_positions(sentence)
        for i, tok in enumerate(tokens):
            if pronoun_gender(tok) in ("male", "female"):
                resolved = resolve_pronoun(tokens, i)
                print(f"  '{sentence}'")
                print(f"    '{tok}' (position {i}) -> resolved to: '{resolved}'")
        print()


def demo_ambiguous_same_gender():
    sentence = "John talked to Peter . He seemed tired ."
    tokens = tokenize_with_positions(sentence)
    print("=== Two same-gender candidates (genuinely ambiguous) ===")
    print(f"  '{sentence}'")
    for i, tok in enumerate(tokens):
        if pronoun_gender(tok) == "male":
            resolved = resolve_pronoun(tokens, i)
            print(f"    '{tok}' -> resolved to: '{resolved}'  (nearest-candidate heuristic, i.e. most RECENT match)")
    print("\nNote: 'He' is equally consistent with either John or Peter grammatically --")
    print("our simple resolver just picks the NEAREST match (Peter). A stronger system")
    print("would weigh grammatical role (subject vs object) and other cues, but even")
    print("those heuristics can be wrong without deeper semantic understanding.\n")


def demo_winograd_schema():
    """The classic Winograd Schema pattern: syntax alone cannot
    distinguish the correct antecedent; it requires world knowledge."""
    print("=== Winograd Schema: syntax-only resolution hits its limit ===")

    sentence_a = "The council refused the demonstrators a permit because they feared violence ."
    sentence_b = "The council refused the demonstrators a permit because they advocated violence ."

    for sentence in [sentence_a, sentence_b]:
        tokens = tokenize_with_positions(sentence)
        they_idx = tokens.index("they")
        # our simple resolver only handles gendered pronouns; "they" here is
        # syntactically ambiguous between "the council" and "the demonstrators" --
        # simple heuristics have no principled way to choose without semantics
        print(f"  '{sentence}'")
        print(f"    'they' could refer to 'the council' OR 'the demonstrators' -- ")
        print(f"    only the VERB choice ('feared' vs 'advocated') combined with world")
        print(f"    knowledge about who fears vs advocates violence disambiguates this,")
        print(f"    which is exactly what a syntax-only rule-based resolver cannot do.\n")


if __name__ == "__main__":
    demo_straightforward_cases()
    demo_ambiguous_same_gender()
    demo_winograd_schema()
