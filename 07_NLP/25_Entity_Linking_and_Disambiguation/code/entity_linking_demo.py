"""
A complete candidate generation + context-based disambiguation
pipeline over a small synthetic knowledge base with genuinely
ambiguous entity names.
"""

KNOWLEDGE_BASE = {
    "Q_PARIS_FR": {
        "names": ["Paris"],
        "type": "LOCATION",
        "description": "capital city of france olympics eiffel tower seine river european",
    },
    "Q_PARIS_TX": {
        "names": ["Paris"],
        "type": "LOCATION",
        "description": "small city in texas county seat united states lamar county",
    },
    "Q_PARIS_HILTON": {
        "names": ["Paris", "Paris Hilton"],
        "type": "PERSON",
        "description": "socialite television personality businesswoman social media influencer",
    },
    "Q_AMAZON_RIVER": {
        "names": ["Amazon"],
        "type": "LOCATION",
        "description": "river south america rainforest brazil largest river by volume",
    },
    "Q_AMAZON_COMPANY": {
        "names": ["Amazon"],
        "type": "ORGANIZATION",
        "description": "online retailer technology company cloud computing ecommerce founded",
    },
}

# A simple popularity prior: default candidate per mention string when
# context alone is too weak to disambiguate confidently.
POPULARITY_PRIOR = {
    "Paris": "Q_PARIS_FR",
    "Amazon": "Q_AMAZON_COMPANY",
}


def generate_candidates(mention, knowledge_base):
    candidates = []
    for entity_id, info in knowledge_base.items():
        if any(mention.lower() == name.lower() for name in info["names"]):
            candidates.append(entity_id)
    return candidates


def disambiguate(mention_context, candidates, knowledge_base, prior=None, min_score=1):
    context_words = set(mention_context.lower().split())

    scores = {}
    for entity_id in candidates:
        description_words = set(knowledge_base[entity_id]["description"].lower().split())
        scores[entity_id] = len(context_words & description_words)

    best_candidate = max(scores, key=scores.get)
    best_score = scores[best_candidate]

    if best_score < min_score and prior is not None:
        return prior, scores, "fell back to popularity prior (weak context signal)"

    return best_candidate, scores, "resolved via context overlap"


def demo_entity_linking():
    test_cases = [
        ("Paris", "Paris hosted the summer olympics near the eiffel tower this year."),
        ("Paris", "Paris is a small county seat in texas with a population under thirty thousand."),
        ("Paris", "Paris posted a new video about her latest business venture on social media."),
        ("Amazon", "The amazon river flows through the rainforest in brazil."),
        ("Amazon", "Amazon announced new cloud computing services for retailer customers."),
    ]

    print("=== Entity linking with context-based disambiguation ===\n")
    for mention, context in test_cases:
        candidates = generate_candidates(mention, KNOWLEDGE_BASE)
        prior = POPULARITY_PRIOR.get(mention)
        resolved, scores, method = disambiguate(context, candidates, KNOWLEDGE_BASE, prior=prior)

        print(f"Mention: '{mention}'")
        print(f"Context: '{context}'")
        print(f"Candidates generated: {candidates}")
        print(f"Context-overlap scores: {scores}")
        print(f"Resolved to: {resolved}  ({KNOWLEDGE_BASE[resolved]['type']})  [{method}]\n")

    print("=== When context is too weak: popularity prior fallback ===")
    mention, context = "Paris", "I saw Paris yesterday."
    candidates = generate_candidates(mention, KNOWLEDGE_BASE)
    prior = POPULARITY_PRIOR.get(mention)
    resolved, scores, method = disambiguate(context, candidates, KNOWLEDGE_BASE, prior=prior)
    print(f"Mention: '{mention}'   Context: '{context}'")
    print(f"Context-overlap scores: {scores}")
    print(f"Resolved to: {resolved}  [{method}]")
    print("\nWith no real disambiguating context, the popularity prior (Paris, France --")
    print("the most commonly referenced entity for this name) provides a sensible")
    print("default. But note this is exactly the mechanism that would silently and")
    print("incorrectly override a correct link to a legitimate but rarer entity, if the")
    print("actual context word overlap score doesn't clear the confidence threshold.")


if __name__ == "__main__":
    demo_entity_linking()
