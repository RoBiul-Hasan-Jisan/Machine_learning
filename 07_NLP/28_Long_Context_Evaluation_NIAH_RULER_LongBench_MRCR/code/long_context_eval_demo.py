"""
A from-scratch Needle-in-a-Haystack (NIAH) test: varying context length
and needle position, testing a simple exact-match retrieval baseline
and producing the characteristic accuracy-by-position-and-length
results table, plus a simplified multi-needle RULER-style test.
"""

import re

import numpy as np


def make_haystack(n_words, seed=0):
    rng = np.random.default_rng(seed)
    filler_words = ["the", "forest", "was", "quiet", "and", "still", "birds",
                     "flew", "over", "trees", "while", "clouds", "drifted", "slowly",
                     "across", "sky", "wind", "blew", "gently", "leaves", "rustled"]
    return rng.choice(filler_words, size=n_words).tolist()


def build_niah_test(haystack_words, needle, position_fraction):
    insert_position = int(len(haystack_words) * position_fraction)
    needle_words = needle.split()
    test_words = haystack_words[:insert_position] + needle_words + haystack_words[insert_position:]
    return test_words


def simple_retrieval_baseline(test_words, query_pattern, context_limit=None):
    """A simple, deterministic 'retrieval' baseline: can the needle still
    be found via exact pattern match within the (possibly truncated)
    context? This validates the test harness itself, separate from the
    simulated 'effective attention' behavior modeled below."""
    text = " ".join(test_words[:context_limit] if context_limit else test_words)
    match = re.search(query_pattern, text)
    return match is not None


def simulate_lost_in_the_middle(position_fraction, context_length, base_rate=0.95, seed=None):
    """Simulate a realistic 'lost in the middle' accuracy curve: high near
    the start/end of context, dipping in the middle, with the dip getting
    WORSE as context length grows -- the well-documented empirical pattern
    this lesson describes. This is a SIMULATION standing in for actually
    querying a real long-context model, which this environment cannot do."""
    rng = np.random.default_rng(seed)
    edge_distance = min(position_fraction, 1 - position_fraction)  # 0 = at an edge, 0.5 = dead center
    length_factor = min(context_length / 8000, 1.0)  # longer context -> bigger middle dip
    dip = 0.5 * length_factor * (edge_distance / 0.5)
    accuracy_prob = base_rate - dip
    return rng.random() < accuracy_prob


def demo_niah_position_length_grid():
    context_lengths = [500, 2000, 8000]
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_trials = 30

    print("=== NIAH: exact-match retrieval accuracy (verifies the test harness works) ===\n")
    needle = "the special magic number is fortytwo"
    query_pattern = r"special magic number is fortytwo"

    for length in context_lengths:
        haystack = make_haystack(length)
        for pos in positions:
            test_words = build_niah_test(haystack, needle, pos)
            found = simple_retrieval_baseline(test_words, query_pattern)
            assert found, "the needle must always be findable via exact match in the full context"
    print("Confirmed: exact-match retrieval succeeds at every length/position combination")
    print("when given the FULL context -- this validates the test harness itself works")
    print("correctly before using it to probe a real model's EFFECTIVE attention.\n")

    print("=== Simulated 'lost in the middle' accuracy grid ===")
    print("(Simulates the well-documented empirical pattern from real long-context")
    print("evaluations -- not a literal query to a real language model.)\n")

    header = "position:  " + "  ".join(f"{p:>5.0%}" for p in positions)
    print(f"{'context len':>12s} | {header}")
    for length in context_lengths:
        row_scores = []
        for pos in positions:
            successes = sum(
                simulate_lost_in_the_middle(pos, length, seed=i) for i in range(n_trials)
            )
            row_scores.append(successes / n_trials)
        row_str = "  ".join(f"{s:>5.0%}" for s in row_scores)
        print(f"{length:>12d} | {row_str}")

    print("\nNote the dip in the MIDDLE column (50%) that gets more pronounced as context")
    print("length increases -- this is the characteristic 'lost in the middle' pattern")
    print("NIAH is specifically designed to surface.")


def demo_ruler_style_multi_needle():
    """A simplified RULER-style test: TWO needles must both be found and
    combined to answer the query correctly, harder than single-needle NIAH."""
    print("\n=== Simplified RULER-style multi-needle test ===\n")

    haystack = make_haystack(300, seed=1)
    needle1 = "the codename for the project is falcon"
    needle2 = "falcon project budget is two million dollars"

    test_words = haystack[:100] + needle1.split() + haystack[100:200] + needle2.split() + haystack[200:]
    text = " ".join(test_words)

    single_needle_found = "codename for the project is falcon" in text
    multi_hop_found = (
        "codename for the project is falcon" in text
        and "falcon project budget is two million dollars" in text
    )

    print(f"Single-needle check (just the codename): found = {single_needle_found}")
    print(f"Multi-hop check (codename AND budget, both needed to answer")
    print(f"  'what is the project's budget'): found = {multi_hop_found}")
    print("\nAnswering 'what is the project's budget' correctly requires connecting BOTH")
    print("needles (first identifying the codename, then finding that codename's")
    print("associated budget) -- a genuinely harder task than single-fact retrieval,")
    print("exactly the kind of multi-hop reasoning RULER adds on top of basic NIAH.")


if __name__ == "__main__":
    demo_niah_position_length_grid()
    demo_ruler_style_multi_needle()
