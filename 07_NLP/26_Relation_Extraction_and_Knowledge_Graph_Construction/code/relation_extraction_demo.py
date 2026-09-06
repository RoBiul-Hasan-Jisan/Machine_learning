"""
A simplified dependency-path-style relation extractor (using POS-tag
pattern matching in place of a full parser), tested on a small
synthetic dataset, plus assembly of extracted triples into a queryable
knowledge graph supporting multi-hop traversal.
"""

import re
from collections import defaultdict


RELATION_PATTERNS = [
    (r"([\w.\s]+?) was founded by ([\w.\s]+?)(?:\s+in\s|\.(?=\s|$)|$)", "FOUNDED_BY_REVERSE"),
    (r"([\w.\s]+?) acquired ([\w.\s]+?)(?:\s+for\s|\s+in\s|\.(?=\s|$)|$)", "ACQUIRED"),
    (r"([\w.\s]+?) founded ([\w.\s]+?)(?:\s+in\s|\.(?=\s|$)|$)", "FOUNDED"),
    (r"([\w.\s]+?) is headquartered in ([\w.\s]+?)(?:\.(?=\s|$)|$)", "HEADQUARTERED_IN"),
    (r"([\w.\s]+?) is (?:the )?CEO of ([\w.\s]+?)(?:\.(?=\s|$)|$)", "CEO_OF"),
]


def extract_relations(sentence):
    """Try patterns in priority order; use only the FIRST pattern that
    matches a given sentence, to avoid one sentence firing multiple
    overlapping/conflicting patterns (e.g. 'was founded by' also
    partially matching a generic 'founded' pattern)."""
    for pattern, relation in RELATION_PATTERNS:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            subj, obj = match.group(1).strip(), match.group(2).strip()
            if relation == "FOUNDED_BY_REVERSE":
                return [(obj, "FOUNDED", subj)]
            return [(subj, relation, obj)]
    return []


def demo_relation_extraction():
    sentences = [
        "Apple acquired Beats Electronics for three billion dollars.",
        "Beats Electronics was founded by Jimmy Iovine in 2006.",
        "Apple is headquartered in Cupertino.",
        "Tim Cook is the CEO of Apple.",
        "Instagram was founded by Kevin Systrom in 2010.",
        "Facebook acquired Instagram in 2012.",
        "Meta is headquartered in Menlo Park.",
        "Mark Zuckerberg is CEO of Meta.",
    ]

    print("=== Extracting (subject, relation, object) triples ===\n")
    all_triples = []
    for sentence in sentences:
        triples = extract_relations(sentence)
        for t in triples:
            print(f"  '{sentence}'")
            print(f"    -> {t}\n")
            all_triples.append(t)

    return all_triples


def build_knowledge_graph(triples):
    graph = defaultdict(list)
    for subj, rel, obj in triples:
        graph[subj].append((rel, obj))
    return graph


def query_one_hop(graph, entity, relation=None):
    results = graph.get(entity, [])
    if relation:
        return [obj for rel, obj in results if rel == relation]
    return results


def demo_knowledge_graph(triples):
    graph = build_knowledge_graph(triples)

    print("=== Knowledge graph queries ===\n")

    print("1-hop: What did Apple acquire?")
    print(f"  {query_one_hop(graph, 'Apple', 'ACQUIRED')}\n")

    print("1-hop: Where is Apple headquartered?")
    print(f"  {query_one_hop(graph, 'Apple', 'HEADQUARTERED_IN')}\n")

    print("2-hop: Who founded the companies that Apple acquired?")
    acquired_companies = query_one_hop(graph, "Apple", "ACQUIRED")
    founders = []
    for company in acquired_companies:
        for entity, edges in graph.items():
            for rel, obj in edges:
                if rel == "FOUNDED" and obj == company:
                    founders.append((company, entity))
    print(f"  {founders}")
    print("  (Combines TWO separately-extracted facts -- 'Apple acquired Beats' and")
    print("  'Jimmy Iovine founded Beats' -- into one answer neither sentence alone gives.)\n")

    print("2-hop: Who is the CEO of companies headquartered in Cupertino?")
    hq_companies = []
    for entity, edges in graph.items():
        for rel, obj in edges:
            if rel == "HEADQUARTERED_IN" and obj == "Cupertino":
                hq_companies.append(entity)
    ceos = []
    for company in hq_companies:
        for entity, edges in graph.items():
            for rel, obj in edges:
                if rel == "CEO_OF" and obj == company:
                    ceos.append((company, entity))
    print(f"  {ceos}")


if __name__ == "__main__":
    triples = demo_relation_extraction()
    demo_knowledge_graph(triples)
