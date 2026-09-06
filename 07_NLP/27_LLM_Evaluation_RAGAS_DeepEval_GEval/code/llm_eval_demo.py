"""
From-scratch implementations of RAGAS-style faithfulness scoring
(word-overlap entailment proxy), answer relevancy (embedding
similarity proxy), and a simulated LLM-as-judge scoring function, run
on RAG-style example outputs with varying quality.
"""

import re

import numpy as np


STOPWORDS = {"the", "a", "an", "is", "was", "were", "of", "to", "in", "on", "for",
             "and", "or", "s", "this", "that", "it", "at", "by", "with"}


def tokenize(text):
    words = re.findall(r"\w+", text.lower())
    return set(w for w in words if w not in STOPWORDS)


def split_into_claims(answer):
    """Crude claim splitter: split on conjunctions/commas as a rough
    proxy for decomposing an answer into individual assertions."""
    parts = re.split(r",\s*(?:and\s+)?|\.\s+|\s+and\s+", answer)
    return [p.strip() for p in parts if p.strip()]


def entailment_proxy(premise, hypothesis, threshold=0.5):
    """A simple word-overlap-based entailment proxy standing in for a
    real trained NLI model (Lesson 21): a claim is considered entailed
    if most of its content words appear in the premise/context."""
    premise_words = tokenize(premise)
    hypothesis_words = tokenize(hypothesis)
    if not hypothesis_words:
        return False
    overlap = len(hypothesis_words & premise_words) / len(hypothesis_words)
    return overlap >= threshold


def faithfulness_score(answer, context):
    claims = split_into_claims(answer)
    if not claims:
        return 0.0, []
    results = [(claim, entailment_proxy(context, claim)) for claim in claims]
    entailed_count = sum(1 for _, entailed in results if entailed)
    return entailed_count / len(claims), results


def answer_relevancy_score(answer, question):
    """A simplified relevancy proxy: word-overlap (Jaccard) similarity
    between the answer and the question's CONTENT words, standing in for
    a real embedding-similarity-based check (Lesson 22). A real system
    also generates candidate questions FROM the answer and compares those
    to the original question, which better captures paraphrase-level
    relevance than this direct-overlap shortcut."""
    answer_words = tokenize(answer)
    question_words = tokenize(question)
    if not question_words or not answer_words:
        return 0.0
    intersection = len(answer_words & question_words)
    union = len(answer_words | question_words)
    return intersection / union


def simulated_llm_judge(answer, context, question, rubric="faithfulness"):
    """A simplified stand-in for an LLM-as-judge call: combines the
    faithfulness and relevancy proxies into a single 1-5 style score,
    illustrating the KIND of aggregation a real LLM judge might produce
    -- not a literal simulation of an LLM's reasoning."""
    faith_score, _ = faithfulness_score(answer, context)
    relevancy = answer_relevancy_score(answer, question)

    if rubric == "faithfulness":
        return round(1 + faith_score * 4)  # map [0,1] -> [1,5]
    elif rubric == "relevancy":
        return round(1 + max(0, relevancy) * 4)
    return None


def demo_ragas_metrics():
    examples = [
        {
            "question": "What was the company's Q3 revenue?",
            "context": "The company reported Q3 revenue of 50 million dollars.",
            "answer": "The company's Q3 revenue was 50 million dollars.",
            "note": "Faithful and relevant",
        },
        {
            "question": "What was the company's Q3 revenue?",
            "context": "The company reported Q3 revenue of 50 million dollars.",
            "answer": "The company's Q3 revenue was 50 million dollars, driven by strong overseas sales.",
            "note": "Contains an UNSUPPORTED claim (overseas sales never mentioned in context)",
        },
        {
            "question": "What was the company's Q3 revenue?",
            "context": "The company reported Q3 revenue of 50 million dollars.",
            "answer": "The company plans to open a new office in Austin next year.",
            "note": "Off-topic: does not answer the question at all",
        },
    ]

    print("=== RAGAS-style metrics on example RAG outputs ===\n")
    for ex in examples:
        faith, claim_results = faithfulness_score(ex["answer"], ex["context"])
        relevancy = answer_relevancy_score(ex["answer"], ex["question"])

        print(f"Question: '{ex['question']}'")
        print(f"Context:  '{ex['context']}'")
        print(f"Answer:   '{ex['answer']}'")
        print(f"Note:     {ex['note']}")
        print(f"  Faithfulness score: {faith:.2f}")
        for claim, entailed in claim_results:
            print(f"    claim: '{claim}'  entailed={entailed}")
        print(f"  Answer relevancy score: {relevancy:.3f}\n")


def demo_llm_judge():
    print("=== Simulated LLM-as-judge scoring ===\n")
    context = "The company reported Q3 revenue of 50 million dollars."
    question = "What was the company's Q3 revenue?"

    candidates = [
        "The company's Q3 revenue was 50 million dollars.",
        "The company's Q3 revenue was 50 million dollars, driven by strong overseas sales.",
        "The company plans to open a new office in Austin next year.",
    ]

    for answer in candidates:
        faith_judge_score = simulated_llm_judge(answer, context, question, rubric="faithfulness")
        relevancy_judge_score = simulated_llm_judge(answer, context, question, rubric="relevancy")
        print(f"Answer: '{answer}'")
        print(f"  Judge faithfulness score (1-5): {faith_judge_score}")
        print(f"  Judge relevancy score (1-5):    {relevancy_judge_score}\n")

    print("Note: this is a SIMPLIFIED simulation combining our own proxy metrics into")
    print("a 1-5 scale, purely to illustrate the shape of an LLM-as-judge score -- a")
    print("real LLM judge reasons over the text directly rather than aggregating")
    print("separately-computed sub-scores like this toy example does.")


if __name__ == "__main__":
    demo_ragas_metrics()
    demo_llm_judge()
