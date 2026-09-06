"""
Cross-lingual embedding alignment from scratch: learning a linear
mapping between two independently-trained toy embedding spaces using a
bilingual dictionary, with generalization to unseen translation pairs.
"""

import numpy as np


def make_synthetic_embedding_space(concepts, embedding_dim=10, seed=0):
    rng = np.random.default_rng(seed)
    concept_vectors = {c: rng.normal(0, 1, size=embedding_dim) for c in concepts}
    return concept_vectors


def apply_language_rotation(concept_vectors, rotation_seed, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    dim = len(next(iter(concept_vectors.values())))

    rng_rot = np.random.default_rng(rotation_seed)
    random_matrix = rng_rot.normal(size=(dim, dim))
    q, _ = np.linalg.qr(random_matrix)

    language_vectors = {}
    for concept, vec in concept_vectors.items():
        rotated = q @ vec
        language_vectors[concept] = rotated + rng.normal(0, noise, size=dim)
    return language_vectors


def learn_alignment(source_vectors, target_vectors, dictionary):
    X = np.array([source_vectors[src] for src, tgt in dictionary])
    Y = np.array([target_vectors[tgt] for src, tgt in dictionary])
    W, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    return W.T


def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def nearest_concept(vector, vector_dict, exclude=None):
    best_concept, best_sim = None, -2
    for concept, v in vector_dict.items():
        if concept == exclude:
            continue
        sim = cosine_similarity(vector, v)
        if sim > best_sim:
            best_sim, best_concept = sim, concept
    return best_concept, best_sim


def demo_cross_lingual_alignment():
    concepts = ["dog", "cat", "king", "queen", "man", "woman", "car", "truck", "happy", "sad"]
    shared_structure = make_synthetic_embedding_space(concepts, embedding_dim=10, seed=42)

    english_vectors = apply_language_rotation(shared_structure, rotation_seed=1, seed=10)
    french_vectors = apply_language_rotation(shared_structure, rotation_seed=2, seed=20)

    unaligned_sim = cosine_similarity(english_vectors["dog"], french_vectors["dog"])
    print(f"Cosine similarity, english 'dog' vs french 'dog', BEFORE alignment: {unaligned_sim:.4f}")
    print("(Low and effectively arbitrary -- the two spaces were trained independently")
    print("and have no inherent alignment, even though they represent the same concept.")
    print("Compare this to the near-perfect 1.0000 similarity achieved below, once")
    print("the learned transformation W is applied.)\n")

    train_concepts = ["dog", "cat", "king", "queen", "man", "woman", "happy", "sad"]
    held_out_concepts = ["car", "truck"]

    dictionary = [(c, c) for c in train_concepts]
    W = learn_alignment(english_vectors, french_vectors, dictionary)

    print("=== After learning the alignment (linear transformation W) ===")
    for concept in train_concepts[:3]:
        mapped = W @ english_vectors[concept]
        sim = cosine_similarity(mapped, french_vectors[concept])
        print(f"  '{concept}' (seen in dictionary): similarity after alignment = {sim:.4f}")

    print("\n=== Generalization to UNSEEN translation pairs ===")
    for concept in held_out_concepts:
        mapped = W @ english_vectors[concept]
        sim = cosine_similarity(mapped, french_vectors[concept])
        nearest, nearest_sim = nearest_concept(mapped, french_vectors)
        print(f"  '{concept}' (NEVER in dictionary): similarity to true translation = {sim:.4f}")
        print(f"    nearest French concept to mapped vector: '{nearest}' (similarity {nearest_sim:.4f})")

    print("\nThe alignment learned from only 8 known translation pairs generalizes")
    print("to correctly relate 'car'/'truck' -- concepts it never saw paired during")
    print("training -- because the two embedding spaces share similar INTERNAL")
    print("geometric structure, even though their absolute orientations differ.")


if __name__ == "__main__":
    demo_cross_lingual_alignment()
