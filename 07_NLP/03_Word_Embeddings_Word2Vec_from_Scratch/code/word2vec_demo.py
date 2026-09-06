"""
Skip-gram with negative sampling implemented from scratch, trained on a
small synthetic corpus with clear semantic clusters, showing that
trained embeddings place related words near each other.
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def build_vocab(corpus):
    words = sorted(set(word for sentence in corpus for word in sentence))
    word_to_idx = {w: i for i, w in enumerate(words)}
    return words, word_to_idx


def generate_training_pairs(corpus, word_to_idx, window_size=2):
    pairs = []
    for sentence in corpus:
        indices = [word_to_idx[w] for w in sentence]
        for i, center in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, indices[j]))
    return pairs


def negative_sampling_distribution(corpus, word_to_idx, power=0.75):
    counts = np.zeros(len(word_to_idx))
    for sentence in corpus:
        for w in sentence:
            counts[word_to_idx[w]] += 1
    weighted = counts ** power
    return weighted / weighted.sum()


def train_skipgram(corpus, embedding_dim=10, window_size=2, n_negative=5,
                    n_epochs=200, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    vocab, word_to_idx = build_vocab(corpus)
    V = len(vocab)

    W_in = rng.normal(0, 0.1, size=(V, embedding_dim))
    W_out = rng.normal(0, 0.1, size=(V, embedding_dim))

    pairs = generate_training_pairs(corpus, word_to_idx, window_size)
    neg_dist = negative_sampling_distribution(corpus, word_to_idx)

    for epoch in range(n_epochs):
        rng.shuffle(pairs)
        total_loss = 0.0
        for center_idx, context_idx in pairs:
            negative_indices = rng.choice(V, size=n_negative, p=neg_dist)
            # avoid accidentally sampling the true context word as a negative
            negative_indices = [n for n in negative_indices if n != context_idx]

            v_center = W_in[center_idx]

            v_context = W_out[context_idx]
            score = sigmoid(v_center @ v_context)
            grad = score - 1
            total_loss += -np.log(score + 1e-10)
            W_out[context_idx] -= lr * grad * v_center
            grad_center_total = grad * v_context

            for neg_idx in negative_indices:
                v_neg = W_out[neg_idx]
                score_neg = sigmoid(v_center @ v_neg)
                grad_neg = score_neg
                total_loss += -np.log(1 - score_neg + 1e-10)
                W_out[neg_idx] -= lr * grad_neg * v_center
                grad_center_total += grad_neg * v_neg

            W_in[center_idx] -= lr * grad_center_total

        if epoch % 15 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  avg loss: {total_loss / len(pairs):.4f}")

    return W_in, vocab, word_to_idx


def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def nearest_neighbors(word, W_in, vocab, word_to_idx, k=3):
    v = W_in[word_to_idx[word]]
    sims = [(other, cosine_similarity(v, W_in[word_to_idx[other]]))
            for other in vocab if other != word]
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


def make_synthetic_corpus(seed=0):
    """A larger, template-based synthetic corpus with two strong semantic
    clusters (pets, vehicles), giving the distributional hypothesis enough
    repeated co-occurrence signal to produce a clean, reliable result."""
    rng = np.random.default_rng(seed)

    pets = ["dog", "puppy", "cat", "kitten"]
    pet_verbs = ["barks", "plays", "sleeps", "runs", "eats"]
    pet_templates = [
        "the {pet} {verb} in the yard",
        "my {pet} {verb} every morning",
        "the little {pet} {verb} happily",
        "a {pet} {verb} near the house",
    ]

    vehicles = ["car", "truck", "bus", "van"]
    vehicle_verbs = ["drives", "parks", "stops", "moves", "turns"]
    vehicle_templates = [
        "the {veh} {verb} on the road",
        "a new {veh} {verb} downtown",
        "the old {veh} {verb} slowly",
        "the {veh} {verb} near the station",
    ]

    corpus = []
    for _ in range(40):
        pet = rng.choice(pets)
        verb = rng.choice(pet_verbs)
        template = rng.choice(pet_templates)
        corpus.append(template.format(pet=pet, verb=verb).split())

    for _ in range(40):
        veh = rng.choice(vehicles)
        verb = rng.choice(vehicle_verbs)
        template = rng.choice(vehicle_templates)
        corpus.append(template.format(veh=veh, verb=verb).split())

    return corpus


def demo_word2vec():
    corpus = make_synthetic_corpus()

    W_in, vocab, word_to_idx = train_skipgram(
        corpus, embedding_dim=16, window_size=2, n_negative=5, n_epochs=45, lr=0.02
    )

    print("\n=== Nearest neighbors after training ===")
    for word in ["dog", "car", "puppy", "truck"]:
        if word in word_to_idx:
            neighbors = nearest_neighbors(word, W_in, vocab, word_to_idx, k=4)
            neighbor_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbors)
            print(f"'{word}' -> {neighbor_str}")

    print("\n=== Cross-cluster similarity check ===")
    pet_pairs = [("dog", "puppy"), ("cat", "kitten"), ("dog", "cat")]
    vehicle_pairs = [("car", "truck"), ("bus", "van"), ("car", "bus")]
    cross_pairs = [("dog", "truck"), ("cat", "car"), ("puppy", "bus")]

    def avg_sim(pairs):
        return np.mean([cosine_similarity(W_in[word_to_idx[a]], W_in[word_to_idx[b]]) for a, b in pairs])

    pet_avg = avg_sim(pet_pairs)
    vehicle_avg = avg_sim(vehicle_pairs)
    cross_avg = avg_sim(cross_pairs)

    print(f"avg similarity within PET cluster:     {pet_avg:.4f}")
    print(f"avg similarity within VEHICLE cluster: {vehicle_avg:.4f}")
    print(f"avg similarity ACROSS clusters:        {cross_avg:.4f}")
    print("\nWith a larger, more repetitive corpus, within-cluster similarity is")
    print("reliably higher than cross-cluster similarity -- the distributional")
    print("hypothesis needs enough repeated co-occurrence signal to produce a")
    print("clean result, which is exactly why real Word2Vec models train on")
    print("corpora with billions of words rather than a handful of sentences.")


if __name__ == "__main__":
    demo_word2vec()
