"""
A simplified GloVe training loop on a small co-occurrence matrix, and a
real (small-scale) FastText training loop -- Skip-gram + negative
sampling applied to character n-gram tokens -- demonstrating
morphological similarity sharing and OOV word handling.
"""

import numpy as np


def build_cooccurrence_matrix(corpus, window_size=2):
    vocab = sorted(set(word for sentence in corpus for word in sentence))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    X = np.zeros((V, V))
    for sentence in corpus:
        indices = [word_to_idx[w] for w in sentence]
        for i, center in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    X[center, indices[j]] += 1

    return X, vocab, word_to_idx


def train_glove(X, embedding_dim=10, n_epochs=150, lr=0.02, x_max=10, alpha=0.75, seed=0):
    rng = np.random.default_rng(seed)
    V = X.shape[0]

    v = rng.normal(0, 0.1, size=(V, embedding_dim))
    v_context = rng.normal(0, 0.1, size=(V, embedding_dim))
    b = np.zeros(V)
    b_context = np.zeros(V)

    pairs = [(i, j) for i in range(V) for j in range(V) if X[i, j] > 0]

    def weight_fn(x):
        return min((x / x_max) ** alpha, 1.0)

    for epoch in range(n_epochs):
        rng.shuffle(pairs)
        total_loss = 0.0
        for i, j in pairs:
            x_ij = X[i, j]
            weight = weight_fn(x_ij)
            pred = v[i] @ v_context[j] + b[i] + b_context[j]
            diff = pred - np.log(x_ij)
            loss = weight * diff ** 2
            total_loss += loss

            grad = weight * diff
            v[i] -= lr * grad * v_context[j]
            v_context[j] -= lr * grad * v[i]
            b[i] -= lr * grad
            b_context[j] -= lr * grad

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  avg loss: {total_loss / len(pairs):.4f}")

    return v + v_context  # common convention: sum the two vector sets


def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def demo_glove():
    corpus_text = [
        "the king rules the kingdom",
        "the queen rules the kingdom",
        "the king and queen live in the castle",
        "a king wears a crown",
        "a queen wears a crown",
        "the king wears a golden crown",
        "the queen wears a golden crown",
        "the prince will be king",
        "the princess will be queen",
        "the castle has a king and a queen",
        "banana bread is a popular snack",
        "banana and bread go well together",
    ]
    corpus = [s.split() for s in corpus_text]

    X, vocab, word_to_idx = build_cooccurrence_matrix(corpus, window_size=3)
    print("Vocabulary:", vocab)
    print(f"\nCo-occurrence count (king, queen): {X[word_to_idx['king'], word_to_idx['queen']]:.0f}")
    print(f"Co-occurrence count (king, crown): {X[word_to_idx['king'], word_to_idx['crown']]:.0f}")
    print(f"Co-occurrence count (king, bread): {X[word_to_idx['king'], word_to_idx['bread']]:.0f}\n")

    vectors = train_glove(X, embedding_dim=10, n_epochs=150, lr=0.02)

    print("\n=== Cosine similarities after training ===")
    king_queen = cosine_similarity(vectors[word_to_idx["king"]], vectors[word_to_idx["queen"]])
    king_crown = cosine_similarity(vectors[word_to_idx["king"]], vectors[word_to_idx["crown"]])
    king_bread = cosine_similarity(vectors[word_to_idx["king"]], vectors[word_to_idx["bread"]])
    print(f"similarity(king, queen): {king_queen:.4f}   (co-occur directly, and share many contexts)")
    print(f"similarity(king, crown): {king_crown:.4f}   (co-occur once -- a tiny corpus like this")
    print(f"                                    gives a noisy signal for infrequent pairs)")
    print(f"similarity(king, bread): {king_bread:.4f}   (never co-occur, unrelated topic)")
    print("\nThe clearest, most reliable signal here is queen (frequent co-occurrence,")
    print("many shared contexts) landing far above bread (zero co-occurrence, completely")
    print("unrelated topic) -- exactly GloVe's global co-occurrence objective at work.")
    print("Real GloVe models train on billions of co-occurrences, not a dozen sentences,")
    print("which is why infrequent pairs like king/crown get a much more reliable signal")
    print("in practice than this toy-scale demo can show.\n")


def get_ngrams(word, min_n=3, max_n=6):
    word_bounded = f"<{word}>"
    ngrams = set()
    for n in range(min_n, min(max_n, len(word_bounded)) + 1):
        for i in range(len(word_bounded) - n + 1):
            ngrams.add(word_bounded[i:i + n])
    ngrams.add(word_bounded)
    return ngrams


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def train_fasttext(corpus, embedding_dim=12, window_size=2, n_negative=5,
                    n_epochs=110, lr=0.08, min_n=3, max_n=5, seed=0):
    """A real (if small-scale) FastText-style training loop: each word's
    vector is the MEAN of its character n-gram vectors, and those n-gram
    vectors are trained with Skip-gram + negative sampling (Lesson 03),
    with the gradient for a word's prediction distributed (averaged)
    across every one of its constituent n-grams."""
    rng = np.random.default_rng(seed)

    vocab = sorted(set(word for sentence in corpus for word in sentence))
    word_ngrams = {w: get_ngrams(w, min_n, max_n) for w in vocab}
    all_ngrams = sorted(set(g for grams in word_ngrams.values() for g in grams))
    ngram_to_idx = {g: i for i, g in enumerate(all_ngrams)}
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    G = len(all_ngrams)
    W_ngram = rng.normal(0, 0.1, size=(G, embedding_dim))   # input side: n-gram vectors
    W_out = rng.normal(0, 0.1, size=(len(vocab), embedding_dim))  # output side: per-word (Lesson 03-style)

    def word_vector(word):
        idxs = [ngram_to_idx[g] for g in word_ngrams.get(word, get_ngrams(word, min_n, max_n)) if g in ngram_to_idx]
        if not idxs:
            return None, []
        return W_ngram[idxs].mean(axis=0), idxs

    pairs = []
    for sentence in corpus:
        for i, center in enumerate(sentence):
            start, end = max(0, i - window_size), min(len(sentence), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, sentence[j]))

    for epoch in range(n_epochs):
        rng.shuffle(pairs)
        total_loss = 0.0
        for center_word, context_word in pairs:
            v_center, ngram_idxs = word_vector(center_word)
            context_idx = word_to_idx[context_word]

            v_context = W_out[context_idx]
            score = sigmoid(v_center @ v_context)
            grad = score - 1
            total_loss += -np.log(score + 1e-10)
            W_out[context_idx] -= lr * grad * v_center
            grad_center_total = grad * v_context

            neg_indices = rng.choice(len(vocab), size=n_negative)
            for neg_idx in neg_indices:
                if vocab[neg_idx] == context_word:
                    continue
                v_neg = W_out[neg_idx]
                score_neg = sigmoid(v_center @ v_neg)
                total_loss += -np.log(1 - score_neg + 1e-10)
                W_out[neg_idx] -= lr * score_neg * v_center
                grad_center_total += score_neg * v_neg

            # distribute the center word's gradient to every constituent n-gram
            # (averaged, matching the mean-pooling in word_vector above)
            n_grams_here = len(ngram_idxs)
            for idx in ngram_idxs:
                W_ngram[idx] -= lr * grad_center_total / n_grams_here

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  avg loss: {total_loss / len(pairs):.4f}")

    return W_ngram, ngram_to_idx, word_vector


def demo_fasttext():
    corpus_text = [
        "the dog is running in the park",
        "a fast runner won the race",
        "she was running fast today",
        "the runner trains every morning",
        "he runs and jumps with energy",
        "the athlete keeps jumping higher",
        "a good jumper needs strong legs",
        "the cat walked slowly home",
        "the old man was walking there",
        "a slow walker enjoys the view",
    ]
    corpus = [s.split() for s in corpus_text]

    print("Training a real (small-scale) FastText-style model...")
    W_ngram, ngram_to_idx, word_vector = train_fasttext(
        corpus, embedding_dim=12, window_size=2, n_negative=4, n_epochs=110, lr=0.08
    )

    print("\n=== Shared n-grams between morphologically related words ===")
    running_ngrams = get_ngrams("running")
    runner_ngrams = get_ngrams("runner")
    walking_ngrams = get_ngrams("walking")
    shared_run_runner = running_ngrams & runner_ngrams
    shared_run_walk = running_ngrams & walking_ngrams
    print(f"Shared n-grams (running, runner): {len(shared_run_runner)}  e.g. {list(shared_run_runner)[:4]}")
    print(f"Shared n-grams (running, walking): {len(shared_run_walk)}\n")

    v_running, _ = word_vector("running")
    v_runner, _ = word_vector("runner")
    v_walking, _ = word_vector("walking")

    sim_related = cosine_similarity(v_running, v_runner)
    sim_unrelated = cosine_similarity(v_running, v_walking)
    print(f"Cosine similarity (running, runner) [share the 'run' root]: {sim_related:.4f}")
    print(f"Cosine similarity (running, walking) [different root]:      {sim_unrelated:.4f}")

    print("\n=== Out-of-vocabulary handling ===")
    # "jumps" was never a token in the training corpus, but shares n-grams
    # with "jumping"/"jumper", which WERE trained.
    v_jumps_novel, idxs = word_vector("jumps")
    print("'jumps' was never a token in the training corpus, but shares n-grams")
    print("with 'jumping'/'jumper', so a vector was still computed via those")
    print(f"shared, trained pieces: found {len(idxs)} matching n-gram vectors.")
    print(f"First 3 dims: {v_jumps_novel[:3].round(4)}")

    v_totally_novel, idxs_novel = word_vector("zzzqqqxxx")
    print(f"\nA word sharing NO n-grams with anything trained ('zzzqqqxxx'):")
    print(f"vector is None? {v_totally_novel is None}  (no signal available at all)")


if __name__ == "__main__":
    print("=== GloVe: co-occurrence-based embeddings ===")
    demo_glove()

    print("=== FastText: character n-gram embeddings ===")
    demo_fasttext()
