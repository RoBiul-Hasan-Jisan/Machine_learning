"""
A rule-based chatbot, a retrieval-based chatbot using TF-IDF similarity
over a small conversation database, and a minimal generative chatbot
built on a character-level language model -- compared side by side.
"""

import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Rule-based chatbot ---

RULES = [
    (r"i feel (.*)", "Why do you feel {0}?"),
    (r"i need (.*)", "Why do you need {0}?"),
    (r"i am (.*)", "How long have you been {0}?"),
    (r".*\b(hello|hi|hey)\b.*", "Hello! How can I help you today?"),
    (r".*\b(thanks|thank you)\b.*", "You're welcome!"),
    (r".*", "Can you tell me more about that?"),
]


def eliza_respond(user_input):
    text = user_input.lower().strip()
    for pattern, response_template in RULES:
        match = re.match(pattern, text)
        if match:
            groups = match.groups()
            return response_template.format(*groups) if groups else response_template
    return "I'm not sure I understand."


# --- Retrieval-based chatbot ---

CONVERSATION_DB = [
    {"context": "what is your name", "response": "I'm a simple demo chatbot."},
    {"context": "what time is it", "response": "I don't have access to a clock, sorry!"},
    {"context": "how is the weather today", "response": "I can't check the weather, but I hope it's nice!"},
    {"context": "can you help me with python programming", "response": "Sure, I can try to help with Python questions."},
    {"context": "tell me a joke", "response": "Why did the developer go broke? Because they used up all their cache."},
    {"context": "what is machine learning", "response": "Machine learning is building systems that learn patterns from data."},
    {"context": "goodbye see you later", "response": "Goodbye! Have a great day."},
]


class RetrievalChatbot:
    def __init__(self, database):
        self.database = database
        self.vectorizer = TfidfVectorizer()
        contexts = [entry["context"] for entry in database]
        self.context_vectors = self.vectorizer.fit_transform(contexts)

    def respond(self, user_input):
        query_vector = self.vectorizer.transform([user_input.lower()])
        sims = cosine_similarity(query_vector, self.context_vectors)[0]
        best_idx = sims.argmax()
        return self.database[best_idx]["response"], sims[best_idx]


# --- Minimal generative chatbot (reusing Lesson 16's CharLM machinery) ---

class CharLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


def build_char_vocab(text):
    chars = sorted(set(text))
    return {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(sorted(set(text)))}


def train_generative_chatbot(n_epochs=200, seed=0):
    torch.manual_seed(seed)
    # Small synthetic "conversation" corpus: short turns concatenated with a marker
    corpus = (
        "user: hello bot: hello how can i help you "
        "user: how are you bot: i am doing well thank you "
        "user: what is your name bot: i am a simple chatbot "
        "user: goodbye bot: goodbye have a nice day "
    ) * 15

    char_to_idx, idx_to_char = build_char_vocab(corpus)
    ids = [char_to_idx[c] for c in corpus]
    seq_len = 40
    inputs, targets = [], []
    for i in range(0, len(ids) - seq_len - 1, seq_len):
        inputs.append(ids[i:i + seq_len])
        targets.append(ids[i + 1:i + seq_len + 1])
    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)

    model = CharLM(len(char_to_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = loss_fn(logits.reshape(-1, len(char_to_idx)), Y.reshape(-1))
        loss.backward()
        optimizer.step()

    return model, char_to_idx, idx_to_char


def generative_respond(model, char_to_idx, idx_to_char, user_input, max_len=60):
    model.eval()
    seed_text = f"user: {user_input} bot:"
    ids = [char_to_idx.get(c, 0) for c in seed_text]
    hidden = None
    generated = []
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long)
        logits, hidden = model(x, hidden)
        for _ in range(max_len):
            next_id = logits[0, -1].argmax().item()
            next_char = idx_to_char[next_id]
            if next_char == "u" and len("".join(generated)) > 3:  # crude stop at next "user:"
                break
            generated.append(next_char)
            x = torch.tensor([[next_id]], dtype=torch.long)
            logits, hidden = model(x, hidden)
    return "".join(generated).strip()


def demo_all_three():
    test_inputs = [
        "hello",
        "how are you",
        "what is your name",
        "i feel tired",
        "goodbye",
    ]

    print("=== Rule-based chatbot ===")
    for inp in test_inputs:
        print(f"  '{inp}' -> '{eliza_respond(inp)}'")

    print("\n=== Retrieval-based chatbot ===")
    retrieval_bot = RetrievalChatbot(CONVERSATION_DB)
    for inp in test_inputs:
        response, score = retrieval_bot.respond(inp)
        print(f"  '{inp}' -> '{response}'  (similarity={score:.3f})")

    print("\n=== Generative chatbot ===")
    model, char_to_idx, idx_to_char = train_generative_chatbot()
    for inp in test_inputs:
        response = generative_respond(model, char_to_idx, idx_to_char, inp)
        print(f"  '{inp}' -> '{response}'")

    print("\nNote the qualitative differences: the rule-based bot falls back to a")
    print("generic response for anything unanticipated ('goodbye' has no dedicated")
    print("rule here); the retrieval bot returns a VERBATIM stored response, only as")
    print("good as its database's coverage; the generative bot can produce novel")
    print("character sequences, but is only as coherent as its (here, tiny) training")
    print("data allows -- a real system would need vastly more conversational data.")


if __name__ == "__main__":
    demo_all_three()
