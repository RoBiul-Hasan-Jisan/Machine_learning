"""
Constrained character-level generation enforcing a simple custom
grammar (a restricted {"key": N} format), with a position-dependent
state machine for valid-token masking, compared against unconstrained
generation to show how often the unconstrained version breaks format.
"""

import re

import torch
import torch.nn as nn


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
    return {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}


def train_char_lm(text, n_epochs=200, seed=0):
    torch.manual_seed(seed)
    char_to_idx, idx_to_char = build_char_vocab(text)
    ids = [char_to_idx[c] for c in text]
    seq_len = 20
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


# --- A tiny state machine for the grammar: {"key": N}  where N is 1-3 digits ---
# States: OPEN_BRACE, KEY_QUOTE_OPEN, KEY_CHARS, KEY_QUOTE_CLOSE, COLON,
#         VALUE_DIGIT, VALUE_DIGIT_OR_CLOSE, DONE

def valid_next_chars(generated_so_far, vocab):
    """Given what's been generated so far (a string), return the SET of
    characters that would keep the output on a valid path toward
    {"key": N} with N being 1-3 digits."""
    s = generated_so_far

    if s == "":
        return {"{"}
    if s == "{":
        return {'"'}
    if re.fullmatch(r'\{"[a-z]*', s):
        # inside the key: either another lowercase letter, or close the quote
        # (require at least 1 char before allowing close)
        key_part = s[2:]
        if len(key_part) == 0:
            return set("abcdefghijklmnopqrstuvwxyz")
        return set("abcdefghijklmnopqrstuvwxyz") | {'"'}
    if re.fullmatch(r'\{"[a-z]+"', s):
        return {":"}
    if re.fullmatch(r'\{"[a-z]+":', s):
        return {" "}
    if re.fullmatch(r'\{"[a-z]+": ', s):
        return set("0123456789")
    if re.fullmatch(r'\{"[a-z]+": \d', s):
        return set("0123456789") | {"}"}
    if re.fullmatch(r'\{"[a-z]+": \d\d', s):
        return set("0123456789") | {"}"}
    if re.fullmatch(r'\{"[a-z]+": \d{1,3}', s):
        return {"}"}
    if re.fullmatch(r'\{"[a-z]+": \d{1,3}\}', s):
        return set()  # DONE -- nothing more is valid
    return set()  # anything not matching a known valid prefix -> dead end (shouldn't happen if constrained)


def constrained_generate(model, char_to_idx, idx_to_char, max_len=20):
    model.eval()
    generated = ""
    hidden = None
    last_char = None
    with torch.no_grad():
        for _ in range(max_len):
            valid_chars = valid_next_chars(generated, char_to_idx)
            if not valid_chars:
                break

            if last_char is None:
                # bootstrap: no prior context yet, deterministically pick the
                # (only) valid start character -- no model call needed
                next_char = sorted(valid_chars)[0]
            else:
                x = torch.tensor([[char_to_idx[last_char]]], dtype=torch.long)
                logits, hidden = model(x, hidden)
                step_logits = logits[0, -1].clone()
                mask = torch.full((len(char_to_idx),), float("-inf"))
                for c in valid_chars:
                    if c in char_to_idx:
                        mask[char_to_idx[c]] = 0.0
                step_logits = step_logits + mask
                next_id = step_logits.argmax().item()
                next_char = idx_to_char[next_id]

            generated += next_char
            last_char = next_char

    return generated


def unconstrained_generate(model, char_to_idx, idx_to_char, seed_text='{"', max_len=18, temperature=0.8, seed=0):
    torch.manual_seed(seed)
    model.eval()
    ids = [char_to_idx.get(c, 0) for c in seed_text]
    hidden = None
    generated = list(seed_text)
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long)
        logits, hidden = model(x, hidden)
        for _ in range(max_len):
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(idx_to_char[next_id])
            x = torch.tensor([[next_id]], dtype=torch.long)
            logits, hidden = model(x, hidden)
    return "".join(generated)


def is_valid_output(s):
    return re.fullmatch(r'\{"[a-z]+": \d{1,3}\}', s) is not None


def demo_constrained_vs_unconstrained():
    # Train on a corpus of mostly-valid examples of the target format,
    # deliberately mixed with some OFF-FORMAT noise, so the model has a
    # strong but imperfect tendency toward the format -- the realistic
    # situation constrained decoding is meant to fix.
    corpus = (
        '{"age": 25} {"count": 3} {"score": 100} {"id": 7} {"total": 42} '
        '{"value": 9} {"level": 12} {"num": 88} {"rank": 1} {"size": 256} '
        'the age was twenty five and the count seemed pretty high overall '
        'a score of one hundred is considered a very good result indeed '
    ) * 20

    print("=== Training character LM on a corpus of valid {\"key\": N} examples ===")
    model, char_to_idx, idx_to_char = train_char_lm(corpus, n_epochs=200)

    print("\n=== Unconstrained generation (20 samples, temperature=0.8) ===")
    valid_count = 0
    for i in range(20):
        output = unconstrained_generate(model, char_to_idx, idx_to_char, seed=i)
        # only check the part up to the first closing brace, if any
        match = re.search(r'\{.*?\}', output)
        candidate = match.group(0) if match else output
        valid = is_valid_output(candidate)
        valid_count += valid
        if i < 6:
            print(f"  '{candidate}'  valid={valid}")
    print(f"\n  {valid_count}/20 unconstrained samples were valid.")

    print("\n=== Constrained generation (guaranteed valid by construction) ===")
    valid_count_constrained = 0
    for i in range(6):
        output = constrained_generate(model, char_to_idx, idx_to_char)
        valid = is_valid_output(output)
        valid_count_constrained += valid
        print(f"  '{output}'  valid={valid}")
    print("(Deterministic argmax-based masking always finds SOME valid completion --")
    print("here it repeatedly lands on the same one, since we're taking the single")
    print("most-likely valid token at every step; sampling within the valid set,")
    print("the same way Lesson 16 samples within an unconstrained distribution,")
    print("would recover output diversity while keeping the same hard guarantee.)")

    print("\nConstrained decoding masks out every token that would break the grammar,")
    print("so the output is structurally guaranteed valid -- not just usually valid")
    print("because the model happened to be trained well.")


if __name__ == "__main__":
    demo_constrained_vs_unconstrained()
