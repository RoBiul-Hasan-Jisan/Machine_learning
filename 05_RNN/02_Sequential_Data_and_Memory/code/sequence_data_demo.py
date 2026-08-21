"""
Tokenizing/numericalizing text, padding a batch of variable-length
sequences, constructing a mask, and comparing manual masking to
PyTorch's pack_padded_sequence.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


def tokenize(sentence):
    return sentence.lower().split()


def build_vocab(sentences):
    vocab = {"<pad>": 0, "<unk>": 1}
    for sentence in sentences:
        for token in tokenize(sentence):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def numericalize(sentence, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenize(sentence)]


def pad_batch(sequences, pad_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded = []
    mask = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_value] * pad_len)
        mask.append([1] * len(seq) + [0] * pad_len)
    return torch.tensor(padded), torch.tensor(mask)


def demo_tokenize_and_numericalize():
    sentences = [
        "the cat sat",
        "the dog barked loudly at night",
        "birds fly",
    ]
    vocab = build_vocab(sentences)
    print("Vocabulary:", vocab)

    numericalized = [numericalize(s, vocab) for s in sentences]
    for s, n in zip(sentences, numericalized):
        print(f"'{s}' -> {n}")
    return numericalized


def demo_padding_and_mask(numericalized):
    padded, mask = pad_batch(numericalized)
    print("\nPadded batch:\n", padded)
    print("Mask:\n", mask)

    lengths_from_mask = mask.sum(dim=1)
    true_lengths = torch.tensor([len(seq) for seq in numericalized])
    print("\nLengths recovered from mask:", lengths_from_mask.tolist())
    print("True lengths:                ", true_lengths.tolist())
    assert torch.equal(lengths_from_mask, true_lengths)
    print("Mask correctly encodes each sequence's true length.\n")

    return padded, mask


def demo_pack_padded_sequence(padded, mask):
    lengths = mask.sum(dim=1)
    embedding_dim = 4
    vocab_size = int(padded.max().item()) + 1

    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    rnn = nn.RNN(input_size=embedding_dim, hidden_size=6, batch_first=True)

    embedded = embedding(padded)  # (batch, T, embedding_dim)

    # Naive: run the RNN on the padded tensor directly (computes on pad positions too)
    naive_output, naive_hidden = rnn(embedded)

    # Packed: skip computation on padded positions entirely
    packed_input = pack_padded_sequence(
        embedded, lengths, batch_first=True, enforce_sorted=False
    )
    packed_output, packed_hidden = rnn(packed_input)

    print("Naive (padded) final hidden state shape:", tuple(naive_hidden.shape))
    print("Packed final hidden state shape:         ", tuple(packed_hidden.shape))
    print("\nNote: the packed version's final hidden state reflects each sequence's")
    print("TRUE last time step, not the last (possibly padding) position -- this")
    print("is the concrete benefit of using pack_padded_sequence over naive padding.")


if __name__ == "__main__":
    print("=== Tokenize and numericalize ===")
    numericalized = demo_tokenize_and_numericalize()

    print("\n=== Padding and mask ===")
    padded, mask = demo_padding_and_mask(numericalized)

    print("=== pack_padded_sequence comparison ===")
    demo_pack_padded_sequence(padded, mask)
