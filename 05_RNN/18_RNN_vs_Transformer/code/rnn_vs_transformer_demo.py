"""
Self-attention implemented from scratch, verified against
torch.nn.MultiheadAttention, a runtime comparison between an RNN and a
self-attention layer as sequence length grows, and a sinusoidal
positional encoding implementation.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def self_attention(X, W_q, W_k, W_v):
    """X: (T, d_model). W_q, W_k, W_v: (d_model, d_k). Returns (T, d_k), attention_weights (T, T)."""
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    d_k = W_q.shape[1]
    scores = (Q @ K.T) / math.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ V
    return output, attention_weights


def demo_self_attention_vs_torch():
    torch.manual_seed(0)
    T, d_model = 5, 8

    X = torch.randn(T, d_model)
    W_q = torch.randn(d_model, d_model) * 0.3
    W_k = torch.randn(d_model, d_model) * 0.3
    W_v = torch.randn(d_model, d_model) * 0.3

    our_output, our_weights = self_attention(X, W_q, W_k, W_v)

    # torch.nn.MultiheadAttention with num_heads=1 and matching in_proj weights
    mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, bias=False, batch_first=True)
    with torch.no_grad():
        # in_proj_weight stacks [W_q; W_k; W_v] each of shape (d_model, d_model),
        # and MultiheadAttention computes X @ W.T internally (note the transpose).
        combined = torch.cat([W_q.T, W_k.T, W_v.T], dim=0)
        mha.in_proj_weight.copy_(combined)
        mha.out_proj.weight.copy_(torch.eye(d_model))  # identity output projection for a clean comparison

    X_batched = X.unsqueeze(0)  # (1, T, d_model)
    torch_output, torch_weights = mha(X_batched, X_batched, X_batched)
    torch_output = torch_output.squeeze(0)
    torch_weights = torch_weights.squeeze(0)

    print("Our output shape:  ", tuple(our_output.shape))
    print("Torch output shape:", tuple(torch_output.shape))
    assert torch.allclose(our_output, torch_output, atol=1e-4), \
        f"max diff: {(our_output - torch_output).abs().max()}"
    assert torch.allclose(our_weights, torch_weights, atol=1e-4)
    print("Match confirmed against torch.nn.MultiheadAttention (single head).\n")


def demo_runtime_comparison():
    print(f"{'seq_len':>8} | {'RNN time (s)':>14} | {'Self-attn time (s)':>20}")
    hidden_size = 64
    for seq_len in [50, 200, 800]:
        rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        x = torch.randn(8, seq_len, hidden_size)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                rnn(x)
        rnn_time = (time.perf_counter() - start) / 5

        W_q = torch.randn(hidden_size, hidden_size) * 0.1
        W_k = torch.randn(hidden_size, hidden_size) * 0.1
        W_v = torch.randn(hidden_size, hidden_size) * 0.1

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                for b in range(x.shape[0]):
                    self_attention(x[b], W_q, W_k, W_v)
        attn_time = (time.perf_counter() - start) / 5

        print(f"{seq_len:8d} | {rnn_time:14.5f} | {attn_time:20.5f}")

    print("\n(On CPU with this naive per-sample attention loop, the gap may not favor")
    print("attention the way it would on a GPU with a properly batched, vectorized")
    print("implementation -- the key structural point is that RNN cost is forced to")
    print("be SEQUENTIAL across time steps, while attention's cost is parallelizable")
    print("across positions, which matters enormously once real parallel hardware")
    print("and batched matrix multiplication are used, as in any real Transformer.)\n")


def sinusoidal_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def demo_positional_encoding():
    seq_len, d_model = 20, 16
    pe = sinusoidal_positional_encoding(seq_len, d_model)

    print("Positional encoding shape:", pe.shape)
    print("\nFirst 4 dimensions at position 0:", pe[0, :4].round(4))
    print("First 4 dimensions at position 5:", pe[5, :4].round(4))
    print("First 4 dimensions at position 10:", pe[10, :4].round(4))
    print("\nEach position gets a distinct encoding vector (varying frequencies")
    print("per dimension), giving self-attention -- which otherwise has no")
    print("built-in notion of order -- a way to distinguish positions.")

    # Confirm every position's encoding is unique
    unique_rows = len(set(map(tuple, pe.round(6))))
    assert unique_rows == seq_len
    print(f"\nConfirmed: all {seq_len} positions have distinct encodings.")


if __name__ == "__main__":
    print("=== Self-attention vs torch.nn.MultiheadAttention ===")
    demo_self_attention_vs_torch()

    print("=== Runtime comparison: RNN vs self-attention ===")
    demo_runtime_comparison()

    print("=== Sinusoidal positional encoding ===")
    demo_positional_encoding()
