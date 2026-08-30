"""Causal Transformer baseline with KV-cached autoregressive decoding."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention supporting cached keys and values.

    ``x`` is ``(B, T, D)``. An optional cache contains ``(K, V)`` tensors, each
    ``(B, H, P, D/H)``, from the preceding ``P`` positions. The output is
    ``(B, T, D)`` and the returned cache, when requested, has ``P + T`` tokens.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = float(dropout)

    def forward(self, x, cache=None, use_cache=False):
        """Apply attention; new queries attend to all valid prefix positions."""
        batch_size, length, hidden_size = x.shape
        qkv = self.qkv(x).view(batch_size, length, 3, self.num_heads, self.head_dim)
        q, key, value = qkv.unbind(dim=2)
        q, key, value = (tensor.transpose(1, 2) for tensor in (q, key, value))
        past_length = 0 if cache is None else cache[0].shape[2]
        if cache is not None:
            key = torch.cat((cache[0], key), dim=2)
            value = torch.cat((cache[1], value), dim=2)

        # (B, H, T, Dh) @ (B, H, Dh, P+T) -> (B, H, T, P+T).
        scores = (q @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_positions = past_length + torch.arange(length, device=x.device)
        key_positions = torch.arange(past_length + length, device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
        probabilities = F.softmax(scores, dim=-1)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        h = probabilities @ value  # (B, H, T, Dh)
        h = h.transpose(1, 2).reshape(batch_size, length, hidden_size)
        output = self.proj(h)  # (B, T, D)
        return (output, (key, value)) if use_cache else output


class CausalTransformerBlock(nn.Module):
    """Pre-norm causal Transformer block with an optional attention KV cache."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        """Return updated ``(B, T, D)`` features and optional per-layer cache."""
        if use_cache:
            attention, cache = self.attn(self.norm1(x), cache, use_cache=True)
        else:
            attention = self.attn(self.norm1(x))
        x = x + attention
        x = x + self.mlp(self.norm2(x))
        return (x, cache) if use_cache else x


class CausalTransformerLM(nn.Module):
    """Decoder-only language model with efficient KV-cached generation.

    ``tokens`` has shape ``(B, L)``. Normal forward returns logits of shape
    ``(B, L, vocab_size)``. With ``use_cache=True`` it additionally returns one
    ``(K, V)`` pair per layer; each has shape ``(B, H, L, D/H)``.
    """

    def __init__(
        self, vocab_size, max_seq_len, hidden_size=384, depth=8, num_heads=6,
        mlp_ratio=4.0, dropout=0.0,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.max_seq_len = int(max_seq_len)
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        self.blocks = nn.ModuleList(
            [CausalTransformerBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.last_nfe = 0
        nn.init.normal_(self.position_embed, std=0.02)

    def forward(self, tokens, past_key_values=None, use_cache=False):
        """Return logits for new tokens and optionally append to a KV cache.

        With a cache, ``tokens`` must contain only the newly appended positions;
        the output shape is ``(B, new_length, vocab_size)``.
        """
        _, length = tokens.shape
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_length + length > self.max_seq_len:
            raise ValueError("Input sequence exceeds max_seq_len.")
        h = self.token_embed(tokens) + self.position_embed[:, past_length:past_length + length]
        next_cache = []
        for index, block in enumerate(self.blocks):
            cache = None if past_key_values is None else past_key_values[index]
            if use_cache:
                h, cache = block(h, cache, use_cache=True)
                next_cache.append(cache)
            else:
                h = block(h)
        logits = self.head(self.norm(h))
        return (logits, tuple(next_cache)) if use_cache else logits

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens, temperature=1.0):
        """Append tokens using per-layer KV caches; return ``(B, prefix_L + new_L)``.

        Only the prefix prefill uses more than one input token. Every following
        forward pass has sequence length one and reuses all cached K/V tensors.
        ``last_nfe`` counts forward calls, not their FLOPs.
        """
        if prefix.shape[1] + max_new_tokens > self.max_seq_len:
            raise ValueError("Requested generation exceeds max_seq_len.")
        tokens = prefix
        logits, past_key_values = self(tokens, use_cache=True)
        self.last_nfe = 1
        for index in range(max_new_tokens):
            next_logits = logits[:, -1]
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1)
            else:
                next_token = torch.multinomial(
                    torch.softmax(next_logits / temperature, dim=-1), num_samples=1
                ).squeeze(1)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            if index + 1 < max_new_tokens:
                logits, past_key_values = self(
                    next_token[:, None], past_key_values=past_key_values, use_cache=True
                )
                self.last_nfe += 1
        return tokens
