"""Small causal Transformer baseline for the Text8 MDLM experiment."""

import torch
from torch import nn


class CausalTransformerLM(nn.Module):
    """Decoder-only language model returning next-token logits.

    Inputs:
        tokens: Integer prefix tokens of shape ``(B, L)`` with values in
            ``[0, vocab_size)`` and ``L <= max_seq_len``.

    Output:
        Logits of shape ``(B, L, vocab_size)``. Position ``i`` predicts the
        token at position ``i + 1`` during teacher-forced training.
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        hidden_size=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.max_seq_len = int(max_seq_len)
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.normal_(self.position_embed, std=0.02)

    def forward(self, tokens):
        """Return next-token logits with shape ``(B, L, vocab_size)``."""
        batch_size, length = tokens.shape
        if length > self.max_seq_len:
            raise ValueError("Input sequence exceeds max_seq_len.")
        h = self.token_embed(tokens) + self.position_embed[:, :length]
        # True values above the diagonal prohibit attending to future positions.
        causal_mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        h = self.blocks(h, mask=causal_mask)
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens, temperature=1.0):
        """Autoregressively append tokens; returns ``(B, prefix_L + new_L)``.

        This deliberately recomputes each prefix so its ``nfe`` counterpart is
        exactly ``max_new_tokens`` model forwards. It is a simple, fair baseline
        rather than an optimized KV-cache implementation.
        """
        tokens = prefix
        for _ in range(max_new_tokens):
            logits = self(tokens[:, -self.max_seq_len :])[:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1)
            else:
                next_token = torch.multinomial(
                    torch.softmax(logits / temperature, dim=-1), num_samples=1
                ).squeeze(1)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
        return tokens
