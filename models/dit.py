"""Compact conditional DiT denoiser for discrete diffusion experiments."""

import math

import torch
from torch import nn


class DiTBlock(nn.Module):
    """AdaLN-modulated Transformer block.

    Inputs:
        x: Token features with shape ``(B, N, D)``.
        cond: Per-example conditioning features with shape ``(B, D)``.

    Output:
        Updated token features with shape ``(B, N, D)``.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size))

    def forward(self, x, cond):
        """Apply the block.

        Args:
            x: Token tensor of shape ``(B, N, D)``.
            cond: Conditioning tensor of shape ``(B, D)``.

        Returns:
            Tensor of shape ``(B, N, D)``.
        """
        shift1, scale1, shift2, scale2 = self.modulation(cond).chunk(4, dim=-1)
        # (B, D) -> (B, 1, D), broadcast over N tokens: (B, N, D).
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + self.attn(h, h, h, need_weights=False)[0]
        # Attention preserves token count, so x and h remain (B, N, D).
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + self.mlp(h)


class DiT(nn.Module):
    """A small class-conditional DiT that predicts categorical clean-data logits.

    The model is intentionally shape-agnostic: images use ``input_shape=(H, W)``;
    a two-dimensional point can use ``input_shape=(2,)``.  Each scalar discrete
    state becomes one token, so its output has one categorical logit vector per
    input state.

    Args:
        input_shape: Non-batch data dimensions, e.g. ``(28, 28)`` or ``(2,)``.
        num_classes: Number of D3PM states ``K``.
        num_timesteps: Number of diffusion timesteps ``T``.
        hidden_size: Transformer token width ``D``.
        depth: Number of DiT blocks.
        num_heads: Number of attention heads; must divide ``hidden_size``.
        condition_classes: Number of labels. ``None`` makes the model unconditional.
        class_dropout_prob: Probability of replacing a label with the learned null
            label during training, enabling classifier-free guidance.

    Inputs:
        ``x`` has shape ``(B, *input_shape)`` with integer values in ``[0, K-1]``.
        ``t`` has shape ``(B,)``. It may contain continuous diffusion times in
        ``[0, 1]`` or integer indices in ``[0, T-1]``. Optional ``y`` has shape
        ``(B,)``.

    Output:
        Tensor of shape ``(B, *input_shape, K)`` containing clean-state logits.
    """

    def __init__(
        self, input_shape, num_classes, num_timesteps, hidden_size=192,
        depth=6, num_heads=6, condition_classes=None, class_dropout_prob=0.1,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.input_shape = tuple(input_shape)
        self.num_tokens = math.prod(self.input_shape)
        self.num_classes = num_classes
        self.condition_classes = condition_classes
        self.class_dropout_prob = class_dropout_prob
        self.token_embed = nn.Embedding(num_classes, hidden_size)
        self.position_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size))
        self.time_embed = nn.Embedding(num_timesteps, hidden_size)
        self.class_embed = (
            nn.Embedding(condition_classes + 1, hidden_size)
            if condition_classes is not None else None
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.position_embed, std=0.02)

    def _forward_logits(self, x, t, y):
        """Predict logits for one fixed conditioning assignment.

        Args:
            x: Integer noisy-state tensor of shape ``(B, *input_shape)``.
            t: Continuous ``(B,)`` time in ``[0, 1]`` or integer timestep index.
            y: Class-label tensor of shape ``(B,)`` or ``None``.

        Returns:
            Tensor of shape ``(B, *input_shape, K)``.
        """
        # Flatten data axes: (B, *input_shape) -> (B, N), then embed -> (B, N, D).
        tokens = self.token_embed(x.long().reshape(x.shape[0], self.num_tokens))
        # MDLM supplies continuous times. Quantize only for the learned lookup table.
        if torch.is_floating_point(t):
            t_index = (t.clamp(0, 1) * (self.time_embed.num_embeddings - 1)).round().long()
        else:
            t_index = t.long()
        if bool(((t_index < 0) | (t_index >= self.time_embed.num_embeddings)).any()):
            raise ValueError("timestep indices must be in [0, num_timesteps - 1].")
        cond = self.time_embed(t_index)  # (B,) -> (B, D)
        if self.class_embed is not None:
            if y is None:
                y = torch.full_like(t, self.condition_classes)
            if y.shape != t.shape:
                raise ValueError("y must have shape (batch_size,).")
            y = y.long()
            if self.training and self.class_dropout_prob:
                dropped = torch.rand_like(y, dtype=torch.float) < self.class_dropout_prob
                y = torch.where(dropped, torch.full_like(y, self.condition_classes), y)
            cond = cond + self.class_embed(y)
        h = tokens + self.position_embed  # (B, N, D) + (1, N, D) -> (B, N, D)
        for block in self.blocks:
            h = block(h, cond)
        logits = self.head(self.norm(h))  # (B, N, D) -> (B, N, K)
        return logits.reshape(x.shape[0], *self.input_shape, self.num_classes)

    def forward(self, x, t, y=None, cfg_scale=1.0):
        """Predict clean categorical logits, optionally with CFG at inference.

        Args:
            x: Integer noisy-state tensor of shape ``(B, *input_shape)``.
            t: Continuous ``(B,)`` time in ``[0, 1]`` or integer timestep index.
            y: Optional class-label tensor of shape ``(B,)``.
            cfg_scale: Classifier-free guidance scale. ``1.0`` uses ordinary
                conditional logits; ``0.0`` uses null-label logits; values above
                one strengthen the conditional prediction. It is ignored for an
                unconditional model or when ``y`` is ``None``.

        Returns:
            Tensor of shape ``(B, *input_shape, K)``.
        """
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected x.shape[1:]={self.input_shape}, got {tuple(x.shape[1:])}."
            )
        if t.shape != (x.shape[0],):
            raise ValueError("t must have shape (batch_size,).")
        if self.class_embed is None or y is None or cfg_scale == 1.0:
            return self._forward_logits(x, t, y)
        conditional_logits = self._forward_logits(x, t, y)
        unconditional_logits = self._forward_logits(x, t, None)
        return unconditional_logits + cfg_scale * (
            conditional_logits - unconditional_logits
        )
