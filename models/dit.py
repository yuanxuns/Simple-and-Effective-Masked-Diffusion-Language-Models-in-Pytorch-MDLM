"""Compact conditional DiT denoiser for discrete diffusion experiments."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(timesteps, dim, max_period=10_000):
    """Embed continuous or discrete times into Fourier features.

    Args:
        timesteps: Floating time coordinates of shape ``(B,)``.
        dim: Feature width ``D``.

    Returns:
        Sinusoidal features of shape ``(B, D)``.
    """
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / max(half, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]  # (B, 1) * (1, D/2)
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    return F.pad(embedding, (0, dim % 2)) if dim % 2 else embedding


class DiTBlock(nn.Module):
    """AdaLN-modulated Transformer block.

    Inputs:
        x: Token features with shape ``(B, N, D)``.
        cond: Per-example conditioning features with shape ``(B, D)``.

    Output:
        Updated token features with shape ``(B, N, D)``.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        if (hidden_size // num_heads) % 2:
            raise ValueError("Each attention head dimension must be even for RoPE.")
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = float(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size))

    def _apply_rope(self, q_or_k):
        """Apply rotary position encoding to ``(B, H, N, Dh)`` queries or keys."""
        length = q_or_k.shape[2]
        half = self.head_dim // 2
        positions = torch.arange(length, device=q_or_k.device, dtype=torch.float32)
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(half, device=q_or_k.device, dtype=torch.float32) / half
        )
        angles = positions[:, None] * frequencies[None, :]
        cos, sin = angles.cos().to(q_or_k.dtype)[None, None], angles.sin().to(q_or_k.dtype)[None, None]
        first, second = q_or_k[..., :half], q_or_k[..., half:]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

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
        batch_size, length, _ = h.shape
        qkv = self.qkv(h).view(batch_size, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # (B, N, H, Dh) -> (B, H, N, Dh); RoPE preserves this shape.
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        q, k = self._apply_rope(q), self._apply_rope(k)
        attention = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        attention = attention.transpose(1, 2).reshape(batch_size, length, -1)
        x = x + self.residual_dropout(self.attn_out(attention))
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
        num_heads: Number of attention heads; must divide ``hidden_size`` and
            leave an even per-head width for RoPE.
        dropout: Attention and MLP dropout probability during training.
        condition_classes: Number of labels. ``None`` makes the model unconditional.
        class_dropout_prob: Probability of replacing a label with the learned null
            label during training, enabling classifier-free guidance.

    Inputs:
        ``x`` has shape ``(B, *input_shape)`` with integer values in ``[0, K-1]``.
        ``t`` has shape ``(B,)`` and contains continuous log-noise levels
        ``sigma=-log(1-t_mask)`` (or integer indices in ``[0, T-1]``). Optional
        ``y`` has shape ``(B,)``.

    Output:
        Tensor of shape ``(B, *input_shape, K)`` containing clean-state logits.
    """

    def __init__(
        self, input_shape, num_classes, num_timesteps, hidden_size=192,
        depth=6, num_heads=6, condition_classes=None, class_dropout_prob=0.1,
        dropout=0.1,
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
        self.num_timesteps = int(num_timesteps)
        # Continuous Fourier features avoid sparsely-trained discrete time bins.
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.class_embed = (
            nn.Embedding(condition_classes + 1, hidden_size)
            if condition_classes is not None else None
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def _forward_logits(self, x, t, y):
        """Predict logits for one fixed conditioning assignment.

        Args:
            x: Integer noisy-state tensor of shape ``(B, *input_shape)``.
            t: Continuous log-noise tensor ``(B,)`` or integer timestep index.
            y: Class-label tensor of shape ``(B,)`` or ``None``.

        Returns:
            Tensor of shape ``(B, *input_shape, K)``.
        """
        # Flatten data axes: (B, *input_shape) -> (B, N), then embed -> (B, N, D).
        tokens = self.token_embed(x.long().reshape(x.shape[0], self.num_tokens))
        # MDLM supplies continuous log-noise sigma. Fourier features retain its
        # full high-noise resolution: (B,) -> (B, D) -> (B, D).
        if torch.is_floating_point(t):
            t_coordinate = t
        else:
            t_coordinate = t.float()
        if bool((~torch.isfinite(t_coordinate) | (t_coordinate < 0)).any()):
            raise ValueError("time coordinates must be finite and non-negative.")
        cond = self.time_embed(sinusoidal_timestep_embedding(t_coordinate, tokens.shape[-1]))
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
        # Positional information is injected into attention queries/keys by RoPE.
        h = tokens  # (B, N, D)
        for block in self.blocks:
            h = block(h, cond)
        logits = self.head(self.norm(h))  # (B, N, D) -> (B, N, K)
        return logits.reshape(x.shape[0], *self.input_shape, self.num_classes)

    def forward(self, x, t, y=None, cfg_scale=1.0):
        """Predict clean categorical logits, optionally with CFG at inference.

        Args:
            x: Integer noisy-state tensor of shape ``(B, *input_shape)``.
            t: Continuous log-noise tensor ``(B,)`` or integer timestep index.
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
