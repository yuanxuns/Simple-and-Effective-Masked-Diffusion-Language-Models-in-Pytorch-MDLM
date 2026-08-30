import torch
from torch.nn import functional as F


def log_min_exp(a, b, eps=1e-6):
    """
    Compute log(exp(a) - exp(b)) in a numerically stable way.

    Args:
        a: Tensor of shape (...,), with a > b element-wise.
        b: Tensor of shape (...,), with b < a element-wise.
        eps: Small safety constant for numerical stability.

    Returns:
        Tensor of shape (...) containing log(exp(a) - exp(b)).
    """
    assert torch.all(b < a), "b must be less than a for log_min_exp."
    return a + torch.log1p(-torch.exp(b - a) + eps)


def gumbel_argmax(logits):
    """
    Sample from a categorical distribution using the Gumbel-max trick.

    Args:
        logits: Tensor of shape (..., K) containing unnormalized category scores.

    Returns:
        Tensor of shape (...) with sampled category indices in [0, K-1].
    """
    noise = torch.rand_like(logits)
    noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.0)
    gumbel_noise = -torch.log(-torch.log(noise))
    return torch.argmax(logits + gumbel_noise, dim=-1)


def categorical_kl_probs(probs1, probs2, eps=1.0e-6):
    """
    Compute the KL divergence between two categorical probability tensors.

    Args:
        probs1: Tensor of shape (..., K) containing probability distribution p.
        probs2: Tensor of shape (..., K) containing probability distribution q.
        eps: Small constant to avoid log(0).

    Returns:
        Tensor of shape (...) with the per-sample KL divergence KL(probs1 || probs2).
    """
    out = probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps))
    return torch.sum(out, dim=-1)


def categorical_kl_logits(logits1, logits2, eps=1.0e-6):
    """
    Compute KL divergence from logits without explicitly normalizing the logits.

    Args:
        logits1: Tensor of shape (..., K) representing log-probabilities or logits.
        logits2: Tensor of shape (..., K) representing log-probabilities or logits.
        eps: Small constant for numerical stability.

    Returns:
        Tensor of shape (...) with KL(logits1 || logits2) evaluated after softmax normalization.
    """
    return categorical_kl_probs(
        F.softmax(logits1 + eps, dim=-1), F.softmax(logits2 + eps, dim=-1)
    )


def categorical_log_likelihood(x, logits):
    """
    Compute the log-likelihood of target class indices under a categorical distribution.

    Args:
        x: Target indices tensor of shape (batch_size, ...).
        logits: Unnormalized categorical logits of shape (batch_size, ..., K).

    Returns:
        Tensor of shape (batch_size, ...) containing the log-likelihood of each target class.
    """
    # (bs, ..., K)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, x.to(torch.int64).unsqueeze(-1)).squeeze(-1)


def meanflat(x):
    """
    Take the mean over all dimensions except the leading batch dimension.

    Args:
        x: Tensor of shape (batch_size, ...).

    Returns:
        Tensor of shape (batch_size,) when x has at least one non-batch dimension;
        for scalar tensors, returns a scalar tensor.
    """

    # (B, *data_shape) -> (B,).  No reduction is needed for a batch vector.
    return x if x.ndim == 1 else x.mean(dim=tuple(range(1, x.ndim)))
