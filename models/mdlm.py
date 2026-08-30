import numpy as np
import torch
from torch import nn

from .mdlm_utils import categorical_log_likelihood, gumbel_argmax, meanflat


class MaskedDiffusion(nn.Module):
    """Continuous-time absorbing-mask diffusion for discrete token tensors.

    The forward marginal independently preserves each clean token with
    ``alpha(t) = 1 - t`` and otherwise replaces it by ``mask_id``.

    Inputs used by the public methods:
        ``x_start``/``x_t``: integer tensors of shape ``(B, *data_shape)``.
        ``t``: scalar or tensor of shape ``(B,)`` containing times in ``[0, 1]``.
        A denoiser returns logits of shape ``(B, *data_shape, V)``.

    Outputs:
        Training and BPD methods return tensors of shape ``(B,)``; sampling
        returns integer samples of requested shape ``(B, *data_shape)``.
    """

    def __init__(
        self,
        V,
        mask_id,
        num_timesteps=1000,
        eps=1e-3,
        eps_eval=None,
        low_discrepancy=True,
        time_conditioning=True,
        K=None,
        bpd_num_steps=256,
        sampler="ancestral",
        cache=False,
        confidence_temp=0.0,
        confidence_k="binomial",
    ):
        super().__init__()
        self.V = int(V)
        self.mask_id = int(mask_id)
        self.num_timesteps = int(num_timesteps)
        self.eps = float(eps)
        self.low_discrepancy = bool(low_discrepancy)
        self.time_conditioning = time_conditioning
        self.K = self.V - 1 if K is None else int(K)
        self.bpd_num_steps = min(int(bpd_num_steps), self.num_timesteps)
        self.eps_eval = (
            0.5 / self.bpd_num_steps if eps_eval is None else eps_eval
        )
        self.last_nfe = 0
        self.sampler = sampler
        self.cache = cache
        self.confidence_temp = confidence_temp
        self.confidence_k = confidence_k
        self._check_mask_id()
        self.register_buffer("_device_anchor", torch.zeros(1))

    def _check_mask_id(self):
        if not 0 <= self.mask_id < self.V:
            raise ValueError("Mask id is out side of vocabulary.")

        if self.mask_id < self.K:
            raise ValueError("Mask id is inside the data range.")

    @property
    def device(self):
        return self._device_anchor.device

    #  === Schedule ===
    @staticmethod
    def alpha(t):
        """Return token-survival probability ``1 - t`` with shape matching ``t``."""
        return 1.0 - t

    @staticmethod
    def expand(t, ndim):
        """Broadcast ``(B,)`` times to ``(B, 1, ..., 1)`` with ``ndim`` axes."""
        # (B,) -> (B, 1, ..., 1), compatible with (B, *data_shape).
        return t.reshape(t.shape[0], *([1] * (ndim - 1)))

    def _as_continuous(self, t):
        """Convert integer indices to ``(B,)`` times in ``[0, 1]``; pass floats through."""
        if not torch.is_floating_point(t):
            return t.to(torch.float32) / max(1, self.num_timesteps - 1)
        return t

    # === Timestep Conventions ===
    def sample_t_timesteps(self, bs, device):
        """Draw training times with output shape ``(bs,)`` in ``[eps, 1)``."""
        if self.low_discrepancy:
            offset = torch.rand((), device=device)
            strata = (torch.arange(bs, device=device) + offset) / bs
            t = self.eps + (1.0 - self.eps) * strata
        else:
            t = self.eps + (1.0 - self.eps) * torch.rand(bs, device=device)
        return t

    def to_internal_t(self, frac, bs, device):
        """Expand one fractional time into a float32 tensor of shape ``(bs,)``."""
        value = min(max(float(frac), self.eps), 1.0)
        return torch.full((bs,), value, dtype=torch.float32, device=device)

    # === Forward Process ===
    def q_sample(self, x_start, t, noise=None):
        """Sample ``x_t`` by independently masking each input position.

        Inputs:
            x_start: Clean integer tokens of shape ``(B, *data_shape)``.
            t: Times of shape ``(B,)`` (or integer timestep indices).
            noise: Optional uniform ``[0, 1)`` tensor of shape
                ``(B, *data_shape)`` for reproducible sampling.

        Returns:
            Integer noisy tokens of shape ``(B, *data_shape)``.
        """
        t = self._as_continuous(t)
        u = (
            torch.rand(x_start.shape, device=x_start.device)
            if noise is None
            else noise
        )
        if u.shape != x_start.shape or u.device != x_start.device:
            raise ValueError("noise must have the same shape as x_start.")
        keep = self.expand(self.alpha(t), x_start.dim())
        # (B,) -> (B, 1, ...); comparison broadcasts to (B, *data_shape).
        return torch.where(
            u < keep, x_start, torch.full_like(x_start, self.mask_id)
        )

    def prior_sample(self, shape, device):
        """Return the all-mask prior ``x_1`` of integer shape ``shape``."""
        return torch.full(shape, self.mask_id, dtype=torch.int64, device=device)

    # === SUBS Parametrization ===
    def model_x_start_logits(self, model, x_t, t):
        """Call the denoiser and return raw logits of shape ``(B, *data_shape, V)``."""
        t = self._as_continuous(t)
        if self.time_conditioning:
            t_net = t * self.time_conditioning
        else:
            t_net = torch.zeros_like(t)
        return model(x_t, t_net)

    def subs_logits(self, logits, x_t):
        """Apply the SUBS reverse-process parameterization.

        Inputs:
            logits: Raw clean-token logits of shape ``(B, *data_shape, V)``.
            x_t: Current tokens of shape ``(B, *data_shape)``.

        Returns:
            Constrained logits with shape ``(B, *data_shape, V)``. Mask has zero
            probability, and positions already unmasked are one-hot frozen.
        """
        if logits.shape[:-1] != x_t.shape or logits.shape[-1] != self.V:
            raise ValueError("logits must have shape (*x_t.shape, V).")
        neg = torch.finfo(logits.dtype).min / 2

        # 1. p(mask) = 0
        logits = logits.clone()
        logits[..., self.mask_id] = neg

        # 2. carry-over unmasking
        # (B, *data_shape) -> (B, *data_shape, 1) for categorical broadcast.
        is_masked = (x_t == self.mask_id).unsqueeze(-1)
        frozen = torch.full_like(logits, neg)
        frozen.scatter_(-1, x_t.unsqueeze(-1), 0.0)
        return torch.where(is_masked, logits, frozen)

    def x_start_logits(self, model, x_t, t):
        """Return SUBS-constrained clean-token logits of shape ``(B, *data_shape, V)``."""
        raw = self.model_x_start_logits(model, x_t, t)
        return self.subs_logits(raw, x_t)

    # === Loss ===
    def _nelbo_bpd_terms(self, model, x_start, t, x_t=None):
        """
        For the log-linear schedule the continuous-time NELBO is:
          alpha'_t(i)/(1-alpha_t(i)) sum_{i: x_t(i)=mask} - log(p_theta(x(i))),
          where alpha'_t(i)/(1-alpha_t(i)) = 1/t

        Inputs ``x_start`` and optional ``x_t`` have shape ``(B, *data_shape)``;
        ``t`` has shape ``(B,)``. Returns per-example BPD terms of shape ``(B,)``.
        """

        t = self._as_continuous(t)
        if x_t is None:
            x_t = self.q_sample(x_start, t)

        logits = self.x_start_logits(model, x_t, t)
        nll = -categorical_log_likelihood(x_start, logits)
        # zero out unmasked positions
        nll = nll * (x_t == self.mask_id).to(nll.dtype)

        # (B,) -> (B, 1, ...), then weighted NLL remains (B, *data_shape).
        weight = 1.0 / self.expand(t, nll.dim()).clamp_min(self.eps)
        return meanflat(weight * nll) / np.log(2.0)

    def training_losses(self, model, x_start, t):
        """Return per-example continuous-time NELBO estimates with shape ``(B,)``."""
        losses = self._nelbo_bpd_terms(model, x_start, t)
        assert losses.shape == (x_start.shape[0],)
        return losses

    @torch.no_grad()
    def calc_bpd_loop(self, model, x_start, num_steps=None):
        """Numerically integrate NELBO estimates.

        Input ``x_start`` has shape ``(B, *data_shape)``. Returned ``total`` and
        ``prior`` have shape ``(B,)``; ``vbterms`` has shape ``(B, num_steps)``.
        """
        n = self.bpd_num_steps if num_steps is None else int(num_steps)
        batch_size = x_start.shape[0]
        device = x_start.device
        vbterms = []

        for j in range(n):
            frac = (j + 0.5) / n
            t = torch.full(
                (batch_size,), max(frac, self.eps_eval), device=device
            )
            vbterms.append(self._nelbo_bpd_terms(model, x_start, t) / n)

        vbterms = torch.stack(vbterms, dim=1)
        prior = torch.zeros(batch_size, device=device)
        return {"total": vbterms.sum(dim=1), "vbterms": vbterms, "prior": prior}

    # === Sampling ===
    def forward_jump(self, x_s, s, t):
        """Re-noise ``x_s`` into ``x_t`` for scalar times ``t > s``.

        Input and output token tensors both have shape ``(B, *data_shape)``.
        """
        alpha_s = self.alpha(s)
        alpha_t = self.alpha(t)
        remask_p = 1.0 - (alpha_t / max(alpha_s, 1e-12))
        u = torch.rand(x_s.shape, device=x_s.device)
        return torch.where(
            u < remask_p, torch.full_like(x_s, self.mask_id), x_s
        )

    def _decoder_step(self, logits, x_t, t, s, greedy, sampler, temp, k_mode):
        """Sample one reverse transition from time ``t`` to scalar ``s <= t``.

        ``logits`` has shape ``(B, *data_shape, V)`` and ``x_t`` has shape
        ``(B, *data_shape)``. Returns ``x_s`` with shape ``(B, *data_shape)``.
        """
        is_masked = x_t == self.mask_id
        if not bool(is_masked.any()):
            return x_t

        decoded = (
            torch.argmax(logits, dim=-1) if greedy else gumbel_argmax(logits)
        )
        stay_p = (s / t) if t > 0 else 0.0

        if sampler == "ancestral":
            unmask = (
                torch.rand(x_t.shape, device=x_t.device) >= stay_p
            ) & is_masked
        else:
            unmask = self._confidence_select(
                logits, is_masked, stay_p, temp, k_mode
            )
        return torch.where(unmask, decoded, x_t)

    def _confidence_select(self, logits, is_masked, stay_p, temp, k_mode):
        """Choose masked positions to reveal using MaskGIT-style confidence.

        ``logits`` is ``(B, *data_shape, V)`` and ``is_masked`` is
        ``(B, *data_shape)``. The returned Boolean reveal mask is
        ``(B, *data_shape)``.
        """
        bs = logits.shape[0]
        flat_mask = is_masked.reshape(bs, -1)
        n_masked = flat_mask.sum(dim=1)

        # Reduce category probabilities: (B, *data_shape, V) -> (B, N).
        conf = (
            torch.log_softmax(logits, dim=-1).max(dim=-1).values.reshape(bs, -1)
        )
        if temp == float("inf"):
            score = torch.zeros_like(conf)
        else:
            score = conf / max(temp, 1e-8) if temp > 0 else conf
        gumbel = -torch.log(
            -torch.log(
                torch.rand_like(score).clamp(torch.finfo(score.dtype).tiny)
            )
        )
        score = score + gumbel if temp != 0 else conf
        score = score.masked_fill(~flat_mask, -float("inf"))

        if k_mode == "binomial":
            k = torch.binomial(
                n_masked.float(),
                torch.full_like(n_masked.float(), 1.0 - stay_p),
            )
        else:
            k = torch.round(n_masked.float() * (1.0 - stay_p)).long()

        # Convert sorted positions into ranks, both with shape (B, N).
        order = score.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1,
            order,
            torch.arange(order.shape[1], device=order.device).expand_as(order),
        )
        chosen = (rank < k.unsqueeze(1)) & flat_mask
        return chosen.reshape(is_masked.shape)

    @torch.no_grad()
    def p_sample_loop_strided(
        self,
        model,
        shape,
        num_steps=None,
        device=None,
        greedy_final=False,
        return_intermediates=False,
        resample_r=1,
        resample_jump=1.0,
        sampler=None,
        cache=None,
        confidence_temp=None,
        confidence_k=None,
    ):
        """Generate discrete samples by strided reverse diffusion.

        Args:
            model: Denoiser mapping ``(B, *data_shape), (B,)`` to
                ``(B, *data_shape, V)`` logits.
            shape: Output token shape ``(B, *data_shape)``.

        Returns:
            Integer samples of shape ``shape``; if ``return_intermediates`` is
            true, also returns a list of ``num_steps + 1`` tensors of that shape.
        """
        if device is None:
            device = next(model.parameters()).device
        if num_steps is None:
            num_steps = self.num_timesteps
        num_steps = max(1, int(num_steps))
        repeats = max(1, int(resample_r))
        sampler = self.sampler if sampler is None else sampler
        cache = self.cache if cache is None else cache
        confidence_temp = (
            self.confidence_temp if confidence_temp is None else confidence_temp
        )
        confidence_k = (
            self.confidence_k if confidence_k is None else confidence_k
        )

        if cache and self.time_conditioning:
            cache = False

        # ts goes 1 -> 0.
        ts = np.linspace(1.0, 0.0, num_steps + 1)

        x = self.prior_sample(shape, device)
        intermediates = [x]
        nfe = 0

        cached_logits, cached_x = None, None
        for i in range(num_steps):
            t_cur = float(ts[i])
            t_next = float(ts[i + 1])
            t_now = t_cur
            for r in range(repeats):
                if (
                    cache
                    and cached_x is not None
                    and cached_logits is not None
                    and torch.equal(x, cached_x)
                ):
                    logits = cached_logits
                else:
                    t_b = torch.full(
                        (shape[0],), max(t_now, self.eps), device=device
                    )
                    logits = self.subs_logits(
                        self.model_x_start_logits(model, x, t_b), x
                    )
                    nfe += 1
                    if cache:
                        cached_logits, cached_x = logits, x.clone()

                greedy = greedy_final and t_next <= 0.0
                x_next = self._decoder_step(
                    logits,
                    x,
                    t_now,
                    t_next,
                    greedy,
                    sampler,
                    confidence_temp,
                    confidence_k,
                )
                if r == repeats - 1 or t_next <= 0.0:
                    x = x_next
                    break

                span = resample_jump * (t_now - t_next)
                t_back = min(t_cur, t_next + span)
                if t_back <= t_next:
                    x = x_next
                    break
                x = self.forward_jump(x_next, t_next, t_back)
                t_now = t_back
            intermediates.append(x)

        self.last_nfe = nfe
        assert not bool((x == self.mask_id).any()), (
            "mask token survived sampling"
        )
        assert x.shape == shape
        return (x, intermediates) if return_intermediates else x

    @torch.no_grad()
    def p_sample_loop(
        self, model, shape, num_timesteps=None, return_x_innit=False
    ):
        """Compatibility wrapper around strided sampling.

        Returns a sample of shape ``shape``. When ``return_x_innit`` is true,
        returns ``(x_1, sample)`` where both integer tensors have shape ``shape``.
        """
        device = next(model.parameters()).device
        x_init = self.prior_sample(tuple(shape), device)
        x = self.p_sample_loop_strided(
            model,
            tuple(shape),
            num_steps=num_timesteps or self.num_timesteps,
            device=device,
        )
        return (x_init, x) if return_x_innit else x


def build_maksed_diffusion(
    K,
    mask_id,
    num_timesteps=1000,
    eps=1e-3,
    low_discrepancy=True,
    time_conditioning=True,
    V=None,
    bpd_num_steps=256,
    sampler="ancestral",
    cache=False,
    confidence_temp=0.0,
    confidence_k="binomial",
):
    """Build a :class:`MaskedDiffusion` with ``K`` clean states plus one mask.

    ``mask_id`` must equal ``K``: clean states are ``[0, K)`` and the mask is
    their final vocabulary entry. Returns a process with vocabulary size ``K + 1``
    and token logits whose last dimension is ``K + 1``.
    """
    if mask_id != K:
        raise ValueError("build_maksed_diffusion requires mask_id == K.")
    if V is not None and V != K + 1:
        raise ValueError("V, when supplied, must equal K+1.")
    return MaskedDiffusion(
        V=V,
        K=K,
        mask_id=mask_id,
        num_timesteps=num_timesteps,
        eps=eps,
        low_discrepancy=low_discrepancy,
        time_conditioning=time_conditioning,
        bpd_num_steps=bpd_num_steps,
        sampler=sampler,
        cache=cache,
        confidence_temp=confidence_temp,
        confidence_k=confidence_k,
    )


# Keep the original misspelled factory as a compatibility alias.
build_masked_diffusion = build_maksed_diffusion
