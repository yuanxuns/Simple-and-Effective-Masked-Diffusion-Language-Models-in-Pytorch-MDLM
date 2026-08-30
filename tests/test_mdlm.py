"""Unit tests for the absorbing-mask MDLM process.

Run from the repository root with ``python -m unittest tests.test_mdlm``.
"""

import unittest

import torch
from torch import nn

from models.mdlm import MaskedDiffusion, build_masked_diffusion


class DummyDenoiser(nn.Module):
    """Minimal denoiser producing logits of shape ``(B, *data_shape, V)``."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Sampling uses this parameter to infer the model device.
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x_t, t):
        """Return broadcast logits for ``x_t: (B, *data_shape)`` and ``t: (B,)``."""
        return self.bias.view(*([1] * x_t.ndim), self.vocab_size).expand(
            *x_t.shape, self.vocab_size
        )


class MaskedDiffusionTest(unittest.TestCase):
    """Check shapes and invariants of forward, loss, and reverse processes."""

    def setUp(self):
        self.k = 4
        self.mask_id = self.k
        self.diffusion = MaskedDiffusion(
            V=self.k + 1, K=self.k, mask_id=self.mask_id, num_timesteps=8
        )
        self.model = DummyDenoiser(self.k + 1)

    def test_q_sample_uses_given_noise(self):
        """Known uniform noise yields the expected ``(B, L)`` masked tokens."""
        x_start = torch.tensor([[0, 1], [2, 3], [1, 0]])
        t = torch.tensor([0.0, 0.5, 1.0])
        noise = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.0, 0.9]])

        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        expected = torch.tensor([[0, 1], [2, self.mask_id], [self.mask_id, self.mask_id]])
        self.assertTrue(torch.equal(x_t, expected))

    def test_subs_logits_masks_and_freezes(self):
        """SUBS disables mask predictions and freezes unmasked ``(B, L)`` values."""
        x_t = torch.tensor([[self.mask_id, 2]])
        raw_logits = torch.randn(1, 2, self.k + 1)

        logits = self.diffusion.subs_logits(raw_logits, x_t)

        self.assertEqual(tuple(logits.shape), (1, 2, self.k + 1))
        self.assertLess(logits[0, 0, self.mask_id].item(), -1e20)
        self.assertEqual(logits[0, 1, 2].item(), 0.0)
        self.assertTrue(torch.all(logits[0, 1, torch.arange(self.k + 1) != 2] < -1e20))

    def test_losses_and_bpd_have_batch_shape(self):
        """Loss and BPD integration return finite values per batch element."""
        x_start = torch.randint(self.k, (3, 2, 3))
        t = self.diffusion.sample_t_timesteps(x_start.shape[0], x_start.device)

        losses = self.diffusion.training_losses(self.model, x_start, t)
        bpd = self.diffusion.calc_bpd_loop(self.model, x_start, num_steps=3)

        self.assertEqual(tuple(losses.shape), (3,))
        self.assertTrue(torch.isfinite(losses).all())
        self.assertEqual(tuple(bpd["vbterms"].shape), (3, 3))
        self.assertEqual(tuple(bpd["total"].shape), (3,))

    def test_sampling_never_leaves_a_mask(self):
        """Both samplers return token tensors of requested shape without mask IDs."""
        for sampler in ("ancestral", "confidence"):
            sample = self.diffusion.p_sample_loop_strided(
                self.model, shape=(2, 3, 2), num_steps=4, sampler=sampler
            )
            self.assertEqual(tuple(sample.shape), (2, 3, 2))
            self.assertFalse(bool((sample == self.mask_id).any()))

    def test_factory_builds_clean_states_plus_mask(self):
        """Factory exposes ``V = K + 1`` and validates the canonical mask ID."""
        diffusion = build_masked_diffusion(self.k, self.mask_id, V=self.k + 1)
        self.assertEqual((diffusion.V, diffusion.K, diffusion.mask_id), (5, 4, 4))
        with self.assertRaises(ValueError):
            build_masked_diffusion(self.k, self.mask_id, V=self.k + 2)


if __name__ == "__main__":
    unittest.main()
