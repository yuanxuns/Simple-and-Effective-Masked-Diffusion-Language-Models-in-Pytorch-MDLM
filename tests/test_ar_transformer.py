"""Tests for the cached causal Transformer baseline."""

import unittest

import torch

from models.ar_transformer import CausalTransformerLM


class CausalTransformerCacheTest(unittest.TestCase):
    """Ensure incremental KV-cache inference matches complete-prefix inference."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = CausalTransformerLM(
            vocab_size=11, max_seq_len=12, hidden_size=24, depth=2, num_heads=3
        ).eval()

    def test_cached_next_token_logits_match_full_forward(self):
        """Appending one cached token gives logits equal to a full ``(B, L+1)`` pass."""
        prefix = torch.randint(11, (2, 5))
        next_token = torch.randint(11, (2, 1))

        prefix_logits, cache = self.model(prefix, use_cache=True)
        full_logits = self.model(prefix)
        cached_logits, updated_cache = self.model(
            next_token, past_key_values=cache, use_cache=True
        )
        full_extended_logits = self.model(torch.cat((prefix, next_token), dim=1))

        self.assertTrue(torch.allclose(prefix_logits, full_logits, atol=1e-6))
        self.assertTrue(torch.allclose(cached_logits[:, -1], full_extended_logits[:, -1], atol=1e-6))
        self.assertEqual(updated_cache[0][0].shape, (2, 3, 6, 8))

    def test_generation_populates_and_uses_cache(self):
        """Generation returns the requested length and counts sequential forwards."""
        prefix = torch.randint(11, (2, 1))
        sample = self.model.generate(prefix, max_new_tokens=7, temperature=0)

        self.assertEqual(tuple(sample.shape), (2, 8))
        self.assertEqual(self.model.last_nfe, 7)


if __name__ == "__main__":
    unittest.main()
