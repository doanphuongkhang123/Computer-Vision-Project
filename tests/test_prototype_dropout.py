import unittest

import torch

from models.vit_transformer_model import AdaptiveCrossAttentionPooling, VTransAdaptive


class PrototypeDropoutTest(unittest.TestCase):
    def test_rejects_invalid_probability(self):
        for value in (-0.1, 1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    AdaptiveCrossAttentionPooling(dim=8, heads=2, prototype_dropout=value)

    def test_training_drops_fixed_fraction_per_sample(self):
        pooler = AdaptiveCrossAttentionPooling(
            dim=8,
            max_tokens=20,
            heads=2,
            prototype_dropout=0.25,
        )
        pooler.train()

        keep_mask = pooler._prototype_keep_mask(batch_size=3, retrieved_count=4, device=torch.device("cpu"))

        self.assertEqual(keep_mask.shape, (3, 20))
        self.assertTrue(torch.equal(keep_mask.sum(dim=1), torch.tensor([15, 15, 15])))

    def test_dropout_never_removes_needed_retrieval_candidates(self):
        pooler = AdaptiveCrossAttentionPooling(
            dim=8,
            max_tokens=20,
            heads=2,
            prototype_dropout=0.9,
        )
        pooler.train()

        keep_mask = pooler._prototype_keep_mask(batch_size=2, retrieved_count=18, device=torch.device("cpu"))

        self.assertTrue(torch.equal(keep_mask.sum(dim=1), torch.tensor([18, 18])))

    def test_eval_uses_full_prototype_bank(self):
        pooler = AdaptiveCrossAttentionPooling(
            dim=8,
            max_tokens=20,
            heads=2,
            prototype_dropout=0.5,
        )
        pooler.eval()

        keep_mask = pooler._prototype_keep_mask(batch_size=2, retrieved_count=4, device=torch.device("cpu"))

        self.assertTrue(keep_mask.all())

    def test_model_forward_is_finite_with_prototype_dropout(self):
        model = VTransAdaptive(
            num_classes=3,
            hidden_dim=8,
            ratio=0.5,
            max_retrieved=4,
            prototype_dropout=0.5,
            dropout=0.0,
        )
        model.train()
        feats = torch.randn(2, 8, 8)
        attn_mask = torch.zeros(2, 8, dtype=torch.bool)

        output = model(feats, attn_mask)

        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
