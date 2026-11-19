import unittest

import torch
import torch.nn.functional as F

from cet_graphsage.model import CETGraphSAGE, EdgeQueryHistoricalAggregator, ModelOutput


class IndependentTimeComponentsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.edge = torch.randn(2, 4)
        self.profile = torch.randn(2, 3)
        self.history = {"edge": torch.randn(2, 3, 4), "neighbour": torch.randn(2, 3, 3),
                        "direction": torch.tensor([[0, 1, 0], [1, 0, 1]]),
                        "lag": torch.tensor([[.1, 1., 3.], [.2, 2., 4.]]),
                        "mask": torch.ones(2, 3, dtype=torch.bool)}

    def aggregator(self, **kwargs):
        return EdgeQueryHistoricalAggregator(4, 3, 8, 0., **kwargs).eval()

    def test_legacy_default_keeps_coupled_behavior_and_state_shapes(self):
        for decay in (False, True):
            legacy = self.aggregator(use_time_decay=decay)
            explicit = self.aggregator(use_time_decay=decay, use_lag_encoding=decay)
            explicit.load_state_dict(legacy.state_dict(), strict=True)
            for left, right in zip(legacy(self.edge, self.profile, self.history),
                                   explicit(self.edge, self.profile, self.history)):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_message_lag_and_score_decay_have_independent_switches(self):
        for lag, decay in ((False, False), (True, False), (False, True), (True, True)):
            model = self.aggregator(use_lag_encoding=lag, use_time_decay=decay)
            captured = []
            hook = model.value_encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
            before = model(self.edge, self.profile, self.history)
            expected = self.history["lag"] if lag else torch.zeros_like(self.history["lag"])
            torch.testing.assert_close(captured[0][..., -1], expected)
            with torch.no_grad():
                model.raw_time_decay.fill_(5.)
            after = model(self.edge, self.profile, self.history)
            hook.remove()
            if decay:
                self.assertFalse(torch.allclose(before[1], after[1]))
            else:
                for left, right in zip(before, after):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            if lag and not decay:
                changed = {**self.history, "lag": torch.zeros_like(self.history["lag"])}
                context, _ = model(self.edge, self.profile, changed)
                self.assertFalse(torch.allclose(context, after[0]))

    def test_unweighted_control_is_plain_cross_entropy(self):
        model = CETGraphSAGE(4, 3, 8, 8, 3, use_lag_encoding=True)
        logits = torch.tensor([[2., 0., -1.], [0., 1., 2.]], requires_grad=True)
        labels = torch.tensor([0, 1])
        output = ModelOutput(logits, torch.zeros(2, 8), torch.zeros(2, 3), torch.zeros(2, 3))
        result, _ = model.loss(output, labels, torch.ones(3))
        torch.testing.assert_close(result, F.cross_entropy(logits, labels))
        result.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
