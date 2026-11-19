import unittest

import torch

from cet_graphsage.model import EdgeQueryHistoricalAggregator


class DecayOnlyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = EdgeQueryHistoricalAggregator(3, 2, 8, 0., aggregation='decay_only',
                                                   use_lag_encoding=True, use_time_decay=True,
                                                   fixed_time_decay=2.).eval()
        self.history = {'edge': torch.randn(2, 3, 3), 'neighbour': torch.randn(2, 3, 2),
                        'direction': torch.tensor([[0, 1, 0], [1, 0, 1]]),
                        'lag': torch.tensor([[.1, .7, 2.], [.2, .4, .8]]),
                        'mask': torch.tensor([[True, True, False], [False, False, False]])}

    def test_recency_weights_masks_and_no_content_matching(self):
        first, weights = self.model(torch.randn(2, 3), torch.randn(2, 2), self.history)
        expected = torch.tensor([1., torch.exp(torch.tensor(-1.2)), 0.])
        expected /= expected.sum()
        torch.testing.assert_close(weights[0], expected)
        torch.testing.assert_close(weights[1], torch.zeros(3), rtol=0, atol=0)
        torch.testing.assert_close(first[1], torch.zeros(8), rtol=0, atol=0)
        second, other = self.model(torch.randn(2, 3)*100, torch.randn(2, 2)*100, self.history)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(weights, other, rtol=0, atol=0)

    def test_lag_value_path_and_gradients_are_retained(self):
        context, _ = self.model(torch.randn(2, 3), torch.randn(2, 2), self.history)
        context[0, 0].backward()
        self.assertGreater(self.model.value_encoder.network[0].weight.grad[:, -1].abs().sum().item(), 0.)
        self.assertIsNone(self.model.query_encoder.network[0].weight.grad)
        self.assertIsNone(self.model.key_encoder.network[0].weight.grad)
        self.assertFalse(self.model.raw_time_decay.requires_grad)


if __name__ == '__main__':
    unittest.main()
