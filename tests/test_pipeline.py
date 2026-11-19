import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cet_graphsage.data import IndexDataset, make_collate_fn, prepare_dataset
from cet_graphsage.model import CETGraphSAGE


def synthetic_frame(rows: int = 90) -> pd.DataFrame:
    records = []
    attacks = ["Benign", "Exploits", "Reconnaissance"]
    for index in range(rows):
        records.append(
            {
                "IPV4_SRC_ADDR": f"10.0.0.{index % 5}",
                "L4_SRC_PORT": 1000 + index,
                "IPV4_DST_ADDR": f"10.0.1.{(index * 3) % 7}",
                "L4_DST_PORT": 20 + index % 8,
                "PROTOCOL": 6 if index % 3 else 17,
                "L7_PROTO": float(index % 4),
                "IN_BYTES": 20 + index,
                "IN_PKTS": 1 + index % 5,
                "OUT_BYTES": 50 + index * 2,
                "OUT_PKTS": 2 + index % 7,
                "FLOW_DURATION_MILLISECONDS": index % 20,
                "DURATION_IN": index % 4,
                "DURATION_OUT": index % 6,
                "MIN_TTL": 30,
                "MAX_TTL": 64,
                "SRC_TO_DST_SECOND_BYTES": 100 + index,
                "DST_TO_SRC_SECOND_BYTES": 80 + index,
                "RETRANSMITTED_IN_PKTS": index % 2,
                "RETRANSMITTED_OUT_PKTS": (index + 1) % 2,
                "SRC_TO_DST_AVG_THROUGHPUT": 200 + index,
                "DST_TO_SRC_AVG_THROUGHPUT": 100 + index,
                "NUM_PKTS_UP_TO_128_BYTES": 1 + index % 3,
                "NUM_PKTS_1024_TO_1514_BYTES": index % 2,
                "Label": int(index % 3 != 0),
                "Attack": attacks[index % len(attacks)],
            }
        )
    return pd.DataFrame(records)


class PipelineTest(unittest.TestCase):
    def test_causal_preprocessing_and_model_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "flows.csv"
            synthetic_frame().to_csv(csv_path, index=False)
            data = prepare_dataset(
                csv_path,
                root / "cache",
                sample_fraction=1.0,
                windows=(2, 4, 8),
                history_size=4,
                train_ratio=0.6,
                val_ratio=0.2,
                progress=lambda _: None,
            )

            self.assertTrue(np.allclose(data.src_profiles[0], 0.0))
            self.assertTrue(np.allclose(data.dst_profiles[0], 0.0))
            for index in range(data.num_samples):
                valid_src = data.src_history[index][data.src_history[index] >= 0]
                valid_dst = data.dst_history[index][data.dst_history[index] >= 0]
                self.assertTrue(np.all(valid_src < index))
                self.assertTrue(np.all(valid_dst < index))

            collate = make_collate_fn(data)
            batch = collate([10, 11, 12, 13])
            model = CETGraphSAGE(
                edge_dim=data.edge_dim,
                profile_dim=data.profile_dim,
                hidden_dim=16,
                representation_dim=16,
                num_classes=data.num_classes,
                dropout=0.0,
            )
            output = model(batch)
            self.assertEqual(output.logits.shape, (4, data.num_classes))
            loss, _ = model.loss(output, batch["label"])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

            for config in (
                {"use_profiles": False, "use_history": False, "use_time_decay": False},
                {"use_history": False, "use_time_decay": False},
                {"use_lag_encoding": False, "use_time_decay": False},
                {"use_lag_encoding": True, "use_time_decay": False},
                {"use_lag_encoding": True, "fixed_time_decay": 2.0},
                {"aggregation": "decay_only", "fixed_time_decay": 2.0},
            ):
                variant_model = CETGraphSAGE(
                    edge_dim=data.edge_dim,
                    profile_dim=data.profile_dim,
                    hidden_dim=16,
                    representation_dim=16,
                    num_classes=data.num_classes,
                    dropout=0.0,
                    **config,
                )
                variant_output = variant_model(batch)
                variant_loss, _ = variant_model.loss(
                    variant_output, batch["label"]
                )
                self.assertTrue(
                    torch.isfinite(variant_loss), f"Non-finite loss for {config}"
                )


if __name__ == "__main__":
    unittest.main()
