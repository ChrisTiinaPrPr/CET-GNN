import tempfile
import unittest
from pathlib import Path

import numpy as np

from cet_graphsage.cic_data import attack_family, prepare_cic_dataset, select_rows
from test_pipeline import synthetic_frame


class CICDataTest(unittest.TestCase):
    def test_attack_families_and_unknown_labels(self):
        self.assertEqual(attack_family("FTP-BruteForce"), "BruteForce")
        self.assertEqual(attack_family("Brute Force -Web"), "Web Attacks")
        self.assertEqual(attack_family("Infilteration"), "Infiltration")
        with self.assertRaises(ValueError):
            attack_family("unknown attack")

    def test_time_ties_and_independent_partition_histories(self):
        frame = synthetic_frame(180)
        frame["Attack"] = np.tile(["Benign", "Bot", "FTP-BruteForce"], 60)
        frame["FLOW_END_MILLISECONDS"] = (np.arange(180) // 6) * 1000
        frame["FLOW_START_MILLISECONDS"] = frame["FLOW_END_MILLISECONDS"] - 1
        splits = select_rows(frame)
        for label in frame["Attack"].unique():
            sets = []
            for ids in splits:
                part = frame.loc[ids]
                sets.append(set(part.loc[part["Attack"] == label, "FLOW_END_MILLISECONDS"]))
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[1] & sets[2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.to_csv(root / "cic.csv", index=False)
            data = prepare_cic_dataset(root / "cic.csv", root / "cache", window=4,
                                       history_size=3, progress=lambda _: None)
            self.assertEqual(data.profile_dim, 9)
            self.assertNotIn("FLOW_END_MILLISECONDS", data.edge_feature_names)
            for lo, hi in ((0, data.train_end), (data.train_end, data.val_end), (data.val_end, data.num_samples)):
                np.testing.assert_array_equal(data.src_profiles[lo], np.zeros(9))
                for history in (data.src_history, data.dst_history):
                    valid = history[lo:hi] >= 0
                    self.assertTrue(np.all(history[lo:hi][valid] >= lo))
                    self.assertTrue(np.all((history[lo:hi] < np.arange(lo, hi)[:, None]) | ~valid))


if __name__ == "__main__":
    unittest.main()
