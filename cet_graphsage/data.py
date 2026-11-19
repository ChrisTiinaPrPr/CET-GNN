from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset


IP_COLUMNS = ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")
LABEL_COLUMN = "Attack"
DROP_FEATURE_COLUMNS = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack", "Label"}

# These values are accumulated causally for each endpoint.  Their ratios form
# the nine behaviour features used by EF-GraphSAGE.
BEHAVIOUR_COLUMNS = (
    "OUT_BYTES",
    "IN_BYTES",
    "OUT_PKTS",
    "IN_PKTS",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "RETRANSMITTED_IN_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "DURATION_OUT",
    "DURATION_IN",
    "MIN_TTL",
    "MAX_TTL",
)

LONG_TAIL_COLUMNS = {
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "OUT_BYTES",
    "IN_BYTES",
    "OUT_PKTS",
    "IN_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
}


@dataclass
class PreparedTrafficData:
    edge_features: np.ndarray
    src_profiles: np.ndarray
    dst_profiles: np.ndarray
    src_history: np.ndarray
    dst_history: np.ndarray
    src_history_direction: np.ndarray
    dst_history_direction: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    edge_feature_names: list[str]
    windows: tuple[int, ...]
    train_end: int
    val_end: int
    metadata: dict

    @property
    def profile_dim(self) -> int:
        return int(self.src_profiles.shape[1])

    @property
    def edge_dim(self) -> int:
        return int(self.edge_features.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])

    def split_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train = np.arange(0, self.train_end, dtype=np.int64)
        val = np.arange(self.train_end, self.val_end, dtype=np.int64)
        test = np.arange(self.val_end, self.num_samples, dtype=np.int64)
        return train, val, test

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            edge_features=self.edge_features,
            src_profiles=self.src_profiles,
            dst_profiles=self.dst_profiles,
            src_history=self.src_history,
            dst_history=self.dst_history,
            src_history_direction=self.src_history_direction,
            dst_history_direction=self.dst_history_direction,
            labels=self.labels,
            class_names=np.asarray(self.class_names),
            edge_feature_names=np.asarray(self.edge_feature_names),
            windows=np.asarray(self.windows, dtype=np.int64),
            train_end=np.asarray(self.train_end, dtype=np.int64),
            val_end=np.asarray(self.val_end, dtype=np.int64),
            metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PreparedTrafficData":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                edge_features=data["edge_features"],
                src_profiles=data["src_profiles"],
                dst_profiles=data["dst_profiles"],
                src_history=data["src_history"],
                dst_history=data["dst_history"],
                src_history_direction=data["src_history_direction"],
                dst_history_direction=data["dst_history_direction"],
                labels=data["labels"],
                class_names=data["class_names"].tolist(),
                edge_feature_names=data["edge_feature_names"].tolist(),
                windows=tuple(int(x) for x in data["windows"].tolist()),
                train_end=int(data["train_end"]),
                val_end=int(data["val_end"]),
                metadata=json.loads(str(data["metadata"])),
            )


class IndexDataset(Dataset):
    """Dataset that lets one vectorized collator gather complete flow batches."""

    def __init__(self, indices: Iterable[int]):
        if isinstance(indices, np.ndarray):
            self.indices = indices.astype(np.int64, copy=False)
        else:
            self.indices = np.fromiter(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> int:
        return int(self.indices[index])


class _NodeState:
    def __init__(self, windows: Sequence[int], stat_dim: int):
        self.buffers = [deque() for _ in windows]
        self.sums = [np.zeros(stat_dim, dtype=np.float64) for _ in windows]
        self.windows = windows

    def profile(self) -> np.ndarray:
        return np.concatenate([_ratios_from_sum(values) for values in self.sums])

    def append(self, values: np.ndarray) -> None:
        for limit, buffer, total in zip(self.windows, self.buffers, self.sums):
            buffer.append(values)
            total += values
            if len(buffer) > limit:
                total -= buffer.popleft()


def _ratios_from_sum(values: np.ndarray) -> np.ndarray:
    eps = 1e-8
    (
        out_bytes,
        in_bytes,
        out_pkts,
        in_pkts,
        tiny_pkts,
        large_pkts,
        retrans_out,
        retrans_in,
        throughput_out,
        throughput_in,
        duration_out,
        duration_in,
        min_ttl,
        max_ttl,
    ) = values
    total_pkts = out_pkts + in_pkts
    profile = np.asarray(
        [
            out_bytes / (out_bytes + in_bytes + eps),
            out_pkts / (total_pkts + eps),
            throughput_out / (throughput_out + throughput_in + eps),
            tiny_pkts / (total_pkts + eps),
            large_pkts / (total_pkts + eps),
            retrans_out / (out_pkts + eps),
            retrans_in / (in_pkts + eps),
            duration_out / (duration_out + duration_in + eps),
            min_ttl / (max_ttl + eps),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(profile, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _cache_key(
    csv_path: Path,
    sample_fraction: float,
    max_rows: int | None,
    windows: Sequence[int],
    history_size: int,
    train_ratio: float,
    val_ratio: float,
    *,
    order_by: str | None = None,
    exclude_features: Sequence[str] = (),
) -> str:
    config = {
        "path": str(csv_path.resolve()),
        "size": csv_path.stat().st_size,
        "mtime": csv_path.stat().st_mtime_ns,
        "sample_fraction": sample_fraction,
        "max_rows": max_rows,
        "windows": list(windows),
        "history_size": history_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }
    if order_by is not None:
        config["order_by"] = order_by
    if exclude_features:
        config["exclude_features"] = sorted(set(exclude_features))
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def _read_csv(
    csv_path: Path,
    sample_fraction: float,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is not None:
        # This option exists for smoke tests.  Research runs should sample the
        # complete file so that rare classes are not lost.
        frame = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    else:
        frame = pd.read_csv(csv_path, low_memory=False)
        if sample_fraction < 1.0:
            frame = (
                frame.groupby(LABEL_COLUMN, group_keys=False, sort=False)
                .sample(frac=sample_fraction, random_state=random_state)
                .sort_index()
            )
    frame.columns = frame.columns.str.strip()
    missing = set(IP_COLUMNS + (LABEL_COLUMN,)) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(frame) < 10:
        raise ValueError("At least ten flows are required.")
    return frame.reset_index(drop=True)


def _numeric_edge_features(
    frame: pd.DataFrame, exclude_features: Sequence[str] = ()
) -> tuple[np.ndarray, list[str]]:
    excluded = DROP_FEATURE_COLUMNS.union(exclude_features)
    feature_names = [c for c in frame.columns if c not in excluded]
    numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for column in feature_names:
        values = numeric[column].to_numpy(dtype=np.float64, copy=False)
        if column in LONG_TAIL_COLUMNS:
            numeric[column] = np.log1p(np.clip(values, 0.0, None))
    values = numeric.to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values, feature_names


def _causal_profiles_and_history(
    frame: pd.DataFrame,
    windows: tuple[int, ...],
    history_size: int,
    progress: Callable[[str], None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_ips = pd.concat([frame[IP_COLUMNS[0]], frame[IP_COLUMNS[1]]], ignore_index=True)
    codes, uniques = pd.factorize(all_ips, sort=False)
    n = len(frame)
    src_codes = codes[:n]
    dst_codes = codes[n:]

    behaviour = np.column_stack(
        [
            pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0).to_numpy()
            if column in frame.columns
            else np.zeros(n, dtype=np.float64)
            for column in BEHAVIOUR_COLUMNS
        ]
    ).astype(np.float64, copy=False)
    behaviour = np.nan_to_num(behaviour, nan=0.0, posinf=0.0, neginf=0.0)

    profile_dim = 9 * len(windows)
    src_profiles = np.zeros((n, profile_dim), dtype=np.float32)
    dst_profiles = np.zeros((n, profile_dim), dtype=np.float32)
    src_history = np.full((n, history_size), -1, dtype=np.int32)
    dst_history = np.full((n, history_size), -1, dtype=np.int32)
    src_direction = np.zeros((n, history_size), dtype=np.int8)
    dst_direction = np.zeros((n, history_size), dtype=np.int8)

    node_states: list[_NodeState | None] = [None] * len(uniques)
    node_history: list[deque] = [deque(maxlen=history_size) for _ in range(len(uniques))]

    def state_for(node: int) -> _NodeState:
        state = node_states[node]
        if state is None:
            state = _NodeState(windows, len(BEHAVIOUR_COLUMNS))
            node_states[node] = state
        return state

    report_every = max(1, n // 20)
    for index in range(n):
        src = int(src_codes[index])
        dst = int(dst_codes[index])
        src_state = state_for(src)
        dst_state = src_state if src == dst else state_for(dst)

        # Profiles and histories are captured before the current edge updates
        # either endpoint.  This is the core causal/test-then-update guarantee.
        src_profiles[index] = src_state.profile()
        dst_profiles[index] = dst_state.profile()

        src_items = list(reversed(node_history[src]))
        dst_items = src_items if src == dst else list(reversed(node_history[dst]))
        for position, (edge_index, direction) in enumerate(src_items):
            src_history[index, position] = edge_index
            src_direction[index, position] = direction
        for position, (edge_index, direction) in enumerate(dst_items):
            dst_history[index, position] = edge_index
            dst_direction[index, position] = direction

        src_state.append(behaviour[index])
        node_history[src].append((index, 0))  # Historical flow left this endpoint.
        if dst != src:
            dst_state.append(behaviour[index])
            node_history[dst].append((index, 1))  # Historical flow entered this endpoint.

        if (index + 1) % report_every == 0 or index + 1 == n:
            progress(f"Causal history preprocessing: {index + 1:,}/{n:,}")

    return (
        src_profiles,
        dst_profiles,
        src_history,
        dst_history,
        src_direction,
        dst_direction,
    )


def prepare_dataset(
    csv_path: str | Path,
    cache_dir: str | Path,
    sample_fraction: float = 0.1,
    max_rows: int | None = None,
    windows: Sequence[int] = (8, 32, 128),
    history_size: int = 32,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_state: int = 42,
    rebuild_cache: bool = False,
    progress: Callable[[str], None] = print,
    *,
    order_by: str | None = None,
    exclude_features: Sequence[str] = (),
) -> PreparedTrafficData:
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir)
    windows = tuple(sorted({int(value) for value in windows}))
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not windows or min(windows) <= 0:
        raise ValueError("All causal windows must be positive.")
    if history_size <= 0:
        raise ValueError("history_size must be positive.")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1].")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave a non-empty test split.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        csv_path,
        sample_fraction,
        max_rows,
        windows,
        history_size,
        train_ratio,
        val_ratio,
        order_by=order_by,
        exclude_features=exclude_features,
    )
    cache_path = cache_dir / f"prepared_unsw_{key}.npz"
    if cache_path.exists() and not rebuild_cache:
        progress(f"Loading prepared causal dataset from {cache_path}")
        return PreparedTrafficData.load(cache_path)

    progress(f"Reading {csv_path}")
    frame = _read_csv(csv_path, sample_fraction, max_rows, random_state)
    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    order_metadata = {}
    if order_by is not None:
        if order_by not in frame.columns or frame[order_by].isna().any():
            raise ValueError(f"Expected a complete ordering column: {order_by}")
        frame = frame.sort_values(order_by, kind="stable", ignore_index=True)
        order = frame[order_by].to_numpy()
        nominal_boundaries = [train_end, val_end]
        # Keep equal timestamps in the same partition.
        train_end, val_end = map(int, np.searchsorted(order, order[nominal_boundaries]))
        if not 0 < train_end < val_end < n:
            raise ValueError("Ordering column does not allow three nonempty partitions")
        order_metadata = {
            "history_unit": "incident-flow count",
            "split": f"ascending {order_by}; stable CSV order for timestamp ties",
            "nominal_split_boundaries": [0, *nominal_boundaries, n],
            "split_order_ranges": [[int(order[lo]), int(order[hi - 1])]
                                   for lo, hi in ((0, train_end), (train_end, val_end), (val_end, n))],
        }

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(frame[LABEL_COLUMN].astype(str)).astype(np.int64)
    raw_edge_features, feature_names = _numeric_edge_features(frame, exclude_features)

    scaler = StandardScaler()
    scaler.fit(raw_edge_features[:train_end])
    edge_features = scaler.transform(raw_edge_features).astype(np.float32)
    edge_features = np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)

    (
        src_profiles,
        dst_profiles,
        src_history,
        dst_history,
        src_direction,
        dst_direction,
    ) = _causal_profiles_and_history(frame, windows, history_size, progress)

    metadata = {
        "source": str(csv_path.resolve()),
        "sample_fraction": sample_fraction,
        "max_rows": max_rows,
        "random_state": random_state,
        "history_unit": "incident-flow count (source CSV has no timestamp)",
        "split": "chronological row order",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        **order_metadata,
    }
    if exclude_features:
        metadata["excluded_extra_features"] = sorted(set(exclude_features))
    prepared = PreparedTrafficData(
        edge_features=edge_features,
        src_profiles=src_profiles,
        dst_profiles=dst_profiles,
        src_history=src_history,
        dst_history=dst_history,
        src_history_direction=src_direction,
        dst_history_direction=dst_direction,
        labels=labels,
        class_names=label_encoder.classes_.tolist(),
        edge_feature_names=feature_names,
        windows=windows,
        train_end=train_end,
        val_end=val_end,
        metadata=metadata,
    )
    prepared.save(cache_path)
    progress(f"Saved prepared causal dataset to {cache_path}")
    return prepared


def make_collate_fn(data: PreparedTrafficData):
    """Create a vectorized batch gatherer for current and historical flows."""

    def history_tensors(
        batch_indices: np.ndarray,
        history_indices: np.ndarray,
        history_direction: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        selected = history_indices[batch_indices]
        mask = selected >= 0
        safe = selected.clip(min=0).astype(np.int64, copy=False)
        direction = history_direction[batch_indices].astype(np.int64, copy=False)

        historical_src = data.src_profiles[safe]
        historical_dst = data.dst_profiles[safe]
        # 0 = the historical edge left the current endpoint, so its neighbour
        # was the historical destination; 1 means the reverse.
        neighbour_profiles = np.where(
            (direction == 0)[..., None], historical_dst, historical_src
        )
        lag = batch_indices[:, None] - safe
        lag = np.log1p(np.maximum(lag, 0)).astype(np.float32) / 10.0
        lag[~mask] = 0.0
        return {
            "edge": torch.from_numpy(data.edge_features[safe]),
            "neighbour": torch.from_numpy(neighbour_profiles.astype(np.float32, copy=False)),
            "direction": torch.from_numpy(direction),
            "lag": torch.from_numpy(lag),
            "mask": torch.from_numpy(mask),
        }

    def collate(batch: list[int]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        indices = np.asarray(batch, dtype=np.int64)
        return {
            "index": torch.from_numpy(indices),
            "edge": torch.from_numpy(data.edge_features[indices]),
            "src_profile": torch.from_numpy(data.src_profiles[indices]),
            "dst_profile": torch.from_numpy(data.dst_profiles[indices]),
            "src_history": history_tensors(
                indices, data.src_history, data.src_history_direction
            ),
            "dst_history": history_tensors(
                indices, data.dst_history, data.dst_history_direction
            ),
            "label": torch.from_numpy(data.labels[indices]),
        }

    return collate
