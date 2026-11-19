"""Seven-class CIC V3 preparation with classwise end-time splits and caps."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data import PreparedTrafficData, _causal_profiles_and_history, _numeric_edge_features


END_TIME = "FLOW_END_MILLISECONDS"
TIMESTAMPS = ("FLOW_START_MILLISECONDS", END_TIME)


def attack_family(label: str) -> str:
    """Map original attack labels to the seven categories used in the paper."""
    key = re.sub(r"[^a-z0-9]", "", str(label).lower())
    if key == "benign":
        return "Benign"
    if key == "bot":
        return "Bot"
    if key.startswith("ddos"):
        return "DDoS"
    if key.startswith("dos"):
        return "DoS"
    if key in ("infiltration", "infilteration"):
        return "Infiltration"
    if key in ("bruteforce", "ftpbruteforce", "sshbruteforce"):
        return "BruteForce"
    if key in ("webattacks", "bruteforceweb", "bruteforcexss", "sqlinjection"):
        return "Web Attacks"
    raise ValueError(f"Unrecognised CIC attack label: {label!r}")


def select_rows(meta: pd.DataFrame, random_state: int = 42) -> list[np.ndarray]:
    """Return original CSV row indices for each sampled temporal partition."""
    labels = meta["Attack"].map({label: attack_family(label) for label in meta["Attack"].unique()})
    times = pd.to_numeric(meta[END_TIME], errors="raise").to_numpy()
    if not np.isfinite(times).all():
        raise ValueError("CIC end timestamps must be finite")
    partitions = [[], [], []]
    for label in sorted(labels.unique()):
        ids = np.flatnonzero((labels == label).to_numpy())
        ids = ids[np.argsort(times[ids], kind="stable")]
        ordered_times = times[ids]
        boundaries = [int(len(ids) * 0.6), int(len(ids) * 0.8)]
        boundaries = [int(np.searchsorted(ordered_times, ordered_times[b], side="left")) for b in boundaries]
        limits = (252000, 84000, 84000) if label == "Benign" else (60000, 20000, 20000)
        for split, (piece, limit) in enumerate(zip(np.split(ids, boundaries), limits)):
            if len(piece) > limit:
                piece = np.random.RandomState(random_state).choice(piece, size=limit, replace=False)
            partitions[split].append(piece)
    result = []
    for pieces in partitions:
        ids = np.concatenate(pieces)
        # End-time order, with original CSV row order as the tie breaker.
        ids = ids[np.lexsort((ids, times[ids]))]
        if len(ids) == 0:
            raise ValueError("CIC classwise splitting produced an empty partition; use more rows")
        result.append(ids)
    return result


def prepare_cic_dataset(csv_path, cache_dir, window=32, history_size=32,
                        max_rows=None, rebuild_cache=False, progress=print):
    csv_path, cache_dir = Path(csv_path), Path(cache_dir)
    if window <= 0 or history_size <= 0:
        raise ValueError("Window and history size must be positive")
    stat = csv_path.stat()
    config = dict(protocol="cic-v3-classwise-endtime-v1", path=str(csv_path.resolve()),
                  size=stat.st_size, mtime=stat.st_mtime_ns, window=window,
                  history_size=history_size, max_rows=max_rows, sampling_seed=42)
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"prepared_cic_{key}.npz"
    if cache.exists() and not rebuild_cache:
        return PreparedTrafficData.load(cache)
    progress("Reading CIC labels and end timestamps for classwise splitting")
    meta = pd.read_csv(csv_path, usecols=["Attack", END_TIME], nrows=max_rows,
                       dtype={"Attack": "category"})
    split_rows = select_rows(meta)
    selected = np.sort(np.concatenate(split_rows))
    del meta
    # A second, chunked pass reads only the selected rows' complete features.
    pieces, offset = [], 0
    for chunk in pd.read_csv(csv_path, chunksize=200000, nrows=max_rows, low_memory=False):
        lo, hi = np.searchsorted(selected, (offset, offset + len(chunk)))
        ids = selected[lo:hi]
        if len(ids):
            piece = chunk.iloc[ids - offset].copy()
            piece.index = ids
            pieces.append(piece)
        offset += len(chunk)
    sampled = pd.concat(pieces)
    sampled.columns = sampled.columns.str.strip()
    frames = [sampled.loc[ids].reset_index(drop=True) for ids in split_rows]
    frame = pd.concat(frames, ignore_index=True)
    train_end = len(frames[0])
    val_end = train_end + len(frames[1])
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(frame["Attack"].map(attack_family)).astype(np.int64)
    features, names = _numeric_edge_features(frame, exclude_features=TIMESTAMPS)
    scaler = StandardScaler().fit(features[:train_end])
    features = np.nan_to_num(scaler.transform(features).astype(np.float32), nan=0., posinf=0., neginf=0.)
    profiles = []
    for split, part in enumerate(frames):
        progress(f"Constructing independent causal histories for CIC partition {split + 1}/3")
        arrays = list(_causal_profiles_and_history(part, (window,), history_size, progress))
        # History indices must address the concatenated prepared arrays.
        shift = (0, train_end, val_end)[split]
        for position in (2, 3):
            valid = arrays[position] >= 0
            arrays[position][valid] += shift
        profiles.append(arrays)
    arrays = [np.concatenate([part[i] for part in profiles]) for i in range(6)]
    metadata = {**config, "class_mapping": "seven attack families", "split": "classwise end time 60/20/20; per-partition caps",
                "history_state": "independently initialised in each partition",
                "excluded_extra_features": list(TIMESTAMPS),
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
    data = PreparedTrafficData(features, *arrays, labels, label_encoder.classes_.tolist(),
                               names, (window,), train_end, val_end, metadata)
    data.save(cache)
    return data
