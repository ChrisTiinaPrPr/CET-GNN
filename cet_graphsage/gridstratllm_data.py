"""Prepare GridStratLLM flow statistics and capture-local, causal histories."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import IP_COLUMNS, PreparedTrafficData, _causal_profiles_and_history, _numeric_edge_features
from .io_utils import replace_with_retry

CAPTURES = {
    "20260228_multi_openai": "train",
    "20260301_multi_google": "train",
    "20260301_multi_sonnet": "validation",
    "20260302_multi_minimax": "test",
    "20260303_multi_sonnet": "test",
}
# Number of labelled attack and candidate-background segments in the released split.
SPLIT_COUNTS = {"train": (1602, 36895), "validation": (574, 2510), "test": (2219, 13564)}
SPLITS = tuple(SPLIT_COUNTS)
CLASS_NAMES = ["Background", "Attack"]
WINDOW = HISTORY_SIZE = 32
FLOW_FEATURES = [
    "OUT_PKTS", "IN_PKTS", "OUT_BYTES", "IN_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "DURATION_OUT", "DURATION_IN", "FLOW_DURATION_MILLISECONDS",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "PACKET_SIZE_MIN", "PACKET_SIZE_MAX", "PACKET_SIZE_MEAN", "PACKET_SIZE_STD",
]
HEADER_FEATURES = ["MIN_TTL", "MAX_TTL", "RETRANSMITTED_OUT_PKTS", "RETRANSMITTED_IN_PKTS"]


FLOW_SQL = """
WITH matches AS (
    SELECT packet_id, attack_session_id FROM attack_datamod_history WHERE state = 'match'
), events AS (
    SELECT s.dataset_id, s.start_ts,
           greatest(s.planned_end_ts, e.stop_ts, h.last_match) AS end_ts
    FROM attack_session s
    LEFT JOIN (
        SELECT attack_session_id, max(exec_ts) AS stop_ts FROM exec_steps WHERE kind = 'stop' GROUP BY 1
    ) e ON e.attack_session_id = s.id
    LEFT JOIN (
        SELECT attack_session_id, max(ts) AS last_match
        FROM attack_datamod_history WHERE state = 'match' GROUP BY 1
    ) h ON h.attack_session_id = s.id
), occupied AS (
    SELECT DISTINCT dataset_id, bucket
    FROM events CROSS JOIN LATERAL generate_series(
        time_bucket(INTERVAL '5 seconds', start_ts - INTERVAL '10 seconds'),
        time_bucket(INTERVAL '5 seconds', end_ts + INTERVAL '10 seconds'), INTERVAL '5 seconds'
    ) AS slots(bucket)
), packet_rows AS (
    SELECT id, dataset_id, community_id, ts, size, src_ip, dst_ip,
           (src_ip, coalesce(src_port, 0)) <= (dst_ip, coalesce(dst_port, 0)) AS forward,
           time_bucket(INTERVAL '5 seconds', ts) AS bucket
    FROM packets
    WHERE transport = 'tcp' AND community_id IS NOT NULL AND src_ip IS NOT NULL AND dst_ip IS NOT NULL
), flow_rows AS (
    SELECT dataset_id, bucket, community_id,
           arg_min(src_ip, (ts, id)) AS observed_src,
           arg_min(dst_ip, (ts, id)) AS observed_dst,
           arg_min(forward, (ts, id)) AS first_forward,
           count(*) FILTER (WHERE forward) AS forward_packets,
           count(*) FILTER (WHERE NOT forward) AS reverse_packets,
           coalesce(sum(size) FILTER (WHERE forward), 0) AS forward_bytes,
           coalesce(sum(size) FILTER (WHERE NOT forward), 0) AS reverse_bytes,
           coalesce(epoch(max(ts) FILTER (WHERE forward) - min(ts) FILTER (WHERE forward)), 0) AS forward_seconds,
           coalesce(epoch(max(ts) FILTER (WHERE NOT forward) - min(ts) FILTER (WHERE NOT forward)), 0) AS reverse_seconds,
           epoch(max(ts) - min(ts)) AS seconds,
           count(*) FILTER (WHERE size <= 128) AS small_packets,
           count(*) FILTER (WHERE size BETWEEN 1024 AND 1514) AS large_packets,
           min(size) AS min_size, max(size) AS max_size, avg(size) AS mean_size,
           coalesce(stddev_pop(size), 0) AS size_std,
           count(attack_session_id) AS matched_packets
    FROM packet_rows LEFT JOIN matches ON matches.packet_id = packet_rows.id
    GROUP BY dataset_id, bucket, community_id
)
SELECT regexp_extract(d.relative_path, '[^/]+$') AS capture, epoch(w.bucket) + 5 AS window_end,
       w.* EXCLUDE (dataset_id, bucket),
       CASE WHEN w.matched_packets > 0 THEN 1 WHEN occupied.bucket IS NOT NULL THEN -1 ELSE 0 END AS label
FROM flow_rows w JOIN dataset d ON d.id = w.dataset_id
LEFT JOIN occupied ON occupied.dataset_id = w.dataset_id AND occupied.bucket = w.bucket
"""


HEADER_SQL = """
WITH header_bytes AS (
    SELECT p.id, p.community_id, p.ts, p.src_ip, p.dst_ip, p.size,
           (p.src_ip, coalesce(p.src_port, 0)) <= (p.dst_ip, coalesce(p.dst_port, 0)) AS forward,
           time_bucket(INTERVAL '5 seconds', p.ts) AS bucket,
           substr(hex(p.raw_packet), 1, 188) AS header,
           octet_length(p.raw_packet) AS captured_size
    FROM packets p JOIN dataset d ON d.id = p.dataset_id
    WHERE d.relative_path LIKE ? AND p.transport = 'tcp'
      AND p.community_id IS NOT NULL AND p.src_ip IS NOT NULL AND p.dst_ip IS NOT NULL
), ip AS (
    SELECT *,
           CASE WHEN substr(header, 25, 4) = '0800' AND substr(header, 29, 1) = '4'
                THEN try_cast('0x' || substr(header, 45, 2) AS USMALLINT) END AS ttl,
           14 + 4 * try_cast('0x' || substr(header, 30, 1) AS UINTEGER) AS tcp_offset,
           try_cast('0x' || substr(header, 33, 4) AS INTEGER) AS ip_size,
           try_cast('0x' || substr(header, 41, 4) AS UINTEGER) & 16383 AS fragment_bits
    FROM header_bytes
), segments AS MATERIALIZED (
    SELECT id, community_id, ts, src_ip, dst_ip, size, forward, bucket, ttl,
           CASE WHEN ttl IS NOT NULL AND substr(header, 47, 2) = '06' AND fragment_bits = 0
                     AND tcp_offset >= 34 AND captured_size >= tcp_offset + 20
                THEN try_cast('0x' || substr(header, 2 * (tcp_offset + 4) + 1, 8) AS UBIGINT) END AS seq,
           ip_size - (tcp_offset - 14)
              - 4 * try_cast('0x' || substr(header, 2 * (tcp_offset + 12) + 1, 1) AS INTEGER) AS data_size
    FROM ip
), repeats AS (
    SELECT id, ts - lag(ts) OVER (
        PARTITION BY community_id, forward, seq, data_size ORDER BY ts, id
    ) AS gap
    FROM segments WHERE seq IS NOT NULL AND data_size > 0
), candidates AS (
    SELECT id FROM repeats WHERE gap BETWEEN INTERVAL '10 milliseconds' AND INTERVAL '30 seconds'
)
SELECT epoch(bucket) + 5 AS window_end, community_id,
       arg_min(src_ip, (ts, s.id)) AS observed_src, arg_min(dst_ip, (ts, s.id)) AS observed_dst,
       arg_min(forward, (ts, s.id)) AS first_forward,
       min(ttl) AS min_ttl, max(ttl) AS max_ttl, count(*) FILTER (WHERE ttl IS NULL) AS missing_ttl,
       count(*) FILTER (WHERE forward) AS forward_packets, count(*) FILTER (WHERE NOT forward) AS reverse_packets,
       coalesce(sum(size) FILTER (WHERE forward), 0) AS forward_bytes,
       coalesce(sum(size) FILTER (WHERE NOT forward), 0) AS reverse_bytes,
       count(*) FILTER (WHERE forward AND c.id IS NOT NULL) AS forward_repeats,
       count(*) FILTER (WHERE NOT forward AND c.id IS NOT NULL) AS reverse_repeats
FROM segments s LEFT JOIN candidates c ON s.id = c.id
GROUP BY bucket, community_id ORDER BY bucket, community_id
"""


def directional_features(raw):
    first = raw.groupby(["capture", "community_id"], sort=False)[
        ["observed_src", "observed_dst", "first_forward"]
    ].transform("first")
    features = {}
    for suffix, forward, reverse in [
        ("PKTS", "forward_packets", "reverse_packets"),
        ("BYTES", "forward_bytes", "reverse_bytes"),
        ("SECONDS", "forward_seconds", "reverse_seconds"),
    ]:
        features[f"OUT_{suffix}"] = np.where(first.first_forward, raw[forward], raw[reverse])
        features[f"IN_{suffix}"] = np.where(first.first_forward, raw[reverse], raw[forward])
    out_seconds = features.pop("OUT_SECONDS")
    in_seconds = features.pop("IN_SECONDS")
    for name, direction, seconds in [
        ("SRC_TO_DST_AVG_THROUGHPUT", "OUT", out_seconds),
        ("DST_TO_SRC_AVG_THROUGHPUT", "IN", in_seconds),
    ]:
        features[name] = np.divide(8 * features[f"{direction}_BYTES"], seconds,
                                   out=np.zeros(len(raw)), where=seconds > 0)
    features.update(
        DURATION_OUT=out_seconds * 1000,
        DURATION_IN=in_seconds * 1000,
        FLOW_DURATION_MILLISECONDS=raw.seconds * 1000,
        NUM_PKTS_UP_TO_128_BYTES=raw.small_packets,
        NUM_PKTS_1024_TO_1514_BYTES=raw.large_packets,
        PACKET_SIZE_MIN=raw.min_size,
        PACKET_SIZE_MAX=raw.max_size,
        PACKET_SIZE_MEAN=raw.mean_size,
        PACKET_SIZE_STD=raw.size_std,
    )
    frame = raw[["capture", "split", "window_end", "label"]].copy()
    frame[["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]] = first[["observed_src", "observed_dst"]].to_numpy()
    names = list(features)
    frame[names] = pd.DataFrame(features, index=frame.index).astype("float32")
    return frame


def align_headers(headers, expected):
    first = headers.groupby("community_id", sort=False)[
        ["observed_src", "observed_dst", "first_forward"]
    ].transform("first")
    if not (
        np.array_equal(headers.window_end, expected.window_end)
        and np.array_equal(first.observed_src, expected.IPV4_SRC_ADDR)
        and np.array_equal(first.observed_dst, expected.IPV4_DST_ADDR)
    ):
        raise ValueError("Flow keys or direction differ from the aggregated flows")
    for name, forward, reverse in [
        ("OUT_PKTS", "forward_packets", "reverse_packets"),
        ("IN_PKTS", "reverse_packets", "forward_packets"),
        ("OUT_BYTES", "forward_bytes", "reverse_bytes"),
        ("IN_BYTES", "reverse_bytes", "forward_bytes"),
    ]:
        if not np.array_equal(np.where(first.first_forward, headers[forward], headers[reverse]), expected[name]):
            raise ValueError(f"Prepared flow row mismatch for {name}")
    if headers.min_ttl.isna().any() or headers.max_ttl.isna().any():
        raise ValueError("At least one flow has no observable TTL")
    return pd.DataFrame({
        "MIN_TTL": headers.min_ttl,
        "MAX_TTL": headers.max_ttl,
        "RETRANSMITTED_OUT_PKTS": np.where(first.first_forward, headers.forward_repeats, headers.reverse_repeats),
        "RETRANSMITTED_IN_PKTS": np.where(first.first_forward, headers.reverse_repeats, headers.forward_repeats),
    }).set_index(expected.index)


def prepare(database: Path, output: Path) -> None:
    """Convert the released DuckDB archive to the paper's flow table."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Preparation requires DuckDB. Install requirements-gridstratllm.txt.") from exc

    if not database.is_file():
        raise FileNotFoundError(database)
    if output.resolve() == database.resolve():
        raise ValueError("The output must differ from the input database")
    output.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database), read_only=True, config={
        "threads": 4, "memory_limit": "4GB",
        "temp_directory": str(output.parent / "gridstratllm_sql_spill"),
    }) as db:
        raw = db.execute(FLOW_SQL).fetchdf()
        raw = raw.loc[raw["capture"].isin(CAPTURES)].copy()
        raw["split"] = raw["capture"].map(CAPTURES)
        raw["capture_order"] = raw["capture"].map({name: i for i, name in enumerate(CAPTURES)})
        raw.sort_values(["capture_order", "window_end", "community_id"], ignore_index=True, inplace=True)
        frame = directional_features(raw)
        features = []
        for capture, rows in frame.groupby("capture", sort=False):
            headers = db.execute(HEADER_SQL, ["%/" + capture]).fetchdf()
            expected_keys = raw.loc[rows.index, "community_id"].to_numpy()
            if not np.array_equal(headers["community_id"].to_numpy(), expected_keys):
                raise ValueError(f"Connection keys do not match for {capture}")
            features.append(align_headers(headers, rows))
        if features:
            frame[HEADER_FEATURES] = pd.concat(features)

    for split, (positive, background) in SPLIT_COUNTS.items():
        labels = frame.loc[frame["split"] == split, "label"].to_numpy()
        if (np.count_nonzero(labels == 1), np.count_nonzero(labels == 0)) != (positive, background):
            raise ValueError(f"Unexpected labelled segment counts in {split}. Check the dataset release.")
    if not np.isfinite(frame[FLOW_FEATURES + HEADER_FEATURES].to_numpy()).all():
        raise ValueError("The prepared flow table contains nonfinite statistics")
    frame.index.name = "row_id"
    temporary = output.with_name(output.name + ".tmp")
    frame.to_csv(temporary, compression="gzip" if output.suffix == ".gz" else None)
    replace_with_retry(temporary, output)


def read_flows(path: Path) -> pd.DataFrame:
    """Read capture-ordered rows and retain their row identifiers for predictions."""
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"capture", "window_end", *IP_COLUMNS, *FLOW_FEATURES, *HEADER_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing flow columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("The flow table is empty")
    if "row_id" not in frame:
        frame["row_id"] = frame.index
    if not frame["row_id"].is_unique:
        raise ValueError("Flow row identifiers must be unique")
    if "label" not in frame:
        frame["label"] = -1
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([-1, 0, 1]).all():
        raise ValueError("Flow labels must be -1, 0, or 1")
    frame["label"] = labels.astype(np.int64)
    if "split" not in frame:
        frame["split"] = "inference"
    frame["window_end"] = pd.to_numeric(frame["window_end"], errors="raise")
    if not np.isfinite(frame["window_end"]).all() or (frame[["capture", *IP_COLUMNS]] == "").any().any():
        raise ValueError("Flows need finite window times and nonempty capture and endpoint identifiers")
    # Preserve the prepared connection-key order for rows ending in the same window.
    groups = []
    for capture, rows in frame.groupby("capture", sort=False):
        if rows["split"].nunique() != 1:
            raise ValueError(f"Capture {capture} cannot span multiple splits")
        groups.append(rows.sort_values("window_end", kind="stable"))
    return pd.concat(groups, ignore_index=True)


def prepare_model_data(frame: pd.DataFrame, *, train_rows=None, scaling=None):
    """Create the model inputs and return the training-only feature scaling."""
    features, names = _numeric_edge_features(frame[FLOW_FEATURES])
    if scaling is None:
        if train_rows is None or not len(train_rows):
            raise ValueError("Training rows are required to fit the scaler")
        scaler = StandardScaler().fit(features[train_rows])
        scaling = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    else:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(scaling["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(scaling["scale"], dtype=np.float64)
        scaler.n_features_in_ = len(names)
        if (scaler.mean_.shape != (len(names),) or scaler.scale_.shape != (len(names),)
                or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all()
                or (scaler.scale_ <= 0).any()):
            raise ValueError("Invalid checkpoint feature scaling")
    features = scaler.transform(features)
    parts = []
    for _, rows in frame.groupby("capture", sort=False):
        arrays = _causal_profiles_and_history(rows, (WINDOW,), HISTORY_SIZE, lambda _: None)
        for indices in arrays[2:4]:
            indices[indices >= 0] += int(rows.index[0])
        parts.append(arrays)
    history = tuple(np.concatenate(fields) for fields in zip(*parts))
    return PreparedTrafficData(
        features, *history, frame["label"].to_numpy(dtype=np.int64), CLASS_NAMES, names,
        (WINDOW,), 0, 0, {},
    ), scaling
