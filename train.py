"""Train the final CET-GNN model or a paper ablation on either dataset."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from cet_graphsage.data import prepare_dataset
from cet_graphsage.training import fit_model


VARIANTS = {
    "A0": dict(use_profiles=False, use_history=False, use_lag_encoding=False, use_time_decay=False),
    "A1": dict(use_history=False, use_lag_encoding=False, use_time_decay=False),
    "A2": dict(use_lag_encoding=False, use_time_decay=False),
    "A3": dict(use_lag_encoding=True, use_time_decay=False),
    "A4": dict(use_lag_encoding=True, use_time_decay=True),
    "Q": dict(aggregation="decay_only", use_lag_encoding=True, use_time_decay=True),
    "U": dict(use_lag_encoding=True, use_time_decay=True, class_weighting=False),
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("unsw-v2", "cic-v3"), default="unsw-v2")
    parser.add_argument("--data", type=Path, help="Path to the downloaded dataset CSV")
    parser.add_argument("--output-dir", type=Path, help="New directory for this experiment")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--variant", choices=VARIANTS, default="A4")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 62, 82])
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--history-size", type=int, default=32)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--learned-decay", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--representation-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-backend", choices=("cpu", "device"), default="cpu",
                        help="Keep prepared arrays on CPU or on the model device")
    parser.add_argument("--max-rows", type=int, help="Read only the first N rows; smoke tests only")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if any(getattr(args, name) <= 0 for name in ("window", "history_size", "epochs", "patience", "batch_size", "hidden_dim", "representation_dim")):
        parser.error("Window, history length, epochs, patience, batch size and dimensions must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be distinct")
    return args


def main():
    args = arguments()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    name = "NF-UNSW-NB15-v2.csv" if args.dataset == "unsw-v2" else "NF-CICIDS2018-v3.csv"
    data_path = args.data or Path("data") / name
    output = args.output_dir or Path("outputs") / f"{args.dataset}_{args.variant}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    if args.dataset == "unsw-v2":
        data = prepare_dataset(data_path, args.cache_dir, sample_fraction=1.0,
                               max_rows=args.max_rows, windows=(args.window,),
                               history_size=args.history_size, rebuild_cache=args.rebuild_cache)
    else:
        from cet_graphsage.cic_data import prepare_cic_dataset

        data = prepare_cic_dataset(data_path, args.cache_dir, window=args.window,
                                   history_size=args.history_size, max_rows=args.max_rows,
                                   rebuild_cache=args.rebuild_cache)
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data": str(data_path), "device": str(device)}
    (output / "run_config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    options = dict(VARIANTS[args.variant])
    options["fixed_time_decay"] = None if args.learned_decay or not options.get("use_time_decay", True) else args.beta
    rows = []
    for seed in args.seeds:
        _, summary = fit_model(
            data, output / f"seed_{seed}", hidden_dim=args.hidden_dim,
            representation_dim=args.representation_dim, dropout=args.dropout,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            num_workers=0, seed=seed, device=device, data_backend=args.data_backend,
            run_metadata={"dataset": args.dataset, "variant": args.variant}, **options)
        rows.append({"seed": seed, **summary["test"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "all_runs_metrics.csv", index=False)
    frame.drop(columns="seed").agg(["mean", "std"]).to_csv(output / "metrics_mean_std.csv")
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
