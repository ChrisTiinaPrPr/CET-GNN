"""Prepare, train, and replay CET-GNN on GridStratLLM traffic."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix

from .gpu_data import GPUPreparedTraffic
from .gridstratllm_data import FLOW_FEATURES, HISTORY_SIZE, SPLITS, WINDOW, prepare, prepare_model_data, read_flows
from .io_utils import replace_with_retry
from .model import CETGraphSAGE
from .training import class_balancing_weights, seed_everything, train_one_epoch

BATCH_SIZE = 512
MIN_EPOCHS = 20
PATIENCE = 16
THRESHOLDS = np.arange(1, 100) / 100
MODEL_CONFIG = {
    "edge_dim": len(FLOW_FEATURES),
    "profile_dim": 9,
    "hidden_dim": 128,
    "representation_dim": 128,
    "num_classes": 2,
    "dropout": 0.3,
    "fixed_time_decay": 2.0,
    "use_lag_encoding": True,
}


def validation_metrics(labels, scores):
    predicted = scores[:, None] >= THRESHOLDS
    positive = labels[:, None] == 1
    tp = (positive & predicted).sum(0)
    fn = (positive & ~predicted).sum(0)
    fp = (~positive & predicted).sum(0)
    tn = (~positive & ~predicted).sum(0)
    f1 = tp / (2 * tp + fn + fp) + tn / (2 * tn + fn + fp)
    maxima = np.flatnonzero(np.isclose(f1, f1.max(), atol=1e-12, rtol=0))
    k = maxima[np.argmin(np.abs(THRESHOLDS[maxima] - 0.5))]
    return {
        "macro_f1": float(f1[k]),
        "threshold": float(THRESHOLDS[k]),
        "attack_ap": float(average_precision_score(labels, scores)),
        "attack_recall": float(tp[k] / (tp[k] + fn[k])),
        "background_fpr": float(fp[k] / (fp[k] + tn[k])),
    }


@torch.no_grad()
def probability(model, loader):
    model.eval()
    scores = np.concatenate([model(batch).logits.softmax(1)[:, 1].cpu().numpy() for batch in loader])
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite attack scores")
    return scores


def save_predictions(frame, indices, scores, threshold, path):
    rows = frame.iloc[indices][["row_id", "capture", "split", "window_end", "label"]].copy()
    rows["attack_score"] = scores
    rows["predicted_attack"] = scores >= threshold
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    rows.to_csv(pending, index=False)
    replace_with_retry(pending, path)
    return rows


def test_metrics(rows):
    tn, fp, fn, tp = confusion_matrix(rows["label"], rows["predicted_attack"], labels=[0, 1]).ravel().tolist()
    attack_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    background_f1 = 2 * tn / max(1, 2 * tn + fp + fn)
    return {
        "rows": len(rows), "positive": tp + fn, "background": tn + fp,
        "macro_f1": (attack_f1 + background_f1) / 2,
        "attack_f1": attack_f1,
        "attack_recall": tp / (tp + fn) if tp + fn else None,
        "candidate_background_fpr": fp / (tn + fp) if tn + fp else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def make_loader(resident, indices, shuffle=False):
    return resident.loader(indices, BATCH_SIZE, shuffle, WINDOW, HISTORY_SIZE)


def fit(frame, data, resident, scaling, targets, seed, max_epochs, folder):
    folder.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    model = CETGraphSAGE(**MODEL_CONFIG).to(resident.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    weights = class_balancing_weights(data.labels[targets["train"]], 2).to(resident.device)
    loaders = {split: make_loader(resident, rows, shuffle=split == "train") for split, rows in targets.items()}
    y_validation = data.labels[targets["validation"]]
    checkpoint = folder / "best_model.pt"
    best = None
    best_epoch = stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        loss = train_one_epoch(model, loaders["train"], optimizer, weights, resident.device)["loss"]
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
        metrics = validation_metrics(y_validation, probability(model, loaders["validation"]))
        history.append({"epoch": epoch, "loss": loss, **{f"validation_{k}": v for k, v in metrics.items()}})
        if best is None or metrics["macro_f1"] > best["macro_f1"]:
            best, best_epoch, stale = metrics, epoch, 0
            pending = checkpoint.with_suffix(".pt.tmp")
            torch.save({
                "model": model.state_dict(), "configuration": MODEL_CONFIG, "scaling": scaling,
                "threshold": best["threshold"], "epoch": best_epoch, "seed": seed,
            }, pending)
            replace_with_retry(pending, checkpoint)
        else:
            stale += 1
        print(f"Seed {seed}, epoch {epoch}: validation Macro-F1 {metrics['macro_f1']:.4f}", flush=True)
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    selected = torch.load(checkpoint, map_location=resident.device, weights_only=True)
    model.load_state_dict(selected["model"])
    scores = probability(model, loaders["test"])
    rows = save_predictions(frame, targets["test"], scores, selected["threshold"], folder / "predictions.csv")
    report = {
        "seed": seed, "epoch": best_epoch, "epochs_trained": epoch, "threshold": best["threshold"],
        "validation": best, "test": test_metrics(rows),
        "by_capture": {name: test_metrics(group) for name, group in rows.groupby("capture", sort=False)},
    }
    pending = folder / "metrics.json.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(pending, folder / "metrics.json")
    return report


def train(flows, output_dir, seeds, max_epochs, device):
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Training seeds must be unique unsigned 32-bit integers")
    destinations = [output_dir / f"seed_{seed}" for seed in seeds]
    if any(path.exists() for path in destinations):
        raise FileExistsError("At least one seed directory already exists. Choose a new output directory.")
    frame = read_flows(flows)
    frame = frame.loc[frame["split"].isin(SPLITS)].reset_index(drop=True)
    targets = {
        split: np.flatnonzero((frame["split"] == split) & (frame["label"] >= 0)) for split in SPLITS
    }
    for split, rows in targets.items():
        if set(frame["label"].iloc[rows]) != {0, 1}:
            raise ValueError(f"The {split} split needs both attack and candidate-background labels")
    data, scaling = prepare_model_data(frame, train_rows=targets["train"])
    resident = GPUPreparedTraffic(data, device)
    for seed, path in zip(seeds, destinations):
        metrics = fit(frame, data, resident, scaling, targets, seed, max_epochs, path)
        print(f"Seed {seed}: test Macro-F1 {metrics['test']['macro_f1']:.4f}, saved to {path}", flush=True)


def predict(flows, checkpoint, output, split, labelled_only, device):
    if output.resolve() in (flows.resolve(), checkpoint.resolve()):
        raise ValueError("The output must differ from the input flows and checkpoint")
    frame = read_flows(flows)
    if split is not None:
        frame = frame.loc[frame["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No flow rows for split {split}")
    indices = np.flatnonzero(frame["label"] >= 0) if labelled_only else frame.index.to_numpy()
    if not len(indices):
        raise ValueError("No labelled rows to predict")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    threshold = float(state["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid checkpoint threshold")
    data, _ = prepare_model_data(frame, scaling=state["scaling"])
    resident = GPUPreparedTraffic(data, device)
    model = CETGraphSAGE(**state["configuration"]).to(device)
    model.load_state_dict(state["model"])
    scores = probability(model, make_loader(resident, indices))
    save_predictions(frame, indices, scores, threshold, output)
    print(f"Saved {len(indices)} predictions to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Build flow rows from the official DuckDB file")
    preparation.add_argument("--database", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)

    training = commands.add_parser("train", help="Train the detector and save its best validation checkpoint")
    training.add_argument("--flows", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, default=Path("outputs/gridstratllm"))
    training.add_argument("--seeds", nargs="+", type=int, default=[42, 62, 82])
    training.add_argument("--max-epochs", type=int, default=60)

    prediction = commands.add_parser("predict", help="Score ordered flow rows with a trained checkpoint")
    prediction.add_argument("--flows", type=Path, required=True)
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--split", help="Optional split to score, such as test")
    prediction.add_argument("--labelled-only", action="store_true")
    for command in (training, prediction):
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.database, args.output)
        print(f"Saved {args.output}")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    if args.command == "train":
        if args.max_epochs < 1:
            parser.error("--max-epochs must be positive")
        train(args.flows, args.output_dir, args.seeds, args.max_epochs, device)
    else:
        predict(args.flows, args.checkpoint, args.output, args.split, args.labelled_only, device)


if __name__ == "__main__":
    main()
