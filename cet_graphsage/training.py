from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .data import IndexDataset, PreparedTrafficData, make_collate_fn
from .model import CETGraphSAGE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def make_loader(
    data: PreparedTrafficData,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        IndexDataset(indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(data),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def class_balancing_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Mild inverse-square-root weights are more stable than raw inverse counts."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    if present.any():
        weights[present] /= weights[present].mean()
    return torch.from_numpy(weights)


def train_one_epoch(
    model: CETGraphSAGE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "classification": 0.0}
    samples = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, parts = model.loss(
            output,
            batch["label"],
            class_weights=class_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        count = int(batch["label"].shape[0])
        samples += count
        totals["loss"] += float(loss.detach()) * count
        for name, value in parts.items():
            totals[name] += value * count
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: CETGraphSAGE,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        output = model(batch)
        predictions.append(output.logits.argmax(dim=-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    if not labels:
        return {}, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(model.num_classes), average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=np.arange(model.num_classes), average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=np.arange(model.num_classes), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    return metrics, y_true, y_pred


def save_evaluation(
    artifacts_dir: Path,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    class_ids = np.arange(len(class_names))
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    (artifacts_dir / f"{split_name}_classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        artifacts_dir / f"{split_name}_confusion_matrix.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "true_id": y_true,
            "predicted_id": y_pred,
            "true_label": [class_names[index] for index in y_true],
            "predicted_label": [class_names[index] for index in y_pred],
        }
    ).to_csv(artifacts_dir / f"{split_name}_predictions.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        figure_size = max(8, len(class_names))
        plt.figure(figsize=(figure_size, figure_size))
        sns.heatmap(
            matrix,
            cmap="Blues",
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"CET-GNN {split_name} confusion matrix")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(artifacts_dir / f"{split_name}_confusion_matrix.png", dpi=180)
        plt.close()
    except ImportError:
        pass


def fit_model(
    data: PreparedTrafficData,
    artifacts_dir: str | Path,
    hidden_dim: int,
    representation_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    use_profiles: bool = True,
    use_history: bool = True,
    aggregation: str = "attention",
    use_time_decay: bool = True,
    run_metadata: dict | None = None,
    use_lag_encoding: bool = True,
    fixed_time_decay: float | None = 2.0,
    class_weighting: bool = True,
    data_backend: str = "cpu",
) -> tuple[CETGraphSAGE, dict]:
    seed_everything(seed)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_indices, val_indices, test_indices = data.split_indices()
    if data_backend == "device":
        from .gpu_data import GPUPreparedTraffic

        resident = GPUPreparedTraffic(data, device)
        loaders = [resident.loader(indices, batch_size, shuffle, data.windows, data.src_history.shape[1])
                   for indices, shuffle in ((train_indices, True), (val_indices, False), (test_indices, False))]
        train_loader, val_loader, test_loader = loaders
    elif data_backend == "cpu":
        train_loader = make_loader(data, train_indices, batch_size, True, num_workers)
        val_loader = make_loader(data, val_indices, batch_size, False, num_workers)
        test_loader = make_loader(data, test_indices, batch_size, False, num_workers)
    else:
        raise ValueError("data_backend must be 'cpu' or 'device'")

    model_config = {
        "edge_dim": data.edge_dim,
        "profile_dim": data.profile_dim,
        "hidden_dim": hidden_dim,
        "representation_dim": representation_dim,
        "num_classes": data.num_classes,
        "dropout": dropout,
        "use_profiles": use_profiles,
        "use_history": use_history,
        "aggregation": aggregation,
        "use_time_decay": use_time_decay,
        "use_lag_encoding": use_lag_encoding,
        "fixed_time_decay": fixed_time_decay,
    }
    model = CETGraphSAGE(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    class_weights = class_balancing_weights(
        data.labels[train_indices], data.num_classes
    ).to(device) if class_weighting else None

    best_score = -1.0
    epochs_without_improvement = 0
    checkpoint_path = artifacts_dir / "best_cet_graphsage.pt"
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            class_weights,
            device,
        )
        val_metrics, _, _ = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(
            f"Epoch {epoch:03d} | loss={train_metrics['loss']:.4f} "
            f"| val_macro_f1={val_metrics.get('macro_f1', 0.0):.4f} "
            f"| val_acc={val_metrics.get('accuracy', 0.0):.4f}"
        )

        score = val_metrics.get("macro_f1", 0.0)
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "class_names": data.class_names,
                    "edge_feature_names": data.edge_feature_names,
                    "windows": data.windows,
                    "metadata": data.metadata,
                    "best_val_macro_f1": best_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    pd.DataFrame(history).to_csv(artifacts_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    val_metrics, val_true, val_pred = evaluate(model, val_loader, device)
    test_metrics, test_true, test_pred = evaluate(model, test_loader, device)
    for metrics, truth, prediction in ((val_metrics, val_true, val_pred), (test_metrics, test_true, test_pred)):
        report = classification_report(truth, prediction, labels=np.arange(data.num_classes),
                                       target_names=data.class_names, output_dict=True, zero_division=0)
        attack_classes = [name for name in data.class_names if name.lower() != "benign"]
        metrics["attack_macro_f1"] = float(np.mean([report[name]["f1-score"] for name in attack_classes])) if attack_classes else 0.0
        if all(name in data.class_names for name in ("Analysis", "Backdoor", "Shellcode", "Worms")):
            metrics["tail_f1"] = float(np.mean([report[name]["f1-score"] for name in ("Analysis", "Backdoor", "Shellcode", "Worms")]))
        if "Benign" in data.class_names:
            benign = truth == data.class_names.index("Benign")
            metrics["benign_false_positive_rate"] = float(np.mean(prediction[benign] != truth[benign])) if benign.any() else None
    save_evaluation(artifacts_dir, "validation", val_true, val_pred, data.class_names)
    save_evaluation(artifacts_dir, "test", test_true, test_pred, data.class_names)
    summary = {
        "best_validation_macro_f1": best_score,
        "validation": val_metrics,
        "test": test_metrics,
        "num_samples": data.num_samples,
        "split_sizes": {
            "train": len(train_indices),
            "validation": len(val_indices),
            "test": len(test_indices),
        },
        "class_names": data.class_names,
        "windows": data.windows,
        "history_size": int(data.src_history.shape[1]),
        "device": str(device),
        "model_config": model_config,
        "seed": seed,
        "class_weighting": "inverse_sqrt" if class_weighting else "none",
        "data_backend": data_backend,
        "run_metadata": run_metadata or {},
    }
    (artifacts_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return model, summary
