from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    representation: torch.Tensor
    src_attention: torch.Tensor
    dst_attention: torch.Tensor


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class EdgeQueryHistoricalAggregator(nn.Module):
    """Aggregate endpoint history using the current flow as the query.

    Unlike mean aggregation, each historical message receives a learned,
    target-flow-specific weight. Fixed or learned non-negative decay penalises
    older records; masked empty histories return zero context.
    """

    def __init__(
        self,
        edge_dim: int,
        profile_dim: int,
        hidden_dim: int,
        dropout: float,
        direction_dim: int = 8,
        aggregation: str = "attention",
        use_time_decay: bool = True,
        use_lag_encoding: bool | None = None,
        fixed_time_decay: float | None = None,
    ):
        super().__init__()
        if aggregation not in {"attention", "mean", "decay_only"}:
            raise ValueError("aggregation must be 'attention', 'mean', or 'decay_only'")
        if aggregation == "decay_only" and not use_time_decay:
            raise ValueError("decay_only aggregation requires enabled time decay")
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.use_time_decay = use_time_decay
        # None preserves the original coupled switch for existing callers.
        self.use_lag_encoding = use_time_decay if use_lag_encoding is None else use_lag_encoding
        if fixed_time_decay is not None:
            if not use_time_decay or not math.isfinite(fixed_time_decay) or fixed_time_decay < 0:
                raise ValueError("fixed_time_decay requires enabled decay and a finite nonnegative coefficient")
        self.fixed_time_decay = fixed_time_decay
        self.direction_embedding = nn.Embedding(2, direction_dim)
        self.query_encoder = MLP(
            edge_dim + profile_dim, hidden_dim, hidden_dim, dropout
        )
        message_dim = edge_dim + profile_dim + direction_dim + 1
        self.key_encoder = MLP(message_dim, hidden_dim, hidden_dim, dropout)
        self.value_encoder = MLP(message_dim, hidden_dim, hidden_dim, dropout)
        self.raw_time_decay = nn.Parameter(torch.tensor(0.0))
        if fixed_time_decay is not None:
            self.raw_time_decay.requires_grad_(False)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        current_edge: torch.Tensor,
        endpoint_profile: torch.Tensor,
        history: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        historical_edge = history["edge"]
        neighbour_profile = history["neighbour"]
        direction = history["direction"]
        lag = history["lag"]
        mask = history["mask"]

        # Message encoding and the explicit attention penalty are independent.
        message_lag = lag if self.use_lag_encoding else torch.zeros_like(lag)

        query = self.query_encoder(
            torch.cat([current_edge, endpoint_profile], dim=-1)
        )
        direction_features = self.direction_embedding(direction)
        messages = torch.cat(
            [
                historical_edge,
                neighbour_profile,
                direction_features,
                message_lag.unsqueeze(-1),
            ],
            dim=-1,
        )
        values = self.value_encoder(messages)

        if self.aggregation == "mean":
            attention = mask.to(values.dtype)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            keys = self.key_encoder(messages)
            scores = torch.einsum("bd,bkd->bk", query, keys) / math.sqrt(
                self.hidden_dim
            )
            if self.aggregation == "decay_only":
                # Consume the same query/key dropout calls as full attention,
                # but remove content matching from the aggregation weights.
                scores = torch.zeros_like(scores)
            if self.use_time_decay:
                beta = F.softplus(self.raw_time_decay) if self.fixed_time_decay is None else self.fixed_time_decay
                scores = scores - beta * lag
            scores = scores.masked_fill(~mask, -1e9)
            attention = torch.softmax(scores, dim=-1)
            attention = attention * mask.to(attention.dtype)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
                1e-8
            )

        context = torch.einsum("bk,bkd->bd", attention, values)
        context = self.output_norm(context)
        has_history = mask.any(dim=-1, keepdim=True)
        context = torch.where(has_history, context, torch.zeros_like(context))
        return context, attention


class CETGraphSAGE(nn.Module):
    """CET-GNN flow classifier; the class name preserves checkpoint compatibility."""

    def __init__(
        self,
        edge_dim: int,
        profile_dim: int,
        hidden_dim: int,
        representation_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        use_profiles: bool = True,
        use_history: bool = True,
        aggregation: str = "attention",
        use_time_decay: bool = True,
        use_lag_encoding: bool | None = None,
        fixed_time_decay: float | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_profiles = use_profiles
        self.use_history = use_history

        self.edge_encoder = MLP(edge_dim, hidden_dim, hidden_dim, dropout)
        self.profile_encoder = MLP(profile_dim, hidden_dim, hidden_dim, dropout)
        self.history_aggregator = EdgeQueryHistoricalAggregator(
            edge_dim=edge_dim,
            profile_dim=profile_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            aggregation=aggregation,
            use_time_decay=use_time_decay,
            use_lag_encoding=use_lag_encoding,
            fixed_time_decay=fixed_time_decay,
        )

        fusion_dim = hidden_dim * 5
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
        )
        self.classifier = nn.Linear(representation_dim, num_classes)

    def forward(self, batch: dict) -> ModelOutput:
        current_edge = batch["edge"]
        src_profile = batch["src_profile"]
        dst_profile = batch["dst_profile"]

        if not self.use_profiles:
            src_profile = torch.zeros_like(src_profile)
            dst_profile = torch.zeros_like(dst_profile)

        edge_hidden = self.edge_encoder(current_edge)
        src_hidden = self.profile_encoder(src_profile)
        dst_hidden = self.profile_encoder(dst_profile)
        if self.use_history:
            src_history = batch["src_history"]
            dst_history = batch["dst_history"]
            if not self.use_profiles:
                src_history = {
                    **src_history,
                    "neighbour": torch.zeros_like(src_history["neighbour"]),
                }
                dst_history = {
                    **dst_history,
                    "neighbour": torch.zeros_like(dst_history["neighbour"]),
                }
            src_context, src_attention = self.history_aggregator(
                current_edge, src_profile, src_history
            )
            dst_context, dst_attention = self.history_aggregator(
                current_edge, dst_profile, dst_history
            )
        else:
            src_context = torch.zeros_like(src_hidden)
            dst_context = torch.zeros_like(dst_hidden)
            src_attention = torch.zeros_like(batch["src_history"]["lag"])
            dst_attention = torch.zeros_like(batch["dst_history"]["lag"])

        if not self.use_profiles:
            src_hidden = torch.zeros_like(src_hidden)
            dst_hidden = torch.zeros_like(dst_hidden)

        representation = self.fusion(
            torch.cat(
                [src_hidden, dst_hidden, src_context, dst_context, edge_hidden], dim=-1
            )
        )
        logits = self.classifier(representation)
        return ModelOutput(
            logits=logits,
            representation=representation,
            src_attention=src_attention,
            dst_attention=dst_attention,
        )

    def loss(
        self,
        output: ModelOutput,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        classification = F.cross_entropy(output.logits, labels, weight=class_weights)
        return classification, {"classification": float(classification.detach())}
