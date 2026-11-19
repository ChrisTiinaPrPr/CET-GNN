"""CET-GNN: causal edge-query temporal graph neural network."""

from .data import PreparedTrafficData, prepare_dataset
from .model import CETGraphSAGE

CETGNN = CETGraphSAGE

__all__ = ["CETGNN", "CETGraphSAGE", "PreparedTrafficData", "prepare_dataset"]
