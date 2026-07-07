from __future__ import annotations
import torch
from typing import Optional, Dict
from torch_geometric.loader import DataLoader
from graph_signal_diffusion.baselines import BASELINE_REGISTRY
from graph_signal_diffusion.baselines.base import BaseBaseline


@BASELINE_REGISTRY.register("diffstg")
class DiffSTG(BaseBaseline):
    """DiffSTG baseline for stock price forecasting.

    .. warning::
        This is an unfinished stub.  ``fit``, ``predict``, and ``evaluate``
        all raise ``NotImplementedError``.  Do not use in production runs.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Args:
            model: The DiffSTG model instance.
            device: Torch device string (default ``"cpu"``).
        """
        super().__init__(device=torch.device(device))
        self.model = model

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs,
    ):
        raise NotImplementedError("DiffSTG is not yet implemented.")

    def predict(self, data) -> torch.Tensor:
        raise NotImplementedError("DiffSTG is not yet implemented.")

    def evaluate(
        self,
        loader: DataLoader,
        task,
        max_batches: Optional[int] = None,
        viz_save_dir: Optional[str] = None,
        eval_split_name: str = "eval",
    ) -> Dict[str, float]:
        raise NotImplementedError("DiffSTG is not yet implemented.")
